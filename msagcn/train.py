# ============================ train.py ============================
# -*- coding: utf-8 -*-
"""
train.py — Multi-Stream AGCN trainer (joints/bones/velocity, mask-aware, TTA)

Фишки:
- MultiStreamGestureDataset + MultiStreamAGCN (mask-aware).
- WeightedRandomSampler по частотам классов *и* качеству видео (meta).
- AMP + grad-accum + grad-clip.
- Cosine LR с warmup.
- Mirror-TTA валидация.
- Профайлер: dataloader_time и fwd+bwd_time за эпоху.
- Перформанс флаги: TF32, channels_last, torch.compile.
- FIX: NaN-guard в train/eval + bf16 autocast на CUDA.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, classification_report
from torch import amp
from torch.utils.data import DataLoader, WeightedRandomSampler

from msagcn.dataset_multistream import (
    DSConfig,
    MultiStreamGestureDataset,
    build_sample_weights,
)
from msagcn.model import MultiStreamAGCN

# ---------------------------- Utils ------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def move_to_device(batch_x: Dict[str, torch.Tensor], device: torch.device):
    out = {}
    for k, v in batch_x.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
    return out


def build_mirror_idx(ds: MultiStreamGestureDataset) -> torch.LongTensor:
    V = ds.V
    idx = list(range(21, 42)) + list(range(0, 21))  # swap hands
    if V > 42:
        pose_keep = list(ds.pose_keep)
        pairs = {(9,10), (11,12), (13,14), (15,16), (23,24)}
        pos_map = {abs_i: i for i, abs_i in enumerate(pose_keep)}
        for abs_i in pose_keep:
            if abs_i == 0:
                idx.append(42 + pos_map[0]); continue
            pair_abs = None
            for a, b in pairs:
                if abs_i == a: pair_abs = b
                elif abs_i == b: pair_abs = a
            if pair_abs is None or pair_abs not in pos_map:
                idx.append(42 + pos_map[abs_i])
            else:
                idx.append(42 + pos_map[pair_abs])
    return torch.tensor(idx, dtype=torch.long)

# ------------------------- Prefetcher -------------------------------------

class PrefetchLoader:
    """Асинхронное копирование батчей (dict of tensors) в GPU."""
    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None

    def __iter__(self):
        self.iter = iter(self.loader)
        if self.stream is not None:
            self._preload()
        return self

    def __next__(self):
        if self.stream is None:
            return next(self.iter)
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        X, y, metas = batch
        for t in X.values():
            if isinstance(t, torch.Tensor):
                t.record_stream(torch.cuda.current_stream())
        y.record_stream(torch.cuda.current_stream())
        self._preload()
        return X, y, metas

    def _preload(self):
        try:
            X, y, metas = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return
        if self.stream is None:
            self.next_batch = (X, y, metas)
            return
        with torch.cuda.stream(self.stream):
            X = move_to_device(X, self.device)
            y = y.to(self.device, non_blocking=True)
            self.next_batch = (X, y, metas)

    def __len__(self):
        return len(self.loader)

# ------------------------- EMA -----------------------------------------------


class ModelEma:
    """Exponential Moving Average wrapper for more stable eval."""
    def __init__(self, model: nn.Module, decay: float = 0.999, device: torch.device | None = None):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        if device is not None:
            self.module.to(device)
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for k, v in ema_state.items():
            src = model_state[k]
            if not v.dtype.is_floating_point:
                v.copy_(src)
                continue
            v.mul_(self.decay).add_(src, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)

# ------------------------- Loss (Logit-Adjusted) ------------------------------

class LogitAdjustedCrossEntropyLoss(nn.Module):
    """Cross-entropy with logit adjustment based on class frequencies (Menon et al., 2021)."""
    def __init__(self, class_freq: torch.Tensor, reduction: str = "mean"):
        super().__init__()
        freq = class_freq.clone().detach().float()
        freq = torch.clamp(freq, min=1.0)  # avoid log(0)
        log_freq = torch.log(freq)
        self.register_buffer("log_freq", log_freq)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_freq = self.log_freq.to(logits.device, logits.dtype)
        adjusted_logits = logits - log_freq
        loss = nn.functional.cross_entropy(adjusted_logits, target, reduction=self.reduction)
        return loss

# ---------------------------- Train/Eval -------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    device,
    accum_steps: int,
    grad_clip: float,
    A: torch.Tensor,
    use_channels_last: bool,
    *,
    max_batches: int | None = None,
    log_interval: int = 50,
    ema: ModelEma | None = None,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_n = 0
    t_load_sum = 0.0
    t_fwd_sum  = 0.0
    steps = 0
    max_batches = max_batches if max_batches and max_batches > 0 else None

    for step, (X, y, _) in enumerate(loader, start=1):
        if max_batches is not None and step > max_batches:
            break
        t0 = time.time()
        # Ensure batch is on the right device (когда PrefetchLoader выключен)
        if any(hasattr(v, 'device') and v.device.type != device.type for v in X.values()):
            X = move_to_device(X, device)
            y = y.to(device, non_blocking=True)
        t_load_sum += (time.time() - t0)

        mask = X.get("mask", None)
        x_streams = {k: v for k, v in X.items() if k != "mask"}
        if device.type == "cuda" and use_channels_last:
            for k in x_streams:
                x_streams[k] = x_streams[k].to(memory_format=torch.channels_last)

        if step == 1 and "joints" in x_streams:
            print("debug | |joints|_mean =", float(x_streams["joints"].abs().mean()))

        t1 = time.time()
        # --- SAFE autocast: CUDA → bf16 (широкий экспонент), CPU → bf16 ---
        with amp.autocast(
            device_type=("cuda" if device.type == "cuda" else "cpu"),
            dtype=(torch.bfloat16),
            enabled=True,
        ):
            logits = model(x_streams, mask=mask, A=A, y=y)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            loss = criterion(logits, y) / accum_steps
        t_fwd_sum += (time.time() - t1)

        # NaN guard
        if not torch.isfinite(loss):
            print("WARN: non-finite loss encountered → batch skipped")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        last_step = (step == len(loader)) if max_batches is None else (step == max_batches)
        if (step % accum_steps == 0) or last_step:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        bs = y.size(0)
        total_loss += float(loss.item()) * accum_steps * bs
        total_n += bs
        steps += 1

        if log_interval > 0 and (step % log_interval == 0):
            with torch.no_grad():
                probs = logits.detach().softmax(dim=1)
                entropy = float((-probs * torch.log(probs.clamp_min(1e-9))).sum(dim=1).mean())
                conf = float(probs.max(dim=1).values.mean())
            print(f"   step {step:04d} | loss={float(loss.item() * accum_steps):.4f} | entropy={entropy:.3f} | conf={conf:.3f}")

    if total_n > 0:
        print(f"   dataloader_time≈{t_load_sum:.1f}s | fwd+bwd_time≈{t_fwd_sum:.1f}s | steps={steps}")
    return total_loss / max(1, total_n)

@torch.no_grad()
def evaluate(model, loader, criterion, device, A: torch.Tensor, mirror_idx: torch.LongTensor | None,
             use_channels_last: bool, *, max_batches: int | None = None):
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    max_batches = max_batches if max_batches and max_batches > 0 else None

    for step, (X, y, _) in enumerate(loader, start=1):
        if max_batches is not None and step > max_batches:
            break
        # Если лоадер без префетчера — переместим
        if any(hasattr(v, 'device') and v.device.type != device.type for v in X.values()):
            X = move_to_device(X, device)
            y = y.to(device, non_blocking=True)

        mask = X.get("mask", None)
        x_streams = {k: v for k, v in X.items() if k != "mask"}
        if device.type == "cuda" and use_channels_last:
            for k in x_streams:
                x_streams[k] = x_streams[k].to(memory_format=torch.channels_last)

        print("debug/eval | mask_coverage =", float((mask>0).float().mean()) if mask is not None else -1.0)

        # --- SAFE autocast: CUDA/CPU → bf16 ---
        with amp.autocast(
            device_type=("cuda" if device.type == "cuda" else "cpu"),
            dtype=(torch.bfloat16),
            enabled=True,
        ):
            if mirror_idx is not None:
                # compile/cudagraph-safe TTA: один прогон на батче, склеиваем по B
                B0 = y.size(0)
                x_cat = {
                    k: torch.cat([v, v.index_select(dim=2, index=mirror_idx)], dim=0)
                    for k, v in x_streams.items()
                }
                m_cat = (torch.cat([mask, mask.index_select(dim=2, index=mirror_idx)], dim=0)
                         if mask is not None else None)
                try:
                    torch.compiler.cudagraph_mark_step_begin()
                except Exception:
                    pass
                logits_cat = model(x_cat, mask=m_cat, A=A)
                logits = 0.5 * (logits_cat[:B0] + logits_cat[B0:])
            else:
                logits = model(x_streams, mask=mask, A=A)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            loss = criterion(logits, y)

        total_loss += float(loss.item()) * y.size(0)
        total_n += y.size(0)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro") if total_n else 0.0
    acc = (np.array(all_preds) == np.array(all_labels)).mean() if total_n else 0.0
    return total_loss / max(1, total_n), acc, macro_f1, all_preds, all_labels

# ---------------------------- CLI --------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Train Multi-Stream AGCN (max F1)")

    # Data
    p.add_argument("--json", required=True, help="skeletons source: combined .json OR a directory with per-video *.json")
    p.add_argument("--csv", required=True, help="annotations CSV with splits")
    p.add_argument("--out", default="outputs/runs/agcn", help="output dir (checkpoints, logs)")

    # Dataset config
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument("--end_is_exclusive", action="store_true",
                   help="Treat CSV 'end' as python-slice exclusive (default: end is inclusive → +1).")
    p.add_argument(
        "--temporal_crop",
        type=str,
        default="random",
        choices=["random", "best", "center", "resample"],
        help=(
            "Temporal strategy inside annotated [begin,end]. "
            "random/best/center: take a contiguous window of max_frames (may cut off long gestures). "
            "resample: time-resample the whole segment to exactly max_frames (keeps segment boundaries)."
        ),
    )
    p.add_argument("--streams", type=str, default="joints,bones,velocity")
    p.add_argument("--include_pose", action="store_true")
    p.add_argument("--pose_keep", type=str, default="0,9,10,11,12,13,14,15,16,23,24")
    p.add_argument("--pose_vis_thr", type=float, default=0.5)
    p.add_argument("--connect_cross_edges", action="store_true")

    p.add_argument("--hand_score_thr", type=float, default=0.45)
    p.add_argument("--hand_score_thr_fallback", type=float, default=0.35)
    p.add_argument("--window_valid_ratio", type=float, default=0.60)
    p.add_argument("--window_valid_ratio_fallback", type=float, default=0.50)

    p.add_argument("--center", action="store_true")
    p.add_argument("--center_mode", type=str, default="masked_mean", choices=["masked_mean","wrists"])
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--norm_method", type=str, default="p95", choices=["p95","max","mad"])
    p.add_argument("--norm_scale", type=float, default=1.0)

    p.add_argument("--augment", action="store_true")
    p.add_argument("--mirror_prob", type=float, default=0.5)
    p.add_argument("--rot_deg", type=float, default=10.0)
    p.add_argument("--scale_jitter", type=float, default=0.10)
    p.add_argument("--noise_sigma", type=float, default=0.01)
    p.add_argument("--mirror_swap_only", action="store_true")
    p.add_argument("--time_drop_prob", type=float, default=0.0)
    p.add_argument("--hand_drop_prob", type=float, default=0.0)

    # Small temporal augs (train only): robust to annotation noise & speed variation
    p.add_argument("--boundary_jitter_prob", type=float, default=0.3)
    p.add_argument("--boundary_jitter_max", type=int, default=2)
    p.add_argument("--speed_perturb_prob", type=float, default=0.3)
    p.add_argument("--speed_perturb_kmin", type=int, default=60)
    p.add_argument("--speed_perturb_kmax", type=int, default=68)

    # Train config
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--ema_decay", type=float, default=0.0, help="EMA decay for model weights (0=off)")
    p.add_argument("--warmup_frac", type=float, default=0.10, help="Fraction of total epochs to use for LR warmup (0 disables)")
    p.add_argument("--early_stop_patience", type=int, default=10, help="Stop if val_f1 does not improve for N epochs (0 disables)")
    p.add_argument("--early_stop_min_delta", type=float, default=0.0, help="Minimum val_f1 increase to reset early stopping patience")
    p.add_argument("--limit_train_batches", type=int, default=0, help="Debug: cap train batches per epoch")
    p.add_argument("--limit_val_batches", type=int, default=0, help="Debug: cap val batches")
    p.add_argument("--overfit_batches", type=int, default=0, help="Debug: use same limited #batches for train+val")
    p.add_argument("--log_interval", type=int, default=50, help="Print train stats every N steps")
    p.add_argument("--keep_aug_in_debug", action="store_true", help="Do not auto-disable augmentations when limiting batches/overfitting")
    p.add_argument("--disable_norm_in_debug", action="store_true", help="When in debug/overfit mode, turn off normalization")

    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 8))
    p.add_argument("--prefetch", type=int, default=6)
    p.add_argument("--no_prefetch", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # Sampling & Loss options
    p.add_argument("--weighted_sampler", action="store_true", help="Use WeightedRandomSampler for train")
    p.add_argument("--use_logit_adjustment", action="store_true", help="Use Logit-Adjusted CrossEntropy")
    # Dataset I/O perf
    p.add_argument("--file_cache", type=int, default=64, help="small per-dataset file cache for per-video JSON (0=off)")
    p.add_argument(
        "--prefer_pp",
        dest="prefer_pp",
        action="store_true",
        default=True,
        help="Prefer *_pp.json when using a per-video JSON directory (fallback to raw .json).",
    )
    p.add_argument(
        "--no_prefer_pp",
        dest="prefer_pp",
        action="store_false",
        help="Use raw *.json even if *_pp.json exists.",
    )

    # Model
    p.add_argument("--depths", type=str, default="64,128,256,256")
    p.add_argument("--temp_ks", type=str, default="9,7,5,5")
    p.add_argument("--droppath", type=float, default=0.05)
    p.add_argument("--drop", type=float, default=0.10)
    p.add_argument("--stream_drop_p", type=float, default=0.10)
    p.add_argument("--use_groupnorm_stem", action="store_true")
    # NEW: cosine head flags
    p.add_argument("--use_cosine_head", action="store_true")
    p.add_argument("--cosine_margin", type=float, default=0.2)
    p.add_argument("--cosine_scale", type=float, default=30.0)

    # Perf toggles
    p.add_argument("--tf32", action="store_true", help="Enable TF32 (matmul & cudnn)")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--channels_last", action="store_true", help="Use channels_last memory format")

    # Prior & TTA
    p.add_argument("--tta_mirror", action="store_true")

    return p.parse_args()

# ---------------------------- Main -------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    debug_mode = (args.overfit_batches > 0) or (args.limit_train_batches > 0)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| cudnn.benchmark =", (device.type == "cuda"))
    torch.backends.cudnn.benchmark = (device.type == "cuda")
    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Dataset config
    augment_flag = args.augment
    time_drop_prob = args.time_drop_prob
    hand_drop_prob = args.hand_drop_prob
    normalize_flag = args.normalize
    if debug_mode and args.augment and not args.keep_aug_in_debug:
        print("Debug mode detected → disabling dataset augmentations/time-drop/hand-drop for reproducibility.")
        augment_flag = False
        time_drop_prob = 0.0
        hand_drop_prob = 0.0
    if debug_mode and args.normalize and args.disable_norm_in_debug:
        print("Debug mode detected → disabling dataset normalization for reproducibility (flagged).")
        normalize_flag = False

    ds_cfg = DSConfig(
        max_frames=args.max_frames,
        end_inclusive=(not args.end_is_exclusive),
        temporal_crop=str(args.temporal_crop),
        use_streams=tuple(s.strip() for s in args.streams.split(",") if s.strip()),
        include_pose=args.include_pose,
        pose_keep=tuple(int(x) for x in args.pose_keep.split(",") if x.strip()),
        pose_vis_thr=args.pose_vis_thr,
        connect_cross_edges=args.connect_cross_edges,
        hand_score_thr=args.hand_score_thr,
        hand_score_thr_fallback=args.hand_score_thr_fallback,
        window_valid_ratio=args.window_valid_ratio,
        window_valid_ratio_fallback=args.window_valid_ratio_fallback,
        center=args.center,
        center_mode=args.center_mode,
        normalize=normalize_flag,
        norm_method=args.norm_method,
        norm_scale=args.norm_scale,
        augment=augment_flag,
        mirror_prob=args.mirror_prob,
        rot_deg=args.rot_deg,
        scale_jitter=args.scale_jitter,
        noise_sigma=args.noise_sigma,
        mirror_swap_only=args.mirror_swap_only,
        time_drop_prob=time_drop_prob,
        hand_drop_prob=hand_drop_prob,
        file_cache_size=args.file_cache,
        prefer_pp=args.prefer_pp,
    )

    # Datasets
    train_ds = MultiStreamGestureDataset(args.json, args.csv, split="train", cfg=ds_cfg)
    label2idx = train_ds.label2idx  # фиксируем единую карту меток от train
    val_ds = MultiStreamGestureDataset(
        args.json, args.csv, split="val",
        cfg=DSConfig(**{**asdict(ds_cfg), "augment": False}),
        label2idx=label2idx,
    )

    # Detect storage mode (dir vs combined)
    json_path = Path(args.json)
    is_dir_mode = json_path.is_dir()
    if (os.name == "nt") and (not is_dir_mode) and args.workers > 0:
        print("Note: Windows + combined JSON detected → forcing workers=0 to avoid RAM blowups.")
        args.workers = 0
    print(f"Dataset mode: {'dir' if is_dir_mode else 'combined'} | workers={args.workers}")

    # Save label map & ds cfg
    with (out_dir / "label2idx.json").open("w", encoding="utf-8") as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)
    with (out_dir / "ds_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(ds_cfg), f, ensure_ascii=False, indent=2)

    # Sampler weights
    weights = build_sample_weights(
        train_ds.samples, label2idx, train_ds._meta_by_vid,
        quality_floor=0.4, quality_power=1.0, cover_key="both_coverage", cover_floor=0.3
    )
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True) if args.weighted_sampler else None

    # DataLoaders (всегда создаём; префетчер — опционально поверх)
    dl_kwargs = dict(
        batch_size=args.batch,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        collate_fn=MultiStreamGestureDataset.collate_fn,
    )
    if device.type == "cuda":
        dl_kwargs["pin_memory"] = True
        dl_kwargs["pin_memory_device"] = f"cuda:{torch.cuda.current_device()}"
    else:
        dl_kwargs["pin_memory"] = False
        
    if args.workers > 0:
        dl_kwargs["prefetch_factor"] = max(2, int(args.prefetch))

    train_loader = DataLoader(train_ds, sampler=sampler if (args.weighted_sampler and sampler is not None) else None,
                               shuffle=False if (args.weighted_sampler and sampler is not None) else True,
                               drop_last=True, **dl_kwargs)
    val_loader   = DataLoader(val_ds, shuffle=False, **dl_kwargs)

    if device.type == "cuda" and not args.no_prefetch:
        train_loader = PrefetchLoader(train_loader, device)
        val_loader   = PrefetchLoader(val_loader, device)

    # Model
    depths = tuple(int(x) for x in args.depths.split(","))
    temp_ks = tuple(int(x) for x in args.temp_ks.split(","))
    model_drop = args.drop
    model_stream_drop = args.stream_drop_p
    if debug_mode and not args.keep_aug_in_debug:
        if (model_drop > 0) or (model_stream_drop > 0):
            print("Debug mode detected → forcing model dropout rates to 0.")
        model_drop = 0.0
        model_stream_drop = 0.0
    model = MultiStreamAGCN(
        num_classes=len(label2idx),
        V=train_ds.V,
        A=train_ds.build_adjacency(normalize=False),
        in_ch=3,
        streams=ds_cfg.use_streams,
        drop=model_drop,
        droppath=args.droppath,
        depths=depths,
        temp_ks=temp_ks,
        use_groupnorm_stem=args.use_groupnorm_stem,
        stream_drop_p=model_stream_drop,
        # cosine head flags
        use_cosine_head=args.use_cosine_head,
        cosine_margin=args.cosine_margin,
        cosine_scale=args.cosine_scale,
    ).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")

    # Static adjacency
    A = train_ds.build_adjacency(normalize=False).to(device)

    # Mirror index for TTA
    mirror_idx = build_mirror_idx(train_ds).to(device) if args.tta_mirror else None

    # Loss
    cls_counts = np.bincount([label2idx[lbl] for _, lbl, *_ in train_ds.samples], minlength=len(label2idx))
    cls_counts = np.maximum(cls_counts, 1)
    cls_counts = torch.tensor(cls_counts, dtype=torch.float32, device=device)
    if args.use_logit_adjustment:
        criterion = LogitAdjustedCrossEntropyLoss(cls_counts)
    else:
        smoothing = max(0.0, float(args.label_smoothing))
        criterion = nn.CrossEntropyLoss(label_smoothing=smoothing if smoothing > 0 else 0.0)

    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warmup_epochs = int(max(0, round(args.warmup_frac * args.epochs)))
    if warmup_epochs >= args.epochs:
        warmup_epochs = max(0, args.epochs - 1)
    if warmup_epochs > 0:
        main_epochs = max(1, args.epochs - warmup_epochs)
        sched = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
                optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=main_epochs)
            ],
            milestones=[warmup_epochs]
        )
    else:
        sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
        )
    scaler = amp.GradScaler("cuda") if device.type == "cuda" else amp.GradScaler(enabled=False)
    ema = ModelEma(model, decay=args.ema_decay, device=device) if args.ema_decay > 0 else None

    # Training
    if args.overfit_batches > 0:
        train_limit = val_limit = args.overfit_batches
    else:
        train_limit = args.limit_train_batches if args.limit_train_batches > 0 else None
        val_limit = args.limit_val_batches if args.limit_val_batches > 0 else None
    if train_limit or val_limit:
        print(f"Debug limits → train:{train_limit or 'full'} | val:{val_limit or 'full'}")

    best_f1 = -1.0
    es_patience = max(0, int(args.early_stop_patience))
    es_min_delta = max(0.0, float(args.early_stop_min_delta))
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device,
            accum_steps=args.accum, grad_clip=args.grad_clip, A=A,
            use_channels_last=args.channels_last, max_batches=train_limit,
            log_interval=args.log_interval, ema=ema,
        )
        eval_model = ema.module if ema is not None else model
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
            eval_model, val_loader, criterion, device, A=A, mirror_idx=mirror_idx,
            use_channels_last=args.channels_last, max_batches=val_limit,
        )
        sched.step()

        dt = time.time() - t0
        print(f"[Ep {epoch:03d}] TL={tr_loss:.4f} | VL={val_loss:.4f} | VA={val_acc:.3f} | VF1={val_f1:.3f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | {dt:.1f}s")
        from collections import Counter
        print("pred top5:", Counter(val_preds).most_common(5))

        history.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=val_loss, val_acc=val_acc, val_f1=val_f1))

        improved = val_f1 > (best_f1 + es_min_delta)
        if improved:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "sched_state": sched.state_dict(),
                "scaler_state": scaler.state_dict(),
                "ema_state": (ema.state_dict() if ema is not None else None),
                "label2idx": label2idx,
                "ds_cfg": asdict(ds_cfg),
                "best_f1": best_f1,
            }, out_dir / "best.ckpt")
            print(f"  ↳ 🔖 new best macro-F1 = {best_f1:.4f} (checkpoint saved)")
        else:
            epochs_no_improve += 1

        # Per-class F1 (раз в 5 эпох)
        if epoch % 5 == 0:
            report = classification_report(val_labels, val_preds, output_dict=True, zero_division=0)
            with (out_dir / f"report_ep{epoch:03d}.json").open("w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

        # Save history
        with (out_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if es_patience > 0 and epochs_no_improve >= es_patience:
            print(f"Early stopping: no val_f1 improvement for {es_patience} epochs. Best_f1={best_f1:.4f}")
            break

    print(f"Done. Best macro-F1 = {best_f1:.4f}")

if __name__ == "__main__":
    main()
