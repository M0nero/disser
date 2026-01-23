from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bio.core.config_utils import load_config_section, write_run_config
from bio.core.datasets.shard_dataset import ShardedBiosDataset, make_boundary_aware_sampler
from bio.core.model import BioModelConfig, BioTagger


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _as_bool(raw: object, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        val = raw.strip().lower()
        if val in ("true", "1", "yes", "y"):
            return True
        if val in ("false", "0", "no", "n"):
            return False
    return default


def _is_missing(raw: object) -> bool:
    if raw is None:
        return True
    if isinstance(raw, str) and not raw.strip():
        return True
    return False


class JsonlLogger:
    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def log(self, payload: Dict[str, object]) -> None:
        payload = dict(payload)
        payload["ts"] = time.time()
        line = json.dumps(payload, ensure_ascii=False)
        print(line)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


def collate_drop_meta(batch):
    if not batch:
        return {}
    if isinstance(batch[0], dict):
        batch = [{k: v for k, v in item.items() if k != "meta"} for item in batch]
    return default_collate(batch)


def compute_boundary_f1_tolerant(y_true: torch.Tensor, y_pred: torch.Tensor, tol: int = 2) -> Dict[str, float]:
    """
    Tolerant boundary F1 for 'B' events.
    Matches predicted B positions to true B positions within +/- tol frames.

    y_true, y_pred: (B,T) int labels with B=1.
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    tp = 0
    fp = 0
    fn = 0

    for t, p in zip(y_true, y_pred):
        gt = np.where(t == 1)[0].tolist()
        pr = np.where(p == 1)[0].tolist()

        used = set()
        for pi in pr:
            # find closest unmatched gt within tol
            best = None
            best_d = None
            for gi in gt:
                if gi in used:
                    continue
                d = abs(int(pi) - int(gi))
                if d <= tol and (best_d is None or d < best_d):
                    best_d = d
                    best = gi
            if best is not None:
                tp += 1
                used.add(best)
            else:
                fp += 1
        fn += (len(gt) - len(used))

    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {"b_prec_tol": float(prec), "b_rec_tol": float(rec), "b_f1_tol": float(f1)}


def frame_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    """
    Framewise accuracy + per-class F1 for labels 0/1/2.
    """
    y_true = y_true.detach().cpu().numpy().reshape(-1)
    y_pred = y_pred.detach().cpu().numpy().reshape(-1)

    acc = float((y_true == y_pred).mean())

    out = {"acc": acc}
    for cls, name in [(0, "O"), (1, "B"), (2, "I")]:
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        out[f"p_{name}"] = float(prec)
        out[f"r_{name}"] = float(rec)
        out[f"f1_{name}"] = float(f1)
    return out


def save_checkpoint(out_dir: Path, step: int, model: torch.nn.Module, optim: torch.optim.Optimizer, cfg: BioModelConfig, stats: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict(),
        "cfg": asdict(cfg),
        "stats": stats,
    }
    torch.save(ckpt, out_dir / "last.pt")
    # keep periodic snapshots
    if step % max(1, stats.get("save_every_steps", 1000)) == 0:
        torch.save(ckpt, out_dir / f"step_{step:07d}.pt")


def train_one_epoch(
    model: BioTagger,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
    class_weights: torch.Tensor,
    grad_clip: float,
    log_every: int,
    step0: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    epoch: int,
    logger: JsonlLogger,
) -> int:
    model.train()
    step = step0
    t_load_sum = 0.0
    t_fwd_sum = 0.0
    steps = 0
    total_loss = 0.0
    total_samples = 0
    epoch_start = time.time()
    last_log_time = epoch_start
    last_log_samples = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for i, batch in enumerate(loader):
        t0 = time.time()
        pts = batch["pts"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        bio = batch["bio"].to(device, non_blocking=True)
        t_load_sum += (time.time() - t0)

        t1 = time.time()
        with amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits, _ = model(pts, mask)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            loss = F.cross_entropy(logits.reshape(-1, 3), bio.reshape(-1), weight=class_weights)

        optim.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            print("WARN: non-finite loss encountered → batch skipped")
            continue
        scaler.scale(loss).backward()
        grad_norm = None
        if grad_clip > 0:
            scaler.unscale_(optim)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optim)
        scaler.update()
        t_fwd_sum += (time.time() - t1)

        bs = int(bio.size(0))
        total_samples += bs
        total_loss += float(loss.item()) * bs

        if (step + 1) % log_every == 0:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                m1 = frame_metrics(bio, pred)
                m2 = compute_boundary_f1_tolerant(bio, pred, tol=2)
                f1_macro = (m1["f1_O"] + m1["f1_B"] + m1["f1_I"]) / 3.0
            now = time.time()
            elapsed = max(1e-6, now - last_log_time)
            sps = (total_samples - last_log_samples) / elapsed
            lr = float(optim.param_groups[0]["lr"]) if optim.param_groups else 0.0
            payload = {
                "event": "train_step",
                "epoch": epoch,
                "step": step + 1,
                "loss": float(loss.item()),
                "acc": float(m1["acc"]),
                "f1_macro": float(f1_macro),
                "b_f1_tol": float(m2["b_f1_tol"]),
                "lr": lr,
                "samples_per_sec": float(sps),
            }
            if grad_norm is not None:
                payload["grad_norm"] = float(grad_norm)
            if device.type == "cuda":
                payload["mem_alloc_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
                payload["mem_reserved_mb"] = float(torch.cuda.max_memory_reserved() / (1024 ** 2))
            logger.log(payload)
            last_log_time = now
            last_log_samples = total_samples

        step += 1
        steps += 1

    if steps > 0:
        epoch_time = max(1e-6, time.time() - epoch_start)
        avg_loss = total_loss / max(1, total_samples)
        lr = float(optim.param_groups[0]["lr"]) if optim.param_groups else 0.0
        payload = {
            "event": "train_epoch",
            "epoch": epoch,
            "step": step,
            "avg_loss": float(avg_loss),
            "samples": int(total_samples),
            "steps": int(steps),
            "epoch_time_sec": float(epoch_time),
            "samples_per_sec": float(total_samples / epoch_time),
            "dataloader_time_sec": float(t_load_sum),
            "fwd_bwd_time_sec": float(t_fwd_sum),
            "lr": lr,
        }
        if device.type == "cuda":
            payload["mem_alloc_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            payload["mem_reserved_mb"] = float(torch.cuda.max_memory_reserved() / (1024 ** 2))
        logger.log(payload)
    return step


@torch.no_grad()
def eval_epoch(
    model: BioTagger,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    class_weights: torch.Tensor,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    ys = []
    ps = []
    for batch in loader:
        pts = batch["pts"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        bio = batch["bio"].to(device, non_blocking=True)

        with amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits, _ = model(pts, mask)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            loss = F.cross_entropy(logits.reshape(-1, 3), bio.reshape(-1), weight=class_weights)
        pred = logits.argmax(dim=-1)
        ys.append(bio)
        ps.append(pred)
        bs = int(bio.size(0))
        total_loss += float(loss.item()) * bs
        total_n += bs

    y = torch.cat(ys, dim=0)
    p = torch.cat(ps, dim=0)

    m1 = frame_metrics(y, p)
    m2 = compute_boundary_f1_tolerant(y, p, tol=2)
    avg_loss = total_loss / max(1, total_n)
    return {"loss": float(avg_loss), **m1, **m2}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, _ = pre.parse_known_args(argv)
    defaults = load_config_section(pre_args.config, "train")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=pre_args.config, help="Path to config JSON (section: train).")
    ap.add_argument(
        "--train_dir",
        type=str,
        default=defaults.get("train_dir"),
        required=_is_missing(defaults.get("train_dir")),
        help="Step2 synth train dir (has index.json and shards/)",
    )
    ap.add_argument("--val_dir", type=str, default=defaults.get("val_dir", ""), help="Step2 synth val dir (optional)")
    ap.add_argument(
        "--out_dir",
        type=str,
        default=defaults.get("out_dir"),
        required=_is_missing(defaults.get("out_dir")),
    )

    ap.add_argument("--epochs", type=int, default=int(defaults.get("epochs", 5)))
    ap.add_argument("--batch_size", type=int, default=int(defaults.get("batch_size", 64)))
    ap.add_argument("--num_workers", type=int, default=int(defaults.get("num_workers", 4)))
    ap.add_argument("--prefetch", type=int, default=int(defaults.get("prefetch", 2)))
    ap.add_argument("--lr", type=float, default=float(defaults.get("lr", 2e-3)))
    ap.add_argument("--weight_decay", type=float, default=float(defaults.get("weight_decay", 1e-4)))
    ap.add_argument("--grad_clip", type=float, default=float(defaults.get("grad_clip", 1.0)))
    ap.add_argument("--no_amp", action="store_true", default=_as_bool(defaults.get("no_amp"), False), help="Disable AMP even on CUDA.")
    ap.add_argument("--tf32", action="store_true", default=_as_bool(defaults.get("tf32"), False), help="Enable TF32 on CUDA (matmul + cudnn).")

    default_device = defaults.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=int(defaults.get("seed", 1337)))
    ap.add_argument("--device", type=str, default=default_device)

    # boundary-aware sampler: target fraction of samples that contain at least one B
    ap.add_argument("--p_with_b", type=float, default=float(defaults.get("p_with_b", 0.85)))

    # class weights for O,B,I (B should be big)
    ap.add_argument("--wO", type=float, default=float(defaults.get("wO", 1.5)))
    ap.add_argument("--wB", type=float, default=float(defaults.get("wB", 25.0)))
    ap.add_argument("--wI", type=float, default=float(defaults.get("wI", 1.0)))

    # model config
    ap.add_argument("--embed_dim", type=int, default=int(defaults.get("embed_dim", 128)))
    ap.add_argument("--conv_kernel", type=int, default=int(defaults.get("conv_kernel", 5)))
    ap.add_argument("--conv_layers", type=int, default=int(defaults.get("conv_layers", 2)))
    ap.add_argument("--gru_hidden", type=int, default=int(defaults.get("gru_hidden", 192)))
    ap.add_argument("--gru_layers", type=int, default=int(defaults.get("gru_layers", 1)))
    ap.add_argument("--drop_conv", type=float, default=float(defaults.get("drop_conv", 0.10)))
    ap.add_argument("--drop_head", type=float, default=float(defaults.get("drop_head", 0.10)))

    ap.add_argument("--log_every", type=int, default=int(defaults.get("log_every", 100)))
    ap.add_argument("--eval_every", type=int, default=int(defaults.get("eval_every", 500)))
    ap.add_argument("--save_every", type=int, default=int(defaults.get("save_every", 1000)))
    ap.add_argument(
        "--log_jsonl",
        type=str,
        default=defaults.get("log_jsonl", ""),
        help="Write JSONL logs to this path (default: <out_dir>/train_log.jsonl).",
    )

    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available → falling back to CPU")
        device = torch.device("cpu")

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.float32
    use_amp = (device.type == "cuda") and (not args.no_amp)

    train_ds = ShardedBiosDataset(args.train_dir, shard_cache_items=2)
    sampler = make_boundary_aware_sampler(train_ds, p_with_b=float(args.p_with_b), replacement=True)

    dl_kwargs = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": (device.type == "cuda"),
    }
    if int(args.num_workers) > 0:
        dl_kwargs["prefetch_factor"] = int(args.prefetch)
        dl_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_ds,
        sampler=sampler,
        drop_last=True,
        collate_fn=collate_drop_meta,
        **dl_kwargs,
    )

    val_loader = None
    if args.val_dir:
        val_ds = ShardedBiosDataset(args.val_dir, shard_cache_items=1)
        val_workers = max(1, int(args.num_workers) // 2)
        val_kwargs = {
            "batch_size": int(args.batch_size),
            "num_workers": val_workers,
            "pin_memory": (device.type == "cuda"),
        }
        if val_workers > 0:
            val_kwargs["prefetch_factor"] = int(args.prefetch)
            val_kwargs["persistent_workers"] = True

        val_loader = DataLoader(
            val_ds,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_drop_meta,
            **val_kwargs,
        )

    cfg = BioModelConfig(
        num_joints=42,
        in_coords=3,
        embed_dim=int(args.embed_dim),
        conv_kernel=int(args.conv_kernel),
        conv_layers=int(args.conv_layers),
        conv_dropout=float(args.drop_conv),
        gru_hidden=int(args.gru_hidden),
        gru_layers=int(args.gru_layers),
        head_dropout=float(args.drop_head),
        use_vel=True,
        use_acc=True,
        use_mask=True,
        use_aggs=True,
    )

    model = BioTagger(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = amp.GradScaler(enabled=use_amp)

    class_weights = torch.tensor([args.wO, args.wB, args.wI], dtype=torch.float32, device=device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(out_dir, args, config_path=args.config, section="train", extra={"model": asdict(cfg)})

    log_path = Path(args.log_jsonl).resolve() if args.log_jsonl else (out_dir / "train_log.jsonl")
    logger = JsonlLogger(log_path)

    step = 0
    best_bf1 = -1.0
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log({
        "event": "start",
        "device": str(device),
        "amp": bool(use_amp),
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "train_samples": int(len(train_ds)),
        "val_samples": int(len(val_ds)) if args.val_dir else 0,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "p_with_b": float(args.p_with_b),
        "num_params": int(num_params),
    })

    for epoch in range(int(args.epochs)):
        step = train_one_epoch(
            model=model,
            loader=train_loader,
            optim=optim,
            scaler=scaler,
            device=device,
            class_weights=class_weights,
            grad_clip=float(args.grad_clip),
            log_every=int(args.log_every),
            step0=step,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            epoch=epoch + 1,
            logger=logger,
        )

        stats = {"epoch": epoch + 1, "step": step, "save_every_steps": int(args.save_every)}
        if val_loader is not None:
            metrics = eval_epoch(
                model,
                val_loader,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                class_weights=class_weights,
            )
            stats.update(metrics)
            logger.log({
                "event": "val_epoch",
                "epoch": epoch + 1,
                "step": step,
                **{k: float(v) for k, v in metrics.items()},
            })
            if metrics.get("b_f1_tol", -1.0) > best_bf1:
                best_bf1 = metrics["b_f1_tol"]
                torch.save({"model_state": model.state_dict(), "cfg": asdict(cfg), "metrics": metrics}, out_dir / "best.pt")

        save_checkpoint(out_dir, step, model, optim, cfg, stats)

    logger.log({"event": "done", "step": step})
    print("Done. last.pt saved in", str(out_dir))


if __name__ == "__main__":
    main()
