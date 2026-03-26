from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch import amp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from msagcn.data import DSConfig, MultiStreamGestureDataset, build_sample_weights
from msagcn.training.ema import ModelEma
from msagcn.training.engine import train_one_epoch
from msagcn.training.losses import LogitAdjustedCrossEntropyLoss, SupervisedContrastiveLoss
from msagcn.training.oof_utils import (
    build_model_from_saved_args,
    checkpoint_arg,
    compute_kinematic_signature,
    load_checkpoint_training_state,
    load_checkpoint_weights,
    make_sample_id,
    resolve_amp_settings,
)
from msagcn.training.oof_storage import write_predictions_table, write_sharded_array
from msagcn.training.runner import (
    _apply_loader_profile,
    _build_train_loader,
    _build_val_loader,
    _detect_cache_mode,
    _resolve_supcon_start_epoch,
)
from msagcn.training.utils import set_seed
from msagcn.training.auto_workers import resolve_loader_profile


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build strict train-only OOF cache for family construction.")
    p.add_argument("--json", required=True, type=str)
    p.add_argument("--csv", required=True, type=str)
    p.add_argument("--ckpt", required=True, type=str, help="Base checkpoint to fine-tune from")
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fold_epochs", type=int, default=6, help="Fixed fine-tune epochs per fold (no holdout early stopping)")
    p.add_argument("--workers", type=int, default=-1, help="Override worker count (-1 = use checkpoint args)")
    p.add_argument("--batch", type=int, default=0, help="Override batch size (0 = use checkpoint args)")
    p.add_argument("--save_full_logits", action="store_true")
    p.add_argument("--array_shard_size", type=int, default=4096, help="Rows per .npz shard for OOF feature/kinematic storage")
    p.add_argument("--limit_train_batches", type=int, default=0)
    p.add_argument("--limit_holdout_batches", type=int, default=0)
    return p.parse_args()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _build_fold_rows(samples: list[tuple[str, str, int, int | None]], fold_ids: np.ndarray, fold_idx: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample, assigned_fold in zip(samples, fold_ids.tolist()):
        vid, label, begin, end = sample
        rows.append(
            {
                "attachment_id": vid,
                "text": label,
                "begin": int(begin),
                "end": ("" if end is None else int(end)),
                "split": ("val" if int(assigned_fold) == int(fold_idx) else "train"),
            }
        )
    return rows


def _infer_end_excl(end_hint: int | None, end_inclusive: bool) -> int:
    if end_hint is None:
        return -1
    return int(end_hint + 1) if end_inclusive else int(end_hint)


def _build_fold_assignments(samples: list[tuple[str, str, int, int | None]], *, folds: int, seed: int, end_inclusive: bool) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels = np.asarray([label for _, label, _, _ in samples])
    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    fold_ids = np.full(len(samples), -1, dtype=np.int64)
    for fold_idx, (_, holdout_idx) in enumerate(splitter.split(np.zeros(len(samples)), labels)):
        fold_ids[np.asarray(holdout_idx, dtype=np.int64)] = int(fold_idx)
    if np.any(fold_ids < 0):
        raise RuntimeError("Failed to assign all train samples to OOF folds")

    rows: list[dict[str, Any]] = []
    for idx, (sample, fold_id) in enumerate(zip(samples, fold_ids.tolist())):
        vid, label, begin, end_hint = sample
        end_excl = _infer_end_excl(end_hint, end_inclusive)
        rows.append(
            {
                "row_id": int(idx),
                "fold": int(fold_id),
                "video": vid,
                "label": label,
                "begin": int(begin),
                "end_excl": int(end_excl),
                "sample_id": make_sample_id(vid, label, int(begin), int(end_excl)),
            }
        )
    return fold_ids, rows


def _build_loader_args(saved_args: dict[str, Any], *, batch: int, seed: int, workers_override: int) -> tuple[SimpleNamespace, Any]:
    args = SimpleNamespace(**saved_args)
    args.batch = int(batch)
    args.seed = int(seed)
    if workers_override >= 0:
        args.workers = int(workers_override)
    args.auto_workers = False
    requested_workers = int(checkpoint_arg(saved_args, "workers", 0) if workers_override < 0 else workers_override)
    return args, requested_workers


def _run_holdout_inference(
    *,
    model,
    loader,
    device: torch.device,
    A: torch.Tensor,
    use_amp: bool,
    amp_dtype: torch.dtype,
    idx2label: dict[int, str],
    end_inclusive: bool,
    save_full_logits: bool,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray | None]:
    model.eval()
    rows: list[dict[str, Any]] = []
    features_all: list[np.ndarray] = []
    kin_all: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    with torch.no_grad():
        for X, y, metas in loader:
            if any(hasattr(v, "device") and v.device.type != device.type for v in X.values()):
                X = {k: v.to(device, non_blocking=True) for k, v in X.items()}
                y = y.to(device, non_blocking=True)
            mask = X.get("mask")
            x_streams = {k: v for k, v in X.items() if k != "mask"}
            with amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits, features = model(x_streams, mask=mask, A=A, return_features=True)
            probs = torch.softmax(logits.float(), dim=1)
            topk_vals, topk_idx = torch.topk(probs, k=min(5, probs.shape[1]), dim=1)
            preds = probs.argmax(dim=1)
            margins = probs.gather(1, preds.unsqueeze(1)).squeeze(1) - topk_vals[:, 1].float().cpu().to(probs.device) if probs.shape[1] > 1 else probs.gather(1, preds.unsqueeze(1)).squeeze(1)

            batch_size = int(y.size(0))
            for i in range(batch_size):
                meta = metas[i]
                true_id = int(y[i].item())
                pred_id = int(preds[i].item())
                video = str(meta.get("video", ""))
                begin = int(meta.get("clip_begin", 0))
                end_excl = int(meta.get("clip_end_excl", -1))
                row = {
                    "sample_id": make_sample_id(video, idx2label[true_id], begin, end_excl),
                    "video": video,
                    "true_class": true_id,
                    "pred_class": pred_id,
                    "true_label": idx2label[true_id],
                    "pred_label": idx2label[pred_id],
                    "clip_begin": begin,
                    "clip_end_excl": end_excl,
                    "t0": int(meta.get("t0", begin)),
                    "t1": int(meta.get("t1", end_excl)),
                    "conf": float(probs[i, pred_id].item()),
                    "margin": float(margins[i].item()),
                    "top5_ids": " ".join(str(int(v)) for v in topk_idx[i].detach().cpu().tolist()),
                    "top5_probs": " ".join(f"{float(v):.6f}" for v in topk_vals[i].detach().cpu().tolist()),
                }
                rows.append(row)
                features_all.append(features[i].detach().float().cpu().numpy().astype(np.float32, copy=False))
                kin_all.append(compute_kinematic_signature({k: v[i].detach().cpu() for k, v in X.items()}))
                if save_full_logits:
                    logits_all.append(logits[i].detach().float().cpu().numpy().astype(np.float32, copy=False))
    return (
        rows,
        np.stack(features_all, axis=0).astype(np.float32, copy=False),
        np.stack(kin_all, axis=0).astype(np.float32, copy=False),
        (np.stack(logits_all, axis=0).astype(np.float32, copy=False) if save_full_logits else None),
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    folds_dir = out_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    state = load_checkpoint_training_state(args.ckpt)
    saved_args = state["args"]
    label2idx = state["label2idx"]
    idx2label = {int(v): str(k) for k, v in label2idx.items()}
    ds_cfg = DSConfig(**state["ds_cfg"])
    end_inclusive = bool(ds_cfg.end_inclusive)

    set_seed(int(args.seed))
    base_train_ds = MultiStreamGestureDataset(
        args.json,
        args.csv,
        split="train",
        cfg=ds_cfg,
        label2idx=label2idx,
        use_packed_cache=bool(checkpoint_arg(saved_args, "use_packed_skeleton_cache", False) and not checkpoint_arg(saved_args, "use_decoded_skeleton_cache", False)),
        packed_cache_dir=checkpoint_arg(saved_args, "packed_skeleton_cache_dir", "") or None,
        packed_cache_rebuild=False,
        use_decoded_cache=bool(checkpoint_arg(saved_args, "use_decoded_skeleton_cache", False)),
        decoded_cache_dir=checkpoint_arg(saved_args, "decoded_skeleton_cache_dir", "") or None,
        decoded_cache_rebuild=False,
    )
    samples = list(base_train_ds.samples)
    if int(args.folds) < 2:
        raise ValueError("--folds must be >= 2")
    fold_ids, fold_assignment_rows = _build_fold_assignments(samples, folds=int(args.folds), seed=int(args.seed), end_inclusive=end_inclusive)
    _write_csv(out_dir / "fold_assignments.csv", ["row_id", "fold", "video", "label", "begin", "end_excl", "sample_id"], fold_assignment_rows)

    record_bundles: list[dict[str, Any]] = []
    cache_mode = _detect_cache_mode(args=SimpleNamespace(**saved_args), is_dir_mode=Path(args.json).is_dir())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp, amp_dtype = resolve_amp_settings(saved_args, device)
    batch_size = int(args.batch if args.batch > 0 else checkpoint_arg(saved_args, "batch", 64))
    loader_args, requested_workers = _build_loader_args(saved_args, batch=batch_size, seed=int(args.seed), workers_override=int(args.workers))
    profile = resolve_loader_profile(
        requested_workers=int(requested_workers),
        requested_prefetch=int(checkpoint_arg(saved_args, "prefetch", 2)),
        requested_file_cache_size=int(checkpoint_arg(saved_args, "file_cache", 0)),
        os_name=os.name,
        is_dir_mode=Path(args.json).is_dir(),
        cache_mode=cache_mode,
        device_type=device.type,
        no_prefetch=bool(checkpoint_arg(saved_args, "no_prefetch", False)),
        cuda_index=(torch.cuda.current_device() if device.type == "cuda" else None),
    )

    meta = {
        "version": 1,
        "created_at_unix": time.time(),
        "base_checkpoint": state["path"],
        "json": str(Path(args.json).resolve()),
        "csv": str(Path(args.csv).resolve()),
        "folds": int(args.folds),
        "seed": int(args.seed),
        "fold_epochs": int(args.fold_epochs),
        "num_train_samples": int(len(samples)),
        "num_classes": int(len(label2idx)),
        "feature_dim": int(getattr(base_train_ds, "V", 0)),
        "save_full_logits": bool(args.save_full_logits),
        "class_names": {str(k): idx2label[k] for k in sorted(idx2label)},
        "protocol": "train_only_oof_fixed_epochs",
    }

    for fold_idx in range(int(args.folds)):
        print(f"=== OOF fold {fold_idx + 1}/{int(args.folds)} ===")
        fold_csv = folds_dir / f"fold_{fold_idx:02d}.csv"
        _write_csv(fold_csv, ["attachment_id", "text", "begin", "end", "split"], _build_fold_rows(samples, fold_ids, fold_idx))
        set_seed(int(args.seed) + int(fold_idx))

        train_ds = MultiStreamGestureDataset(
            args.json,
            fold_csv,
            split="train",
            cfg=ds_cfg,
            label2idx=label2idx,
            use_packed_cache=bool(checkpoint_arg(saved_args, "use_packed_skeleton_cache", False) and not checkpoint_arg(saved_args, "use_decoded_skeleton_cache", False)),
            packed_cache_dir=checkpoint_arg(saved_args, "packed_skeleton_cache_dir", "") or None,
            packed_cache_rebuild=False,
            use_decoded_cache=bool(checkpoint_arg(saved_args, "use_decoded_skeleton_cache", False)),
            decoded_cache_dir=checkpoint_arg(saved_args, "decoded_skeleton_cache_dir", "") or None,
            decoded_cache_rebuild=False,
        )
        holdout_ds = MultiStreamGestureDataset(
            args.json,
            fold_csv,
            split="val",
            cfg=DSConfig(**{**state["ds_cfg"], "augment": False}),
            label2idx=label2idx,
            use_packed_cache=bool(checkpoint_arg(saved_args, "use_packed_skeleton_cache", False) and not checkpoint_arg(saved_args, "use_decoded_skeleton_cache", False)),
            packed_cache_dir=checkpoint_arg(saved_args, "packed_skeleton_cache_dir", "") or None,
            packed_cache_rebuild=False,
            use_decoded_cache=bool(checkpoint_arg(saved_args, "use_decoded_skeleton_cache", False)),
            decoded_cache_dir=checkpoint_arg(saved_args, "decoded_skeleton_cache_dir", "") or None,
            decoded_cache_rebuild=False,
        )
        _apply_loader_profile(train_ds, holdout_ds, profile)
        weights = build_sample_weights(
            train_ds.samples,
            label2idx,
            train_ds._meta_by_vid,
            quality_floor=0.4,
            quality_power=1.0,
            cover_key="both_coverage",
            cover_floor=0.3,
        )
        weights_tensor = torch.tensor(weights, dtype=torch.double)
        train_loader, _, train_batch_sampler = _build_train_loader(
            train_ds=train_ds,
            label2idx=label2idx,
            weights_tensor=weights_tensor,
            args=loader_args,
            profile=profile,
            device=device,
            benchmark=False,
        )
        holdout_loader, _ = _build_val_loader(
            val_ds=holdout_ds,
            args=loader_args,
            profile=profile,
            device=device,
            benchmark=False,
        )

        model = build_model_from_saved_args(saved_args=saved_args, train_ds=train_ds, num_classes=len(label2idx)).to(device)
        if bool(checkpoint_arg(saved_args, "channels_last", False)) and device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        load_checkpoint_weights(model, state["payload"], prefer_ema=True)

        cls_counts = np.bincount([label2idx[lbl] for _, lbl, *_ in train_ds.samples], minlength=len(label2idx))
        cls_counts = np.maximum(cls_counts, 1)
        cls_counts = torch.tensor(cls_counts, dtype=torch.float32, device=device)
        if bool(checkpoint_arg(saved_args, "use_logit_adjustment", False)):
            criterion = LogitAdjustedCrossEntropyLoss(cls_counts)
        else:
            criterion = torch.nn.CrossEntropyLoss(
                label_smoothing=max(0.0, float(checkpoint_arg(saved_args, "label_smoothing", 0.0)))
            )
        supcon_criterion = (
            SupervisedContrastiveLoss(temperature=float(checkpoint_arg(saved_args, "supcon_temp", 0.07)))
            if bool(checkpoint_arg(saved_args, "use_supcon", False))
            else None
        )
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(checkpoint_arg(saved_args, "lr", 5e-4)),
            weight_decay=float(checkpoint_arg(saved_args, "wd", 5e-4)),
        )
        warmup_epochs = int(max(0, round(float(checkpoint_arg(saved_args, "warmup_frac", 0.1)) * int(args.fold_epochs))))
        if warmup_epochs >= int(args.fold_epochs):
            warmup_epochs = max(0, int(args.fold_epochs) - 1)
        if warmup_epochs > 0:
            sched = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
                    optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.fold_epochs) - warmup_epochs)),
                ],
                milestones=[warmup_epochs],
            )
        else:
            sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.fold_epochs)))
        scaler = amp.GradScaler("cuda") if use_amp else amp.GradScaler(enabled=False)
        ema = ModelEma(model, decay=float(checkpoint_arg(saved_args, "ema_decay", 0.0)), device=device) if float(checkpoint_arg(saved_args, "ema_decay", 0.0)) > 0 else None
        A = train_ds.build_adjacency(normalize=False).to(device)
        supcon_start_epoch = _resolve_supcon_start_epoch(
            total_epochs=int(args.fold_epochs),
            warmup_epochs=int(warmup_epochs),
            requested=int(checkpoint_arg(saved_args, "supcon_start_epoch", -1)),
        )

        for epoch in range(1, int(args.fold_epochs) + 1):
            if train_batch_sampler is not None and hasattr(train_batch_sampler, "set_epoch"):
                train_batch_sampler.set_epoch(int(epoch))
            train_out = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                criterion,
                device,
                accum_steps=int(checkpoint_arg(saved_args, "accum", 1)),
                grad_clip=float(checkpoint_arg(saved_args, "grad_clip", 1.0)),
                A=A,
                use_channels_last=bool(checkpoint_arg(saved_args, "channels_last", False)),
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                max_batches=(int(args.limit_train_batches) if int(args.limit_train_batches) > 0 else None),
                log_interval=int(checkpoint_arg(saved_args, "log_interval", 50)),
                ema=ema,
                tb_logger=None,
                tb_log_every=0,
                global_step=0,
                supcon_criterion=supcon_criterion,
                supcon_weight=float(checkpoint_arg(saved_args, "supcon_weight", 0.05)),
                supcon_start_epoch=int(supcon_start_epoch),
                epoch=int(epoch),
            )
            sched.step()
            print(f"[Fold {fold_idx:02d} Ep {epoch:03d}] train_loss={float(train_out.loss):.4f} total={float(train_out.total_loss):.4f}")

        infer_model = ema.module if ema is not None else model
        prediction_rows, feature_arr, kin_arr, logits_arr = _run_holdout_inference(
            model=infer_model,
            loader=holdout_loader,
            device=device,
            A=A,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            idx2label=idx2label,
            end_inclusive=end_inclusive,
            save_full_logits=bool(args.save_full_logits),
        )
        for i, row in enumerate(prediction_rows):
            row["fold"] = int(fold_idx)
            record_bundles.append(
                {
                    "row": dict(row),
                    "feature": feature_arr[i],
                    "kin": kin_arr[i],
                    "logits": (None if logits_arr is None else logits_arr[i]),
                }
            )

    record_bundles.sort(key=lambda item: str(item["row"]["sample_id"]))
    prediction_rows_sorted = [item["row"] for item in record_bundles]
    feature_matrix = np.stack([item["feature"] for item in record_bundles], axis=0).astype(np.float32, copy=False)
    kin_matrix = np.stack([item["kin"] for item in record_bundles], axis=0).astype(np.float32, copy=False)
    logits_matrix = (
        np.stack([item["logits"] for item in record_bundles], axis=0).astype(np.float32, copy=False)
        if bool(args.save_full_logits)
        else None
    )

    if len(prediction_rows_sorted) != len(samples):
        raise RuntimeError(f"Expected {len(samples)} OOF rows, got {len(prediction_rows_sorted)}")
    unique_ids = {row["sample_id"] for row in prediction_rows_sorted}
    if len(unique_ids) != len(samples):
        raise RuntimeError("OOF predictions do not contain exactly one unique sample_id per train sample")
    if feature_matrix.shape[0] != len(samples) or kin_matrix.shape[0] != len(samples):
        raise RuntimeError("OOF feature/kinematic arrays are misaligned with prediction rows")

    sample_ids = [str(row["sample_id"]) for row in prediction_rows_sorted]
    table_info = write_predictions_table(rows=prediction_rows_sorted, out_dir=out_dir, basename="oof_predictions")
    np.save(out_dir / "oof_features.npy", feature_matrix)
    np.save(out_dir / "oof_kinematics.npy", kin_matrix)
    features_shards = write_sharded_array(
        out_dir=out_dir,
        name="oof_features",
        sample_ids=sample_ids,
        array=feature_matrix,
        shard_size=int(args.array_shard_size),
    )
    kin_shards = write_sharded_array(
        out_dir=out_dir,
        name="oof_kinematics",
        sample_ids=sample_ids,
        array=kin_matrix,
        shard_size=int(args.array_shard_size),
    )
    if logits_matrix is not None:
        np.save(out_dir / "oof_logits.npy", logits_matrix)
        logits_shards = write_sharded_array(
            out_dir=out_dir,
            name="oof_logits",
            sample_ids=sample_ids,
            array=logits_matrix,
            shard_size=int(args.array_shard_size),
        )
    else:
        logits_shards = {}

    meta["feature_dim"] = int(feature_matrix.shape[1]) if feature_matrix.ndim == 2 else 0
    meta["kinematic_dim"] = int(kin_matrix.shape[1]) if kin_matrix.ndim == 2 else 0
    meta["num_records"] = int(len(prediction_rows_sorted))
    meta["predictions_table"] = table_info
    meta["sharded_arrays"] = {
        "oof_features": features_shards,
        "oof_kinematics": kin_shards,
        "oof_logits": logits_shards,
    }
    meta["loader_profile"] = {
        "workers": int(profile.workers),
        "persistent_workers": bool(profile.persistent_workers),
        "prefetch_factor": (None if profile.prefetch_factor is None else int(profile.prefetch_factor)),
        "file_cache_size": int(profile.file_cache_size),
        "use_prefetch_loader": bool(profile.use_prefetch_loader),
    }
    _write_json(out_dir / "oof_meta.json", meta)
    print(f"Done. Wrote OOF cache to {out_dir}")


if __name__ == "__main__":
    main()
