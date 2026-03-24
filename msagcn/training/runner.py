from __future__ import annotations

import csv
import gc
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from torch import amp
from torch.utils.data import DataLoader, WeightedRandomSampler

from msagcn.data import ClassBalancedBatchSampler, DSConfig, HybridSupConBatchSampler, MultiStreamGestureDataset, build_sample_weights
from msagcn.models import MultiStreamAGCN
from utils.tensorboard_logger import TensorboardLogger

from .ema import ModelEma
from .engine import evaluate, train_one_epoch
from .losses import LogitAdjustedCrossEntropyLoss, SupervisedContrastiveLoss
from .metrics import (
    _sanitize_label,
    build_confusion_pairs,
    build_confusion_image,
    format_confusion_pairs_text,
    format_examples_text,
    format_per_class_rows_text,
    format_prediction_rows_text,
    format_table_text,
)
from .prefetch import PrefetchLoader
from .utils import build_mirror_idx, select_hist_params, set_seed
from .auto_workers import (
    LoaderProfile,
    build_auto_workers_fingerprint,
    build_worker_candidates,
    choose_best_candidate,
    get_auto_workers_cache_path,
    load_auto_workers_record,
    profile_to_dict,
    resolve_loader_profile,
    save_auto_workers_record,
)


def _load_history_file(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: failed to read history file {path}: {exc}")
        return []
    return data if isinstance(data, list) else []


def _optimizer_to_device(optimizer: optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _diff_dict_keys(a: dict, b: dict) -> list[str]:
    mismatched = []
    for key in sorted(set(a.keys()) | set(b.keys())):
        if a.get(key) != b.get(key):
            mismatched.append(key)
    return mismatched


def _preview_keys(keys: list[str] | tuple[str, ...], limit: int = 8) -> str:
    items = list(keys)
    if not items:
        return ""
    preview = ", ".join(items[:limit])
    return preview + (" ..." if len(items) > limit else "")


def _build_checkpoint(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    sched,
    scaler,
    ema: ModelEma | None,
    label2idx: dict[str, int],
    ds_cfg_dict: dict,
    best_f1: float,
    epochs_no_improve: int,
    global_step: int,
    history: list[dict],
    analysis_state: dict | None,
    args,
) -> dict:
    return {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "sched_state": sched.state_dict(),
        "scaler_state": scaler.state_dict(),
        "ema_state": (ema.state_dict() if ema is not None else None),
        "label2idx": label2idx,
        "ds_cfg": ds_cfg_dict,
        "best_f1": float(best_f1),
        "epochs_no_improve": int(epochs_no_improve),
        "global_step": int(global_step),
        "history": history,
        "analysis_state": analysis_state or {},
        "args": vars(args),
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _compute_train_support(train_ds: MultiStreamGestureDataset, label2idx: dict[str, int]) -> np.ndarray:
    counts = np.zeros(len(label2idx), dtype=np.int64)
    for _, label, *_ in train_ds.samples:
        counts[int(label2idx[label])] += 1
    return counts


def _support_is_uniform(train_support: np.ndarray) -> bool:
    nonzero = np.asarray(train_support, dtype=np.int64)
    nonzero = nonzero[nonzero > 0]
    return bool(nonzero.size == 0 or np.unique(nonzero).size <= 1)


def _resolve_bucket_mode(train_support: np.ndarray, requested_mode: str) -> tuple[str, bool]:
    support_is_uniform = _support_is_uniform(train_support)
    if requested_mode == "auto":
        return ("difficulty" if support_is_uniform else "support"), support_is_uniform
    return str(requested_mode), support_is_uniform


def _bucket_mode_spec(bucket_mode: str) -> dict[str, Any]:
    if bucket_mode == "difficulty":
        return {
            "mode": "difficulty",
            "bucket_names": ("easy", "mid", "hard"),
            "focus_bucket": "hard",
            "focus_label": "hard",
            "secondary_bucket": "easy",
            "secondary_label": "easy",
            "tb_prefix": "val_difficulty",
        }
    return {
        "mode": "support",
        "bucket_names": ("head", "mid", "tail"),
        "focus_bucket": "tail",
        "focus_label": "tail",
        "secondary_bucket": "head",
        "secondary_label": "head",
        "tb_prefix": "val_bucket",
    }


def _build_support_bucket_rows(train_support: np.ndarray, idx2label: dict[int, str]) -> tuple[dict[int, str], list[dict[str, Any]]]:
    valid = [idx for idx, support in enumerate(train_support.tolist()) if int(support) > 0]
    order = sorted(valid, key=lambda idx: (-int(train_support[idx]), int(idx)))
    n_valid = len(order)
    head_end = max(1, int(np.ceil(0.20 * n_valid))) if n_valid > 0 else 0
    mid_end = max(head_end, int(np.ceil(0.50 * n_valid))) if n_valid > 0 else 0
    bucket_by_class: dict[int, str] = {}
    rows: list[dict[str, Any]] = []
    for rank, class_id in enumerate(order):
        if rank < head_end:
            bucket = "head"
        elif rank < mid_end:
            bucket = "mid"
        else:
            bucket = "tail"
        bucket_by_class[int(class_id)] = bucket
        rows.append(
            {
                "class_id": int(class_id),
                "label": str(idx2label.get(int(class_id), str(class_id))),
                "support_train": int(train_support[class_id]),
                "bucket": bucket,
                "bucket_mode": "support",
                "rank_by_metric": int(rank),
                "rank_by_support": int(rank),
                "difficulty_score": "",
            }
        )
    return bucket_by_class, rows


def _build_support_watchlist_rows(
    train_support: np.ndarray,
    idx2label: dict[int, str],
    bucket_by_class: dict[int, str],
    watchlist_k: int,
) -> list[dict[str, Any]]:
    valid = [idx for idx, support in enumerate(train_support.tolist()) if int(support) > 0]
    order = sorted(valid, key=lambda idx: (int(train_support[idx]), int(idx)))
    watch = order[: min(max(1, int(watchlist_k)), len(order))]
    return [
        {
            "class_id": int(class_id),
            "label": str(idx2label.get(int(class_id), str(class_id))),
            "support_train": int(train_support[class_id]),
            "bucket": str(bucket_by_class.get(int(class_id), "none")),
            "bucket_mode": "support",
            "difficulty_score": "",
            "watch_reason": "low_support",
        }
        for class_id in watch
    ]


def _update_difficulty_scores(
    score_state: dict[str, float],
    per_class: dict[int, dict[str, Any]],
    *,
    ema_decay: float,
) -> None:
    decay = float(np.clip(ema_decay, 0.0, 0.999))
    for class_id, row in per_class.items():
        if int(row.get("support_val", 0)) <= 0:
            continue
        key = str(int(class_id))
        current = float(row.get("f1", 0.0))
        prev = score_state.get(key)
        score_state[key] = current if prev is None else (decay * float(prev) + (1.0 - decay) * current)


def _build_difficulty_bucket_rows(
    score_state: dict[str, float],
    train_support: np.ndarray,
    idx2label: dict[int, str],
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    valid = [idx for idx, support in enumerate(train_support.tolist()) if int(support) > 0]
    order = sorted(valid, key=lambda idx: (-float(score_state.get(str(int(idx)), 0.0)), int(idx)))
    n_valid = len(order)
    easy_end = max(1, int(np.ceil(0.20 * n_valid))) if n_valid > 0 else 0
    mid_end = max(easy_end, int(np.ceil(0.50 * n_valid))) if n_valid > 0 else 0
    bucket_by_class: dict[int, str] = {}
    rows: list[dict[str, Any]] = []
    for rank, class_id in enumerate(order):
        if rank < easy_end:
            bucket = "easy"
        elif rank < mid_end:
            bucket = "mid"
        else:
            bucket = "hard"
        bucket_by_class[int(class_id)] = bucket
        rows.append(
            {
                "class_id": int(class_id),
                "label": str(idx2label.get(int(class_id), str(class_id))),
                "support_train": int(train_support[class_id]),
                "bucket": bucket,
                "bucket_mode": "difficulty",
                "rank_by_metric": int(rank),
                "rank_by_support": "",
                "difficulty_score": float(score_state.get(str(int(class_id)), 0.0)),
            }
        )
    return bucket_by_class, rows


def _build_difficulty_watchlist_rows(
    score_state: dict[str, float],
    train_support: np.ndarray,
    idx2label: dict[int, str],
    bucket_by_class: dict[int, str],
    watchlist_k: int,
) -> list[dict[str, Any]]:
    valid = [idx for idx, support in enumerate(train_support.tolist()) if int(support) > 0]
    order = sorted(valid, key=lambda idx: (float(score_state.get(str(int(idx)), 0.0)), int(idx)))
    watch = order[: min(max(1, int(watchlist_k)), len(order))]
    return [
        {
            "class_id": int(class_id),
            "label": str(idx2label.get(int(class_id), str(class_id))),
            "support_train": int(train_support[class_id]),
            "bucket": str(bucket_by_class.get(int(class_id), "none")),
            "bucket_mode": "difficulty",
            "difficulty_score": float(score_state.get(str(int(class_id)), 0.0)),
            "watch_reason": "low_difficulty_score",
        }
        for class_id in watch
    ]


def _bucket_aggregates(
    per_class: dict[int, dict[str, Any]],
    bucket_by_class: dict[int, str],
    bucket_names: tuple[str, ...],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for bucket in bucket_names:
        rows = [row for class_id, row in per_class.items() if bucket_by_class.get(int(class_id)) == bucket]
        if not rows:
            out[f"f1_{bucket}"] = 0.0
            out[f"p_{bucket}"] = 0.0
            out[f"r_{bucket}"] = 0.0
            out[f"zero_f1_{bucket}_count"] = 0.0
            continue
        f1_vals = [float(row["f1"]) for row in rows]
        p_vals = [float(row["precision"]) for row in rows]
        r_vals = [float(row["recall"]) for row in rows]
        out[f"f1_{bucket}"] = float(np.mean(f1_vals))
        out[f"p_{bucket}"] = float(np.mean(p_vals))
        out[f"r_{bucket}"] = float(np.mean(r_vals))
        out[f"zero_f1_{bucket}_count"] = float(sum(1 for value in f1_vals if value <= 1e-12))
    return out


def _compute_per_class_distribution_metrics(per_class: dict[int, dict[str, Any]]) -> dict[str, float]:
    rows = [row for row in per_class.values() if int(row.get("support_val", 0)) > 0]
    if not rows:
        return {
            "per_class_f1_p10": 0.0,
            "per_class_f1_p25": 0.0,
            "per_class_f1_median": 0.0,
            "per_class_recall_p10": 0.0,
        }
    f1_vals = np.asarray([float(row.get("f1", 0.0)) for row in rows], dtype=np.float64)
    recall_vals = np.asarray([float(row.get("recall", 0.0)) for row in rows], dtype=np.float64)
    return {
        "per_class_f1_p10": float(np.quantile(f1_vals, 0.10)),
        "per_class_f1_p25": float(np.quantile(f1_vals, 0.25)),
        "per_class_f1_median": float(np.median(f1_vals)),
        "per_class_recall_p10": float(np.quantile(recall_vals, 0.10)),
    }


def _write_bucket_artifacts(analysis_dir: Path, bucket_rows: list[dict[str, Any]], watchlist_rows: list[dict[str, Any]]) -> None:
    _write_csv(
        analysis_dir / "buckets.csv",
        ["class_id", "label", "support_train", "bucket", "bucket_mode", "rank_by_metric", "rank_by_support", "difficulty_score"],
        bucket_rows,
    )
    _write_csv(
        analysis_dir / "watchlist.csv",
        ["class_id", "label", "support_train", "bucket", "bucket_mode", "difficulty_score", "watch_reason"],
        watchlist_rows,
    )


def _apply_bucket_annotations(
    eval_out,
    *,
    idx2label: dict[int, str],
    bucket_by_class: dict[int, str],
) -> None:
    for class_id, row in eval_out.per_class.items():
        row["bucket"] = str(bucket_by_class.get(int(class_id), "none"))
    eval_out.confusion_pairs = build_confusion_pairs(
        eval_out.labels,
        eval_out.preds,
        idx2label,
        support_true=[int(v.get("support_val", 0)) for _, v in sorted(eval_out.per_class.items())] if eval_out.per_class else None,
        bucket_by_class=bucket_by_class,
    )


def _prediction_csv_rows(records: list[dict[str, Any]], train_support: np.ndarray, bucket_by_class: dict[int, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        true_id = int(rec["true_id"])
        rows.append(
            {
                "epoch": int(rec["epoch"]),
                "global_step": int(rec["global_step"]),
                "video": rec.get("video", ""),
                "t0": rec.get("t0", ""),
                "t1": rec.get("t1", ""),
                "true_id": true_id,
                "pred_id": int(rec["pred_id"]),
                "true_label": rec.get("true_label", ""),
                "pred_label": rec.get("pred_label", ""),
                "correct": int(rec.get("correct", 0)),
                "conf": float(rec.get("conf", 0.0)),
                "margin": float(rec.get("margin", 0.0)),
                "entropy": float(rec.get("entropy", 0.0)),
                "top5_ids": "|".join(str(x) for x in rec.get("top5_ids", [])),
                "top5_labels": "|".join(str(x) for x in rec.get("top5_labels", [])),
                "top5_probs": "|".join(f"{float(x):.6f}" for x in rec.get("top5_probs", [])),
                "support_train_true": int(train_support[true_id]) if true_id < len(train_support) else 0,
                "bucket_true": str(bucket_by_class.get(true_id, "none")),
            }
        )
    return rows


def _select_error_rows(rows: list[dict[str, Any]], *, bucket_filter: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    filtered = [row for row in rows if int(row.get("correct", 0)) == 0]
    if bucket_filter:
        filtered = [row for row in filtered if row.get("bucket_true") == bucket_filter]
    filtered.sort(key=lambda row: (-float(row.get("conf", 0.0)), -float(row.get("margin", 0.0)), str(row.get("video", ""))))
    return filtered[: max(1, int(limit))]


def _compute_biggest_late_drops(
    per_class: dict[int, dict[str, Any]],
    peak_state: dict[str, dict[str, Any]],
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_id, row in per_class.items():
        state = peak_state.get(str(int(class_id)))
        peak_f1 = float(state.get("peak_f1", row["f1"])) if state else float(row["f1"])
        peak_epoch = int(state.get("peak_epoch", row.get("epoch", 0))) if state else 0
        drop = float(row["f1"]) - peak_f1
        rows.append(
            {
                "class_id": int(class_id),
                "label": row.get("label", str(class_id)),
                "bucket": row.get("bucket", "none"),
                "support_train": int(row.get("support_train", 0)),
                "support_val": int(row.get("support_val", 0)),
                "f1": float(row.get("f1", 0.0)),
                "peak_f1": float(peak_f1),
                "peak_epoch": int(peak_epoch),
                "delta_vs_peak": float(drop),
            }
        )
    rows.sort(key=lambda row: (float(row["delta_vs_peak"]), int(row["support_train"]), int(row["class_id"])))
    return rows[: max(1, int(limit))]


def _update_peak_state(peak_state: dict[str, dict[str, Any]], per_class: dict[int, dict[str, Any]], epoch: int) -> None:
    for class_id, row in per_class.items():
        key = str(int(class_id))
        current = float(row.get("f1", 0.0))
        prev = peak_state.get(key)
        if prev is None or current > float(prev.get("peak_f1", -1.0)):
            peak_state[key] = {"peak_f1": float(current), "peak_epoch": int(epoch)}


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _module_tag_name(name: str) -> str:
    if not name:
        return "root"
    return name.replace(".", "/")


def _resolve_early_stop_start_epoch(*, total_epochs: int, warmup_epochs: int, patience: int, requested: int) -> int:
    total = max(1, int(total_epochs))
    if int(requested) > 0:
        return min(total, int(requested))
    warmup = max(0, int(warmup_epochs))
    burn_in = warmup + max(int(patience), warmup)
    return min(total, burn_in)


def _resolve_early_stop_patience(*, total_epochs: int, requested: int) -> int:
    if int(requested) == 0:
        return 0
    if int(requested) > 0:
        return int(requested)
    total = max(1, int(total_epochs))
    return int(min(12, max(8, int(np.ceil(total / 6.0)))))


def _resolve_early_stop_min_delta(*, total_epochs: int, requested: float) -> float:
    if float(requested) >= 0.0:
        return float(requested)
    total = max(1, int(total_epochs))
    auto = 1e-3 * float(np.sqrt(60.0 / float(total)))
    return float(np.clip(auto, 5e-4, 2e-3))


def _resolve_supcon_start_epoch(*, total_epochs: int, warmup_epochs: int, requested: int) -> int:
    total = max(1, int(total_epochs))
    req = int(requested)
    if req > 0:
        return min(total, req)
    if req == 0:
        return 1
    warmup = max(0, int(warmup_epochs))
    return min(total, warmup + 1)


def _collect_topology_scalars(model: nn.Module) -> dict[str, float]:
    scalars: dict[str, float] = {}
    for name, module in model.named_modules():
        stats = getattr(module, "last_topology_stats", None)
        if not isinstance(stats, dict) or not stats:
            continue
        prefix = f"topology/{_module_tag_name(name)}"
        for key, value in stats.items():
            try:
                scalars[f"{prefix}/{key}"] = float(value)
            except Exception:
                continue
    return scalars


def _batch_label_count_summary(label_ids: list[int]) -> dict[str, int]:
    counts = Counter(int(x) for x in label_ids)
    repeated = [int(v) for v in counts.values() if int(v) > 1]
    singletons = [int(v) for v in counts.values() if int(v) == 1]
    return {
        "unique_classes": int(len(counts)),
        "min_count": int(min(counts.values()) if counts else 0),
        "max_count": int(max(counts.values()) if counts else 0),
        "repeated_classes": int(len(repeated)),
        "repeated_min": int(min(repeated) if repeated else 0),
        "repeated_max": int(max(repeated) if repeated else 0),
        "singleton_classes": int(len(singletons)),
    }


def _detect_cache_mode(*, args, is_dir_mode: bool) -> str:
    if not is_dir_mode:
        return "none"
    if bool(args.use_decoded_skeleton_cache):
        return "decoded"
    if bool(args.use_packed_skeleton_cache):
        return "packed"
    return "none"


def _train_sampler_mode(args) -> str:
    if bool(args.supcon_class_balanced_batch):
        return "class_balanced"
    if bool(getattr(args, "supcon_mixed_batch", False)):
        return "mixed"
    if bool(args.weighted_sampler):
        return "weighted"
    return "random"


def _loader_profile_from_record(record: dict[str, Any] | None) -> LoaderProfile | None:
    if not isinstance(record, dict):
        return None
    payload = record.get("profile")
    if not isinstance(payload, dict):
        return None
    try:
        return LoaderProfile(
            workers=int(payload.get("workers", 0)),
            persistent_workers=bool(payload.get("persistent_workers", False)),
            prefetch_factor=(None if payload.get("prefetch_factor") is None else int(payload.get("prefetch_factor"))),
            file_cache_size=int(payload.get("file_cache_size", 0)),
            pin_memory=bool(payload.get("pin_memory", False)),
            pin_memory_device=(str(payload["pin_memory_device"]) if payload.get("pin_memory_device") else None),
            use_prefetch_loader=bool(payload.get("use_prefetch_loader", False)),
            notes=tuple(payload.get("notes", ())),
        )
    except Exception:
        return None


def _apply_loader_profile(
    train_ds: MultiStreamGestureDataset,
    val_ds: MultiStreamGestureDataset | None,
    profile: LoaderProfile,
) -> None:
    train_ds.cfg.file_cache_size = int(profile.file_cache_size)
    if val_ds is not None:
        val_ds.cfg.file_cache_size = int(profile.file_cache_size)


def _build_loader_kwargs(
    *,
    profile: LoaderProfile,
    collate_fn,
    benchmark: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": int(profile.workers),
        "persistent_workers": bool(profile.persistent_workers),
        "collate_fn": collate_fn,
        "pin_memory": bool(profile.pin_memory),
    }
    if profile.pin_memory_device:
        kwargs["pin_memory_device"] = profile.pin_memory_device
    if profile.prefetch_factor is not None and int(profile.workers) > 0:
        kwargs["prefetch_factor"] = int(profile.prefetch_factor)
    if benchmark and int(profile.workers) > 0:
        kwargs["timeout"] = 20
    return kwargs


def _build_train_loader(
    *,
    train_ds: MultiStreamGestureDataset,
    label2idx: dict[str, int],
    weights_tensor: torch.Tensor,
    args,
    profile: LoaderProfile,
    device: torch.device,
    benchmark: bool = False,
) -> tuple[DataLoader | PrefetchLoader, DataLoader, Any | None]:
    dl_worker_kwargs = _build_loader_kwargs(
        profile=profile,
        collate_fn=MultiStreamGestureDataset.collate_fn,
        benchmark=benchmark,
    )
    class_balanced_batch_sampler = (
        ClassBalancedBatchSampler(
            train_ds.samples,
            label2idx,
            classes_per_batch=int(args.supcon_classes_per_batch),
            samples_per_class=int(args.supcon_samples_per_class),
            seed=int(args.seed),
        )
        if args.supcon_class_balanced_batch
        else None
    )
    mixed_batch_sampler = (
        HybridSupConBatchSampler(
            train_ds.samples,
            label2idx,
            batch_size=int(args.batch),
            repeated_classes_per_batch=int(args.supcon_mixed_repeated_classes),
            repeated_samples_per_class=int(args.supcon_mixed_repeated_samples),
            seed=int(args.seed),
        )
        if getattr(args, "supcon_mixed_batch", False)
        else None
    )
    batch_sampler = class_balanced_batch_sampler if class_balanced_batch_sampler is not None else mixed_batch_sampler
    if batch_sampler is not None:
        base_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            **dl_worker_kwargs,
        )
    else:
        sampler = (
            WeightedRandomSampler(weights_tensor, num_samples=int(weights_tensor.numel()), replacement=True)
            if args.weighted_sampler
            else None
        )
        generator = torch.Generator()
        generator.manual_seed(int(args.seed))
        base_loader = DataLoader(
            train_ds,
            batch_size=int(args.batch),
            sampler=sampler if sampler is not None else None,
            shuffle=False if sampler is not None else True,
            drop_last=True,
            generator=generator if sampler is None else None,
            **dl_worker_kwargs,
        )
    loader: DataLoader | PrefetchLoader = base_loader
    if profile.use_prefetch_loader:
        loader = PrefetchLoader(base_loader, device)
    return loader, base_loader, batch_sampler


def _build_val_loader(
    *,
    val_ds: MultiStreamGestureDataset,
    args,
    profile: LoaderProfile,
    device: torch.device,
    benchmark: bool = False,
) -> tuple[DataLoader | PrefetchLoader, DataLoader]:
    dl_worker_kwargs = _build_loader_kwargs(
        profile=profile,
        collate_fn=MultiStreamGestureDataset.collate_fn,
        benchmark=benchmark,
    )
    base_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch),
        shuffle=False,
        **dl_worker_kwargs,
    )
    loader: DataLoader | PrefetchLoader = base_loader
    if profile.use_prefetch_loader:
        loader = PrefetchLoader(base_loader, device)
    return loader, base_loader


def _shutdown_loader_workers(loader: DataLoader | PrefetchLoader | None) -> None:
    if loader is None:
        return
    base = loader.loader if isinstance(loader, PrefetchLoader) else loader
    iterator = getattr(base, "_iterator", None)
    if iterator is not None:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass


def _benchmark_train_loader_profile(
    *,
    train_ds: MultiStreamGestureDataset,
    label2idx: dict[str, int],
    weights_tensor: torch.Tensor,
    args,
    profile: LoaderProfile,
    device: torch.device,
) -> dict[str, Any]:
    loader: DataLoader | PrefetchLoader | None = None
    base_loader: DataLoader | None = None
    train_batch_sampler: Any | None = None
    warmup_batches = max(0, int(args.auto_workers_warmup_batches))
    measure_batches = max(1, int(args.auto_workers_measure_batches))
    total_target = warmup_batches + measure_batches
    result: dict[str, Any] = {
        "workers": int(profile.workers),
        "success": False,
        "samples_per_sec": 0.0,
        "batch_per_sec": 0.0,
        "mean_batch_time": 0.0,
        "first_batch_time": 0.0,
        "measured_batches": 0,
        "measured_samples": 0,
        "persistent_workers": bool(profile.persistent_workers),
        "prefetch_factor": (int(profile.prefetch_factor) if profile.prefetch_factor is not None else 0),
        "file_cache_size": int(profile.file_cache_size),
        "use_prefetch_loader": bool(profile.use_prefetch_loader),
        "pin_memory": bool(profile.pin_memory),
        "error": "",
        "notes": " | ".join(str(note) for note in profile.notes),
    }
    try:
        _apply_loader_profile(train_ds, None, profile)
        loader, base_loader, train_batch_sampler = _build_train_loader(
            train_ds=train_ds,
            label2idx=label2idx,
            weights_tensor=weights_tensor,
            args=args,
            profile=profile,
            device=device,
            benchmark=True,
        )
        if train_batch_sampler is not None and hasattr(train_batch_sampler, "set_epoch"):
            train_batch_sampler.set_epoch(0)
        iterator = iter(loader)
        measured_batches = 0
        measured_samples = 0
        measured_time = 0.0
        first_batch_time = None
        for batch_idx in range(total_target):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _, y, _ = next(iterator)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dt = float(time.perf_counter() - t0)
            if first_batch_time is None:
                first_batch_time = dt
            if batch_idx >= warmup_batches:
                measured_batches += 1
                measured_samples += int(y.size(0))
                measured_time += dt
        if measured_batches <= 0 or measured_time <= 0.0:
            raise RuntimeError("benchmark did not observe any measured batches")
        result["success"] = True
        result["measured_batches"] = int(measured_batches)
        result["measured_samples"] = int(measured_samples)
        result["first_batch_time"] = float(first_batch_time or 0.0)
        result["mean_batch_time"] = float(measured_time / measured_batches)
        result["batch_per_sec"] = float(measured_batches / measured_time)
        result["samples_per_sec"] = float(measured_samples / measured_time)
    except StopIteration:
        result["error"] = "loader exhausted before enough benchmark batches were collected"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _shutdown_loader_workers(loader)
        _shutdown_loader_workers(base_loader)
        del loader
        del base_loader
        del train_batch_sampler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def _resolve_auto_workers(
    *,
    train_ds: MultiStreamGestureDataset,
    label2idx: dict[str, int],
    weights_tensor: torch.Tensor,
    args,
    device: torch.device,
    json_path: Path,
    is_dir_mode: bool,
    cache_mode: str,
) -> tuple[LoaderProfile, dict[str, Any]]:
    sampler_mode = _train_sampler_mode(args)
    fingerprint_payload, fingerprint_hash = build_auto_workers_fingerprint(
        args=args,
        json_path=json_path,
        dataset_mode=("dir" if is_dir_mode else "combined"),
        cache_mode=cache_mode,
        train_size=len(train_ds),
        sampler_mode=sampler_mode,
    )
    info: dict[str, Any] = {
        "enabled": True,
        "from_cache": False,
        "fingerprint": fingerprint_payload,
        "fingerprint_hash": fingerprint_hash,
        "cache_hit": False,
        "cache_path": "",
        "fallback_reason": "",
        "candidates": [],
        "selected_candidate": {},
        "profile": {},
        "benchmark_samples_per_sec": 0.0,
    }

    cached_record = None if args.auto_workers_rebench else load_auto_workers_record(fingerprint_hash)
    cached_profile = _loader_profile_from_record(cached_record)
    if cached_profile is not None:
        info["from_cache"] = True
        info["cache_hit"] = True
        info["cache_path"] = str(get_auto_workers_cache_path())
        info["profile"] = profile_to_dict(cached_profile)
        info["selected_candidate"] = dict(cached_record.get("benchmark", {})) if isinstance(cached_record, dict) else {}
        info["candidates"] = list(cached_record.get("candidates", [])) if isinstance(cached_record, dict) else []
        info["benchmark_samples_per_sec"] = float(info["selected_candidate"].get("samples_per_sec", 0.0))
        return cached_profile, info

    if args.auto_workers_rebench:
        print("Auto-workers: cache bypass requested -> benchmarking loader candidates again.")
    else:
        print("Auto-workers: cache miss -> benchmarking loader candidates.")
    cpu_count = int(os.cpu_count() or 1)
    candidate_workers = build_worker_candidates(
        auto_workers_max=int(args.auto_workers_max),
        cpu_count=cpu_count,
        os_name=os.name,
        is_dir_mode=is_dir_mode,
        cache_mode=cache_mode,
    )
    print(f"Auto-workers candidates: {candidate_workers}")
    for requested_workers in candidate_workers:
        profile = resolve_loader_profile(
            requested_workers=int(requested_workers),
            requested_prefetch=int(args.prefetch),
            requested_file_cache_size=int(args.file_cache),
            os_name=os.name,
            is_dir_mode=is_dir_mode,
            cache_mode=cache_mode,
            device_type=device.type,
            no_prefetch=bool(args.no_prefetch),
            cuda_index=(torch.cuda.current_device() if device.type == "cuda" else None),
        )
        print(
            "Auto-workers bench | "
            f"requested={requested_workers} -> workers={profile.workers}, "
            f"persistent={int(profile.persistent_workers)}, "
            f"prefetch={profile.prefetch_factor if profile.prefetch_factor is not None else 0}, "
            f"file_cache={profile.file_cache_size}"
        )
        row = _benchmark_train_loader_profile(
            train_ds=train_ds,
            label2idx=label2idx,
            weights_tensor=weights_tensor,
            args=args,
            profile=profile,
            device=device,
        )
        row["requested_workers"] = int(requested_workers)
        row["profile"] = profile_to_dict(profile)
        info["candidates"].append(row)
        if row.get("success"):
            print(
                "  -> success | "
                f"samples_per_sec={float(row['samples_per_sec']):.2f} | "
                f"batch_per_sec={float(row['batch_per_sec']):.2f} | "
                f"first_batch_time={float(row['first_batch_time']):.3f}s"
            )
        else:
            print(f"  -> failed | {row.get('error', 'unknown error')}")

    selected = choose_best_candidate(info["candidates"])
    if selected is not None:
        chosen_profile = LoaderProfile(**selected["profile"])
        info["selected_candidate"] = {k: v for k, v in selected.items() if k != "profile"}
        info["profile"] = profile_to_dict(chosen_profile)
        info["benchmark_samples_per_sec"] = float(selected.get("samples_per_sec", 0.0))
        print(
            "Auto-workers selected | "
            f"workers={chosen_profile.workers}, "
            f"persistent={int(chosen_profile.persistent_workers)}, "
            f"prefetch={chosen_profile.prefetch_factor if chosen_profile.prefetch_factor is not None else 0}, "
            f"file_cache={chosen_profile.file_cache_size}, "
            f"samples_per_sec={float(selected.get('samples_per_sec', 0.0)):.2f}"
        )
    else:
        info["fallback_reason"] = "no successful benchmark candidates; using safe fallback workers=0"
        chosen_profile = resolve_loader_profile(
            requested_workers=0,
            requested_prefetch=int(args.prefetch),
            requested_file_cache_size=int(args.file_cache),
            os_name=os.name,
            is_dir_mode=is_dir_mode,
            cache_mode=cache_mode,
            device_type=device.type,
            no_prefetch=bool(args.no_prefetch),
            cuda_index=(torch.cuda.current_device() if device.type == "cuda" else None),
        )
        info["profile"] = profile_to_dict(chosen_profile)
        print(f"Auto-workers fallback | {info['fallback_reason']}")

    cache_record = {
        "version": 1,
        "timestamp_unix": int(time.time()),
        "profile": info["profile"],
        "benchmark": dict(info["selected_candidate"]),
        "candidates": list(info["candidates"]),
    }
    cache_path = save_auto_workers_record(fingerprint_hash, cache_record)
    info["cache_path"] = str(cache_path)
    return chosen_profile, info


def run_training(args) -> None:
    set_seed(args.seed)

    debug_mode = (args.overfit_batches > 0) or (args.limit_train_batches > 0)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = out_dir / "analysis"
    predictions_dir = analysis_dir / "predictions"
    errors_dir = analysis_dir / "errors"
    confusion_dir = analysis_dir / "confusion"
    per_class_dir = analysis_dir / "per_class"
    for path in (analysis_dir, predictions_dir, errors_dir, confusion_dir, per_class_dir):
        path.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name.strip() or out_dir.name
    tb_logger = TensorboardLogger(
        log_dir=args.logdir,
        run_name=run_name,
        enabled=bool(args.tensorboard),
        flush_secs=int(args.flush_secs),
    )
    tb_path = ""
    if tb_logger.enabled:
        tb_path = str(tb_logger.log_dir.resolve())
        print(f"TensorBoard logs: {tb_path}")
        print(f"Run: tensorboard --logdir \"{tb_path}\"")
        with (out_dir / "tensorboard_logdir.txt").open("w", encoding="utf-8") as f:
            f.write(str(tb_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| cudnn.benchmark =", (device.type == "cuda"))
    torch.backends.cudnn.benchmark = device.type == "cuda"
    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.float32
    use_amp = (device.type == "cuda") and (not args.no_amp)

    # Dataset config
    augment_flag = args.augment
    time_drop_prob = args.time_drop_prob
    hand_drop_prob = args.hand_drop_prob
    normalize_flag = args.normalize
    if debug_mode and args.augment and not args.keep_aug_in_debug:
        print("Debug mode detected -> disabling dataset augmentations/time-drop/hand-drop for reproducibility.")
        augment_flag = False
        time_drop_prob = 0.0
        hand_drop_prob = 0.0
    if debug_mode and args.normalize and args.disable_norm_in_debug:
        print("Debug mode detected -> disabling dataset normalization for reproducibility (flagged).")
        normalize_flag = False

    connect_cross_edges = bool(args.connect_cross_edges)
    if args.include_pose and not args.no_cross_edges:
        connect_cross_edges = True
    ds_cfg = DSConfig(
        max_frames=args.max_frames,
        end_inclusive=(not args.end_is_exclusive),
        temporal_crop=str(args.temporal_crop),
        use_streams=tuple(s.strip() for s in args.streams.split(",") if s.strip()),
        include_pose=args.include_pose,
        pose_keep=tuple(int(x) for x in args.pose_keep.split(",") if x.strip()),
        pose_vis_thr=args.pose_vis_thr,
        connect_cross_edges=connect_cross_edges,
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
        boundary_jitter_prob=args.boundary_jitter_prob,
        boundary_jitter_max=args.boundary_jitter_max,
        speed_perturb_prob=args.speed_perturb_prob,
        speed_perturb_kmin=args.speed_perturb_kmin,
        speed_perturb_kmax=args.speed_perturb_kmax,
        file_cache_size=args.file_cache,
        prefer_pp=args.prefer_pp,
    )
    # Keep the checkpoint-compatible dataset config separate from runtime loader tweaks.
    ds_cfg_dict = asdict(ds_cfg)

    # Datasets
    if args.use_decoded_skeleton_cache and args.use_packed_skeleton_cache:
        print("Note: --use_decoded_skeleton_cache supersedes --use_packed_skeleton_cache for dataset loading.")
    packed_cache_dir = args.packed_skeleton_cache_dir.strip() or None
    decoded_cache_dir = args.decoded_skeleton_cache_dir.strip() or None
    train_ds = MultiStreamGestureDataset(
        args.json,
        args.csv,
        split="train",
        cfg=ds_cfg,
        use_packed_cache=bool(args.use_packed_skeleton_cache and not args.use_decoded_skeleton_cache),
        packed_cache_dir=packed_cache_dir,
        packed_cache_rebuild=bool(args.packed_skeleton_cache_rebuild),
        use_decoded_cache=bool(args.use_decoded_skeleton_cache),
        decoded_cache_dir=decoded_cache_dir,
        decoded_cache_rebuild=bool(args.decoded_skeleton_cache_rebuild),
    )
    label2idx = train_ds.label2idx  # single label map from train
    idx2label = {v: k for k, v in label2idx.items()}
    val_ds = MultiStreamGestureDataset(
        args.json,
        args.csv,
        split="val",
        cfg=DSConfig(**{**asdict(ds_cfg), "augment": False}),
        label2idx=label2idx,
        use_packed_cache=bool(args.use_packed_skeleton_cache and not args.use_decoded_skeleton_cache),
        packed_cache_dir=packed_cache_dir,
        packed_cache_rebuild=False,
        use_decoded_cache=bool(args.use_decoded_skeleton_cache),
        decoded_cache_dir=decoded_cache_dir,
        decoded_cache_rebuild=False,
    )

    # Detect storage mode (dir vs combined)
    json_path = Path(args.json)
    is_dir_mode = json_path.is_dir()
    if args.use_packed_skeleton_cache and not is_dir_mode:
        print("Note: --use_packed_skeleton_cache is only used for per-video JSON directories; ignoring it for combined JSON.")
    if args.use_decoded_skeleton_cache and not is_dir_mode:
        print("Note: --use_decoded_skeleton_cache is only used for per-video JSON directories; ignoring it for combined JSON.")
    requested_workers = int(args.workers)
    cache_mode = _detect_cache_mode(args=args, is_dir_mode=is_dir_mode)
    dataset_mode_name = "dir" if is_dir_mode else "combined"
    if args.auto_workers:
        print(f"Dataset mode: {dataset_mode_name} | requested_workers={requested_workers} | auto_workers=on")
    else:
        print(f"Dataset mode: {dataset_mode_name} | workers={requested_workers}")

    # Save label map & ds cfg
    with (out_dir / "label2idx.json").open("w", encoding="utf-8") as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)
    with (out_dir / "ds_config.json").open("w", encoding="utf-8") as f:
        json.dump(ds_cfg_dict, f, ensure_ascii=False, indent=2)

    train_support = _compute_train_support(train_ds, label2idx)
    _write_csv(
        analysis_dir / "train_support.csv",
        ["class_id", "label", "support_train"],
        [
            {
                "class_id": int(class_id),
                "label": str(idx2label.get(int(class_id), str(class_id))),
                "support_train": int(train_support[class_id]),
            }
            for class_id in range(len(train_support))
        ],
    )

    # Sampler weights / train batch composition
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

    auto_workers_info: dict[str, Any] = {
        "enabled": False,
        "from_cache": False,
        "fingerprint": {},
        "fingerprint_hash": "",
        "cache_hit": False,
        "cache_path": "",
        "fallback_reason": "",
        "candidates": [],
        "selected_candidate": {},
        "profile": {},
        "benchmark_samples_per_sec": 0.0,
    }
    if args.auto_workers:
        effective_profile, auto_workers_info = _resolve_auto_workers(
            train_ds=train_ds,
            label2idx=label2idx,
            weights_tensor=weights_tensor,
            args=args,
            device=device,
            json_path=json_path,
            is_dir_mode=is_dir_mode,
            cache_mode=cache_mode,
        )
        if auto_workers_info.get("from_cache"):
            print(f"Auto-workers cache hit: {auto_workers_info.get('cache_path', '')}")
    else:
        effective_profile = resolve_loader_profile(
            requested_workers=requested_workers,
            requested_prefetch=int(args.prefetch),
            requested_file_cache_size=int(args.file_cache),
            os_name=os.name,
            is_dir_mode=is_dir_mode,
            cache_mode=cache_mode,
            device_type=device.type,
            no_prefetch=bool(args.no_prefetch),
            cuda_index=(torch.cuda.current_device() if device.type == "cuda" else None),
        )
    _apply_loader_profile(train_ds, val_ds, effective_profile)
    for note in effective_profile.notes:
        print(f"Note: {note}")
    print(
        "Loader profile: "
        f"workers={effective_profile.workers} | "
        f"persistent={int(effective_profile.persistent_workers)} | "
        f"prefetch={effective_profile.prefetch_factor if effective_profile.prefetch_factor is not None else 0} | "
        f"file_cache={effective_profile.file_cache_size} | "
        f"prefetch_loader={int(effective_profile.use_prefetch_loader)}"
    )

    candidate_fieldnames = [
        "requested_workers",
        "workers",
        "success",
        "samples_per_sec",
        "batch_per_sec",
        "mean_batch_time",
        "first_batch_time",
        "measured_batches",
        "measured_samples",
        "persistent_workers",
        "prefetch_factor",
        "file_cache_size",
        "use_prefetch_loader",
        "pin_memory",
        "error",
        "notes",
    ]
    if args.auto_workers:
        _write_json(analysis_dir / "auto_workers_decision.json", auto_workers_info)
        _write_csv(
            analysis_dir / "auto_workers_candidates.csv",
            candidate_fieldnames,
            [
                {key: row.get(key, "") for key in candidate_fieldnames}
                for row in auto_workers_info.get("candidates", [])
            ],
        )

    # Reset RNG after loader autotune so the real training run starts from the configured seed.
    if args.auto_workers:
        set_seed(args.seed)

    train_loader, train_loader_base, train_batch_sampler = _build_train_loader(
        train_ds=train_ds,
        label2idx=label2idx,
        weights_tensor=weights_tensor,
        args=args,
        profile=effective_profile,
        device=device,
        benchmark=False,
    )
    if train_batch_sampler is not None:
        preview_batch = next(iter(train_batch_sampler))
        preview_labels = [int(label2idx[train_ds.samples[idx][1]]) for idx in preview_batch]
        preview_summary = _batch_label_count_summary(preview_labels)
        if args.supcon_class_balanced_batch:
            print(
                "Using class-balanced batch sampler: "
                f"{int(args.supcon_classes_per_batch)} classes x {int(args.supcon_samples_per_class)} samples "
                f"(batch={int(args.batch)}, batches/epoch~{len(train_batch_sampler)})."
            )
            print(
                "Class-balanced preview | "
                f"unique_classes={preview_summary['unique_classes']} | "
                f"min_count={preview_summary['min_count']} | "
                f"max_count={preview_summary['max_count']}"
            )
        elif args.supcon_mixed_batch:
            singleton_slots = int(args.batch) - int(args.supcon_mixed_repeated_classes * args.supcon_mixed_repeated_samples)
            print(
                "Using mixed SupCon batch sampler: "
                f"{int(args.supcon_mixed_repeated_classes)} repeated classes x {int(args.supcon_mixed_repeated_samples)} samples "
                f"+ {singleton_slots} singleton classes "
                f"(batch={int(args.batch)}, batches/epoch~{len(train_batch_sampler)})."
            )
            print(
                "Mixed batch preview | "
                f"unique_classes={preview_summary['unique_classes']} | "
                f"repeated_classes={preview_summary['repeated_classes']} | "
                f"repeated_min={preview_summary['repeated_min']} | "
                f"repeated_max={preview_summary['repeated_max']} | "
                f"singleton_classes={preview_summary['singleton_classes']}"
            )
    val_loader, val_loader_base = _build_val_loader(
        val_ds=val_ds,
        args=args,
        profile=effective_profile,
        device=device,
        benchmark=False,
    )

    # Model
    depths = tuple(int(x) for x in args.depths.split(","))
    temp_ks = tuple(int(x) for x in args.temp_ks.split(","))
    model_drop = args.drop
    model_stream_drop = args.stream_drop_p
    if debug_mode and not args.keep_aug_in_debug:
        if (model_drop > 0) or (model_stream_drop > 0):
            print("Debug mode detected -> forcing model dropout rates to 0.")
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
        use_ctr_hand_refine=args.use_ctr_hand_refine,
        ctr_in_stream_encoder=args.ctr_in_stream_encoder,
        ctr_groups=args.ctr_groups,
        ctr_hand_nodes=args.ctr_hand_nodes,
        ctr_rel_channels=args.ctr_rel_channels,
        ctr_alpha_init=args.ctr_alpha_init,
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
    supcon_criterion = SupervisedContrastiveLoss(temperature=args.supcon_temp) if args.use_supcon else None

    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warmup_epochs = int(max(0, round(args.warmup_frac * args.epochs)))
    if warmup_epochs >= args.epochs:
        warmup_epochs = max(0, args.epochs - 1)
    supcon_start_epoch = _resolve_supcon_start_epoch(
        total_epochs=int(args.epochs),
        warmup_epochs=int(warmup_epochs),
        requested=int(args.supcon_start_epoch),
    )
    if warmup_epochs > 0:
        main_epochs = max(1, args.epochs - warmup_epochs)
        sched = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
                optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=main_epochs),
            ],
            milestones=[warmup_epochs],
        )
    else:
        sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
        )
    scaler = amp.GradScaler("cuda") if use_amp else amp.GradScaler(enabled=False)
    ema = ModelEma(model, decay=args.ema_decay, device=device) if args.ema_decay > 0 else None

    # Training
    if args.overfit_batches > 0:
        train_limit = val_limit = args.overfit_batches
    else:
        train_limit = args.limit_train_batches if args.limit_train_batches > 0 else None
        val_limit = args.limit_val_batches if args.limit_val_batches > 0 else None
    if train_limit or val_limit:
        print(f"Debug limits -> train:{train_limit or 'full'} | val:{val_limit or 'full'}")

    best_f1 = -1.0
    es_patience = _resolve_early_stop_patience(total_epochs=int(args.epochs), requested=int(args.early_stop_patience))
    es_min_delta = _resolve_early_stop_min_delta(total_epochs=int(args.epochs), requested=float(args.early_stop_min_delta))
    es_start_epoch = _resolve_early_stop_start_epoch(
        total_epochs=int(args.epochs),
        warmup_epochs=int(warmup_epochs),
        patience=int(es_patience),
        requested=int(args.early_stop_start_epoch),
    )
    epochs_no_improve = 0
    history = []
    global_step = 0
    start_epoch = 1
    analysis_state: dict[str, Any] = {
        "per_class_peak_f1": {},
        "epoch_index": [],
        "difficulty_scores": {},
        "difficulty_frozen": False,
        "difficulty_freeze_epoch": 0,
        "bucket_mode": "",
        "support_is_uniform": False,
    }

    if args.resume:
        resume_path = Path(args.resume).expanduser()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        ckpt_label2idx = ckpt.get("label2idx")
        if ckpt_label2idx is not None and ckpt_label2idx != label2idx:
            raise ValueError(
                "Checkpoint label2idx does not match the current train split. "
                "Use the same dataset/split or start a fresh run."
            )
        ckpt_ds_cfg = ckpt.get("ds_cfg")
        if (not args.resume_model_only) and isinstance(ckpt_ds_cfg, dict) and ckpt_ds_cfg != ds_cfg_dict:
            diff_keys = _diff_dict_keys(ckpt_ds_cfg, ds_cfg_dict)
            preview = ", ".join(diff_keys[:8])
            suffix = " ..." if len(diff_keys) > 8 else ""
            raise ValueError(
                "Checkpoint dataset config does not match the current run. "
                f"Mismatched keys: {preview}{suffix}. "
                "Use identical data flags or pass --resume_model_only."
            )

        if args.resume_model_only:
            missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
            if missing or unexpected:
                print(
                    "Model-only resume loaded with non-strict state_dict matching. "
                    f"missing=[{_preview_keys(missing)}] unexpected=[{_preview_keys(unexpected)}]"
                )
            if ema is not None:
                ema_state = ckpt.get("ema_state")
                if ema_state is not None:
                    ema_missing, ema_unexpected = ema.module.load_state_dict(ema_state, strict=False)
                    if ema_missing or ema_unexpected:
                        print(
                            "EMA model-only resume loaded with non-strict state_dict matching. "
                            f"missing=[{_preview_keys(ema_missing)}] unexpected=[{_preview_keys(ema_unexpected)}]"
                        )
                else:
                    ema.module.load_state_dict(model.state_dict())
            print(f"Loaded model weights from {resume_path} (model-only resume).")
        else:
            model.load_state_dict(ckpt["model_state"], strict=True)
            if ema is not None:
                if ckpt.get("ema_state") is not None:
                    ema.load_state_dict(ckpt["ema_state"])
                else:
                    ema.module.load_state_dict(model.state_dict())
            elif ckpt.get("ema_state") is not None:
                print("Warning: checkpoint contains EMA weights, but current run has ema disabled; ignoring ema_state.")
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
                _optimizer_to_device(optimizer, device)
            else:
                print("Warning: optimizer_state missing in checkpoint; optimizer will start fresh.")
            if "sched_state" in ckpt:
                sched.load_state_dict(ckpt["sched_state"])
            else:
                print("Warning: sched_state missing in checkpoint; scheduler will start fresh.")
            if "scaler_state" in ckpt:
                try:
                    scaler.load_state_dict(ckpt["scaler_state"])
                except Exception as exc:
                    print(f"Warning: failed to restore GradScaler state: {exc}")

            history = ckpt.get("history")
            if not isinstance(history, list):
                history = _load_history_file(resume_path.parent / "history.json")
            analysis_state = ckpt.get("analysis_state") if isinstance(ckpt.get("analysis_state"), dict) else analysis_state
            analysis_state.setdefault("per_class_peak_f1", {})
            analysis_state.setdefault("epoch_index", [])
            analysis_state.setdefault("difficulty_scores", {})
            analysis_state.setdefault("difficulty_frozen", False)
            analysis_state.setdefault("difficulty_freeze_epoch", 0)
            analysis_state.setdefault("bucket_mode", "")
            analysis_state.setdefault("support_is_uniform", False)
            best_f1 = float(
                ckpt.get(
                    "best_f1",
                    max((row.get("val_f1", -1.0) for row in history), default=-1.0),
                )
            )
            epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
            global_step = int(ckpt.get("global_step", 0))
            if global_step <= 0 and history:
                steps_per_epoch = int(train_limit) if train_limit is not None else len(train_loader)
                global_step = len(history) * max(1, steps_per_epoch)
            start_epoch = int(ckpt.get("epoch", 0)) + 1

            saved_args = ckpt.get("args")
            if isinstance(saved_args, dict):
                saved_total_epochs = saved_args.get("epochs")
                if saved_total_epochs is not None and int(saved_total_epochs) != int(args.epochs):
                    print(
                        "Warning: --epochs differs from the checkpoint's original training plan. "
                        "Scheduler state will continue from the saved schedule."
                    )
            if start_epoch > int(args.epochs):
                raise ValueError(
                    f"Checkpoint is already at epoch {start_epoch - 1}, but --epochs={args.epochs}. "
                    "Pass a larger total epoch count to continue training."
                )
            if start_epoch <= es_start_epoch and epochs_no_improve != 0:
                print(
                    f"Resetting epochs_no_improve from checkpoint because early stopping is not armed until epoch {es_start_epoch}."
                )
                epochs_no_improve = 0
            print(
                f"Resuming full training state from {resume_path} | "
                f"next_epoch={start_epoch} | best_f1={best_f1:.4f} | global_step={global_step}"
            )

    bucket_mode, support_is_uniform = _resolve_bucket_mode(train_support, str(args.tb_bucket_mode))
    if analysis_state.get("bucket_mode") and str(analysis_state.get("bucket_mode")) != bucket_mode:
        print(
            "Warning: checkpoint analytics bucket mode differs from the current run. "
            f"saved={analysis_state.get('bucket_mode')} current={bucket_mode}"
        )
    analysis_state["bucket_mode"] = bucket_mode
    analysis_state["support_is_uniform"] = bool(support_is_uniform)
    bucket_spec = _bucket_mode_spec(bucket_mode)
    if bucket_mode == "difficulty":
        if support_is_uniform:
            print("Note: train support is uniform -> switching TensorBoard buckets to difficulty mode.")
        bucket_by_class, bucket_rows = _build_difficulty_bucket_rows(
            analysis_state.setdefault("difficulty_scores", {}),
            train_support,
            idx2label,
        )
        watchlist_rows = _build_difficulty_watchlist_rows(
            analysis_state.setdefault("difficulty_scores", {}),
            train_support,
            idx2label,
            bucket_by_class,
            int(args.tb_watchlist_k),
        )
    else:
        bucket_by_class, bucket_rows = _build_support_bucket_rows(train_support, idx2label)
        watchlist_rows = _build_support_watchlist_rows(train_support, idx2label, bucket_by_class, int(args.tb_watchlist_k))
    watchlist_ids = [int(row["class_id"]) for row in watchlist_rows]
    _write_bucket_artifacts(analysis_dir, bucket_rows, watchlist_rows)

    hist_params = select_hist_params(model, ("stems.", "head."))

    run_manifest = {
        "run_name": run_name,
        "out_dir": str(out_dir.resolve()),
        "tb_log_dir": tb_path,
        "resume_from": str(Path(args.resume).expanduser().resolve()) if args.resume else "",
        "label2idx_path": str((out_dir / "label2idx.json").resolve()),
        "best_ckpt_path": str((out_dir / "best.ckpt").resolve()),
        "last_ckpt_path": str((out_dir / "last.ckpt").resolve()),
        "tb_full_logging_enabled": bool(args.tb_full_logging),
        "bucket_mode_requested": str(args.tb_bucket_mode),
        "bucket_mode_effective": bucket_mode,
        "support_is_uniform": bool(support_is_uniform),
        "early_stop": {
            "patience_requested": int(args.early_stop_patience),
            "patience_effective": int(es_patience),
            "min_delta_requested": float(args.early_stop_min_delta),
            "min_delta_effective": float(es_min_delta),
            "start_epoch_requested": int(args.early_stop_start_epoch),
            "start_epoch_effective": int(es_start_epoch),
            "warmup_epochs": int(warmup_epochs),
        },
        "features": {
            "tensorboard": bool(args.tensorboard),
            "use_packed_skeleton_cache": bool(args.use_packed_skeleton_cache),
            "use_decoded_skeleton_cache": bool(args.use_decoded_skeleton_cache),
            "auto_workers": bool(args.auto_workers),
            "use_supcon": bool(args.use_supcon),
            "supcon_class_balanced_batch": bool(args.supcon_class_balanced_batch),
            "supcon_mixed_batch": bool(args.supcon_mixed_batch),
            "tb_log_all_classes": bool(args.tb_log_all_classes),
            "tb_log_tail_buckets": bool(args.tb_log_tail_buckets),
            "tb_log_confusion_pairs": bool(args.tb_log_confusion_pairs),
            "tb_log_predictions_csv": bool(args.tb_log_predictions_csv),
            "tb_log_errors_csv": bool(args.tb_log_errors_csv),
            "tb_log_topology": bool(args.tb_log_topology),
            "tb_log_confusion": bool(args.tb_log_confusion),
            "tb_log_examples": bool(args.tb_log_examples),
        },
        "tb_flags": {
            "tb_watchlist_k": int(args.tb_watchlist_k),
            "tb_bucket_mode": str(args.tb_bucket_mode),
            "tb_difficulty_freeze_epoch": int(args.tb_difficulty_freeze_epoch),
            "tb_difficulty_ema": float(args.tb_difficulty_ema),
            "tb_confusion_every": int(args.tb_confusion_every),
            "tb_predictions_every": int(args.tb_predictions_every),
            "tb_tables_k": int(args.tb_tables_k),
        },
        "supcon": {
            "enabled": bool(args.use_supcon),
            "weight": float(args.supcon_weight),
            "temperature": float(args.supcon_temp),
            "start_epoch_requested": int(args.supcon_start_epoch),
            "start_epoch_effective": int(supcon_start_epoch),
            "class_balanced_batch": bool(args.supcon_class_balanced_batch),
            "classes_per_batch": int(args.supcon_classes_per_batch),
            "samples_per_class": int(args.supcon_samples_per_class),
            "mixed_batch": bool(args.supcon_mixed_batch),
            "mixed_repeated_classes": int(args.supcon_mixed_repeated_classes),
            "mixed_repeated_samples": int(args.supcon_mixed_repeated_samples),
        },
        "auto_workers": {
            "enabled": bool(args.auto_workers),
            "from_cache": bool(auto_workers_info.get("from_cache", False)),
            "effective_workers": int(effective_profile.workers),
            "profile": profile_to_dict(effective_profile),
            "fingerprint_hash": str(auto_workers_info.get("fingerprint_hash", "")),
            "candidates_tested": int(len(auto_workers_info.get("candidates", []))),
            "benchmark_samples_per_sec": float(auto_workers_info.get("benchmark_samples_per_sec", 0.0)),
            "fallback_reason": str(auto_workers_info.get("fallback_reason", "")),
            "cache_path": str(auto_workers_info.get("cache_path", "")),
        },
    }
    _write_json(analysis_dir / "run_manifest.json", run_manifest)
    analysis_artifacts_enabled = bool(
        args.tb_full_logging
        or args.tb_log_all_classes
        or args.tb_log_tail_buckets
        or args.tb_log_confusion_pairs
        or args.tb_log_predictions_csv
        or args.tb_log_errors_csv
        or args.tb_log_topology
    )
    if tb_logger.enabled:
        tb_logger.scalar("meta/num_classes", float(len(label2idx)), 0)
        tb_logger.scalar("meta/train_samples", float(len(train_ds)), 0)
        tb_logger.scalar("meta/val_samples", float(len(val_ds)), 0)
        tb_logger.scalar("meta/tb_full_logging", float(bool(args.tb_full_logging)), 0)
        tb_logger.scalar("meta/early_stop_patience", float(es_patience), 0)
        tb_logger.scalar("meta/early_stop_min_delta", float(es_min_delta), 0)
        tb_logger.scalar("meta/early_stop_start_epoch", float(es_start_epoch), 0)
        tb_logger.scalar("meta/support_is_uniform", float(bool(support_is_uniform)), 0)
        tb_logger.scalar("meta/bucket_mode_support", float(bucket_mode == "support"), 0)
        tb_logger.scalar("meta/bucket_mode_difficulty", float(bucket_mode == "difficulty"), 0)
        tb_logger.scalar("meta/auto_workers_enabled", float(bool(args.auto_workers)), 0)
        tb_logger.scalar("meta/auto_workers_effective", float(effective_profile.workers), 0)
        tb_logger.scalar("meta/auto_workers_from_cache", float(bool(auto_workers_info.get("from_cache", False))), 0)
        tb_logger.scalar(
            "meta/auto_workers_benchmark_samples_per_sec",
            float(auto_workers_info.get("benchmark_samples_per_sec", 0.0)),
            0,
        )
        tb_logger.scalar("meta/auto_workers_profile_persistent", float(bool(effective_profile.persistent_workers)), 0)
        tb_logger.scalar(
            "meta/auto_workers_profile_prefetch",
            float(effective_profile.prefetch_factor if effective_profile.prefetch_factor is not None else 0),
            0,
        )
        tb_logger.scalar("meta/use_supcon", float(bool(args.use_supcon)), 0)
        tb_logger.scalar("meta/supcon_weight", float(args.supcon_weight), 0)
        tb_logger.scalar("meta/supcon_temp", float(args.supcon_temp), 0)
        tb_logger.scalar("meta/supcon_start_epoch", float(supcon_start_epoch), 0)
        tb_logger.scalar("meta/supcon_class_balanced_batch", float(bool(args.supcon_class_balanced_batch)), 0)
        tb_logger.scalar("meta/supcon_classes_per_batch", float(args.supcon_classes_per_batch), 0)
        tb_logger.scalar("meta/supcon_samples_per_class", float(args.supcon_samples_per_class), 0)
        tb_logger.scalar("meta/supcon_mixed_batch", float(bool(args.supcon_mixed_batch)), 0)
        tb_logger.scalar("meta/supcon_mixed_repeated_classes", float(args.supcon_mixed_repeated_classes), 0)
        tb_logger.scalar("meta/supcon_mixed_repeated_samples", float(args.supcon_mixed_repeated_samples), 0)
        tb_logger.text("meta/run_manifest", json.dumps(run_manifest, ensure_ascii=False, indent=2), 0)

    if es_patience > 0:
        print(
            f"Early stopping armed from epoch {es_start_epoch} "
            f"(warmup_epochs={warmup_epochs}, patience={es_patience}, min_delta={es_min_delta:.6f})."
        )
    else:
        print("Early stopping disabled.")
    if args.use_supcon:
        print(
            f"SupCon auxiliary armed from epoch {supcon_start_epoch} "
            f"(warmup_epochs={warmup_epochs}, requested_start={int(args.supcon_start_epoch)})."
        )

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        if train_batch_sampler is not None and hasattr(train_batch_sampler, "set_epoch"):
            train_batch_sampler.set_epoch(int(epoch))
        train_out = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            device,
            accum_steps=args.accum,
            grad_clip=args.grad_clip,
            A=A,
            use_channels_last=args.channels_last,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            max_batches=train_limit,
            log_interval=args.log_interval,
            ema=ema,
            tb_logger=tb_logger,
            tb_log_every=int(args.log_every_steps),
            global_step=global_step,
            supcon_criterion=supcon_criterion,
            supcon_weight=float(args.supcon_weight),
            supcon_start_epoch=int(supcon_start_epoch),
            epoch=int(epoch),
            log_batch_label_stats=bool(args.supcon_class_balanced_batch or args.supcon_mixed_batch),
            expected_batch_unique_classes=(int(args.supcon_classes_per_batch) if args.supcon_class_balanced_batch else None),
            expected_batch_samples_per_class=(int(args.supcon_samples_per_class) if args.supcon_class_balanced_batch else None),
            expected_mixed_repeated_classes=(int(args.supcon_mixed_repeated_classes) if args.supcon_mixed_batch else None),
            expected_mixed_repeated_samples=(int(args.supcon_mixed_repeated_samples) if args.supcon_mixed_batch else None),
            expected_mixed_singleton_classes=(
                int(args.batch - (args.supcon_mixed_repeated_classes * args.supcon_mixed_repeated_samples))
                if args.supcon_mixed_batch
                else None
            ),
        )
        tr_loss = float(train_out.loss)
        tr_total_loss = float(train_out.total_loss)
        global_step = int(train_out.global_step)
        eval_model = ema.module if ema is not None else model
        collect_examples = bool(
            tb_logger.enabled and args.tb_log_examples and (epoch % max(1, int(args.tb_examples_every)) == 0)
        )
        eval_out = evaluate(
            eval_model,
            val_loader,
            criterion,
            device,
            A=A,
            mirror_idx=mirror_idx,
            use_channels_last=args.channels_last,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            max_batches=val_limit,
            topk=(3, 5),
            collect_examples=collect_examples,
            examples_k=int(args.tb_examples_k),
            epoch=epoch,
            global_step=global_step,
            idx2label=idx2label,
            train_support=train_support,
            bucket_by_class=bucket_by_class,
        )
        if bucket_mode == "difficulty":
            difficulty_scores = analysis_state.setdefault("difficulty_scores", {})
            difficulty_frozen = bool(analysis_state.get("difficulty_frozen", False))
            if not difficulty_frozen:
                _update_difficulty_scores(
                    difficulty_scores,
                    eval_out.per_class,
                    ema_decay=float(args.tb_difficulty_ema),
                )
                if epoch >= int(args.tb_difficulty_freeze_epoch):
                    difficulty_frozen = True
                    analysis_state["difficulty_freeze_epoch"] = int(epoch)
            analysis_state["difficulty_frozen"] = bool(difficulty_frozen)
            bucket_by_class, bucket_rows = _build_difficulty_bucket_rows(difficulty_scores, train_support, idx2label)
            watchlist_rows = _build_difficulty_watchlist_rows(
                difficulty_scores,
                train_support,
                idx2label,
                bucket_by_class,
                int(args.tb_watchlist_k),
            )
            watchlist_ids = [int(row["class_id"]) for row in watchlist_rows]
            _apply_bucket_annotations(eval_out, idx2label=idx2label, bucket_by_class=bucket_by_class)
            _write_bucket_artifacts(analysis_dir, bucket_rows, watchlist_rows)
        sched.step()
        val_loss = float(eval_out.loss)
        val_acc = float(eval_out.acc)
        val_f1 = float(eval_out.macro_f1)
        topk_acc = eval_out.topk_acc
        val_preds = eval_out.preds
        val_labels = eval_out.labels
        examples = eval_out.examples

        dt = time.time() - t0
        print(
            f"[Ep {epoch:03d}] TL={tr_loss:.4f} | VL={val_loss:.4f} | VA={val_acc:.3f} | VF1={val_f1:.3f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | {dt:.1f}s"
        )
        print("pred top5:", Counter(val_preds).most_common(5))

        f1_micro = f1_score(val_labels, val_preds, average="micro") if val_labels else 0.0
        f1_weighted = f1_score(val_labels, val_preds, average="weighted") if val_labels else 0.0
        p_macro, r_macro, _, _ = (
            precision_recall_fscore_support(val_labels, val_preds, average="macro", zero_division=0)
            if val_labels
            else (0.0, 0.0, 0.0, None)
        )
        bucket_metrics = _bucket_aggregates(eval_out.per_class, bucket_by_class, tuple(bucket_spec["bucket_names"]))
        per_class_dist = _compute_per_class_distribution_metrics(eval_out.per_class)
        zero_f1_count = int(sum(1 for row in eval_out.per_class.values() if float(row["f1"]) <= 1e-12))
        nonzero_f1_count = int(len(eval_out.per_class) - zero_f1_count)
        conf_arr = np.asarray(eval_out.probs_top1, dtype=np.float64) if eval_out.probs_top1 else np.zeros(0, dtype=np.float64)
        margin_arr = np.asarray(eval_out.margin, dtype=np.float64) if eval_out.margin else np.zeros(0, dtype=np.float64)
        entropy_arr = np.asarray(eval_out.entropy, dtype=np.float64) if eval_out.entropy else np.zeros(0, dtype=np.float64)
        correct_arr = np.asarray([int(p == t) for p, t in zip(val_preds, val_labels)], dtype=np.int64) if val_labels else np.zeros(0, dtype=np.int64)
        high_conf_wrong_mask = (correct_arr == 0) & (conf_arr >= 0.90) if conf_arr.size else np.zeros(0, dtype=bool)
        high_conf_wrong_count = int(high_conf_wrong_mask.sum()) if high_conf_wrong_mask.size else 0
        high_conf_wrong_rate = float(high_conf_wrong_count / max(1, len(val_labels))) if val_labels else 0.0
        confidence_mean = float(conf_arr.mean()) if conf_arr.size else 0.0
        confidence_correct_mean = float(conf_arr[correct_arr == 1].mean()) if np.any(correct_arr == 1) else 0.0
        confidence_wrong_mean = float(conf_arr[correct_arr == 0].mean()) if np.any(correct_arr == 0) else 0.0
        prob_margin_mean = float(margin_arr.mean()) if margin_arr.size else 0.0
        prob_margin_wrong_mean = float(margin_arr[correct_arr == 0].mean()) if np.any(correct_arr == 0) else 0.0
        entropy_mean = float(entropy_arr.mean()) if entropy_arr.size else 0.0
        f1_gain_vs_prev = float(val_f1 - history[-1]["val_f1"]) if history else 0.0
        loss_gap_train_minus_val = float(tr_loss - val_loss)
        f1_gap_macro_minus_weighted = float(val_f1 - f1_weighted)
        es_armed = bool(es_patience > 0 and epoch >= es_start_epoch)

        history.append(
            dict(
                epoch=epoch,
                train_loss=tr_loss,
                train_total_loss=tr_total_loss,
                train_supcon_loss=float(train_out.supcon_loss),
                train_supcon_valid_anchors=int(train_out.supcon_valid_anchors),
                train_supcon_positive_pairs=int(train_out.supcon_positive_pairs),
                val_loss=val_loss,
                val_acc=val_acc,
                val_f1=val_f1,
                val_bucket_mode=bucket_mode,
                val_f1_tail=(float(bucket_metrics.get("f1_tail", 0.0)) if bucket_mode == "support" else None),
                val_f1_hard=(float(bucket_metrics.get("f1_hard", 0.0)) if bucket_mode == "difficulty" else None),
                val_ece=float(eval_out.calibration.get("ece_15bin", 0.0)),
                val_brier=float(eval_out.calibration.get("brier", 0.0)),
                val_nll=float(eval_out.calibration.get("nll", 0.0)),
            )
        )

        per_class_rows = []
        for class_id, row in sorted(eval_out.per_class.items()):
            row_out = dict(row)
            row_out["epoch"] = int(epoch)
            row_out["global_step"] = int(global_step)
            row_out["bucket_mode"] = bucket_mode
            per_class_rows.append(row_out)
        worst_classes_rows = sorted(per_class_rows, key=lambda row: (float(row["f1"]), int(row["support_train"]), int(row["class_id"])))
        biggest_late_drop_rows = _compute_biggest_late_drops(
            eval_out.per_class,
            analysis_state.get("per_class_peak_f1", {}),
            limit=int(args.tb_tables_k),
        )
        prediction_rows = _prediction_csv_rows(eval_out.records, train_support, bucket_by_class)
        error_rows = [row for row in prediction_rows if int(row["correct"]) == 0]
        worst_error_rows = _select_error_rows(prediction_rows, bucket_filter=None, limit=int(args.tb_tables_k))
        focus_error_rows = _select_error_rows(
            prediction_rows,
            bucket_filter=str(bucket_spec["focus_bucket"]),
            limit=int(args.tb_tables_k),
        )
        confusion_pairs_top = eval_out.confusion_pairs[: max(1, int(args.tb_tables_k))]
        confusion_pairs_tail = [
            row for row in eval_out.confusion_pairs if str(row.get("bucket_true")) == str(bucket_spec["focus_bucket"])
        ][: max(1, int(args.tb_tables_k))]
        watchlist_id_set = set(watchlist_ids)
        confusion_pairs_watchlist = [
            row for row in eval_out.confusion_pairs if int(row.get("true_id", -1)) in watchlist_id_set
        ][: max(1, int(args.tb_tables_k))]

        if tb_logger.enabled:
            tb_logger.scalar("meta/early_stop_armed", float(es_armed), global_step)
            tb_logger.scalar("val/loss", float(val_loss), global_step)
            tb_logger.scalar("val/acc", float(val_acc), global_step)
            tb_logger.scalar("val/f1_macro", float(val_f1), global_step)
            if 3 in topk_acc:
                tb_logger.scalar("val/acc_top3", float(topk_acc[3]), global_step)
            if 5 in topk_acc:
                tb_logger.scalar("val/acc_top5", float(topk_acc[5]), global_step)
            if val_labels:
                tb_logger.scalar("val/f1_micro", float(f1_micro), global_step)
                tb_logger.scalar("val/f1_weighted", float(f1_weighted), global_step)
                tb_logger.scalar("val/ece_15bin", float(eval_out.calibration.get("ece_15bin", 0.0)), global_step)
                tb_logger.scalar("val/brier", float(eval_out.calibration.get("brier", 0.0)), global_step)
                tb_logger.scalar("val/nll", float(eval_out.calibration.get("nll", 0.0)), global_step)
                tb_logger.scalar("val/confidence_mean", confidence_mean, global_step)
                tb_logger.scalar("val/confidence_correct_mean", confidence_correct_mean, global_step)
                tb_logger.scalar("val/confidence_wrong_mean", confidence_wrong_mean, global_step)
                tb_logger.scalar("val/prob_margin_mean", prob_margin_mean, global_step)
                tb_logger.scalar("val/prob_margin_wrong_mean", prob_margin_wrong_mean, global_step)
                tb_logger.scalar("val/entropy_mean", entropy_mean, global_step)
                tb_logger.scalar("val/per_class_f1_p10", float(per_class_dist["per_class_f1_p10"]), global_step)
                tb_logger.scalar("val/per_class_f1_p25", float(per_class_dist["per_class_f1_p25"]), global_step)
                tb_logger.scalar("val/per_class_f1_median", float(per_class_dist["per_class_f1_median"]), global_step)
                tb_logger.scalar("val/per_class_recall_p10", float(per_class_dist["per_class_recall_p10"]), global_step)
                tb_logger.scalar("val/high_conf_wrong_count", float(high_conf_wrong_count), global_step)
                tb_logger.scalar("val/high_conf_wrong_rate", float(high_conf_wrong_rate), global_step)
                tb_logger.scalar("val/precision_macro", float(p_macro), global_step)
                tb_logger.scalar("val/recall_macro", float(r_macro), global_step)
                tb_logger.scalar("val/zero_f1_count", float(zero_f1_count), global_step)
                tb_logger.scalar("val/nonzero_f1_count", float(nonzero_f1_count), global_step)
                tb_logger.scalar("val/f1_gap_macro_minus_weighted", float(f1_gap_macro_minus_weighted), global_step)
                tb_logger.scalar("val/f1_gain_vs_prev", float(f1_gain_vs_prev), global_step)
                tb_logger.scalar("val/loss_gap_train_minus_val", float(loss_gap_train_minus_val), global_step)
                topk_support = int(max(0, args.tb_support_topk))
                if topk_support > 0:
                    order = sorted(
                        (row for row in per_class_rows if int(row["support_val"]) > 0),
                        key=lambda row: (-int(row["support_val"]), int(row["class_id"])),
                    )
                    for row in order[: min(topk_support, len(order))]:
                        idx = int(row["class_id"])
                        label_name = _sanitize_label(idx2label.get(idx, str(idx)))
                        tb_logger.scalar(f"val/support/{label_name}", float(row["support_val"]), global_step)
                        tb_logger.scalar(f"val/p_class/{label_name}", float(row["precision"]), global_step)
                        tb_logger.scalar(f"val/r_class/{label_name}", float(row["recall"]), global_step)
                        tb_logger.scalar(f"val/f1_class/{label_name}", float(row["f1"]), global_step)
                worstk = int(max(0, args.tb_worstk_f1))
                if worstk > 0:
                    for row in worst_classes_rows[: min(worstk, len(worst_classes_rows))]:
                        idx = int(row["class_id"])
                        label_name = _sanitize_label(idx2label.get(idx, str(idx)))
                        tb_logger.scalar(f"val/f1_worst/{label_name}", float(row["f1"]), global_step)
                        tb_logger.scalar(f"val/p_worst/{label_name}", float(row["precision"]), global_step)
                        tb_logger.scalar(f"val/r_worst/{label_name}", float(row["recall"]), global_step)
                if args.tb_log_tail_buckets or args.tb_full_logging:
                    tb_logger.scalar("meta/difficulty_frozen", float(bool(analysis_state.get("difficulty_frozen", False))), global_step)
                    tb_logger.scalar("meta/bucket_mode_support", float(bucket_mode == "support"), global_step)
                    tb_logger.scalar("meta/bucket_mode_difficulty", float(bucket_mode == "difficulty"), global_step)
                    if bucket_mode == "support":
                        tb_logger.scalar("val_bucket/f1_head", float(bucket_metrics.get("f1_head", 0.0)), global_step)
                        tb_logger.scalar("val_bucket/f1_mid", float(bucket_metrics.get("f1_mid", 0.0)), global_step)
                        tb_logger.scalar("val_bucket/f1_tail", float(bucket_metrics.get("f1_tail", 0.0)), global_step)
                        tb_logger.scalar("val_bucket/p_head", float(bucket_metrics.get("p_head", 0.0)), global_step)
                        tb_logger.scalar("val_bucket/p_mid", float(bucket_metrics.get("p_mid", 0.0)), global_step)
                        tb_logger.scalar("val_bucket/p_tail", float(bucket_metrics.get("p_tail", 0.0)), global_step)
                        tb_logger.scalar("val_bucket/r_head", float(bucket_metrics.get("r_head", 0.0)), global_step)
                        tb_logger.scalar("val_bucket/r_mid", float(bucket_metrics.get("r_mid", 0.0)), global_step)
                        tb_logger.scalar("val_bucket/r_tail", float(bucket_metrics.get("r_tail", 0.0)), global_step)
                        tb_logger.scalar("val_bucket/zero_f1_head_count", float(bucket_metrics.get("zero_f1_head_count", 0.0)), global_step)
                        tb_logger.scalar("val_bucket/zero_f1_mid_count", float(bucket_metrics.get("zero_f1_mid_count", 0.0)), global_step)
                        tb_logger.scalar("val_bucket/zero_f1_tail_count", float(bucket_metrics.get("zero_f1_tail_count", 0.0)), global_step)
                    else:
                        tb_logger.scalar("val_difficulty/f1_easy", float(bucket_metrics.get("f1_easy", 0.0)), global_step)
                        tb_logger.scalar("val_difficulty/f1_mid", float(bucket_metrics.get("f1_mid", 0.0)), global_step)
                        tb_logger.scalar("val_difficulty/f1_hard", float(bucket_metrics.get("f1_hard", 0.0)), global_step)
                        tb_logger.scalar("val_difficulty/p_easy", float(bucket_metrics.get("p_easy", 0.0)), global_step)
                        tb_logger.scalar("val_difficulty/p_mid", float(bucket_metrics.get("p_mid", 0.0)), global_step)
                        tb_logger.scalar("val_difficulty/p_hard", float(bucket_metrics.get("p_hard", 0.0)), global_step)
                        tb_logger.scalar("val_difficulty/r_easy", float(bucket_metrics.get("r_easy", 0.0)), global_step)
                        tb_logger.scalar("val_difficulty/r_mid", float(bucket_metrics.get("r_mid", 0.0)), global_step)
                        tb_logger.scalar("val_difficulty/r_hard", float(bucket_metrics.get("r_hard", 0.0)), global_step)
                        tb_logger.scalar("val_difficulty/zero_f1_easy_count", float(bucket_metrics.get("zero_f1_easy_count", 0.0)), global_step)
                        tb_logger.scalar("val_difficulty/zero_f1_mid_count", float(bucket_metrics.get("zero_f1_mid_count", 0.0)), global_step)
                        tb_logger.scalar("val_difficulty/zero_f1_hard_count", float(bucket_metrics.get("zero_f1_hard_count", 0.0)), global_step)
                if args.tb_log_all_classes or args.tb_full_logging:
                    for row in per_class_rows:
                        class_id = int(row["class_id"])
                        tb_logger.scalar(f"val_all/f1/{class_id}", float(row["f1"]), global_step)
                        tb_logger.scalar(f"val_all/p/{class_id}", float(row["precision"]), global_step)
                        tb_logger.scalar(f"val_all/r/{class_id}", float(row["recall"]), global_step)
                        tb_logger.scalar(f"val_all/support_val/{class_id}", float(row["support_val"]), global_step)
                        tb_logger.scalar(f"val_all/support_train/{class_id}", float(row["support_train"]), global_step)
                        tb_logger.scalar(f"val_all/pred_count/{class_id}", float(row["pred_count"]), global_step)
                        tb_logger.scalar(f"val_all/true_count/{class_id}", float(row["true_count"]), global_step)
                        tb_logger.scalar(f"val_all/conf_mean/{class_id}", float(row["conf_mean"]), global_step)
                        tb_logger.scalar(f"val_all/conf_wrong_mean/{class_id}", float(row["conf_wrong_mean"]), global_step)
                        tb_logger.scalar(f"val_all/margin_mean/{class_id}", float(row["margin_mean"]), global_step)
                    for class_id in watchlist_ids:
                        row = eval_out.per_class.get(int(class_id))
                        if row is None:
                            continue
                        tb_logger.scalar(f"val_watch/f1/{class_id}", float(row["f1"]), global_step)
                        tb_logger.scalar(f"val_watch/p/{class_id}", float(row["precision"]), global_step)
                        tb_logger.scalar(f"val_watch/r/{class_id}", float(row["recall"]), global_step)
                        tb_logger.scalar(f"val_watch/conf_mean/{class_id}", float(row["conf_mean"]), global_step)
                        tb_logger.scalar(f"val_watch/margin_mean/{class_id}", float(row["margin_mean"]), global_step)
            log_confusion_now = epoch % max(1, int(args.tb_confusion_every)) == 0
            if (args.tb_log_confusion or args.tb_full_logging) and val_labels and log_confusion_now:
                try:
                    img, _ = build_confusion_image(
                        val_labels, val_preds, idx2label, topk=int(args.tb_confusion_topk), normalize=True
                    )
                    tb_logger.image("val/confusion_topk", img, global_step, dataformats="HWC")
                    secondary_ids = [
                        class_id
                        for class_id, bucket in bucket_by_class.items()
                        if bucket == str(bucket_spec["secondary_bucket"])
                    ]
                    focus_ids = [
                        class_id
                        for class_id, bucket in bucket_by_class.items()
                        if bucket == str(bucket_spec["focus_bucket"])
                    ]
                    img_secondary, _ = build_confusion_image(
                        val_labels,
                        val_preds,
                        idx2label,
                        topk=int(args.tb_confusion_topk),
                        normalize=True,
                        class_ids=secondary_ids,
                    )
                    tb_logger.image(
                        f"val/confusion_{bucket_spec['secondary_label']}",
                        img_secondary,
                        global_step,
                        dataformats="HWC",
                    )
                    img_focus, _ = build_confusion_image(
                        val_labels,
                        val_preds,
                        idx2label,
                        topk=int(args.tb_confusion_topk),
                        normalize=True,
                        class_ids=focus_ids,
                    )
                    tb_logger.image(
                        f"val/confusion_{bucket_spec['focus_label']}",
                        img_focus,
                        global_step,
                        dataformats="HWC",
                    )
                    tb_logger.image("val/confusion_focus", img_focus, global_step, dataformats="HWC")
                    img_watch, _ = build_confusion_image(
                        val_labels, val_preds, idx2label, topk=int(args.tb_confusion_topk), normalize=True, class_ids=watchlist_ids
                    )
                    tb_logger.image("val/confusion_watchlist", img_watch, global_step, dataformats="HWC")
                except Exception as exc:
                    print(f"Warning: failed to log confusion image to TensorBoard: {exc}")
            if collect_examples and examples:
                wrong = [ex for ex in examples if int(ex["pred"]) != int(ex["true"])]
                wrong.sort(key=lambda x: x.get("conf", 1.0))
                selected = wrong
                if len(selected) < int(args.tb_examples_k):
                    rest = [ex for ex in examples if int(ex["pred"]) == int(ex["true"])]
                    rest.sort(key=lambda x: x.get("conf", 1.0))
                    selected = selected + rest
                text = format_examples_text(selected, idx2label, int(args.tb_examples_k))
                tb_logger.text("val/examples", text, global_step)
            if args.tb_log_confusion_pairs or args.tb_full_logging:
                tb_logger.text("tables/confusion_pairs_top", format_confusion_pairs_text(confusion_pairs_top, max_rows=int(args.tb_tables_k)), global_step)
                tb_logger.text(
                    "tables/confusion_pairs_focus",
                    format_confusion_pairs_text(confusion_pairs_tail, max_rows=int(args.tb_tables_k)),
                    global_step,
                )
                if bucket_mode == "support":
                    tb_logger.text(
                        "tables/confusion_pairs_tail",
                        format_confusion_pairs_text(confusion_pairs_tail, max_rows=int(args.tb_tables_k)),
                        global_step,
                    )
                tb_logger.text(
                    "tables/confusion_pairs_watchlist",
                    format_confusion_pairs_text(confusion_pairs_watchlist, max_rows=int(args.tb_tables_k)),
                    global_step,
                )
            if analysis_artifacts_enabled:
                tb_logger.text("tables/worst_classes", format_per_class_rows_text(worst_classes_rows, max_rows=int(args.tb_tables_k)), global_step)
                tb_logger.text("tables/worst_errors", format_prediction_rows_text(worst_error_rows, max_rows=int(args.tb_tables_k)), global_step)
                tb_logger.text("tables/focus_errors", format_prediction_rows_text(focus_error_rows, max_rows=int(args.tb_tables_k)), global_step)
                if bucket_mode == "support":
                    tb_logger.text("tables/tail_errors", format_prediction_rows_text(focus_error_rows, max_rows=int(args.tb_tables_k)), global_step)
                tb_logger.text(
                    "tables/biggest_late_drops",
                    format_table_text(
                        biggest_late_drop_rows,
                        ("class_id", "label", "bucket", "support_train", "support_val", "f1", "peak_f1", "peak_epoch", "delta_vs_peak"),
                        max_rows=int(args.tb_tables_k),
                    ),
                    global_step,
                )
            if args.tb_log_topology or args.tb_full_logging:
                topology_scalars = _collect_topology_scalars(eval_model)
                if topology_scalars:
                    tb_logger.scalars(topology_scalars, global_step)
            for name, param in hist_params:
                tb_logger.histogram(f"weights/{name}", param, global_step)
                if param.grad is not None:
                    tb_logger.histogram(f"grads/{name}", param.grad, global_step)
            tb_logger.flush()

        should_write_predictions = analysis_artifacts_enabled and (epoch % max(1, int(args.tb_predictions_every)) == 0)
        per_class_csv_path = per_class_dir / f"per_class_ep{epoch:03d}.csv"
        per_class_json_path = per_class_dir / f"per_class_ep{epoch:03d}.json"
        if analysis_artifacts_enabled:
            _write_csv(
                per_class_csv_path,
                [
                    "epoch",
                    "global_step",
                    "class_id",
                    "label",
                    "bucket",
                    "bucket_mode",
                    "support_train",
                    "support_val",
                    "precision",
                    "recall",
                    "f1",
                    "pred_count",
                    "true_count",
                    "conf_mean",
                    "conf_wrong_mean",
                    "margin_mean",
                ],
                per_class_rows,
            )
            _write_json(per_class_json_path, per_class_rows)
        predictions_csv_path = predictions_dir / f"predictions_ep{epoch:03d}.csv"
        errors_csv_path = errors_dir / f"errors_ep{epoch:03d}.csv"
        if should_write_predictions and (args.tb_log_predictions_csv or args.tb_full_logging):
            _write_csv(
                predictions_csv_path,
                [
                    "epoch",
                    "global_step",
                    "video",
                    "t0",
                    "t1",
                    "true_id",
                    "pred_id",
                    "true_label",
                    "pred_label",
                    "correct",
                    "conf",
                    "margin",
                    "entropy",
                    "top5_ids",
                    "top5_labels",
                    "top5_probs",
                    "support_train_true",
                    "bucket_true",
                ],
                prediction_rows,
            )
        if should_write_predictions and (args.tb_log_errors_csv or args.tb_full_logging):
            _write_csv(
                errors_csv_path,
                [
                    "epoch",
                    "global_step",
                    "video",
                    "t0",
                    "t1",
                    "true_id",
                    "pred_id",
                    "true_label",
                    "pred_label",
                    "correct",
                    "conf",
                    "margin",
                    "entropy",
                    "top5_ids",
                    "top5_labels",
                    "top5_probs",
                    "support_train_true",
                    "bucket_true",
                ],
                error_rows,
            )
        confusion_csv_path = confusion_dir / f"confusion_pairs_ep{epoch:03d}.csv"
        confusion_json_path = confusion_dir / f"confusion_pairs_ep{epoch:03d}.json"
        if analysis_artifacts_enabled and (args.tb_log_confusion_pairs or args.tb_full_logging):
            _write_csv(
                confusion_csv_path,
                ["true_id", "pred_id", "true_label", "pred_label", "count", "rate_within_true", "support_true", "bucket_true", "bucket_pred"],
                eval_out.confusion_pairs,
            )
            _write_json(confusion_json_path, eval_out.confusion_pairs)

        _update_peak_state(analysis_state.setdefault("per_class_peak_f1", {}), eval_out.per_class, epoch)

        improved = val_f1 > (best_f1 + es_min_delta)
        if improved:
            best_f1 = val_f1
            epochs_no_improve = 0
        else:
            if es_armed:
                epochs_no_improve += 1
            else:
                epochs_no_improve = 0

        if analysis_artifacts_enabled:
            epoch_index_rows = analysis_state.setdefault("epoch_index", [])
            epoch_index_rows.append(
                {
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "best_f1_so_far": float(best_f1),
                    "event_log_dir": tb_path,
                    "bucket_mode": bucket_mode,
                }
            )
            _write_csv(
                analysis_dir / "epoch_index.csv",
                ["epoch", "global_step", "best_f1_so_far", "event_log_dir", "bucket_mode"],
                epoch_index_rows,
            )

        ckpt_payload = _build_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            sched=sched,
            scaler=scaler,
            ema=ema,
            label2idx=label2idx,
            ds_cfg_dict=ds_cfg_dict,
            best_f1=best_f1,
            epochs_no_improve=epochs_no_improve,
            global_step=global_step,
            history=history,
            analysis_state=analysis_state,
            args=args,
        )
        torch.save(ckpt_payload, out_dir / "last.ckpt")
        if improved:
            torch.save(ckpt_payload, out_dir / "best.ckpt")
            if analysis_artifacts_enabled:
                _copy_if_exists(per_class_dir / f"per_class_ep{epoch:03d}.csv", analysis_dir / "best_per_class.csv")
                _copy_if_exists(per_class_dir / f"per_class_ep{epoch:03d}.json", analysis_dir / "best_per_class.json")
                _copy_if_exists(predictions_dir / f"predictions_ep{epoch:03d}.csv", analysis_dir / "best_predictions.csv")
                _copy_if_exists(errors_dir / f"errors_ep{epoch:03d}.csv", analysis_dir / "best_errors.csv")
                _copy_if_exists(confusion_dir / f"confusion_pairs_ep{epoch:03d}.csv", analysis_dir / "best_confusion_pairs.csv")
                _copy_if_exists(confusion_dir / f"confusion_pairs_ep{epoch:03d}.json", analysis_dir / "best_confusion_pairs.json")
            print(f"  -> new best macro-F1 = {best_f1:.4f} (checkpoint saved)")

        # Per-class F1 (every 5 epochs)
        if epoch % 5 == 0:
            report = classification_report(val_labels, val_preds, output_dict=True, zero_division=0)
            with (out_dir / f"report_ep{epoch:03d}.json").open("w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

        # Save history
        with (out_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if es_armed and epochs_no_improve >= es_patience:
            print(f"Early stopping: no val_f1 improvement for {es_patience} epochs. Best_f1={best_f1:.4f}")
            break

    print(f"Done. Best macro-F1 = {best_f1:.4f}")
    if tb_logger.enabled:
        tb_logger.hparams(
            {
                "epochs": int(args.epochs),
                "batch": int(args.batch),
                "lr": float(args.lr),
                "wd": float(args.wd),
                "accum": int(args.accum),
                "streams": str(args.streams),
                "use_logit_adjustment": bool(args.use_logit_adjustment),
                "use_cosine_head": bool(args.use_cosine_head),
                "use_ctr_hand_refine": bool(args.use_ctr_hand_refine),
                "ctr_in_stream_encoder": bool(args.ctr_in_stream_encoder),
                "use_packed_skeleton_cache": bool(args.use_packed_skeleton_cache),
                "use_decoded_skeleton_cache": bool(args.use_decoded_skeleton_cache),
                "tb_full_logging": bool(args.tb_full_logging),
                "tb_watchlist_k": int(args.tb_watchlist_k),
                "tb_bucket_mode": str(args.tb_bucket_mode),
                "bucket_mode_effective": bucket_mode,
                "early_stop_patience_effective": int(es_patience),
                "early_stop_min_delta_effective": float(es_min_delta),
                "early_stop_start_epoch_effective": int(es_start_epoch),
            },
            {"best_val_f1": float(best_f1)},
        )
        tb_logger.close()
