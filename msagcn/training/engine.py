from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch import amp

from .metrics import build_confusion_pairs, compute_calibration_metrics, push_lowest_conf
from .utils import move_to_device
from utils.tensorboard_logger import TensorboardLogger


@dataclass
class TrainOutputs:
    loss: float
    total_loss: float
    supcon_loss: float
    supcon_valid_anchors: int
    supcon_positive_pairs: int
    global_step: int
    nonfinite_batches: int
    skipped_updates: int


@dataclass
class EvalOutputs:
    loss: float
    acc: float
    macro_f1: float
    topk_acc: Dict[int, float]
    labels: List[int]
    preds: List[int]
    probs_top1: List[float]
    probs_top2: List[float]
    entropy: List[float]
    margin: List[float]
    records: List[Dict[str, object]]
    per_class: Dict[int, Dict[str, object]]
    confusion_pairs: List[Dict[str, object]]
    examples: List[Dict[str, object]]
    calibration: Dict[str, float]


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
    use_amp: bool,
    amp_dtype: torch.dtype,
    *,
    max_batches: int | None = None,
    log_interval: int = 50,
    ema=None,
    tb_logger: TensorboardLogger | None = None,
    tb_log_every: int = 1,
    global_step: int = 0,
    supcon_criterion=None,
    supcon_weight: float = 0.0,
    supcon_start_epoch: int = 0,
    epoch: int = 0,
    log_batch_label_stats: bool = False,
    expected_batch_unique_classes: int | None = None,
    expected_batch_samples_per_class: int | None = None,
    expected_mixed_repeated_classes: int | None = None,
    expected_mixed_repeated_samples: int | None = None,
    expected_mixed_singleton_classes: int | None = None,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    epoch_start = time.time()
    total_loss = 0.0
    total_objective = 0.0
    total_n = 0
    t_load_sum = 0.0
    t_fwd_sum = 0.0
    steps = 0
    nonfinite_batches = 0
    skipped_updates = 0
    supcon_loss_sum = 0.0
    supcon_valid_anchor_sum = 0
    supcon_positive_pair_sum = 0
    max_batches = max_batches if max_batches and max_batches > 0 else None
    tb_enabled = tb_logger is not None and tb_logger.enabled and tb_log_every > 0
    supcon_active = bool(supcon_criterion is not None and float(supcon_weight) > 0.0 and int(epoch) >= int(supcon_start_epoch))

    for step, (X, y, metas) in enumerate(loader, start=1):
        if max_batches is not None and step > max_batches:
            break
        global_step += 1
        t0 = time.time()
        # Ensure batch is on the right device (when PrefetchLoader is off)
        if any(hasattr(v, "device") and v.device.type != device.type for v in X.values()):
            X = move_to_device(X, device)
            y = y.to(device, non_blocking=True)
        t_load_sum += time.time() - t0

        mask = X.get("mask", None)
        x_streams = {k: v for k, v in X.items() if k != "mask"}
        if device.type == "cuda" and use_channels_last:
            for k in x_streams:
                x_streams[k] = x_streams[k].to(memory_format=torch.channels_last)

        if step == 1 and "joints" in x_streams:
            print("debug | |joints|_mean =", float(x_streams["joints"].abs().mean()))
        if step == 1 and expected_batch_unique_classes is not None and expected_batch_samples_per_class is not None:
            _, label_counts = torch.unique(y.detach(), return_counts=True)
            unique_classes = int(label_counts.numel())
            min_label_count = int(label_counts.min().item()) if label_counts.numel() > 0 else 0
            max_label_count = int(label_counts.max().item()) if label_counts.numel() > 0 else 0
            print(
                "class-balanced batch check | "
                f"unique_classes={unique_classes} | min_count={min_label_count} | max_count={max_label_count}"
            )
            expected_unique = int(expected_batch_unique_classes)
            expected_count = int(expected_batch_samples_per_class)
            if unique_classes != expected_unique or min_label_count != expected_count or max_label_count != expected_count:
                raise RuntimeError(
                    "Class-balanced batch sampler verification failed on the first train batch: "
                    f"expected {expected_unique} classes x {expected_count} samples, got "
                    f"{unique_classes} unique classes with count range [{min_label_count}, {max_label_count}]."
                )
        if (
            step == 1
            and expected_mixed_repeated_classes is not None
            and expected_mixed_repeated_samples is not None
            and expected_mixed_singleton_classes is not None
        ):
            _, label_counts = torch.unique(y.detach(), return_counts=True)
            unique_classes = int(label_counts.numel())
            repeated_counts = label_counts[label_counts > 1]
            repeated_classes = int(repeated_counts.numel())
            repeated_min = int(repeated_counts.min().item()) if repeated_counts.numel() > 0 else 0
            repeated_max = int(repeated_counts.max().item()) if repeated_counts.numel() > 0 else 0
            singleton_classes = int((label_counts == 1).sum().item())
            print(
                "mixed supcon batch check | "
                f"unique_classes={unique_classes} | "
                f"repeated_classes={repeated_classes} | "
                f"repeated_count_min={repeated_min} | repeated_count_max={repeated_max} | "
                f"singleton_classes={singleton_classes}"
            )
            expected_repeated_classes = int(expected_mixed_repeated_classes)
            expected_repeated_samples = int(expected_mixed_repeated_samples)
            expected_singletons = int(expected_mixed_singleton_classes)
            if (
                repeated_classes != expected_repeated_classes
                or repeated_min != expected_repeated_samples
                or repeated_max != expected_repeated_samples
                or singleton_classes != expected_singletons
            ):
                raise RuntimeError(
                    "Hybrid SupCon batch sampler verification failed on the first train batch: "
                    f"expected {expected_repeated_classes} repeated classes x {expected_repeated_samples} samples "
                    f"+ {expected_singletons} singleton classes, got "
                    f"{repeated_classes} repeated classes with count range [{repeated_min}, {repeated_max}] "
                    f"and {singleton_classes} singleton classes."
                )

        t1 = time.time()
        # --- autocast (AMP) ---
        with amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            if supcon_active:
                logits, features = model(x_streams, mask=mask, A=A, y=y, return_features=True)
                features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                logits = model(x_streams, mask=mask, A=A, y=y)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            cls_loss = criterion(logits, y)
            if supcon_active:
                supcon_loss = supcon_criterion(features, y).to(dtype=cls_loss.dtype)
                total_batch_loss = cls_loss + float(supcon_weight) * supcon_loss
            else:
                supcon_loss = cls_loss.new_zeros(())
                total_batch_loss = cls_loss
            loss = total_batch_loss / accum_steps
        t_fwd_sum += time.time() - t1

        # NaN guard
        if not torch.isfinite(loss):
            print("WARN: non-finite loss encountered -> batch skipped")
            optimizer.zero_grad(set_to_none=True)
            nonfinite_batches += 1
            skipped_updates += 1
            continue

        if tb_enabled and (global_step % tb_log_every == 0):
            lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0
            supcon_stats = getattr(supcon_criterion, "last_stats", {}) if supcon_active else {}
            tb_logger.scalar("train/loss", float(cls_loss.item()), global_step)
            tb_logger.scalar("train/total_loss", float(total_batch_loss.item()), global_step)
            tb_logger.scalar("train/supcon_loss", float(supcon_loss.item()), global_step)
            tb_logger.scalar(
                "train/supcon_valid_anchors",
                float(supcon_stats.get("valid_anchor_count", 0)),
                global_step,
            )
            tb_logger.scalar(
                "train/supcon_positive_pairs",
                float(supcon_stats.get("positive_pair_count", 0)),
                global_step,
            )
            tb_logger.scalar("train/lr", lr, global_step)
            tb_logger.scalar("train/amp_scale", float(scaler.get_scale()) if hasattr(scaler, "get_scale") else 1.0, global_step)
            tb_logger.scalar("train/nonfinite_batches", float(nonfinite_batches), global_step)
            tb_logger.scalar("train/skipped_updates", float(skipped_updates), global_step)
            if log_batch_label_stats:
                _, label_counts = torch.unique(y.detach(), return_counts=True)
                repeated_counts = label_counts[label_counts > 1]
                tb_logger.scalar("train/batch_unique_classes", float(label_counts.numel()), global_step)
                tb_logger.scalar("train/batch_max_label_count", float(label_counts.max().item()), global_step)
                tb_logger.scalar("train/batch_min_label_count", float(label_counts.min().item()), global_step)
                tb_logger.scalar("train/batch_repeated_classes", float(repeated_counts.numel()), global_step)
                tb_logger.scalar("train/batch_singleton_classes", float((label_counts == 1).sum().item()), global_step)
                tb_logger.scalar(
                    "train/batch_repeat_count_min",
                    float(repeated_counts.min().item()) if repeated_counts.numel() > 0 else 0.0,
                    global_step,
                )
                tb_logger.scalar(
                    "train/batch_repeat_count_max",
                    float(repeated_counts.max().item()) if repeated_counts.numel() > 0 else 0.0,
                    global_step,
                )

        scaler.scale(loss).backward()

        last_step = (step == len(loader)) if max_batches is None else (step == max_batches)
        if (step % accum_steps == 0) or last_step:
            grad_norm = None
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            if tb_enabled and grad_norm is not None:
                tb_logger.scalar("train/grad_norm", float(grad_norm), global_step)

        bs = y.size(0)
        total_loss += float(cls_loss.item()) * bs
        total_objective += float(total_batch_loss.item()) * bs
        total_n += bs
        steps += 1
        supcon_loss_sum += float(supcon_loss.item()) * bs
        if supcon_active:
            supcon_stats = getattr(supcon_criterion, "last_stats", {})
            supcon_valid_anchor_sum += int(supcon_stats.get("valid_anchor_count", 0))
            supcon_positive_pair_sum += int(supcon_stats.get("positive_pair_count", 0))

        if log_interval > 0 and (step % log_interval == 0):
            with torch.no_grad():
                probs = logits.detach().softmax(dim=1)
                entropy = float((-probs * torch.log(probs.clamp_min(1e-9))).sum(dim=1).mean())
                top2_vals = probs.topk(k=min(2, probs.size(1)), dim=1).values
                conf = float(top2_vals[:, 0].mean())
                if top2_vals.size(1) > 1:
                    margin = top2_vals[:, 0] - top2_vals[:, 1]
                else:
                    margin = top2_vals[:, 0]
                margin_f = margin.float()
                margin_mean = float(margin_f.mean())
                margin_p10 = float(torch.quantile(margin_f, q=0.10)) if margin_f.numel() > 1 else margin_mean
            print(
                f"   step {step:04d} | loss={float(total_batch_loss.item()):.4f} | "
                f"entropy={entropy:.3f} | conf={conf:.3f}"
            )
            if tb_enabled:
                tb_logger.scalar("train/entropy", entropy, global_step)
                tb_logger.scalar("train/confidence", conf, global_step)
                tb_logger.scalar("train/confidence_mean", conf, global_step)
                tb_logger.scalar("train/prob_margin_mean", margin_mean, global_step)
                tb_logger.scalar("train/prob_margin_p10", margin_p10, global_step)

    if total_n > 0:
        print(f"   dataloader_time~{t_load_sum:.1f}s | fwd+bwd_time~{t_fwd_sum:.1f}s | steps={steps}")
        if tb_enabled:
            epoch_time = max(1e-6, time.time() - epoch_start)
            tb_logger.scalar("train/samples_per_sec", float(total_n / epoch_time), global_step)
            tb_logger.scalar("train/dataloader_time_sec", float(t_load_sum), global_step)
            tb_logger.scalar("train/fwd_bwd_time_sec", float(t_fwd_sum), global_step)
            tb_logger.scalar("train/epoch_time_sec", float(epoch_time), global_step)
    return TrainOutputs(
        loss=(total_loss / max(1, total_n)),
        total_loss=(total_objective / max(1, total_n)),
        supcon_loss=(supcon_loss_sum / max(1, total_n)),
        supcon_valid_anchors=int(supcon_valid_anchor_sum),
        supcon_positive_pairs=int(supcon_positive_pair_sum),
        global_step=global_step,
        nonfinite_batches=int(nonfinite_batches),
        skipped_updates=int(skipped_updates),
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    A: torch.Tensor,
    mirror_idx: torch.LongTensor | None,
    use_channels_last: bool,
    use_amp: bool,
    amp_dtype: torch.dtype,
    *,
    max_batches: int | None = None,
    topk: Tuple[int, ...] = (3, 5),
    collect_examples: bool = False,
    examples_k: int = 5,
    epoch: int = 0,
    global_step: int = 0,
    idx2label: Mapping[int, str] | None = None,
    train_support: np.ndarray | None = None,
    bucket_by_class: Mapping[int, str] | None = None,
) -> EvalOutputs:
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_conf: List[float] = []
    all_top2: List[float] = []
    all_margin: List[float] = []
    all_entropy: List[float] = []
    all_correct: List[float] = []
    all_true_prob: List[float] = []
    all_prob_sq_sum: List[float] = []
    wrong_examples: List[Dict[str, object]] = []
    correct_examples: List[Dict[str, object]] = []
    records: List[Dict[str, object]] = []
    topk = tuple(sorted({int(k) for k in topk if int(k) > 0}))
    topk_correct = {k: 0 for k in topk}
    max_batches = max_batches if max_batches and max_batches > 0 else None
    idx2label = dict(idx2label or {})
    num_classes = int(len(train_support)) if train_support is not None else 0

    for step, (X, y, metas) in enumerate(loader, start=1):
        if max_batches is not None and step > max_batches:
            break
        # If loader has no prefetcher
        if any(hasattr(v, "device") and v.device.type != device.type for v in X.values()):
            X = move_to_device(X, device)
            y = y.to(device, non_blocking=True)

        mask = X.get("mask", None)
        x_streams = {k: v for k, v in X.items() if k != "mask"}
        if device.type == "cuda" and use_channels_last:
            for k in x_streams:
                x_streams[k] = x_streams[k].to(memory_format=torch.channels_last)

        # --- autocast (AMP) ---
        with amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            if mirror_idx is not None:
                # compile/cudagraph-safe TTA: one forward on concatenated batch
                B0 = y.size(0)
                x_cat = {k: torch.cat([v, v.index_select(dim=2, index=mirror_idx)], dim=0) for k, v in x_streams.items()}
                m_cat = (
                    torch.cat([mask, mask.index_select(dim=2, index=mirror_idx)], dim=0)
                    if mask is not None
                    else None
                )
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
        probs = torch.softmax(logits, dim=1)
        top2_vals, top2_idx = probs.topk(k=min(2, probs.size(1)), dim=1)
        conf = top2_vals[:, 0]
        second = top2_vals[:, 1] if top2_vals.size(1) > 1 else torch.zeros_like(conf)
        margin = conf - second
        entropy = (-probs * torch.log(probs.clamp_min(1e-9))).sum(dim=1)
        true_prob = probs.gather(1, y.view(-1, 1)).squeeze(1).clamp_min(1e-9)
        prob_sq_sum = probs.square().sum(dim=1)
        correct = preds.eq(y).float()
        all_conf.extend(conf.detach().cpu().tolist())
        all_top2.extend(second.detach().cpu().tolist())
        all_margin.extend(margin.detach().cpu().tolist())
        all_entropy.extend(entropy.detach().cpu().tolist())
        all_true_prob.extend(true_prob.detach().cpu().tolist())
        all_prob_sq_sum.extend(prob_sq_sum.detach().cpu().tolist())
        all_correct.extend(correct.detach().cpu().tolist())
        if topk:
            maxk = min(max(topk), logits.size(1))
            _, pred_topk = logits.topk(maxk, dim=1)
            correct = pred_topk.eq(y.view(-1, 1))
            for k in topk:
                kk = min(k, pred_topk.size(1))
                topk_correct[k] += int(correct[:, :kk].any(dim=1).sum().item())
        if metas is not None:
            top5_k = min(5, probs.size(1))
            top5_vals, top5_idx = probs.topk(top5_k, dim=1)
            for i, meta in enumerate(metas):
                item = {
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "video": meta.get("video", ""),
                    "t0": meta.get("t0", ""),
                    "t1": meta.get("t1", ""),
                    "true": int(y[i].item()),
                    "pred": int(preds[i].item()),
                    "true_id": int(y[i].item()),
                    "pred_id": int(preds[i].item()),
                    "true_label": str(idx2label.get(int(y[i].item()), str(int(y[i].item())))),
                    "pred_label": str(idx2label.get(int(preds[i].item()), str(int(preds[i].item())))),
                    "conf": float(conf[i].item()),
                    "margin": float(margin[i].item()),
                    "entropy": float(entropy[i].item()),
                    "top5_ids": [int(x) for x in top5_idx[i].detach().cpu().tolist()],
                    "top5_labels": [str(idx2label.get(int(x), str(int(x)))) for x in top5_idx[i].detach().cpu().tolist()],
                    "top5_probs": [float(x) for x in top5_vals[i].detach().cpu().tolist()],
                    "correct": int(preds[i].item() == y[i].item()),
                }
                records.append(item)
                if collect_examples:
                    if item["true"] != item["pred"]:
                        push_lowest_conf(wrong_examples, item, int(max(1, examples_k)))
                    else:
                        push_lowest_conf(correct_examples, item, int(max(1, examples_k)))

    macro_f1 = f1_score(all_labels, all_preds, average="macro") if total_n else 0.0
    acc = (np.array(all_preds) == np.array(all_labels)).mean() if total_n else 0.0
    topk_acc = {k: (topk_correct[k] / max(1, total_n)) for k in topk}
    examples = wrong_examples
    if len(examples) < int(max(1, examples_k)):
        examples = examples + correct_examples
    if num_classes <= 0:
        num_classes = (max(max(all_labels, default=-1), max(all_preds, default=-1)) + 1) if total_n else 0

    if num_classes > 0:
        labels_range = list(range(num_classes))
        p_all, r_all, f1_all, _ = precision_recall_fscore_support(
            all_labels,
            all_preds,
            labels=labels_range,
            zero_division=0,
        )
        support_val = np.bincount(np.asarray(all_labels, dtype=np.int64), minlength=num_classes)
        pred_count = np.bincount(np.asarray(all_preds, dtype=np.int64), minlength=num_classes)
        conf_sum = np.zeros(num_classes, dtype=np.float64)
        conf_wrong_sum = np.zeros(num_classes, dtype=np.float64)
        conf_wrong_cnt = np.zeros(num_classes, dtype=np.float64)
        margin_sum = np.zeros(num_classes, dtype=np.float64)
        for cls, pred, conf_v, margin_v in zip(all_labels, all_preds, all_conf, all_margin):
            conf_sum[int(cls)] += float(conf_v)
            margin_sum[int(cls)] += float(margin_v)
            if int(cls) != int(pred):
                conf_wrong_sum[int(cls)] += float(conf_v)
                conf_wrong_cnt[int(cls)] += 1.0
        per_class: Dict[int, Dict[str, object]] = {}
        for class_id in range(num_classes):
            val_support = int(support_val[class_id])
            per_class[class_id] = {
                "class_id": int(class_id),
                "label": str(idx2label.get(class_id, str(class_id))),
                "bucket": str(bucket_by_class.get(class_id, "none")) if bucket_by_class else "none",
                "support_val": val_support,
                "support_train": int(train_support[class_id]) if train_support is not None and class_id < len(train_support) else 0,
                "precision": float(p_all[class_id]),
                "recall": float(r_all[class_id]),
                "f1": float(f1_all[class_id]),
                "pred_count": int(pred_count[class_id]),
                "true_count": val_support,
                "conf_mean": float(conf_sum[class_id] / max(1, val_support)),
                "conf_wrong_mean": float(conf_wrong_sum[class_id] / max(1.0, conf_wrong_cnt[class_id])),
                "margin_mean": float(margin_sum[class_id] / max(1, val_support)),
            }
    else:
        per_class = {}

    confusion_pairs = build_confusion_pairs(
        all_labels,
        all_preds,
        idx2label,
        support_true=[int(v.get("support_val", 0)) for _, v in sorted(per_class.items())] if per_class else None,
        bucket_by_class=bucket_by_class,
    )
    calibration = compute_calibration_metrics(all_conf, all_correct, all_true_prob, all_prob_sq_sum, num_bins=15)
    return EvalOutputs(
        loss=(total_loss / max(1, total_n)),
        acc=float(acc),
        macro_f1=float(macro_f1),
        topk_acc=topk_acc,
        labels=all_labels,
        preds=all_preds,
        probs_top1=all_conf,
        probs_top2=all_top2,
        entropy=all_entropy,
        margin=all_margin,
        records=records,
        per_class=per_class,
        confusion_pairs=confusion_pairs,
        examples=examples,
        calibration=calibration,
    )
