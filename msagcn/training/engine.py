from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch import amp

from .metrics import push_lowest_conf
from .utils import move_to_device
from utils.tensorboard_logger import TensorboardLogger


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
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    epoch_start = time.time()
    total_loss = 0.0
    total_n = 0
    t_load_sum = 0.0
    t_fwd_sum = 0.0
    steps = 0
    max_batches = max_batches if max_batches and max_batches > 0 else None
    tb_enabled = tb_logger is not None and tb_logger.enabled and tb_log_every > 0

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

        t1 = time.time()
        # --- autocast (AMP) ---
        with amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            logits = model(x_streams, mask=mask, A=A, y=y)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            loss = criterion(logits, y) / accum_steps
        t_fwd_sum += time.time() - t1

        # NaN guard
        if not torch.isfinite(loss):
            print("WARN: non-finite loss encountered -> batch skipped")
            optimizer.zero_grad(set_to_none=True)
            continue

        if tb_enabled and (global_step % tb_log_every == 0):
            lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0
            tb_logger.scalar("train/loss", float(loss.item()) * accum_steps, global_step)
            tb_logger.scalar("train/lr", lr, global_step)

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
        total_loss += float(loss.item()) * accum_steps * bs
        total_n += bs
        steps += 1

        if log_interval > 0 and (step % log_interval == 0):
            with torch.no_grad():
                probs = logits.detach().softmax(dim=1)
                entropy = float((-probs * torch.log(probs.clamp_min(1e-9))).sum(dim=1).mean())
                conf = float(probs.max(dim=1).values.mean())
            print(
                f"   step {step:04d} | loss={float(loss.item() * accum_steps):.4f} | "
                f"entropy={entropy:.3f} | conf={conf:.3f}"
            )
            if tb_enabled:
                tb_logger.scalar("train/entropy", entropy, global_step)
                tb_logger.scalar("train/confidence", conf, global_step)

    if total_n > 0:
        print(f"   dataloader_time~{t_load_sum:.1f}s | fwd+bwd_time~{t_fwd_sum:.1f}s | steps={steps}")
        if tb_enabled:
            epoch_time = max(1e-6, time.time() - epoch_start)
            tb_logger.scalar("train/samples_per_sec", float(total_n / epoch_time), global_step)
            tb_logger.scalar("train/dataloader_time_sec", float(t_load_sum), global_step)
            tb_logger.scalar("train/fwd_bwd_time_sec", float(t_fwd_sum), global_step)
            tb_logger.scalar("train/epoch_time_sec", float(epoch_time), global_step)
    return total_loss / max(1, total_n), global_step


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
):
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    wrong_examples: List[Dict[str, object]] = []
    correct_examples: List[Dict[str, object]] = []
    topk = tuple(sorted({int(k) for k in topk if int(k) > 0}))
    topk_correct = {k: 0 for k in topk}
    max_batches = max_batches if max_batches and max_batches > 0 else None

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
        if topk:
            maxk = min(max(topk), logits.size(1))
            _, pred_topk = logits.topk(maxk, dim=1)
            correct = pred_topk.eq(y.view(-1, 1))
            for k in topk:
                kk = min(k, pred_topk.size(1))
                topk_correct[k] += int(correct[:, :kk].any(dim=1).sum().item())
        if collect_examples and metas is not None:
            probs = torch.softmax(logits, dim=1)
            conf = probs.max(dim=1).values.detach().cpu().tolist()
            for i, meta in enumerate(metas):
                item = {
                    "video": meta.get("video", ""),
                    "t0": meta.get("t0", ""),
                    "t1": meta.get("t1", ""),
                    "true": int(y[i].item()),
                    "pred": int(preds[i].item()),
                    "conf": float(conf[i]),
                }
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
    return total_loss / max(1, total_n), acc, macro_f1, topk_acc, all_preds, all_labels, examples
