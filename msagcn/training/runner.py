from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from torch import amp
from torch.utils.data import DataLoader, WeightedRandomSampler

from msagcn.data import DSConfig, MultiStreamGestureDataset, build_sample_weights
from msagcn.models import MultiStreamAGCN
from utils.tensorboard_logger import TensorboardLogger

from .ema import ModelEma
from .engine import evaluate, train_one_epoch
from .losses import LogitAdjustedCrossEntropyLoss
from .metrics import build_confusion_image, format_examples_text, _sanitize_label
from .prefetch import PrefetchLoader
from .utils import build_mirror_idx, select_hist_params, set_seed


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
        "args": vars(args),
    }


def run_training(args) -> None:
    set_seed(args.seed)

    debug_mode = (args.overfit_batches > 0) or (args.limit_train_batches > 0)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name.strip() or out_dir.name
    tb_logger = TensorboardLogger(
        log_dir=args.logdir,
        run_name=run_name,
        enabled=bool(args.tensorboard),
        flush_secs=int(args.flush_secs),
    )
    if tb_logger.enabled:
        tb_path = tb_logger.log_dir.resolve()
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
    train_ds = MultiStreamGestureDataset(args.json, args.csv, split="train", cfg=ds_cfg)
    label2idx = train_ds.label2idx  # single label map from train
    idx2label = {v: k for k, v in label2idx.items()}
    val_ds = MultiStreamGestureDataset(
        args.json,
        args.csv,
        split="val",
        cfg=DSConfig(**{**asdict(ds_cfg), "augment": False}),
        label2idx=label2idx,
    )

    # Detect storage mode (dir vs combined)
    json_path = Path(args.json)
    is_dir_mode = json_path.is_dir()
    if (os.name == "nt") and (not is_dir_mode) and args.workers > 0:
        print("Note: Windows + combined JSON detected -> forcing workers=0 to avoid RAM blowups.")
        args.workers = 0
    print(f"Dataset mode: {'dir' if is_dir_mode else 'combined'} | workers={args.workers}")

    persistent_workers = args.workers > 0
    prefetch_factor = max(2, int(args.prefetch))
    if (os.name == "nt") and is_dir_mode and args.workers > 0:
        # On Windows each spawned worker keeps its own file cache. With high worker
        # counts plus aggressive prefetch this can grow RAM over epochs and kill a worker.
        if persistent_workers:
            print("Note: Windows + per-video JSON -> disabling persistent_workers for loader stability.")
            persistent_workers = False
        if prefetch_factor > 2:
            print(f"Note: Windows + per-video JSON -> capping prefetch_factor {prefetch_factor} -> 2.")
            prefetch_factor = 2
        safe_file_cache = min(int(train_ds.cfg.file_cache_size), 8)
        if safe_file_cache != int(train_ds.cfg.file_cache_size):
            print(
                f"Note: Windows + per-video JSON -> capping per-worker file_cache "
                f"{train_ds.cfg.file_cache_size} -> {safe_file_cache}."
            )
            train_ds.cfg.file_cache_size = safe_file_cache
            val_ds.cfg.file_cache_size = safe_file_cache

    # Save label map & ds cfg
    with (out_dir / "label2idx.json").open("w", encoding="utf-8") as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)
    with (out_dir / "ds_config.json").open("w", encoding="utf-8") as f:
        json.dump(ds_cfg_dict, f, ensure_ascii=False, indent=2)

    # Sampler weights
    weights = build_sample_weights(
        train_ds.samples,
        label2idx,
        train_ds._meta_by_vid,
        quality_floor=0.4,
        quality_power=1.0,
        cover_key="both_coverage",
        cover_floor=0.3,
    )
    sampler = (
        WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
        if args.weighted_sampler
        else None
    )

    # DataLoaders (always build; prefetch optional)
    dl_kwargs = dict(
        batch_size=args.batch,
        num_workers=args.workers,
        persistent_workers=persistent_workers,
        collate_fn=MultiStreamGestureDataset.collate_fn,
    )
    if device.type == "cuda":
        dl_kwargs["pin_memory"] = True
        dl_kwargs["pin_memory_device"] = f"cuda:{torch.cuda.current_device()}"
    else:
        dl_kwargs["pin_memory"] = False

    if args.workers > 0:
        dl_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_ds,
        sampler=sampler if (args.weighted_sampler and sampler is not None) else None,
        shuffle=False if (args.weighted_sampler and sampler is not None) else True,
        drop_last=True,
        **dl_kwargs,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)

    if device.type == "cuda" and not args.no_prefetch:
        train_loader = PrefetchLoader(train_loader, device)
        val_loader = PrefetchLoader(val_loader, device)

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
    es_patience = max(0, int(args.early_stop_patience))
    es_min_delta = max(0.0, float(args.early_stop_min_delta))
    epochs_no_improve = 0
    history = []
    global_step = 0
    start_epoch = 1

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
            print(
                f"Resuming full training state from {resume_path} | "
                f"next_epoch={start_epoch} | best_f1={best_f1:.4f} | global_step={global_step}"
            )

    hist_params = select_hist_params(model, ("stems.", "head."))

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr_loss, global_step = train_one_epoch(
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
        )
        eval_model = ema.module if ema is not None else model
        collect_examples = bool(
            tb_logger.enabled and args.tb_log_examples and (epoch % max(1, int(args.tb_examples_every)) == 0)
        )
        val_loss, val_acc, val_f1, topk_acc, val_preds, val_labels, examples = evaluate(
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
        )
        sched.step()

        dt = time.time() - t0
        print(
            f"[Ep {epoch:03d}] TL={tr_loss:.4f} | VL={val_loss:.4f} | VA={val_acc:.3f} | VF1={val_f1:.3f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | {dt:.1f}s"
        )
        from collections import Counter

        print("pred top5:", Counter(val_preds).most_common(5))

        history.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=val_loss, val_acc=val_acc, val_f1=val_f1))

        if tb_logger.enabled:
            tb_logger.scalar("val/loss", float(val_loss), global_step)
            tb_logger.scalar("val/acc", float(val_acc), global_step)
            tb_logger.scalar("val/f1_macro", float(val_f1), global_step)
            if 3 in topk_acc:
                tb_logger.scalar("val/acc_top3", float(topk_acc[3]), global_step)
            if 5 in topk_acc:
                tb_logger.scalar("val/acc_top5", float(topk_acc[5]), global_step)
            if val_labels:
                f1_micro = f1_score(val_labels, val_preds, average="micro")
                f1_weighted = f1_score(val_labels, val_preds, average="weighted")
                tb_logger.scalar("val/f1_micro", float(f1_micro), global_step)
                tb_logger.scalar("val/f1_weighted", float(f1_weighted), global_step)
            if val_labels:
                p_macro, r_macro, _, _ = precision_recall_fscore_support(
                    val_labels, val_preds, average="macro", zero_division=0
                )
                tb_logger.scalar("val/precision_macro", float(p_macro), global_step)
                tb_logger.scalar("val/recall_macro", float(r_macro), global_step)
                p_all, r_all, f1s, _ = precision_recall_fscore_support(
                    val_labels, val_preds, labels=list(range(len(label2idx))), zero_division=0
                )
                support = np.bincount(np.array(val_labels), minlength=len(label2idx))
                valid = np.where(support > 0)[0]
                if valid.size > 0:
                    topk_support = int(max(0, args.tb_support_topk))
                    if topk_support > 0:
                        order = valid[np.argsort(-support[valid])]
                        for idx in order[: min(topk_support, len(order))]:
                            label_name = _sanitize_label(idx2label.get(idx, str(idx)))
                            tb_logger.scalar(f"val/support/{label_name}", float(support[idx]), global_step)
                            tb_logger.scalar(f"val/p_class/{label_name}", float(p_all[idx]), global_step)
                            tb_logger.scalar(f"val/r_class/{label_name}", float(r_all[idx]), global_step)
                            tb_logger.scalar(f"val/f1_class/{label_name}", float(f1s[idx]), global_step)
                    worstk = int(max(0, args.tb_worstk_f1))
                    if worstk > 0:
                        order = valid[np.argsort(f1s[valid])]
                        for idx in order[: min(worstk, len(order))]:
                            label_name = _sanitize_label(idx2label.get(idx, str(idx)))
                            tb_logger.scalar(f"val/f1_worst/{label_name}", float(f1s[idx]), global_step)
                            tb_logger.scalar(f"val/p_worst/{label_name}", float(p_all[idx]), global_step)
                            tb_logger.scalar(f"val/r_worst/{label_name}", float(r_all[idx]), global_step)
            if args.tb_log_confusion and val_labels:
                try:
                    img, _ = build_confusion_image(
                        val_labels, val_preds, idx2label, topk=int(args.tb_confusion_topk), normalize=True
                    )
                    tb_logger.image("val/confusion_topk", img, global_step, dataformats="HWC")
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
            for name, param in hist_params:
                tb_logger.histogram(f"weights/{name}", param, global_step)
                if param.grad is not None:
                    tb_logger.histogram(f"grads/{name}", param.grad, global_step)

        improved = val_f1 > (best_f1 + es_min_delta)
        if improved:
            best_f1 = val_f1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

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
            args=args,
        )
        torch.save(ckpt_payload, out_dir / "last.ckpt")
        if improved:
            torch.save(ckpt_payload, out_dir / "best.ckpt")
            print(f"  -> new best macro-F1 = {best_f1:.4f} (checkpoint saved)")

        # Per-class F1 (every 5 epochs)
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
            },
            {"best_val_f1": float(best_f1)},
        )
        tb_logger.close()
