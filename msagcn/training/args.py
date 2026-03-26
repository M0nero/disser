from __future__ import annotations

import argparse
import os


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser("Train Multi-Stream AGCN (max F1)")

    # Data
    p.add_argument("--json", required=True, help="skeletons source: combined .json OR a directory with per-video *.json")
    p.add_argument("--csv", required=True, help="annotations CSV with splits")
    p.add_argument("--out", default="outputs/runs/agcn", help="output dir (checkpoints, logs)")

    # Dataset config
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument(
        "--end_is_exclusive",
        action="store_true",
        help="Treat CSV 'end' as python-slice exclusive (default: end is inclusive -> +1).",
    )
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
    p.add_argument(
        "--connect_cross_edges",
        action="store_true",
        help="Enable pose<->hand cross edges (overrides default when --include_pose).",
    )
    p.add_argument(
        "--no_cross_edges",
        action="store_true",
        help="Disable pose<->hand cross edges even when --include_pose.",
    )

    p.add_argument("--hand_score_thr", type=float, default=0.45)
    p.add_argument("--hand_score_thr_fallback", type=float, default=0.35)
    p.add_argument("--window_valid_ratio", type=float, default=0.60)
    p.add_argument("--window_valid_ratio_fallback", type=float, default=0.50)

    p.add_argument("--center", action="store_true")
    p.add_argument("--center_mode", type=str, default="masked_mean", choices=["masked_mean", "wrists"])
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--norm_method", type=str, default="p95", choices=["p95", "max", "mad"])
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
    p.add_argument("--epochs", type=int, default=80, help="Total target epochs. When resuming, training continues until this epoch number.")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--ema_decay", type=float, default=0.0, help="EMA decay for model weights (0=off)")
    p.add_argument("--warmup_frac", type=float, default=0.10, help="Fraction of total epochs to use for LR warmup (0 disables)")
    p.add_argument(
        "--early_stop_patience",
        type=int,
        default=-1,
        help="Stop if val_f1 does not improve for N armed epochs. -1 = auto, 0 disables.",
    )
    p.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=-1.0,
        help="Minimum val_f1 increase to reset early stopping patience. Negative = auto.",
    )
    p.add_argument(
        "--early_stop_start_epoch",
        type=int,
        default=0,
        help="Epoch at which early stopping becomes active. 0 = auto burn-in based on warmup and patience.",
    )
    p.add_argument("--limit_train_batches", type=int, default=0, help="Debug: cap train batches per epoch")
    p.add_argument("--limit_val_batches", type=int, default=0, help="Debug: cap val batches")
    p.add_argument("--overfit_batches", type=int, default=0, help="Debug: use same limited #batches for train+val")
    p.add_argument("--log_interval", type=int, default=50, help="Print train stats every N steps")
    p.add_argument("--keep_aug_in_debug", action="store_true", help="Do not auto-disable augmentations when limiting batches/overfitting")
    p.add_argument("--disable_norm_in_debug", action="store_true", help="When in debug/overfit mode, turn off normalization")
    p.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    p.add_argument("--logdir", type=str, default="runs", help="TensorBoard base log dir")
    p.add_argument("--run_name", type=str, default="", help="TensorBoard run name (default: out dir name)")
    p.add_argument("--flush_secs", type=int, default=30, help="TensorBoard flush_secs")
    p.add_argument("--log_every_steps", type=int, default=1, help="TensorBoard step logging frequency")
    p.add_argument("--tb_support_topk", type=int, default=50, help="Top-K classes by support to log in TensorBoard")
    p.add_argument("--tb_worstk_f1", type=int, default=50, help="Worst-K classes by F1 to log in TensorBoard")
    p.add_argument("--tb_confusion_topk", type=int, default=50, help="Top-K classes for confusion matrix")
    p.add_argument("--tb_log_confusion", action="store_true", help="Log confusion matrix image to TensorBoard")
    p.add_argument("--tb_log_examples", action="store_true", help="Log sample predictions to TensorBoard")
    p.add_argument("--tb_examples_k", type=int, default=5, help="Number of examples to log")
    p.add_argument("--tb_examples_every", type=int, default=5, help="Log examples every N epochs")
    p.add_argument("--tb_full_logging", action="store_true", help="Enable full TensorBoard + analysis artifact logging")
    p.add_argument("--tb_log_all_classes", action="store_true", help="Log per-class val metrics for every class")
    p.add_argument("--tb_log_tail_buckets", action="store_true", help="Log aggregate bucket metrics (support buckets or difficulty buckets)")
    p.add_argument("--tb_log_confusion_pairs", action="store_true", help="Write machine-readable confusion pairs and text tables")
    p.add_argument("--tb_log_predictions_csv", action="store_true", help="Write per-sample validation predictions CSV")
    p.add_argument("--tb_log_errors_csv", action="store_true", help="Write per-sample validation errors CSV")
    p.add_argument("--tb_log_topology", action="store_true", help="Log CTR/adaptive-topology diagnostics")
    p.add_argument(
        "--tb_watchlist_k",
        type=int,
        default=64,
        help="Size of the fixed watchlist (tail classes in support mode, hardest classes in difficulty mode)",
    )
    p.add_argument(
        "--tb_bucket_mode",
        type=str,
        default="auto",
        choices=("auto", "support", "difficulty"),
        help="Bucket mode for aggregate class analytics (auto switches to difficulty mode when train support is uniform).",
    )
    p.add_argument(
        "--tb_difficulty_freeze_epoch",
        type=int,
        default=10,
        help="Freeze difficulty buckets/watchlist after this epoch when using difficulty mode.",
    )
    p.add_argument(
        "--tb_difficulty_ema",
        type=float,
        default=0.8,
        help="EMA decay for per-class F1 difficulty scores before freeze.",
    )
    p.add_argument("--tb_confusion_every", type=int, default=5, help="Log confusion images/tables every N epochs")
    p.add_argument("--tb_predictions_every", type=int, default=1, help="Write predictions/errors artifacts every N epochs")
    p.add_argument("--tb_tables_k", type=int, default=50, help="Max rows for TensorBoard text tables")

    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 8))
    p.add_argument("--auto_workers", action="store_true", help="Auto-benchmark and cache a safe/effective train loader worker profile")
    p.add_argument(
        "--auto_workers_max",
        type=int,
        default=0,
        help="Upper bound for auto worker search (0 = pick a sensible machine/data-mode bound automatically).",
    )
    p.add_argument("--auto_workers_rebench", action="store_true", help="Ignore cached auto-workers decisions and benchmark again")
    p.add_argument("--auto_workers_warmup_batches", type=int, default=4, help="Warmup batches for auto worker benchmark")
    p.add_argument("--auto_workers_measure_batches", type=int, default=16, help="Measured batches for auto worker benchmark")
    p.add_argument("--prefetch", type=int, default=6)
    p.add_argument("--no_prefetch", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default="", help="Resume training from a checkpoint (.ckpt)")
    p.add_argument(
        "--resume_model_only",
        action="store_true",
        help="Load model/EMA weights from --resume, but reset optimizer, scheduler, scaler, best score, and history.",
    )

    # Sampling & Loss options
    p.add_argument("--weighted_sampler", action="store_true", help="Use WeightedRandomSampler for train")
    p.add_argument(
        "--supcon_class_balanced_batch",
        action="store_true",
        help="Use a class-balanced train batch sampler (classes_per_batch x samples_per_class), intended for SupCon experiments.",
    )
    p.add_argument(
        "--supcon_mixed_batch",
        action="store_true",
        help="Use a hybrid SupCon batch sampler: repeated-label groups plus singleton labels to preserve class diversity.",
    )
    p.add_argument(
        "--supcon_classes_per_batch",
        type=int,
        default=16,
        help="Number of unique labels per train batch when --supcon_class_balanced_batch is enabled.",
    )
    p.add_argument(
        "--supcon_samples_per_class",
        type=int,
        default=4,
        help="Number of samples per label in a class-balanced train batch.",
    )
    p.add_argument(
        "--supcon_mixed_repeated_classes",
        type=int,
        default=16,
        help="Number of repeated labels per train batch when --supcon_mixed_batch is enabled.",
    )
    p.add_argument(
        "--supcon_mixed_repeated_samples",
        type=int,
        default=2,
        help="Samples per repeated label when --supcon_mixed_batch is enabled.",
    )
    p.add_argument("--use_logit_adjustment", action="store_true", help="Use Logit-Adjusted CrossEntropy")
    p.add_argument("--use_supcon", action="store_true", help="Enable supervised contrastive auxiliary loss on pooled embeddings")
    p.add_argument("--supcon_weight", type=float, default=0.05, help="Weight for the supervised contrastive auxiliary loss")
    p.add_argument("--supcon_temp", type=float, default=0.07, help="Temperature for supervised contrastive loss")
    p.add_argument(
        "--supcon_start_epoch",
        type=int,
        default=-1,
        help="Start applying the supervised contrastive auxiliary from this 1-based epoch (-1 = auto after warmup, 0 = from epoch 1).",
    )
    p.add_argument("--use_family_head", action="store_true", help="Enable an auxiliary family-classification head on top of pooled features")
    p.add_argument("--family_map", type=str, default="", help="Path to a train-only family_map.json")
    p.add_argument("--family_loss_weight", type=float, default=0.25, help="Weight for the auxiliary family classification loss")
    p.add_argument("--family_warmup_epochs", type=int, default=0, help="Delay family supervision for the first N epochs")
    p.add_argument("--family_label_smoothing", type=float, default=0.0, help="Label smoothing for the family classification loss")
    p.add_argument("--num_families", type=int, default=0, help="Expected number of families (0 = infer from family_map)")
    p.add_argument("--family_eval", action="store_true", help="Log validation family metrics when the auxiliary family head is enabled")
    p.add_argument("--head_only_rebalance_epochs", type=int, default=0, help="Optional second-stage epochs that freeze the backbone and train only the class head")
    p.add_argument("--head_only_rebalance_lr", type=float, default=1e-4, help="Learning rate for the optional classifier-only rebalance stage")
    p.add_argument(
        "--head_only_rebalance_only",
        action="store_true",
        help="Skip the main training loop and run only the classifier-only rebalance stage from the resumed checkpoint.",
    )
    p.add_argument(
        "--auto_head_only_rebalance_stop",
        action="store_true",
        help="Enable early stopping inside the optional classifier-only rebalance stage; --head_only_rebalance_epochs stays the maximum cap.",
    )
    p.add_argument(
        "--head_only_rebalance_use_logit_adjustment",
        action="store_true",
        help="Use logit-adjusted CE during the optional classifier-only rebalance stage",
    )
    p.add_argument(
        "--head_only_rebalance_weighted_sampler",
        action="store_true",
        help="Use the legacy weighted sampler during the optional classifier-only rebalance stage",
    )
    # Dataset I/O perf
    p.add_argument("--file_cache", type=int, default=64, help="small per-dataset file cache for per-video JSON (0=off)")
    p.add_argument(
        "--use_packed_skeleton_cache",
        action="store_true",
        help="Use a sidecar packed mmap cache for per-video skeleton JSON to improve high-worker throughput.",
    )
    p.add_argument(
        "--use_decoded_skeleton_cache",
        action="store_true",
        help="Use a sidecar decoded mmap cache that removes JSON parsing from Dataset.__getitem__.",
    )
    p.add_argument(
        "--packed_skeleton_cache_dir",
        type=str,
        default="",
        help="Optional directory for the packed skeleton cache (default: sidecar next to --json directory).",
    )
    p.add_argument(
        "--decoded_skeleton_cache_dir",
        type=str,
        default="",
        help="Optional directory for the decoded skeleton cache (default: sidecar next to --json directory).",
    )
    p.add_argument(
        "--packed_skeleton_cache_rebuild",
        action="store_true",
        help="Force rebuild of the packed skeleton cache before training.",
    )
    p.add_argument(
        "--decoded_skeleton_cache_rebuild",
        action="store_true",
        help="Force rebuild of the decoded skeleton cache before training.",
    )
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
    p.add_argument("--use_ctr_hand_refine", action="store_true", help="Enable hand-only CTR-style topology refinement")
    p.add_argument(
        "--ctr_in_stream_encoder",
        action="store_true",
        help="Also enable the hand-only CTR refinement inside the pre-fusion per-stream encoder.",
    )
    p.add_argument("--ctr_groups", type=int, default=4, help="Channel groups for hand-only adaptive topology")
    p.add_argument("--ctr_hand_nodes", type=int, default=42, help="Number of leading nodes treated as hands")
    p.add_argument(
        "--ctr_rel_channels",
        type=int,
        default=None,
        help="Optional relation channels per group for CTR refinement (default: auto)",
    )
    p.add_argument("--ctr_alpha_init", type=float, default=0.0, help="Initial scale for adaptive hand correction")
    # cosine head flags
    p.add_argument("--use_cosine_head", action="store_true")
    p.add_argument("--cosine_margin", type=float, default=0.2)
    p.add_argument("--cosine_scale", type=float, default=30.0)
    p.add_argument(
        "--cosine_subcenters",
        type=int,
        default=1,
        help="Number of cosine prototypes per class (1 keeps the original single-center cosine head).",
    )

    # Perf toggles
    p.add_argument("--tf32", action="store_true", help="Enable TF32 (matmul & cudnn)")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--channels_last", action="store_true", help="Use channels_last memory format")
    p.add_argument("--no_amp", action="store_true", help="Disable AMP/autocast (bf16) even on CUDA.")

    # Prior & TTA
    p.add_argument("--tta_mirror", action="store_true")

    args = p.parse_args(args=argv)
    if args.tb_full_logging:
        if not args.tensorboard:
            p.error("--tb_full_logging requires --tensorboard")
        args.tb_log_all_classes = True
        args.tb_log_tail_buckets = True
        args.tb_log_confusion_pairs = True
        args.tb_log_predictions_csv = True
        args.tb_log_errors_csv = True
        args.tb_log_topology = True
        args.tb_log_confusion = True
        args.tb_predictions_every = 1
        args.tb_confusion_every = 5
        args.tb_tables_k = 50
    args.tb_difficulty_freeze_epoch = max(1, int(args.tb_difficulty_freeze_epoch))
    args.tb_difficulty_ema = float(min(max(args.tb_difficulty_ema, 0.0), 0.999))
    args.auto_workers_max = max(0, int(args.auto_workers_max))
    args.auto_workers_warmup_batches = max(1, int(args.auto_workers_warmup_batches))
    args.auto_workers_measure_batches = max(1, int(args.auto_workers_measure_batches))
    args.cosine_subcenters = max(1, int(args.cosine_subcenters))
    args.family_loss_weight = max(0.0, float(args.family_loss_weight))
    args.family_warmup_epochs = max(0, int(args.family_warmup_epochs))
    args.family_label_smoothing = float(min(max(args.family_label_smoothing, 0.0), 0.999))
    args.num_families = max(0, int(args.num_families))
    args.head_only_rebalance_epochs = max(0, int(args.head_only_rebalance_epochs))
    args.head_only_rebalance_lr = max(0.0, float(args.head_only_rebalance_lr))
    if args.head_only_rebalance_only and args.head_only_rebalance_epochs <= 0:
        p.error("--head_only_rebalance_only requires --head_only_rebalance_epochs > 0")
    args.supcon_classes_per_batch = max(1, int(args.supcon_classes_per_batch))
    args.supcon_samples_per_class = max(1, int(args.supcon_samples_per_class))
    args.supcon_mixed_repeated_classes = max(1, int(args.supcon_mixed_repeated_classes))
    args.supcon_mixed_repeated_samples = max(2, int(args.supcon_mixed_repeated_samples))
    if args.supcon_class_balanced_batch and args.supcon_mixed_batch:
        p.error("--supcon_class_balanced_batch cannot be combined with --supcon_mixed_batch")
    if args.supcon_class_balanced_batch:
        if args.weighted_sampler:
            p.error("--supcon_class_balanced_batch cannot be combined with --weighted_sampler")
        expected_batch = int(args.supcon_classes_per_batch * args.supcon_samples_per_class)
        if expected_batch != int(args.batch):
            p.error(
                "--supcon_class_balanced_batch requires "
                "--supcon_classes_per_batch * --supcon_samples_per_class == --batch "
                f"(got {args.supcon_classes_per_batch} * {args.supcon_samples_per_class} = {expected_batch}, batch={args.batch})"
            )
    if args.supcon_mixed_batch:
        if args.weighted_sampler:
            p.error("--supcon_mixed_batch cannot be combined with --weighted_sampler")
        repeated_total = int(args.supcon_mixed_repeated_classes * args.supcon_mixed_repeated_samples)
        if repeated_total > int(args.batch):
            p.error(
                "--supcon_mixed_batch requires "
                "--supcon_mixed_repeated_classes * --supcon_mixed_repeated_samples <= --batch "
                f"(got {args.supcon_mixed_repeated_classes} * {args.supcon_mixed_repeated_samples} = {repeated_total}, batch={args.batch})"
            )
    if args.use_family_head and not args.family_map:
        p.error("--use_family_head requires --family_map")
    if args.family_eval and not args.use_family_head:
        p.error("--family_eval requires --use_family_head")
    return args
