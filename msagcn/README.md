# MSAGCN

Multi-stream Attention-GCN for isolated sign classification on hand keypoints (with optional pose).

## Quickstart

1) Prepare skeletons (see `scripts/README.md` for extraction).
2) Train:

```
python -m msagcn.train --json datasets/skeletons --csv datasets/data/annotations.csv --out outputs/runs/agcn_run
```

## Training Example (PowerShell)

```
python -m msagcn.train `
  --json datasets/skeletons `
  --csv datasets/data/annotations.csv `
  --out outputs/runs/agcn_run `
  --max_frames 64 `
  --temporal_crop resample `
  --streams joints,bones,velocity `
  --include_pose `
  --pose_keep 0,9,10,11,12,13,14,15,16,23,24 `
  --connect_cross_edges `
  --center `
  --center_mode masked_mean `
  --normalize `
  --norm_method p95 `
  --norm_scale 1.0 `
  --augment `
  --mirror_swap_only `
  --mirror_prob 0.5 `
  --rot_deg 10 `
  --scale_jitter 0.1 `
  --noise_sigma 0.01 `
  --epochs 120 `
  --batch 64 `
  --lr 5e-4 `
  --wd 5e-4 `
  --grad_clip 1.0 `
  --label_smoothing 0.05 `
  --depths 64,128,256,320 `
  --temp_ks 9,7,5,5 `
  --drop 0.05 `
  --droppath 0.03 `
  --stream_drop_p 0.05 `
  --weighted_sampler `
  --use_logit_adjustment `
  --use_cosine_head `
  --cosine_margin 0.2 `
  --cosine_scale 30 `
  --ema_decay 0.999 `
  --warmup_frac 0.1 `
  --tensorboard `
  --logdir runs `
  --run_name agcn_run `
  --log_every_steps 10 `
  --flush_secs 30 `
  --tb_support_topk 50 `
  --tb_worstk_f1 50 `
  --tb_confusion_topk 50 `
  --tb_log_confusion `
  --tb_log_examples `
  --tb_examples_k 5 `
  --tb_examples_every 5 `
  --workers 24
```

## TensorBoard

Enable event logs:

```
python -m msagcn.train ... --tensorboard --logdir runs --run_name agcn_run --log_every_steps 1 \
  --tb_log_examples --tb_examples_k 5 --tb_examples_every 5
```

Logs are written to `runs/<run_name>`. Start TensorBoard with:

```
tensorboard --logdir runs
```

Enable the full monitoring bundle for macro-F1, tail classes, confusion pairs, per-class curves,
prediction/error CSVs, and CTR topology diagnostics:

```
python -m msagcn.train ... --tensorboard --logdir outputs/runs --run_name agcn_full \
  --tb_full_logging --tb_confusion_every 5 --tb_predictions_every 1 --tb_tables_k 50
```

With `--tb_full_logging`, the run writes:
- stable per-class TensorBoard tags under `val_all/*`
- support buckets under `val_bucket/*` when train support is non-uniform
- difficulty buckets under `val_difficulty/*` when train support is uniform (balanced few-shot splits)
- fixed support-tail or hardest-class watchlist curves under `val_watch/*`
- CTR diagnostics under `topology/*`
- text summaries under `tables/*`
- structured artifacts under `--out/analysis/`

Bucket analytics defaults to `--tb_bucket_mode auto`. In balanced runs where every class has the same train support, the logger switches from support buckets to difficulty buckets and freezes the assignment after `--tb_difficulty_freeze_epoch` using an EMA of per-class F1.

## Outputs

Training writes into `--out`:
- `best.ckpt` (best by macro-F1)
- `last.ckpt` (latest epoch, for crash-safe resume)
- `history.json` (epoch metrics)
- `report_epXXX.json` (per-class report, every 5 epochs)
- `label2idx.json`, `ds_config.json`

When full logging is enabled, `--out/analysis/` also contains:
- `run_manifest.json`, `train_support.csv`, `buckets.csv`, `watchlist.csv`, `epoch_index.csv`
- `per_class/per_class_epXXX.{csv,json}`
- `predictions/predictions_epXXX.csv`
- `errors/errors_epXXX.csv`
- `confusion/confusion_pairs_epXXX.{csv,json}`
- `best_per_class.*`, `best_predictions.csv`, `best_errors.csv`, `best_confusion_pairs.*`

Resume training from the latest checkpoint:

```
python -m msagcn.train ... --out outputs/runs/agcn_run --resume outputs/runs/agcn_run/last.ckpt --epochs 120
```

Notes:
- `--epochs` is the total target epoch count, not "extra epochs after resume".
- `--resume_model_only` loads model / EMA weights from a checkpoint but resets optimizer, scheduler, scaler, best score, and history.
- early stopping defaults to auto-resolved patience/min-delta; use positive values to override, or `--early_stop_patience 0` to disable it.

## Experimental Graph Refinement

An optional hand-first CTR-style refinement branch can be enabled on top of the existing static graph prior.
It only adapts the leading hand subgraph (default: the first 42 nodes) and leaves pose / cross-edge context on the static path.

Example:

```
python -m msagcn.train --json datasets/skeletons --csv datasets/data/annotations.csv --out outputs/runs/agcn_ctr \
  --use_ctr_hand_refine --ctr_groups 4 --ctr_hand_nodes 42 --ctr_alpha_init 0.0
```

For Windows runs with many workers on a per-video JSON directory, you can enable a sidecar packed
memory-mapped cache to reduce tiny-file overhead and let the OS page cache use RAM more effectively:

```
python -m msagcn.train --json datasets/skeletons --csv datasets/data/annotations.csv --out outputs/runs/agcn_ctr \
  --workers 24 --use_packed_skeleton_cache
```

Optional flags:
- `--packed_skeleton_cache_dir <path>` to choose a custom cache location
- `--packed_skeleton_cache_rebuild` to rebuild the packed cache

If the bottleneck is no longer tiny-file I/O but JSON parsing / Python frame traversal, use the
stronger decoded cache instead. It stores already-decoded hand / pose arrays in a single mmap-backed
sidecar and removes JSON parsing from `Dataset.__getitem__`:

```
python -m msagcn.train --json datasets/skeletons --csv datasets/data/annotations.csv --out outputs/runs/agcn_ctr \
  --workers 24 --use_decoded_skeleton_cache
```

Optional flags:
- `--decoded_skeleton_cache_dir <path>` to choose a custom cache location
- `--decoded_skeleton_cache_rebuild` to rebuild the decoded cache
- if both decoded and packed cache flags are passed, the decoded cache path wins

## Auto Workers

For portable training on different machines, you can let `msagcn` benchmark and cache a safe
train-loader profile automatically instead of hard-coding `--workers`.

- `--auto_workers` benchmarks a small set of worker counts on the current machine
- it keeps the existing Windows-safe loader rules for raw / packed / decoded skeleton paths
- the chosen profile is cached locally by machine fingerprint and reused on later matching runs
- if every benchmark candidate fails, training falls back to a safe `workers=0` profile instead of crashing

Run-local artifacts:
- `analysis/auto_workers_decision.json`
- `analysis/auto_workers_candidates.csv`

Example:

```
python -m msagcn.train --json datasets/skeletons --csv datasets/data/annotations.csv --out outputs/runs/agcn_auto \
  --use_decoded_skeleton_cache --auto_workers
```

Optional flags:
- `--auto_workers_max <N>` to cap the benchmark ladder
- `--auto_workers_rebench` to ignore the cached decision and benchmark again
- `--auto_workers_warmup_batches <N>` and `--auto_workers_measure_batches <N>` to control benchmark length

To ablate a stronger variant that also applies the same hand-only refinement inside the
pre-fusion stream encoder, add:

```
  --ctr_in_stream_encoder
```

## Supervised Contrastive Auxiliary

An optional supervised-contrastive auxiliary loss can be added on the pooled embedding
after graph pooling and `embed_norm`, while keeping the classifier head unchanged.
This is a training-only auxiliary objective:

- the cosine / classifier head remains the main classifier
- evaluation stays logits-based
- anchors without same-label positives in the current batch are safely ignored
- the effect may be weak when a batch has very few repeated labels
- by default the auxiliary starts right after LR warmup; use `0` to force it from epoch 1

Example:

```
python -m msagcn.train --json datasets/skeletons --csv datasets/data/annotations.csv --out outputs/runs/agcn_supcon \
  --use_supcon --supcon_weight 0.05 --supcon_temp 0.07
```

To delay the auxiliary until later in training, add for example:

```
  --supcon_start_epoch 5
```

Use `--supcon_start_epoch -1` for the default auto mode, which resolves to `warmup_epochs + 1`.

## Class-Balanced SupCon Batches

Batch-local SupCon can be weak when ordinary random shuffle produces too few repeated labels inside
each train batch. For SupCon experiments you can optionally replace random shuffle with a simple
class-balanced batch sampler:

- each train batch is built as `classes_per_batch x samples_per_class`
- this affects only the train loader
- validation stays unchanged
- this is not the old `--weighted_sampler`

Example for a `64`-sample batch with `16` classes and `4` samples per class:

```
python -m msagcn.train --json datasets/skeletons --csv datasets/data/annotations.csv --out outputs/runs/agcn_supcon_cb \
  --batch 64 --use_supcon --supcon_weight 0.05 --supcon_temp 0.07 \
  --supcon_class_balanced_batch --supcon_classes_per_batch 16 --supcon_samples_per_class 4
```

When enabled, `--supcon_classes_per_batch * --supcon_samples_per_class` must exactly equal `--batch`.

## Hybrid SupCon Batches

If fully class-balanced `N x K` batches reduce class diversity too much, you can use a hybrid
SupCon batch sampler instead:

- a fixed number of labels are repeated inside the batch to create positive pairs
- the remaining slots are filled with singleton labels to keep many unique classes per batch
- this affects only the train loader
- validation stays unchanged

Example for `batch=64` with `16` repeated labels, `2` samples each, plus `32` singleton labels:

```
python -m msagcn.train --json datasets/skeletons --csv datasets/data/annotations.csv --out outputs/runs/agcn_supcon_mixed \
  --batch 64 --use_supcon --supcon_weight 0.03 --supcon_temp 0.07 \
  --supcon_mixed_batch --supcon_mixed_repeated_classes 16 --supcon_mixed_repeated_samples 2
```

When enabled:
- `--supcon_mixed_repeated_samples` must be at least `2`
- `--supcon_mixed_repeated_classes * --supcon_mixed_repeated_samples <= --batch`

## Notes

- When `--json` points to a per-video directory, training prefers `*_pp.json` if present.
  Use `--no_prefer_pp` to force raw `*.json`.
- Validation is deterministic even if `--temporal_crop=random`.
- If `--include_pose` is set, pose↔hand cross edges are enabled by default.
  Use `--no_cross_edges` to disable.
- `--use_ctr_hand_refine` keeps the full static graph path and adds a residual adaptive correction only on hand↔hand relations.
- `--ctr_in_stream_encoder` extends the same CTR hand refinement into the per-stream encoder before fusion; keep it off for the cleaner backbone-only ablation.
  If you enable it on top of an older checkpoint, prefer `--resume_model_only` or start a new `--out` directory.
- `--ctr_rel_channels` is optional; when omitted the model uses a small auto-sized relation width per group.
- Use `--no_amp` to disable autocast/AMP (useful for reproducibility or CPU runs).
- For 1000 classes with tiny val support, rely on `val/f1_micro`, `val/f1_weighted`,
  top-K metrics, and worst-K examples in TensorBoard.
