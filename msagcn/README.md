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
- head/mid/tail aggregates under `val_bucket/*`
- fixed tail watchlist curves under `val_watch/*`
- CTR diagnostics under `topology/*`
- text summaries under `tables/*`
- structured artifacts under `--out/analysis/`

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

To ablate a stronger variant that also applies the same hand-only refinement inside the
pre-fusion stream encoder, add:

```
  --ctr_in_stream_encoder
```

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
