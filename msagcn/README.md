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

## Outputs

Training writes into `--out`:
- `best.ckpt` (best by macro-F1)
- `history.json` (epoch metrics)
- `report_epXXX.json` (per-class report, every 5 epochs)
- `label2idx.json`, `ds_config.json`

## Experimental Graph Refinement

An optional hand-first CTR-style refinement branch can be enabled on top of the existing static graph prior.
It only adapts the leading hand subgraph (default: the first 42 nodes) and leaves pose / cross-edge context on the static path.

Example:

```
python -m msagcn.train --json datasets/skeletons --csv datasets/data/annotations.csv --out outputs/runs/agcn_ctr \
  --use_ctr_hand_refine --ctr_groups 4 --ctr_hand_nodes 42 --ctr_alpha_init 0.0
```

## Notes

- When `--json` points to a per-video directory, training prefers `*_pp.json` if present.
  Use `--no_prefer_pp` to force raw `*.json`.
- Validation is deterministic even if `--temporal_crop=random`.
- If `--include_pose` is set, pose↔hand cross edges are enabled by default.
  Use `--no_cross_edges` to disable.
- `--use_ctr_hand_refine` keeps the full static graph path and adds a residual adaptive correction only on hand↔hand relations.
- `--ctr_rel_channels` is optional; when omitted the model uses a small auto-sized relation width per group.
- Use `--no_amp` to disable autocast/AMP (useful for reproducibility or CPU runs).
- For 1000 classes with tiny val support, rely on `val/f1_micro`, `val/f1_weighted`,
  top-K metrics, and worst-K examples in TensorBoard.
