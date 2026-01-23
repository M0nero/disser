# MSAGCN

## Last Best Run

PowerShell (Windows) example:

```
python -m msagcn.train `
  --json datasets/skeletons `
  --csv datasets/data/annotations.csv `
  --out datasets/skeletons_new_crop `
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
  --workers 24
```

Note: when `--json` points to a per-video directory, training prefers `*_pp.json` if present. Use `--no_prefer_pp` to force raw `*.json`.
