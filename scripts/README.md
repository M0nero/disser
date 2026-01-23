# Scripts

## Last Best Export Run

Run from the repo root:

```
python3 scripts/extract_keypoints.py \
  --in-dir datasets/SLOVO/all \
  --out-dir datasets/skeletons/Slovo \
  --eval-report outputs/out_dataset_default/eval_report.json \
  --seed 0 \
  --image-coords \
  --stride 1 \
  --pose-every 1 \
  --keep-pose-indices 0,9,10,11,12,13,14,15,16,23,24 \
  --min-det 0.40 \
  --min-track 0.35 \
  --second-pass \
  --min-hand-score 0.10 \
  --hand-score-lo 0.40 \
  --hand-score-hi 0.80 \
  --hand-score-source presence \
  --anchor-score 0.85 \
  --tracker-init-score 0.75 \
  --tracker-update-score 0.65 \
  --pose-dist-qual-min 0.55 \
  --pose-side-reassign-ratio 0.60 \
  --occ-hyst-frames 20 \
  --occ-return-k 1.30 \
  --track-max-gap 20 \
  --track-score-decay 0.93 \
  --track-reset-ms 300 \
  --sp-trigger-below 0.80 \
  --sp-roi-frac 0.30 \
  --sp-margin 0.45 \
  --sp-escalate-step 0.25 \
  --sp-escalate-max 2.0 \
  --sp-overlap-iou 0.12 \
  --sp-overlap-shrink 0.60 \
  --sp-overlap-penalty-mult 1.5 \
  --sp-center-penalty 0.20 \
  --sp-label-relax 0.25 \
  --sanity-scale-range "0.70,1.35" \
  --sanity-wrist-k 2.0 \
  --sanity-bone-tol 0.30 \
  --sanity-anchor-max-gap 30 \
  --interp-hold 7 \
  --postprocess \
  --pp-max-gap 20 \
  --pp-smoother ema \
  --mp-backend tasks \
  --jobs 18
```

Note: export writes both raw and `_pp` JSON files; downstream training (msagcn/bio) prefers `_pp` by default.
