# Scripts

Utility scripts for dataset preparation and keypoint extraction.

## Extract Keypoints (PowerShell)

Run from repo root:

```
python scripts/extract_keypoints.py `
  --in-dir datasets/SLOVO/all `
  --out-dir datasets/skeletons/Slovo `
  --eval-report outputs/Slovo/eval_report.json `
  --seed 0 `
  --image-coords `
  --stride 1 `
  --pose-every 1 `
  --keep-pose-indices 0,9,10,11,12,13,14,15,16,23,24 `
  --min-det 0.40 `
  --min-track 0.35 `
  --second-pass `
  --min-hand-score 0.10 `
  --hand-score-lo 0.40 `
  --hand-score-hi 0.80 `
  --hand-score-source presence `
  --anchor-score 0.85 `
  --tracker-init-score 0.75 `
  --tracker-update-score 0.65 `
  --pose-dist-qual-min 0.55 `
  --pose-side-reassign-ratio 0.60 `
  --occ-hyst-frames 20 `
  --occ-return-k 1.30 `
  --track-max-gap 20 `
  --track-score-decay 0.93 `
  --track-reset-ms 300 `
  --sp-trigger-below 0.80 `
  --sp-roi-frac 0.30 `
  --sp-margin 0.45 `
  --sp-escalate-step 0.25 `
  --sp-escalate-max 2.0 `
  --sp-overlap-iou 0.12 `
  --sp-overlap-shrink 0.60 `
  --sp-overlap-penalty-mult 1.5 `
  --sp-center-penalty 0.20 `
  --sp-label-relax 0.25 `
  --sanity-scale-range "0.70,1.35" `
  --sanity-wrist-k 2.0 `
  --sanity-bone-tol 0.30 `
  --sanity-anchor-max-gap 30 `
  --interp-hold 7 `
  --postprocess `
  --pp-max-gap 20 `
  --pp-smoother ema `
  --mp-backend tasks `
  --jobs 12 `
  --skip-existing
```

## Outputs

- Per-video JSON files in `--out-dir`
- Optional `_pp.json` (post-processed)
- Eval report JSON (if `--eval-report` is set)

## Extract Keypoints For Manifest Segments

Use this for IPNHand `D0X` background chunks or any other pre-cut segments manifest:

```bash
python scripts/extract_keypoints.py \
  --in-dir datasets/ipnhand/videos \
  --segments-manifest outputs/ipnhand/manifests/ipn_d0x_manifest.jsonl \
  --out-dir datasets/skeletons/ipnhand \
  --eval-report outputs/ipnhand/eval_report.json \
  --image-coords \
  --stride 1 \
  --pose-every 1 \
  --keep-pose-indices 0,9,10,11,12,13,14,15,16,23,24 \
  --postprocess \
  --jobs 8 \
  --skip-existing
```

Notes:

- In `--segments-manifest` mode the script matches rows by `video_id`, cuts `[start,end)` from the source video, and writes one output file per `seg_uid`.
- Outputs are JSON / `_pp.json` segment files; `bio ipn-prelabel` can now consume those directly, so an extra JSON->NPZ converter is no longer required.
- The intended next step is to run `bio ipn-prelabel` separately for `--split train` and `--split val`, consuming `datasets/skeletons/ipnhand` and producing `outputs/ipnhand/prelabels_train` and `outputs/ipnhand/prelabels_val`.
- If you are rebuilding the full BIO dataset, the canonical next step is `python -m bio build-dataset`, which will reuse these IPN pools and write `outputs/bio_out_v2/...`.
- For IPNHand `D0X`, generate the manifest first with `python -m bio ipn-make-manifest --config bio/configs/ipn_default.json`.

## Notes

- Export writes both raw and `_pp` JSON files; downstream training (msagcn/bio) prefers `_pp` by default.
- For Windows, use PowerShell backticks or convert to a single-line command.

## BIO Rebuild + Train (PowerShell)

For the canonical BIO dataset rebuild followed by training:

```powershell
.\scripts\run_bio_rebuild_and_train.ps1
```

Useful modes:

```powershell
# Full Step1 + Step2 rebuild, then train
.\scripts\run_bio_rebuild_and_train.ps1 `
  -OutRoot outputs\bio_out_v4 `
  -RunDir outputs\runs\bio_v4_run

# Faster path: reuse existing Step1 prelabels, rebuild only Step2 synth shards, then train
.\scripts\run_bio_rebuild_and_train.ps1 `
  -Step2Only `
  -OutRoot outputs\bio_out_v4 `
  -RunDir outputs\runs\bio_v4_run

# Faster debug retrain: smaller offline synth + train
.\scripts\run_bio_rebuild_and_train.ps1 `
  -Step2Only `
  -FastDebug `
  -OutRoot outputs\bio_out_v4 `
  -RunDir outputs\runs\bio_v4_run

# Explicit sample counts with synth auto-workers
.\scripts\run_bio_rebuild_and_train.ps1 `
  -Step2Only `
  -TrainSynthSamples 30000 `
  -ValSynthSamples 3000 `
  -OutRoot outputs\bio_out_v4 `
  -RunDir outputs\runs\bio_v4_run

# Force a manual synth worker count instead of auto-workers
.\scripts\run_bio_rebuild_and_train.ps1 `
  -Step2Only `
  -SynthWorkers 8 `
  -OutRoot outputs\bio_out_v4 `
  -RunDir outputs\runs\bio_v4_run
```

The `-Step2Only` mode is usually much faster than `python -m bio build-dataset`
because it skips the full Step1 prelabel rebuild and only regenerates
`synth_train` / `synth_val` before training.
The script now uses synth auto-workers by default. Use `-SynthWorkers <N>` only
when you want to force a manual shard-generation process count.
