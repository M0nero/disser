# Scripts

Utility scripts for dataset preparation and keypoint extraction.

## Extract Keypoints (PowerShell)

Run from repo root:

```
python scripts/extract_keypoints.py `
  --in-dir datasets/SLOVO/all `
  --out-dir datasets/skeletons/Slovo `
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

- `landmarks.zarr` as the main landmark store (`samples/<sample_id>/{raw,pp}/...`)
- `videos.parquet` with one row per sample and extractor-level metrics
- `frames.parquet` with one row per frame and scalar diagnostics only
- `runs.parquet` with one row per extractor run
- Optional debug NDJSON under `--out-dir/debug/` when `--ndjson` is enabled

Notes:

- Combined JSON is removed.
- Raw per-video `.json` and `_pp.json` are removed from the main extractor output.
- Full landmarks now live only in Zarr; Parquet contains metadata and scalar diagnostics.
- For refactors, compare two extractor artifact roots with `python scripts/check_kp_parity.py --baseline <old_out_dir> --candidate <new_out_dir>`.

## Runpod Scale-Out Workflow

For large Secure Cloud Runpod extraction runs, use shard manifests and per-pod shard outputs instead of writing all pods into one artifact root.

### 1. Prepare manifests and run spec

`prepare` enumerates tasks once using the normal extractor discovery logic, writes `all_tasks.jsonl`, splits it into deterministic shard manifests, and writes `run_spec.json`.

```bash
python scripts/runpod_extract.py prepare \
  --run-id phoenix_4090_v1 \
  --output-root /workspace/kp_export_runs \
  --scratch-root /tmp/kp_export \
  --pod-count 4 \
  --gpu-type "NVIDIA RTX 4090" \
  --container-image your-repo/kp-export:latest \
  --network-volume-id <volume_id> \
  -- \
  --in-dir datasets/phoenix/videos_phoenix/videos \
  --pattern "*/*.mp4" \
  --image-coords \
  --pose-every 1 \
  --postprocess
```

### 2. Launch pods through Runpod API

`launch` reads the generated `run_spec.json` and creates one pod per shard.

```bash
python scripts/runpod_extract.py launch \
  --run-spec /workspace/kp_export_runs/phoenix_4090_v1/run_spec.json
```

Set `RUNPOD_API_KEY` in the environment or pass `--api-key`.

### 3. Watch shard progress

Each pod writes:

- `status/shard-xxxxx.json`
- `logs/shard-xxxxx.events.jsonl`
- `logs/shard-xxxxx.failed_samples.txt`

Aggregate them with:

```bash
python scripts/runpod_extract.py watch \
  --run-spec /workspace/kp_export_runs/phoenix_4090_v1/run_spec.json
```

Live follow mode:

```bash
python scripts/runpod_extract.py watch \
  --run-spec /workspace/kp_export_runs/phoenix_4090_v1/run_spec.json \
  --follow \
  --compact
```

### 4. Merge shard artifacts

```bash
python scripts/runpod_extract.py merge \
  --run-root /workspace/kp_export_runs/phoenix_4090_v1
```

This writes the merged final artifact under:

- `/workspace/kp_export_runs/<run_id>/merged/landmarks.zarr`
- `/workspace/kp_export_runs/<run_id>/merged/videos.parquet`
- `/workspace/kp_export_runs/<run_id>/merged/frames.parquet`
- `/workspace/kp_export_runs/<run_id>/merged/runs.parquet`

### 5. Validate shard or merged outputs

```bash
python scripts/runpod_extract.py validate \
  --run-root /workspace/kp_export_runs/phoenix_4090_v1 \
  --include-shards \
  --include-merged
```

### 6. Build retry manifests from failures

```bash
python scripts/runpod_extract.py retry \
  --all-tasks /workspace/kp_export_runs/phoenix_4090_v1/manifests/all_tasks.jsonl \
  --failure-file /workspace/kp_export_runs/phoenix_4090_v1/logs/shard-00000.failed_samples.txt \
  --out-manifest /workspace/kp_export_runs/phoenix_4090_v1/manifests/retry_failed.jsonl
```

### 7. Stop or terminate pods

Terminate the whole run after merge/validate:

```bash
python scripts/runpod_extract.py terminate \
  --run-spec /workspace/kp_export_runs/phoenix_4090_v1/run_spec.json
```

Or stop without deleting:

```bash
python scripts/runpod_extract.py terminate \
  --run-spec /workspace/kp_export_runs/phoenix_4090_v1/run_spec.json \
  --stop-only
```

Notes:

- `scripts/runpod_entrypoint.sh` is the default pod entrypoint used by `Dockerfile.runpod`.
- `requirements.runpod.txt` is a minimal extractor-only runtime environment for Ubuntu pods.
- For GPU delegate mode on Ubuntu pods, extractor now forces `--jobs 1` by default unless you override it explicitly.

## Prepare PHOENIX-2014T Annotations

Before extraction, unpack PHOENIX annotation gzip files into TSV / JSONL:

```bash
python3 utils/prepare_phoenix_annotations.py \
  --annotations-dir datasets/phoenix \
  --videos-dir datasets/phoenix/videos_phoenix/videos \
  --out-dir datasets/phoenix/prepared
```

This writes:

- `datasets/phoenix/prepared/phoenix14t.all.tsv`
- `datasets/phoenix/prepared/phoenix14t.train.tsv`
- `datasets/phoenix/prepared/phoenix14t.val.tsv`
- `datasets/phoenix/prepared/phoenix14t.test.tsv`
- matching `jsonl` files and `summary.json`

Notes:

- `attachment_id` is written as `train__...`, `dev__...`, `test__...` to match the extractor output naming when you scan nested split folders from the PHOENIX videos root.
- The training `split` column is normalized as `train` / `val` / `test`, so PHOENIX `dev` becomes `val` there while the file ID stays `dev__...`.
- By default the generic `text` column is filled from the PHOENIX `gloss` field and the spoken translation is preserved as `translation`.

## Extract Keypoints For PHOENIX-2014T

Scan all split folders from the common videos root and keep the split name in the output slug:

```bash
python3 scripts/extract_keypoints.py \
  --in-dir datasets/phoenix/videos_phoenix/videos \
  --pattern "*/*.mp4" \
  --out-dir datasets/skeletons/phoenix \
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
  --jobs 18 \
  --skip-existing
```

After extraction, train from the prepared TSV plus the skeleton directory. For this repo, prefer `--use_decoded_skeleton_cache` during training rather than a single giant combined JSON.

## Extract Keypoints For Manifest Segments

Use this for IPNHand `D0X` background chunks or any other pre-cut segments manifest:

```bash
python scripts/extract_keypoints.py \
  --in-dir datasets/ipnhand/videos \
  --segments-manifest outputs/ipnhand/manifests/ipn_d0x_manifest.jsonl \
  --out-dir datasets/skeletons/ipnhand \
  --image-coords \
  --stride 1 \
  --pose-every 1 \
  --keep-pose-indices 0,9,10,11,12,13,14,15,16,23,24 \
  --postprocess \
  --jobs 8 \
  --skip-existing
```

Notes:

- In `--segments-manifest` mode the script matches rows by `video_id`, cuts `[start,end)` from the source video, and stores one Zarr group per `seg_uid`.
- The extractor no longer produces segment JSON files. `bio ipn-prelabel`, `scripts/ipn_json_quality_report.py`, and related IPN downstream tools still need a separate migration to consume Zarr/Parquet.
- For this extractor-only phase, stop after extraction if your next step still expects legacy JSON segment files.
- If you are rebuilding the full BIO dataset, the canonical next step is `python -m bio build-dataset`, which will reuse these IPN pools and write `outputs/bio_out_v2/...`.
- For IPNHand `D0X`, generate the manifest first with `python -m bio ipn-make-manifest --config bio/configs/ipn_default.json`.

## Notes

- Export writes `raw/` and optional `pp/` landmark arrays into the same Zarr sample group.
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
