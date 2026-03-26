# BIO Pipeline

This module contains a canonical BIO segmentation pipeline:

1) `prelabel`: build Step1 BIO clips from skeletons + CSV
2) `synth-build`: generate Step2 synthetic continuous sequences
3) `train`: train the BIO tagger on synthetic shards
4) `build-dataset`: canonical v2 rebuild with overlap audit + Step1 + Step2

All outputs now live under `outputs/`, and datasets under `datasets/`.

## Quickstart

```bash
python -m bio build-dataset
python -m bio train \
  --config bio/configs/bio_default.json \
  --train_dir outputs/bio_out_v2/synth_train \
  --val_dir outputs/bio_out_v2/synth_val \
  --out_dir outputs/runs/bio_gru_v2
```

`build-dataset` is the preferred entrypoint. It writes rebuilt artifacts into
`outputs/bio_out_v2`. Older `outputs/bio_out` snapshots are unsupported and
should not be used for new experiments. Rebuilds are staged transactionally
before promotion so failed runs do not leave mixed canonical artifacts.

## CLI

Entry point:

```
python -m bio <command> [args]
```

Commands:
- `prelabel`
- `synth-build`
- `train`
- `build-dataset`
- `smoke-test`
- `ipn-make-manifest`
- `ipn-prelabel`

Run `python -m bio <command> -h` for details.

## Config Templates

Templates live in `bio/configs/`:
- `bio/configs/bio_default.json`
- `bio/configs/ipn_default.json`
- `bio/configs/ipn_val.json`
- `bio/configs/bio_val.json`

For the full `SLOVO + IPNHand no_event` recipe, see:
`docs/bio_slovo_ipn_pipeline.md`

Each stage writes a `config.json` into its output directory for reproducibility.
Step1 also writes `csv_audit.json` and `rejected_rows.json`; Step2 writes
decision-grade `stats.json` with `O` source breakdown and dataset signature.
Canonical stages also write `dataset_manifest.json`.

Prelabel prefers `*_pp.json` when both raw and post-processed files exist. Override with `--no_prefer_pp` or set `"prefer_pp": false` in config.
Default SLOVO skeletons path is `datasets/skeletons/Slovo` (flat directory of per-video JSONs).

## Canonical v2 Rebuild

```bash
python -m bio build-dataset \
  --out_root outputs/bio_out_v2 \
  --ipn_out_root outputs/ipnhand
```

This performs:

1. split overlap audit for SLOVO and IPNHand
2. SLOVO Step1 prelabels for `train` and `val`
3. SLOVO `no_event` Step1 pools for `train` and `val`
4. IPNHand `D0X` Step1 pools for `train` and `val`
5. Step2 synth rebuild with native SLOVO tails first, SLOVO `no_event` second, IPNHand third

It writes:

- `outputs/bio_out_v2/overlap_report.json`
- `outputs/bio_out_v2/build_summary.json`
- `outputs/bio_out_v2/dataset_manifest.json`
- `outputs/bio_out_v2/prelabels_*`
- `outputs/bio_out_v2/synth_train`
- `outputs/bio_out_v2/synth_val`

## Manual SLOVO/IPN Run

```
python -m bio prelabel \
  --config bio/configs/bio_default.json \
  --split train \
  --out outputs/bio_out_v2/prelabels_slovo_train

python -m bio prelabel \
  --config bio/configs/bio_default.json \
  --split val \
  --out outputs/bio_out_v2/prelabels_slovo_val

python -m bio prelabel \
  --config bio/configs/bio_default.json \
  --skeletons datasets/skeletons/Slovo \
  --csv datasets/data/annotations_no_event.csv \
  --split train \
  --out outputs/bio_out_v2/prelabels_slovo_noev_train

python -m bio prelabel \
  --config bio/configs/bio_default.json \
  --skeletons datasets/skeletons/Slovo \
  --csv datasets/data/annotations_no_event.csv \
  --split val \
  --out outputs/bio_out_v2/prelabels_slovo_noev_val

python -m bio synth-build \
  --config bio/configs/bio_default.json \
  --prelabel_dir outputs/bio_out_v2/prelabels_slovo_train \
  --preferred_noev_prelabel_dir outputs/bio_out_v2/prelabels_slovo_noev_train \
  --extra_noev_prelabel_dir outputs/ipnhand/prelabels_train \
  --out_dir outputs/bio_out_v2/synth_train

python -m bio synth-build \
  --config bio/configs/bio_val.json \
  --prelabel_dir outputs/bio_out_v2/prelabels_slovo_val \
  --preferred_noev_prelabel_dir outputs/bio_out_v2/prelabels_slovo_noev_val \
  --extra_noev_prelabel_dir outputs/ipnhand/prelabels_val \
  --out_dir outputs/bio_out_v2/synth_val
```

`preferred_noev_prelabel_dir` and `extra_noev_prelabel_dir` must be canonical
`O`-only Step1 pools. Mixed directories that still contain sign clips are
rejected.

Note: current `synth-build` now forms `O` in this order:
1. tails before/after the sign span inside ordinary SLOVO sign clips
2. dedicated SLOVO `no_event`
3. external IPN negatives

Defaults now use:

- `min_tail_len = 4`
- `primary_noev_prob = 0.90`
- `source_sampling = uniform_source`
- `sign_sampling = uniform_label_source`
- `stitch_noev_chunks = true`

```
python -m bio train \
  --config bio/configs/bio_default.json \
  --train_dir outputs/bio_out_v2/synth_train \
  --val_dir outputs/bio_out_v2/synth_val \
  --out_dir outputs/runs/bio_gru_v2 \
  --tensorboard \
  --console_log_format text \
  --logdir runs \
  --run_name bio_gru_v2 \
  --log_every_steps 10 \
  --flush_secs 30 \
  --tb_log_examples \
  --tb_examples_k 5 \
  --tb_examples_every 5
```

Stdout is human-readable by default. Use `--console_log_format json` only if you
want raw JSON events in the console; `train_log.jsonl` stays structured JSONL
either way.

## TensorBoard

Enable event logs for training:

```
python -m bio train ... --tensorboard --logdir runs --run_name bio_gru_v2 --log_every_steps 1 \
  --tb_log_examples --tb_examples_k 5 --tb_examples_every 5
```

Logs are written to `runs/<run_name>`. Start TensorBoard with:

```
tensorboard --logdir runs
```

## Streaming inference (camera / iOS)

Use `BioTagger.stream_step(...)` with a short temporal buffer. Recommended window:

- **W = 9–16 frames** (covers the causal conv receptive field; good balance for latency/quality).

Example (Python-side logic; same flow for iOS):

```python
from bio.core.model import BioTagger, BioModelConfig

cfg = BioModelConfig(num_joints=42)
model = BioTagger(cfg).eval()

W = 12  # recommended: 9–16
state = model.init_stream_state(batch_size=1, window=W, device="cpu")

for pt, mask in stream_frames():  # pt: (V,3), mask: (V,1)
    logits, state = model.stream_step(pt, mask, state)
    pred = int(logits.argmax(dim=-1).item())  # 0=O,1=B,2=I
    # use pred for the current frame
```

## Outputs

Training writes into `--out_dir`:
- `last.pt`
- `last_model.pt`
- `epoch_XXXX.pt` compact resume snapshots every `--save_every_epochs`
- `best_boundary.pt`
- `best_boundary_model.pt`
- `best_balanced.pt`
- `best_balanced_model.pt`
- `best.pt` (compatibility alias for `best_boundary.pt`)
- `best_model.pt` (model-only alias for `best_boundary_model.pt`)
- `train_log.jsonl`
- `history.json`
- `analysis/runtime_summary.json` (startup phases, loader benchmark results, effective worker selection, meta parsing flags)
- `analysis/` (when `--save_analysis_artifacts` is enabled; threshold sweeps every epoch, prediction/error tables on `--prediction_artifacts_every` cadence plus best/final epochs)
- `config.json`
- `dataset_manifest.json`

## IPN Hand (suggested layout)

Suggested in-repo structure:

```
datasets/ipnhand/
  annotations/
    Annot_TrainList.txt
    Annot_TestList.txt
  videos/                 # raw .avi files
datasets/skeletons/
  ipnhand/                # extracted per-segment JSON / _pp.json
outputs/ipnhand/
  manifests/
  quality_report/
  prelabels_train/        # Step1-compatible IPN no_event pool for train
  prelabels_val/          # Step1-compatible IPN no_event pool for val
```

Example commands:

```
python -m bio ipn-make-manifest \
  --config bio/configs/ipn_default.json

python scripts/extract_keypoints.py \
  --in-dir datasets/ipnhand/videos \
  --segments-manifest outputs/ipnhand/manifests/ipn_d0x_manifest.jsonl \
  --out-dir datasets/skeletons/ipnhand \
  --eval-report outputs/ipnhand/eval_report.json \
  --image-coords \
  --postprocess \
  --jobs 8 \
  --skip-existing

python scripts/ipn_json_quality_report.py \
  --ipn_dir datasets/skeletons/ipnhand \
  --manifest outputs/ipnhand/manifests/ipn_d0x_manifest.jsonl \
  --out outputs/ipnhand/quality_report \
  --write_clean_manifest outputs/ipnhand/manifests/ipn_d0x_manifest_clean.jsonl

python -m bio ipn-prelabel \
  --config bio/configs/ipn_default.json \
  --manifest outputs/ipnhand/manifests/ipn_d0x_manifest_clean.jsonl \
  --segments_dir datasets/skeletons/ipnhand \
  --split train \
  --out_dir outputs/ipnhand/prelabels_train

python -m bio ipn-prelabel \
  --config bio/configs/ipn_val.json \
  --manifest outputs/ipnhand/manifests/ipn_d0x_manifest_clean.jsonl \
  --segments_dir datasets/skeletons/ipnhand \
  --split val \
  --out_dir outputs/ipnhand/prelabels_val
```

`bio ipn-prelabel` accepts either per-segment `.npz` files or JSON / `_pp.json`
files produced by `scripts/extract_keypoints.py --segments-manifest ...`, and it
can now filter the manifest by `--split` so train/val negatives stay separate.

Inside `synth-build`, ordinary sign clips now also contribute their leading and
trailing `O` tails into the primary no_event pool. Sampling is source-aware:
the builder first chooses `primary` vs `extra`, then samples uniformly by
`source_group` inside that pool. Long `O` spans are stitched from multiple
chunks by default instead of repeat-padding one clip.

If you want a one-shot helper for the IPN `O/no_event` pool build,
use:

```
bash bio/tools/build_ipn_noevent_pool.sh
```

## Smoke Test

```
bash bio/tools/smoke_test.sh
```

This runs a tiny prelabel + synth-build sanity check.
