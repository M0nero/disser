# BIO Pipeline

This module contains a canonical BIO segmentation pipeline:

1) `prelabel`: build Step1 BIO clips from skeletons + CSV
2) `synth-build`: generate Step2 synthetic continuous sequences
3) `train`: train the BIO tagger on synthetic shards
4) `build-dataset`: canonical Slovo-only trimmed rebuild with overlap audit + Step1 + Step2

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
decision-grade `stats.json` with `O` source breakdown, seam realism metrics,
and dataset signature. Canonical stages also write `dataset_manifest.json`.

Prelabel prefers `*_pp.json` when both raw and post-processed files exist. Override with `--no_prefer_pp` or set `"prefer_pp": false` in config.
Default SLOVO skeletons path is `datasets/skeletons/Slovo` (flat directory of per-video JSONs).

The canonical rebuild now uses a runtime-first preprocessing contract:

- Step1 stores raw canonical hand skeletons as `pts_raw`
- Step1 still writes `pts` for train/debug, but these are derived by the shared
  `canonical_hands42_v3` preprocessing path
- Step1 `trimmed_mode = true` keeps the full trimmed clip and treats CSV
  `begin/end` as gold BIO boundaries inside that full clip
- Step1 now maps `source_group` to `user_id` when the CSV provides it, so split
  audit and same-source sampling become signer-aware instead of clip-aware
- runtime and offline replay both use the same causal center/scale update
- Step2 stitches raw chunks, aligns them geometrically, then applies the same
  shared preprocessing before writing synth shards

## Canonical Rebuild

```bash
python -m bio build-dataset \
  --out_root outputs/bio_out_v2
```

This performs:

1. split overlap audit for SLOVO sign / SLOVO no_event
2. SLOVO Step1 prelabels for `train` and `val`
3. SLOVO `no_event` Step1 pools for `train` and `val`
4. Step2 main `main_continuous` synth rebuild with strict same-source semantics
5. optional separate stress dataset when `--emit_stress_dataset` is enabled
6. raw-space alignment + boundary transitions before shared v3 preprocessing

It writes:

- `outputs/bio_out_v2/overlap_report.json`
- `outputs/bio_out_v2/build_summary.json`
- `outputs/bio_out_v2/dataset_manifest.json`
- `outputs/bio_out_v2/prelabels_*`
- `outputs/bio_out_v2/synth_train`
- `outputs/bio_out_v2/synth_val`

The default rebuild is now Slovo-only. No external `ipnhand` no-event pool is
used in the main path.

If the original Slovo CSV is not signer-disjoint, rewrite it first:

```bash
python -m bio signer-split \
  --csv datasets/data/annotations.csv \
  --csv datasets/data/annotations_no_event.csv \
  --out_dir datasets/data/slovo_signer_split
```

Then point `build-dataset` / `prelabel` at
`datasets/data/slovo_signer_split/annotations.csv` and
`datasets/data/slovo_signer_split/annotations_no_event.csv`.

## Manual Slovo-Only Run

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
  --out_dir outputs/bio_out_v2/synth_train

python -m bio synth-build \
  --config bio/configs/bio_val.json \
  --prelabel_dir outputs/bio_out_v2/prelabels_slovo_val \
  --preferred_noev_prelabel_dir outputs/bio_out_v2/prelabels_slovo_noev_val \
  --out_dir outputs/bio_out_v2/synth_val
```

`preferred_noev_prelabel_dir` must be a canonical SLOVO `O`-only Step1 pool.
Mixed directories that still contain sign clips are rejected.

Note: current `synth-build` now forms `O` in this order:
1. tails before/after the sign span inside ordinary SLOVO sign clips
2. dedicated SLOVO `no_event`
3. optional pause-style synthetic filler when same-source `O` is unavailable

Defaults now use:

- `min_tail_len = 4`
- `primary_noev_prob = 1.00`
- `source_sampling = uniform_source`
- `sign_sampling = uniform_label_source`
- `stitch_noev_chunks = true`
- `pad_mode = both_no_event`
- `leading_noev_prob = 0.65`
- `leading_noev_min = 12`
- `leading_noev_max = 96`
- `all_noev_prob = 0.15`
- `dataset_profile = main_continuous`
- `same_source_sequence_prob = 1.0`
- `cross_source_boundary_prob = 0.0`
- `continuous_mode_weight = 0.85`
- `hard_negative_mode_weight = 0.15`
- `stress_mode_weight = 0.0`
- `sampling_profile = prelabel_empirical`
- `blend_prob = 0.0`
- `transition_k_min = 2`
- `transition_k_max = 6`

This matters for startup negatives and train/infer parity:

- Step1 `trimmed_mode = true` keeps the full trimmed clip and treats CSV
  `begin/end` as gold BIO boundaries inside that clip instead of recropping by
  motion.

- Step2 can now prepend an explicit leading `O/no_event` prefix before the
  first sign instead of over-sampling sequences that begin with `B` at frame 0.
- Step2 can also emit fully negative `all-O` sequences with no `B/I` at all.
- `stats.json` now reports:
  - fraction of samples with a leading `O` prefix
  - fraction of `all-O` samples
  - `first_B_frame` stats/distribution
  - how often `first_B_frame == 0`

Step2 now also applies runtime-like hand corruption after assembly:

- single-hand dropout spans
- both-hands dropout spans
- short mask flicker spans
- small coordinate jitter on visible joints

This is meant to make training inputs look closer to real MediaPipe failures.
Step2 also now assembles sequences in raw coordinate space and reports seam
realism directly. `stats.json` / `summary.json` now reports:

- fraction of samples with post-assembly hand corruption
- startup-prefix frames that remain fully no-hand after corruption
- longest no-hand span statistics
- corruption counts by type
- `semantic_seam_realism` for real chunk boundaries
- `expanded_seam_realism` for transition-expanded boundaries
- semantic boundary/internal jump ratios for center and scale
- semantic boundary deltas for wrist position and valid-joint counts
- `semantic_source_mixing_report` and `expanded_source_mixing_report`

Relevant new config knobs:

- `preprocessing_version = canonical_hands42_v3`
- `align_chunks = true`
- `transition_all_boundaries = true`
- `dataset_profile = main_continuous | stress`
- `same_source_sequence_prob`
- `cross_source_boundary_prob`
- `continuous_mode_weight`
- `hard_negative_mode_weight`
- `stress_mode_weight`
- `sampling_profile`
- `continuous_stats_dir`

To build `runtime_empirical` sampling stats from real continuous sequences:

```bash
python -m bio continuous-stats \
  --session_dir outputs/review_sessions/first_raw \
  --out_dir outputs/bio_out_v5/continuous_stats
```

Then point `synth_build.continuous_stats_dir` at that output directory. New
`stats.json` files now include the effective synth config plus an `acceptance`
block. For `main_continuous`, the realism gate now uses semantic seam metrics,
not transition-expanded ones. `python -m bio train` will refuse to start if the
synth dataset fails its realism gate unless `--allow_bad_synth_stats` is set.

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

The current training recipe can also enable an auxiliary `signness` head
(`O` vs `B/I`) plus startup-aware penalties. With the default config this is on
for new runs, but old checkpoints remain loadable because the legacy BIO head
and runtime API stay compatible.

New train-time knobs in `bio_default.json` include:

- `use_signness_head`
- `signness_head_dropout`
- `loss_lambda_signness`
- `loss_lambda_startup_nohand`
- `startup_visible_joint_threshold`
- `startup_visible_hand_frames`
- `balanced_lambda_startup_false_start`
- `balanced_lambda_startup_nohand_active`

The val/eval path now tracks startup-aware diagnostics such as:

- `startup_false_start_rate`
- `startup_segment_before_first_hand_rate`
- `startup_nohand_pred_B_rate`
- `startup_nohand_pred_active_rate`

`best_balanced` selection now penalizes startup false starts and can reject
checkpoints that frequently start a segment before the first visible hand.

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

## Runtime startup guard

The runtime decoder now supports a startup hand-presence guard that blocks
opening a new segment while no hands are stably visible. This is meant to
reduce the failure mode:

- no hands visible -> false `B/I`
- first visible hand frame -> sudden `O`

Relevant CLI flags on `bio infer-*`:

- `--require_hand_presence_to_start`
- `--min_visible_hand_frames_to_start`
- `--min_valid_hand_joints_to_start`
- `--allow_one_hand_to_start`
- `--require_both_hands_to_start`

The guard only blocks starting a new segment. It does not force-close an
already active segment, and it uses hand mask visibility, not pose or model
probabilities.

When the checkpoint includes the auxiliary signness head, runtime can also gate
segment starts/continuation with:

- `--use_signness_gate`
- `--signness_start_threshold`
- `--signness_continue_threshold`

Old checkpoints without the auxiliary head automatically fall back to the
BIO-only decoder path.

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
