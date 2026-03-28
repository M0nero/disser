# BIO SLOVO + IPNHand Pipeline

This is the explicit end-to-end recipe for the setup used in this repo:

- SLOVO provides sign clips plus optional SLOVO `no_event`
- IPNHand provides extra `O/no_event` background chunks (`D0X`)
- Step2 mixes SLOVO sign segments with native SLOVO `O` first, then optional IPNHand negatives
- `bio` learns only `O/B/I` boundaries, not lexical sign classes

Canonical rebuild target:

- `outputs/bio_out_v2`

Old `outputs/bio_out` snapshots are unsupported and should not be used as the
reference build for new experiments.

`msagcn` remains the isolated-sign classifier. `bio` is the boundary detector in
front of it.

## How O Is Built

The synthetic BIO dataset needs many negative `O` frames. SLOVO alone has some
`no_event`, and the current code also harvests the `O` tails around each sign
span from ordinary SLOVO sign clips. The intended priority is:

1. native SLOVO `O`
2. explicit SLOVO `no_event`
3. external IPNHand `D0X`

Concretely:

- Train:
  - `O` tails from `outputs/bio_out_v2/prelabels_slovo_train`
  - `outputs/bio_out_v2/prelabels_slovo_noev_train`
  - `outputs/ipnhand/prelabels_train`
- Val:
  - `O` tails from `outputs/bio_out_v2/prelabels_slovo_val`
  - `outputs/bio_out_v2/prelabels_slovo_noev_val`
  - `outputs/ipnhand/prelabels_val`

`bio/configs/bio_default.json` and `bio/configs/bio_val.json` now encode this
priority explicitly: native SLOVO `O` is sampled before external IPN negatives.

## Stage 0. Skeleton Extraction

### 0a. SLOVO videos -> skeleton JSON

If SLOVO skeletons are not already prepared:

```bash
python scripts/extract_keypoints.py \
  --in-dir datasets/SLOVO/all \
  --out-dir datasets/skeletons/Slovo \
  --eval-report outputs/Slovo/eval_report.json \
  --image-coords \
  --postprocess \
  --jobs 8 \
  --skip-existing
```

The detailed extractor options live in `scripts/README.md`.

### 0b. IPNHand `D0X` manifest

Build a manifest of IPNHand background intervals:

```bash
python -m bio ipn-make-manifest \
  --config bio/configs/ipn_default.json
```

This creates:

- `outputs/ipnhand/manifests/ipn_d0x_manifest.jsonl`
- `outputs/ipnhand/manifests/ipn_d0x_manifest.csv`
- `outputs/ipnhand/manifests/ipn_d0x_manifest.stats.json`

By default the manifest keeps label `D0X`, chunks intervals into 96-frame
windows, and uses the IPN train/test lists as BIO train/val splits.

### 0c. IPNHand manifest segments -> skeleton JSON

Extract only the manifest segments from the raw IPNHand videos:

```bash
python scripts/extract_keypoints.py \
  --in-dir datasets/ipnhand/videos \
  --segments-manifest outputs/ipnhand/manifests/ipn_d0x_manifest.jsonl \
  --out-dir datasets/skeletons/ipnhand \
  --eval-report outputs/ipnhand/eval_report.json \
  --image-coords \
  --postprocess \
  --jobs 8 \
  --skip-existing
```

This writes one JSON per `seg_uid` into `datasets/skeletons/ipnhand/`.

### 0d. Optional quality filtering for IPNHand segments

The repo has a quality filter for these segment JSONs:

```bash
python scripts/ipn_json_quality_report.py \
  --ipn_dir datasets/skeletons/ipnhand \
  --manifest outputs/ipnhand/manifests/ipn_d0x_manifest.jsonl \
  --out outputs/ipnhand/quality_report \
  --write_clean_manifest outputs/ipnhand/manifests/ipn_d0x_manifest_clean.jsonl
```

Recommended outcome:

- use `outputs/ipnhand/manifests/ipn_d0x_manifest_clean.jsonl` for the next step
- if you want no filtering, keep using the original manifest JSONL

## Canonical One-Shot Rebuild

```bash
python -m bio build-dataset \
  --out_root outputs/bio_out_v2
```

This command runs:

1. overlap audit for SLOVO sign / SLOVO no_event
2. Step1 SLOVO sign prelabels
3. Step1 SLOVO `no_event` pools
4. Step2 main `main_continuous` synth train/val rebuild with trimmed Slovo defaults
5. optional separate stress dataset when `--emit_stress_dataset` is enabled

Key runtime-first v3 defaults:

- Step1 `trimmed_mode = true`
- Step1 `source_group = user_id` when the CSV provides signer identity
- If the original Slovo CSV is not signer-disjoint, first run:
  `python -m bio signer-split --csv datasets/data/annotations.csv --csv datasets/data/annotations_no_event.csv --out_dir datasets/data/slovo_signer_split`
- `min_tail_len = 4`
- `primary_noev_prob = 0.90`
- `source_sampling = uniform_source`
- `sign_sampling = uniform_label_source`
- `stitch_noev_chunks = true`
- `dataset_profile = main_continuous`
- `same_source_sequence_prob = 1.0`
- `cross_source_boundary_prob = 0.0`
- `sampling_profile = prelabel_empirical`
- synthetic mix defaults: `continuous=85%`, `hard_negative=15%`, `stress=0%` in the main dataset

Artifacts written by the orchestration step:

- `outputs/bio_out_v2/overlap_report.json`
- `outputs/bio_out_v2/build_summary.json`
- `outputs/bio_out_v2/dataset_manifest.json`
- `outputs/bio_out_v2/prelabels_*`
- `outputs/bio_out_v2/synth_train`
- `outputs/bio_out_v2/synth_val`

The rebuild is staged into temporary directories and promoted only after
overlap checks and artifact validation pass.

The default rebuild is now Slovo-only. No external IPN no-event pool is used in
the main train set.

## Stage 1. Step1 Prelabels

### 1a. SLOVO sign clips

```bash
python -m bio prelabel \
  --config bio/configs/bio_default.json \
  --split train \
  --out outputs/bio_out_v2/prelabels_slovo_train

python -m bio prelabel \
  --config bio/configs/bio_default.json \
  --split val \
  --out outputs/bio_out_v2/prelabels_slovo_val
```

### 1b. SLOVO no_event clips

```bash
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
```

### 1c. IPNHand segments -> Step1-compatible no_event prelabels

`bio ipn-prelabel` now accepts either:

- per-segment `.npz`
- or JSON / `_pp.json` files from `scripts/extract_keypoints.py --segments-manifest`

Recommended:

```bash
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

If you skipped quality filtering, replace `ipn_d0x_manifest_clean.jsonl` with the
original manifest JSONL.

If you already restored ready-made IPN pools, these directories are equivalent
and already usable as Step1 input:

- `datasets/ipnhand/ipn_prelabels_train`
- `datasets/ipnhand/ipn_prelabels_val`

The repo now treats the canonical artifact copies as:

- `outputs/ipnhand/prelabels_train`
- `outputs/ipnhand/prelabels_val`

## Stage 2. Synthetic Continuous BIO Dataset

### Train synth

```bash
python -m bio synth-build \
  --config bio/configs/bio_default.json \
  --prelabel_dir outputs/bio_out_v2/prelabels_slovo_train \
  --preferred_noev_prelabel_dir outputs/bio_out_v2/prelabels_slovo_noev_train \
  --extra_noev_prelabel_dir outputs/ipnhand/prelabels_train \
  --out_dir outputs/bio_out_v2/synth_train
```

This uses:

- sign pool from `outputs/bio_out_v2/prelabels_slovo_train`
- primary `O/no_event` pool from:
  - `O` tails harvested from ordinary sign clips in `outputs/bio_out_v2/prelabels_slovo_train`
  - `outputs/bio_out_v2/prelabels_slovo_noev_train`
- secondary external `O/no_event` pool from:
  - `outputs/ipnhand/prelabels_train`

Step2 defaults are now intentionally biased away from "sign starts at frame 0":

- `pad_mode = both_no_event`
- `leading_noev_prob = 0.65`
- `leading_noev_min = 12`
- `leading_noev_max = 96`
- `all_noev_prob = 0.15`

This means train/val synth now contains:

- explicit leading `O` prefixes before the first sign
- fully negative `all-O` sequences
- the usual inter-sign gaps and end padding

`stats.json` / `summary.json` now exposes startup-negative diagnostics:

- fraction of samples with a leading `O` prefix
- fraction of `all-O` samples
- `first_B_frame` stats/distribution
- how often `first_B_frame == 0`

Step2 now also assembles sequences in raw canonical coordinate space:

- Step1 contributes `pts_raw` runtime-like skeletons
- the next chunk is aligned to the previous tail by wrist-center and scale
- transitions can be inserted for all boundary types
- only after assembly/corruption the shared `canonical_hands42_v3`
  preprocessing is applied

Step2 also applies post-assembly runtime-like corruption so synth better matches
real detector failures:

- single-hand dropout spans
- both-hands dropout spans
- short mask flicker spans
- visible-joint coordinate jitter

`stats.json` / `summary.json` also tracks:

- fraction of samples with hand corruption
- startup-prefix frames that remain fully no-hand after corruption
- longest no-hand span stats
- corruption counts by type
- seam realism:
  - boundary/internal center jump ratio
  - boundary/internal scale jump ratio
  - wrist and valid-joint deltas at seams
  - same-source vs cross-source boundary breakdown

### Val synth

```bash
python -m bio synth-build \
  --config bio/configs/bio_val.json \
  --prelabel_dir outputs/bio_out_v2/prelabels_slovo_val \
  --preferred_noev_prelabel_dir outputs/bio_out_v2/prelabels_slovo_noev_val \
  --extra_noev_prelabel_dir outputs/ipnhand/prelabels_val \
  --out_dir outputs/bio_out_v2/synth_val
```

`preferred_noev_prelabel_dir` and `extra_noev_prelabel_dir` must be canonical
`O`-only Step1 pools. Mixed directories that contain sign clips are rejected
during build.

This uses:

- sign pool from `outputs/bio_out_v2/prelabels_slovo_val`
- primary `O/no_event` pool from:
  - `O` tails harvested from ordinary sign clips in `outputs/bio_out_v2/prelabels_slovo_val`
  - `outputs/bio_out_v2/prelabels_slovo_noev_val`
- secondary external `O/no_event` pool from:
  - `outputs/ipnhand/prelabels_val`

## Stage 3. BIO Boundary Tagger Training

```bash
python -m bio train \
  --config bio/configs/bio_default.json \
  --train_dir outputs/bio_out_v2/synth_train \
  --val_dir outputs/bio_out_v2/synth_val \
  --out_dir outputs/runs/bio_gru_v2 \
  --tensorboard \
  --logdir runs \
  --run_name bio_gru_v2
```

Current train defaults also enable an auxiliary `signness` head (`O` vs `B/I`)
plus startup-aware losses/selection. Important config knobs:

- `use_signness_head`
- `signness_head_dropout`
- `loss_lambda_signness`
- `loss_lambda_startup_nohand`
- `startup_visible_joint_threshold`
- `startup_visible_hand_frames`
- `balanced_lambda_startup_false_start`
- `balanced_lambda_startup_nohand_active`

The val/eval path now logs startup diagnostics such as:

- `startup_false_start_rate`
- `startup_segment_before_first_hand_rate`
- `startup_nohand_pred_B_rate`
- `startup_nohand_pred_active_rate`

## Runtime Startup Guard

The runtime decoder now supports a hand-presence startup guard for inference.
Use it when long silent / no-hands prefixes cause false `B/I` at frame 0.

Relevant `bio infer-*` options:

- `--require_hand_presence_to_start`
- `--min_visible_hand_frames_to_start`
- `--min_valid_hand_joints_to_start`
- `--allow_one_hand_to_start`
- `--require_both_hands_to_start`

The guard:

- blocks only the start of a new segment
- does not interfere with an already open segment
- uses hand mask validity only
- does not depend on pose

When the checkpoint includes the auxiliary signness head, runtime can also gate
starts/continuation with:

- `--use_signness_gate`
- `--signness_start_threshold`
- `--signness_continue_threshold`

## Minimal Artifact Layout

```text
datasets/
  skeletons/
    Slovo/
    ipnhand/
  ipnhand/
    annotations/
    videos/

outputs/
  bio_out_v2/
    prelabels_slovo_train/
    prelabels_slovo_val/
    prelabels_slovo_noev_train/
    prelabels_slovo_noev_val/
    synth_train/
    synth_val/
  ipnhand/
    manifests/
    quality_report/
    prelabels_train/
    prelabels_val/
```

## Important Notes

- `bio` predicts only `O/B/I`. Sign labels survive in metadata, but training
  targets are boundary tags.
- In the current code, ordinary sign clips contribute twice:
  once as trimmed `B/I` spans, and again as `O`-prefix/`O`-suffix tails in the
  primary no_event pool.
- `bio ipn-prelabel` now writes flat Step1-compatible outputs:
  `.npz` files in the directory root plus `index.json`, `index.csv`, and
  `summary.json` and `dataset_manifest.json`.
- `bio prelabel` now writes canonical Step1 index rows with:
  `vid`, `label_str`, `path_to_npz`, `T_total`, `start_idx`, `end_idx`,
  `is_no_event`, `split`, `dataset`, `source_group`
- `bio prelabel` also writes `csv_audit.json` and `rejected_rows.json` so bad
  CSV rows are visible instead of being silently swallowed.
- Canonical stages now also write `dataset_manifest.json`, which stores config
  SHA, git SHA, input paths, counts, and runtime/build metadata.
- The current code writes POSIX shard paths going forward, and the loader also
  normalizes old Windows-style `shards\\...` paths. This matters if synth shards
  were generated on Windows and later consumed from WSL/Linux.
- `bio train` now saves `best_boundary.pt` and `best_balanced.pt`. The balanced
  checkpoint applies guardrails on predicted `B` ratio and segment-length ratio.
  Boundary and balanced checkpoint selection both use the validation threshold
  sweep, not only raw argmax logits.
- `bio train` also writes `history.json`, a full-state `last.pt`, compact
  resume snapshots for `epoch_XXXX.pt` / `best_*.pt`, and optional `analysis/`
  artifacts with threshold sweeps every epoch plus prediction/error tables on
  a configurable cadence.
- Periodic snapshots are epoch-boundary artifacts controlled by
  `--save_every_epochs`. The legacy `--save_every` alias remains only for
  migration.
- If you only want the IPNHand negative-pool build, use:

```bash
bash bio/tools/build_ipn_noevent_pool.sh
```
