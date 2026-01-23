# BIO Pipeline

This module contains a 3-step pipeline for BIO segmentation:

1) prelabel: build per-segment BIO labels from skeletons + CSV
2) synth-build: generate synthetic continuous sequences (offline shards)
3) train: train the BIO tagger on the synthetic dataset

All outputs now live under `outputs/`, and datasets under `datasets/`.

## CLI

Entry point:

```
python -m bio <command> [args]
```

Commands:
- `prelabel`
- `synth-build`
- `train`
- `smoke-test`
- `ipn-make-manifest`
- `ipn-prelabel`

Run `python -m bio <command> -h` for details.

## Config Templates

Templates live in `bio/configs/`:
- `bio/configs/bio_default.json`
- `bio/configs/ipn_default.json`
- `bio/configs/bio_val.json` (val synth-build only; no IPN)

Each stage writes a `config.json` into its output directory for reproducibility.

Prelabel prefers `*_pp.json` when both raw and post-processed files exist. Override with `--no_prefer_pp` or set `"prefer_pp": false` in config.
Default SLOVO skeletons path is `datasets/skeletons/Slovo` (flat directory of per-video JSONs).

## Typical SLOVO Run (updated paths)

```
python -m bio prelabel \
  --config bio/configs/bio_default.json \
  --split train \
  --out outputs/bio_out/prelabels_slovo_train

python -m bio prelabel \
  --config bio/configs/bio_default.json \
  --split val \
  --out outputs/bio_out/prelabels_slovo_val

python -m bio prelabel \
  --config bio/configs/bio_default.json \
  --skeletons datasets/skeletons/Slovo \
  --csv datasets/data/annotations_no_event.csv \
  --split train \
  --out outputs/bio_out/prelabels_slovo_noev_train

python -m bio prelabel \
  --config bio/configs/bio_default.json \
  --skeletons datasets/skeletons/Slovo \
  --csv datasets/data/annotations_no_event.csv \
  --split val \
  --out outputs/bio_out/prelabels_slovo_noev_val

python -m bio synth-build \
  --config bio/configs/bio_default.json \
  --prelabel_dir outputs/bio_out/prelabels_slovo_train \
  --out_dir outputs/bio_out/synth_train

python -m bio synth-build \
  --config bio/configs/bio_val.json

Note: `bio/configs/bio_default.json` includes the no_event pool from:
`outputs/bio_out/prelabels_slovo_noev_train` and `outputs/ipnhand/prelabels_no_event`.
Use only SLOVO no_event for validation to avoid IPN split leakage.

python -m bio train \
  --config bio/configs/bio_default.json \
  --train_dir outputs/bio_out/synth_train \
  --val_dir outputs/bio_out/synth_val \
  --out_dir outputs/runs/bio_gru_v1
```

## IPN Hand (suggested layout)

Suggested in-repo structure:

```
datasets/ipnhand/
  annotations/
    Annot_TrainList.txt
    Annot_TestList.txt
  videos/                 # raw .avi files
  segments_npz/           # extracted per-segment npz (pts/mask)
outputs/ipnhand/
  manifests/
  prelabels_no_event/     # output of ipn-prelabel (all splits in one dir)
  quality_report/
  step1/
```

Example commands:

```
python -m bio ipn-make-manifest \
  --config bio/configs/ipn_default.json

python -m bio ipn-prelabel \
  --config bio/configs/ipn_default.json
```

If you imported an `ipnhand/` folder into the repo root, move its contents
into `datasets/ipnhand/` (data) and `outputs/ipnhand/` (artifacts) using the
layout above.

## Smoke Test

```
bash bio/tools/smoke_test.sh
```

This runs a tiny prelabel + synth-build sanity check.
