#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python}"

IPN_VIDEOS="${IPN_VIDEOS:-$ROOT/datasets/ipnhand/videos}"
IPN_MANIFEST="${IPN_MANIFEST:-$ROOT/outputs/ipnhand/manifests/ipn_d0x_manifest.jsonl}"
IPN_SEGMENTS="${IPN_SEGMENTS:-$ROOT/datasets/skeletons/ipnhand}"
IPN_QUALITY_DIR="${IPN_QUALITY_DIR:-$ROOT/outputs/ipnhand/quality_report}"
IPN_CLEAN_MANIFEST="${IPN_CLEAN_MANIFEST:-$ROOT/outputs/ipnhand/manifests/ipn_d0x_manifest_clean.jsonl}"
IPN_PRELABEL_TRAIN_DIR="${IPN_PRELABEL_TRAIN_DIR:-$ROOT/outputs/ipnhand/prelabels_train}"
IPN_PRELABEL_VAL_DIR="${IPN_PRELABEL_VAL_DIR:-$ROOT/outputs/ipnhand/prelabels_val}"
JOBS="${JOBS:-8}"
KEEP_BUCKET="${KEEP_BUCKET:-OK}"
MIN_QUALITY_SCORE="${MIN_QUALITY_SCORE:-0.35}"
EXTRACT_ARGS="${EXTRACT_ARGS:---image-coords --postprocess --skip-existing}"

PYTHONPATH="$ROOT" "$PYTHON" -m bio ipn-make-manifest \
  --config "$ROOT/bio/configs/ipn_default.json"

# shellcheck disable=SC2086
PYTHONPATH="$ROOT" "$PYTHON" "$ROOT/scripts/extract_keypoints.py" \
  --in-dir "$IPN_VIDEOS" \
  --segments-manifest "$IPN_MANIFEST" \
  --out-dir "$IPN_SEGMENTS" \
  --eval-report "$ROOT/outputs/ipnhand/eval_report.json" \
  --jobs "$JOBS" \
  $EXTRACT_ARGS

PYTHONPATH="$ROOT" "$PYTHON" "$ROOT/scripts/ipn_json_quality_report.py" \
  --ipn_dir "$IPN_SEGMENTS" \
  --manifest "$IPN_MANIFEST" \
  --out "$IPN_QUALITY_DIR" \
  --keep_bucket "$KEEP_BUCKET" \
  --min_quality_score "$MIN_QUALITY_SCORE" \
  --write_clean_manifest "$IPN_CLEAN_MANIFEST"

PYTHONPATH="$ROOT" "$PYTHON" -m bio ipn-prelabel \
  --config "$ROOT/bio/configs/ipn_default.json" \
  --manifest "$IPN_CLEAN_MANIFEST" \
  --segments_dir "$IPN_SEGMENTS" \
  --split train \
  --out_dir "$IPN_PRELABEL_TRAIN_DIR"

PYTHONPATH="$ROOT" "$PYTHON" -m bio ipn-prelabel \
  --config "$ROOT/bio/configs/ipn_val.json" \
  --manifest "$IPN_CLEAN_MANIFEST" \
  --segments_dir "$IPN_SEGMENTS" \
  --split val \
  --out_dir "$IPN_PRELABEL_VAL_DIR"
