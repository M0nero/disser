#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SKELETONS_SIGN=${1:-"$ROOT/datasets/skeletons"}
CSV=${2:-"$ROOT/datasets/data/annotations.csv"}
WORKDIR=${3:-"$ROOT/outputs/out_smoke"}
SKELETONS_NOEV=${SKELETONS_NOEV:-"$ROOT/datasets/skeletons/no_event_old"}
PYTHON=${PYTHON:-python}

PYTHONPATH="$ROOT" "$PYTHON" -m bio smoke-test \
  --skeletons_sign "$SKELETONS_SIGN" \
  --skeletons_no_event "$SKELETONS_NOEV" \
  --csv "$CSV" \
  --workdir "$WORKDIR" \
  --split train \
  --num_synth_samples 128 \
  --seq_len 256 \
  --shard_size 64
