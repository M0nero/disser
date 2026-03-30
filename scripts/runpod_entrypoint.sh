#!/usr/bin/env bash
set -euo pipefail

STDERR_LOG_PATH="${STDERR_LOG_PATH:-}"
if [[ -n "${STDERR_LOG_PATH}" ]]; then
  mkdir -p "$(dirname "${STDERR_LOG_PATH}")"
  exec 2>>"${STDERR_LOG_PATH}"
fi

cd /app
python3 scripts/runpod_extract.py pod-run "$@"
