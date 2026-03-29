#!/usr/bin/env python3
from __future__ import annotations

import sys


def main() -> int:
    print(
        "scripts/rebuild_manifest_eval.py is obsolete.\n"
        "The extractor no longer writes per-video JSON, manifest.json, or eval_report.json.\n"
        "Use landmarks.zarr together with videos.parquet, frames.parquet, and runs.parquet.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
