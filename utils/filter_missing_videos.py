#!/usr/bin/env python3
"""
Filter out video IDs that already exist in a manifest.

Example:
  python utils/filter_missing_videos.py \
    --videos outputs/runs/agcn_latest_balanced/low_f1_videos.txt \
    --manifest_ids outputs/artifacts/manifest_ids.txt \
    --out outputs/runs/agcn_latest_balanced/low_f1_not_in_manifest.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description="Remove IDs present in manifest from a list.")
    ap.add_argument("--videos", required=True, help="Path to input video-id list (one per line).")
    ap.add_argument("--manifest_ids", required=True, help="Path to manifest id list (one per line).")
    ap.add_argument("--out", required=False, help="Where to save filtered IDs (default: stdout only).")
    args = ap.parse_args()

    vids = read_ids(Path(args.videos))
    manifest = set(read_ids(Path(args.manifest_ids)))

    filtered = [v for v in vids if v not in manifest]

    text = "\n".join(filtered)
    print(f"Input videos: {len(vids)} | Manifest IDs: {len(manifest)} | Kept: {len(filtered)}")
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Saved to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
