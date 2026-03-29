#!/usr/bin/env python3
"""
Extract sample IDs from videos.parquet and save one ID per line.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pyarrow.parquet as pq


def collect_ids(data) -> List[str]:
    ids: List[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "sample_id" in item:
                ids.append(str(item["sample_id"]))
            elif isinstance(item, dict) and "slug" in item:
                ids.append(str(item["slug"]))
            else:
                ids.append(str(item))
    elif isinstance(data, dict):
        # fallback: treat keys as ids
        ids.extend(str(k) for k in data.keys())
    return ids


def main():
    ap = argparse.ArgumentParser(description="Extract sample IDs from videos.parquet")
    ap.add_argument(
        "--videos-parquet",
        type=Path,
        default=Path("datasets/skeletons/videos.parquet"),
        help="Path to videos.parquet",
    )
    ap.add_argument("--out", type=Path, default=None, help="Path to save IDs (one per line)")
    args = ap.parse_args()

    data = pq.read_table(args.videos_parquet).to_pylist()
    ids = collect_ids(data)

    text = "\n".join(ids)
    print(f"Found {len(ids)} IDs")
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"Saved to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
