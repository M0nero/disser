#!/usr/bin/env python3
"""
Extract video IDs from datasets/skeletons/manifest.json and save one ID per line.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List


def collect_ids(data) -> List[str]:
    ids: List[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "id" in item:
                ids.append(str(item["id"]))
            else:
                ids.append(str(item))
    elif isinstance(data, dict):
        # fallback: treat keys as ids
        ids.extend(str(k) for k in data.keys())
    return ids


def main():
    ap = argparse.ArgumentParser(description="Extract IDs from manifest.json")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/skeletons/manifest.json"),
        help="Path to manifest.json",
    )
    ap.add_argument("--out", type=Path, default=None, help="Path to save IDs (one per line)")
    args = ap.parse_args()

    data = json.load(args.manifest.open("r", encoding="utf-8"))
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
