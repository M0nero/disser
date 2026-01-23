#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a clean manifest JSONL into Step1 CSV for bio/pipeline/prelabel.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def _split_norm(val: Optional[str]) -> str:
    s = str(val or "").strip().lower()
    return s or "unknown"


def _read_manifest_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _resolve_segment_path(segments_dir: Path, seg_uid: str, prefer_pp: bool) -> Path:
    pp = segments_dir / f"{seg_uid}_pp.json"
    raw = segments_dir / f"{seg_uid}.json"
    if prefer_pp and pp.exists():
        return pp
    if raw.exists():
        return raw
    if pp.exists():
        return pp
    return raw


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser("Clean manifest -> Step1 CSV")
    ap.add_argument("--clean_manifest", required=True, help="Path to clean manifest JSONL")
    ap.add_argument("--segments_dir", required=True, help="Directory with <seg_uid>.json segments")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--verify_files", action="store_true", default=True, help="Check that segment files exist")
    ap.add_argument("--no_verify_files", dest="verify_files", action="store_false")
    ap.add_argument("--drop_missing", action="store_true", default=True)
    ap.add_argument("--keep_missing", dest="drop_missing", action="store_false")
    ap.add_argument("--split", choices=["train", "val", "all"], default="all")
    ap.add_argument(
        "--prefer_pp",
        dest="prefer_pp",
        action="store_true",
        default=True,
        help="Prefer *_pp.json when both raw and post-processed files exist.",
    )
    ap.add_argument(
        "--no_prefer_pp",
        dest="prefer_pp",
        action="store_false",
        help="Use raw *.json even if *_pp.json exists.",
    )
    args = ap.parse_args(argv)

    manifest_path = Path(args.clean_manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    segments_dir = Path(args.segments_dir)
    if not segments_dir.exists():
        raise FileNotFoundError(segments_dir)

    rows = _read_manifest_rows(manifest_path)
    if not rows:
        raise RuntimeError("No rows found in clean manifest JSONL.")

    out_rows: List[Dict[str, object]] = []
    missing = 0
    kept_missing = 0
    counts = {"train": 0, "val": 0, "unknown": 0}

    for row in rows:
        seg_uid = str(row.get("seg_uid") or "").strip()
        if not seg_uid:
            continue
        split = _split_norm(row.get("split"))
        if args.split != "all" and split != args.split:
            continue

        fname = f"{seg_uid}.json"
        if args.verify_files:
            seg_path = _resolve_segment_path(segments_dir, seg_uid, args.prefer_pp)
            if not seg_path.exists():
                missing += 1
                if args.drop_missing:
                    continue
                kept_missing += 1

        out_rows.append({
            "attachment_id": fname,
            "text": "no_event",
            "split": split,
            "begin": 0,
            "end": 0,
        })
        if split in counts:
            counts[split] += 1
        else:
            counts["unknown"] += 1

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["attachment_id", "text", "split", "begin", "end"])
        w.writeheader()
        w.writerows(out_rows)

    print("[OK] Wrote CSV:", str(out_path))
    print("rows:", len(out_rows), "| train:", counts["train"], "| val:", counts["val"], "| unknown:", counts["unknown"])
    if args.verify_files:
        print("missing:", missing, "| kept_missing:", kept_missing)


if __name__ == "__main__":
    main()
