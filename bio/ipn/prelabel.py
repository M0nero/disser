#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create Step-1 compatible prelabels for IPN Hand O-background segments.

Input:
  - a directory with extracted skeleton npz files (one per segment)
    each must contain:
      pts: (T,V,3) float32
      mask: (T,V,1) float32
  - a manifest jsonl/csv from ipn/make_manifest.py (to keep split info)

Output (like bio/pipeline/prelabel.py):
  out_dir/
    npz/<seg_uid>.npz   (with keys expected by Step2)
    index.json
    summary.json

All outputs are labeled as no_event (BIO all O).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bio.core.config_utils import load_config_section, write_run_config


BIO_O = np.uint8(0)


@dataclass
class IndexRow:
    vid: str
    label_str: str
    path_to_npz: str
    T_total: int
    start_idx: int
    end_idx: int
    is_no_event: bool
    split: str
    dataset: str


def _read_manifest(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".csv":
        rows = []
        with path.open("r", encoding="utf-8", newline="") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                rows.append(dict(r))
        return rows
    raise ValueError(f"Unsupported manifest format: {path}")


def _is_missing(raw: object) -> bool:
    if raw is None:
        return True
    if isinstance(raw, str) and not raw.strip():
        return True
    return False


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, _ = pre.parse_known_args(argv)
    defaults = load_config_section(pre_args.config, "ipn_prelabel")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=pre_args.config, help="Path to config JSON (section: ipn_prelabel).")
    ap.add_argument("--manifest", type=str, default=defaults.get("manifest"), required=_is_missing(defaults.get("manifest")), help="ipn_d0x_manifest.jsonl or .csv")
    ap.add_argument("--skeleton_npz_dir", type=str, default=defaults.get("skeleton_npz_dir"), required=_is_missing(defaults.get("skeleton_npz_dir")), help="dir with extracted keypoints .npz per seg_uid")
    ap.add_argument("--out_dir", type=str, default=defaults.get("out_dir"), required=_is_missing(defaults.get("out_dir")), help="output Step-1 prelabel dir (for Step2 as extra no_event)")
    ap.add_argument("--expect_V", type=int, default=int(defaults.get("expect_V", 42)), help="expected number of keypoints V (default 42 for hands)")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    manifest = Path(args.manifest)
    sk_dir = Path(args.skeleton_npz_dir)
    out_dir = Path(args.out_dir)
    out_npz = out_dir / "npz"
    out_npz.mkdir(parents=True, exist_ok=True)
    write_run_config(out_dir, args, config_path=args.config, section="ipn_prelabel")

    rows = _read_manifest(manifest)

    index: List[IndexRow] = []
    missing = 0
    bad = 0

    for r in rows:
        seg_uid = str(r.get("seg_uid", "")).strip()
        if not seg_uid:
            continue
        split = str(r.get("split", "train")).strip() or "train"

        src = sk_dir / f"{seg_uid}.npz"
        if not src.exists():
            missing += 1
            continue

        try:
            d = np.load(src, allow_pickle=True)
            pts = d["pts"]
            mask = d["mask"]
        except Exception:
            bad += 1
            continue

        if pts.ndim != 3 or mask.ndim != 3:
            bad += 1
            continue
        T, V, C = pts.shape
        if C != 3 or mask.shape[0] != T or mask.shape[1] != V:
            bad += 1
            continue
        if int(args.expect_V) > 0 and V != int(args.expect_V):
            # hard fail because Step2 expects same V everywhere
            raise RuntimeError(f"V mismatch in {src}: got V={V}, expected {int(args.expect_V)}")

        bio = np.zeros((T,), dtype=np.uint8)  # all O
        label_str = "no_event"

        out_path_rel = Path("npz") / f"{seg_uid}.npz"
        out_path = out_dir / out_path_rel

        np.savez(
            out_path,
            pts=pts.astype(np.float32, copy=False),
            mask=mask.astype(np.float32, copy=False),
            bio=bio,
            label_str=np.asarray(label_str),
            is_no_event=np.asarray(True),
            start_idx=np.asarray(-1, dtype=np.int32),
            end_idx=np.asarray(-1, dtype=np.int32),
        )

        index.append(
            IndexRow(
                vid=seg_uid,
                label_str=label_str,
                path_to_npz=str(out_path_rel).replace("\\", "/"),
                T_total=int(T),
                start_idx=-1,
                end_idx=-1,
                is_no_event=True,
                split=split,
                dataset="ipn_hand",
            )
        )

    # write index.json
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.json").write_text(
        json.dumps([asdict(x) for x in index], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "num_rows_manifest": int(len(rows)),
        "num_index_written": int(len(index)),
        "missing_skeleton_npz": int(missing),
        "bad_npz": int(bad),
        "expect_V": int(args.expect_V),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] out_dir:", out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
