#!/usr/bin/env python3
"""
List classes with low F1 from a classification_report JSON and show their videos.

Inputs:
  --report    Path to report_epXXX.json (sklearn classification_report output).
  --label2idx Path to label2idx.json (label -> idx mapping).
  --csv       Path to annotations CSV/TSV (columns: attachment_id, text, train/val or split).
  --f1_thr    Threshold: keep classes with F1 <= this value (default: 0.0).

Output: human‑readable summary to stdout.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_label_map(path: Path) -> Dict[int, str]:
    """Invert label2idx.json (label -> idx) to idx -> label."""
    data = json.load(path.open("r", encoding="utf-8"))
    out = {}
    for label, idx in data.items():
        out[int(idx)] = label
    return out


def parse_annotations(csv_path: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Read annotations, returning {label: {"train": [...], "val": [...]}} with video ids (stem).
    Mirrors the split logic from dataset_multistream: prefers 'split', otherwise 'train'/is_train.
    """
    by_label: Dict[str, Dict[str, List[str]]] = {}

    def _as_bool(s: str):
        if s is None:
            return None
        v = s.strip().lower()
        if v in ("true", "1", "yes", "y"):
            return True
        if v in ("false", "0", "no", "n"):
            return False
        return None

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        except csv.Error:
            class _D(csv.Dialect):
                delimiter = "\t" if ("\t" in sample and "," not in sample) else ","
                quotechar = '"'
                doublequote = True
                skipinitialspace = False
                lineterminator = "\n"
                quoting = csv.QUOTE_MINIMAL

            dialect = _D()

        rdr = csv.DictReader(f, dialect=dialect)
        for row in rdr:
            label = (row.get("text") or "").strip()
            if not label or label.lower() == "no_event":
                continue

            raw_split = (row.get("split") or "").strip()
            raw_train = row.get("train") or row.get("is_train")

            if raw_split:
                split = "train" if raw_split.lower() == "train" else "val"
            elif raw_train is not None:
                bt = _as_bool(str(raw_train))
                if bt is None:
                    split = "train" if str(raw_train).strip().lower() == "train" else "val"
                else:
                    split = "train" if bt else "val"
            else:
                # default: treat as train if not specified
                split = "train"

            vid = (row.get("attachment_id") or "").strip()
            vid = Path(vid).stem
            if not vid:
                continue

            bucket = by_label.setdefault(label, {"train": [], "val": []})
            bucket[split].append(vid)

    return by_label


def collect_low_f1(
    report_path: Path, idx2label: Dict[int, str], f1_thr: float
) -> List[Tuple[int, str, float, float]]:
    """Return list of (idx, label, f1, support) with F1 <= f1_thr."""
    rep = json.load(report_path.open("r", encoding="utf-8"))
    out = []
    for k, v in rep.items():
        if k in ("accuracy", "macro avg", "weighted avg"):
            continue
        if not isinstance(v, dict):
            continue
        f1 = float(v.get("f1-score", 0.0))
        supp = float(v.get("support", 0.0))
        try:
            cls_idx = int(k)
        except ValueError:
            # keys in report might be labels; map back if they exist in idx2label inversely
            label = k
            cls_idx = next((i for i, lbl in idx2label.items() if lbl == label), -1)
        label = idx2label.get(cls_idx, str(k))
        if f1 <= f1_thr:
            out.append((cls_idx, label, f1, supp))
    out.sort(key=lambda x: x[0])
    return out


def main():
    ap = argparse.ArgumentParser(description="List classes with low F1 and their videos.")
    ap.add_argument("--report", required=True, help="Path to report_epXXX.json")
    ap.add_argument("--label2idx", required=True, help="Path to label2idx.json (label -> idx)")
    ap.add_argument("--csv", required=True, help="annotations CSV/TSV")
    ap.add_argument("--f1_thr", type=float, default=0.0, help="Threshold for F1 (<=)")
    ap.add_argument("--out", type=str, default=None, help="Optional path to save the list (text)")
    ap.add_argument(
        "--videos_only",
        action="store_true",
        help="If set, only save unique video IDs (one per line) for classes under threshold.",
    )
    args = ap.parse_args()

    idx2label = load_label_map(Path(args.label2idx))
    low = collect_low_f1(Path(args.report), idx2label, args.f1_thr)
    ann = parse_annotations(Path(args.csv))

    # Collect outputs
    if args.videos_only:
        vids_all = []
        for _, label, _, _ in low:
            vids = ann.get(label, {"train": [], "val": []})
            vids_all.extend(vids.get("train", []))
            vids_all.extend(vids.get("val", []))
        # unique preserve order
        seen = set()
        vids_unique = []
        for v in vids_all:
            if v not in seen:
                vids_unique.append(v)
                seen.add(v)
        text = "\n".join(vids_unique)
        print(f"Collected {len(vids_unique)} unique videos for classes with F1 <= {args.f1_thr}")
        print(text)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"\nSaved to {args.out}")
    else:
        lines = []
        lines.append(f"Report: {args.report}")
        lines.append(f"Label map: {args.label2idx}")
        lines.append(f"Annotations: {args.csv}")
        lines.append(f"Classes with F1 <= {args.f1_thr}: {len(low)}\n")
        for cls_idx, label, f1, supp in low:
            vids = ann.get(label, {"train": [], "val": []})
            train_v = vids.get("train", [])
            val_v = vids.get("val", [])
            lines.append(f"[{cls_idx:04d}] {label} | F1={f1:.4f} | supp={supp}")
            lines.append(f"  train ({len(train_v)}): {', '.join(train_v) if train_v else '-'}")
            lines.append(f"  val   ({len(val_v)}): {', '.join(val_v) if val_v else '-'}")
            lines.append("")

        text = "\n".join(lines)
        print(text)

        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
