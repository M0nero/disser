#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove per-video JSON files whose IDs are labeled with specific classes in annotations CSV.
- IDs are matched against file stems (filename without extension).
- Defaults assume your CSV has columns: 'attachment_id' (ID) and 'text' (label), tab-separated.
- Safe by default: dry-run mode. Use --yes to actually delete (or --move-to to move).

Examples:
  python remove_by_labels.py --json-dir datasets/skeletons --annotations datasets/data/annotations.csv --labels no_event --yes
  python remove_by_labels.py --json-dir datasets/skeletons --annotations datasets/data/annotations.csv --labels no_event --move-to datasets/trash
"""
import csv
import sys
import argparse
from pathlib import Path
from typing import List, Set, Optional
import shutil

def _detect_delim(header_line: str) -> str:
    cands = ["\t", ";", ",", "|"]
    counts = {d: header_line.count(d) for d in cands}
    return max(counts, key=counts.get) if any(counts.values()) else ","


def _stem_no_pp(path: Path) -> str:
    stem = path.stem
    return stem[:-3] if stem.endswith("_pp") else stem

def load_ids_by_labels(csv_path: Path,
                       labels: List[str],
                       id_col_candidates = ("attachment_id","id","video","vid","filename","file","name"),
                       label_col_candidates = ("text","label","class","tag"),
                       delimiter: Optional[str] = None) -> Set[str]:
    labels = [s.strip().lower() for s in labels if s and s.strip()]
    ids: Set[str] = set()
    if not csv_path or not csv_path.exists():
        return ids

    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline()
        delim = delimiter or _detect_delim(header)
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delim)
        cols = [c.lower() for c in (reader.fieldnames or [])]

        id_col = next((c for c in id_col_candidates if c in cols), None)
        lab_col = next((c for c in label_col_candidates if c in cols), None)
        if not id_col or not lab_col:
            print(f"[ERROR] Could not find id/label columns in CSV. Have: {cols}.", file=sys.stderr)
            return ids

        for row in reader:
            try:
                lbl = str(row[lab_col]).strip().lower()
                if lbl in labels:
                    raw = str(row[id_col]).strip()
                    if raw:
                        ids.add(Path(raw).stem)
            except Exception:
                continue
    return ids

def main(argv=None):
    ap = argparse.ArgumentParser("Remove per-video JSONs by labels from annotations CSV")
    ap.add_argument("--json-dir", required=True, help="Directory with per-video JSON files")
    ap.add_argument("--annotations", required=True, help="annotations.csv path")
    ap.add_argument("--labels", required=True, help="Comma-separated labels to remove (e.g., no_event)")
    ap.add_argument("--pattern", default="*.json", help="Glob pattern for JSON files (default: *.json)")
    ap.add_argument("--yes", action="store_true", help="Actually delete/move files (otherwise dry-run)")
    ap.add_argument("--move-to", type=str, default="", help="Instead of delete, move files to this directory")
    ap.add_argument("--delimiter", type=str, default="", help="CSV delimiter override (auto-detect if empty)")
    ap.add_argument("--report", type=str, default="removed_ids.txt", help="Path to write list of removed IDs")
    args = ap.parse_args(argv)

    json_dir = Path(args.json_dir)
    if not json_dir.exists() or not json_dir.is_dir():
        print(f"[ERR] JSON dir not found: {json_dir}", file=sys.stderr); sys.exit(2)

    csv_path = Path(args.annotations)
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    delim = args.delimiter or None

    to_remove = load_ids_by_labels(csv_path, labels, delimiter=delim)
    if not to_remove:
        print("[INFO] No IDs found for provided labels. Nothing to do."); sys.exit(0)

    json_files = list(json_dir.glob(args.pattern))
    matched = [p for p in json_files if _stem_no_pp(p) in to_remove]

    print(f"[INFO] Found {len(to_remove)} IDs in CSV for labels={labels}.")
    print(f"[INFO] Matched {len(matched)} JSON files in {json_dir} with pattern {args.pattern}.")
    if not matched:
        print("[INFO] No files to process."); sys.exit(0)

    moveroot = None
    if args.move_to:
        moveroot = Path(args.move_to)
        moveroot.mkdir(parents=True, exist_ok=True)

    action = "MOVE" if moveroot else "DELETE"
    print(f"[PLAN] {action} {len(matched)} files.")
    for p in matched[:10]:
        print("  -", p.name)
    if len(matched) > 10:
        print(f"  ... and {len(matched)-10} more")

    removed_ids = []
    if args.yes:
        for p in matched:
            try:
                if moveroot:
                    target = moveroot / p.name
                    shutil.move(str(p), str(target))
                else:
                    p.unlink(missing_ok=True)
                removed_ids.append(_stem_no_pp(p))
            except Exception as e:
                print(f"[WARN] Failed {action.lower()} {p.name}: {e}", file=sys.stderr)
        if removed_ids:
            Path(args.report).write_text("\n".join(sorted(removed_ids)), encoding="utf-8")
            print(f"[OK] {action} done. Wrote report with {len(removed_ids)} IDs to {args.report}")
        else:
            print("[INFO] Nothing was removed.")
    else:
        print("[DRY-RUN] Use --yes to execute.")

if __name__ == "__main__":
    main()
