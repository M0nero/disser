"""
make_labels.py — Build labels.txt from CSV annotations

Matches mapping used in your GestureDataset:
- takes only rows where train == True
- collects unique `text`
- sorts them lexicographically (stable)
- writes one label per line to labels.txt (UTF-8)
"""
import argparse, csv, sys
from pathlib import Path

def parse_bool(x: str) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes", "y"}

def main():
    ap = argparse.ArgumentParser(description="Generate labels.txt from CSV annotations")
    ap.add_argument("csv", type=str, help="Path to annotations CSV (tab- or comma-separated)")
    ap.add_argument("--out", type=str, default="labels.txt", help="Output file path (default: labels.txt)")
    ap.add_argument("--delimiter", type=str, default="\t", help="CSV delimiter (default: TAB)" )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[labels] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    uniq = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f, delimiter=args.delimiter)
        need_cols = {"attachment_id", "text", "train"}
        miss = [c for c in need_cols if c not in rdr.fieldnames]
        if miss:
            print(f"[labels] Missing columns: {miss}. Found: {rdr.fieldnames}", file=sys.stderr)
            sys.exit(2)
        for row in rdr:
            if not parse_bool(row.get("train", "")):
                continue
            lbl = (row.get("text") or "").strip()
            if lbl:
                uniq.add(lbl)

    labels = sorted(uniq)
    out_path = Path(args.out)
    out_path.write_text("\n".join(labels), encoding="utf-8")
    print(f"[labels] Wrote {len(labels)} labels → {out_path}")
    if labels[:5]:
        print("[labels] First 5:", labels[:5])

if __name__ == "__main__":
    main()
