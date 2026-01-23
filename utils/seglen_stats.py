import argparse
import pandas as pd
import csv
from pathlib import Path


def read_table(path: str, sep: str | None):
    if sep is None:
        sample = Path(path).read_bytes()[:4096].decode("utf-8", errors="ignore")
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
            sep = dialect.delimiter
        except Exception:
            sep = "\t" 
    try:
        return pd.read_csv(path, sep=sep, engine="python", encoding="utf-8-sig", on_bad_lines="warn")
    except TypeError:
        return pd.read_csv(path, sep=sep, engine="python", encoding="utf-8-sig", error_bad_lines=False, warn_bad_lines=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to annotations.csv")
    ap.add_argument("--skip_label", default="no_event", help="Label to skip (text column)")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--begin_col", default="begin")
    ap.add_argument("--end_col", default="end")
    ap.add_argument("--sep", default=None, help="Delimiter: ',' '\\t' ';' etc. Default: auto-detect")
    ap.add_argument("--end_exclusive", action="store_true",
                    help="If set, seg_len = end - begin (treat end as exclusive)")
    args = ap.parse_args()

    df = read_table(args.csv, args.sep)

    # Drop invalid rows early
    for c in [args.text_col, args.begin_col, args.end_col]:
        if c not in df.columns:
            raise SystemExit(f"Missing column '{c}' in CSV. Columns: {list(df.columns)}")

    # Skip no_event
    df2 = df[df[args.text_col].astype(str) != args.skip_label].copy()

    # Coerce numeric
    df2[args.begin_col] = pd.to_numeric(df2[args.begin_col], errors="coerce")
    df2[args.end_col] = pd.to_numeric(df2[args.end_col], errors="coerce")
    df2 = df2.dropna(subset=[args.begin_col, args.end_col])

    begin = df2[args.begin_col].astype(int)
    end = df2[args.end_col].astype(int)

    if args.end_exclusive:
        seg_len = end - begin
    else:
        seg_len = end - begin + 1

    # Filter weird rows
    df2["seg_len"] = seg_len
    df2 = df2[df2["seg_len"] > 0]

    def print_stats(name, x):
        x = x.astype(int)
        q = x.quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        print(f"\n[{name}] n={len(x)}")
        print(f"mean={x.mean():.2f}  median={x.median():.0f}  std={x.std():.2f}")
        print(f"min={x.min()}  max={x.max()}")
        print("p50={:.0f}  p75={:.0f}  p90={:.0f}  p95={:.0f}  p99={:.0f}".format(
            q[0.5], q[0.75], q[0.9], q[0.95], q[0.99]
        ))

    print_stats("ALL (no_event skipped)", df2["seg_len"])

    if "train" in df2.columns:
        # train can be bool or string
        tr = df2[df2["train"].astype(str).str.lower().isin(["true", "1", "yes"])]
        te = df2[~df2.index.isin(tr.index)]
        if len(tr) > 0:
            print_stats("TRAIN", tr["seg_len"])
        if len(te) > 0:
            print_stats("TEST/VAL", te["seg_len"])

    # crude max_frames suggestion
    p95 = df2["seg_len"].quantile(0.95)
    if p95 <= 32:
        rec = 32
    elif p95 <= 48:
        rec = 48
    else:
        rec = 64
    print(f"\nSuggested max_frames by p95 rule: {rec}")

if __name__ == "__main__":
    main()
