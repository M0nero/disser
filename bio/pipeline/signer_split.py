from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _norm_split_name(raw: Any) -> str:
    val = str(raw or "").strip().lower()
    if val == "test":
        return "val"
    return val


def _as_bool(raw: str) -> Optional[bool]:
    val = str(raw or "").strip().lower()
    if not val:
        return None
    if val in {"1", "true", "t", "yes", "y", "train"}:
        return True
    if val in {"0", "false", "f", "no", "n", "val", "test"}:
        return False
    return None


def _sniff_dialect(csv_path: Path) -> csv.Dialect:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        class _D(csv.Dialect):
            delimiter = "\t" if ("\t" in sample and "," not in sample) else ","
            quotechar = '"'
            doublequote = True
            skipinitialspace = False
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL

        return _D()


def _resolve_split(row: Dict[str, Any]) -> str:
    raw_split = _norm_split_name(row.get("split"))
    if raw_split:
        return raw_split
    raw_train = row.get("train") or row.get("is_train")
    if raw_train is not None and str(raw_train).strip():
        bt = _as_bool(str(raw_train))
        if bt is None:
            return _norm_split_name(str(raw_train).strip())
        return "train" if bt else "val"
    raise RuntimeError("CSV row is missing split/train information.")


def _resolve_signer_id(row: Dict[str, Any]) -> str:
    signer_id = str(
        row.get("user_id")
        or row.get("signer_id")
        or row.get("user")
        or row.get("signer")
        or ""
    ).strip()
    if signer_id:
        return signer_id
    vid_raw = str(row.get("attachment_id") or "").strip()
    return Path(vid_raw).stem if vid_raw else ""


@dataclass(frozen=True)
class SplitRow:
    csv_index: int
    row_index: int
    signer_id: str
    label: str
    split: str
    row: Dict[str, Any]


@dataclass(frozen=True)
class CsvLoaded:
    path: Path
    dialect: csv.Dialect
    fieldnames: Tuple[str, ...]
    rows: Tuple[Dict[str, Any], ...]


def load_csv_rows(csv_path: Path) -> CsvLoaded:
    dialect = _sniff_dialect(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        fieldnames = tuple(str(x) for x in (reader.fieldnames or []))
        rows = tuple(dict(row) for row in reader)
    return CsvLoaded(path=csv_path, dialect=dialect, fieldnames=fieldnames, rows=rows)


def collect_split_rows(csv_paths: Sequence[Path]) -> Tuple[List[CsvLoaded], List[SplitRow]]:
    loaded: List[CsvLoaded] = []
    split_rows: List[SplitRow] = []
    for csv_index, csv_path in enumerate(csv_paths):
        loaded_csv = load_csv_rows(csv_path)
        loaded.append(loaded_csv)
        for row_index, row in enumerate(loaded_csv.rows):
            signer_id = _resolve_signer_id(row)
            if not signer_id:
                raise RuntimeError(f"Missing signer identity in {csv_path} row {row_index + 2}.")
            split_rows.append(
                SplitRow(
                    csv_index=csv_index,
                    row_index=row_index,
                    signer_id=signer_id,
                    label=str(row.get("text") or "").strip(),
                    split=_resolve_split(row),
                    row=row,
                )
            )
    return loaded, split_rows


def _build_targets(rows: Sequence[SplitRow], splits: Sequence[str]) -> Dict[str, Dict[str, float]]:
    total = max(1, len(rows))
    split_counts = Counter(r.split for r in rows)
    label_totals: Counter[str] = Counter(r.label for r in rows)
    label_split_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        label_split_counts[row.split][row.label] += 1
    targets: Dict[str, Dict[str, float]] = {}
    for split in splits:
        ratio = float(split_counts.get(split, 0)) / float(total)
        targets[split] = {label: float(label_totals.get(label, 0)) * ratio for label in label_totals}
        targets[split]["__total__"] = float(total) * ratio
    return targets


def assign_signers(rows: Sequence[SplitRow]) -> Dict[str, str]:
    split_names = sorted({r.split for r in rows})
    if not split_names:
        raise RuntimeError("No rows found for signer split assignment.")
    if "train" not in split_names:
        raise RuntimeError("Expected at least one train row in source CSV.")
    targets = _build_targets(rows, split_names)
    signer_rows: Dict[str, List[SplitRow]] = defaultdict(list)
    for row in rows:
        signer_rows[row.signer_id].append(row)

    assigned_label_counts: Dict[str, Counter[str]] = {split: Counter() for split in split_names}
    assigned_total_counts: Counter[str] = Counter()
    assigned_signer_counts: Counter[str] = Counter()
    assignments: Dict[str, str] = {}
    signer_split_hist: Dict[str, Counter[str]] = {signer_id: Counter(r.split for r in signer_group_rows) for signer_id, signer_group_rows in signer_rows.items()}
    target_signer_counts: Counter[str] = Counter(hist.most_common(1)[0][0] for hist in signer_split_hist.values())

    def _score_candidate(split: str, label_counts: Counter[str], total_count: int, preferred_split: str) -> Tuple[float, float, float, float, float, float, str]:
        signer_target = float(target_signer_counts.get(split, 0))
        current_signers = float(assigned_signer_counts.get(split, 0))
        after_signers = current_signers + 1.0
        signer_deficit_before = max(0.0, signer_target - current_signers)
        signer_overflow_penalty = max(0.0, after_signers - signer_target)
        signer_remaining_penalty = abs(signer_deficit_before - 1.0)

        total_target = float(targets[split]["__total__"])
        current_total = float(assigned_total_counts.get(split, 0))
        after_total = current_total + float(total_count)
        deficit_before = max(0.0, total_target - current_total)
        row_overflow_penalty = max(0.0, after_total - total_target)
        row_remaining_penalty = abs(deficit_before - float(total_count))

        for label, count in label_counts.items():
            target = float(targets[split].get(label, 0.0))
            current = float(assigned_label_counts[split].get(label, 0))
            row_remaining_penalty += 0.05 * abs((current + count) - target) / max(1.0, target)

        preference_penalty = 0.0 if split == preferred_split else 1.0
        return (
            signer_overflow_penalty,
            signer_remaining_penalty,
            row_overflow_penalty,
            row_remaining_penalty,
            preference_penalty,
            float(current_total),
            split,
        )

    for signer_id, signer_group_rows in signer_rows.items():
        split_hist = signer_split_hist[signer_id]
        if len(split_hist) != 1:
            continue
        label_counts = Counter(r.label for r in signer_group_rows)
        total_count = len(signer_group_rows)
        fixed_split = next(iter(split_hist))
        assignments[signer_id] = fixed_split
        assigned_label_counts[fixed_split].update(label_counts)
        assigned_total_counts[fixed_split] += total_count
        assigned_signer_counts[fixed_split] += 1

    signer_order = sorted(
        ((signer_id, signer_group_rows) for signer_id, signer_group_rows in signer_rows.items() if signer_id not in assignments),
        key=lambda item: (-len(item[1]), -max((Counter(r.label for r in item[1]).values()), default=0), item[0]),
    )
    for signer_id, signer_group_rows in signer_order:
        label_counts = Counter(r.label for r in signer_group_rows)
        total_count = len(signer_group_rows)
        preferred_split = signer_split_hist[signer_id].most_common(1)[0][0]
        best_split = min(
            (_score_candidate(split, label_counts, total_count, preferred_split) for split in split_names),
            key=lambda item: item,
        )[-1]
        assignments[signer_id] = best_split
        assigned_label_counts[best_split].update(label_counts)
        assigned_total_counts[best_split] += total_count
        assigned_signer_counts[best_split] += 1
    return assignments


def _overlap_by_signer(rows: Sequence[SplitRow]) -> Dict[str, List[str]]:
    signer_splits: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        signer_splits[row.signer_id].add(row.split)
    return {
        signer_id: sorted(splits)
        for signer_id, splits in signer_splits.items()
        if len(splits) > 1
    }


def write_rewritten_csvs(
    loaded_csvs: Sequence[CsvLoaded],
    assignments: Dict[str, str],
    out_dir: Path,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for loaded in loaded_csvs:
        out_path = out_dir / loaded.path.name
        fieldnames = list(loaded.fieldnames)
        for extra in ("split", "orig_split", "orig_train", "signer_id"):
            if extra not in fieldnames:
                fieldnames.append(extra)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, dialect=loaded.dialect)
            writer.writeheader()
            for row in loaded.rows:
                out_row = dict(row)
                orig_split = _resolve_split(row)
                signer_id = _resolve_signer_id(row)
                new_split = assignments[signer_id]
                out_row["orig_split"] = orig_split
                if "train" in fieldnames or "is_train" in fieldnames:
                    out_row["orig_train"] = str(row.get("train") or row.get("is_train") or "")
                out_row["split"] = new_split
                if "train" in out_row:
                    out_row["train"] = "1" if new_split == "train" else "0"
                if "is_train" in out_row:
                    out_row["is_train"] = "1" if new_split == "train" else "0"
                out_row["signer_id"] = signer_id
                writer.writerow(out_row)
        written.append(out_path)
    return written


def build_summary(rows: Sequence[SplitRow], assignments: Dict[str, str]) -> Dict[str, Any]:
    before_overlap = _overlap_by_signer(rows)
    split_row_counts: Counter[str] = Counter()
    split_signer_counts: Counter[str] = Counter(assignments.values())
    label_split_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        new_split = assignments[row.signer_id]
        split_row_counts[new_split] += 1
        label_split_counts[new_split][row.label] += 1
    return {
        "total_rows": int(len(rows)),
        "total_signers": int(len(assignments)),
        "source": "user_id_signer_split",
        "before_overlap_signer_count": int(len(before_overlap)),
        "before_overlap_examples": {k: before_overlap[k] for k in sorted(before_overlap)[:20]},
        "after_overlap_signer_count": 0,
        "split_row_counts": {k: int(v) for k, v in sorted(split_row_counts.items())},
        "split_signer_counts": {k: int(v) for k, v in sorted(split_signer_counts.items())},
        "split_label_counts": {
            split: {label: int(count) for label, count in sorted(counter.items())}
            for split, counter in sorted(label_split_counts.items())
        },
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rewrite Slovo CSVs into signer-disjoint splits based on user_id.")
    ap.add_argument("--csv", action="append", required=True, help="Input CSV/TSV path. Repeat for annotations.csv and annotations_no_event.csv.")
    ap.add_argument("--out_dir", required=True, help="Directory for rewritten signer-split CSV files.")
    ap.add_argument("--summary_name", default="signer_split_summary.json", help="Summary JSON filename written inside out_dir.")
    ap.add_argument("--mapping_name", default="signer_assignments.json", help="Signer->split JSON filename written inside out_dir.")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    csv_paths = [Path(x).resolve() for x in args.csv]
    out_dir = Path(args.out_dir).resolve()
    loaded_csvs, rows = collect_split_rows(csv_paths)
    assignments = assign_signers(rows)
    written = write_rewritten_csvs(loaded_csvs, assignments, out_dir)
    summary = build_summary(rows, assignments)
    (out_dir / args.summary_name).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / args.mapping_name).write_text(json.dumps(dict(sorted(assignments.items())), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[signer-split] wrote {len(written)} CSV files to {out_dir}")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
