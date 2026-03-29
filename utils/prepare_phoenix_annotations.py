from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


_FILE_RE = re.compile(r"phoenix14t\.pami0\.(train|dev|test)\.annotations_only\.gzip$")


def _discover_annotation_files(root: Path) -> List[Path]:
    files = []
    for path in sorted(root.glob("*.gzip")):
        if _FILE_RE.match(path.name):
            files.append(path)
    if not files:
        raise FileNotFoundError(f"No PHOENIX annotation gzip files found in {root}")
    return files


def _split_to_training_split(split: str) -> str:
    split_norm = str(split).strip().lower()
    if split_norm == "dev":
        return "val"
    return split_norm


def _load_pickle_gzip(path: Path) -> List[Dict[str, Any]]:
    with gzip.open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected list payload in {path}, got {type(payload).__name__}")
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise RuntimeError(f"Expected dict row in {path} item {idx}, got {type(row).__name__}")
        out.append(dict(row))
    return out


def _build_row(
    row: Dict[str, Any],
    *,
    source_file: Path,
    videos_dir: Path,
    label_source: str,
) -> Dict[str, Any]:
    source_match = _FILE_RE.match(source_file.name)
    if source_match is None:
        raise RuntimeError(f"Unexpected PHOENIX annotation filename: {source_file.name}")
    source_split = source_match.group(1)

    raw_name = str(row.get("name") or "").strip()
    if not raw_name or "/" not in raw_name:
        raise RuntimeError(f"Missing or invalid 'name' in {source_file}: {row!r}")

    split_prefix, stem = raw_name.split("/", 1)
    if split_prefix != source_split:
        raise RuntimeError(
            f"Split mismatch in {source_file.name}: row says {split_prefix!r}, file says {source_split!r}"
        )

    gloss = str(row.get("gloss") or "").strip()
    translation = str(row.get("text") or "").strip()
    signer = str(row.get("signer") or "").strip()
    video_relpath = f"{raw_name}.mp4"
    attachment_id = raw_name.replace("/", "__")
    video_path = videos_dir / video_relpath

    if label_source == "gloss":
        label_text = gloss
    elif label_source == "translation":
        label_text = translation
    else:
        raise ValueError(f"Unsupported label_source: {label_source}")

    return {
        "attachment_id": attachment_id,
        "name": raw_name,
        "video_relpath": video_relpath.replace("\\", "/"),
        "video_path": str(video_path).replace("\\", "/"),
        "video_exists": bool(video_path.exists()),
        "split": _split_to_training_split(split_prefix),
        "orig_split": split_prefix,
        "train": "True" if split_prefix == "train" else "False",
        "text": label_text,
        "gloss": gloss,
        "translation": translation,
        "signer_id": signer,
        "begin": 0,
        "end": "",
        "dataset": "phoenix14t",
    }


def _write_tsv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Unpack PHOENIX-2014T *.annotations_only.gzip files into TSV/JSONL manifests "
            "with attachment_id values aligned to this repo's keypoint extractor."
        )
    )
    ap.add_argument("--annotations-dir", default="datasets/phoenix", help="Directory with PHOENIX *.gzip files.")
    ap.add_argument(
        "--videos-dir",
        default="datasets/phoenix/videos_phoenix/videos",
        help="Root directory with train/dev/test PHOENIX videos.",
    )
    ap.add_argument(
        "--out-dir",
        default="datasets/phoenix/prepared",
        help="Output directory for prepared PHOENIX manifests.",
    )
    ap.add_argument(
        "--label-source",
        choices=["gloss", "translation"],
        default="gloss",
        help="Which PHOENIX target to copy into the generic 'text' column.",
    )
    args = ap.parse_args(argv)

    annotations_dir = Path(args.annotations_dir)
    videos_dir = Path(args.videos_dir)
    out_dir = Path(args.out_dir)

    files = _discover_annotation_files(annotations_dir)
    prepared_rows: List[Dict[str, Any]] = []
    by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    signer_counts: Counter[str] = Counter()
    missing_videos: List[str] = []

    for path in files:
        for raw_row in _load_pickle_gzip(path):
            row = _build_row(raw_row, source_file=path, videos_dir=videos_dir, label_source=args.label_source)
            prepared_rows.append(row)
            split = str(row["split"])
            by_split.setdefault(split, []).append(row)
            signer_id = str(row.get("signer_id") or "").strip()
            if signer_id:
                signer_counts[signer_id] += 1
            if not bool(row["video_exists"]):
                missing_videos.append(str(row["video_relpath"]))

    prepared_rows.sort(key=lambda item: str(item["attachment_id"]))
    for rows in by_split.values():
        rows.sort(key=lambda item: str(item["attachment_id"]))

    fieldnames = [
        "attachment_id",
        "text",
        "gloss",
        "translation",
        "signer_id",
        "split",
        "orig_split",
        "train",
        "begin",
        "end",
        "dataset",
        "name",
        "video_relpath",
        "video_path",
        "video_exists",
    ]

    _write_tsv(out_dir / "phoenix14t.all.tsv", prepared_rows, fieldnames)
    _write_jsonl(out_dir / "phoenix14t.all.jsonl", prepared_rows)

    for split, rows in sorted(by_split.items()):
        _write_tsv(out_dir / f"phoenix14t.{split}.tsv", rows, fieldnames)
        _write_jsonl(out_dir / f"phoenix14t.{split}.jsonl", rows)

    summary = {
        "annotation_files": [str(p).replace("\\", "/") for p in files],
        "videos_dir": str(videos_dir).replace("\\", "/"),
        "label_source": str(args.label_source),
        "rows_total": int(len(prepared_rows)),
        "rows_by_split": {k: int(len(v)) for k, v in sorted(by_split.items())},
        "unique_signers": int(len(signer_counts)),
        "signer_counts": {k: int(v) for k, v in sorted(signer_counts.items())},
        "missing_videos_count": int(len(missing_videos)),
        "missing_videos_sample": sorted(set(missing_videos))[:20],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote {out_dir / 'phoenix14t.all.tsv'}")
    print(f"[OK] wrote {out_dir / 'phoenix14t.all.jsonl'}")
    for split in sorted(by_split):
        print(f"[OK] wrote {out_dir / f'phoenix14t.{split}.tsv'}")
        print(f"[OK] wrote {out_dir / f'phoenix14t.{split}.jsonl'}")
    print(f"[OK] wrote {out_dir / 'summary.json'}")
    print(
        "[SUMMARY]",
        json.dumps(
            {
                "rows_total": summary["rows_total"],
                "rows_by_split": summary["rows_by_split"],
                "unique_signers": summary["unique_signers"],
                "missing_videos_count": summary["missing_videos_count"],
            },
            ensure_ascii=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
