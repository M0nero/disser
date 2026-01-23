#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IPN Hand: build a single manifest for extracting O-background segments (e.g., D0X).

Reads:
  - Annot_TrainList.txt
  - Annot_TestList.txt
(also supports Annot_List.txt if you want)

Each line is expected to be CSV:
  video,label,id,t_start,t_end,frames

IMPORTANT:
  IPN files use 1-based inclusive [t_start..t_end].
  We convert to 0-based half-open [start0, end_excl):
    start0  = t_start - 1
    end_excl = t_end       (because inclusive end in 1-based)

Optionally chunks long intervals into fixed windows (chunk_frames/stride).

Writes:
  - out_manifest.jsonl (one object per segment)
  - out_manifest.csv   (same content)
  - out_manifest.stats.json (summary)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bio.core.config_utils import load_config_section, write_run_config


@dataclass(frozen=True)
class Segment:
    dataset: str
    split: str              # train / val
    video_id: str
    label: str              # original label, e.g. D0X
    seg_uid: str            # unique id for this segment (stable)
    start: int              # 0-based, inclusive
    end: int                # 0-based, exclusive
    length: int             # end - start


def _read_rows(path: Path) -> Iterator[Dict[str, str]]:
    """
    Supports:
      - with header: video,label,id,t_start,t_end,frames
      - without header: same order
    """
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        # sniff header
        first = f.readline()
        if not first:
            return
        f.seek(0)

        # detect if first token is "video"
        peek = first.strip().split(",")
        has_header = bool(peek) and peek[0].strip().lower() == "video"

        if has_header:
            rdr = csv.DictReader(f)
            for r in rdr:
                yield {k: (v or "").strip() for k, v in r.items()}
        else:
            rdr2 = csv.reader(f)
            for row in rdr2:
                if not row:
                    continue
                # video,label,id,t_start,t_end,frames
                row = [x.strip() for x in row]
                if len(row) < 5:
                    continue
                yield {
                    "video": row[0],
                    "label": row[1] if len(row) > 1 else "",
                    "id": row[2] if len(row) > 2 else "",
                    "t_start": row[3] if len(row) > 3 else "",
                    "t_end": row[4] if len(row) > 4 else "",
                    "frames": row[5] if len(row) > 5 else "",
                }


def _to_int(x: str, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _is_missing(raw: object) -> bool:
    if raw is None:
        return True
    if isinstance(raw, str) and not raw.strip():
        return True
    return False


def _iter_segments_from_list(
    list_path: Path,
    split: str,
    keep_labels: Sequence[str],
    min_frames: int,
    max_frames: int,
    chunk_frames: int,
    chunk_stride: int,
    include_tail: bool,
) -> Iterator[Segment]:
    keep = {s.strip() for s in keep_labels if s.strip()}
    dataset = "ipn_hand"

    for r in _read_rows(list_path):
        video = (r.get("video") or "").strip()
        label = (r.get("label") or "").strip()

        if not video or not label:
            continue
        if keep and label not in keep:
            continue

        t_start = _to_int(r.get("t_start", ""), default=0)
        t_end = _to_int(r.get("t_end", ""), default=-1)
        if t_start <= 0 or t_end <= 0 or t_end < t_start:
            continue

        # Convert 1-based inclusive to 0-based [start, end)
        start0 = t_start - 1
        end_excl = t_end
        length = end_excl - start0
        if length <= 0:
            continue

        if max_frames > 0 and length > max_frames:
            # we still can chunk it down; only apply max_frames if chunking disabled
            if chunk_frames <= 0:
                continue

        if chunk_frames and chunk_frames > 0:
            stride = chunk_stride if chunk_stride > 0 else chunk_frames
            last_start = None
            s = start0
            while s + chunk_frames <= end_excl:
                e = s + chunk_frames
                if (e - s) >= min_frames and (max_frames <= 0 or (e - s) <= max_frames):
                    seg_uid = f"{video}__{label}__{t_start}-{t_end}__{s}-{e}"
                    yield Segment(dataset, split, video, label, seg_uid, s, e, e - s)
                last_start = s
                s += stride

            # optionally add tail-aligned chunk
            if include_tail and end_excl - start0 >= chunk_frames:
                tail_s = end_excl - chunk_frames
                if last_start is None or tail_s != last_start:
                    tail_e = end_excl
                    if (tail_e - tail_s) >= min_frames and (max_frames <= 0 or (tail_e - tail_s) <= max_frames):
                        seg_uid = f"{video}__{label}__{t_start}-{t_end}__{tail_s}-{tail_e}"
                        yield Segment(dataset, split, video, label, seg_uid, tail_s, tail_e, tail_e - tail_s)
        else:
            if length < min_frames:
                continue
            if max_frames > 0 and length > max_frames:
                continue
            seg_uid = f"{video}__{label}__{t_start}-{t_end}"
            yield Segment(dataset, split, video, label, seg_uid, start0, end_excl, length)


def _write_jsonl(path: Path, segs: Sequence[Segment]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for s in segs:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")


def _write_csv(path: Path, segs: Sequence[Segment]) -> None:
    fields = ["dataset", "split", "video_id", "label", "seg_uid", "start", "end", "length"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in segs:
            w.writerow(asdict(s))


def _stats(segs: Sequence[Segment]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    by_split: Dict[str, List[int]] = {}
    for s in segs:
        by_split.setdefault(s.split, []).append(int(s.length))
    out["num_segments"] = int(len(segs))
    out["splits"] = {}
    for k, lens in by_split.items():
        lens = sorted(lens)
        if not lens:
            out["splits"][k] = {"count": 0}
            continue
        def pct(p: float) -> int:
            idx = int(round((len(lens) - 1) * p))
            return int(lens[max(0, min(len(lens) - 1, idx))])
        out["splits"][k] = {
            "count": int(len(lens)),
            "min": int(lens[0]),
            "p25": pct(0.25),
            "median": pct(0.50),
            "p75": pct(0.75),
            "max": int(lens[-1]),
            "mean": float(sum(lens) / max(1, len(lens))),
        }
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, _ = pre.parse_known_args(argv)
    defaults = load_config_section(pre_args.config, "ipn_make_manifest")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=pre_args.config, help="Path to config JSON (section: ipn_make_manifest).")
    ap.add_argument("--train_list", type=str, default=defaults.get("train_list"), required=_is_missing(defaults.get("train_list")), help="Annot_TrainList.txt")
    ap.add_argument("--test_list", type=str, default=defaults.get("test_list"), required=_is_missing(defaults.get("test_list")), help="Annot_TestList.txt (will become val split)")
    ap.add_argument("--out", type=str, default=defaults.get("out"), required=_is_missing(defaults.get("out")), help="Output base path (without ext) OR .jsonl path")
    ap.add_argument("--labels", type=str, default=defaults.get("labels", "D0X"), help="Comma-separated labels to keep, default: D0X")
    ap.add_argument("--min_frames", type=int, default=int(defaults.get("min_frames", 48)), help="Filter out segments shorter than this")
    ap.add_argument("--max_frames", type=int, default=int(defaults.get("max_frames", 0)), help="0 = no max (if chunking off)")
    ap.add_argument("--chunk_frames", type=int, default=int(defaults.get("chunk_frames", 96)), help="0 = keep full intervals; else chunk length")
    ap.add_argument("--chunk_stride", type=int, default=int(defaults.get("chunk_stride", 96)), help="Stride between chunks (default=chunk_frames)")
    ap.add_argument("--include_tail", action="store_true", default=bool(defaults.get("include_tail", False)), help="Add tail-aligned chunk per interval")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    train_list = Path(args.train_list)
    test_list = Path(args.test_list)
    labels = [s.strip() for s in (args.labels or "").split(",") if s.strip()]

    segs: List[Segment] = []
    segs.extend(
        list(
            _iter_segments_from_list(
                train_list,
                split="train",
                keep_labels=labels,
                min_frames=int(args.min_frames),
                max_frames=int(args.max_frames),
                chunk_frames=int(args.chunk_frames),
                chunk_stride=int(args.chunk_stride),
                include_tail=bool(args.include_tail),
            )
        )
    )
    segs.extend(
        list(
            _iter_segments_from_list(
                test_list,
                split="val",
                keep_labels=labels,
                min_frames=int(args.min_frames),
                max_frames=int(args.max_frames),
                chunk_frames=int(args.chunk_frames),
                chunk_stride=int(args.chunk_stride),
                include_tail=bool(args.include_tail),
            )
        )
    )

    # de-dup by seg_uid
    uniq: Dict[str, Segment] = {}
    for s in segs:
        uniq[s.seg_uid] = s
    segs = list(uniq.values())
    segs.sort(key=lambda x: (x.split, x.video_id, x.start, x.end))

    out_path = Path(args.out)
    if out_path.suffix.lower() == ".jsonl":
        base = out_path.with_suffix("")
    else:
        base = out_path
    base.parent.mkdir(parents=True, exist_ok=True)
    write_run_config(base.parent, args, config_path=args.config, section="ipn_make_manifest", extra={"base": str(base)})

    jsonl_path = base.with_suffix(".jsonl")
    csv_path = base.with_suffix(".csv")
    stats_path = base.with_suffix(".stats.json")

    _write_jsonl(jsonl_path, segs)
    _write_csv(csv_path, segs)
    stats = _stats(segs)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote: {jsonl_path}")
    print(f"[OK] wrote: {csv_path}")
    print(f"[OK] wrote: {stats_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
