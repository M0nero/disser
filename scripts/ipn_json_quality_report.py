#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quality report for IPNHand segment JSONs.

Reads per-segment JSON files (each ~96 frames) and computes tracking quality
metrics for left/right hands. Outputs stats.json, segments.csv, keep_list.txt,
drop_list.txt, and can write a cleaned manifest JSONL.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

HAND_JOINTS = 21


def _stem_no_pp(path: Path) -> str:
    stem = path.stem
    return stem[:-3] if stem.endswith("_pp") else stem


def load_manifest_records(manifest_path: str) -> Dict[str, Dict[str, Any]]:
    if not manifest_path:
        return {}
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    out: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    if suffix == ".csv":
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",\t;|")
        except csv.Error:
            class _D(csv.Dialect):
                delimiter = "," if "," in text[:4096] else "\t"
                quotechar = '"'
                doublequote = True
                skipinitialspace = False
                lineterminator = "\n"
                quoting = csv.QUOTE_MINIMAL
            dialect = _D()
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rdr = csv.DictReader(f, dialect=dialect)
            for row in rdr:
                if isinstance(row, dict):
                    rows.append(row)
        return _rows_to_manifest_map(rows)

    # json / jsonl
    if suffix in (".json", ".jsonl"):
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                rows = [r for r in obj if isinstance(r, dict)]
                return _rows_to_manifest_map(rows)
            if isinstance(obj, dict):
                rows = obj.get("items") or obj.get("rows")
                if isinstance(rows, list):
                    rows = [r for r in rows if isinstance(r, dict)]
                    return _rows_to_manifest_map(rows)
                if all(isinstance(v, str) for v in obj.values()):
                    rows = [{"seg_uid": k, "split": v} for k, v in obj.items()]
                    return _rows_to_manifest_map(rows)
                if all(isinstance(v, dict) for v in obj.values()):
                    rows = []
                    for k, v in obj.items():
                        row = dict(v)
                        row.setdefault("seg_uid", k)
                        rows.append(row)
                    return _rows_to_manifest_map(rows)
        except Exception:
            pass

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        rows.append(row)

    return _rows_to_manifest_map(rows)


def _rows_to_manifest_map(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        seg_uid = Path(str(row.get("seg_uid") or "")).stem
        if seg_uid:
            out[seg_uid] = row
    return out


def load_manifest_splits(manifest_path: str) -> Dict[str, str]:
    records = load_manifest_records(manifest_path)
    out: Dict[str, str] = {}
    for seg_uid, row in records.items():
        split = (str(row.get("split") or "")).strip().lower() or "unknown"
        out[seg_uid] = split
    return out


def iter_json_files(ipn_dir: Path, max_files: int, prefer_pp: bool = True) -> Iterator[Path]:
    files = sorted(ipn_dir.glob("*.json"))
    by_id: Dict[str, Dict[str, Path]] = {}
    for f in files:
        stem = f.stem
        if stem.endswith("_pp"):
            key = stem[:-3]
            by_id.setdefault(key, {})["pp"] = f
        else:
            by_id.setdefault(stem, {})["raw"] = f

    chosen: List[Path] = []
    for entry in by_id.values():
        if prefer_pp and "pp" in entry:
            chosen.append(entry["pp"])
        elif "raw" in entry:
            chosen.append(entry["raw"])
        elif "pp" in entry:
            chosen.append(entry["pp"])

    chosen = sorted(chosen)
    if max_files and max_files > 0:
        chosen = chosen[: max_files]
    return iter(chosen)


def parse_frames_from_json(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        frames = payload.get("frames", [])
        if isinstance(frames, list):
            return frames
    if isinstance(payload, list):
        return payload
    raise ValueError("Unsupported JSON payload shape (expected dict with frames or list)")


def _pick_points_from_hand_obj(hand_obj: Any) -> Any:
    if isinstance(hand_obj, dict):
        for key in ("landmarks", "hand_landmarks", "keypoints", "points", "pts", "xyz"):
            if key in hand_obj:
                return hand_obj[key]
        # single-point dict with x/y?
        if "x" in hand_obj or "X" in hand_obj:
            return [hand_obj]
    return hand_obj


def _points_to_array(points: Any) -> Optional[np.ndarray]:
    if points is None:
        return None
    if isinstance(points, np.ndarray):
        arr = points.astype(np.float32, copy=False)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            if arr.shape[1] == 2:
                arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=arr.dtype)], axis=1)
            return arr[:, :3]
        return None
    if not isinstance(points, list):
        return None
    n = len(points)
    if n == 0:
        return None
    arr = np.full((n, 3), np.nan, dtype=np.float32)
    for i, pt in enumerate(points):
        x = y = z = None
        if isinstance(pt, dict):
            if "x" in pt or "y" in pt:
                x = pt.get("x")
                y = pt.get("y")
                z = pt.get("z", 0.0)
            elif "X" in pt or "Y" in pt:
                x = pt.get("X")
                y = pt.get("Y")
                z = pt.get("Z", 0.0)
            elif "xyz" in pt and isinstance(pt["xyz"], (list, tuple)):
                vals = pt["xyz"]
                if len(vals) >= 2:
                    x, y = vals[0], vals[1]
                    z = vals[2] if len(vals) >= 3 else 0.0
        elif isinstance(pt, (list, tuple)):
            if len(pt) >= 2:
                x, y = pt[0], pt[1]
                z = pt[2] if len(pt) >= 3 else 0.0
        if x is None or y is None:
            continue
        try:
            arr[i, 0] = float(x)
            arr[i, 1] = float(y)
            arr[i, 2] = float(z if z is not None else 0.0)
        except Exception:
            continue
    return arr


def _to_hand_array(hand_obj: Any) -> Optional[np.ndarray]:
    points = _pick_points_from_hand_obj(hand_obj)
    arr = _points_to_array(points)
    if arr is None:
        return None
    if arr.shape[0] < HAND_JOINTS:
        out = np.full((HAND_JOINTS, 3), np.nan, dtype=np.float32)
        out[: arr.shape[0]] = arr
        return out
    if arr.shape[0] > HAND_JOINTS:
        return arr[:HAND_JOINTS].astype(np.float32, copy=False)
    return arr.astype(np.float32, copy=False)


def _hand_side_from_obj(hand_obj: Any) -> str:
    if not isinstance(hand_obj, dict):
        return ""
    for key in ("handedness", "handedness_label", "label", "type", "side", "hand"):
        val = hand_obj.get(key)
        if isinstance(val, dict):
            val = val.get("label") or val.get("type")
        if val is None:
            continue
        s = str(val).strip().lower()
        if "left" in s:
            return "left"
        if "right" in s:
            return "right"
    return ""


def extract_hand_landmarks(frame: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not isinstance(frame, dict):
        return None, None

    if ("hand 1" in frame) or ("hand 2" in frame):
        left = _to_hand_array(frame.get("hand 1"))
        right = _to_hand_array(frame.get("hand 2"))
        return left, right

    if ("left_hand" in frame) or ("right_hand" in frame):
        left = _to_hand_array(frame.get("left_hand"))
        right = _to_hand_array(frame.get("right_hand"))
        return left, right

    if "hands" in frame:
        hands = frame.get("hands")
        if isinstance(hands, dict):
            left = _to_hand_array(hands.get("left") or hands.get("left_hand"))
            right = _to_hand_array(hands.get("right") or hands.get("right_hand"))
            return left, right
        if isinstance(hands, list):
            left = right = None
            for h in hands:
                side = _hand_side_from_obj(h)
                pts = _to_hand_array(_pick_points_from_hand_obj(h))
                if side == "left" and left is None:
                    left = pts
                elif side == "right" and right is None:
                    right = pts
                else:
                    if left is None:
                        left = pts
                    elif right is None:
                        right = pts
            return left, right

    for key in ("hand_landmarks", "landmarks"):
        if key not in frame:
            continue
        val = frame.get(key)
        if isinstance(val, dict):
            left = _to_hand_array(val.get("left") or val.get("left_hand"))
            right = _to_hand_array(val.get("right") or val.get("right_hand"))
            return left, right
        if isinstance(val, list):
            if len(val) == 2 and all(isinstance(v, list) for v in val):
                return _to_hand_array(val[0]), _to_hand_array(val[1])
            if len(val) == 1 and isinstance(val[0], list):
                return _to_hand_array(val[0]), None
            if len(val) == HAND_JOINTS:
                return _to_hand_array(val), None

    return None, None


def _count_visible_points(hand: Optional[np.ndarray]) -> int:
    if hand is None:
        return 0
    mask = np.isfinite(hand).all(axis=1)
    return int(mask.sum())


def _combine_hands(left: Optional[np.ndarray], right: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.full((HAND_JOINTS * 2, 3), np.nan, dtype=np.float32)
    mask = np.zeros((HAND_JOINTS * 2,), dtype=bool)
    if left is not None:
        pts[:HAND_JOINTS] = left
        mask[:HAND_JOINTS] = np.isfinite(left).all(axis=1)
    if right is not None:
        pts[HAND_JOINTS:] = right
        mask[HAND_JOINTS:] = np.isfinite(right).all(axis=1)
    return pts, mask


def compute_metrics_for_segment(
    frames: List[Dict[str, Any]],
    min_points_per_hand: int,
) -> Dict[str, float]:
    T = len(frames)
    if T == 0:
        return {
            "T": 0,
            "any_present_rate": 0.0,
            "left_present_rate": 0.0,
            "right_present_rate": 0.0,
            "both_present_rate": 0.0,
            "any_present_frames": 0,
            "max_consecutive_missing": 0,
            "mean_points_visible": 0.0,
            "motion_score": 0.0,
            "jitter_score": 0.0,
        }

    left_present = 0
    right_present = 0
    both_present = 0
    any_present = 0
    total_points_visible = 0

    max_missing = 0
    cur_missing = 0

    motion_vals: List[float] = []
    prev_pts = None
    prev_mask = None

    for fr in frames:
        left, right = extract_hand_landmarks(fr)
        left_cnt = _count_visible_points(left)
        right_cnt = _count_visible_points(right)

        left_vis = left_cnt >= min_points_per_hand
        right_vis = right_cnt >= min_points_per_hand
        any_vis = left_vis or right_vis
        both_vis = left_vis and right_vis

        if left_vis:
            left_present += 1
        if right_vis:
            right_present += 1
        if both_vis:
            both_present += 1
        if any_vis:
            any_present += 1
            cur_missing = 0
        else:
            cur_missing += 1
            if cur_missing > max_missing:
                max_missing = cur_missing

        total_points_visible += (left_cnt + right_cnt)

        cur_pts, cur_mask = _combine_hands(left, right)
        if prev_pts is not None and prev_mask is not None:
            valid = cur_mask & prev_mask
            if valid.any():
                diff = cur_pts[valid] - prev_pts[valid]
                speed = np.linalg.norm(diff, axis=1)
                motion_vals.append(float(speed.mean()))
            else:
                motion_vals.append(0.0)
        prev_pts, prev_mask = cur_pts, cur_mask

    any_present_rate = any_present / T
    left_present_rate = left_present / T
    right_present_rate = right_present / T
    both_present_rate = both_present / T
    mean_points_visible = total_points_visible / T

    if motion_vals:
        motion_score = float(np.mean(motion_vals))
        jitter_score = float(np.std(motion_vals))
    else:
        motion_score = 0.0
        jitter_score = 0.0

    return {
        "T": int(T),
        "any_present_rate": float(any_present_rate),
        "left_present_rate": float(left_present_rate),
        "right_present_rate": float(right_present_rate),
        "both_present_rate": float(both_present_rate),
        "any_present_frames": int(any_present),
        "max_consecutive_missing": int(max_missing),
        "mean_points_visible": float(mean_points_visible),
        "motion_score": float(motion_score),
        "jitter_score": float(jitter_score),
    }


def _percentiles(values: List[float], ps: Iterable[int]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float32)
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def _to_float(val: Any, default: float = float("inf")) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _split_norm(val: Any) -> str:
    s = str(val or "").strip().lower()
    return s or "unknown"


def _parse_keep_buckets(raw: str) -> List[str]:
    buckets = [b.strip().upper() for b in (raw or "").split(",") if b.strip()]
    return buckets or ["OK"]


def write_reports(
    out_dir: Path,
    rows: List[Dict[str, Any]],
    stats: Dict[str, Any],
    drop_weak: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=True, indent=2), encoding="utf-8")

    csv_path = out_dir / "segments.csv"
    cols = [
        "seg_uid",
        "split",
        "T",
        "any_present_rate",
        "left_present_rate",
        "right_present_rate",
        "both_present_rate",
        "max_consecutive_missing",
        "mean_points_visible",
        "motion_score",
        "jitter_score",
        "quality_score",
        "bucket",
        "path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    keep_path = out_dir / "keep_list.txt"
    drop_path = out_dir / "drop_list.txt"
    with keep_path.open("w", encoding="utf-8") as f_keep, drop_path.open("w", encoding="utf-8") as f_drop:
        for r in rows:
            uid = r.get("seg_uid", "")
            bucket = r.get("bucket", "")
            if not uid:
                continue
            if bucket == "OK":
                f_keep.write(uid + "\n")
            if bucket == "EMPTY" or (drop_weak and bucket == "WEAK"):
                f_drop.write(uid + "\n")


def _write_clean_manifest(
    out_path: Path,
    manifest_rows: Dict[str, Dict[str, Any]],
    rows: List[Dict[str, Any]],
    keep_buckets: List[str],
    min_quality_score: float,
    topk: int,
    per_split: bool,
) -> Dict[str, Any]:
    keep_set = {b.upper() for b in keep_buckets}
    kept: List[Dict[str, Any]] = []
    drop_counts: Dict[str, int] = {"EMPTY": 0, "WEAK": 0, "OK": 0}
    drop_low_quality = 0
    missing_manifest = 0

    for r in rows:
        bucket = str(r.get("bucket", "")).upper()
        if bucket not in keep_set:
            drop_counts[bucket] = drop_counts.get(bucket, 0) + 1
            continue
        if float(r.get("quality_score", 0.0)) < float(min_quality_score):
            drop_low_quality += 1
            continue
        seg_uid = r.get("seg_uid")
        row = manifest_rows.get(seg_uid)
        if row is None:
            missing_manifest += 1
            continue
        split_norm = _split_norm(row.get("split"))
        clean_row = {
            "dataset": row.get("dataset"),
            "split": split_norm,
            "video_id": row.get("video_id"),
            "label": row.get("label"),
            "seg_uid": row.get("seg_uid", seg_uid),
            "start": row.get("start"),
            "end": row.get("end"),
            "length": row.get("length"),
            "quality_score": float(r.get("quality_score", 0.0)),
            "bucket": r.get("bucket"),
            "any_present_rate": float(r.get("any_present_rate", 0.0)),
            "both_present_rate": float(r.get("both_present_rate", 0.0)),
            "motion_score": float(r.get("motion_score", 0.0)),
            "max_consecutive_missing": int(r.get("max_consecutive_missing", 0)),
        }
        kept.append(clean_row)

    dropped_topk = 0
    if topk and topk > 0:
        if per_split:
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for r in kept:
                grouped.setdefault(_split_norm(r.get("split")), []).append(r)
            new_kept: List[Dict[str, Any]] = []
            for split, group in grouped.items():
                group_sorted = sorted(group, key=lambda x: float(x.get("quality_score", 0.0)), reverse=True)
                if topk < len(group_sorted):
                    dropped_topk += (len(group_sorted) - topk)
                new_kept.extend(group_sorted[:topk] if topk > 0 else group_sorted)
            kept = new_kept
        else:
            kept_sorted = sorted(kept, key=lambda x: float(x.get("quality_score", 0.0)), reverse=True)
            if topk < len(kept_sorted):
                dropped_topk = len(kept_sorted) - topk
            kept = kept_sorted[:topk] if topk > 0 else kept_sorted

    kept_sorted = sorted(
        kept,
        key=lambda x: (
            _split_norm(x.get("split")),
            str(x.get("video_id") or ""),
            _to_float(x.get("start")),
        ),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in kept_sorted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    kept_by_split: Dict[str, int] = {}
    for row in kept_sorted:
        split = _split_norm(row.get("split"))
        kept_by_split[split] = kept_by_split.get(split, 0) + 1

    stats = {
        "kept_total": int(len(kept_sorted)),
        "kept_by_split": kept_by_split,
        "dropped_by_bucket": drop_counts,
        "dropped_low_quality": int(drop_low_quality),
        "dropped_topk": int(dropped_topk),
        "missing_manifest": int(missing_manifest),
        "min_quality_score": float(min_quality_score),
        "keep_bucket": list(keep_set),
        "topk": int(topk),
        "per_split": bool(per_split),
    }
    return stats


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser("IPNHand JSON quality report")
    ap.add_argument("--ipn_dir", required=True, help="Directory with IPNHand segment JSONs")
    ap.add_argument("--manifest", default="", help="Optional manifest (csv/jsonl) with seg_uid + split")
    ap.add_argument("--out", required=True, help="Output directory for reports")
    ap.add_argument("--min_points_per_hand", type=int, default=5)
    ap.add_argument("--min_present_rate", type=float, default=0.20)
    ap.add_argument("--min_any_present_frames", type=int, default=5)
    ap.add_argument("--max_files", type=int, default=0)
    ap.add_argument("--drop_weak", action="store_true", help="Add WEAK segments to drop_list.txt")
    ap.add_argument("--write_clean_manifest", type=str, default="", help="Write filtered manifest JSONL to this path")
    ap.add_argument("--keep_bucket", type=str, default="OK", help="Comma-separated buckets to keep (default: OK)")
    ap.add_argument("--min_quality_score", type=float, default=0.35)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--per_split", action="store_true", help="Apply topk separately per split")
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

    ipn_dir = Path(args.ipn_dir)
    out_dir = Path(args.out)
    if not ipn_dir.exists():
        raise FileNotFoundError(ipn_dir)

    manifest_rows = load_manifest_records(args.manifest)

    rows: List[Dict[str, Any]] = []
    parse_errors = 0
    total_files = 0

    t0 = time.time()
    for path in iter_json_files(ipn_dir, args.max_files, prefer_pp=args.prefer_pp):
        total_files += 1
        seg_uid = _stem_no_pp(path)
        manifest_row = manifest_rows.get(seg_uid)
        split = _split_norm(manifest_row.get("split") if manifest_row else "unknown")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            frames = parse_frames_from_json(payload)
            metrics = compute_metrics_for_segment(frames, int(args.min_points_per_hand))
        except Exception:
            parse_errors += 1
            continue

        any_present_frames = int(metrics["any_present_frames"])
        any_present_rate = float(metrics["any_present_rate"])
        if any_present_frames < int(args.min_any_present_frames):
            bucket = "EMPTY"
        elif any_present_rate < float(args.min_present_rate):
            bucket = "WEAK"
        else:
            bucket = "OK"

        rows.append({
            "seg_uid": seg_uid,
            "split": split,
            "T": int(metrics["T"]),
            "any_present_rate": float(metrics["any_present_rate"]),
            "left_present_rate": float(metrics["left_present_rate"]),
            "right_present_rate": float(metrics["right_present_rate"]),
            "both_present_rate": float(metrics["both_present_rate"]),
            "max_consecutive_missing": int(metrics["max_consecutive_missing"]),
            "mean_points_visible": float(metrics["mean_points_visible"]),
            "motion_score": float(metrics["motion_score"]),
            "jitter_score": float(metrics["jitter_score"]),
            "quality_score": 0.0,
            "bucket": bucket,
            "path": str(path),
        })

    parsed_ok = len(rows)
    if parsed_ok == 0:
        raise RuntimeError("No segments parsed successfully.")

    motion_scores = [float(r["motion_score"]) for r in rows]
    p90_motion = float(np.percentile(np.asarray(motion_scores, dtype=np.float32), 90)) if motion_scores else 0.0
    denom = p90_motion if p90_motion > 1e-8 else 1.0

    for r in rows:
        motion_norm = min(1.0, float(r["motion_score"]) / denom)
        miss_ratio = float(r["max_consecutive_missing"]) / max(1, int(r["T"]))
        score = (
            0.5 * float(r["any_present_rate"])
            + 0.2 * float(r["both_present_rate"])
            + 0.2 * motion_norm
            - 0.1 * min(1.0, miss_ratio)
        )
        r["quality_score"] = float(max(0.0, min(1.0, score)))

    counts_by_split: Dict[str, Dict[str, int]] = {}
    for r in rows:
        split = r["split"]
        bucket = r["bucket"]
        if split not in counts_by_split:
            counts_by_split[split] = {"EMPTY": 0, "WEAK": 0, "OK": 0, "TOTAL": 0}
        counts_by_split[split][bucket] += 1
        counts_by_split[split]["TOTAL"] += 1

    stats = {
        "total_files": int(total_files),
        "parsed_ok": int(parsed_ok),
        "parse_errors": int(parse_errors),
        "elapsed_sec": float(time.time() - t0),
        "counts_by_split": counts_by_split,
        "percentiles": {
            "any_present_rate": _percentiles([float(r["any_present_rate"]) for r in rows], [5, 25, 50, 75, 95]),
            "both_present_rate": _percentiles([float(r["both_present_rate"]) for r in rows], [5, 25, 50, 75, 95]),
            "motion_score": _percentiles(motion_scores, [5, 25, 50, 75, 95]),
        },
        "motion_norm_p90": float(p90_motion),
    }

    write_reports(out_dir, rows, stats, drop_weak=bool(args.drop_weak))
    print(f"[OK] Wrote reports to: {out_dir}")

    if args.write_clean_manifest:
        if not manifest_rows:
            raise RuntimeError("--write_clean_manifest requires --manifest with seg_uid rows.")
        keep_buckets = _parse_keep_buckets(args.keep_bucket)
        clean_stats = _write_clean_manifest(
            out_path=Path(args.write_clean_manifest),
            manifest_rows=manifest_rows,
            rows=rows,
            keep_buckets=keep_buckets,
            min_quality_score=float(args.min_quality_score),
            topk=int(args.topk),
            per_split=bool(args.per_split),
        )
        stats_path = Path(args.write_clean_manifest).resolve().parent / "clean_manifest.stats.json"
        stats_path.write_text(json.dumps(clean_stats, ensure_ascii=True, indent=2), encoding="utf-8")
        kept_by_split = clean_stats.get("kept_by_split", {})
        dropped = clean_stats.get("dropped_by_bucket", {})
        print(
            "[CLEAN]",
            f"kept_by_split={kept_by_split}",
            f"dropped_EMPTY={dropped.get('EMPTY', 0)}",
            f"dropped_WEAK={dropped.get('WEAK', 0)}",
            f"dropped_low_quality={clean_stats.get('dropped_low_quality', 0)}",
            f"dropped_topk={clean_stats.get('dropped_topk', 0)}",
        )
        print(f"[CLEAN] stats -> {stats_path}")


if __name__ == "__main__":
    main()
