#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create Step-1 compatible prelabels for IPN Hand O-background segments.

Input:
  - a directory with extracted segment files (one per seg_uid), either:
      * .npz with:
          pts: (T,V,3) float32
          mask: (T,V,1) float32
      * .json / _pp.json from scripts/extract_keypoints.py segments mode
  - a manifest jsonl/csv from ipn/make_manifest.py (to keep split info)

Output (like bio/pipeline/prelabel.py):
  out_dir/
    <seg_uid>.npz
    index.json
    index.csv
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

from bio.core.config_utils import load_config_section, write_dataset_manifest, write_run_config


BIO_O = np.uint8(0)
HAND_JOINTS = 21
NUM_HAND_NODES = 42


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
    source_group: str


def _norm_split(raw: object) -> str:
    s = str(raw or "").strip().lower()
    if s == "test":
        return "val"
    return s


def _pick_points_from_hand_obj(hand_obj: Any) -> Any:
    if isinstance(hand_obj, dict):
        for key in ("landmarks", "hand_landmarks", "keypoints", "points", "pts", "xyz"):
            if key in hand_obj:
                return hand_obj[key]
        if "x" in hand_obj or "X" in hand_obj:
            return [hand_obj]
    return hand_obj


def _points_to_array(points: Any) -> Optional[np.ndarray]:
    if points is None:
        return None
    if isinstance(points, np.ndarray):
        arr = points.astype(np.float32, copy=False)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None
        if arr.shape[1] == 2:
            arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=arr.dtype)], axis=1)
        return arr[:, :3]
    if not isinstance(points, list):
        return None
    if not points:
        return None

    arr = np.full((len(points), 3), np.nan, dtype=np.float32)
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
        elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
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
    arr = _points_to_array(_pick_points_from_hand_obj(hand_obj))
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


def _extract_hand_landmarks(frame: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not isinstance(frame, dict):
        return None, None

    if ("hand 1" in frame) or ("hand 2" in frame):
        return _to_hand_array(frame.get("hand 1")), _to_hand_array(frame.get("hand 2"))
    if ("left_hand" in frame) or ("right_hand" in frame):
        return _to_hand_array(frame.get("left_hand")), _to_hand_array(frame.get("right_hand"))

    hands = frame.get("hands")
    if isinstance(hands, dict):
        left = _to_hand_array(hands.get("left") or hands.get("left_hand"))
        right = _to_hand_array(hands.get("right") or hands.get("right_hand"))
        return left, right
    if isinstance(hands, list):
        left = right = None
        for hand in hands:
            side = _hand_side_from_obj(hand)
            pts = _to_hand_array(hand)
            if side == "left" and left is None:
                left = pts
            elif side == "right" and right is None:
                right = pts
            elif left is None:
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


def _combine_hands(left: Optional[np.ndarray], right: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.full((NUM_HAND_NODES, 3), np.nan, dtype=np.float32)
    mask = np.zeros((NUM_HAND_NODES, 1), dtype=np.float32)
    if left is not None:
        pts[:HAND_JOINTS] = left
        mask[:HAND_JOINTS, 0] = np.isfinite(left).all(axis=1).astype(np.float32)
    if right is not None:
        pts[HAND_JOINTS:] = right
        mask[HAND_JOINTS:, 0] = np.isfinite(right).all(axis=1).astype(np.float32)
    pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)
    return pts, mask


def _parse_frames_from_json(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        frames = payload.get("frames", [])
        if isinstance(frames, list):
            return frames
    if isinstance(payload, list):
        return payload
    raise ValueError("Unsupported segment JSON payload; expected list or dict with 'frames'.")


def _load_segment_json(path: Path, expect_v: int) -> Tuple[np.ndarray, np.ndarray]:
    if expect_v > 0 and expect_v != NUM_HAND_NODES:
        raise RuntimeError(
            f"JSON segment loader currently supports V={NUM_HAND_NODES} only; got expect_V={expect_v}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = _parse_frames_from_json(payload)
    T = len(frames)
    pts = np.zeros((T, NUM_HAND_NODES, 3), dtype=np.float32)
    mask = np.zeros((T, NUM_HAND_NODES, 1), dtype=np.float32)
    for i, frame in enumerate(frames):
        left, right = _extract_hand_landmarks(frame)
        cur_pts, cur_mask = _combine_hands(left, right)
        pts[i] = cur_pts
        mask[i] = cur_mask
    return pts, mask


def _load_segment_npz(path: Path, expect_v: int) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        pts = d["pts"]
        mask = d["mask"]
    if pts.ndim != 3 or mask.ndim != 3:
        raise RuntimeError(f"Bad shapes in {path}: pts={pts.shape}, mask={mask.shape}")
    T, V, C = pts.shape
    if C != 3 or mask.shape[0] != T or mask.shape[1] != V:
        raise RuntimeError(f"Inconsistent pts/mask shapes in {path}: pts={pts.shape}, mask={mask.shape}")
    if expect_v > 0 and V != int(expect_v):
        raise RuntimeError(f"V mismatch in {path}: got V={V}, expected {int(expect_v)}")
    return pts.astype(np.float32, copy=False), mask.astype(np.float32, copy=False)


def _resolve_segment_path(segments_dir: Path, seg_uid: str, prefer_pp: bool) -> Optional[Path]:
    npz_path = segments_dir / f"{seg_uid}.npz"
    raw_json = segments_dir / f"{seg_uid}.json"
    pp_json = segments_dir / f"{seg_uid}_pp.json"
    if npz_path.exists():
        return npz_path
    if prefer_pp and pp_json.exists():
        return pp_json
    if raw_json.exists():
        return raw_json
    if pp_json.exists():
        return pp_json
    return None


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
    default_segments_dir = defaults.get("segments_dir", defaults.get("skeleton_npz_dir"))
    ap.add_argument(
        "--segments_dir",
        type=str,
        default=default_segments_dir,
        required=_is_missing(default_segments_dir),
        help="dir with extracted segment files per seg_uid (.npz or .json/_pp.json).",
    )
    ap.add_argument(
        "--skeleton_npz_dir",
        dest="segments_dir",
        type=str,
        default=None,
        help="Deprecated alias for --segments_dir.",
    )
    ap.add_argument(
        "--split",
        type=str,
        default=str(defaults.get("split", "all")),
        help="Manifest split to keep: train / val / all. Empty or 'all' keeps every row.",
    )
    ap.add_argument("--out_dir", type=str, default=defaults.get("out_dir"), required=_is_missing(defaults.get("out_dir")), help="output Step-1 prelabel dir (for Step2 as extra no_event)")
    ap.add_argument("--expect_V", type=int, default=int(defaults.get("expect_V", 42)), help="expected number of keypoints V (default 42 for hands)")
    prefer_pp_default = bool(defaults.get("prefer_pp", True))
    ap.add_argument("--prefer_pp", dest="prefer_pp", action="store_true", default=prefer_pp_default, help="Prefer *_pp.json over raw .json when both exist.")
    ap.add_argument("--no_prefer_pp", dest="prefer_pp", action="store_false")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    manifest = Path(args.manifest)
    seg_dir = Path(args.segments_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(out_dir, args, config_path=args.config, section="ipn_prelabel")

    rows = _read_manifest(manifest)
    requested_split = _norm_split(args.split)
    if requested_split in ("", "all"):
        requested_split = ""
    if requested_split:
        rows = [r for r in rows if _norm_split(r.get("split", "train")) == requested_split]

    index: List[IndexRow] = []
    missing = 0
    bad = 0
    loaded_npz = 0
    loaded_json = 0

    for r in rows:
        seg_uid = str(r.get("seg_uid", "")).strip()
        if not seg_uid:
            continue
        split = _norm_split(r.get("split", "train")) or "train"

        src = _resolve_segment_path(seg_dir, seg_uid, prefer_pp=bool(args.prefer_pp))
        if src is None or not src.exists():
            missing += 1
            continue

        try:
            if src.suffix.lower() == ".npz":
                pts, mask = _load_segment_npz(src, int(args.expect_V))
                loaded_npz += 1
            else:
                pts, mask = _load_segment_json(src, int(args.expect_V))
                loaded_json += 1
        except Exception:
            bad += 1
            continue
        T, V, _ = pts.shape

        bio = np.zeros((T,), dtype=np.uint8)  # all O
        label_str = "no_event"
        start_frame = int(r.get("start", 0) or 0)
        ts = np.arange(start_frame, start_frame + T, dtype=np.float32)
        meta = {
            "dataset": str(r.get("dataset", "ipn_hand")),
            "split": split,
            "video_id": str(r.get("video_id", "")),
            "label": str(r.get("label", "")),
            "seg_uid": seg_uid,
            "start": start_frame,
            "end": int(r.get("end", start_frame + T) or (start_frame + T)),
            "length": int(r.get("length", T) or T),
            "source_file": str(src).replace("\\", "/"),
        }

        out_path_rel = Path(f"{seg_uid}.npz")
        out_path = out_dir / out_path_rel

        np.savez(
            out_path,
            pts=pts.astype(np.float32, copy=False),
            mask=mask.astype(np.float32, copy=False),
            ts=ts,
            bio=bio,
            label_str=np.asarray(label_str),
            is_no_event=np.asarray(True),
            start_idx=np.asarray(-1, dtype=np.int32),
            end_idx=np.asarray(-1, dtype=np.int32),
            meta=np.asarray(json.dumps(meta, ensure_ascii=False)),
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
                source_group=str(r.get("video_id", "") or seg_uid),
            )
        )

    # write index.json
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.json").write_text(
        json.dumps([asdict(x) for x in index], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "index.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["vid", "label_str", "path_to_npz", "T_total", "start_idx", "end_idx", "is_no_event", "split", "dataset", "source_group"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in index:
            w.writerow(asdict(row))

    summary = {
        "num_rows_manifest": int(len(rows)),
        "num_index_written": int(len(index)),
        "missing_segment_files": int(missing),
        "bad_segment_files": int(bad),
        "loaded_npz": int(loaded_npz),
        "loaded_json": int(loaded_json),
        "expect_V": int(args.expect_V),
        "segments_dir": str(seg_dir),
        "prefer_pp": bool(args.prefer_pp),
        "split": requested_split or "all",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_dataset_manifest(
        out_dir,
        stage="ipn_prelabel",
        args=args,
        config_path=args.config,
        section="ipn_prelabel",
        inputs={
            "manifest": str(manifest),
            "segments_dir": str(seg_dir),
        },
        counts={
            "manifest_rows": int(len(rows)),
            "written_samples": int(len(index)),
            "missing_segment_files": int(missing),
            "bad_segment_files": int(bad),
            "split": requested_split or "all",
        },
        extra={"summary": summary},
    )

    print("[OK] out_dir:", out_dir)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
