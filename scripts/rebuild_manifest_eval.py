#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild manifest.json and eval_report.json from existing keypoints JSON files.

This scans a skeletons directory (raw + _pp JSONs), recomputes per-video metrics
from the stored frames, and writes:
  - manifest.json (one entry per raw file)
  - eval_report.json (aggregate + per-video eval)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kp_export.algos.postprocess import postprocess_sequence

MP_HAND_NUM_LANDMARKS = 21


def _wrist_xy(pts: Optional[List[Dict[str, float]]]) -> Optional[Tuple[float, float]]:
    if not pts or len(pts) == 0:
        return None
    try:
        return float(pts[0]["x"]), float(pts[0]["y"])
    except Exception:
        return None


def _hand_scale(pts: Optional[List[Dict[str, float]]]) -> float:
    if not pts or len(pts) < 5:
        return 0.0
    c = _wrist_xy(pts)
    if c is None:
        return 0.0
    cx, cy = c
    d = []
    for j in range(1, min(MP_HAND_NUM_LANDMARKS, len(pts))):
        try:
            px, py = float(pts[j]["x"]), float(pts[j]["y"])
        except Exception:
            continue
        d.append(math.hypot(px - cx, py - cy))
    if not d:
        return 0.0
    d.sort()
    return d[len(d) // 2]


def _mean_l2_xy(a: Optional[List[Dict[str, float]]], b: Optional[List[Dict[str, float]]]) -> float:
    if not a or not b:
        return float("inf")
    L = min(len(a), len(b), MP_HAND_NUM_LANDMARKS)
    s = 0.0
    for j in range(L):
        try:
            ax, ay = float(a[j]["x"]), float(a[j]["y"])
            bx, by = float(b[j]["x"]), float(b[j]["y"])
        except Exception:
            continue
        s += math.hypot(ax - bx, ay - by)
    return s / float(L) if L else float("inf")


def _count_outliers(frames_list: List[Dict[str, Any]], hand_key: str) -> int:
    count = 0
    prev = None
    for fr in frames_list:
        cur = fr.get(hand_key)
        if cur is None:
            prev = None
            continue
        if prev is not None:
            dist = _mean_l2_xy(cur, prev)
            prev_scale = _hand_scale(prev)
            cur_scale = _hand_scale(cur)
            denom = max(prev_scale, 1e-6)
            jump = dist / denom
            scale_ratio = None
            if prev_scale > 0.0 and cur_scale > 0.0:
                scale_ratio = max(prev_scale, cur_scale) / min(prev_scale, cur_scale)
            if jump > 2.5 or (scale_ratio is not None and scale_ratio > 1.6):
                count += 1
        prev = cur
    return count


EVAL_KEYS = {
    "missing_frames_1",
    "missing_frames_2",
    "occluded_frames_1",
    "occluded_frames_2",
    "swap_frames",
    "dedup_trigger_frames",
    "sp_attempt_frames_1",
    "sp_attempt_frames_2",
    "sp_recovered_frames_1",
    "sp_recovered_frames_2",
    "hold_frames_1",
    "hold_frames_2",
    "outlier_frames_1",
    "outlier_frames_2",
    "sanity_reject_frames_1",
    "sanity_reject_frames_2",
    "missing_gap_p50_1",
    "missing_gap_p90_1",
    "missing_gap_max_1",
    "missing_gap_p50_2",
    "missing_gap_p90_2",
    "missing_gap_max_2",
    "occluded_gap_p50_1",
    "occluded_gap_p90_1",
    "occluded_gap_max_1",
    "occluded_gap_p50_2",
    "occluded_gap_p90_2",
    "occluded_gap_max_2",
    "pp_filled_left",
    "pp_filled_right",
    "pp_gaps_filled_left",
    "pp_gaps_filled_right",
    "pp_smoothing_delta_left",
    "pp_smoothing_delta_right",
}


def _norm_path(path: Path, base: Optional[Path]) -> str:
    try:
        if base:
            rel = path.resolve().relative_to(base.resolve())
            return str(rel).replace("/", "\\")
    except Exception:
        pass
    return str(path).replace("/", "\\")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _frames_meta(payload: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if isinstance(payload, dict):
        frames = payload.get("frames", []) or []
        meta = payload.get("meta", {}) or {}
    elif isinstance(payload, list):
        frames = payload
        meta = {}
    else:
        frames, meta = [], {}
    frames = [fr for fr in frames if isinstance(fr, dict)]
    return frames, meta


def _mean(vals: List[float]) -> float:
    return float(statistics.mean(vals)) if vals else 0.0


def _median(vals: List[float]) -> float:
    return float(statistics.median(vals)) if vals else 0.0


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except Exception:
        return default


def _percentile_sorted(sorted_vals: List[int], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def _gap_stats(frames: List[Dict[str, Any]], pred) -> Tuple[float, float, int]:
    gaps: List[int] = []
    cur = 0
    for fr in frames:
        if pred(fr):
            cur += 1
        else:
            if cur > 0:
                gaps.append(cur)
                cur = 0
    if cur > 0:
        gaps.append(cur)
    if not gaps:
        return 0.0, 0.0, 0
    gaps_sorted = sorted(gaps)
    return (
        _percentile_sorted(gaps_sorted, 50),
        _percentile_sorted(gaps_sorted, 90),
        int(max(gaps)),
    )


def _is_occluded(fr: Dict[str, Any], hand_idx: int) -> bool:
    key = f"occluded_{hand_idx}"
    if key in fr:
        return bool(fr.get(key))
    state = fr.get(f"hand {hand_idx}_state")
    if state is not None:
        return str(state) == "occluded"
    source = fr.get(f"hand {hand_idx}_source")
    if source is not None:
        return str(source) == "occluded"
    return False


def _is_track_ok(fr: Dict[str, Any], hand_idx: int) -> bool:
    key = f"hand_{hand_idx}_track_ok"
    if key in fr:
        return bool(fr.get(key))
    source = fr.get(f"hand {hand_idx}_source")
    if source is not None:
        return str(source) == "tracked"
    return False


def _is_sp_attempt(fr: Dict[str, Any], hand_idx: int) -> bool:
    key = f"hand_{hand_idx}_sp_attempted"
    if key in fr:
        return bool(fr.get(key))
    return False


def _is_sp_recovered(fr: Dict[str, Any], hand_idx: int) -> bool:
    key = f"hand_{hand_idx}_sp_recovered"
    if key in fr:
        return bool(fr.get(key))
    return False


def _is_sanity_reject(reason: Any) -> bool:
    if not reason:
        return False
    return "sanity" in str(reason).lower()


def _extract_hand_score_hi(frames: List[Dict[str, Any]]) -> float:
    for fr in frames:
        val = fr.get("hand_score_hi")
        if val is None:
            continue
        try:
            return float(val)
        except Exception:
            continue
    return 0.90


def _compute_pp_stats(
    frames: List[Dict[str, Any]],
    meta: Dict[str, Any],
    pp_payload: Optional[Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if pp_payload is None:
        return None, None
    pp_frames, pp_meta = _frames_meta(pp_payload)
    post_meta = {}
    if isinstance(pp_meta, dict):
        post_meta = pp_meta.get("postprocess", {}) or {}
    enabled = post_meta.get("enabled", True)
    if not enabled:
        return None, None

    # Try to recompute pp stats using the original postprocess params.
    hi = _extract_hand_score_hi(frames)
    max_gap = int(post_meta.get("max_gap", 15))
    smoother = str(post_meta.get("smoother", "ema"))
    only_anchors = bool(post_meta.get("only_anchors", True))
    coords = str(meta.get("coords") or "").lower()
    world_coords = coords.startswith("world")
    try:
        _, stats = postprocess_sequence(
            frames,
            hi=hi,
            max_gap=max_gap,
            smoother=smoother,
            only_anchors=only_anchors,
            world_coords=world_coords,
        )
        return stats, None
    except Exception as exc:
        # Fallback: compute simple diffs vs pp frames (approximate).
        stats = _compare_pp_stats(frames, pp_frames)
        return stats, f"postprocess_sequence_failed: {exc}"


def _wrist(pts: Any) -> Optional[Tuple[float, float, float]]:
    if not isinstance(pts, list) or not pts:
        return None
    p0 = pts[0]
    if not isinstance(p0, dict):
        return None
    try:
        return (float(p0.get("x", 0.0)), float(p0.get("y", 0.0)), float(p0.get("z", 0.0)))
    except Exception:
        return None


def _count_gaps(mask: List[bool]) -> int:
    gaps = 0
    cur = 0
    for m in mask:
        if m:
            cur += 1
        else:
            if cur > 0:
                gaps += 1
                cur = 0
    if cur > 0:
        gaps += 1
    return gaps


def _compare_pp_stats(
    raw_frames: List[Dict[str, Any]],
    pp_frames: List[Dict[str, Any]],
) -> Dict[str, Any]:
    n = min(len(raw_frames), len(pp_frames))
    filled_left_mask: List[bool] = []
    filled_right_mask: List[bool] = []
    delta_sum_left = 0.0
    delta_cnt_left = 0
    delta_sum_right = 0.0
    delta_cnt_right = 0

    for i in range(n):
        raw = raw_frames[i]
        pp = pp_frames[i]
        raw_left = raw.get("hand 1")
        pp_left = pp.get("hand 1")
        raw_right = raw.get("hand 2")
        pp_right = pp.get("hand 2")

        filled_left = raw_left is None and pp_left is not None
        filled_right = raw_right is None and pp_right is not None
        filled_left_mask.append(bool(filled_left))
        filled_right_mask.append(bool(filled_right))

        if raw_left is not None and pp_left is not None:
            w0 = _wrist(raw_left)
            w1 = _wrist(pp_left)
            if w0 and w1:
                dx = w1[0] - w0[0]
                dy = w1[1] - w0[1]
                dz = w1[2] - w0[2]
                delta_sum_left += math.sqrt(dx * dx + dy * dy + dz * dz)
                delta_cnt_left += 1

        if raw_right is not None and pp_right is not None:
            w0 = _wrist(raw_right)
            w1 = _wrist(pp_right)
            if w0 and w1:
                dx = w1[0] - w0[0]
                dy = w1[1] - w0[1]
                dz = w1[2] - w0[2]
                delta_sum_right += math.sqrt(dx * dx + dy * dy + dz * dz)
                delta_cnt_right += 1

    return {
        "pp_filled_left": int(sum(1 for v in filled_left_mask if v)),
        "pp_filled_right": int(sum(1 for v in filled_right_mask if v)),
        "pp_gaps_filled_left": int(_count_gaps(filled_left_mask)),
        "pp_gaps_filled_right": int(_count_gaps(filled_right_mask)),
        "pp_smoothing_delta_left": float(delta_sum_left / delta_cnt_left) if delta_cnt_left else 0.0,
        "pp_smoothing_delta_right": float(delta_sum_right / delta_cnt_right) if delta_cnt_right else 0.0,
    }


def _process_one(args: Tuple[str, str, str, bool, bool]) -> Dict[str, Any]:
    raw_path_str, base_dir_str, pp_suffix, compute_eval, compute_pp = args
    raw_path = Path(raw_path_str)
    base_dir = Path(base_dir_str) if base_dir_str else None
    try:
        payload = _load_json(raw_path)
        frames, meta = _frames_meta(payload)
    except Exception as exc:
        return {"ok": False, "path": raw_path_str, "error": str(exc)}

    frames_total = max(1, len(frames))
    fps = _safe_float(meta.get("fps"), 0.0)

    hands_present = sum(
        1 for fr in frames
        if fr.get("hand 1") is not None or fr.get("hand 2") is not None
    )
    left_cov = sum(1 for fr in frames if fr.get("hand 1") is not None) / frames_total
    right_cov = sum(1 for fr in frames if fr.get("hand 2") is not None) / frames_total
    both_cov = sum(
        1 for fr in frames
        if fr.get("hand 1") is not None and fr.get("hand 2") is not None
    ) / frames_total
    pose_cov = sum(1 for fr in frames if fr.get("pose") is not None) / frames_total
    pose_interp_frac = sum(1 for fr in frames if fr.get("pose_interpolated")) / frames_total

    left_scores: List[float] = []
    right_scores: List[float] = []
    for fr in frames:
        v1 = fr.get("hand 1_score")
        if v1 is not None:
            try:
                left_scores.append(float(v1))
            except Exception:
                pass
        v2 = fr.get("hand 2_score")
        if v2 is not None:
            try:
                right_scores.append(float(v2))
            except Exception:
                pass
    left_mean = _mean(left_scores)
    right_mean = _mean(right_scores)
    left_med = _median(left_scores)
    right_med = _median(right_scores)

    dts: List[float] = []
    for fr in frames:
        v = fr.get("dt")
        try:
            if v is not None and float(v) > 0:
                dts.append(float(v))
        except Exception:
            pass
    dt_median = _median(dts) if dts else 0.0
    fps_est = (1000.0 / dt_median) if dt_median > 0 else fps

    sp_recovered_left = sum(1 for fr in frames if _is_sp_recovered(fr, 1))
    sp_recovered_right = sum(1 for fr in frames if _is_sp_recovered(fr, 2))
    sp_recovered_left_frac = float(sp_recovered_left) / float(frames_total)
    sp_recovered_right_frac = float(sp_recovered_right) / float(frames_total)

    track_recovered_left = sum(1 for fr in frames if _is_track_ok(fr, 1))
    track_recovered_right = sum(1 for fr in frames if _is_track_ok(fr, 2))
    track_recovered_left_frac = float(track_recovered_left) / float(frames_total)
    track_recovered_right_frac = float(track_recovered_right) / float(frames_total)

    outlier_frames_1 = _count_outliers(frames, "hand 1") if compute_eval else 0
    outlier_frames_2 = _count_outliers(frames, "hand 2") if compute_eval else 0

    quality_score = (
        0.4 * both_cov +
        0.2 * (left_cov + right_cov) / 2.0 +
        0.2 * pose_cov +
        0.2 * (left_med + right_med) / 2.0
    )

    manifest_entry: Dict[str, Any] = {
        "id": raw_path.stem,
        "file": _norm_path(raw_path, base_dir),
        "num_frames": int(len(frames)),
        "fps": float(fps),
        "hands_frames": int(hands_present),
        "hands_coverage": round(hands_present / frames_total, 4),
        "left_score_mean": round(left_mean, 4),
        "right_score_mean": round(right_mean, 4),
        "left_coverage": round(left_cov, 4),
        "right_coverage": round(right_cov, 4),
        "both_coverage": round(both_cov, 4),
        "pose_coverage": round(pose_cov, 4),
        "pose_interpolated_frac": round(pose_interp_frac, 4),
        "dt_median_ms": round(dt_median, 2),
        "fps_est": round(fps_est, 3),
        "left_score_median": round(left_med, 4),
        "right_score_median": round(right_med, 4),
        "quality_score": round(float(quality_score), 4),
        "sp_recovered_left_frac": round(sp_recovered_left_frac, 4),
        "sp_recovered_right_frac": round(sp_recovered_right_frac, 4),
        "track_recovered_left_frac": round(track_recovered_left_frac, 4),
        "track_recovered_right_frac": round(track_recovered_right_frac, 4),
    }

    if compute_eval:
        missing_frames_1 = sum(1 for fr in frames if fr.get("hand 1") is None)
        missing_frames_2 = sum(1 for fr in frames if fr.get("hand 2") is None)
        occluded_frames_1 = sum(1 for fr in frames if _is_occluded(fr, 1))
        occluded_frames_2 = sum(1 for fr in frames if _is_occluded(fr, 2))
        swap_frames = sum(1 for fr in frames if fr.get("swap_applied"))
        dedup_trigger_frames = sum(1 for fr in frames if fr.get("dedup_triggered"))
        sp_attempt_frames_1 = sum(1 for fr in frames if _is_sp_attempt(fr, 1))
        sp_attempt_frames_2 = sum(1 for fr in frames if _is_sp_attempt(fr, 2))
        sp_recovered_frames_1 = sp_recovered_left
        sp_recovered_frames_2 = sp_recovered_right
        hold_frames_1 = track_recovered_left
        hold_frames_2 = track_recovered_right
        sanity_reject_frames_1 = sum(
            1 for fr in frames if _is_sanity_reject(fr.get("hand 1_reject_reason"))
        )
        sanity_reject_frames_2 = sum(
            1 for fr in frames if _is_sanity_reject(fr.get("hand 2_reject_reason"))
        )

        missing_gap_p50_1, missing_gap_p90_1, missing_gap_max_1 = _gap_stats(
            frames, lambda fr: fr.get("hand 1") is None
        )
        missing_gap_p50_2, missing_gap_p90_2, missing_gap_max_2 = _gap_stats(
            frames, lambda fr: fr.get("hand 2") is None
        )
        occluded_gap_p50_1, occluded_gap_p90_1, occluded_gap_max_1 = _gap_stats(
            frames, lambda fr: _is_occluded(fr, 1)
        )
        occluded_gap_p50_2, occluded_gap_p90_2, occluded_gap_max_2 = _gap_stats(
            frames, lambda fr: _is_occluded(fr, 2)
        )

        manifest_entry.update({
            "missing_frames_1": int(missing_frames_1),
            "missing_frames_2": int(missing_frames_2),
            "occluded_frames_1": int(occluded_frames_1),
            "occluded_frames_2": int(occluded_frames_2),
            "swap_frames": int(swap_frames),
            "dedup_trigger_frames": int(dedup_trigger_frames),
            "sp_attempt_frames_1": int(sp_attempt_frames_1),
            "sp_attempt_frames_2": int(sp_attempt_frames_2),
            "sp_recovered_frames_1": int(sp_recovered_frames_1),
            "sp_recovered_frames_2": int(sp_recovered_frames_2),
            "hold_frames_1": int(hold_frames_1),
            "hold_frames_2": int(hold_frames_2),
            "outlier_frames_1": int(outlier_frames_1),
            "outlier_frames_2": int(outlier_frames_2),
            "sanity_reject_frames_1": int(sanity_reject_frames_1),
            "sanity_reject_frames_2": int(sanity_reject_frames_2),
            "missing_gap_p50_1": float(missing_gap_p50_1),
            "missing_gap_p90_1": float(missing_gap_p90_1),
            "missing_gap_max_1": int(missing_gap_max_1),
            "missing_gap_p50_2": float(missing_gap_p50_2),
            "missing_gap_p90_2": float(missing_gap_p90_2),
            "missing_gap_max_2": int(missing_gap_max_2),
            "occluded_gap_p50_1": float(occluded_gap_p50_1),
            "occluded_gap_p90_1": float(occluded_gap_p90_1),
            "occluded_gap_max_1": int(occluded_gap_max_1),
            "occluded_gap_p50_2": float(occluded_gap_p50_2),
            "occluded_gap_p90_2": float(occluded_gap_p90_2),
            "occluded_gap_max_2": int(occluded_gap_max_2),
        })

    pp_stats = None
    pp_error = None
    pp_path = raw_path.with_name(f"{raw_path.stem}{pp_suffix}.json")
    if compute_pp and pp_path.exists():
        try:
            pp_payload = _load_json(pp_path)
        except Exception as exc:
            pp_payload = None
            pp_error = str(exc)
        if pp_payload is not None:
            pp_stats, pp_error = _compute_pp_stats(frames, meta, pp_payload)
        if pp_stats is None:
            pp_stats = {
                "pp_filled_left": 0,
                "pp_filled_right": 0,
                "pp_gaps_filled_left": 0,
                "pp_gaps_filled_right": 0,
                "pp_smoothing_delta_left": 0.0,
                "pp_smoothing_delta_right": 0.0,
            }
        manifest_entry["file_pp"] = _norm_path(pp_path, base_dir)
        manifest_entry["pp_filled_left"] = int(pp_stats.get("pp_filled_left", 0))
        manifest_entry["pp_filled_right"] = int(pp_stats.get("pp_filled_right", 0))
        manifest_entry["pp_gaps_filled_left"] = int(pp_stats.get("pp_gaps_filled_left", 0))
        manifest_entry["pp_gaps_filled_right"] = int(pp_stats.get("pp_gaps_filled_right", 0))
        manifest_entry["pp_smoothing_delta_left"] = float(pp_stats.get("pp_smoothing_delta_left", 0.0))
        manifest_entry["pp_smoothing_delta_right"] = float(pp_stats.get("pp_smoothing_delta_right", 0.0))

    input_path = ""
    if isinstance(meta, dict):
        input_path = str(meta.get("video") or "")

    return {
        "ok": True,
        "manifest": manifest_entry,
        "input": input_path,
        "pp_error": pp_error,
        "path": raw_path_str,
    }


def _safe_rate(num: Optional[float], denom: Optional[float]) -> Optional[float]:
    if num is None:
        return None
    if denom is None or denom <= 0:
        return None
    try:
        return float(num) / float(denom)
    except Exception:
        return None


def _manifest_sort_key(item: Dict[str, Any]) -> tuple:
    seg_uid = item.get("seg_uid")
    if seg_uid:
        return (str(seg_uid), str(item.get("file", "")))
    slug = item.get("slug")
    if slug:
        return (str(slug), str(item.get("file", "")))
    return (str(item.get("id", "")), str(item.get("file", "")))


def _stem_from_path_str(raw: str) -> str:
    if not raw:
        return ""
    return Path(raw.replace("\\", "/")).stem


def _build_eval_report(
    manifest: List[Dict[str, Any]],
    output_to_input: Dict[str, str],
    args: argparse.Namespace,
    out_path: Path,
) -> Dict[str, Any]:
    quality_scores: List[float] = []
    swap_rates: List[float] = []
    missing_rates: List[float] = []
    outlier_rates: List[float] = []
    sanity_rates: List[float] = []
    pp_filled_fracs: List[float] = []
    pp_smoothing_deltas: List[float] = []
    videos_report: List[Dict[str, Any]] = []

    for entry in manifest:
        entry_id = entry.get("id")
        output_path = entry.get("file", "")
        slug = entry.get("slug")
        if not slug and output_path:
            slug = _stem_from_path_str(output_path)

        input_path = output_to_input.get(output_path) or ""

        eval_part = {k: entry[k] for k in EVAL_KEYS if k in entry}
        meta_part = {
            k: v for k, v in entry.items()
            if k not in EVAL_KEYS and k not in ("id", "file")
        }

        num_frames = entry.get("num_frames") or 0
        q = entry.get("quality_score")
        if q is not None:
            quality_scores.append(float(q))

        if eval_part:
            swap_rate = _safe_rate(eval_part.get("swap_frames"), num_frames)
            missing_rate = _safe_rate(
                (eval_part.get("missing_frames_1", 0) + eval_part.get("missing_frames_2", 0)),
                2 * num_frames,
            )
            outlier_rate = _safe_rate(
                (eval_part.get("outlier_frames_1", 0) + eval_part.get("outlier_frames_2", 0)),
                2 * num_frames,
            )
            sanity_rate = _safe_rate(
                (eval_part.get("sanity_reject_frames_1", 0) + eval_part.get("sanity_reject_frames_2", 0)),
                2 * num_frames,
            )
            if swap_rate is not None:
                swap_rates.append(swap_rate)
            if missing_rate is not None:
                missing_rates.append(missing_rate)
            if outlier_rate is not None:
                outlier_rates.append(outlier_rate)
            if sanity_rate is not None:
                sanity_rates.append(sanity_rate)

            pp_filled = (
                eval_part.get("pp_filled_left", 0) + eval_part.get("pp_filled_right", 0)
            )
            pp_filled_frac = _safe_rate(pp_filled, 2 * num_frames)
            if pp_filled_frac is not None:
                pp_filled_fracs.append(pp_filled_frac)

            pp_delta_vals = []
            if eval_part.get("pp_smoothing_delta_left") is not None:
                pp_delta_vals.append(float(eval_part.get("pp_smoothing_delta_left")))
            if eval_part.get("pp_smoothing_delta_right") is not None:
                pp_delta_vals.append(float(eval_part.get("pp_smoothing_delta_right")))
            if pp_delta_vals:
                pp_smoothing_deltas.append(sum(pp_delta_vals) / len(pp_delta_vals))

        video_item = {
            "id": entry_id,
            "slug": slug,
            "input": input_path,
            "output": output_path,
            "meta": meta_part,
        }
        if eval_part:
            video_item["eval"] = eval_part
        videos_report.append(video_item)

    videos_report.sort(
        key=lambda v: (str(v.get("slug") or v.get("id") or ""), str(v.get("output") or ""))
    )

    def _mean(vals: List[float]) -> Optional[float]:
        return (sum(vals) / len(vals)) if vals else None

    aggregate = {
        "quality_score": _mean(quality_scores),
        "swap_rate": _mean(swap_rates),
        "missing_rate": _mean(missing_rates),
        "outlier_rate": _mean(outlier_rates),
        "sanity_reject_rate": _mean(sanity_rates),
        "pp_filled_frac": _mean(pp_filled_fracs),
        "pp_smoothing_delta": _mean(pp_smoothing_deltas),
    }

    versions = {"python": sys.version}
    try:
        import cv2  # type: ignore
        versions["cv2"] = cv2.__version__
    except Exception:
        versions["cv2"] = None
    try:
        import mediapipe as mp  # type: ignore
        versions["mediapipe"] = getattr(mp, "__version__", None)
    except Exception:
        versions["mediapipe"] = None

    run_meta = {
        "args": vars(args),
        "jobs": int(args.jobs),
        "seed": None,
        "versions": versions,
    }

    report = {
        "run": run_meta,
        "videos": videos_report,
        "aggregate": aggregate,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
    return report


def _discover_raw_files(skeletons_dir: Path) -> List[Path]:
    raw_files: List[Path] = []
    for path in skeletons_dir.rglob("*.json"):
        name = path.name.lower()
        if name.endswith("_pp.json"):
            continue
        if name in {"manifest.json", "eval_report.json"}:
            continue
        raw_files.append(path)
    return raw_files


def main() -> int:
    ap = argparse.ArgumentParser("Rebuild manifest + eval_report from existing keypoints JSONs.")
    ap.add_argument("--skeletons-dir", default="datasets/skeletons", help="Root folder with keypoints JSONs.")
    ap.add_argument("--manifest-out", default="", help="Path to output manifest.json.")
    ap.add_argument("--eval-report", default="", help="Path to output eval_report.json (optional).")
    ap.add_argument("--path-base", default="datasets", help="Base path for manifest file paths (empty to use absolute).")
    ap.add_argument("--pp-suffix", default="_pp", help="Suffix for postprocess files (default: _pp).")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel workers.")
    ap.add_argument("--no-eval", action="store_true", help="Skip eval metrics + eval_report.")
    ap.add_argument("--no-pp", action="store_true", help="Skip pp stats even if _pp files exist.")
    ap.add_argument("--limit", type=int, default=0, help="Process only N files (debug).")
    ap.add_argument("--progress-every", type=int, default=250, help="Progress log interval.")
    args = ap.parse_args()

    skeletons_dir = Path(args.skeletons_dir)
    if not skeletons_dir.exists():
        print(f"[ERR] skeletons dir not found: {skeletons_dir}")
        return 2

    manifest_out = Path(args.manifest_out) if args.manifest_out else (skeletons_dir / "manifest.json")
    if args.eval_report:
        eval_out = Path(args.eval_report)
    else:
        default_eval = Path("outputs/Slovo/eval_report.json")
        eval_out = default_eval if default_eval.parent.exists() else Path("outputs/eval_report.json")

    base_dir = Path(args.path_base) if args.path_base else None

    raw_files = _discover_raw_files(skeletons_dir)
    if args.limit and args.limit > 0:
        raw_files = raw_files[: args.limit]

    print(f"[INFO] Found raw files: {len(raw_files)}")

    started = time.time()
    manifest: List[Dict[str, Any]] = []
    output_to_input: Dict[str, str] = {}
    errors: List[Tuple[str, str]] = []
    pp_errors: List[Tuple[str, str]] = []

    compute_eval = not args.no_eval
    compute_pp = not args.no_pp
    jobs = max(1, int(args.jobs))

    payloads = [
        (str(p), str(base_dir) if base_dir else "", args.pp_suffix, compute_eval, compute_pp)
        for p in raw_files
    ]

    if jobs <= 1:
        for idx, pl in enumerate(payloads, 1):
            res = _process_one(pl)
            if not res.get("ok"):
                errors.append((res.get("path", ""), res.get("error", "")))
            else:
                entry = res["manifest"]
                manifest.append(entry)
                if res.get("input"):
                    output_to_input[entry.get("file", "")] = res.get("input", "")
                if res.get("pp_error"):
                    pp_errors.append((res.get("path", ""), res.get("pp_error", "")))
            if args.progress_every and (idx % args.progress_every) == 0:
                print(f"[INFO] processed {idx}/{len(payloads)}")
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futures = [ex.submit(_process_one, pl) for pl in payloads]
            for idx, fut in enumerate(as_completed(futures), 1):
                try:
                    res = fut.result()
                except Exception as exc:
                    errors.append(("<worker>", str(exc)))
                    continue
                if not res.get("ok"):
                    errors.append((res.get("path", ""), res.get("error", "")))
                else:
                    entry = res["manifest"]
                    manifest.append(entry)
                    if res.get("input"):
                        output_to_input[entry.get("file", "")] = res.get("input", "")
                    if res.get("pp_error"):
                        pp_errors.append((res.get("path", ""), res.get("pp_error", "")))
                if args.progress_every and (idx % args.progress_every) == 0:
                    print(f"[INFO] processed {idx}/{len(payloads)}")

    manifest.sort(key=_manifest_sort_key)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with manifest_out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[OK] Manifest written to {manifest_out}")

    if compute_eval and eval_out:
        _build_eval_report(manifest, output_to_input, args, eval_out)
        print(f"[OK] Eval report written to {eval_out}")

    elapsed = time.time() - started
    print(f"[DONE] processed={len(manifest)} errors={len(errors)} pp_errors={len(pp_errors)} elapsed_sec={elapsed:.1f}")
    if errors:
        print("[WARN] sample errors:")
        for path, err in errors[:5]:
            print(f"  - {path}: {err}")
    if pp_errors:
        print("[WARN] sample pp errors:")
        for path, err in pp_errors[:5]:
            print(f"  - {path}: {err}")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
