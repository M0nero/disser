from __future__ import annotations

from typing import Any, Callable, List, Tuple

import numpy as np

from ..contracts import FrameRecord
from ..heuristics.geometry import _hand_scale, _mean_l2_xy


def diag(record: FrameRecord, key: str, default=None):
    return record.diagnostics.get(key, default)


def gap_stats(records: List[FrameRecord], pred: Callable[[FrameRecord], bool]) -> Tuple[float, float, int]:
    gaps: List[int] = []
    cur = 0
    for record in records:
        if pred(record):
            cur += 1
        else:
            if cur > 0:
                gaps.append(cur)
                cur = 0
    if cur > 0:
        gaps.append(cur)
    if not gaps:
        return 0.0, 0.0, 0
    return (
        float(np.percentile(gaps, 50)),
        float(np.percentile(gaps, 90)),
        int(max(gaps)),
    )


def mean_gap_len(records: List[FrameRecord], hand_idx: int) -> float:
    gaps: List[int] = []
    cur = 0
    for record in records:
        hand = record.hand_1 if hand_idx == 1 else record.hand_2
        if hand.landmarks is None:
            cur += 1
        else:
            if cur > 0:
                gaps.append(cur)
                cur = 0
    if cur > 0:
        gaps.append(cur)
    return float(np.mean(gaps)) if gaps else 0.0


def normalize_hand_key(hand_key: str) -> str:
    value = str(hand_key).strip().lower().replace(" ", "_")
    if value in {"hand_1", "1", "left"}:
        return "hand_1"
    if value in {"hand_2", "2", "right"}:
        return "hand_2"
    raise ValueError(f"Unsupported hand key: {hand_key}")


def get_hand_points(frame: Any, hand_key: str):
    key = normalize_hand_key(hand_key)
    if hasattr(frame, "hand_1") and hasattr(frame, "hand_2"):
        return frame.hand_1.landmarks if key == "hand_1" else frame.hand_2.landmarks
    legacy_key = "hand 1" if key == "hand_1" else "hand 2"
    return frame.get(legacy_key)


def count_outliers(frames_list: List[Any], hand_key: str) -> int:
    count = 0
    prev = None
    for fr in frames_list:
        cur = get_hand_points(fr, hand_key)
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


def is_sanity_reject(reason) -> bool:
    if not reason:
        return False
    return "sanity:" in str(reason)
