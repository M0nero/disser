from __future__ import annotations
from typing import Any, Dict, List

from .process_geometry import _hand_scale, _mean_l2_xy


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
