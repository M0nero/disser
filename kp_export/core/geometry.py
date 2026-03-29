from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


def wrist_xy(pts: Optional[List[Dict[str, float]]]) -> Optional[Tuple[float, float]]:
    if not pts:
        return None
    try:
        return float(pts[0]["x"]), float(pts[0]["y"])
    except Exception:
        return None


def hand_scale(pts: Optional[List[Dict[str, float]]], *, min_points: int = 2) -> float:
    if not pts or len(pts) < max(2, int(min_points)):
        return 0.0
    wrist = wrist_xy(pts)
    if wrist is None:
        return 0.0
    wx, wy = wrist
    dists = []
    for point in pts[1:]:
        try:
            dx = float(point["x"]) - wx
            dy = float(point["y"]) - wy
        except Exception:
            continue
        dists.append(math.hypot(dx, dy))
    if not dists:
        return 0.0
    dists.sort()
    return float(dists[len(dists) // 2])


def mean_l2_xy(a: Optional[List[Dict[str, float]]], b: Optional[List[Dict[str, float]]], *, limit: Optional[int] = None) -> float:
    if not a or not b:
        return float("inf")
    size = min(len(a), len(b), int(limit) if limit is not None else min(len(a), len(b)))
    if size <= 0:
        return float("inf")
    total = 0.0
    for idx in range(size):
        ax, ay = float(a[idx]["x"]), float(a[idx]["y"])
        bx, by = float(b[idx]["x"]), float(b[idx]["y"])
        total += math.hypot(ax - bx, ay - by)
    return total / float(size)


def bbox_norm(pts: List[Dict[str, float]]) -> Tuple[float, float, float, float]:
    xs = [float(point["x"]) for point in pts]
    ys = [float(point["y"]) for point in pts]
    return min(xs), min(ys), max(xs), max(ys)


def iou_norm(a: List[Dict[str, float]], b: List[Dict[str, float]]) -> float:
    ax0, ay0, ax1, ay1 = bbox_norm(a)
    bx0, by0, bx1, by1 = bbox_norm(b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 1e-9 else 0.0
