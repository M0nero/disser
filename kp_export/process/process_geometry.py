from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math

from .process_constants import MP_HAND_NUM_LANDMARKS


def _wrist_xy(pts: Optional[List[Dict[str, float]]]) -> Optional[Tuple[float, float]]:
    if not pts or len(pts) == 0:
        return None
    return float(pts[0]["x"]), float(pts[0]["y"])


def _hand_scale(pts: Optional[List[Dict[str, float]]]) -> float:
    if not pts or len(pts) < 5:
        return 0.0
    c = _wrist_xy(pts)
    if c is None:
        return 0.0
    cx, cy = c
    d = []
    for j in range(1, min(MP_HAND_NUM_LANDMARKS, len(pts))):
        px, py = float(pts[j]["x"]), float(pts[j]["y"])
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
        ax, ay = float(a[j]["x"]), float(a[j]["y"])
        bx, by = float(b[j]["x"]), float(b[j]["y"])
        s += math.hypot(ax - bx, ay - by)
    return s / float(L)


def _bbox_norm(pts: List[Dict[str, float]]) -> Tuple[float, float, float, float]:
    xs = [float(p["x"]) for p in pts]
    ys = [float(p["y"]) for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _iou_norm(a: List[Dict[str, float]], b: List[Dict[str, float]]) -> float:
    ax0, ay0, ax1, ay1 = _bbox_norm(a)
    bx0, by0, bx1, by1 = _bbox_norm(b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 1e-9 else 0.0
