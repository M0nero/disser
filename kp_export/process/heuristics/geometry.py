from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from ...core.geometry import bbox_norm, hand_scale, iou_norm, mean_l2_xy, wrist_xy
from .constants import MP_HAND_NUM_LANDMARKS


def _wrist_xy(pts: Optional[List[Dict[str, float]]]) -> Optional[Tuple[float, float]]:
    return wrist_xy(pts)


def _hand_scale(pts: Optional[List[Dict[str, float]]]) -> float:
    return hand_scale(pts, min_points=min(5, MP_HAND_NUM_LANDMARKS))


def _mean_l2_xy(a: Optional[List[Dict[str, float]]], b: Optional[List[Dict[str, float]]]) -> float:
    return mean_l2_xy(a, b, limit=MP_HAND_NUM_LANDMARKS)


def _bbox_norm(pts: List[Dict[str, float]]) -> Tuple[float, float, float, float]:
    return bbox_norm(pts)


def _iou_norm(a: List[Dict[str, float]], b: List[Dict[str, float]]) -> float:
    return iou_norm(a, b)
