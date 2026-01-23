from __future__ import annotations
from typing import Dict, List, Optional, TypedDict


class Landmark(TypedDict):
    x: float
    y: float
    z: float
    visibility: Optional[float]


class FrameData(TypedDict, total=False):
    ts: int
    dt: int
    hand_1: Optional[List[Dict[str, float]]]
    hand_1_score: Optional[float]
    hand_1_source: Optional[str]
    hand_1_state: Optional[str]
    hand_1_is_anchor: Optional[bool]
    hand_1_reject_reason: Optional[str]
    hand_2: Optional[List[Dict[str, float]]]
    hand_2_score: Optional[float]
    hand_2_source: Optional[str]
    hand_2_state: Optional[str]
    hand_2_is_anchor: Optional[bool]
    hand_2_reject_reason: Optional[str]
    pose: Optional[List[Dict[str, float]]]
    pose_vis: Optional[List[float]]
    pose_interpolated: bool
    hand_mask: Optional[List[int]]
    both_hands: Optional[int]
    hand_1_sp_roi_px: Optional[List[int]]
    hand_2_sp_roi_px: Optional[List[int]]
