from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math

from .process_constants import (
    MP_POSE_LEFT_WRIST_IDX,
    MP_POSE_RIGHT_WRIST_IDX,
    POSE_WRIST_CROSS_SCALE,
    POSE_WRIST_MIN_VIS,
    POSE_WRIST_OFF_MARGIN,
    POSE_WRIST_REJECT_MIN_IMAGE,
    POSE_WRIST_REJECT_MIN_WORLD,
)
from .process_geometry import _hand_scale, _wrist_xy
from ..algos.tracking import HandTracker
from ..core.utils import bbox_from_pts_px


def _pose_wrist_refs(world_coords, pose_img_landmarks, pose_world_full, last_pose_world_full):
    if world_coords:
        pw = pose_world_full or last_pose_world_full
        if isinstance(pw, list) and len(pw) >= 17:
            lw = pw[MP_POSE_LEFT_WRIST_IDX]
            rw = pw[MP_POSE_RIGHT_WRIST_IDX]
            return (float(lw["x"]), float(lw["y"])), (float(rw["x"]), float(rw["y"]))
        return None, None
    else:
        if pose_img_landmarks is not None and len(pose_img_landmarks) >= 17:
            lw = pose_img_landmarks[MP_POSE_LEFT_WRIST_IDX]
            rw = pose_img_landmarks[MP_POSE_RIGHT_WRIST_IDX]
            return (float(lw.x), float(lw.y)), (float(rw.x), float(rw.y))
        return None, None


def _pose_wrist_out_of_frame(hand: str, pose_img_landmarks, margin: float = POSE_WRIST_OFF_MARGIN) -> bool:
    if pose_img_landmarks is None or len(pose_img_landmarks) < 17:
        return False
    idx = MP_POSE_LEFT_WRIST_IDX if hand == "left" else MP_POSE_RIGHT_WRIST_IDX
    lm = pose_img_landmarks[idx]
    x = float(lm.x)
    y = float(lm.y)
    return (x < -margin) or (x > 1.0 + margin) or (y < -margin) or (y > 1.0 + margin)


def _pose_wrist_low_visibility(hand: str, pose_img_landmarks, min_vis: float = POSE_WRIST_MIN_VIS) -> bool:
    if pose_img_landmarks is None or len(pose_img_landmarks) < 17:
        return False
    idx = MP_POSE_LEFT_WRIST_IDX if hand == "left" else MP_POSE_RIGHT_WRIST_IDX
    lm = pose_img_landmarks[idx]
    vis = float(getattr(lm, "visibility", 1.0))
    return vis < min_vis


def _pose_gate_allows_second_pass(hand: str, pose_img_landmarks) -> bool:
    if pose_img_landmarks is None:
        return True
    if _pose_wrist_out_of_frame(hand, pose_img_landmarks):
        return False
    if _pose_wrist_low_visibility(hand, pose_img_landmarks):
        return False
    return True


def _pose_wrist_px(hand: str, proc_w: int, proc_h: int, pose_img_landmarks) -> Optional[Tuple[float, float]]:
    if pose_img_landmarks is None or len(pose_img_landmarks) < 17:
        return None
    if _pose_wrist_out_of_frame(hand, pose_img_landmarks) or _pose_wrist_low_visibility(hand, pose_img_landmarks):
        return None
    idx = MP_POSE_LEFT_WRIST_IDX if hand == "left" else MP_POSE_RIGHT_WRIST_IDX
    lm = pose_img_landmarks[idx]
    return float(lm.x) * proc_w, float(lm.y) * proc_h


def _center_hint_for(
    hand: str,
    proc_w: int,
    proc_h: int,
    pose_img_landmarks,
    tracker: HandTracker,
    tracker_ready: bool,
    last_good_px,
) -> Optional[Tuple[float, float]]:
    if tracker_ready and getattr(tracker, "last_valid_landmarks", None):
        lms = tracker.last_valid_landmarks
        if lms and len(lms) > 0:
            try:
                wx = float(lms[0]["x"]) * proc_w
                wy = float(lms[0]["y"]) * proc_h
                return wx, wy
            except Exception:
                pass
    pose_xy = _pose_wrist_px(hand, proc_w, proc_h, pose_img_landmarks)
    if pose_xy is not None:
        return pose_xy
    if last_good_px is not None:
        x0, y0, x1, y1 = bbox_from_pts_px(last_good_px)
        return 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    return None


def _hand_expected(hand: str, pose_img_landmarks, tracker_ready: bool, last_good_px) -> bool:
    if tracker_ready:
        return True
    if pose_img_landmarks is not None and not _pose_wrist_out_of_frame(hand, pose_img_landmarks):
        if not _pose_wrist_low_visibility(hand, pose_img_landmarks):
            return True
    return last_good_px is not None


def _pose_distance_quality(
    hand: str,
    hand_pts,
    world_coords: bool,
    pose_img_landmarks,
    pose_world_full,
    last_pose_world_full,
) -> Optional[float]:
    if hand_pts is None or pose_img_landmarks is None:
        return None
    if _pose_wrist_out_of_frame(hand, pose_img_landmarks) or _pose_wrist_low_visibility(hand, pose_img_landmarks):
        return None
    Lref, Rref = _pose_wrist_refs(world_coords, pose_img_landmarks, pose_world_full, last_pose_world_full)
    ref = Lref if hand == "left" else Rref
    if ref is None:
        return None
    wrist_xy = _wrist_xy(hand_pts)
    if wrist_xy is None:
        return None
    dist = math.hypot(wrist_xy[0] - ref[0], wrist_xy[1] - ref[1])
    min_scale = POSE_WRIST_REJECT_MIN_WORLD if world_coords else POSE_WRIST_REJECT_MIN_IMAGE
    scale = max(_hand_scale(hand_pts), min_scale)
    if scale <= 0.0:
        return None
    dist_norm = dist / scale
    return 1.0 / (1.0 + dist_norm)


def _pose_wrist_dists(
    hand_pts,
    world_coords: bool,
    pose_img_landmarks,
    pose_world_full,
    last_pose_world_full,
) -> Tuple[Optional[float], Optional[float]]:
    if hand_pts is None or pose_img_landmarks is None:
        return None, None
    if _pose_wrist_out_of_frame("left", pose_img_landmarks) or _pose_wrist_out_of_frame("right", pose_img_landmarks):
        return None, None
    if _pose_wrist_low_visibility("left", pose_img_landmarks) or _pose_wrist_low_visibility("right", pose_img_landmarks):
        return None, None
    Lref, Rref = _pose_wrist_refs(world_coords, pose_img_landmarks, pose_world_full, last_pose_world_full)
    if Lref is None or Rref is None:
        return None, None
    wrist_xy = _wrist_xy(hand_pts)
    if wrist_xy is None:
        return None, None
    dL = math.hypot(wrist_xy[0] - Lref[0], wrist_xy[1] - Lref[1])
    dR = math.hypot(wrist_xy[0] - Rref[0], wrist_xy[1] - Rref[1])
    return dL, dR


def _pose_wrists_close(
    world_coords: bool,
    pose_img_landmarks,
    pose_world_full,
    last_pose_world_full,
    pts_candidates: Optional[List[Optional[List[Dict[str, float]]]]] = None,
) -> bool:
    Lref, Rref = _pose_wrist_refs(world_coords, pose_img_landmarks, pose_world_full, last_pose_world_full)
    if Lref is None or Rref is None:
        return False
    dist = math.hypot(Lref[0] - Rref[0], Lref[1] - Rref[1])
    scale = 0.0
    if pts_candidates:
        for pts in pts_candidates:
            scale = max(scale, _hand_scale(pts))
    min_scale = POSE_WRIST_REJECT_MIN_WORLD if world_coords else POSE_WRIST_REJECT_MIN_IMAGE
    scale = max(scale, min_scale)
    return dist < (POSE_WRIST_CROSS_SCALE * scale)
