from __future__ import annotations
from typing import Dict, List, Optional

from .process_constants import (
    MP_POSE_LEFT_WRIST_IDX,
    MP_POSE_RIGHT_WRIST_IDX,
    OCCLUSION_IOU_THRESHOLD_WORLD,
    OCCLUSION_Z_THRESHOLD_WORLD,
)
from .process_geometry import _iou_norm


def _resolve_pose_world(world_coords, pose_world_current, last_pose_world_full):
    return pose_world_current or last_pose_world_full


def is_hand_occluded(
    hand_pts: List[Dict[str, float]],
    other_pts: List[Dict[str, float]],
    hand_wrist_z: Optional[float],
    other_wrist_z: Optional[float],
    iou_threshold: float = OCCLUSION_IOU_THRESHOLD_WORLD,
    z_threshold: float = OCCLUSION_Z_THRESHOLD_WORLD,  # default tuned for world coords; override for image coords
    iou_pts: Optional[List[Dict[str, float]]] = None,
    other_iou_pts: Optional[List[Dict[str, float]]] = None,
) -> bool:
    """
    Returns True if hand_pts is occluded by other_pts.
    Occlusion is defined as:
    1. Significant 2D overlap (IoU > threshold)
    2. hand_pts is physically behind other_pts (Z value is larger)
    `z_threshold` and `iou_threshold` should be tuned depending on coordinate space (world vs image).
    """
    if not hand_pts or not other_pts or hand_wrist_z is None or other_wrist_z is None:
        return False

    # Prefer pixel-space overlap if provided, otherwise use input points
    pts_a = iou_pts if iou_pts else hand_pts
    pts_b = other_iou_pts if other_iou_pts else other_pts
    if not pts_a or not pts_b:
        return False

    # Check 2D overlap
    iou = _iou_norm(pts_a, pts_b)
    if iou < iou_threshold:
        return False

    # Check depth ordering using wrist Z
    # MediaPipe Pose World Z: negative is in front of hip center, positive is behind.
    # So smaller Z (more negative) is closer to camera.
    # We want to know if 'hand' is BEHIND 'other'.
    # So hand_z > other_z
    if hand_wrist_z > (other_wrist_z + z_threshold):
        return True

    return False


def _pick_wrist_depth(
    pts: Optional[List[Dict[str, float]]],
    pose_img_landmarks,
    pose_world_full,
    hand: str,
    world_coords: bool,
) -> Optional[float]:
    """Choose best-available wrist depth for occlusion checks."""
    depth_hand = float(pts[0]["z"]) if pts and isinstance(pts, list) and len(pts) > 0 and "z" in pts[0] else None
    depth_pose_world = None
    depth_pose_img = None
    if pose_world_full and len(pose_world_full) >= 17:
        depth_pose_world = float(
            pose_world_full[MP_POSE_LEFT_WRIST_IDX]["z"]
            if hand == "left"
            else pose_world_full[MP_POSE_RIGHT_WRIST_IDX]["z"]
        )
    if pose_img_landmarks is not None and len(pose_img_landmarks) >= 17:
        depth_pose_img = float(
            pose_img_landmarks[MP_POSE_LEFT_WRIST_IDX].z
            if hand == "left"
            else pose_img_landmarks[MP_POSE_RIGHT_WRIST_IDX].z
        )

    if world_coords:
        return depth_hand if depth_hand is not None else depth_pose_world
    # image coords: prefer pose world Z if available, then pose image Z
    if depth_pose_world is not None:
        return depth_pose_world
    if depth_pose_img is not None:
        return depth_pose_img
    return depth_hand


def _overlap_iou(
    a_pts: Optional[List[Dict[str, float]]],
    b_pts: Optional[List[Dict[str, float]]],
    fallback: float = 0.0,
) -> float:
    if not a_pts or not b_pts:
        return fallback
    try:
        return _iou_norm(a_pts, b_pts)
    except Exception:
        return fallback
