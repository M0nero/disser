from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math

from .constants import (
    DEDUP_D0_SCALE,
    DEDUP_DIST_SCALE,
    DEDUP_IOU_THRESHOLD,
    POSE_WRIST_REJECT_MIN_IMAGE,
    POSE_WRIST_REJECT_MIN_WORLD,
    POSE_WRIST_REJECT_SCALE,
)
from .geometry import _hand_scale, _iou_norm, _mean_l2_xy, _wrist_xy
from .pose import (
    _pose_wrists_close,
    _pose_wrist_dists,
    _pose_wrist_out_of_frame,
    _pose_wrist_refs,
)


def _last_good_wrist_dists(hand_pts, left_ref, right_ref) -> Tuple[Optional[float], Optional[float]]:
    if hand_pts is None:
        return None, None
    wxy = _wrist_xy(hand_pts)
    if wxy is None:
        return None, None
    dL = None
    dR = None
    if left_ref is not None:
        lxy = _wrist_xy(left_ref)
        if lxy is not None:
            dL = math.hypot(wxy[0] - lxy[0], wxy[1] - lxy[1])
    if right_ref is not None:
        rxy = _wrist_xy(right_ref)
        if rxy is not None:
            dR = math.hypot(wxy[0] - rxy[0], wxy[1] - rxy[1])
    return dL, dR


def _swap_by_last_good(
    left,
    right,
    left_score,
    right_score,
    left_px,
    right_px,
    left_img,
    right_img,
    last_left_img,
    last_right_img,
    ratio: float,
):
    if left is None or right is None or last_left_img is None or last_right_img is None:
        return left, right, left_score, right_score, left_px, right_px, left_img, right_img, False
    dLL, dLR = _last_good_wrist_dists(left_img, last_left_img, last_right_img)
    dRL, dRR = _last_good_wrist_dists(right_img, last_left_img, last_right_img)
    if dLL is None or dLR is None or dRL is None or dRR is None:
        return left, right, left_score, right_score, left_px, right_px, left_img, right_img, False
    cost_current = dLL + dRR
    cost_swap = dLR + dRL
    if cost_swap < (cost_current * ratio):
        return (
            right, left,
            right_score, left_score,
            right_px, left_px,
            right_img, left_img,
            True,
        )
    return left, right, left_score, right_score, left_px, right_px, left_img, right_img, False


def _side_consistent(
    side: str,
    hand_pts,
    world_coords: bool,
    pose_img_landmarks,
    pose_world_full,
    last_pose_world_full,
    last_left_good_img,
    last_right_good_img,
    ratio: float,
    use_last_good: bool,
) -> bool:
    if hand_pts is None:
        return False
    dL = None
    dR = None
    if use_last_good and (last_left_good_img is not None or last_right_good_img is not None):
        dL, dR = _last_good_wrist_dists(hand_pts, last_left_good_img, last_right_good_img)
    if dL is None or dR is None:
        dL, dR = _pose_wrist_dists(hand_pts, world_coords, pose_img_landmarks, pose_world_full, last_pose_world_full)
    if dL is None or dR is None:
        return False
    if side == "left":
        return dL <= (dR * ratio)
    return dR <= (dL * ratio)


def _hand_too_far_from_pose(hand_pts, pose_ref, world_coords):
    if hand_pts is None or pose_ref is None:
        return False
    wrist_xy = _wrist_xy(hand_pts)
    if wrist_xy is None:
        return False
    hx, hy = wrist_xy
    dist = math.hypot(hx - pose_ref[0], hy - pose_ref[1])
    min_scale = POSE_WRIST_REJECT_MIN_WORLD if world_coords else POSE_WRIST_REJECT_MIN_IMAGE
    scale = max(_hand_scale(hand_pts), min_scale)
    return dist > (POSE_WRIST_REJECT_SCALE * scale)


def _reject_hands_far_from_pose(
    left,
    right,
    left_score,
    right_score,
    world_coords,
    pose_img_landmarks,
    pose_world_full,
    last_pose_world_full,
):
    Lref, Rref = _pose_wrist_refs(world_coords, pose_img_landmarks, pose_world_full, last_pose_world_full)
    if _pose_wrist_out_of_frame("left", pose_img_landmarks) or _hand_too_far_from_pose(left, Lref, world_coords):
        left, left_score = None, None
    if _pose_wrist_out_of_frame("right", pose_img_landmarks) or _hand_too_far_from_pose(right, Rref, world_coords):
        right, right_score = None, None
    return left, right, left_score, right_score


def _antiswap_and_dedup(
    left,
    right,
    left_score,
    right_score,
    world_coords,
    pose_img_landmarks,
    pose_world_full,
    last_pose_world_full,
    left_px=None,
    right_px=None,
    left_img=None,
    right_img=None,
):
    swapped = False
    # 1) Anti-swap via pose wrists proximity
    Lref, Rref = _pose_wrist_refs(world_coords, pose_img_landmarks, pose_world_full, last_pose_world_full)
    pose_ambiguous = _pose_wrists_close(
        world_coords,
        pose_img_landmarks,
        pose_world_full,
        last_pose_world_full,
        [left, right],
    )

    if Lref and Rref and not pose_ambiguous:
        # Helper to get distance to wrists
        def get_dists(hand):
            xy = _wrist_xy(hand)
            if xy is None:
                return None, None
            dL = math.hypot(xy[0] - Lref[0], xy[1] - Lref[1])
            dR = math.hypot(xy[0] - Rref[0], xy[1] - Rref[1])
            return dL, dR

        # Case 1: Both hands present -> Swap if total distance is minimized by swapping
        if left is not None and right is not None:
            dLL, dLR = get_dists(left)
            dRL, dRR = get_dists(right)

            if dLL is not None and dRL is not None:
                cost_current = dLL + dRR
                cost_swap = dLR + dRL

                if cost_swap < cost_current:
                    left, right = right, left
                    left_score, right_score = right_score, left_score
                    left_px, right_px = right_px, left_px
                    left_img, right_img = right_img, left_img
                    swapped = True

        # Case 2: Single hand present -> Check if it belongs to the other side
        elif left is not None and right is None:
            dLL, dLR = get_dists(left)
            if dLL is not None:
                if dLR < dLL * 0.8:
                    right, right_score = left, left_score
                    left, left_score = None, None
                    right_px, right_img = left_px, left_img
                    left_px, left_img = None, None
                    swapped = True

        elif right is not None and left is None:
            dRL, dRR = get_dists(right)
            if dRL is not None:
                if dRL < dRR * 0.8:
                    left, left_score = right, right_score
                    right, right_score = None, None
                    left_px, left_img = right_px, right_img
                    right_px, right_img = None, None
                    swapped = True

    left, right, left_score, right_score = _reject_hands_far_from_pose(
        left,
        right,
        left_score,
        right_score,
        world_coords,
        pose_img_landmarks,
        pose_world_full,
        last_pose_world_full,
    )
    if left is None:
        left_px, left_img = None, None
    if right is None:
        right_px, right_img = None, None

    # 2) Duplicate suppression by mean landmark distance normalized by hand scale
    if left is not None and right is not None and not pose_ambiguous:
        sL = _hand_scale(left)
        sR = _hand_scale(right)
        scale = max(1e-6, 0.5 * (sL + sR))
        dist = _mean_l2_xy(left, right)

        d0 = math.hypot(float(left[0]["x"]) - float(right[0]["x"]), float(left[0]["y"]) - float(right[0]["y"]))

        iou = _iou_norm(left, right)

        if (dist < DEDUP_DIST_SCALE * scale) or (d0 < DEDUP_D0_SCALE * scale) or (iou > DEDUP_IOU_THRESHOLD):
            if (right_score or 0.0) > (left_score or 0.0):
                left, left_score = None, None
                left_px, left_img = None, None
            else:
                right, right_score = None, None
                right_px, right_img = None, None

    return left, right, left_score, right_score, left_px, right_px, left_img, right_img, swapped
