from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional

from ...core.utils import norm_to_px
from ..heuristics.geometry import _hand_scale, _wrist_xy
from ..heuristics.pose import (
    _pose_wrist_low_visibility,
    _pose_wrist_out_of_frame,
    _pose_wrist_refs,
)

from ..heuristics.constants import OVERLAP_GUARD_IOU


@dataclass
class SecondPassContext:
    overlap_iou_val: float
    overlap_guard_pre: bool
    overlap: bool
    effective_params: Dict[str, float | bool | int]
    strict: bool
    strict_params: Dict[str, float | bool | int]


@dataclass
class HandFrameState:
    landmarks: Optional[List[Dict[str, float]]]
    score: Optional[float]
    source: Optional[str]
    reject_reason: Optional[str]
    cur_img: Optional[List[Dict[str, float]]] = None
    cur_px: Optional[List[Dict[str, float]]] = None


@dataclass
class OcclusionTransitionResult:
    hand: HandFrameState
    occluded: bool
    occ_ttl: int
    occ_freeze_age: int
    hold: int
    missing_pre_occ: bool
    occlusion_saved: bool


@dataclass
class TrackerTransitionResult:
    hand: HandFrameState
    tracker_ready: bool
    track_age: int
    hold: int
    track_ok: bool
    track_reset: bool
    track_recovered_inc: int


def wrist_dist_norm(det_img, exp_img):
    if not det_img or not exp_img:
        return None
    det_xy = _wrist_xy(det_img)
    exp_xy = _wrist_xy(exp_img)
    if det_xy is None or exp_xy is None:
        return None
    d = math.hypot(det_xy[0] - exp_xy[0], det_xy[1] - exp_xy[1])
    scale = max(_hand_scale(exp_img), 1e-6)
    return d / scale


def pose_guided_freeze(
    hand_pts,
    *,
    last_good_img,
    hand: str,
    world_coords: bool,
    pose_img_landmarks,
    pose_world_full,
    last_pose_world_full,
    occ_return_k: float,
):
    if not hand_pts:
        return hand_pts
    if world_coords:
        return hand_pts
    if pose_img_landmarks is None:
        return hand_pts
    if _pose_wrist_out_of_frame(hand, pose_img_landmarks):
        return hand_pts
    if _pose_wrist_low_visibility(hand, pose_img_landmarks):
        return hand_pts
    left_ref, right_ref = _pose_wrist_refs(
        world_coords, pose_img_landmarks, pose_world_full, last_pose_world_full
    )
    pose_xy = left_ref if hand == "left" else right_ref
    if pose_xy is None:
        return hand_pts
    ref_pts = last_good_img or hand_pts
    wrist_xy = _wrist_xy(ref_pts)
    if wrist_xy is None:
        return hand_pts
    dx = pose_xy[0] - wrist_xy[0]
    dy = pose_xy[1] - wrist_xy[1]
    scale = _hand_scale(ref_pts)
    if scale <= 1e-6:
        return hand_pts
    max_shift = occ_return_k * scale
    dist = math.hypot(dx, dy)
    if max_shift > 0 and dist > max_shift:
        scale_factor = max_shift / dist
        dx *= scale_factor
        dy *= scale_factor
    shifted = []
    for point in hand_pts:
        if not isinstance(point, dict):
            shifted.append(point)
            continue
        nx = float(point["x"]) + dx
        ny = float(point["y"]) + dy
        nx = min(1.0, max(0.0, nx))
        ny = min(1.0, max(0.0, ny))
        shifted.append({**point, "x": nx, "y": ny})
    return shifted


def apply_occlusion_transition(
    hand: HandFrameState,
    *,
    side: str,
    occluded: bool,
    occ_ttl: int,
    occ_freeze_age: int,
    hold: int,
    overlap_guard: bool,
    overlap_freeze_side: Optional[str],
    score_source: str,
    hand_hi: float,
    anchor_score_eff: float,
    score_gate,
    pose_ok: bool,
    side_ok_accept: bool,
    det_img,
    last_good_img,
    last_export,
    last_export_score,
    occ_freeze_max_frames: int,
    occ_return_k: float,
    world_coords: bool,
    proc_w: int,
    proc_h: int,
    pose_img_landmarks,
    pose_world_full,
    last_pose_world_full,
) -> OcclusionTransitionResult:
    if occluded and hand.landmarks is not None and hand.source in ("pass1", "pass2"):
        accept = False
        if overlap_guard and overlap_freeze_side == side:
            accept = False
        elif overlap_guard:
            if (
                score_gate is not None
                and score_gate >= anchor_score_eff
                and pose_ok
                and side_ok_accept
            ):
                accept = True
        else:
            score_for_accept = hand.score if score_source == "handedness" else None
            if score_for_accept is not None and score_for_accept >= hand_hi:
                accept = True
            else:
                dist_norm = wrist_dist_norm(det_img, last_good_img)
                if dist_norm is not None and dist_norm <= occ_return_k:
                    accept = True
        if accept:
            occ_ttl = 0
            occluded = False
        else:
            hand.landmarks = None
            hand.score = None
            hand.source = None
            reason = "occ_freeze" if overlap_guard else "occ_guard"
            hand.reject_reason = hand.reject_reason or reason

    missing_pre_occ = hand.landmarks is None
    occlusion_saved = False

    if occluded:
        hold = 0
        if (
            hand.landmarks is None
            and last_export is not None
            and occ_freeze_age < occ_freeze_max_frames
        ):
            hand.landmarks = pose_guided_freeze(
                last_export,
                last_good_img=last_good_img,
                hand=side,
                world_coords=world_coords,
                pose_img_landmarks=pose_img_landmarks,
                pose_world_full=pose_world_full,
                last_pose_world_full=last_pose_world_full,
                occ_return_k=occ_return_k,
            )
            hand.source = "occluded"
            hand.score = last_export_score
            if not world_coords and hand.landmarks is not None:
                hand.cur_img = hand.landmarks
                hand.cur_px = norm_to_px(hand.landmarks, proc_w, proc_h)
            if missing_pre_occ:
                occlusion_saved = True
            occ_freeze_age += 1
        else:
            hand.landmarks = None

    return OcclusionTransitionResult(
        hand=hand,
        occluded=bool(occluded),
        occ_ttl=int(occ_ttl),
        occ_freeze_age=int(occ_freeze_age),
        hold=int(hold),
        missing_pre_occ=bool(missing_pre_occ),
        occlusion_saved=bool(occlusion_saved),
    )


def update_or_track_hand(
    hand: HandFrameState,
    *,
    tracker: Any,
    tracker_ready: bool,
    track_age: int,
    hold: int,
    world_coords: bool,
    overlap_ambiguous: bool,
    side_ok: bool,
    overlap_guard: bool,
    pose_ok: bool,
    block_track: bool,
    tracker_init_score_eff: float,
    tracker_update_score_eff: Optional[float],
    score_gate,
    ts: float,
    dt: float,
    rgb,
    track_reset_ms: int,
    track_max_gap: int,
    track_score_decay: float,
) -> TrackerTransitionResult:
    track_ok = False
    track_reset = False
    track_recovered_inc = 0

    if hand.landmarks is not None and hand.source in ("pass1", "pass2") and (not overlap_ambiguous or side_ok) and not overlap_guard:
        hand_img_for_tracker = hand.cur_img if hand.cur_img is not None else None
        if hand_img_for_tracker is None and not world_coords:
            hand_img_for_tracker = hand.landmarks
        if hand_img_for_tracker and not world_coords:
            if not tracker_ready:
                if pose_ok and hand.score is not None and hand.score >= tracker_init_score_eff:
                    tracker.reset()
                    tracker.update(hand_img_for_tracker, ts, rgb, score=score_gate)
                    tracker_ready = True
                    track_age = 0
            else:
                update_ok = pose_ok
                if tracker_update_score_eff is not None:
                    update_ok = update_ok and (
                        hand.score is not None and hand.score >= tracker_update_score_eff
                    )
                if update_ok:
                    tracker.update(
                        hand_img_for_tracker,
                        ts,
                        rgb,
                        score=score_gate if score_gate is not None else 1.0,
                    )
                    track_age = 0
        hold = 0
    elif hand.landmarks is None and tracker_ready and (not world_coords) and not block_track:
        if dt > track_reset_ms:
            tracker.reset()
            tracker_ready = False
            track_age = 0
            track_reset = True
        elif track_age < track_max_gap:
            tracked = tracker.track(rgb, ts)
            if tracked is not None:
                hand.landmarks = tracked
                hand.source = "tracked"
                base = getattr(tracker, "last_score", None) or 1.0
                hand.score = base * (track_score_decay ** (track_age + 1))
                track_ok = True
                track_age += 1
                track_recovered_inc = 1
            else:
                tracker.reset()
                tracker_ready = False
                track_age = 0

    return TrackerTransitionResult(
        hand=hand,
        tracker_ready=bool(tracker_ready),
        track_age=int(track_age),
        hold=int(hold),
        track_ok=bool(track_ok),
        track_reset=bool(track_reset),
        track_recovered_inc=int(track_recovered_inc),
    )


def build_second_pass_context(
    *,
    sp_overlap_iou_val: float,
    sp_overlap_iou: float,
    sp_overlap_shrink: float,
    sp_center_penalty: float,
    sp_overlap_penalty_mult: float,
    sp_label_relax: float,
    sp_overlap_require_label: bool,
    sp_roi_frac: float,
    sp_jitter_px: int,
    sp_jitter_rings: int,
) -> SecondPassContext:
    overlap_guard_pre = sp_overlap_iou_val >= OVERLAP_GUARD_IOU
    overlap = sp_overlap_iou_val >= sp_overlap_iou
    sp_roi_frac_eff = sp_roi_frac * (sp_overlap_shrink if overlap else 1.0)
    sp_center_penalty_eff = sp_center_penalty * (sp_overlap_penalty_mult if overlap else 1.0)
    sp_label_relax_eff = 0.0 if overlap else sp_label_relax
    sp_require_label_eff = bool(sp_overlap_require_label and overlap)
    max_center_dist_norm_eff = 0.9 if overlap else 1.25
    effective = {
        "sp_roi_frac": float(sp_roi_frac_eff),
        "center_penalty_lambda": float(sp_center_penalty_eff),
        "label_relax_margin": float(sp_label_relax_eff),
        "require_label_match": bool(sp_require_label_eff),
        "max_center_dist_norm": float(max_center_dist_norm_eff),
        "sp_jitter_px": int(sp_jitter_px),
        "sp_jitter_rings": int(sp_jitter_rings),
    }
    strict = bool(overlap_guard_pre)
    strict_params = {
        "sp_roi_frac": float(sp_roi_frac_eff * 0.7),
        "center_penalty_lambda": float(sp_center_penalty_eff * 1.5),
        "label_relax_margin": 0.0,
        "require_label_match": True,
        "max_center_dist_norm": float(min(max_center_dist_norm_eff, 0.6)),
        "sp_jitter_px": int(sp_jitter_px),
        "sp_jitter_rings": int(sp_jitter_rings),
    }
    return SecondPassContext(
        overlap_iou_val=float(sp_overlap_iou_val),
        overlap_guard_pre=bool(overlap_guard_pre),
        overlap=bool(overlap),
        effective_params=effective,
        strict=bool(strict),
        strict_params=strict_params,
    )
