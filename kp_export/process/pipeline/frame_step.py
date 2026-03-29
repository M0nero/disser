from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple

from ...algos.tracking import HandTracker
from ...core.logging_utils import get_logger
from ...core.utils import bbox_from_pts_px, norm_to_px
from ...mp.mp_utils import _ALIGN_HANDS_TO_POSE, align_hand_xy_to_target
from ...mp.roi import hands_up_gate
from ..contracts import FrameRecord
from ..heuristics.association import (
    _antiswap_and_dedup,
    _hand_too_far_from_pose,
    _last_good_wrist_dists,
    _reject_hands_far_from_pose,
    _side_consistent,
    _swap_by_last_good,
)
from ..heuristics.constants import (
    CROSS_OCCLUSION_FRAMES,
    DELTA_QUALITY,
    DEDUP_D0_SCALE,
    DEDUP_DIST_SCALE,
    DEDUP_IOU_THRESHOLD,
    MIN_QUALITY,
    OCCLUSION_IOU_THRESHOLD_IMAGE,
    OCCLUSION_IOU_THRESHOLD_WORLD,
    OCCLUSION_Z_THRESHOLD_IMAGE,
    OCCLUSION_Z_THRESHOLD_WORLD,
    MP_POSE_LEFT_WRIST_IDX,
    MP_POSE_RIGHT_WRIST_IDX,
    OCCLUSION_FREEZE_MAX_FRAMES,
    OVERLAP_GUARD_IOU,
    OVERLAP_Z_MIN,
)
from ..heuristics.geometry import _hand_scale, _iou_norm, _mean_l2_xy, _wrist_xy
from ..heuristics.occlusion import _overlap_iou, _pick_wrist_depth, _resolve_pose_world
from ..heuristics.pose import (
    _center_hint_for,
    _hand_expected,
    _pose_distance_quality,
    _pose_gate_allows_second_pass,
    _pose_wrist_dists,
    _pose_wrist_refs,
    _pose_wrists_close,
)
from ..records.builder import build_runtime_frame_record
from ..state import SampleRuntime, classify_hand_state
from .decode import DecodedFrame
from .detect import PoseRuntimeState, run_frame_detectors
from .filtering import append_reject_reason, apply_sanity_stage, score_for_gate
from .recover import HandFrameState, apply_occlusion_transition, build_second_pass_context, update_or_track_hand
from .second_pass import _execute_second_pass

LOGGER = get_logger(__name__)


@dataclass
class SideStepResult:
    sp_recovered: bool = False


@dataclass
class FrameStepResult:
    record: FrameRecord
    hand_runtime: float
    pose_runtime: float
    second_pass_runtime: float
    sp_missing_pre_left: bool
    sp_missing_pre_right: bool
    left: SideStepResult
    right: SideStepResult


@dataclass
class FrameStepContext:
    hands_detector: Any
    pose_detector: Any
    hands_sp: Any
    pose_state: PoseRuntimeState
    sample_state: SampleRuntime
    tracker_left: HandTracker
    tracker_right: HandTracker
    world_coords: bool
    keep_pose_indices: Optional[list[int]]
    pose_every: int
    pose_ema_alpha: float
    min_hand_score: float
    hand_lo: float
    hand_hi: float
    score_source: str
    anchor_score_eff: float
    tracker_init_score_eff: float
    tracker_update_score_eff: Optional[float]
    pose_dist_qual_min_eff: float
    pose_side_reassign_ratio_eff: float
    second_pass: bool
    sp_trigger_below: float
    sp_roi_frac: float
    sp_margin: float
    sp_escalate_step: float
    sp_escalate_max: float
    sp_hands_up_only: bool
    sp_jitter_px: int
    sp_jitter_rings: int
    sp_center_penalty: float
    sp_label_relax: float
    sp_overlap_iou: float
    sp_overlap_shrink: float
    sp_overlap_penalty_mult: float
    sp_overlap_require_label: bool
    sp_debug_roi: bool
    occ_hyst_frames: int
    occ_return_k: float
    sanity_enable: bool
    sanity_scale_range: Tuple[float, float]
    sanity_wrist_k: float
    sanity_bone_tol: float
    sanity_pass2: bool
    sanity_anchor_max_gap: int
    sanitize_rejects: bool
    track_max_gap: int
    track_score_decay: float
    track_reset_ms: int
    write_hand_mask: bool


def process_frame_step(decoded: DecodedFrame, *, context: FrameStepContext) -> FrameStepResult:
    hands_detector = context.hands_detector
    pose_detector = context.pose_detector
    hands_sp = context.hands_sp
    pose_state = context.pose_state
    sample_state = context.sample_state
    tracker_left = context.tracker_left
    tracker_right = context.tracker_right
    world_coords = context.world_coords
    keep_pose_indices = context.keep_pose_indices
    pose_every = context.pose_every
    pose_ema_alpha = context.pose_ema_alpha
    min_hand_score = context.min_hand_score
    hand_lo = context.hand_lo
    hand_hi = context.hand_hi
    score_source = context.score_source
    anchor_score_eff = context.anchor_score_eff
    tracker_init_score_eff = context.tracker_init_score_eff
    tracker_update_score_eff = context.tracker_update_score_eff
    pose_dist_qual_min_eff = context.pose_dist_qual_min_eff
    pose_side_reassign_ratio_eff = context.pose_side_reassign_ratio_eff
    second_pass = context.second_pass
    sp_trigger_below = context.sp_trigger_below
    sp_roi_frac = context.sp_roi_frac
    sp_margin = context.sp_margin
    sp_escalate_step = context.sp_escalate_step
    sp_escalate_max = context.sp_escalate_max
    sp_hands_up_only = context.sp_hands_up_only
    sp_jitter_px = context.sp_jitter_px
    sp_jitter_rings = context.sp_jitter_rings
    sp_center_penalty = context.sp_center_penalty
    sp_label_relax = context.sp_label_relax
    sp_overlap_iou = context.sp_overlap_iou
    sp_overlap_shrink = context.sp_overlap_shrink
    sp_overlap_penalty_mult = context.sp_overlap_penalty_mult
    sp_overlap_require_label = context.sp_overlap_require_label
    sp_debug_roi = context.sp_debug_roi
    occ_hyst_frames = context.occ_hyst_frames
    occ_return_k = context.occ_return_k
    sanity_enable = context.sanity_enable
    sanity_scale_range = context.sanity_scale_range
    sanity_wrist_k = context.sanity_wrist_k
    sanity_bone_tol = context.sanity_bone_tol
    sanity_pass2 = context.sanity_pass2
    sanity_anchor_max_gap = context.sanity_anchor_max_gap
    sanitize_rejects = context.sanitize_rejects
    track_max_gap = context.track_max_gap
    track_score_decay = context.track_score_decay
    track_reset_ms = context.track_reset_ms
    write_hand_mask = context.write_hand_mask
    left_state = sample_state.left
    right_state = sample_state.right
    hand_runtime = 0.0
    pose_runtime = 0.0
    second_pass_runtime = 0.0

    i = decoded.frame_index
    proc_w = decoded.proc_w
    proc_h = decoded.proc_h
    ts = decoded.ts_ms
    dt = decoded.dt_ms
    bgr = decoded.bgr
    rgb = decoded.rgb

    left_reject_reason = None
    right_reject_reason = None
    src1 = None
    src2 = None
    detection, pose_state, hand_dt, pose_dt = run_frame_detectors(
        decoded,
        hands_detector=hands_detector,
        pose_detector=pose_detector,
        pose_every=pose_every,
        world_coords=world_coords,
        keep_pose_indices=keep_pose_indices,
        pose_ema_alpha=pose_ema_alpha,
        state=pose_state,
    )
    hand_runtime += hand_dt
    pose_runtime += pose_dt
    left = detection.left
    right = detection.right
    left_score = detection.left_score
    right_score = detection.right_score
    pose_xyz = detection.pose_xyz
    pose_vis = detection.pose_vis
    pose_interpolated = detection.pose_interpolated
    pose_img_landmarks = detection.pose_img_landmarks
    pose_world_current = detection.pose_world_current
    pose_world_full = detection.pose_world_full
    cur_left_img = detection.cur_left_img
    cur_right_img = detection.cur_right_img
    cur_left_px = detection.cur_left_px
    cur_right_px = detection.cur_right_px

    prev_pred_left = left_state.previous_observation(i)
    prev_pred_right = right_state.previous_observation(i)
    anchor_left = left_state.anchor_for_sanity(i, max_gap=sanity_anchor_max_gap)
    anchor_right = right_state.anchor_for_sanity(i, max_gap=sanity_anchor_max_gap)

    last_pose_world_full = pose_state.last_pose_world_full
    left_score_gate = score_for_gate(left_score, left is not None, score_source)
    right_score_gate = score_for_gate(right_score, right is not None, score_source)

    if left is not None and (left_score_gate is None or left_score_gate < min_hand_score):
        left, left_score = None, None
    if right is not None and (right_score_gate is None or right_score_gate < min_hand_score):
        right, right_score = None, None

    left, right, left_score, right_score = _reject_hands_far_from_pose(
        left, right, left_score, right_score,
        world_coords, pose_img_landmarks, pose_world_current, last_pose_world_full
    )

    if left is not None and (left_score_gate is None or left_score_gate < hand_lo):
        left, left_score = None, None
        src1 = None
        left_reject_reason = "score_lo"
    if right is not None and (right_score_gate is None or right_score_gate < hand_lo):
        right, right_score = None, None
        src2 = None
        right_reject_reason = "score_lo"

    sanity_dbg_left = None
    sanity_dbg_right = None
    sanity_stage_left = None
    sanity_stage_right = None

    left, left_score, left_reject_reason, sanity_dbg_left, sanity_stage_left = apply_sanity_stage(
        left,
        left_score,
        left_reject_reason,
        enabled=sanity_enable,
        prev_anchor=anchor_left,
        prev_pred=prev_pred_left,
        world_coords=world_coords,
        scale_range=sanity_scale_range,
        wrist_k=sanity_wrist_k,
        bone_tol=sanity_bone_tol,
        stage="pass1",
    )
    if left is None:
        src1 = None
    right, right_score, right_reject_reason, sanity_dbg_right, sanity_stage_right = apply_sanity_stage(
        right,
        right_score,
        right_reject_reason,
        enabled=sanity_enable,
        prev_anchor=anchor_right,
        prev_pred=prev_pred_right,
        world_coords=world_coords,
        scale_range=sanity_scale_range,
        wrist_k=sanity_wrist_k,
        bone_tol=sanity_bone_tol,
        stage="pass1",
    )
    if right is None:
        src2 = None

    src1 = "pass1" if (left is not None and left_score is not None) else None
    src2 = "pass1" if (right is not None and right_score is not None) else None

    sp_left_ref_px = cur_left_px if cur_left_px is not None else left_state.last_good_px
    sp_right_ref_px = cur_right_px if cur_right_px is not None else right_state.last_good_px
    if sp_left_ref_px is not None and sp_right_ref_px is not None:
        overlap_iou = _overlap_iou(sp_left_ref_px, sp_right_ref_px, fallback=0.0)
    else:
        overlap_iou = 0.0
    sp_overlap_iou_val = float(overlap_iou) if overlap_iou is not None else 0.0
    sp_context = build_second_pass_context(
        sp_overlap_iou_val=sp_overlap_iou_val,
        sp_overlap_iou=sp_overlap_iou,
        sp_overlap_shrink=sp_overlap_shrink,
        sp_center_penalty=sp_center_penalty,
        sp_overlap_penalty_mult=sp_overlap_penalty_mult,
        sp_label_relax=sp_label_relax,
        sp_overlap_require_label=sp_overlap_require_label,
        sp_roi_frac=sp_roi_frac,
        sp_jitter_px=sp_jitter_px,
        sp_jitter_rings=sp_jitter_rings,
    )
    overlap_guard_pre = sp_context.overlap_guard_pre
    overlap = sp_context.overlap
    sp_params_eff = dict(sp_context.effective_params)
    sp_strict = sp_context.strict
    sp_params_eff_strict = dict(sp_context.strict_params)

    sp_kwargs = {
        "bgr": bgr,
        "proc_w": proc_w,
        "proc_h": proc_h,
        "world_coords": world_coords,
        "pose_img_landmarks": pose_img_landmarks,
        "last_left_px": cur_left_px,
        "last_right_px": cur_right_px,
        "hands_sp": hands_sp,
        "sp_trigger_below": sp_trigger_below,
        "sp_roi_frac": sp_params_eff["sp_roi_frac"],
        "sp_margin": sp_margin,
        "sp_escalate_step": sp_escalate_step,
        "sp_escalate_max": sp_escalate_max,
        "sp_hands_up_only": sp_hands_up_only,
        "sp_jitter_px": sp_jitter_px,
        "sp_jitter_rings": sp_jitter_rings,
        "center_penalty_lambda": sp_params_eff["center_penalty_lambda"],
        "label_relax_margin": sp_params_eff["label_relax_margin"],
        "require_label_match": sp_params_eff["require_label_match"],
        "max_center_dist_norm": sp_params_eff["max_center_dist_norm"],
    }

    sp_attempt_left = False
    sp_attempt_right = False
    sp_rec_left = 0
    sp_rec_right = 0
    missing_pre_sp_left = (left is None)
    missing_pre_sp_right = (right is None)

    left_roi = None
    right_roi = None
    sp_dbg_left = None
    sp_dbg_right = None
    sp_params_left = None
    sp_params_right = None

    expected_left = _hand_expected("left", pose_img_landmarks, left_state.tracker_ready, left_state.last_good_px)
    expected_right = _hand_expected("right", pose_img_landmarks, right_state.tracker_ready, right_state.last_good_px)
    center_hint_left = _center_hint_for(
        "left", proc_w, proc_h, pose_img_landmarks, tracker_left, left_state.tracker_ready, left_state.last_good_px
    )
    center_hint_right = _center_hint_for(
        "right", proc_w, proc_h, pose_img_landmarks, tracker_right, right_state.tracker_ready, right_state.last_good_px
    )

    allow_sp_occ_left = (left_state.occ_ttl > 0) and (not world_coords) and (left_state.last_good_px is not None)
    allow_sp_occ_right = (right_state.occ_ttl > 0) and (not world_coords) and (right_state.last_good_px is not None)
    if allow_sp_occ_left:
        x0, y0, x1, y1 = bbox_from_pts_px(left_state.last_good_px)
        center_hint_left = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
    if allow_sp_occ_right:
        x0, y0, x1, y1 = bbox_from_pts_px(right_state.last_good_px)
        center_hint_right = (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    if left_state.occ_ttl > 0 and not allow_sp_occ_left:
        sp_attempt_left = False
        rec_l = False
    elif allow_sp_occ_left or _pose_gate_allows_second_pass("left", pose_img_landmarks):
        need_sp_left = (left is None)
        sp_hands_up_ok_left = (not sp_hands_up_only) or hands_up_gate(
            "left", pose_img_landmarks, proc_w, proc_h
        )
        if allow_sp_occ_left:
            sp_hands_up_ok_left = True
        sp_attempt_left = bool(
            hands_sp is not None and second_pass and expected_left and need_sp_left and sp_hands_up_ok_left
        )
        if sp_attempt_left:
            sp_dbg_left = {} if sp_debug_roi else None
            sp_kwargs_left = sp_kwargs
            if sp_strict:
                sp_kwargs_left = dict(sp_kwargs)
                sp_kwargs_left.update(
                    {
                        "sp_roi_frac": sp_params_eff_strict["sp_roi_frac"],
                        "center_penalty_lambda": sp_params_eff_strict["center_penalty_lambda"],
                        "label_relax_margin": sp_params_eff_strict["label_relax_margin"],
                        "require_label_match": sp_params_eff_strict["require_label_match"],
                        "max_center_dist_norm": sp_params_eff_strict["max_center_dist_norm"],
                    }
                )
            sp_result_left, sp_dt = _execute_second_pass(
                "left",
                left,
                left_score,
                debug_roi=sp_debug_roi,
                center_hint=center_hint_left,
                debug_out=sp_dbg_left,
                **sp_kwargs_left,
            )
            left = sp_result_left.landmarks
            left_score = sp_result_left.score
            rec_l = bool(sp_result_left.recovered)
            left_roi = sp_result_left.roi
            second_pass_runtime += sp_dt
            if rec_l:
                src1 = "pass2"
                sp_rec_left += 1
                if sanitize_rejects:
                    left_reject_reason = None
            if sp_dbg_left is not None:
                sp_params_left = dict(sp_params_eff_strict if sp_strict else sp_params_eff)
        else:
            rec_l = False
    else:
        rec_l = False

    if right_state.occ_ttl > 0 and not allow_sp_occ_right:
        sp_attempt_right = False
        rec_r = False
    elif allow_sp_occ_right or _pose_gate_allows_second_pass("right", pose_img_landmarks):
        need_sp_right = (right is None)
        sp_hands_up_ok_right = (not sp_hands_up_only) or hands_up_gate(
            "right", pose_img_landmarks, proc_w, proc_h
        )
        if allow_sp_occ_right:
            sp_hands_up_ok_right = True
        sp_attempt_right = bool(
            hands_sp is not None and second_pass and expected_right and need_sp_right and sp_hands_up_ok_right
        )
        if sp_attempt_right:
            sp_dbg_right = {} if sp_debug_roi else None
            sp_kwargs_right = sp_kwargs
            if sp_strict:
                sp_kwargs_right = dict(sp_kwargs)
                sp_kwargs_right.update(
                    {
                        "sp_roi_frac": sp_params_eff_strict["sp_roi_frac"],
                        "center_penalty_lambda": sp_params_eff_strict["center_penalty_lambda"],
                        "label_relax_margin": sp_params_eff_strict["label_relax_margin"],
                        "require_label_match": sp_params_eff_strict["require_label_match"],
                        "max_center_dist_norm": sp_params_eff_strict["max_center_dist_norm"],
                    }
                )
            sp_result_right, sp_dt = _execute_second_pass(
                "right",
                right,
                right_score,
                debug_roi=sp_debug_roi,
                center_hint=center_hint_right,
                debug_out=sp_dbg_right,
                **sp_kwargs_right,
            )
            right = sp_result_right.landmarks
            right_score = sp_result_right.score
            rec_r = bool(sp_result_right.recovered)
            right_roi = sp_result_right.roi
            second_pass_runtime += sp_dt
            if rec_r:
                src2 = "pass2"
                sp_rec_right += 1
                if sanitize_rejects:
                    right_reject_reason = None
            if sp_dbg_right is not None:
                sp_params_right = dict(sp_params_eff_strict if sp_strict else sp_params_eff)
        else:
            rec_r = False
    else:
        rec_r = False

    left_score_gate = score_for_gate(left_score, left is not None, score_source)
    right_score_gate = score_for_gate(right_score, right is not None, score_source)

    if left is not None and (left_score_gate is None or left_score_gate < hand_lo):
        left, left_score = None, None
        src1 = None
        left_reject_reason = left_reject_reason or "score_lo_after_sp"
    if right is not None and (right_score_gate is None or right_score_gate < hand_lo):
        right, right_score = None, None
        src2 = None
        right_reject_reason = right_reject_reason or "score_lo_after_sp"

    left, left_score, left_reject_reason, sanity_dbg_left, sanity_stage_left = apply_sanity_stage(
        left,
        left_score,
        left_reject_reason,
        enabled=(sanity_enable and sanity_pass2),
        prev_anchor=anchor_left,
        prev_pred=prev_pred_left,
        world_coords=world_coords,
        scale_range=sanity_scale_range,
        wrist_k=sanity_wrist_k,
        bone_tol=sanity_bone_tol,
        stage="pass2",
    )
    if left is None:
        src1 = None
    right, right_score, right_reject_reason, sanity_dbg_right, sanity_stage_right = apply_sanity_stage(
        right,
        right_score,
        right_reject_reason,
        enabled=(sanity_enable and sanity_pass2),
        prev_anchor=anchor_right,
        prev_pred=prev_pred_right,
        world_coords=world_coords,
        scale_range=sanity_scale_range,
        wrist_k=sanity_wrist_k,
        bone_tol=sanity_bone_tol,
        stage="pass2",
    )
    if right is None:
        src2 = None

    if rec_l and left is not None and not world_coords:
        cur_left_img = left
        cur_left_px = norm_to_px(left, proc_w, proc_h)
    if rec_r and right is not None and not world_coords:
        cur_right_img = right
        cur_right_px = norm_to_px(right, proc_w, proc_h)

    pose_world_full = _resolve_pose_world(world_coords, pose_world_current, last_pose_world_full)

    pose_side_ambiguous = _pose_wrists_close(
        world_coords,
        pose_img_landmarks,
        pose_world_full,
        last_pose_world_full,
        [left, right, left_state.last_good_img, right_state.last_good_img] if not world_coords else [left, right],
    )

    # If second pass grabs the other hand, reassign based on pose wrist proximity.
    if pose_img_landmarks is not None:
        if left is not None and src1 == "pass2" and src2 != "pass2":
            if pose_side_ambiguous and not world_coords:
                dLL, dLR = _last_good_wrist_dists(left, left_state.last_good_img, right_state.last_good_img)
            else:
                dLL, dLR = _pose_wrist_dists(
                    left,
                    world_coords,
                    pose_img_landmarks,
                    pose_world_full,
                    last_pose_world_full,
                )
            if (
                dLL is not None and dLR is not None and
                dLR < (dLL * pose_side_reassign_ratio_eff)
            ):
                if right is None or src2 in ("tracked", "occluded", "hold"):
                    right, right_score = left, left_score
                    src2 = "pass2"
                    if sanitize_rejects:
                        right_reject_reason = None
                    if right_roi is None:
                        right_roi = left_roi
                    if rec_l:
                        rec_r = True
                        rec_l = False
                        if sp_rec_left > 0:
                            sp_rec_left -= 1
                        sp_rec_right += 1
                    sp_attempt_right = True
                    if sp_dbg_right is None and sp_dbg_left is not None:
                        sp_dbg_right = sp_dbg_left
                    if sp_params_right is None and sp_params_left is not None:
                        sp_params_right = sp_params_left
                    cur_right_img, cur_right_px = cur_left_img, cur_left_px
                    left, left_score = None, None
                    src1 = None
                    left_roi = None
                    cur_left_img, cur_left_px = None, None
                    left_reject_reason = append_reject_reason(left_reject_reason, "pose_side_reassign")
                    sp_dbg_left = None
                    sp_params_left = None
                else:
                    left, left_score = None, None
                    src1 = None
                    left_roi = None
                    cur_left_img, cur_left_px = None, None
                    left_reject_reason = append_reject_reason(left_reject_reason, "pose_side_drop")
                    if rec_l:
                        rec_l = False
                        if sp_rec_left > 0:
                            sp_rec_left -= 1
                    sp_dbg_left = None
                    sp_params_left = None

        if right is not None and src2 == "pass2" and src1 != "pass2":
            if pose_side_ambiguous and not world_coords:
                dRL, dRR = _last_good_wrist_dists(right, left_state.last_good_img, right_state.last_good_img)
            else:
                dRL, dRR = _pose_wrist_dists(
                    right,
                    world_coords,
                    pose_img_landmarks,
                    pose_world_full,
                    last_pose_world_full,
                )
            if (
                dRL is not None and dRR is not None and
                dRL < (dRR * pose_side_reassign_ratio_eff)
            ):
                if left is None or src1 in ("tracked", "occluded", "hold"):
                    left, left_score = right, right_score
                    src1 = "pass2"
                    if sanitize_rejects:
                        left_reject_reason = None
                    if left_roi is None:
                        left_roi = right_roi
                    if rec_r:
                        rec_l = True
                        rec_r = False
                        if sp_rec_right > 0:
                            sp_rec_right -= 1
                        sp_rec_left += 1
                    sp_attempt_left = True
                    if sp_dbg_left is None and sp_dbg_right is not None:
                        sp_dbg_left = sp_dbg_right
                    if sp_params_left is None and sp_params_right is not None:
                        sp_params_left = sp_params_right
                    cur_left_img, cur_left_px = cur_right_img, cur_right_px
                    right, right_score = None, None
                    src2 = None
                    right_roi = None
                    cur_right_img, cur_right_px = None, None
                    right_reject_reason = append_reject_reason(right_reject_reason, "pose_side_reassign")
                    sp_dbg_right = None
                    sp_params_right = None
                else:
                    right, right_score = None, None
                    src2 = None
                    right_roi = None
                    cur_right_img, cur_right_px = None, None
                    right_reject_reason = append_reject_reason(right_reject_reason, "pose_side_drop")
                    if rec_r:
                        rec_r = False
                        if sp_rec_right > 0:
                            sp_rec_right -= 1
                    sp_dbg_right = None
                    sp_params_right = None

    swap_by_history = False
    if (
        pose_side_ambiguous and not world_coords and
        left is not None and right is not None and
        src1 in ("pass1", "pass2") and src2 in ("pass1", "pass2")
    ):
        old_rec_l = rec_l
        old_rec_r = rec_r
        left, right, left_score, right_score, cur_left_px, cur_right_px, cur_left_img, cur_right_img, swap_by_history = _swap_by_last_good(
            left,
            right,
            left_score,
            right_score,
            cur_left_px,
            cur_right_px,
            cur_left_img,
            cur_right_img,
            left_state.last_good_img,
            right_state.last_good_img,
            pose_side_reassign_ratio_eff,
        )
        if swap_by_history:
            src1, src2 = src2, src1
            left_reject_reason, right_reject_reason = right_reject_reason, left_reject_reason
            left_roi, right_roi = right_roi, left_roi
            sp_dbg_left, sp_dbg_right = sp_dbg_right, sp_dbg_left
            sp_params_left, sp_params_right = sp_params_right, sp_params_left
            rec_l, rec_r = rec_r, rec_l
            sp_attempt_left, sp_attempt_right = sp_attempt_right, sp_attempt_left
            if old_rec_l:
                if sp_rec_left > 0:
                    sp_rec_left -= 1
                sp_rec_right += 1
            if old_rec_r:
                if sp_rec_right > 0:
                    sp_rec_right -= 1
                sp_rec_left += 1

    if (
        left is not None and right is not None and
        src1 in ("pass1", "pass2") and src2 in ("pass1", "pass2")
    ):
        if pose_side_ambiguous and not world_coords:
            dLL, dLR = _last_good_wrist_dists(left, left_state.last_good_img, right_state.last_good_img)
            dRL, dRR = _last_good_wrist_dists(right, left_state.last_good_img, right_state.last_good_img)
        else:
            dLL, dLR = _pose_wrist_dists(
                left, world_coords, pose_img_landmarks, pose_world_full, last_pose_world_full
            )
            dRL, dRR = _pose_wrist_dists(
                right, world_coords, pose_img_landmarks, pose_world_full, last_pose_world_full
            )

        pref_left = None
        pref_right = None
        if dLL is not None and dLR is not None:
            if dLR < dLL * pose_side_reassign_ratio_eff:
                pref_left = "right"
            elif dLL < dLR * pose_side_reassign_ratio_eff:
                pref_left = "left"
        if dRL is not None and dRR is not None:
            if dRL < dRR * pose_side_reassign_ratio_eff:
                pref_right = "left"
            elif dRR < dRL * pose_side_reassign_ratio_eff:
                pref_right = "right"

        if pref_left == pref_right and pref_left in ("left", "right"):
            keep_left = True
            if pref_left == "left":
                if dLL is None:
                    keep_left = False
                elif dRL is None:
                    keep_left = True
                else:
                    keep_left = (dLL <= dRL)
            else:
                if dLR is None:
                    keep_left = False
                elif dRR is None:
                    keep_left = True
                else:
                    keep_left = (dLR <= dRR)

            if keep_left:
                right, right_score = None, None
                src2 = None
                right_roi = None
                cur_right_img, cur_right_px = None, None
                right_reject_reason = append_reject_reason(right_reject_reason, "pose_side_drop")
                sp_dbg_right = None
                sp_params_right = None
                if rec_r:
                    rec_r = False
                    if sp_rec_right > 0:
                        sp_rec_right -= 1
            else:
                left, left_score = None, None
                src1 = None
                left_roi = None
                cur_left_img, cur_left_px = None, None
                left_reject_reason = append_reject_reason(left_reject_reason, "pose_side_drop")
                sp_dbg_left = None
                sp_params_left = None
                if rec_l:
                    rec_l = False
                    if sp_rec_left > 0:
                        sp_rec_left -= 1

    # Anti-swap by pose wrists + duplicate suppression if both hands overlap
    swap_applied = False
    dedup_triggered = False
    dedup_removed = "none"
    dedup_iou = None
    dedup_dist_norm = None
    try:
        # Soft deduplication for tracked vs detected hands:
        # Only drop one when overlap is strong and one is clearly better by score/pose proximity.
        if not pose_side_ambiguous and left is not None and right is not None:
            is_left_tracked = (src1 == "tracked")
            is_right_tracked = (src2 == "tracked")

            if is_left_tracked != is_right_tracked:  # One tracked, one detected
                dist = _mean_l2_xy(left, right)
                scale = max(1e-6, 0.5 * (_hand_scale(left) + _hand_scale(right)))
                iou = _iou_norm(left, right)
                dedup_iou = float(iou)
                dedup_dist_norm = float(dist / scale) if scale > 0 else None

                overlap_strong = (dist < 0.6 * scale) or (iou > 0.35)
                if overlap_strong:
                    dedup_triggered = True
                    left_score_val = float(left_score or 0.0)
                    right_score_val = float(right_score or 0.0)

                    left_pose_bonus = 0.0
                    right_pose_bonus = 0.0
                    Lref, Rref = _pose_wrist_refs(
                        world_coords, pose_img_landmarks, pose_world_full, last_pose_world_full
                    )
                    if Lref and Rref:
                        lx, ly = _wrist_xy(left)
                        rx, ry = _wrist_xy(right)
                        if lx is not None:
                            left_dist = math.hypot(lx - Lref[0], ly - Lref[1])
                            left_pose_bonus = max(0.0, 1.0 - min(1.0, left_dist / max(scale, 1e-6)))
                        if rx is not None:
                            right_dist = math.hypot(rx - Rref[0], ry - Rref[1])
                            right_pose_bonus = max(0.0, 1.0 - min(1.0, right_dist / max(scale, 1e-6)))

                    left_quality = left_score_val + 0.5 * left_pose_bonus
                    right_quality = right_score_val + 0.5 * right_pose_bonus

                    best_quality = max(left_quality, right_quality)
                    worst_quality = min(left_quality, right_quality)
                    if (
                        best_quality >= MIN_QUALITY and
                        (best_quality - worst_quality) >= DELTA_QUALITY
                    ):
                        if left_quality >= right_quality:
                            right, right_score = None, None
                            dedup_removed = "hand_2"
                        else:
                            left, left_score = None, None
                            dedup_removed = "hand_1"

        left, right, left_score, right_score, cur_left_px, cur_right_px, cur_left_img, cur_right_img, swap_applied = _antiswap_and_dedup(
            left, right, left_score, right_score,
            world_coords, pose_img_landmarks, pose_world_full, last_pose_world_full,
            left_px=cur_left_px,
            right_px=cur_right_px,
            left_img=cur_left_img,
            right_img=cur_right_img,
        )
        swap_applied = bool(swap_applied or swap_by_history)
    except Exception as e:
        LOGGER.warning(f"Antiswap/dedup failed at frame ts={ts}, i={i}: {e}")
        # keep original values on any failure

    left_score_gate = score_for_gate(left_score, left is not None, score_source)
    right_score_gate = score_for_gate(right_score, right is not None, score_source)

    occluded_L = False
    occluded_R = False
    occlusion_iou = None
    occlusion_z_diff = None
    occlusion_behind_diff = None
    occlusion_samples_ok = False
    overlap_z_hint = None

    # --- Occlusion Detection ---
    # IoU is calculated in pixel space (using current or last-good px),
    # even when world_coords=True, because pixel-space overlap is more
    # stable and accurate for detecting visual occlusion.
    left_ref_px = cur_left_px if cur_left_px is not None else left_state.last_good_px
    right_ref_px = cur_right_px if cur_right_px is not None else right_state.last_good_px
    enter_left = False
    hold_left_occ = False
    enter_right = False
    hold_right_occ = False

    if left_ref_px is not None and right_ref_px is not None:
        left_depth = _pick_wrist_depth(
            left if left is not None else None,
            pose_img_landmarks,
            pose_world_full,
            "left",
            world_coords,
        )
        right_depth = _pick_wrist_depth(
            right if right is not None else None,
            pose_img_landmarks,
            pose_world_full,
            "right",
            world_coords,
        )

        iou_cur = _overlap_iou(left_ref_px, right_ref_px, fallback=0.0)
        occlusion_iou = float(iou_cur)
        if left_depth is not None and right_depth is not None:
            occlusion_samples_ok = True
            z_diff = float(left_depth - right_depth)
            behind = float(abs(z_diff))
            occlusion_z_diff = z_diff
            occlusion_behind_diff = behind

            # Tighter thresholds for world coords; tuned for image coords to react quicker
            iou_thr = OCCLUSION_IOU_THRESHOLD_WORLD if world_coords else OCCLUSION_IOU_THRESHOLD_IMAGE
            z_thr = OCCLUSION_Z_THRESHOLD_WORLD if world_coords else OCCLUSION_Z_THRESHOLD_IMAGE
            IOU_ENTER = iou_thr
            IOU_EXIT = iou_thr * 0.6
            Z_ENTER = z_thr
            Z_EXIT = z_thr * 0.6

            enter_left = (iou_cur > IOU_ENTER) and (z_diff > 0.0) and (behind > Z_ENTER)
            hold_left_occ = (iou_cur > IOU_EXIT) and (z_diff > 0.0) and (behind > Z_EXIT)
            enter_right = (iou_cur > IOU_ENTER) and (z_diff < 0.0) and (behind > Z_ENTER)
            hold_right_occ = (iou_cur > IOU_EXIT) and (z_diff < 0.0) and (behind > Z_EXIT)

    if enter_left or hold_left_occ:
        left_state.occ_ttl = occ_hyst_frames
    else:
        left_state.occ_ttl = max(0, left_state.occ_ttl - 1)

    if enter_right or hold_right_occ:
        right_state.occ_ttl = occ_hyst_frames
    else:
        right_state.occ_ttl = max(0, right_state.occ_ttl - 1)

    if pose_side_ambiguous:
        cross_frames = max(2, min(int(occ_hyst_frames), CROSS_OCCLUSION_FRAMES))
        if left is None and left_state.last_export is not None:
            left_state.occ_ttl = max(left_state.occ_ttl, cross_frames)
        if right is None and right_state.last_export is not None:
            right_state.occ_ttl = max(right_state.occ_ttl, cross_frames)

    overlap_guard = False
    overlap_freeze = False
    overlap_freeze_reason = None
    overlap_freeze_side = None
    overlap_iou_guard = None
    overlap_iou_guard_src = None
    overlap_hand_z_diff = None
    overlap_iou_candidates = []
    if occlusion_iou is not None:
        overlap_iou_candidates.append((float(occlusion_iou), "occlusion"))
    if sp_overlap_iou_val is not None:
        overlap_iou_candidates.append((float(sp_overlap_iou_val), "sp"))
    if left_ref_px is not None and right_ref_px is not None:
        ref_iou = _overlap_iou(left_ref_px, right_ref_px, fallback=0.0)
        overlap_iou_candidates.append((float(ref_iou), "ref"))
    if overlap_iou_candidates:
        overlap_iou_guard, overlap_iou_guard_src = max(overlap_iou_candidates, key=lambda item: item[0])
        overlap_guard = overlap_iou_guard >= OVERLAP_GUARD_IOU
    if overlap_guard_pre:
        overlap_guard = True
        if (
            overlap_iou_guard is None
            or (sp_overlap_iou_val is not None and sp_overlap_iou_val >= overlap_iou_guard)
        ):
            overlap_iou_guard = float(sp_overlap_iou_val)
            overlap_iou_guard_src = "sp_pre"
    if overlap_guard:
        overlap_freeze = True
        cross_frames = max(2, min(int(occ_hyst_frames), CROSS_OCCLUSION_FRAMES))
        decided = False

        def _apply_freeze(side: str, reason: str):
            nonlocal overlap_freeze_side, overlap_freeze_reason
            overlap_freeze_side = side
            overlap_freeze_reason = reason
            sample_state.last_overlap_freeze_side = side
            sample_state.last_overlap_freeze_i = i
            if side == "left":
                left_state.occ_ttl = max(left_state.occ_ttl, cross_frames)
                right_state.occ_ttl = 0
            elif side == "right":
                right_state.occ_ttl = max(right_state.occ_ttl, cross_frames)
                left_state.occ_ttl = 0
            else:
                left_state.occ_ttl = max(left_state.occ_ttl, cross_frames)
                right_state.occ_ttl = max(right_state.occ_ttl, cross_frames)

        def _median_hand_z(hand_pts):
            if not hand_pts or not isinstance(hand_pts, list):
                return None
            zs = [p.get("z") for p in hand_pts if isinstance(p, dict) and p.get("z") is not None]
            if not zs:
                return None
            zs.sort()
            mid = len(zs) // 2
            if len(zs) % 2:
                return float(zs[mid])
            return float(0.5 * (zs[mid - 1] + zs[mid]))

        left_z_src = left if left is not None else left_state.last_export
        right_z_src = right if right is not None else right_state.last_export
        if left_z_src is not None and right_z_src is not None:
            try:
                lz_med = _median_hand_z(left_z_src)
                rz_med = _median_hand_z(right_z_src)
                if lz_med is not None and rz_med is not None:
                    overlap_hand_z_diff = float(lz_med - rz_med)
            except Exception:
                overlap_hand_z_diff = None
        if overlap_hand_z_diff is not None and abs(overlap_hand_z_diff) >= OVERLAP_Z_MIN:
            if overlap_hand_z_diff > 0:
                _apply_freeze("left", "hand_z")
                decided = True
            elif overlap_hand_z_diff < 0:
                _apply_freeze("right", "hand_z")
                decided = True

        if not decided and occlusion_samples_ok and occlusion_z_diff is not None:
            if occlusion_z_diff > 0:
                _apply_freeze("left", "z")
                decided = True
            elif occlusion_z_diff < 0:
                _apply_freeze("right", "z")
                decided = True
            elif sample_state.last_overlap_freeze_side and (i - sample_state.last_overlap_freeze_i) <= cross_frames:
                _apply_freeze(sample_state.last_overlap_freeze_side, "last")
                decided = True

        if not decided:
            left_depth_hint = _pick_wrist_depth(None, pose_img_landmarks, pose_world_full, "left", world_coords)
            right_depth_hint = _pick_wrist_depth(None, pose_img_landmarks, pose_world_full, "right", world_coords)
            if left_depth_hint is not None and right_depth_hint is not None:
                z_hint = float(left_depth_hint - right_depth_hint)
                overlap_z_hint = z_hint
                if z_hint > 0:
                    _apply_freeze("left", "z_hint")
                    decided = True
                elif z_hint < 0:
                    _apply_freeze("right", "z_hint")
                    decided = True
                elif sample_state.last_overlap_freeze_side and (i - sample_state.last_overlap_freeze_i) <= cross_frames:
                    _apply_freeze(sample_state.last_overlap_freeze_side, "last")
                    decided = True

        if not decided:
            if sample_state.last_overlap_freeze_side and (i - sample_state.last_overlap_freeze_i) <= cross_frames:
                _apply_freeze(sample_state.last_overlap_freeze_side, "last")
                decided = True

        if not decided:
            left_q = left_score_gate if left_score_gate is not None else 0.0
            right_q = right_score_gate if right_score_gate is not None else 0.0
            if left_q > (right_q + 0.05):
                _apply_freeze("right", "score")
            elif right_q > (left_q + 0.05):
                _apply_freeze("left", "score")
            else:
                _apply_freeze("left", "score")

    occluded_L = (left_state.occ_ttl > 0)
    occluded_R = (right_state.occ_ttl > 0)
    occ_freeze_max_frames = max(0, min(int(occ_hyst_frames), OCCLUSION_FREEZE_MAX_FRAMES))
    if not occluded_L:
        left_state.occ_freeze_age = 0
    if not occluded_R:
        right_state.occ_freeze_age = 0

    side_ok_left_accept = True
    side_ok_right_accept = True
    if overlap_guard or pose_side_ambiguous:
        side_ok_left_accept = _side_consistent(
            "left",
            left,
            world_coords,
            pose_img_landmarks,
            pose_world_full,
            last_pose_world_full,
            left_state.last_good_img,
            right_state.last_good_img,
            pose_side_reassign_ratio_eff,
            use_last_good=True,
        )
        side_ok_right_accept = _side_consistent(
            "right",
            right,
            world_coords,
            pose_img_landmarks,
            pose_world_full,
            last_pose_world_full,
            left_state.last_good_img,
            right_state.last_good_img,
            pose_side_reassign_ratio_eff,
            use_last_good=True,
        )

    if overlap_guard:
        freeze_side = overlap_freeze_side if overlap_freeze else None
        if left is not None and src1 in ("pass1", "pass2") and not side_ok_left_accept:
            if freeze_side == "right":
                pass
            else:
                left, left_score = None, None
                src1 = None
                left_roi = None
                cur_left_img, cur_left_px = None, None
                left_reject_reason = append_reject_reason(left_reject_reason, "pose_side_drop")
                if rec_l:
                    rec_l = False
                    if sp_rec_left > 0:
                        sp_rec_left -= 1
                sp_dbg_left = None
                sp_params_left = None
        if right is not None and src2 in ("pass1", "pass2") and not side_ok_right_accept:
            if freeze_side == "left":
                pass
            else:
                right, right_score = None, None
                src2 = None
                right_roi = None
                cur_right_img, cur_right_px = None, None
                right_reject_reason = append_reject_reason(right_reject_reason, "pose_side_drop")
                if rec_r:
                    rec_r = False
                    if sp_rec_right > 0:
                        sp_rec_right -= 1
                sp_dbg_right = None
                sp_params_right = None

    pose_quality_left = None
    pose_quality_right = None
    pose_ok_left = True
    pose_ok_right = True
    if left is not None and src1 in ("pass1", "pass2"):
        pose_quality_left = _pose_distance_quality(
            "left",
            left,
            world_coords,
            pose_img_landmarks,
            pose_world_full,
            last_pose_world_full,
        )
        if pose_dist_qual_min_eff > 0 and pose_quality_left is not None:
            pose_ok_left = pose_quality_left >= pose_dist_qual_min_eff
    if right is not None and src2 in ("pass1", "pass2"):
        pose_quality_right = _pose_distance_quality(
            "right",
            right,
            world_coords,
            pose_img_landmarks,
            pose_world_full,
            last_pose_world_full,
        )
        if pose_dist_qual_min_eff > 0 and pose_quality_right is not None:
            pose_ok_right = pose_quality_right >= pose_dist_qual_min_eff

    left_transition = apply_occlusion_transition(
        HandFrameState(
            landmarks=left,
            score=left_score,
            source=src1,
            reject_reason=left_reject_reason,
            cur_img=cur_left_img,
            cur_px=cur_left_px,
        ),
        side="left",
        occluded=occluded_L,
        occ_ttl=left_state.occ_ttl,
        occ_freeze_age=left_state.occ_freeze_age,
        hold=left_state.hold,
        overlap_guard=overlap_guard,
        overlap_freeze_side=overlap_freeze_side,
        score_source=score_source,
        hand_hi=hand_hi,
        anchor_score_eff=anchor_score_eff,
        score_gate=left_score_gate,
        pose_ok=pose_ok_left,
        side_ok_accept=side_ok_left_accept,
        det_img=cur_left_img,
        last_good_img=left_state.last_good_img,
        last_export=left_state.last_export,
        last_export_score=left_state.last_export_score,
        occ_freeze_max_frames=occ_freeze_max_frames,
        occ_return_k=occ_return_k,
        world_coords=world_coords,
        proc_w=proc_w,
        proc_h=proc_h,
        pose_img_landmarks=pose_img_landmarks,
        pose_world_full=pose_world_full,
        last_pose_world_full=last_pose_world_full,
    )
    left = left_transition.hand.landmarks
    left_score = left_transition.hand.score
    src1 = left_transition.hand.source
    left_reject_reason = left_transition.hand.reject_reason
    cur_left_img = left_transition.hand.cur_img
    cur_left_px = left_transition.hand.cur_px
    occluded_L = left_transition.occluded
    left_state.occ_ttl = left_transition.occ_ttl
    left_state.occ_freeze_age = left_transition.occ_freeze_age
    left_state.hold = left_transition.hold
    missing_pre_occ_left = left_transition.missing_pre_occ
    occlusion_saved_1 = left_transition.occlusion_saved

    right_transition = apply_occlusion_transition(
        HandFrameState(
            landmarks=right,
            score=right_score,
            source=src2,
            reject_reason=right_reject_reason,
            cur_img=cur_right_img,
            cur_px=cur_right_px,
        ),
        side="right",
        occluded=occluded_R,
        occ_ttl=right_state.occ_ttl,
        occ_freeze_age=right_state.occ_freeze_age,
        hold=right_state.hold,
        overlap_guard=overlap_guard,
        overlap_freeze_side=overlap_freeze_side,
        score_source=score_source,
        hand_hi=hand_hi,
        anchor_score_eff=anchor_score_eff,
        score_gate=right_score_gate,
        pose_ok=pose_ok_right,
        side_ok_accept=side_ok_right_accept,
        det_img=cur_right_img,
        last_good_img=right_state.last_good_img,
        last_export=right_state.last_export,
        last_export_score=right_state.last_export_score,
        occ_freeze_max_frames=occ_freeze_max_frames,
        occ_return_k=occ_return_k,
        world_coords=world_coords,
        proc_w=proc_w,
        proc_h=proc_h,
        pose_img_landmarks=pose_img_landmarks,
        pose_world_full=pose_world_full,
        last_pose_world_full=last_pose_world_full,
    )
    right = right_transition.hand.landmarks
    right_score = right_transition.hand.score
    src2 = right_transition.hand.source
    right_reject_reason = right_transition.hand.reject_reason
    cur_right_img = right_transition.hand.cur_img
    cur_right_px = right_transition.hand.cur_px
    occluded_R = right_transition.occluded
    right_state.occ_ttl = right_transition.occ_ttl
    right_state.occ_freeze_age = right_transition.occ_freeze_age
    right_state.hold = right_transition.hold
    missing_pre_occ_right = right_transition.missing_pre_occ
    occlusion_saved_2 = right_transition.occlusion_saved

    side_ok_left = True
    side_ok_right = True
    overlap_ambiguous = pose_side_ambiguous or overlap_guard
    if overlap_ambiguous:
        side_ok_left = _side_consistent(
            "left",
            left,
            world_coords,
            pose_img_landmarks,
            pose_world_full,
            last_pose_world_full,
            left_state.last_good_img,
            right_state.last_good_img,
            pose_side_reassign_ratio_eff,
            use_last_good=True,
        )
        side_ok_right = _side_consistent(
            "right",
            right,
            world_coords,
            pose_img_landmarks,
            pose_world_full,
            last_pose_world_full,
            left_state.last_good_img,
            right_state.last_good_img,
            pose_side_reassign_ratio_eff,
            use_last_good=True,
        )

    left_state.note_observation(
        i,
        landmarks=left,
        source=src1,
        overlap_ambiguous=overlap_ambiguous,
        side_ok=side_ok_left,
        overlap_guard=overlap_guard,
    )
    right_state.note_observation(
        i,
        landmarks=right,
        source=src2,
        overlap_ambiguous=overlap_ambiguous,
        side_ok=side_ok_right,
        overlap_guard=overlap_guard,
    )

    block_track_left = bool(overlap_guard and left is None)
    block_track_right = bool(overlap_guard and right is None)

    track_ok_left = False
    track_ok_right = False
    track_reset_left = False
    track_reset_right = False
    tracker_last_score_left = getattr(tracker_left, "last_score", None)
    tracker_last_score_right = getattr(tracker_right, "last_score", None)
    tracker_last_ts_left = getattr(tracker_left, "last_valid_ts", None)
    tracker_last_ts_right = getattr(tracker_right, "last_valid_ts", None)

    left_track_transition = update_or_track_hand(
        HandFrameState(
            landmarks=left,
            score=left_score,
            source=src1,
            reject_reason=left_reject_reason,
            cur_img=cur_left_img,
            cur_px=cur_left_px,
        ),
        tracker=tracker_left,
        tracker_ready=left_state.tracker_ready,
        track_age=left_state.track_age,
        hold=left_state.hold,
        world_coords=world_coords,
        overlap_ambiguous=overlap_ambiguous,
        side_ok=side_ok_left,
        overlap_guard=overlap_guard,
        pose_ok=pose_ok_left,
        block_track=block_track_left,
        tracker_init_score_eff=tracker_init_score_eff,
        tracker_update_score_eff=tracker_update_score_eff,
        score_gate=left_score_gate,
        ts=ts,
        dt=dt,
        rgb=rgb,
        track_reset_ms=track_reset_ms,
        track_max_gap=track_max_gap,
        track_score_decay=track_score_decay,
    )
    left = left_track_transition.hand.landmarks
    left_score = left_track_transition.hand.score
    src1 = left_track_transition.hand.source
    left_reject_reason = left_track_transition.hand.reject_reason
    cur_left_img = left_track_transition.hand.cur_img
    cur_left_px = left_track_transition.hand.cur_px
    left_state.tracker_ready = left_track_transition.tracker_ready
    left_state.track_age = left_track_transition.track_age
    left_state.hold = left_track_transition.hold
    track_ok_left = left_track_transition.track_ok
    track_reset_left = left_track_transition.track_reset
    left_state.track_recovered += left_track_transition.track_recovered_inc

    right_track_transition = update_or_track_hand(
        HandFrameState(
            landmarks=right,
            score=right_score,
            source=src2,
            reject_reason=right_reject_reason,
            cur_img=cur_right_img,
            cur_px=cur_right_px,
        ),
        tracker=tracker_right,
        tracker_ready=right_state.tracker_ready,
        track_age=right_state.track_age,
        hold=right_state.hold,
        world_coords=world_coords,
        overlap_ambiguous=overlap_ambiguous,
        side_ok=side_ok_right,
        overlap_guard=overlap_guard,
        pose_ok=pose_ok_right,
        block_track=block_track_right,
        tracker_init_score_eff=tracker_init_score_eff,
        tracker_update_score_eff=tracker_update_score_eff,
        score_gate=right_score_gate,
        ts=ts,
        dt=dt,
        rgb=rgb,
        track_reset_ms=track_reset_ms,
        track_max_gap=track_max_gap,
        track_score_decay=track_score_decay,
    )
    right = right_track_transition.hand.landmarks
    right_score = right_track_transition.hand.score
    src2 = right_track_transition.hand.source
    right_reject_reason = right_track_transition.hand.reject_reason
    cur_right_img = right_track_transition.hand.cur_img
    cur_right_px = right_track_transition.hand.cur_px
    right_state.tracker_ready = right_track_transition.tracker_ready
    right_state.track_age = right_track_transition.track_age
    right_state.hold = right_track_transition.hold
    track_ok_right = right_track_transition.track_ok
    track_reset_right = right_track_transition.track_reset
    right_state.track_recovered += right_track_transition.track_recovered_inc

    # Post-tracking dedup: if a tracked hand overlaps a detected hand, keep the detection.
    if left is not None and right is not None:
        is_left_tracked = (src1 == "tracked")
        is_right_tracked = (src2 == "tracked")
        if is_left_tracked != is_right_tracked:
            dist = _mean_l2_xy(left, right)
            scale = max(1e-6, 0.5 * (_hand_scale(left) + _hand_scale(right)))
            iou = _iou_norm(left, right)
            overlap_strong = (
                (dist is not None and dist < DEDUP_DIST_SCALE * scale) or
                (iou > DEDUP_IOU_THRESHOLD)
            )
            if overlap_strong:
                if is_left_tracked:
                    left, left_score = None, None
                    src1 = None
                    if not dedup_triggered:
                        dedup_triggered = True
                        dedup_removed = "hand_1"
                        dedup_iou = float(iou)
                        dedup_dist_norm = float(dist / scale) if dist is not None else None
                else:
                    right, right_score = None, None
                    src2 = None
                    if not dedup_triggered:
                        dedup_triggered = True
                        dedup_removed = "hand_2"
                        dedup_iou = float(iou)
                        dedup_dist_norm = float(dist / scale) if dist is not None else None


    if _ALIGN_HANDS_TO_POSE and world_coords and (left is not None or right is not None):
        # pose_world_full was already computed earlier in this frame (line 732)
        if pose_world_full and isinstance(pose_world_full, list) and len(pose_world_full) >= 17:
            l_wrist = pose_world_full[15]
            r_wrist = pose_world_full[16]
            if left is not None:
                left = align_hand_xy_to_target(left, l_wrist)
            if right is not None:
                right = align_hand_xy_to_target(right, r_wrist)

    iou_hands = None
    if left is not None and right is not None:
        if world_coords and cur_left_px is not None and cur_right_px is not None:
            iou_hands = _overlap_iou(cur_left_px, cur_right_px, fallback=0.0)
        else:
            iou_hands = _overlap_iou(left, right, fallback=0.0)

    pose_wrist_z_world_left = None
    pose_wrist_z_world_right = None
    if pose_world_full and isinstance(pose_world_full, list) and len(pose_world_full) >= 17:
        pose_wrist_z_world_left = float(pose_world_full[MP_POSE_LEFT_WRIST_IDX]["z"])
        pose_wrist_z_world_right = float(pose_world_full[MP_POSE_RIGHT_WRIST_IDX]["z"])

    pose_wrist_z_img_left = None
    pose_wrist_z_img_right = None
    if pose_img_landmarks is not None and len(pose_img_landmarks) >= 17:
        pose_wrist_z_img_left = float(pose_img_landmarks[MP_POSE_LEFT_WRIST_IDX].z)
        pose_wrist_z_img_right = float(pose_img_landmarks[MP_POSE_RIGHT_WRIST_IDX].z)

    hand_wrist_z_left = None
    if left is not None and isinstance(left, list) and len(left) > 0 and "z" in left[0]:
        hand_wrist_z_left = float(left[0]["z"])
    hand_wrist_z_right = None
    if right is not None and isinstance(right, list) and len(right) > 0 and "z" in right[0]:
        hand_wrist_z_right = float(right[0]["z"])

    side_d_ll = None
    side_d_lr = None
    side_d_rl = None
    side_d_rr = None
    side_cost_current = None
    side_cost_swap = None
    side_pref_left = None
    side_pref_right = None

    if left is not None:
        if pose_side_ambiguous and not world_coords:
            dLL, dLR = _last_good_wrist_dists(left, left_state.last_good_img, right_state.last_good_img)
        else:
            dLL, dLR = _pose_wrist_dists(
                left, world_coords, pose_img_landmarks, pose_world_full, last_pose_world_full
            )
        side_d_ll, side_d_lr = dLL, dLR
        if dLL is not None and dLR is not None:
            if dLR < dLL * pose_side_reassign_ratio_eff:
                side_pref_left = "right"
            elif dLL < dLR * pose_side_reassign_ratio_eff:
                side_pref_left = "left"

    if right is not None:
        if pose_side_ambiguous and not world_coords:
            dRL, dRR = _last_good_wrist_dists(right, left_state.last_good_img, right_state.last_good_img)
        else:
            dRL, dRR = _pose_wrist_dists(
                right, world_coords, pose_img_landmarks, pose_world_full, last_pose_world_full
            )
        side_d_rl, side_d_rr = dRL, dRR
        if dRL is not None and dRR is not None:
            if dRL < dRR * pose_side_reassign_ratio_eff:
                side_pref_right = "left"
            elif dRR < dRL * pose_side_reassign_ratio_eff:
                side_pref_right = "right"

    if None not in (side_d_ll, side_d_lr, side_d_rl, side_d_rr):
        side_cost_current = side_d_ll + side_d_rr
        side_cost_swap = side_d_lr + side_d_rl

    pose_wrist_dist_1 = None
    pose_wrist_dist_2 = None
    Lref, Rref = _pose_wrist_refs(world_coords, pose_img_landmarks, pose_world_full, last_pose_world_full)
    if Lref is not None and left is not None:
        lxy = _wrist_xy(left)
        if lxy is not None:
            lx, ly = lxy
            pose_wrist_dist_1 = math.hypot(lx - Lref[0], ly - Lref[1])
    if Rref is not None and right is not None:
        rxy = _wrist_xy(right)
        if rxy is not None:
            rx, ry = rxy
            pose_wrist_dist_2 = math.hypot(rx - Rref[0], ry - Rref[1])

    left_state.maybe_export(
        landmarks=left,
        score=left_score,
        source=src1,
        overlap_ambiguous=overlap_ambiguous,
        side_ok=side_ok_left,
        overlap_guard=overlap_guard,
        cur_px=cur_left_px,
        cur_img=cur_left_img,
    )
    right_state.maybe_export(
        landmarks=right,
        score=right_score,
        source=src2,
        overlap_ambiguous=overlap_ambiguous,
        side_ok=side_ok_right,
        overlap_guard=overlap_guard,
        cur_px=cur_right_px,
        cur_img=cur_right_img,
    )

    hand1_is_anchor = left_state.maybe_anchor(
        i,
        landmarks=left,
        score=left_score,
        source=src1,
        anchor_score=anchor_score_eff,
        pose_ok=pose_ok_left,
        overlap_ambiguous=overlap_ambiguous,
        side_ok=side_ok_left,
        overlap_guard=overlap_guard,
    )
    hand2_is_anchor = right_state.maybe_anchor(
        i,
        landmarks=right,
        score=right_score,
        source=src2,
        anchor_score=anchor_score_eff,
        pose_ok=pose_ok_right,
        overlap_ambiguous=overlap_ambiguous,
        side_ok=side_ok_right,
        overlap_guard=overlap_guard,
    )

    hand1_state = classify_hand_state(left, src1)
    hand2_state = classify_hand_state(right, src2)

    record = build_runtime_frame_record(
        frame_idx=i,
        ts_ms=ts,
        dt_ms=dt,
        left=left,
        right=right,
        left_score=left_score,
        right_score=right_score,
        left_score_gate=left_score_gate,
        right_score_gate=right_score_gate,
        src1=src1,
        src2=src2,
        hand1_state=hand1_state,
        hand2_state=hand2_state,
        hand1_is_anchor=hand1_is_anchor,
        hand2_is_anchor=hand2_is_anchor,
        left_reject_reason=left_reject_reason,
        right_reject_reason=right_reject_reason,
        pose_quality_left=pose_quality_left,
        pose_quality_right=pose_quality_right,
        hand_wrist_z_left=hand_wrist_z_left,
        hand_wrist_z_right=hand_wrist_z_right,
        sanity_dbg_left=sanity_dbg_left,
        sanity_dbg_right=sanity_dbg_right,
        sanity_stage_left=sanity_stage_left,
        sanity_stage_right=sanity_stage_right,
        track_age_left=left_state.track_age,
        track_age_right=right_state.track_age,
        track_reset_left=track_reset_left,
        track_reset_right=track_reset_right,
        tracker_left_ready=left_state.tracker_ready,
        tracker_right_ready=right_state.tracker_ready,
        tracker_last_score_left=tracker_last_score_left,
        tracker_last_score_right=tracker_last_score_right,
        tracker_last_ts_left=tracker_last_ts_left,
        tracker_last_ts_right=tracker_last_ts_right,
        swap_applied=swap_applied,
        dedup_triggered=dedup_triggered,
        dedup_removed=dedup_removed,
        dedup_iou=dedup_iou,
        dedup_dist_norm=dedup_dist_norm,
        track_ok_left=track_ok_left,
        track_ok_right=track_ok_right,
        sp_attempt_left=sp_attempt_left,
        sp_attempt_right=sp_attempt_right,
        rec_l=rec_l,
        rec_r=rec_r,
        sp_overlap_iou_val=sp_overlap_iou_val,
        overlap=overlap,
        iou_hands=iou_hands,
        occlusion_iou=occlusion_iou,
        overlap_guard_pre=overlap_guard_pre,
        overlap_guard=overlap_guard,
        overlap_freeze=overlap_freeze,
        overlap_freeze_reason=overlap_freeze_reason,
        overlap_freeze_side=overlap_freeze_side,
        overlap_iou_guard=overlap_iou_guard,
        overlap_iou_guard_src=overlap_iou_guard_src,
        overlap_z_hint=overlap_z_hint,
        occlusion_z_diff=occlusion_z_diff,
        overlap_hand_z_diff=overlap_hand_z_diff,
        pose_side_ambiguous=pose_side_ambiguous,
        pose_side_reassign_ratio_eff=pose_side_reassign_ratio_eff,
        side_d_ll=side_d_ll,
        side_d_lr=side_d_lr,
        side_d_rl=side_d_rl,
        side_d_rr=side_d_rr,
        side_cost_current=side_cost_current,
        side_cost_swap=side_cost_swap,
        side_pref_left=side_pref_left,
        side_pref_right=side_pref_right,
        side_ok_left_accept=side_ok_left_accept,
        side_ok_right_accept=side_ok_right_accept,
        occluded_L=occluded_L,
        occluded_R=occluded_R,
        occ_ttl_left=left_state.occ_ttl,
        occ_ttl_right=right_state.occ_ttl,
        occ_freeze_age_left=left_state.occ_freeze_age,
        occ_freeze_age_right=right_state.occ_freeze_age,
        occ_freeze_max_frames=occ_freeze_max_frames,
        occlusion_saved_1=occlusion_saved_1,
        occlusion_saved_2=occlusion_saved_2,
        missing_pre_occ_left=missing_pre_occ_left,
        missing_pre_occ_right=missing_pre_occ_right,
        occlusion_behind_diff=occlusion_behind_diff,
        occlusion_samples_ok=occlusion_samples_ok,
        pose_wrist_dist_1=pose_wrist_dist_1,
        pose_wrist_dist_2=pose_wrist_dist_2,
        pose_wrist_z_world_left=pose_wrist_z_world_left,
        pose_wrist_z_world_right=pose_wrist_z_world_right,
        pose_wrist_z_img_left=pose_wrist_z_img_left,
        pose_wrist_z_img_right=pose_wrist_z_img_right,
        score_source=score_source,
        min_hand_score=min_hand_score,
        hand_lo=hand_lo,
        hand_hi=hand_hi,
        anchor_score_eff=anchor_score_eff,
        tracker_init_score_eff=tracker_init_score_eff,
        tracker_update_score_eff=tracker_update_score_eff,
        pose_dist_qual_min_eff=pose_dist_qual_min_eff,
        pose_xyz=pose_xyz,
        pose_vis=pose_vis,
        pose_interpolated=pose_interpolated,
        write_hand_mask=write_hand_mask,
        left_roi=left_roi if sp_debug_roi else None,
        right_roi=right_roi if sp_debug_roi else None,
        sp_dbg_left=sp_dbg_left if sp_debug_roi else None,
        sp_dbg_right=sp_dbg_right if sp_debug_roi else None,
        sp_params_left=sp_params_left if sp_debug_roi else None,
        sp_params_right=sp_params_right if sp_debug_roi else None,
    )

    context.pose_state = pose_state
    return FrameStepResult(
        record=record,
        hand_runtime=float(hand_runtime),
        pose_runtime=float(pose_runtime),
        second_pass_runtime=float(second_pass_runtime),
        sp_missing_pre_left=bool(missing_pre_sp_left),
        sp_missing_pre_right=bool(missing_pre_sp_right),
        left=SideStepResult(sp_recovered=bool(rec_l)),
        right=SideStepResult(sp_recovered=bool(rec_r)),
    )
