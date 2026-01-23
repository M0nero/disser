from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import numpy as np
from .. import _env  # ensure caps before cv2 import
import cv2
import math
from time import perf_counter
from ..core.logging_utils import get_logger, log_metrics

from ..mp.mp_utils import (
    try_import_mediapipe,
    align_hand_xy_to_target,
    _ALIGN_HANDS_TO_POSE,
    create_hand_detector,
    create_pose_detector,
    normalize_backend_name,
    normalize_tasks_delegate,
    resolve_task_model_path,
)
from ..core.utils import (
    xyz_list_from_lms,
    pick_pose_indices,
    resize_short_side,
    norm_to_px,
    px_to_norm,
    bbox_from_pts_px,
)
from ..mp.roi import run_second_pass_for, hands_up_gate
from ..core.types import VideoMeta
from ..algos.tracking import HandTracker, smooth_tracks
from ..algos.sanity import check_hand_sanity
from ..algos.postprocess import postprocess_sequence

from .process_assignment import (
    _antiswap_and_dedup,
    _hand_too_far_from_pose,
    _last_good_wrist_dists,
    _reject_hands_far_from_pose,
    _side_consistent,
    _swap_by_last_good,
)
from .process_constants import (
    CROSS_OCCLUSION_FRAMES,
    DELTA_QUALITY,
    DEDUP_D0_SCALE,
    DEDUP_DIST_SCALE,
    DEDUP_IOU_THRESHOLD,
    HAND_MIN_DETECTION_CONFIDENCE,
    HAND_MIN_TRACKING_CONFIDENCE,
    HAND_SP_MIN_DETECTION,
    HAND_SP_MIN_TRACKING,
    MIN_QUALITY,
    MP_HAND_NUM_LANDMARKS,
    MP_POSE_LEFT_WRIST_IDX,
    MP_POSE_NUM_LANDMARKS,
    MP_POSE_RIGHT_WRIST_IDX,
    OCCLUSION_FREEZE_MAX_FRAMES,
    OCCLUSION_HYST_FRAMES,
    OCCLUSION_HYST_IOU_SCALE,
    OCCLUSION_IOU_THRESHOLD_IMAGE,
    OCCLUSION_IOU_THRESHOLD_WORLD,
    OCCLUSION_Z_THRESHOLD_IMAGE,
    OCCLUSION_Z_THRESHOLD_WORLD,
    OVERLAP_GUARD_IOU,
    OVERLAP_Z_MIN,
    POSE_SIDE_REASSIGN_RATIO,
    POSE_WRIST_CROSS_SCALE,
    POSE_WRIST_MIN_VIS,
    POSE_WRIST_OFF_MARGIN,
    POSE_WRIST_REJECT_MIN_IMAGE,
    POSE_WRIST_REJECT_MIN_WORLD,
    POSE_WRIST_REJECT_SCALE,
    SP_MIN_DET_MULTIPLIER,
    SP_MIN_TRACK_MULTIPLIER,
)
from .process_geometry import (
    _bbox_norm,
    _hand_scale,
    _iou_norm,
    _mean_l2_xy,
    _wrist_xy,
)
from .process_init import _initialize_detectors, _resolve_model_paths
from .process_metrics import _count_outliers
from .process_models import FrameData, Landmark
from .process_occlusion import (
    _overlap_iou,
    _pick_wrist_depth,
    _resolve_pose_world,
    is_hand_occluded,
)
from .process_pose import (
    _center_hint_for,
    _hand_expected,
    _pose_distance_quality,
    _pose_gate_allows_second_pass,
    _pose_wrist_dists,
    _pose_wrist_low_visibility,
    _pose_wrist_out_of_frame,
    _pose_wrist_px,
    _pose_wrist_refs,
    _pose_wrists_close,
)
from .process_second_pass import _apply_hold_if_needed, _execute_second_pass

LOGGER = get_logger(__name__)


def process_video(
    path: Path,
    out_path: Path,
    world_coords: bool,
    keep_pose_indices: Optional[List[int]],
    stride: int,
    short_side: Optional[int],
    min_det: float,
    min_track: float,
    pose_every: int,
    pose_complexity: int,
    ts_source: str,
    min_hand_score: float = 0.0,
    hand_score_lo: float = 0.55,
    hand_score_hi: float = 0.90,
    hand_score_source: str = "handedness",
    tracker_init_score: float = -1.0,
    anchor_score: float = -1.0,
    pose_dist_qual_min: float = 0.50,
    tracker_update_score: float = -1.0,
    pose_side_reassign_ratio: float = 0.85,
    pose_ema_alpha: float = 0.0,
    ndjson_path: Optional[Path] = None,
    # second pass controls
    second_pass: bool = False,
    sp_trigger_below: float = 0.50,
    sp_roi_frac: float = 0.25,
    sp_margin: float = 0.35,
    sp_escalate_step: float = 0.25,
    sp_escalate_max: float = 2.0,
    sp_hands_up_only: bool = False,
    sp_jitter_px: int = 0,
    sp_jitter_rings: int = 1,
    sp_center_penalty: float = 0.30,
    sp_label_relax: float = 0.15,
    sp_overlap_iou: float = 0.15,
    sp_overlap_shrink: float = 0.70,
    sp_overlap_penalty_mult: float = 2.0,
    sp_overlap_require_label: bool = False,
    sp_debug_roi: bool = False,
    # extras
    interp_hold: int = 0,
    track_max_gap: int = 15,
    track_score_decay: float = 0.90,
    track_reset_ms: int = 250,
    write_hand_mask: bool = False,
    mp_backend: str = "solutions",
    hand_task: Optional[str] = None,
    pose_task: Optional[str] = None,
    mp_tasks_delegate: Optional[str] = None,
    debug_video_path: Optional[Path] = None,
    eval_mode: bool = False,
    seed: int = 0,
    occ_hyst_frames: int = 15,
    occ_return_k: float = 1.2,
    sanity_enable: bool = True,
    sanity_scale_range: Tuple[float, float] = (0.70, 1.35),
    sanity_wrist_k: float = 2.0,
    sanity_bone_tol: float = 0.30,
    sanity_pass2: bool = False,
    sanity_anchor_max_gap: int = 0,
    sanitize_rejects: bool = True,
    postprocess: bool = False,
    pp_max_gap: int = 15,
    pp_smoother: str = "ema",
    pp_only_anchors: bool = True,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    segment_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    mp, mp_solutions = try_import_mediapipe()
    backend = normalize_backend_name(mp_backend)
    delegate_raw = mp_tasks_delegate or "auto"
    tasks_delegate = normalize_tasks_delegate(delegate_raw)
    hand_lo = max(min_hand_score, hand_score_lo)
    hand_lo = min(1.0, max(0.0, hand_lo))
    hand_hi = max(hand_lo, hand_score_hi)
    hand_hi = min(1.0, max(0.0, hand_hi))
    score_source = (hand_score_source or "handedness").strip().lower()
    if score_source not in {"handedness", "presence"}:
        score_source = "handedness"
    tracker_init_score_eff = hand_hi if tracker_init_score < 0 else tracker_init_score
    tracker_init_score_eff = min(1.0, max(hand_lo, tracker_init_score_eff))
    anchor_score_eff = hand_hi if anchor_score < 0 else anchor_score
    anchor_score_eff = min(1.0, max(0.0, anchor_score_eff))
    pose_dist_qual_min_eff = min(1.0, max(0.0, pose_dist_qual_min))
    tracker_update_score_eff = None
    if tracker_update_score >= 0:
        tracker_update_score_eff = min(1.0, max(0.0, tracker_update_score))
    pose_side_reassign_ratio_eff = POSE_SIDE_REASSIGN_RATIO
    if pose_side_reassign_ratio > 0:
        pose_side_reassign_ratio_eff = pose_side_reassign_ratio
    pose_side_reassign_ratio_eff = min(1.0, max(0.5, pose_side_reassign_ratio_eff))

    log_metrics(LOGGER, "process_video.start", {
        "video": path.name,
        "out_path": str(out_path),
        "world_coords": bool(world_coords),
        "stride": stride,
        "short_side": short_side,
        "pose_every": pose_every,
        "pose_complexity": pose_complexity,
        "second_pass": bool(second_pass),
        "sp_debug_roi": bool(sp_debug_roi),
        "mp_backend": backend,
        "mp_tasks_delegate": tasks_delegate,
        "ndjson": str(ndjson_path) if ndjson_path else None,
        "debug_video": str(debug_video_path) if debug_video_path else None,
        "eval_mode": bool(eval_mode),
        "seed": int(seed),
        "frame_start": frame_start,
        "frame_end": frame_end,
    })

    orig_frame_start = frame_start
    orig_frame_end = frame_end
    try:
        frame_start_i = int(frame_start) if frame_start is not None else 0
    except Exception as exc:
        raise RuntimeError(f"Invalid frame_start: {frame_start}") from exc
    try:
        frame_end_i = int(frame_end) if frame_end is not None else None
    except Exception as exc:
        raise RuntimeError(f"Invalid frame_end: {frame_end}") from exc
    if frame_start_i < 0:
        frame_start_i = 0
    if frame_end_i is not None and frame_end_i <= frame_start_i:
        raise RuntimeError(f"Invalid frame range: start={frame_start_i}, end={frame_end_i}")

    hand_model_path, pose_model_path = _resolve_model_paths(backend, hand_task, pose_task)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        log_metrics(LOGGER, "process_video.error", {
            "video": path.name,
            "reason": "open_failed",
            "path": str(path),
        })
        raise RuntimeError(f"Failed to open video: {path}")

    if frame_start_i > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start_i)

    width_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0

    hands_detector, pose_detector, hands_sp = _initialize_detectors(
        backend, mp, mp_solutions,
        hand_model_path, pose_model_path,
        min_det, min_track, pose_complexity,
        tasks_delegate, second_pass, world_coords
    )


    frames: List[Dict[str, Any]] = []
    i = frame_start_i
    proc_w = width_src
    proc_h = height_src

    ndjson_f = None
    if ndjson_path is not None:
        ndjson_path.parent.mkdir(parents=True, exist_ok=True)
        ndjson_f = open(ndjson_path, "wt", encoding="utf-8")
        log_metrics(LOGGER, "process_video.ndjson", {
            "video": path.name,
            "ndjson_path": str(ndjson_path),
        })

    def _emit_ndjson(obj: Dict[str, Any]):
        if ndjson_f is not None:
            ndjson_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    processing_started = perf_counter()
    pose_runtime = 0.0
    hand_runtime = 0.0
    second_pass_runtime = 0.0
    progress_interval = 500

    last_ts = None

    last_left_px: Optional[List[Dict[str, float]]] = None
    last_right_px: Optional[List[Dict[str, float]]] = None
    last_left_export: Optional[List[Dict[str, float]]] = None
    last_right_export: Optional[List[Dict[str, float]]] = None
    last_left_export_score = None
    last_right_export_score = None
    last_left_good_px = None
    last_right_good_px = None
    last_left_good_img = None
    last_right_good_img = None
    last_left_anchor: Optional[List[Dict[str, float]]] = None
    last_right_anchor: Optional[List[Dict[str, float]]] = None
    last_left_anchor_i: int = -10**9
    last_right_anchor_i: int = -10**9
    last_left_obs: Optional[List[Dict[str, float]]] = None
    last_right_obs: Optional[List[Dict[str, float]]] = None
    last_left_obs_i: int = -10**9
    last_right_obs_i: int = -10**9
    hold_left = 0
    hold_right = 0

    sp_rec_left = 0
    sp_rec_right = 0
    sp_missing_left_pre = 0
    sp_missing_right_pre = 0

    last_pose_xyz: Optional[List[Dict[str, float]]] = None
    last_pose_vis: Optional[List[float]] = None
    last_pose_world_full: Optional[List[Dict[str, float]]] = None
    last_pose_img_landmarks = None
    pose_ema: Optional[List[Dict[str, float]]] = None
    hands_frames_running = 0

    occl_left_cnt = 0
    occl_right_cnt = 0
    occ_ttl_left = 0
    occ_ttl_right = 0
    occ_freeze_age_left = 0
    occ_freeze_age_right = 0
    last_overlap_freeze_side: Optional[str] = None
    last_overlap_freeze_i: int = -10**9

    tracker_left = HandTracker()
    tracker_right = HandTracker()
    tracker_left_ready = False
    tracker_right_ready = False
    track_age_left = 0
    track_age_right = 0
    track_rec_left = 0
    track_rec_right = 0

    def _append_reject_reason(cur: Optional[str], reason: Optional[str]) -> Optional[str]:
        if not reason:
            return cur
        if cur:
            return f"{cur}; {reason}"
        return reason

    def _score_for_gate(score_raw: Optional[float], present: bool) -> Optional[float]:
        if not present:
            return None
        if score_source == "presence":
            return 1.0
        return score_raw

    try:
        if hands_detector is None or pose_detector is None:
            raise RuntimeError("Mediapipe detectors failed to initialize")

        while True:
            if frame_end_i is not None and i >= frame_end_i:
                break
            ok, bgr = cap.read()
            if not ok:
                break

            rel_i = i - frame_start_i
            if (rel_i % max(1, stride)) != 0:
                i += 1
                continue
            bgr = resize_short_side(bgr, short_side)
            proc_h, proc_w = bgr.shape[:2]

            ts = None
            if ts_source in ("auto", "pos_msec"):
                t = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                ts = int(round(t)) if t > 0 else None
            if ts is None:
                ts = int(round((i / max(fps, 1e-6)) * 1000.0))
            dt = 0 if last_ts is None else max(0, ts - last_ts)
            last_ts = ts

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            run_pose = (rel_i % max(1, pose_every)) == 0
            if run_pose:
                pose_t0 = perf_counter()
                rp = pose_detector.process(rgb, ts)
                pose_runtime += perf_counter() - pose_t0
            else:
                rp = None

            hand_t0 = perf_counter()
            rh = hands_detector.process(rgb, ts)
            hand_runtime += perf_counter() - hand_t0

            left = right = None
            left_score = right_score = None
            left_idx = right_idx = None
            left_reject_reason = None
            right_reject_reason = None
            src1 = None
            src2 = None

            lm_list_img = getattr(rh, "multi_hand_landmarks", None)
            lm_list_world = getattr(rh, "multi_hand_world_landmarks", None)

            if getattr(rh, "multi_handedness", None):
                for idx_hd, hd in enumerate(rh.multi_handedness):
                    label = str(hd.classification[0].label).lower()
                    score = float(hd.classification[0].score)
                    if "left" in label:
                        left_idx = left_idx if left_idx is not None else idx_hd
                        left_score = max(left_score or 0.0, score)
                    elif "right" in label:
                        right_idx = right_idx if right_idx is not None else idx_hd
                        right_score = max(right_score or 0.0, score)

            lm_list = lm_list_world if world_coords else lm_list_img
            lm_list = lm_list or []

            if left_idx is not None and 0 <= left_idx < len(lm_list):
                left = xyz_list_from_lms(lm_list[left_idx])
            if right_idx is not None and 0 <= right_idx < len(lm_list):
                right = xyz_list_from_lms(lm_list[right_idx])

            prev_pred_left = last_left_obs if (i - last_left_obs_i == 1) else None
            prev_pred_right = last_right_obs if (i - last_right_obs_i == 1) else None
            anchor_left = last_left_anchor
            anchor_right = last_right_anchor
            if sanity_anchor_max_gap > 0:
                if (i - last_left_anchor_i) > sanity_anchor_max_gap:
                    anchor_left = None
                if (i - last_right_anchor_i) > sanity_anchor_max_gap:
                    anchor_right = None

            left_score_gate = _score_for_gate(left_score, left is not None)
            right_score_gate = _score_for_gate(right_score, right is not None)

            if left is not None and (left_score_gate is None or left_score_gate < min_hand_score):
                left, left_score = None, None
            if right is not None and (right_score_gate is None or right_score_gate < min_hand_score):
                right, right_score = None, None

            pose_xyz = None
            pose_vis = None
            pose_interpolated = False
            pose_img_landmarks = None
            pose_world_current = None

            if run_pose and rp is not None:
                if getattr(rp, "pose_world_landmarks", None):
                    pose_world_current = xyz_list_from_lms(rp.pose_world_landmarks)
                    last_pose_world_full = pose_world_current
                    if world_coords:
                        pose_xyz = pose_world_current
                if getattr(rp, "pose_landmarks", None):
                    if pose_xyz is None:
                        pose_xyz = xyz_list_from_lms(rp.pose_landmarks)

                if getattr(rp, "pose_landmarks", None):
                    pose_img_landmarks = rp.pose_landmarks.landmark
                    last_pose_img_landmarks = pose_img_landmarks
                    idxs = keep_pose_indices if keep_pose_indices is not None else range(33)
                    lm = rp.pose_landmarks.landmark
                    pose_vis = []
                    for idx_p in idxs:
                        if 0 <= idx_p < len(lm):
                            pose_vis.append(float(lm[idx_p].visibility))
                        else:
                            pose_vis.append(0.0)

                pose_xyz = pick_pose_indices(pose_xyz, keep_pose_indices)

                if pose_xyz is not None:
                    last_pose_xyz = pose_xyz
                if pose_vis is not None:
                    last_pose_vis = pose_vis

            if pose_xyz is None and last_pose_xyz is not None:
                pose_xyz = last_pose_xyz
                pose_vis = last_pose_vis
                pose_interpolated = True

            if pose_img_landmarks is None and last_pose_img_landmarks is not None:
                pose_img_landmarks = last_pose_img_landmarks

            if pose_xyz is not None and pose_ema_alpha > 0.0:
                if pose_ema is None:
                    pose_ema = pose_xyz
                else:
                    smoothed: List[Dict[str, float]] = []
                    for p_new, p_old in zip(pose_xyz, pose_ema):
                        smoothed.append({
                            "x": pose_ema_alpha * p_new["x"] + (1.0 - pose_ema_alpha) * p_old["x"],
                            "y": pose_ema_alpha * p_new["y"] + (1.0 - pose_ema_alpha) * p_old["y"],
                            "z": pose_ema_alpha * p_new["z"] + (1.0 - pose_ema_alpha) * p_old["z"],
                        })
                    pose_ema = smoothed
                pose_xyz = pose_ema

            # Track pixel-space coordinates for IoU calculations in occlusion detection.
            # These are always maintained regardless of world_coords mode because
            # IoU calculations are more accurate in pixel space.
            cur_left_img = None
            cur_right_img = None
            cur_left_px = None
            cur_right_px = None
            if lm_list_img is not None:
                H, W = proc_h, proc_w
                if left_idx is not None and 0 <= left_idx < len(lm_list_img):
                    cur_left_img = xyz_list_from_lms(lm_list_img[left_idx]) or None
                    if cur_left_img:
                        cur_left_px = norm_to_px(cur_left_img, W, H)
                if right_idx is not None and 0 <= right_idx < len(lm_list_img):
                    cur_right_img = xyz_list_from_lms(lm_list_img[right_idx]) or None
                    if cur_right_img:
                        cur_right_px = norm_to_px(cur_right_img, W, H)

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

            if left is not None and sanity_enable:
                sanity_dbg_left = {}
                sanity_stage_left = "pass1"
                ok, reasons = check_hand_sanity(
                    left,
                    prev_anchor=anchor_left,
                    prev_pred=prev_pred_left,
                    world_coords=world_coords,
                    scale_range=sanity_scale_range,
                    wrist_k=sanity_wrist_k,
                    bone_tol=sanity_bone_tol,
                    debug_out=sanity_dbg_left,
                )
                if not ok:
                    left, left_score = None, None
                    src1 = None
                    sanity_msg = "sanity:" + "|".join(reasons)
                    left_reject_reason = _append_reject_reason(left_reject_reason, sanity_msg)
            if right is not None and sanity_enable:
                sanity_dbg_right = {}
                sanity_stage_right = "pass1"
                ok, reasons = check_hand_sanity(
                    right,
                    prev_anchor=anchor_right,
                    prev_pred=prev_pred_right,
                    world_coords=world_coords,
                    scale_range=sanity_scale_range,
                    wrist_k=sanity_wrist_k,
                    bone_tol=sanity_bone_tol,
                    debug_out=sanity_dbg_right,
                )
                if not ok:
                    right, right_score = None, None
                    src2 = None
                    sanity_msg = "sanity:" + "|".join(reasons)
                    right_reject_reason = _append_reject_reason(right_reject_reason, sanity_msg)

            src1 = "pass1" if (left is not None and left_score is not None) else None
            src2 = "pass1" if (right is not None and right_score is not None) else None

            sp_left_ref_px = cur_left_px if cur_left_px is not None else last_left_good_px
            sp_right_ref_px = cur_right_px if cur_right_px is not None else last_right_good_px
            if sp_left_ref_px is not None and sp_right_ref_px is not None:
                overlap_iou = _overlap_iou(sp_left_ref_px, sp_right_ref_px, fallback=0.0)
            else:
                overlap_iou = 0.0
            sp_overlap_iou_val = float(overlap_iou) if overlap_iou is not None else 0.0
            overlap_guard_pre = sp_overlap_iou_val >= OVERLAP_GUARD_IOU
            overlap = (sp_overlap_iou_val >= sp_overlap_iou)
            sp_roi_frac_eff = sp_roi_frac * (sp_overlap_shrink if overlap else 1.0)
            sp_center_penalty_eff = sp_center_penalty * (sp_overlap_penalty_mult if overlap else 1.0)
            sp_label_relax_eff = 0.0 if overlap else sp_label_relax
            sp_require_label_eff = bool(sp_overlap_require_label and overlap)
            max_center_dist_norm_eff = 0.9 if overlap else 1.25
            sp_params_eff = {
                "sp_roi_frac": float(sp_roi_frac_eff),
                "center_penalty_lambda": float(sp_center_penalty_eff),
                "label_relax_margin": float(sp_label_relax_eff),
                "require_label_match": bool(sp_require_label_eff),
                "max_center_dist_norm": float(max_center_dist_norm_eff),
                "sp_jitter_px": int(sp_jitter_px),
                "sp_jitter_rings": int(sp_jitter_rings),
            }
            sp_strict = bool(overlap_guard_pre)
            sp_roi_frac_strict = sp_roi_frac_eff * 0.7
            sp_center_penalty_strict = sp_center_penalty_eff * 1.5
            sp_label_relax_strict = 0.0
            sp_require_label_strict = True
            max_center_dist_norm_strict = min(max_center_dist_norm_eff, 0.6)
            sp_params_eff_strict = {
                "sp_roi_frac": float(sp_roi_frac_strict),
                "center_penalty_lambda": float(sp_center_penalty_strict),
                "label_relax_margin": float(sp_label_relax_strict),
                "require_label_match": bool(sp_require_label_strict),
                "max_center_dist_norm": float(max_center_dist_norm_strict),
                "sp_jitter_px": int(sp_jitter_px),
                "sp_jitter_rings": int(sp_jitter_rings),
            }

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
                "sp_roi_frac": sp_roi_frac_eff,
                "sp_margin": sp_margin,
                "sp_escalate_step": sp_escalate_step,
                "sp_escalate_max": sp_escalate_max,
                "sp_hands_up_only": sp_hands_up_only,
                "sp_jitter_px": sp_jitter_px,
                "sp_jitter_rings": sp_jitter_rings,
                "center_penalty_lambda": sp_center_penalty_eff,
                "label_relax_margin": sp_label_relax_eff,
                "require_label_match": sp_require_label_eff,
                "max_center_dist_norm": max_center_dist_norm_eff,
            }

            sp_attempt_left = False
            sp_attempt_right = False
            missing_pre_sp_left = (left is None)
            missing_pre_sp_right = (right is None)
            if missing_pre_sp_left:
                sp_missing_left_pre += 1
            if missing_pre_sp_right:
                sp_missing_right_pre += 1

            left_roi = None
            right_roi = None
            sp_dbg_left = None
            sp_dbg_right = None
            sp_params_left = None
            sp_params_right = None

            expected_left = _hand_expected("left", pose_img_landmarks, tracker_left_ready, last_left_good_px)
            expected_right = _hand_expected("right", pose_img_landmarks, tracker_right_ready, last_right_good_px)
            center_hint_left = _center_hint_for(
                "left", proc_w, proc_h, pose_img_landmarks, tracker_left, tracker_left_ready, last_left_good_px
            )
            center_hint_right = _center_hint_for(
                "right", proc_w, proc_h, pose_img_landmarks, tracker_right, tracker_right_ready, last_right_good_px
            )

            allow_sp_occ_left = (occ_ttl_left > 0) and (not world_coords) and (last_left_good_px is not None)
            allow_sp_occ_right = (occ_ttl_right > 0) and (not world_coords) and (last_right_good_px is not None)
            if allow_sp_occ_left:
                x0, y0, x1, y1 = bbox_from_pts_px(last_left_good_px)
                center_hint_left = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
            if allow_sp_occ_right:
                x0, y0, x1, y1 = bbox_from_pts_px(last_right_good_px)
                center_hint_right = (0.5 * (x0 + x1), 0.5 * (y0 + y1))

            if occ_ttl_left > 0 and not allow_sp_occ_left:
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
                                "sp_roi_frac": sp_roi_frac_strict,
                                "center_penalty_lambda": sp_center_penalty_strict,
                                "label_relax_margin": sp_label_relax_strict,
                                "require_label_match": sp_require_label_strict,
                                "max_center_dist_norm": max_center_dist_norm_strict,
                            }
                        )
                    left, left_score, rec_l, sp_dt, left_roi = _execute_second_pass(
                        "left",
                        left,
                        left_score,
                        debug_roi=sp_debug_roi,
                        center_hint=center_hint_left,
                        debug_out=sp_dbg_left,
                        **sp_kwargs_left,
                    )
                    second_pass_runtime += sp_dt
                    if rec_l:
                        src1 = "pass2"
                        sp_rec_left += 1
                        if sanitize_rejects:
                            left_reject_reason = None
                        # Update pixel coordinates after second pass recovery
                        if world_coords and left is not None:
                            # Convert recovered world coords back to pixel space for IoU
                            if lm_list_img and left_idx is not None and 0 <= left_idx < len(lm_list_img):
                                pts_norm = xyz_list_from_lms(lm_list_img[left_idx]) or []
                                last_left_px = norm_to_px(pts_norm, proc_w, proc_h)
                    if sp_dbg_left is not None:
                        sp_params_left = dict(sp_params_eff_strict if sp_strict else sp_params_eff)
                else:
                    rec_l = False
            else:
                rec_l = False

            if occ_ttl_right > 0 and not allow_sp_occ_right:
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
                                "sp_roi_frac": sp_roi_frac_strict,
                                "center_penalty_lambda": sp_center_penalty_strict,
                                "label_relax_margin": sp_label_relax_strict,
                                "require_label_match": sp_require_label_strict,
                                "max_center_dist_norm": max_center_dist_norm_strict,
                            }
                        )
                    right, right_score, rec_r, sp_dt, right_roi = _execute_second_pass(
                        "right",
                        right,
                        right_score,
                        debug_roi=sp_debug_roi,
                        center_hint=center_hint_right,
                        debug_out=sp_dbg_right,
                        **sp_kwargs_right,
                    )
                    second_pass_runtime += sp_dt
                    if rec_r:
                        src2 = "pass2"
                        sp_rec_right += 1
                        if sanitize_rejects:
                            right_reject_reason = None
                        # Update pixel coordinates after second pass recovery
                        if world_coords and right is not None:
                            # Convert recovered world coords back to pixel space for IoU
                            if lm_list_img and right_idx is not None and 0 <= right_idx < len(lm_list_img):
                                pts_norm = xyz_list_from_lms(lm_list_img[right_idx]) or []
                                last_right_px = norm_to_px(pts_norm, proc_w, proc_h)
                    if sp_dbg_right is not None:
                        sp_params_right = dict(sp_params_eff_strict if sp_strict else sp_params_eff)
                else:
                    rec_r = False
            else:
                rec_r = False

            left_score_gate = _score_for_gate(left_score, left is not None)
            right_score_gate = _score_for_gate(right_score, right is not None)

            if left is not None and (left_score_gate is None or left_score_gate < hand_lo):
                left, left_score = None, None
                src1 = None
                left_reject_reason = left_reject_reason or "score_lo_after_sp"
            if right is not None and (right_score_gate is None or right_score_gate < hand_lo):
                right, right_score = None, None
                src2 = None
                right_reject_reason = right_reject_reason or "score_lo_after_sp"

            if left is not None and sanity_enable and sanity_pass2:
                sanity_dbg_left = {}
                sanity_stage_left = "pass2"
                ok, reasons = check_hand_sanity(
                    left,
                    prev_anchor=anchor_left,
                    prev_pred=prev_pred_left,
                    world_coords=world_coords,
                    scale_range=sanity_scale_range,
                    wrist_k=sanity_wrist_k,
                    bone_tol=sanity_bone_tol,
                    debug_out=sanity_dbg_left,
                )
                if not ok:
                    left, left_score = None, None
                    src1 = None
                    sanity_msg = "sanity:" + "|".join(reasons)
                    left_reject_reason = _append_reject_reason(left_reject_reason, sanity_msg)
            if right is not None and sanity_enable and sanity_pass2:
                sanity_dbg_right = {}
                sanity_stage_right = "pass2"
                ok, reasons = check_hand_sanity(
                    right,
                    prev_anchor=anchor_right,
                    prev_pred=prev_pred_right,
                    world_coords=world_coords,
                    scale_range=sanity_scale_range,
                    wrist_k=sanity_wrist_k,
                    bone_tol=sanity_bone_tol,
                    debug_out=sanity_dbg_right,
                )
                if not ok:
                    right, right_score = None, None
                    src2 = None
                    sanity_msg = "sanity:" + "|".join(reasons)
                    right_reject_reason = _append_reject_reason(right_reject_reason, sanity_msg)

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
                [left, right, last_left_good_img, last_right_good_img] if not world_coords else [left, right],
            )

            # If second pass grabs the other hand, reassign based on pose wrist proximity.
            if pose_img_landmarks is not None:
                if left is not None and src1 == "pass2" and src2 != "pass2":
                    if pose_side_ambiguous and not world_coords:
                        dLL, dLR = _last_good_wrist_dists(left, last_left_good_img, last_right_good_img)
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
                            left_reject_reason = _append_reject_reason(left_reject_reason, "pose_side_reassign")
                            sp_dbg_left = None
                            sp_params_left = None
                        else:
                            left, left_score = None, None
                            src1 = None
                            left_roi = None
                            cur_left_img, cur_left_px = None, None
                            left_reject_reason = _append_reject_reason(left_reject_reason, "pose_side_drop")
                            if rec_l:
                                rec_l = False
                                if sp_rec_left > 0:
                                    sp_rec_left -= 1
                            sp_dbg_left = None
                            sp_params_left = None

                if right is not None and src2 == "pass2" and src1 != "pass2":
                    if pose_side_ambiguous and not world_coords:
                        dRL, dRR = _last_good_wrist_dists(right, last_left_good_img, last_right_good_img)
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
                            right_reject_reason = _append_reject_reason(right_reject_reason, "pose_side_reassign")
                            sp_dbg_right = None
                            sp_params_right = None
                        else:
                            right, right_score = None, None
                            src2 = None
                            right_roi = None
                            cur_right_img, cur_right_px = None, None
                            right_reject_reason = _append_reject_reason(right_reject_reason, "pose_side_drop")
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
                    last_left_good_img,
                    last_right_good_img,
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
                    dLL, dLR = _last_good_wrist_dists(left, last_left_good_img, last_right_good_img)
                    dRL, dRR = _last_good_wrist_dists(right, last_left_good_img, last_right_good_img)
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
                        right_reject_reason = _append_reject_reason(right_reject_reason, "pose_side_drop")
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
                        left_reject_reason = _append_reject_reason(left_reject_reason, "pose_side_drop")
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

            left_score_gate = _score_for_gate(left_score, left is not None)
            right_score_gate = _score_for_gate(right_score, right is not None)

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
            left_ref_px = cur_left_px if cur_left_px is not None else last_left_good_px
            right_ref_px = cur_right_px if cur_right_px is not None else last_right_good_px
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
                occ_ttl_left = occ_hyst_frames
            else:
                occ_ttl_left = max(0, occ_ttl_left - 1)

            if enter_right or hold_right_occ:
                occ_ttl_right = occ_hyst_frames
            else:
                occ_ttl_right = max(0, occ_ttl_right - 1)

            if pose_side_ambiguous:
                cross_frames = max(2, min(int(occ_hyst_frames), CROSS_OCCLUSION_FRAMES))
                if left is None and last_left_export is not None:
                    occ_ttl_left = max(occ_ttl_left, cross_frames)
                if right is None and last_right_export is not None:
                    occ_ttl_right = max(occ_ttl_right, cross_frames)

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
                    nonlocal overlap_freeze_side, overlap_freeze_reason, last_overlap_freeze_side, last_overlap_freeze_i
                    nonlocal occ_ttl_left, occ_ttl_right
                    overlap_freeze_side = side
                    overlap_freeze_reason = reason
                    last_overlap_freeze_side = side
                    last_overlap_freeze_i = i
                    if side == "left":
                        occ_ttl_left = max(occ_ttl_left, cross_frames)
                        occ_ttl_right = 0
                    elif side == "right":
                        occ_ttl_right = max(occ_ttl_right, cross_frames)
                        occ_ttl_left = 0
                    else:
                        occ_ttl_left = max(occ_ttl_left, cross_frames)
                        occ_ttl_right = max(occ_ttl_right, cross_frames)

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

                left_z_src = left if left is not None else last_left_export
                right_z_src = right if right is not None else last_right_export
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
                    elif last_overlap_freeze_side and (i - last_overlap_freeze_i) <= cross_frames:
                        _apply_freeze(last_overlap_freeze_side, "last")
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
                        elif last_overlap_freeze_side and (i - last_overlap_freeze_i) <= cross_frames:
                            _apply_freeze(last_overlap_freeze_side, "last")
                            decided = True

                if not decided:
                    if last_overlap_freeze_side and (i - last_overlap_freeze_i) <= cross_frames:
                        _apply_freeze(last_overlap_freeze_side, "last")
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

            occluded_L = (occ_ttl_left > 0)
            occluded_R = (occ_ttl_right > 0)
            occ_freeze_max_frames = max(0, min(int(occ_hyst_frames), OCCLUSION_FREEZE_MAX_FRAMES))
            if not occluded_L:
                occ_freeze_age_left = 0
            if not occluded_R:
                occ_freeze_age_right = 0

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
                    last_left_good_img,
                    last_right_good_img,
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
                    last_left_good_img,
                    last_right_good_img,
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
                        left_reject_reason = _append_reject_reason(left_reject_reason, "pose_side_drop")
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
                        right_reject_reason = _append_reject_reason(right_reject_reason, "pose_side_drop")
                        if rec_r:
                            rec_r = False
                            if sp_rec_right > 0:
                                sp_rec_right -= 1
                        sp_dbg_right = None
                        sp_params_right = None

            def _wrist_dist_norm(det_img, exp_img):
                if not det_img or not exp_img:
                    return None
                det_xy = _wrist_xy(det_img)
                exp_xy = _wrist_xy(exp_img)
                if det_xy is None or exp_xy is None:
                    return None
                d = math.hypot(det_xy[0] - exp_xy[0], det_xy[1] - exp_xy[1])
                scale = max(_hand_scale(exp_img), 1e-6)
                return d / scale

            def _pose_guided_freeze(hand_pts, last_good_img, hand: str):
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
                Lref, Rref = _pose_wrist_refs(
                    world_coords, pose_img_landmarks, pose_world_full, last_pose_world_full
                )
                pose_xy = Lref if hand == "left" else Rref
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
                    s = max_shift / dist
                    dx *= s
                    dy *= s
                shifted = []
                for p in hand_pts:
                    if not isinstance(p, dict):
                        shifted.append(p)
                        continue
                    nx = float(p["x"]) + dx
                    ny = float(p["y"]) + dy
                    nx = min(1.0, max(0.0, nx))
                    ny = min(1.0, max(0.0, ny))
                    shifted.append({**p, "x": nx, "y": ny})
                return shifted

            if occluded_L and left is not None and src1 in ("pass1", "pass2"):
                accept = False
                if overlap_guard and overlap_freeze_side == "left":
                    accept = False
                elif overlap_guard:
                    if (
                        left_score_gate is not None and
                        left_score_gate >= anchor_score_eff and
                        pose_ok_left and
                        side_ok_left_accept
                    ):
                        accept = True
                else:
                    score_for_accept = left_score if score_source == "handedness" else None
                    if score_for_accept is not None and score_for_accept >= hand_hi:
                        accept = True
                    else:
                        det_img = cur_left_img
                        exp_img = last_left_good_img
                        dn = _wrist_dist_norm(det_img, exp_img)
                        if dn is not None and dn <= occ_return_k:
                            accept = True
                if accept:
                    occ_ttl_left = 0
                    occluded_L = False
                else:
                    left = None
                    left_score = None
                    src1 = None
                    reason = "occ_freeze" if overlap_guard else "occ_guard"
                    left_reject_reason = left_reject_reason or reason

            if occluded_R and right is not None and src2 in ("pass1", "pass2"):
                accept = False
                if overlap_guard and overlap_freeze_side == "right":
                    accept = False
                elif overlap_guard:
                    if (
                        right_score_gate is not None and
                        right_score_gate >= anchor_score_eff and
                        pose_ok_right and
                        side_ok_right_accept
                    ):
                        accept = True
                else:
                    score_for_accept = right_score if score_source == "handedness" else None
                    if score_for_accept is not None and score_for_accept >= hand_hi:
                        accept = True
                    else:
                        det_img = cur_right_img
                        exp_img = last_right_good_img
                        dn = _wrist_dist_norm(det_img, exp_img)
                        if dn is not None and dn <= occ_return_k:
                            accept = True
                if accept:
                    occ_ttl_right = 0
                    occluded_R = False
                else:
                    right = None
                    right_score = None
                    src2 = None
                    reason = "occ_freeze" if overlap_guard else "occ_guard"
                    right_reject_reason = right_reject_reason or reason

            missing_pre_occ_left = (left is None)
            missing_pre_occ_right = (right is None)
            occlusion_saved_1 = False
            occlusion_saved_2 = False

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

            if occluded_L:
                hold_left = 0  # Reset hold counter when occlusion triggers
                if (
                    left is None
                    and last_left_export is not None
                    and occ_freeze_age_left < occ_freeze_max_frames
                ):
                    left = _pose_guided_freeze(last_left_export, last_left_good_img, "left")
                    src1 = "occluded"
                    left_score = last_left_export_score
                    if not world_coords and left is not None:
                        cur_left_img = left
                        cur_left_px = norm_to_px(left, proc_w, proc_h)
                    if missing_pre_occ_left:
                        occlusion_saved_1 = True
                    occ_freeze_age_left += 1
                else:
                    left = None

            if occluded_R:
                hold_right = 0  # Reset hold counter when occlusion triggers
                if (
                    right is None
                    and last_right_export is not None
                    and occ_freeze_age_right < occ_freeze_max_frames
                ):
                    right = _pose_guided_freeze(last_right_export, last_right_good_img, "right")
                    src2 = "occluded"
                    right_score = last_right_export_score
                    if not world_coords and right is not None:
                        cur_right_img = right
                        cur_right_px = norm_to_px(right, proc_w, proc_h)
                    if missing_pre_occ_right:
                        occlusion_saved_2 = True
                    occ_freeze_age_right += 1
                else:
                    right = None

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
                    last_left_good_img,
                    last_right_good_img,
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
                    last_left_good_img,
                    last_right_good_img,
                    pose_side_reassign_ratio_eff,
                    use_last_good=True,
                )

            if left is not None and src1 in ("pass1", "pass2") and (not overlap_ambiguous or side_ok_left) and not overlap_guard:
                last_left_obs = left
                last_left_obs_i = i
            if right is not None and src2 in ("pass1", "pass2") and (not overlap_ambiguous or side_ok_right) and not overlap_guard:
                last_right_obs = right
                last_right_obs_i = i

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

            # Update or Track Left
            if left is not None and src1 in ("pass1", "pass2") and (not overlap_ambiguous or side_ok_left) and not overlap_guard:
                # Update filter and prepare for potential tracking next frame
                # IMPORTANT: Always use Image Coords for tracker initialization if available
                left_img_for_tracker = None
                if cur_left_img is not None:
                    left_img_for_tracker = cur_left_img

                # If we don't have specific image coords (e.g. world_coords=False), use 'left' itself
                if left_img_for_tracker is None and not world_coords:
                    left_img_for_tracker = left

                # Only update tracker if we have valid image coords
                # CRITICAL: Pass image coords to tracker, but don't overwrite 'left'
                # because 'left' should stay in the coordinate system we're using (world or image)
                if left_img_for_tracker:
                    if not world_coords:
                        if not tracker_left_ready:
                            if (
                                pose_ok_left and
                                left_score is not None and
                                left_score >= tracker_init_score_eff
                            ):
                                tracker_left.reset()
                                tracker_left.update(left_img_for_tracker, ts, rgb, score=left_score_gate)
                                tracker_left_ready = True
                                track_age_left = 0
                        else:
                            update_ok = pose_ok_left
                            if tracker_update_score_eff is not None:
                                update_ok = update_ok and (
                                    left_score is not None and left_score >= tracker_update_score_eff
                                )
                            if update_ok:
                                tracker_left.update(
                                    left_img_for_tracker,
                                    ts,
                                    rgb,
                                    score=left_score_gate if left_score_gate is not None else 1.0,
                                )
                                track_age_left = 0
                hold_left = 0
            elif left is None and tracker_left_ready and (not world_coords) and not block_track_left:
                if dt > track_reset_ms:
                    tracker_left.reset()
                    tracker_left_ready = False
                    track_age_left = 0
                    track_reset_left = True
                elif track_age_left < track_max_gap:
                    tracked = tracker_left.track(rgb, ts)
                    if tracked is not None:
                        left = tracked
                        src1 = "tracked"
                        base = tracker_left.last_score or 1.0
                        left_score = base * (track_score_decay ** (track_age_left + 1))
                        track_ok_left = True
                        track_age_left += 1
                        track_rec_left += 1
                    else:
                        tracker_left.reset()
                        tracker_left_ready = False
                        track_age_left = 0

            # Update or Track Right
            if right is not None and src2 in ("pass1", "pass2") and (not overlap_ambiguous or side_ok_right) and not overlap_guard:
                # Update filter and prepare for potential tracking next frame
                right_img_for_tracker = None
                if cur_right_img is not None:
                    right_img_for_tracker = cur_right_img

                if right_img_for_tracker is None and not world_coords:
                    right_img_for_tracker = right

                # CRITICAL: Pass image coords to tracker, don't overwrite 'right'
                if right_img_for_tracker:
                    if not world_coords:
                        if not tracker_right_ready:
                            if (
                                pose_ok_right and
                                right_score is not None and
                                right_score >= tracker_init_score_eff
                            ):
                                tracker_right.reset()
                                tracker_right.update(right_img_for_tracker, ts, rgb, score=right_score_gate)
                                tracker_right_ready = True
                                track_age_right = 0
                        else:
                            update_ok = pose_ok_right
                            if tracker_update_score_eff is not None:
                                update_ok = update_ok and (
                                    right_score is not None and right_score >= tracker_update_score_eff
                                )
                            if update_ok:
                                tracker_right.update(
                                    right_img_for_tracker,
                                    ts,
                                    rgb,
                                    score=right_score_gate if right_score_gate is not None else 1.0,
                                )
                                track_age_right = 0
                hold_right = 0
            elif right is None and tracker_right_ready and (not world_coords) and not block_track_right:
                if dt > track_reset_ms:
                    tracker_right.reset()
                    tracker_right_ready = False
                    track_age_right = 0
                    track_reset_right = True
                elif track_age_right < track_max_gap:
                    tracked = tracker_right.track(rgb, ts)
                    if tracked is not None:
                        right = tracked
                        src2 = "tracked"
                        base = tracker_right.last_score or 1.0
                        right_score = base * (track_score_decay ** (track_age_right + 1))
                        track_ok_right = True
                        track_age_right += 1
                        track_rec_right += 1
                    else:
                        tracker_right.reset()
                        tracker_right_ready = False
                        track_age_right = 0

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
                    dLL, dLR = _last_good_wrist_dists(left, last_left_good_img, last_right_good_img)
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
                    dRL, dRR = _last_good_wrist_dists(right, last_left_good_img, last_right_good_img)
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

            if left is not None and src1 in ("pass1", "pass2") and left_score is not None and (not overlap_ambiguous or side_ok_left) and not overlap_guard:
                last_left_export = left
                last_left_export_score = float(left_score)
                last_left_good_px = cur_left_px
                last_left_good_img = cur_left_img
            if right is not None and src2 in ("pass1", "pass2") and right_score is not None and (not overlap_ambiguous or side_ok_right) and not overlap_guard:
                last_right_export = right
                last_right_export_score = float(right_score)
                last_right_good_px = cur_right_px
                last_right_good_img = cur_right_img

            hand1_is_anchor = (
                left is not None and
                left_score is not None and
                left_score >= anchor_score_eff and
                pose_ok_left and
                (not overlap_ambiguous or side_ok_left) and
                not overlap_guard and
                src1 in ("pass1", "pass2")
            )
            hand2_is_anchor = (
                right is not None and
                right_score is not None and
                right_score >= anchor_score_eff and
                pose_ok_right and
                (not overlap_ambiguous or side_ok_right) and
                not overlap_guard and
                src2 in ("pass1", "pass2")
            )
            if hand1_is_anchor:
                last_left_anchor = left
                last_left_anchor_i = i
            if hand2_is_anchor:
                last_right_anchor = right
                last_right_anchor_i = i

            if left is None:
                hand1_state = "missing"
            elif src1 == "occluded":
                hand1_state = "occluded"
            elif src1 == "tracked":
                hand1_state = "predicted"
            else:
                hand1_state = "observed"

            if right is None:
                hand2_state = "missing"
            elif src2 == "occluded":
                hand2_state = "occluded"
            elif src2 == "tracked":
                hand2_state = "predicted"
            else:
                hand2_state = "observed"

            fr_obj: Dict[str, Any] = {
                "ts": int(ts),
                "dt": int(dt),
                "hand 1": left,
                "hand 1_score": float(left_score) if left_score is not None else None,
                "hand 1_score_gate": float(left_score_gate) if left_score_gate is not None else None,
                "hand 1_source": src1,
                "hand 1_state": hand1_state,
                "hand 1_is_anchor": bool(hand1_is_anchor),
                "hand 1_reject_reason": left_reject_reason,
                "hand 1_pose_quality": float(pose_quality_left) if pose_quality_left is not None else None,
                "hand 1_wrist_z": float(hand_wrist_z_left) if hand_wrist_z_left is not None else None,
                "hand 1_sanity_scale_ratio": sanity_dbg_left.get("scale_ratio") if sanity_dbg_left else None,
                "hand 1_sanity_wrist_jump": sanity_dbg_left.get("wrist_jump_dist") if sanity_dbg_left else None,
                "hand 1_sanity_bone_max_err": sanity_dbg_left.get("bone_max_rel_err") if sanity_dbg_left else None,
                "hand 1_sanity_bone_worst": sanity_dbg_left.get("bone_worst") if sanity_dbg_left else None,
                "hand 1_sanity_stage": sanity_stage_left,
                "hand_1_track_age": int(track_age_left),
                "hand_1_track_reset": bool(track_reset_left),
                "hand_1_tracker_ready": bool(tracker_left_ready),
                "hand 1_tracker_last_score": float(tracker_last_score_left) if tracker_last_score_left is not None else None,
                "hand 1_tracker_last_ts": float(tracker_last_ts_left) if tracker_last_ts_left is not None else None,
                "hand 2": right,
                "hand 2_score": float(right_score) if right_score is not None else None,
                "hand 2_score_gate": float(right_score_gate) if right_score_gate is not None else None,
                "hand 2_source": src2,
                "hand 2_state": hand2_state,
                "hand 2_is_anchor": bool(hand2_is_anchor),
                "hand 2_reject_reason": right_reject_reason,
                "hand 2_pose_quality": float(pose_quality_right) if pose_quality_right is not None else None,
                "hand 2_wrist_z": float(hand_wrist_z_right) if hand_wrist_z_right is not None else None,
                "hand 2_sanity_scale_ratio": sanity_dbg_right.get("scale_ratio") if sanity_dbg_right else None,
                "hand 2_sanity_wrist_jump": sanity_dbg_right.get("wrist_jump_dist") if sanity_dbg_right else None,
                "hand 2_sanity_bone_max_err": sanity_dbg_right.get("bone_max_rel_err") if sanity_dbg_right else None,
                "hand 2_sanity_bone_worst": sanity_dbg_right.get("bone_worst") if sanity_dbg_right else None,
                "hand 2_sanity_stage": sanity_stage_right,
                "hand_2_track_age": int(track_age_right),
                "hand_2_track_reset": bool(track_reset_right),
                "hand_2_tracker_ready": bool(tracker_right_ready),
                "hand 2_tracker_last_score": float(tracker_last_score_right) if tracker_last_score_right is not None else None,
                "hand 2_tracker_last_ts": float(tracker_last_ts_right) if tracker_last_ts_right is not None else None,
                "swap_applied": bool(swap_applied),
                "dedup_triggered": bool(dedup_triggered),
                "dedup_removed": dedup_removed,
                "dedup_iou": float(dedup_iou) if dedup_iou is not None else None,
                "dedup_dist_norm": float(dedup_dist_norm) if dedup_dist_norm is not None else None,
                "hand_1_track_ok": bool(track_ok_left),
                "hand_2_track_ok": bool(track_ok_right),
                "hand_1_sp_attempted": bool(sp_attempt_left),
                "hand_2_sp_attempted": bool(sp_attempt_right),
                "hand_1_sp_recovered": bool(rec_l),
                "hand_2_sp_recovered": bool(rec_r),
                "sp_overlap_iou": round(float(sp_overlap_iou_val), 4),
                "overlap_iou_sp": float(sp_overlap_iou_val),
                "sp_overlap": bool(overlap),
                "iou_hands": float(iou_hands) if iou_hands is not None else None,
                "overlap_iou_ref": float(occlusion_iou) if occlusion_iou is not None else None,
                "overlap_guard_pre": bool(overlap_guard_pre),
                "overlap_guard": bool(overlap_guard),
                "overlap_freeze": bool(overlap_freeze),
                "overlap_freeze_reason": overlap_freeze_reason,
                "overlap_freeze_side": overlap_freeze_side,
                "overlap_iou_guard": float(overlap_iou_guard) if overlap_iou_guard is not None else None,
                "overlap_iou_guard_src": overlap_iou_guard_src,
                "overlap_z_hint": float(overlap_z_hint) if overlap_z_hint is not None else None,
                "overlap_z_abs": float(abs(occlusion_z_diff)) if occlusion_z_diff is not None else None,
                "overlap_hand_z_diff": float(overlap_hand_z_diff) if overlap_hand_z_diff is not None else None,
                "pose_side_ambiguous": bool(pose_side_ambiguous),
                "pose_side_ratio": float(pose_side_reassign_ratio_eff),
                "side_d_ll": float(side_d_ll) if side_d_ll is not None else None,
                "side_d_lr": float(side_d_lr) if side_d_lr is not None else None,
                "side_d_rl": float(side_d_rl) if side_d_rl is not None else None,
                "side_d_rr": float(side_d_rr) if side_d_rr is not None else None,
                "side_cost_current": float(side_cost_current) if side_cost_current is not None else None,
                "side_cost_swap": float(side_cost_swap) if side_cost_swap is not None else None,
                "side_pref_left": side_pref_left,
                "side_pref_right": side_pref_right,
                "side_ok_1": bool(side_ok_left_accept),
                "side_ok_2": bool(side_ok_right_accept),
                "occluded_1": bool(occluded_L),
                "occluded_2": bool(occluded_R),
                "occluded_1_ttl": int(occ_ttl_left),
                "occluded_2_ttl": int(occ_ttl_right),
                "occlusion_freeze_age_1": int(occ_freeze_age_left),
                "occlusion_freeze_age_2": int(occ_freeze_age_right),
                "occlusion_freeze_max": int(occ_freeze_max_frames),
                "occlusion_saved_1": bool(occlusion_saved_1),
                "occlusion_saved_2": bool(occlusion_saved_2),
                "missing_pre_occ_1": bool(missing_pre_occ_left),
                "missing_pre_occ_2": bool(missing_pre_occ_right),
                "occlusion_iou": float(occlusion_iou) if occlusion_iou is not None else None,
                "occlusion_z_diff": float(occlusion_z_diff) if occlusion_z_diff is not None else None,
                "occlusion_behind_diff": float(occlusion_behind_diff) if occlusion_behind_diff is not None else None,
                "occlusion_samples_ok": bool(occlusion_samples_ok),
                "pose_wrist_dist_1": float(pose_wrist_dist_1) if pose_wrist_dist_1 is not None else None,
                "pose_wrist_dist_2": float(pose_wrist_dist_2) if pose_wrist_dist_2 is not None else None,
                "pose_wrist_z_left_world": float(pose_wrist_z_world_left) if pose_wrist_z_world_left is not None else None,
                "pose_wrist_z_right_world": float(pose_wrist_z_world_right) if pose_wrist_z_world_right is not None else None,
                "pose_wrist_z_left_img": float(pose_wrist_z_img_left) if pose_wrist_z_img_left is not None else None,
                "pose_wrist_z_right_img": float(pose_wrist_z_img_right) if pose_wrist_z_img_right is not None else None,
                "score_source": score_source,
                "min_hand_score": float(min_hand_score),
                "hand_score_lo": float(hand_lo),
                "hand_score_hi": float(hand_hi),
                "anchor_score": float(anchor_score_eff),
                "tracker_init_score": float(tracker_init_score_eff) if tracker_init_score_eff is not None else None,
                "tracker_update_score": float(tracker_update_score_eff) if tracker_update_score_eff is not None else None,
                "pose_dist_qual_min": float(pose_dist_qual_min_eff) if pose_dist_qual_min_eff is not None else None,
                "pose": pose_xyz,
                "pose_vis": pose_vis,
                "pose_interpolated": bool(pose_interpolated),
            }
            if write_hand_mask:
                h1_ok = 1 if fr_obj["hand 1"] is not None else 0
                h2_ok = 1 if fr_obj["hand 2"] is not None else 0
                fr_obj["hand_mask"] = [h1_ok, h2_ok]
                fr_obj["both_hands"] = 1 if (h1_ok and h2_ok) else 0

            if sp_debug_roi:
                if left_roi is not None:
                    fr_obj["hand 1_sp_roi_px"] = [int(left_roi[0]), int(left_roi[1]), int(left_roi[2]), int(left_roi[3])]
                if right_roi is not None:
                    fr_obj["hand 2_sp_roi_px"] = [int(right_roi[0]), int(right_roi[1]), int(right_roi[2]), int(right_roi[3])]
                if sp_dbg_left is not None:
                    fr_obj["hand 1_sp_center_hint_px"] = sp_dbg_left.get("center_hint_px")
                    fr_obj["hand 1_sp_debug"] = sp_dbg_left
                    fr_obj["hand 1_sp_params"] = sp_params_left
                if sp_dbg_right is not None:
                    fr_obj["hand 2_sp_center_hint_px"] = sp_dbg_right.get("center_hint_px")
                    fr_obj["hand 2_sp_debug"] = sp_dbg_right
                    fr_obj["hand 2_sp_params"] = sp_params_right

            frames.append(fr_obj)
            if fr_obj["hand 1"] is not None or fr_obj["hand 2"] is not None:
                hands_frames_running += 1
            if progress_interval and (len(frames) % progress_interval) == 0:
                log_metrics(LOGGER, "process_video.progress", {
                    "video": path.name,
                    "frames_processed": len(frames),
                    "elapsed_sec": round(perf_counter() - processing_started, 3),
                    "hands_detected_frames": hands_frames_running,
                })
            _emit_ndjson({"video": path.name, **fr_obj})

            i += 1
    finally:
        if hands_sp is not None:
            hands_sp.close()
        if pose_detector is not None:
            pose_detector.close()
        if hands_detector is not None:
            hands_detector.close()

    cap.release()
    if ndjson_f is not None:
        ndjson_f.close()

    meta_header = {
        "video": path.name,
        "fps": fps,
        "size_src": [int(width_src), int(height_src)],
        "size_proc": [int(proc_w), int(proc_h)],
        "version": 5,
        "coords": "world" if world_coords else "image",
        "mp_backend": backend,
        "mp_models": {
            "hand": str(hand_model_path) if hand_model_path is not None else None,
            "pose": str(pose_model_path) if pose_model_path is not None else None,
        } if backend == "tasks" else None,
        "mp_tasks_delegate": delegate_raw if backend == "tasks" else None,
        "pose_indices": keep_pose_indices if keep_pose_indices is not None else "all",
        "hand_mapping": {"hand 1": "left", "hand 2": "right"},
        "second_pass": bool(second_pass),
        "second_pass_params": {
            "trigger_below": float(sp_trigger_below),
            "roi_frac": float(sp_roi_frac),
            "margin": float(sp_margin),
            "escalate_step": float(sp_escalate_step),
            "escalate_max": float(sp_escalate_max),
            "hands_up_only": bool(sp_hands_up_only),
        } if second_pass else None,
        "sp_debug_roi": bool(sp_debug_roi),
        "interp_hold": int(interp_hold),
        "hand_score_gate": {
            "lo": float(hand_lo),
            "hi": float(hand_hi),
            "min_hand_score_legacy": float(min_hand_score),
            "score_source": score_source,
            "tracker_init_score": float(tracker_init_score_eff),
            "anchor_score": float(anchor_score_eff),
            "pose_dist_qual_min": float(pose_dist_qual_min_eff),
            "pose_side_reassign_ratio": float(pose_side_reassign_ratio_eff),
            "sanitize_rejects": bool(sanitize_rejects),
            "tracker_update_score": (
                float(tracker_update_score_eff)
                if tracker_update_score_eff is not None
                else None
            ),
        },
        "tracking": {
            "enabled": bool((not world_coords) and track_max_gap > 0),
            "track_max_gap": int(track_max_gap),
            "track_score_decay": float(track_score_decay),
            "track_reset_ms": int(track_reset_ms),
        },
    }
    if orig_frame_start is not None or orig_frame_end is not None:
        meta_header["frame_range"] = {
            "start": int(frame_start_i),
            "end": int(frame_end_i) if frame_end_i is not None else None,
        }
    if segment_meta:
        meta_header["segment"] = segment_meta
    processing_elapsed = perf_counter() - processing_started
    frames_total = max(1, len(frames))
    hands_present = hands_frames_running
    left_cov = sum(1 for fr in frames if fr["hand 1"] is not None) / frames_total
    right_cov = sum(1 for fr in frames if fr["hand 2"] is not None) / frames_total
    both_cov = sum(1 for fr in frames if (fr["hand 1"] is not None and fr["hand 2"] is not None)) / frames_total
    pose_cov = sum(1 for fr in frames if fr["pose"] is not None) / frames_total
    pose_interp_frac = sum(1 for fr in frames if fr["pose_interpolated"]) / frames_total

    left_scores = [fr["hand 1_score"] for fr in frames if fr["hand 1_score"] is not None]
    right_scores = [fr["hand 2_score"] for fr in frames if fr["hand 2_score"] is not None]
    left_mean = float(np.mean(left_scores)) if left_scores else 0.0
    right_mean = float(np.mean(right_scores)) if right_scores else 0.0
    left_med = float(np.median(left_scores)) if left_scores else 0.0
    right_med = float(np.median(right_scores)) if right_scores else 0.0

    dts = [fr["dt"] for fr in frames if fr["dt"] > 0]
    dt_median = float(np.median(dts)) if dts else 0.0
    fps_est = (1000.0 / dt_median) if dt_median > 0 else (fps or 0.0)

    sp_recovered_left_frac = float(sp_rec_left) / float(frames_total)
    sp_recovered_right_frac = float(sp_rec_right) / float(frames_total)
    track_recovered_count_1 = sum(1 for fr in frames if fr.get("hand_1_track_ok"))
    track_recovered_count_2 = sum(1 for fr in frames if fr.get("hand_2_track_ok"))
    track_recovered_left_frac = track_recovered_count_1 / frames_total
    track_recovered_right_frac = track_recovered_count_2 / frames_total
    outlier_frames_1 = 0
    outlier_frames_2 = 0
    if eval_mode:
        outlier_frames_1 = _count_outliers(frames, "hand 1")
        outlier_frames_2 = _count_outliers(frames, "hand 2")

    # Apply Global Smoothing
    smooth_tracks(frames)

    if ndjson_path is not None:
        smoothed_ndjson_path = ndjson_path.with_name(f"{ndjson_path.stem}_smoothed.ndjson")
        with open(smoothed_ndjson_path, "wt", encoding="utf-8") as smoothed_f:
            for fr in frames:
                smoothed_f.write(json.dumps(fr, ensure_ascii=False) + "\n")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta_header, "frames": frames}, f, ensure_ascii=False)

    pp_stats: Dict[str, Any] = {
        "pp_filled_left": 0,
        "pp_filled_right": 0,
        "pp_gaps_filled_left": 0,
        "pp_gaps_filled_right": 0,
        "pp_smoothing_delta_left": 0.0,
        "pp_smoothing_delta_right": 0.0,
    }
    out_pp_path: Optional[Path] = None
    if postprocess:
        frames_pp, pp_stats = postprocess_sequence(
            frames,
            hi=hand_hi,
            max_gap=pp_max_gap,
            smoother=pp_smoother,
            only_anchors=pp_only_anchors,
            world_coords=world_coords,
        )
        out_pp_path = out_path.with_name(f"{out_path.stem}_pp.json")
        meta_pp = dict(meta_header)
        meta_pp["postprocess"] = {
            "enabled": True,
            "max_gap": int(pp_max_gap),
            "smoother": str(pp_smoother),
            "only_anchors": bool(pp_only_anchors),
            "source_file": out_path.name,
        }
        with open(out_pp_path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta_pp, "frames": frames_pp}, f, ensure_ascii=False)

    eval_metrics = None
    if eval_mode:
        def _gap_stats(pred) -> Tuple[float, float, int]:
            gaps: List[int] = []
            cur = 0
            for fr in frames:
                if pred(fr):
                    cur += 1
                else:
                    if cur > 0:
                        gaps.append(cur)
                        cur = 0
            if cur > 0:
                gaps.append(cur)
            if not gaps:
                return 0.0, 0.0, 0
            return (
                float(np.percentile(gaps, 50)),
                float(np.percentile(gaps, 90)),
                int(max(gaps)),
            )

        missing_frames_1 = sum(1 for fr in frames if fr.get("hand 1") is None)
        missing_frames_2 = sum(1 for fr in frames if fr.get("hand 2") is None)
        occluded_frames_1 = sum(1 for fr in frames if fr.get("occluded_1"))
        occluded_frames_2 = sum(1 for fr in frames if fr.get("occluded_2"))
        swap_frames = sum(1 for fr in frames if fr.get("swap_applied"))
        dedup_trigger_frames = sum(1 for fr in frames if fr.get("dedup_triggered"))
        sp_attempt_frames_1 = sum(1 for fr in frames if fr.get("hand_1_sp_attempted"))
        sp_attempt_frames_2 = sum(1 for fr in frames if fr.get("hand_2_sp_attempted"))
        sp_recovered_frames_1 = sum(1 for fr in frames if fr.get("hand_1_sp_recovered"))
        sp_recovered_frames_2 = sum(1 for fr in frames if fr.get("hand_2_sp_recovered"))
        hold_frames_1 = sum(1 for fr in frames if fr.get("hand_1_track_ok"))
        hold_frames_2 = sum(1 for fr in frames if fr.get("hand_2_track_ok"))
        def _is_sanity_reject(reason) -> bool:
            if not reason:
                return False
            return "sanity:" in str(reason)
        sanity_reject_frames_1 = sum(
            1 for fr in frames if _is_sanity_reject(fr.get("hand 1_reject_reason"))
        )
        sanity_reject_frames_2 = sum(
            1 for fr in frames if _is_sanity_reject(fr.get("hand 2_reject_reason"))
        )

        missing_gap_p50_1, missing_gap_p90_1, missing_gap_max_1 = _gap_stats(
            lambda fr: fr.get("hand 1") is None
        )
        missing_gap_p50_2, missing_gap_p90_2, missing_gap_max_2 = _gap_stats(
            lambda fr: fr.get("hand 2") is None
        )
        occluded_gap_p50_1, occluded_gap_p90_1, occluded_gap_max_1 = _gap_stats(
            lambda fr: fr.get("occluded_1")
        )
        occluded_gap_p50_2, occluded_gap_p90_2, occluded_gap_max_2 = _gap_stats(
            lambda fr: fr.get("occluded_2")
        )

        eval_metrics = {
            "missing_frames_1": int(missing_frames_1),
            "missing_frames_2": int(missing_frames_2),
            "occluded_frames_1": int(occluded_frames_1),
            "occluded_frames_2": int(occluded_frames_2),
            "swap_frames": int(swap_frames),
            "dedup_trigger_frames": int(dedup_trigger_frames),
            "sp_attempt_frames_1": int(sp_attempt_frames_1),
            "sp_attempt_frames_2": int(sp_attempt_frames_2),
            "sp_recovered_frames_1": int(sp_recovered_frames_1),
            "sp_recovered_frames_2": int(sp_recovered_frames_2),
            "hold_frames_1": int(hold_frames_1),
            "hold_frames_2": int(hold_frames_2),
            "outlier_frames_1": int(outlier_frames_1),
            "outlier_frames_2": int(outlier_frames_2),
            "sanity_reject_frames_1": int(sanity_reject_frames_1),
            "sanity_reject_frames_2": int(sanity_reject_frames_2),
            "missing_gap_p50_1": float(missing_gap_p50_1),
            "missing_gap_p90_1": float(missing_gap_p90_1),
            "missing_gap_max_1": int(missing_gap_max_1),
            "missing_gap_p50_2": float(missing_gap_p50_2),
            "missing_gap_p90_2": float(missing_gap_p90_2),
            "missing_gap_max_2": int(missing_gap_max_2),
            "occluded_gap_p50_1": float(occluded_gap_p50_1),
            "occluded_gap_p90_1": float(occluded_gap_p90_1),
            "occluded_gap_max_1": int(occluded_gap_max_1),
            "occluded_gap_p50_2": float(occluded_gap_p50_2),
            "occluded_gap_p90_2": float(occluded_gap_p90_2),
            "occluded_gap_max_2": int(occluded_gap_max_2),
        }
        if postprocess:
            eval_metrics.update({
                "pp_filled_left": int(pp_stats.get("pp_filled_left", 0)),
                "pp_filled_right": int(pp_stats.get("pp_filled_right", 0)),
                "pp_gaps_filled_left": int(pp_stats.get("pp_gaps_filled_left", 0)),
                "pp_gaps_filled_right": int(pp_stats.get("pp_gaps_filled_right", 0)),
                "pp_smoothing_delta_left": float(pp_stats.get("pp_smoothing_delta_left", 0.0)),
                "pp_smoothing_delta_right": float(pp_stats.get("pp_smoothing_delta_right", 0.0)),
            })

    quality_score = (
        0.4 * both_cov +
        0.2 * (left_cov + right_cov) / 2.0 +
        0.2 * pose_cov +
        0.2 * (left_med + right_med) / 2.0
    )

    manifest_entry: VideoMeta = VideoMeta(
        id=path.stem,
        file=str(out_path),
        num_frames=len(frames),
        fps=fps,
        hands_frames=hands_present,
        hands_coverage=round(hands_present / frames_total, 4),
        left_score_mean=round(left_mean, 4),
        right_score_mean=round(right_mean, 4),
        left_coverage=round(left_cov, 4),
        right_coverage=round(right_cov, 4),
        both_coverage=round(both_cov, 4),
        pose_coverage=round(pose_cov, 4),
        pose_interpolated_frac=round(pose_interp_frac, 4),
        dt_median_ms=round(dt_median, 2),
        fps_est=round(fps_est, 3),
        left_score_median=round(left_med, 4),
        right_score_median=round(right_med, 4),
        quality_score=round(float(quality_score), 4),
        sp_recovered_left_frac=round(sp_recovered_left_frac, 4),
        sp_recovered_right_frac=round(sp_recovered_right_frac, 4),
        track_recovered_left_frac=round(track_recovered_left_frac, 4),
        track_recovered_right_frac=round(track_recovered_right_frac, 4),
    )
    if postprocess and out_pp_path is not None:
        manifest_entry.file_pp = str(out_pp_path)
        manifest_entry.pp_filled_left = int(pp_stats.get("pp_filled_left", 0))
        manifest_entry.pp_filled_right = int(pp_stats.get("pp_filled_right", 0))
        manifest_entry.pp_gaps_filled_left = int(pp_stats.get("pp_gaps_filled_left", 0))
        manifest_entry.pp_gaps_filled_right = int(pp_stats.get("pp_gaps_filled_right", 0))
        manifest_entry.pp_smoothing_delta_left = float(pp_stats.get("pp_smoothing_delta_left", 0.0))
        manifest_entry.pp_smoothing_delta_right = float(pp_stats.get("pp_smoothing_delta_right", 0.0))
    else:
        manifest_entry.file_pp = ""
        manifest_entry.pp_filled_left = 0
        manifest_entry.pp_filled_right = 0
        manifest_entry.pp_gaps_filled_left = 0
        manifest_entry.pp_gaps_filled_right = 0
        manifest_entry.pp_smoothing_delta_left = 0.0
        manifest_entry.pp_smoothing_delta_right = 0.0

    def _mean_gap_len(frames_list: List[Dict[str, Any]], key: str) -> float:
        gaps: List[int] = []
        cur = 0
        for fr in frames_list:
            if fr.get(key) is None:
                cur += 1
            else:
                if cur > 0:
                    gaps.append(cur)
                    cur = 0
        if cur > 0:
            gaps.append(cur)
        return float(np.mean(gaps)) if gaps else 0.0

    mean_gap_len_hand_1 = _mean_gap_len(frames, "hand 1")
    mean_gap_len_hand_2 = _mean_gap_len(frames, "hand 2")
    sp_recovered_count_1 = sum(1 for fr in frames if fr.get("hand_1_sp_recovered"))
    sp_recovered_count_2 = sum(1 for fr in frames if fr.get("hand_2_sp_recovered"))
    sp_attempt_count_1 = sum(1 for fr in frames if fr.get("hand_1_sp_attempted"))
    sp_attempt_count_2 = sum(1 for fr in frames if fr.get("hand_2_sp_attempted"))
    occl_saved_1 = sum(1 for fr in frames if fr.get("occlusion_saved_1"))
    occl_saved_2 = sum(1 for fr in frames if fr.get("occlusion_saved_2"))
    miss_pre_1 = sum(1 for fr in frames if fr.get("missing_pre_occ_1"))
    miss_pre_2 = sum(1 for fr in frames if fr.get("missing_pre_occ_2"))
    sp_recovery_rate_1 = sp_recovered_count_1 / frames_total
    sp_recovery_rate_2 = sp_recovered_count_2 / frames_total
    sp_attempt_rate_1 = sp_attempt_count_1 / frames_total
    sp_attempt_rate_2 = sp_attempt_count_2 / frames_total
    sp_success_given_attempt_1 = (sp_recovered_count_1 / sp_attempt_count_1) if sp_attempt_count_1 > 0 else 0.0
    sp_success_given_attempt_2 = (sp_recovered_count_2 / sp_attempt_count_2) if sp_attempt_count_2 > 0 else 0.0
    sp_rescue_rate_1 = (sp_recovered_count_1 / sp_missing_left_pre) if sp_missing_left_pre > 0 else 0.0
    sp_rescue_rate_2 = (sp_recovered_count_2 / sp_missing_right_pre) if sp_missing_right_pre > 0 else 0.0
    occlusion_saved_left_frac = occl_saved_1 / frames_total
    occlusion_saved_right_frac = occl_saved_2 / frames_total
    occlusion_recall_1 = occl_saved_1 / max(1, miss_pre_1)
    occlusion_recall_2 = occl_saved_2 / max(1, miss_pre_2)
    swap_rate = sum(1 for fr in frames if fr.get("swap_applied")) / frames_total
    dedup_trigger_count = sum(1 for fr in frames if fr.get("dedup_triggered"))
    dedup_removed_hand_1_count = sum(1 for fr in frames if fr.get("dedup_removed") == "hand_1")
    dedup_removed_hand_2_count = sum(1 for fr in frames if fr.get("dedup_removed") == "hand_2")
    dedup_trigger_rate = dedup_trigger_count / frames_total
    dedup_removed_rate = (dedup_removed_hand_1_count + dedup_removed_hand_2_count) / frames_total
    dedup_iou_vals = [
        fr.get("dedup_iou")
        for fr in frames
        if fr.get("dedup_triggered") and fr.get("dedup_iou") is not None
    ]
    dedup_dist_vals = [
        fr.get("dedup_dist_norm")
        for fr in frames
        if fr.get("dedup_triggered") and fr.get("dedup_dist_norm") is not None
    ]
    dedup_avg_iou_on_trigger = float(np.mean(dedup_iou_vals)) if dedup_iou_vals else 0.0
    dedup_avg_dist_norm_on_trigger = float(np.mean(dedup_dist_vals)) if dedup_dist_vals else 0.0
    occlusion_rate_1 = sum(1 for fr in frames if fr.get("occluded_1")) / frames_total
    occlusion_rate_2 = sum(1 for fr in frames if fr.get("occluded_2")) / frames_total
    occlusion_samples_count = sum(1 for fr in frames if fr.get("occlusion_samples_ok"))
    occlusion_iou_vals = [fr.get("occlusion_iou") for fr in frames if fr.get("occlusion_iou") is not None]
    occlusion_zdiff_vals = [fr.get("occlusion_z_diff") for fr in frames if fr.get("occlusion_z_diff") is not None]
    occlusion_behind_vals = [fr.get("occlusion_behind_diff") for fr in frames if fr.get("occlusion_behind_diff") is not None]
    occlusion_iou_p50 = float(np.percentile(occlusion_iou_vals, 50)) if occlusion_samples_count > 0 and occlusion_iou_vals else None
    occlusion_iou_p90 = float(np.percentile(occlusion_iou_vals, 90)) if occlusion_samples_count > 0 and occlusion_iou_vals else None
    occlusion_zdiff_p50 = float(np.percentile(occlusion_zdiff_vals, 50)) if occlusion_zdiff_vals else 0.0
    occlusion_zdiff_p90 = float(np.percentile(occlusion_zdiff_vals, 90)) if occlusion_zdiff_vals else 0.0
    occlusion_behind_p50 = float(np.percentile(occlusion_behind_vals, 50)) if occlusion_behind_vals else None
    occlusion_behind_p90 = float(np.percentile(occlusion_behind_vals, 90)) if occlusion_behind_vals else None
    occlusion_iou_not_occ_vals = [
        fr.get("occlusion_iou")
        for fr in frames
        if fr.get("occlusion_samples_ok") and not fr.get("occluded_1") and not fr.get("occluded_2")
        and fr.get("occlusion_iou") is not None
    ]
    occlusion_iou_occ_vals = [
        fr.get("occlusion_iou")
        for fr in frames
        if (fr.get("occluded_1") or fr.get("occluded_2")) and fr.get("occlusion_iou") is not None
    ]
    occlusion_behind_not_occ_vals = [
        fr.get("occlusion_behind_diff")
        for fr in frames
        if fr.get("occlusion_samples_ok") and not fr.get("occluded_1") and not fr.get("occluded_2")
        and fr.get("occlusion_behind_diff") is not None
    ]
    occlusion_behind_occ_vals = [
        fr.get("occlusion_behind_diff")
        for fr in frames
        if fr.get("occlusion_samples_ok") and (fr.get("occluded_1") or fr.get("occluded_2"))
        and fr.get("occlusion_behind_diff") is not None
    ]
    occlusion_iou_not_occ_p90 = float(np.percentile(occlusion_iou_not_occ_vals, 90)) if occlusion_iou_not_occ_vals else None
    occlusion_iou_occ_p50 = float(np.percentile(occlusion_iou_occ_vals, 50)) if occlusion_iou_occ_vals else None
    occlusion_behind_not_occ_p90 = float(np.percentile(occlusion_behind_not_occ_vals, 90)) if occlusion_behind_not_occ_vals else None
    occlusion_behind_occ_p50 = float(np.percentile(occlusion_behind_occ_vals, 50)) if occlusion_behind_occ_vals else None
    both_present_count = sum(1 for fr in frames if fr.get("hand 1") is not None and fr.get("hand 2") is not None)
    occlusion_rate_when_both_present_1 = (
        sum(1 for fr in frames if fr.get("hand 1") is not None and fr.get("hand 2") is not None and fr.get("occluded_1"))
        / both_present_count
        if both_present_count > 0 else 0.0
    )
    occlusion_rate_when_both_present_2 = (
        sum(1 for fr in frames if fr.get("hand 1") is not None and fr.get("hand 2") is not None and fr.get("occluded_2"))
        / both_present_count
        if both_present_count > 0 else 0.0
    )
    occlusion_toggle_count_1 = 0
    occlusion_toggle_count_2 = 0
    if frames:
        prev_occ_1 = bool(frames[0].get("occluded_1"))
        prev_occ_2 = bool(frames[0].get("occluded_2"))
        for fr in frames[1:]:
            cur_occ_1 = bool(fr.get("occluded_1"))
            cur_occ_2 = bool(fr.get("occluded_2"))
            if cur_occ_1 != prev_occ_1:
                occlusion_toggle_count_1 += 1
                prev_occ_1 = cur_occ_1
            if cur_occ_2 != prev_occ_2:
                occlusion_toggle_count_2 += 1
                prev_occ_2 = cur_occ_2

    summary_metrics = {
        "video": path.name,
        "out_path": str(out_path),
        "frames": len(frames),
        "processing": {
            "duration_sec": round(processing_elapsed, 4),
            "hand_runtime_sec": round(hand_runtime, 4),
            "pose_runtime_sec": round(pose_runtime, 4),
            "second_pass_runtime_sec": round(second_pass_runtime, 4),
            "effective_fps": round(len(frames) / max(processing_elapsed, 1e-6), 3),
        },
        "mp_backend": backend,
        "mp_tasks_delegate": tasks_delegate,
        "second_pass_enabled": bool(second_pass),
        "sp_recovered_left": int(sp_rec_left),
        "sp_recovered_right": int(sp_rec_right),
        "track_recovered_left": int(track_recovered_count_1),
        "track_recovered_right": int(track_recovered_count_2),
        "ndjson_written": bool(ndjson_path),
        "mean_gap_len_hand_1": round(mean_gap_len_hand_1, 4),
        "mean_gap_len_hand_2": round(mean_gap_len_hand_2, 4),
        "sp_recovery_rate_1": round(float(sp_recovery_rate_1), 4),
        "sp_recovery_rate_2": round(float(sp_recovery_rate_2), 4),
        "sp_attempt_rate_1": round(float(sp_attempt_rate_1), 4),
        "sp_attempt_rate_2": round(float(sp_attempt_rate_2), 4),
        "sp_success_given_attempt_1": round(float(sp_success_given_attempt_1), 4),
        "sp_success_given_attempt_2": round(float(sp_success_given_attempt_2), 4),
        "sp_rescue_rate_1": round(float(sp_rescue_rate_1), 4),
        "sp_rescue_rate_2": round(float(sp_rescue_rate_2), 4),
        "swap_rate": round(float(swap_rate), 4),
        "dedup_trigger_count": int(dedup_trigger_count),
        "dedup_removed_hand_1_count": int(dedup_removed_hand_1_count),
        "dedup_removed_hand_2_count": int(dedup_removed_hand_2_count),
        "dedup_trigger_rate": round(float(dedup_trigger_rate), 4),
        "dedup_removed_rate": round(float(dedup_removed_rate), 4),
        "dedup_avg_iou_on_trigger": round(float(dedup_avg_iou_on_trigger), 4),
        "dedup_avg_dist_norm_on_trigger": round(float(dedup_avg_dist_norm_on_trigger), 4),
        "occlusion_rate_1": round(float(occlusion_rate_1), 4),
        "occlusion_rate_2": round(float(occlusion_rate_2), 4),
        "occlusion_samples_count": int(occlusion_samples_count),
        "occlusion_iou_p50": round(float(occlusion_iou_p50), 4) if occlusion_iou_p50 is not None else None,
        "occlusion_iou_p90": round(float(occlusion_iou_p90), 4) if occlusion_iou_p90 is not None else None,
        "occlusion_zdiff_p50": round(float(occlusion_zdiff_p50), 4),
        "occlusion_zdiff_p90": round(float(occlusion_zdiff_p90), 4),
        "occlusion_behind_p50": round(float(occlusion_behind_p50), 4) if occlusion_behind_p50 is not None else None,
        "occlusion_behind_p90": round(float(occlusion_behind_p90), 4) if occlusion_behind_p90 is not None else None,
        "occlusion_iou_not_occ_p90": round(float(occlusion_iou_not_occ_p90), 4) if occlusion_iou_not_occ_p90 is not None else None,
        "occlusion_iou_occ_p50": round(float(occlusion_iou_occ_p50), 4) if occlusion_iou_occ_p50 is not None else None,
        "occlusion_behind_not_occ_p90": round(float(occlusion_behind_not_occ_p90), 4) if occlusion_behind_not_occ_p90 is not None else None,
        "occlusion_behind_occ_p50": round(float(occlusion_behind_occ_p50), 4) if occlusion_behind_occ_p50 is not None else None,
        "occlusion_rate_when_both_present_1": round(float(occlusion_rate_when_both_present_1), 4),
        "occlusion_rate_when_both_present_2": round(float(occlusion_rate_when_both_present_2), 4),
        "occlusion_toggle_count_1": int(occlusion_toggle_count_1),
        "occlusion_toggle_count_2": int(occlusion_toggle_count_2),
        "occlusion_saved_left": int(occl_saved_1),
        "occlusion_saved_right": int(occl_saved_2),
        "occlusion_saved_left_frac": round(float(occlusion_saved_left_frac), 4),
        "occlusion_saved_right_frac": round(float(occlusion_saved_right_frac), 4),
        "occlusion_recall_1": round(float(occlusion_recall_1), 4),
        "occlusion_recall_2": round(float(occlusion_recall_2), 4),
        "manifest": manifest_entry,
    }
    log_metrics(LOGGER, "process_video.summary", summary_metrics)
    manifest_dict = manifest_entry.__dict__.copy()
    if eval_metrics:
        manifest_dict.update(eval_metrics)
    return manifest_dict
