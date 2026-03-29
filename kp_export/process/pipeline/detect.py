from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

from ...core.utils import norm_to_px, pick_pose_indices, xyz_list_from_lms
from .decode import DecodedFrame


@dataclass
class PoseRuntimeState:
    last_pose_xyz: Optional[List[Dict[str, float]]] = None
    last_pose_vis: Optional[List[float]] = None
    last_pose_img_landmarks: Any = None
    last_pose_world_full: Optional[List[Dict[str, float]]] = None
    pose_ema: Optional[List[Dict[str, float]]] = None


@dataclass
class FrameDetections:
    pose_xyz: Optional[List[Dict[str, float]]]
    pose_vis: Optional[List[float]]
    pose_interpolated: bool
    pose_img_landmarks: Any
    pose_world_current: Optional[List[Dict[str, float]]]
    pose_world_full: Optional[List[Dict[str, float]]]
    lm_list_img: List[Any]
    lm_list_world: List[Any]
    left: Optional[List[Dict[str, float]]]
    right: Optional[List[Dict[str, float]]]
    left_score: Optional[float]
    right_score: Optional[float]
    left_idx: Optional[int]
    right_idx: Optional[int]
    cur_left_img: Optional[List[Dict[str, float]]]
    cur_right_img: Optional[List[Dict[str, float]]]
    cur_left_px: Optional[List[Dict[str, float]]]
    cur_right_px: Optional[List[Dict[str, float]]]


def run_frame_detectors(
    decoded: DecodedFrame,
    *,
    hands_detector: Any,
    pose_detector: Any,
    pose_every: int,
    world_coords: bool,
    keep_pose_indices: Optional[List[int]],
    pose_ema_alpha: float,
    state: PoseRuntimeState,
) -> Tuple[FrameDetections, PoseRuntimeState, float, float]:
    run_pose = (decoded.rel_index % max(1, pose_every)) == 0

    pose_runtime = 0.0
    rp = None
    if run_pose:
        pose_t0 = perf_counter()
        rp = pose_detector.process(decoded.rgb, decoded.ts_ms)
        pose_runtime = perf_counter() - pose_t0

    hand_t0 = perf_counter()
    rh = hands_detector.process(decoded.rgb, decoded.ts_ms)
    hand_runtime = perf_counter() - hand_t0

    left = right = None
    left_score = right_score = None
    left_idx = right_idx = None

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

    pose_xyz = None
    pose_vis = None
    pose_interpolated = False
    pose_img_landmarks = None
    pose_world_current = None

    last_pose_xyz = state.last_pose_xyz
    last_pose_vis = state.last_pose_vis
    last_pose_img_landmarks = state.last_pose_img_landmarks
    last_pose_world_full = state.last_pose_world_full
    pose_ema = state.pose_ema

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

    cur_left_img = None
    cur_right_img = None
    cur_left_px = None
    cur_right_px = None
    if lm_list_img is not None:
        if left_idx is not None and 0 <= left_idx < len(lm_list_img):
            cur_left_img = xyz_list_from_lms(lm_list_img[left_idx]) or None
            if cur_left_img:
                cur_left_px = norm_to_px(cur_left_img, decoded.proc_w, decoded.proc_h)
        if right_idx is not None and 0 <= right_idx < len(lm_list_img):
            cur_right_img = xyz_list_from_lms(lm_list_img[right_idx]) or None
            if cur_right_img:
                cur_right_px = norm_to_px(cur_right_img, decoded.proc_w, decoded.proc_h)

    new_state = PoseRuntimeState(
        last_pose_xyz=last_pose_xyz,
        last_pose_vis=last_pose_vis,
        last_pose_img_landmarks=last_pose_img_landmarks,
        last_pose_world_full=last_pose_world_full,
        pose_ema=pose_ema,
    )
    return (
        FrameDetections(
            pose_xyz=pose_xyz,
            pose_vis=pose_vis,
            pose_interpolated=pose_interpolated,
            pose_img_landmarks=pose_img_landmarks,
            pose_world_current=pose_world_current,
            pose_world_full=last_pose_world_full,
            lm_list_img=lm_list_img or [],
            lm_list_world=lm_list_world or [],
            left=left,
            right=right,
            left_score=left_score,
            right_score=right_score,
            left_idx=left_idx,
            right_idx=right_idx,
            cur_left_img=cur_left_img,
            cur_right_img=cur_right_img,
            cur_left_px=cur_left_px,
            cur_right_px=cur_right_px,
        ),
        new_state,
        hand_runtime,
        pose_runtime,
    )
