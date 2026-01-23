from __future__ import annotations
from typing import Any, Optional, Tuple
from pathlib import Path

from ..mp.mp_utils import create_hand_detector, create_pose_detector, resolve_task_model_path
from .process_constants import (
    HAND_SP_MIN_DETECTION,
    HAND_SP_MIN_TRACKING,
    SP_MIN_DET_MULTIPLIER,
    SP_MIN_TRACK_MULTIPLIER,
)


def _resolve_model_paths(
    backend: str,
    hand_task: Optional[str],
    pose_task: Optional[str],
) -> Tuple[Optional[Path], Optional[Path]]:
    hand_model_path: Optional[Path] = None
    pose_model_path: Optional[Path] = None
    if backend == "tasks":
        hand_model_path = resolve_task_model_path(hand_task, "hand_landmarker.task")
        pose_model_path = resolve_task_model_path(pose_task, "pose_landmarker_full.task")
        if hand_model_path is None:
            raise FileNotFoundError(
                "hand_landmarker.task not found. Pass --hand-task or place the asset under kp_export/mp/tasks/."
            )
        if pose_model_path is None:
            raise FileNotFoundError(
                "pose_landmarker_full.task not found. Pass --pose-task or place the asset under kp_export/mp/tasks/."
            )
    return hand_model_path, pose_model_path


def _initialize_detectors(
    backend: str,
    mp: Any,
    mp_solutions: Any,
    hand_model_path: Optional[Path],
    pose_model_path: Optional[Path],
    min_det: float,
    min_track: float,
    pose_complexity: int,
    tasks_delegate: Optional[str],
    second_pass: bool,
    world_coords: bool,
) -> Tuple[Any, Any, Optional[Any]]:
    hands_detector = create_hand_detector(
        backend,
        mp,
        mp_solutions,
        max_num_hands=2,
        min_det=min_det,
        min_track=min_track,
        static_image_mode=False,
        model_path=hand_model_path,
        world_coords=world_coords,
        tasks_delegate=tasks_delegate,
    )
    pose_detector = create_pose_detector(
        backend,
        mp,
        mp_solutions,
        model_complexity=pose_complexity,
        min_det=min_det,
        min_track=min_track,
        enable_segmentation=False,
        model_path=pose_model_path,
        tasks_delegate=tasks_delegate,
    )

    hands_sp = None
    if second_pass:
        hands_sp = create_hand_detector(
            backend,
            mp,
            mp_solutions,
            max_num_hands=1,
            min_det=max(HAND_SP_MIN_DETECTION, min_det * SP_MIN_DET_MULTIPLIER),
            min_track=max(HAND_SP_MIN_TRACKING, min_track * SP_MIN_TRACK_MULTIPLIER),
            static_image_mode=True,
            model_path=hand_model_path,
            world_coords=world_coords,
            tasks_delegate=tasks_delegate,
        )
    return hands_detector, pose_detector, hands_sp
