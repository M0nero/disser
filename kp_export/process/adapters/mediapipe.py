from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from pathlib import Path

from ...mp.mp_utils import create_hand_detector, create_pose_detector, resolve_task_model_path
from ..heuristics.constants import (
    HAND_SP_MIN_DETECTION,
    HAND_SP_MIN_TRACKING,
    SP_MIN_DET_MULTIPLIER,
    SP_MIN_TRACK_MULTIPLIER,
)
from .protocols import DetectorFactoryProtocol


@dataclass
class MediaPipeVideoDetectors:
    hands_detector: Any
    pose_detector: Any
    hands_sp: Optional[Any]

    def close(self) -> None:
        if self.pose_detector is not None:
            self.pose_detector.close()
        if self.hands_detector is not None:
            self.hands_detector.close()


class MediaPipeGpuSession:
    def __init__(
        self,
        *,
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
    ) -> None:
        self.backend = str(backend)
        self.mp = mp
        self.mp_solutions = mp_solutions
        self.hand_model_path = hand_model_path
        self.pose_model_path = pose_model_path
        self.min_det = float(min_det)
        self.min_track = float(min_track)
        self.pose_complexity = int(pose_complexity)
        self.tasks_delegate = tasks_delegate
        self.second_pass = bool(second_pass)
        self.world_coords = bool(world_coords)
        self._hands_sp = None
        self._warmed = False

    def _create_static_second_pass(self) -> Optional[Any]:
        if not self.second_pass:
            return None
        if self._hands_sp is None:
            self._hands_sp = create_hand_detector(
                self.backend,
                self.mp,
                self.mp_solutions,
                max_num_hands=1,
                min_det=max(HAND_SP_MIN_DETECTION, self.min_det * SP_MIN_DET_MULTIPLIER),
                min_track=max(HAND_SP_MIN_TRACKING, self.min_track * SP_MIN_TRACK_MULTIPLIER),
                static_image_mode=True,
                model_path=self.hand_model_path,
                world_coords=self.world_coords,
                tasks_delegate=self.tasks_delegate,
            )
        return self._hands_sp

    def warmup(self) -> None:
        if self._warmed:
            return
        hands_detector = create_hand_detector(
            self.backend,
            self.mp,
            self.mp_solutions,
            max_num_hands=2,
            min_det=self.min_det,
            min_track=self.min_track,
            static_image_mode=False,
            model_path=self.hand_model_path,
            world_coords=self.world_coords,
            tasks_delegate=self.tasks_delegate,
        )
        pose_detector = create_pose_detector(
            self.backend,
            self.mp,
            self.mp_solutions,
            model_complexity=self.pose_complexity,
            min_det=self.min_det,
            min_track=self.min_track,
            enable_segmentation=False,
            model_path=self.pose_model_path,
            tasks_delegate=self.tasks_delegate,
        )
        pose_detector.close()
        hands_detector.close()
        self._create_static_second_pass()
        self._warmed = True

    def create_video_detectors(self) -> MediaPipeVideoDetectors:
        hands_detector = create_hand_detector(
            self.backend,
            self.mp,
            self.mp_solutions,
            max_num_hands=2,
            min_det=self.min_det,
            min_track=self.min_track,
            static_image_mode=False,
            model_path=self.hand_model_path,
            world_coords=self.world_coords,
            tasks_delegate=self.tasks_delegate,
        )
        pose_detector = create_pose_detector(
            self.backend,
            self.mp,
            self.mp_solutions,
            model_complexity=self.pose_complexity,
            min_det=self.min_det,
            min_track=self.min_track,
            enable_segmentation=False,
            model_path=self.pose_model_path,
            tasks_delegate=self.tasks_delegate,
        )
        return MediaPipeVideoDetectors(
            hands_detector=hands_detector,
            pose_detector=pose_detector,
            hands_sp=self._create_static_second_pass(),
        )

    def close(self) -> None:
        if self._hands_sp is not None:
            self._hands_sp.close()
            self._hands_sp = None


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


class MediaPipeDetectorFactory:
    def create(
        self,
        *,
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
    ):
        return _initialize_detectors(
            backend=backend,
            mp=mp,
            mp_solutions=mp_solutions,
            hand_model_path=hand_model_path,
            pose_model_path=pose_model_path,
            min_det=min_det,
            min_track=min_track,
            pose_complexity=pose_complexity,
            tasks_delegate=tasks_delegate,
            second_pass=second_pass,
            world_coords=world_coords,
        )
