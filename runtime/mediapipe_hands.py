from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .skeleton import CanonicalSkeletonSequence, HAND_JOINTS, combine_hands, canonicalize_sequence


POSE_LANDMARKS = 33


@dataclass
class MediaPipeHandsConfig:
    static_image_mode: bool = False
    max_num_hands: int = 2
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class MediaPipeHolisticConfig:
    static_image_mode: bool = False
    model_complexity: int = 1
    smooth_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


def _landmarks_to_array(landmarks: Any) -> np.ndarray:
    arr = np.zeros((HAND_JOINTS, 3), dtype=np.float32)
    if landmarks is None:
        return arr
    pts = getattr(landmarks, "landmark", [])
    for i, lm in enumerate(pts[:HAND_JOINTS]):
        arr[i, 0] = float(getattr(lm, "x", 0.0))
        arr[i, 1] = float(getattr(lm, "y", 0.0))
        arr[i, 2] = float(getattr(lm, "z", 0.0))
    return arr


def _pose_landmarks_to_array(landmarks: Any) -> tuple[np.ndarray, np.ndarray]:
    arr = np.zeros((POSE_LANDMARKS, 3), dtype=np.float32)
    vis = np.zeros((POSE_LANDMARKS,), dtype=np.float32)
    if landmarks is None:
        return arr, vis
    pts = getattr(landmarks, "landmark", [])
    for i, lm in enumerate(pts[:POSE_LANDMARKS]):
        arr[i, 0] = float(getattr(lm, "x", 0.0))
        arr[i, 1] = float(getattr(lm, "y", 0.0))
        arr[i, 2] = float(getattr(lm, "z", 0.0))
        vis[i] = float(getattr(lm, "visibility", 0.0) or 0.0)
    return arr, vis


def _assign_detected_hands(
    detections: List[Tuple[str, float, np.ndarray]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    left: Optional[np.ndarray] = None
    right: Optional[np.ndarray] = None
    left_score = -1.0
    right_score = -1.0
    unknown: List[Tuple[float, np.ndarray]] = []
    for side, score, arr in detections:
        s = str(side or "").strip().lower()
        if s == "left":
            if score > left_score:
                left = arr
                left_score = float(score)
        elif s == "right":
            if score > right_score:
                right = arr
                right_score = float(score)
        else:
            unknown.append((float(score), arr))
    unknown.sort(key=lambda x: x[0], reverse=True)
    for score, arr in unknown:
        if left is None:
            left = arr
            left_score = score
        elif right is None:
            right = arr
            right_score = score
    return left, right, {
        "left_score": float(max(left_score, 0.0)),
        "right_score": float(max(right_score, 0.0)),
        "num_detections": int(len(detections)),
    }


class MediaPipeHandsTracker:
    def __init__(self, cfg: Optional[MediaPipeHandsConfig] = None) -> None:
        self.cfg = cfg or MediaPipeHandsConfig()
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=bool(self.cfg.static_image_mode),
            max_num_hands=int(self.cfg.max_num_hands),
            model_complexity=int(self.cfg.model_complexity),
            min_detection_confidence=float(self.cfg.min_detection_confidence),
            min_tracking_confidence=float(self.cfg.min_tracking_confidence),
        )

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()

    def __enter__(self) -> "MediaPipeHandsTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def process_bgr(self, frame_bgr: np.ndarray, *, ts_ms: Optional[float] = None) -> Dict[str, Any]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        detections: List[Tuple[str, float, np.ndarray]] = []
        hands = result.multi_hand_landmarks or []
        handedness = result.multi_handedness or []
        for idx, hand_landmarks in enumerate(hands):
            label = ""
            score = 0.0
            if idx < len(handedness):
                try:
                    cls = handedness[idx].classification[0]
                    label = str(getattr(cls, "label", "") or "").lower()
                    score = float(getattr(cls, "score", 0.0) or 0.0)
                except Exception:
                    label = ""
                    score = 0.0
            detections.append((label, score, _landmarks_to_array(hand_landmarks)))
        left, right, meta = _assign_detected_hands(detections)
        pts, mask = combine_hands(left, right)
        return {
            "pts": pts.astype(np.float32, copy=False),
            "mask": mask.astype(np.float32, copy=False),
            "ts_ms": float(ts_ms if ts_ms is not None else 0.0),
            "meta": {
                **meta,
                "extractor_mode": "hands_only",
                "extractor": "mediapipe.solutions.hands",
            },
        }


class MediaPipeHolisticTracker:
    def __init__(self, cfg: Optional[MediaPipeHolisticConfig] = None) -> None:
        self.cfg = cfg or MediaPipeHolisticConfig()
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=bool(self.cfg.static_image_mode),
            model_complexity=int(self.cfg.model_complexity),
            smooth_landmarks=bool(self.cfg.smooth_landmarks),
            min_detection_confidence=float(self.cfg.min_detection_confidence),
            min_tracking_confidence=float(self.cfg.min_tracking_confidence),
        )

    def close(self) -> None:
        if self._holistic is not None:
            self._holistic.close()

    def __enter__(self) -> "MediaPipeHolisticTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def process_bgr(self, frame_bgr: np.ndarray, *, ts_ms: Optional[float] = None) -> Dict[str, Any]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._holistic.process(rgb)
        left = _landmarks_to_array(getattr(result, "left_hand_landmarks", None)) if getattr(result, "left_hand_landmarks", None) is not None else None
        right = _landmarks_to_array(getattr(result, "right_hand_landmarks", None)) if getattr(result, "right_hand_landmarks", None) is not None else None
        pose_xyz, pose_vis = _pose_landmarks_to_array(getattr(result, "pose_landmarks", None))
        pts, mask = combine_hands(left, right)
        return {
            "pts": pts.astype(np.float32, copy=False),
            "mask": mask.astype(np.float32, copy=False),
            "pose_xyz": pose_xyz.astype(np.float32, copy=False),
            "pose_vis": pose_vis.astype(np.float32, copy=False),
            "pose_indices": list(range(POSE_LANDMARKS)),
            "ts_ms": float(ts_ms if ts_ms is not None else 0.0),
            "meta": {
                "left_detected": bool(left is not None),
                "right_detected": bool(right is not None),
                "extractor_mode": "holistic_hands_pose",
                "extractor": "mediapipe.solutions.holistic",
            },
        }


def extract_video_sequence(
    video_path: str | Path,
    *,
    tracker_cfg: Optional[MediaPipeHandsConfig] = None,
    holistic_cfg: Optional[MediaPipeHolisticConfig] = None,
    require_pose: bool = False,
    max_frames: int = 0,
    return_frames: bool = False,
) -> Tuple[CanonicalSkeletonSequence, List[np.ndarray]]:
    source = Path(video_path)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.0:
        fps = 30.0
    dt_ms = 1000.0 / fps
    pts_rows: List[np.ndarray] = []
    mask_rows: List[np.ndarray] = []
    ts_rows: List[float] = []
    pose_rows: List[np.ndarray] = []
    pose_vis_rows: List[np.ndarray] = []
    raw_frames: List[np.ndarray] = []
    frame_idx = 0
    tracker_cm = (
        MediaPipeHolisticTracker(cfg=holistic_cfg or MediaPipeHolisticConfig()) if require_pose else MediaPipeHandsTracker(cfg=tracker_cfg)
    )
    with tracker_cm as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if return_frames:
                raw_frames.append(frame.copy())
            ts_ms = float(frame_idx) * dt_ms
            item = tracker.process_bgr(frame, ts_ms=ts_ms)
            pts_rows.append(item["pts"])
            mask_rows.append(item["mask"])
            ts_rows.append(ts_ms)
            if require_pose:
                pose_rows.append(np.asarray(item.get("pose_xyz"), dtype=np.float32))
                pose_vis_rows.append(np.asarray(item.get("pose_vis"), dtype=np.float32))
            frame_idx += 1
            if max_frames > 0 and frame_idx >= int(max_frames):
                break
    cap.release()
    seq = canonicalize_sequence(
        np.stack(pts_rows, axis=0) if pts_rows else np.zeros((0, 42, 3), dtype=np.float32),
        np.stack(mask_rows, axis=0) if mask_rows else np.zeros((0, 42, 1), dtype=np.float32),
        np.asarray(ts_rows, dtype=np.float32),
        pose_xyz=(np.stack(pose_rows, axis=0) if pose_rows else None),
        pose_vis=(np.stack(pose_vis_rows, axis=0) if pose_vis_rows else None),
        pose_indices=(list(range(POSE_LANDMARKS)) if require_pose else None),
        meta={
            "source_video": str(source),
            "fps": float(fps),
            "frame_dt_ms": float(dt_ms),
            "coords": "image",
            "extractor_mode": ("holistic_hands_pose" if require_pose else "hands_only"),
            "extractor": ("mediapipe.solutions.holistic" if require_pose else "mediapipe.solutions.hands"),
            "pose_indices": (list(range(POSE_LANDMARKS)) if require_pose else []),
        },
    )
    return seq, raw_frames
