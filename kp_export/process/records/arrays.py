from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..contracts import FrameRecord


HAND_POINTS = 21


def landmark_point_xyz(point: Any) -> tuple[float, float, float]:
    if isinstance(point, dict):
        return (
            float(point.get("x", 0.0)),
            float(point.get("y", 0.0)),
            float(point.get("z", 0.0)),
        )
    return (0.0, 0.0, 0.0)


def _hand_arrays_from_records(records: List[FrameRecord], *, which: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_frames = len(records)
    xyz = np.zeros((num_frames, HAND_POINTS, 3), dtype=np.float32)
    score = np.full((num_frames,), np.nan, dtype=np.float32)
    valid = np.zeros((num_frames,), dtype=np.bool_)
    for idx, record in enumerate(records):
        hand = record.hand_1 if which == "hand_1" else record.hand_2
        pts = hand.landmarks
        if isinstance(pts, list):
            valid[idx] = True
            for point_idx, point in enumerate(pts[:HAND_POINTS]):
                xyz[idx, point_idx] = landmark_point_xyz(point)
        if hand.score is not None:
            score[idx] = float(hand.score)
    return xyz, score, valid


def _pose_arrays_from_records(records: List[FrameRecord], *, pose_joint_count: int) -> tuple[np.ndarray, np.ndarray]:
    num_frames = len(records)
    pose_xyz = np.zeros((num_frames, pose_joint_count, 3), dtype=np.float32)
    pose_vis = np.zeros((num_frames, pose_joint_count), dtype=np.float32)
    if pose_joint_count <= 0:
        return pose_xyz, pose_vis
    for idx, record in enumerate(records):
        pts = record.pose.landmarks
        if isinstance(pts, list):
            limit = min(len(pts), pose_joint_count)
            for point_idx in range(limit):
                pose_xyz[idx, point_idx] = landmark_point_xyz(pts[point_idx])
            if isinstance(record.pose.visibility, list):
                for point_idx in range(min(len(record.pose.visibility), pose_joint_count)):
                    pose_vis[idx, point_idx] = float(record.pose.visibility[point_idx] or 0.0)
            else:
                pose_vis[idx, :limit] = 1.0
    return pose_xyz, pose_vis


def extract_sample_arrays_from_records(records: List[FrameRecord], *, meta_header: Dict[str, Any]) -> Dict[str, np.ndarray]:
    left_xyz, left_score, left_valid = _hand_arrays_from_records(records, which="hand_1")
    right_xyz, right_score, right_valid = _hand_arrays_from_records(records, which="hand_2")
    pose_joint_count = 0
    raw_pose_indices = meta_header.get("pose_indices")
    if isinstance(raw_pose_indices, list):
        pose_joint_count = len(raw_pose_indices)
    if pose_joint_count <= 0:
        for record in records:
            if isinstance(record.pose.landmarks, list):
                pose_joint_count = len(record.pose.landmarks)
                break
    pose_xyz, pose_vis = _pose_arrays_from_records(records, pose_joint_count=pose_joint_count)
    ts_ms = np.asarray([int(record.ts_ms) for record in records], dtype=np.int64)
    return {
        "ts_ms": ts_ms,
        "left_xyz": left_xyz,
        "right_xyz": right_xyz,
        "left_score": left_score,
        "right_score": right_score,
        "left_valid": left_valid,
        "right_valid": right_valid,
        "pose_xyz": pose_xyz,
        "pose_vis": pose_vis,
    }
