from __future__ import annotations

from typing import Any, Dict, List

from ..contracts import FrameDiagnostics, FrameRecord, HandObservation, PoseObservation
from .rows import FRAME_ROW_RAW_KEYS, frame_key_to_column, scalarize


COLUMN_TO_RAW_KEY = {frame_key_to_column(raw_key): raw_key for raw_key in FRAME_ROW_RAW_KEYS}


def frame_record_from_legacy(frame_idx: int, frame: Dict[str, Any]) -> FrameRecord:
    hand_1 = HandObservation(
        landmarks=frame.get("hand 1"),
        score=frame.get("hand 1_score"),
        score_gate=frame.get("hand 1_score_gate"),
        source=frame.get("hand 1_source"),
        state=frame.get("hand 1_state"),
        is_anchor=frame.get("hand 1_is_anchor"),
        reject_reason=frame.get("hand 1_reject_reason"),
        pose_quality=frame.get("hand 1_pose_quality"),
        wrist_z=frame.get("hand 1_wrist_z"),
        track_age=frame.get("hand_1_track_age"),
        track_reset=frame.get("hand_1_track_reset"),
        tracker_ready=frame.get("hand_1_tracker_ready"),
        tracker_last_score=frame.get("hand 1_tracker_last_score"),
        tracker_last_ts=frame.get("hand 1_tracker_last_ts"),
    )
    hand_2 = HandObservation(
        landmarks=frame.get("hand 2"),
        score=frame.get("hand 2_score"),
        score_gate=frame.get("hand 2_score_gate"),
        source=frame.get("hand 2_source"),
        state=frame.get("hand 2_state"),
        is_anchor=frame.get("hand 2_is_anchor"),
        reject_reason=frame.get("hand 2_reject_reason"),
        pose_quality=frame.get("hand 2_pose_quality"),
        wrist_z=frame.get("hand 2_wrist_z"),
        track_age=frame.get("hand_2_track_age"),
        track_reset=frame.get("hand_2_track_reset"),
        tracker_ready=frame.get("hand_2_tracker_ready"),
        tracker_last_score=frame.get("hand 2_tracker_last_score"),
        tracker_last_ts=frame.get("hand 2_tracker_last_ts"),
    )
    pose = PoseObservation(
        landmarks=frame.get("pose"),
        visibility=frame.get("pose_vis"),
        interpolated=bool(frame.get("pose_interpolated")),
    )
    diagnostics = {
        frame_key_to_column(raw_key): scalarize(frame.get(raw_key))
        for raw_key in FRAME_ROW_RAW_KEYS
    }
    return FrameRecord(
        frame_idx=int(frame_idx),
        ts_ms=int(frame.get("ts", 0) or 0),
        dt_ms=int(frame.get("dt", 0) or 0),
        hand_1=hand_1,
        hand_2=hand_2,
        pose=pose,
        both_hands=bool(frame.get("both_hands"))
        if frame.get("both_hands") is not None
        else bool(frame.get("hand 1") is not None and frame.get("hand 2") is not None),
        diagnostics=FrameDiagnostics(values=diagnostics),
        extras={
            key: frame.get(key)
            for key in (
                "hand_mask",
                "hand 1_sp_roi_px",
                "hand 2_sp_roi_px",
                "hand 1_sp_center_hint_px",
                "hand 1_sp_debug",
                "hand 1_sp_params",
                "hand 2_sp_center_hint_px",
                "hand 2_sp_debug",
                "hand 2_sp_params",
            )
            if key in frame
        },
    )


def build_frame_records(frames: List[Dict[str, Any]]) -> List[FrameRecord]:
    return [frame_record_from_legacy(idx, frame) for idx, frame in enumerate(frames)]


def legacy_frame_from_record(record: FrameRecord) -> Dict[str, Any]:
    frame: Dict[str, Any] = {
        "ts": int(record.ts_ms),
        "dt": int(record.dt_ms),
        "hand 1": record.hand_1.landmarks,
        "hand 2": record.hand_2.landmarks,
        "pose": record.pose.landmarks,
        "pose_vis": record.pose.visibility,
        "pose_interpolated": bool(record.pose.interpolated),
        "both_hands": 1 if record.both_hands else 0,
    }
    for key, value in record.diagnostics.to_dict().items():
        raw_key = COLUMN_TO_RAW_KEY.get(key, key)
        frame[raw_key] = value
    frame.update(record.extras)
    return frame


def legacy_frames_from_records(records: List[FrameRecord]) -> List[Dict[str, Any]]:
    return [legacy_frame_from_record(record) for record in records]
