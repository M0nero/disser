from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..contracts import FrameRecord


FRAME_ROW_RAW_KEYS = [
    "hand 1_score",
    "hand 1_score_gate",
    "hand 1_source",
    "hand 1_state",
    "hand 1_is_anchor",
    "hand 1_reject_reason",
    "hand 1_pose_quality",
    "hand 1_wrist_z",
    "hand 1_sanity_scale_ratio",
    "hand 1_sanity_wrist_jump",
    "hand 1_sanity_bone_max_err",
    "hand 1_sanity_bone_worst",
    "hand 1_sanity_stage",
    "hand_1_track_age",
    "hand_1_track_reset",
    "hand_1_tracker_ready",
    "hand 1_tracker_last_score",
    "hand 1_tracker_last_ts",
    "hand 2_score",
    "hand 2_score_gate",
    "hand 2_source",
    "hand 2_state",
    "hand 2_is_anchor",
    "hand 2_reject_reason",
    "hand 2_pose_quality",
    "hand 2_wrist_z",
    "hand 2_sanity_scale_ratio",
    "hand 2_sanity_wrist_jump",
    "hand 2_sanity_bone_max_err",
    "hand 2_sanity_bone_worst",
    "hand 2_sanity_stage",
    "hand_2_track_age",
    "hand_2_track_reset",
    "hand_2_tracker_ready",
    "hand 2_tracker_last_score",
    "hand 2_tracker_last_ts",
    "swap_applied",
    "dedup_triggered",
    "dedup_removed",
    "dedup_iou",
    "dedup_dist_norm",
    "hand_1_track_ok",
    "hand_2_track_ok",
    "hand_1_sp_attempted",
    "hand_2_sp_attempted",
    "hand_1_sp_recovered",
    "hand_2_sp_recovered",
    "sp_overlap_iou",
    "overlap_iou_sp",
    "sp_overlap",
    "iou_hands",
    "overlap_iou_ref",
    "overlap_guard_pre",
    "overlap_guard",
    "overlap_freeze",
    "overlap_freeze_reason",
    "overlap_freeze_side",
    "overlap_iou_guard",
    "overlap_iou_guard_src",
    "overlap_z_hint",
    "overlap_z_abs",
    "overlap_hand_z_diff",
    "pose_side_ambiguous",
    "pose_side_ratio",
    "side_d_ll",
    "side_d_lr",
    "side_d_rl",
    "side_d_rr",
    "side_cost_current",
    "side_cost_swap",
    "side_pref_left",
    "side_pref_right",
    "side_ok_1",
    "side_ok_2",
    "occluded_1",
    "occluded_2",
    "occluded_1_ttl",
    "occluded_2_ttl",
    "occlusion_freeze_age_1",
    "occlusion_freeze_age_2",
    "occlusion_freeze_max",
    "occlusion_saved_1",
    "occlusion_saved_2",
    "missing_pre_occ_1",
    "missing_pre_occ_2",
    "occlusion_iou",
    "occlusion_z_diff",
    "occlusion_behind_diff",
    "occlusion_samples_ok",
    "pose_wrist_dist_1",
    "pose_wrist_dist_2",
    "pose_wrist_z_left_world",
    "pose_wrist_z_right_world",
    "pose_wrist_z_left_img",
    "pose_wrist_z_right_img",
    "score_source",
    "min_hand_score",
    "hand_score_lo",
    "hand_score_hi",
    "anchor_score",
    "tracker_init_score",
    "tracker_update_score",
    "pose_dist_qual_min",
    "pose_interpolated",
]


def frame_key_to_column(key: str) -> str:
    return str(key).replace(" ", "_")


def scalarize(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def frame_row_from_record(record: FrameRecord) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "frame_idx": int(record.frame_idx),
        "ts_ms": int(record.ts_ms),
        "dt_ms": int(record.dt_ms),
        "hand_1_present": bool(record.hand_1.landmarks is not None),
        "hand_2_present": bool(record.hand_2.landmarks is not None),
        "pose_present": bool(record.pose.landmarks is not None),
        "both_hands": bool(record.both_hands),
    }
    row.update(record.diagnostics.to_dict())
    return row


def build_frame_rows_from_records(records: List[FrameRecord]) -> List[Dict[str, Any]]:
    return [frame_row_from_record(record) for record in records]
