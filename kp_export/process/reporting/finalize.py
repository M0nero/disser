from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ...algos.postprocess import postprocess_sequence
from ...algos.tracking import smooth_tracks
from ...core.types import VideoMeta
from ..contracts import FrameRecord
from .io import write_smoothed_ndjson
from .metrics import count_outliers, diag, gap_stats, is_sanity_reject, mean_gap_len


@dataclass
class ReportingContext:
    sample_id: str
    video_name: str
    source_video: str
    fps: float
    backend: str
    tasks_delegate: str
    processing_elapsed: float
    hand_runtime: float
    pose_runtime: float
    second_pass_runtime: float
    second_pass_enabled: bool
    hands_present: int
    sp_rec_left: int
    sp_rec_right: int
    sp_missing_left_pre: int
    sp_missing_right_pre: int
    ndjson_path: Optional[Path]
    eval_mode: bool
    postprocess: bool
    pp_max_gap: int
    pp_smoother: str
    pp_only_anchors: bool
    hand_hi: float
    world_coords: bool


@dataclass
class ReportingResult:
    frame_records_pp: Optional[List[FrameRecord]]
    pp_stats: Dict[str, Any]
    manifest_entry: VideoMeta
    manifest_dict: Dict[str, Any]
    summary_metrics: Dict[str, Any]


def finalize_records(
    *,
    frame_records: List[FrameRecord],
    context: ReportingContext,
) -> ReportingResult:
    frames_total = max(1, len(frame_records))
    left_cov = sum(1 for fr in frame_records if fr.hand_1.landmarks is not None) / frames_total
    right_cov = sum(1 for fr in frame_records if fr.hand_2.landmarks is not None) / frames_total
    both_cov = sum(
        1 for fr in frame_records if fr.hand_1.landmarks is not None and fr.hand_2.landmarks is not None
    ) / frames_total
    pose_cov = sum(1 for fr in frame_records if fr.pose.landmarks is not None) / frames_total
    pose_interp_frac = sum(1 for fr in frame_records if fr.pose.interpolated) / frames_total

    left_scores = [fr.hand_1.score for fr in frame_records if fr.hand_1.score is not None]
    right_scores = [fr.hand_2.score for fr in frame_records if fr.hand_2.score is not None]
    left_mean = float(np.mean(left_scores)) if left_scores else 0.0
    right_mean = float(np.mean(right_scores)) if right_scores else 0.0
    left_med = float(np.median(left_scores)) if left_scores else 0.0
    right_med = float(np.median(right_scores)) if right_scores else 0.0

    dts = [fr.dt_ms for fr in frame_records if fr.dt_ms > 0]
    dt_median = float(np.median(dts)) if dts else 0.0
    fps_est = (1000.0 / dt_median) if dt_median > 0 else (context.fps or 0.0)

    sp_recovered_left_frac = float(context.sp_rec_left) / float(frames_total)
    sp_recovered_right_frac = float(context.sp_rec_right) / float(frames_total)
    track_recovered_count_1 = sum(1 for fr in frame_records if diag(fr, "hand_1_track_ok"))
    track_recovered_count_2 = sum(1 for fr in frame_records if diag(fr, "hand_2_track_ok"))
    track_recovered_left_frac = track_recovered_count_1 / frames_total
    track_recovered_right_frac = track_recovered_count_2 / frames_total

    outlier_frames_1 = 0
    outlier_frames_2 = 0
    if context.eval_mode:
        outlier_frames_1 = count_outliers(frame_records, "hand_1")
        outlier_frames_2 = count_outliers(frame_records, "hand_2")

    smooth_tracks(frame_records)
    write_smoothed_ndjson(frame_records, context.ndjson_path)

    pp_stats: Dict[str, Any] = {
        "pp_filled_left": 0,
        "pp_filled_right": 0,
        "pp_gaps_filled_left": 0,
        "pp_gaps_filled_right": 0,
        "pp_smoothing_delta_left": 0.0,
        "pp_smoothing_delta_right": 0.0,
    }
    frame_records_pp = None
    if context.postprocess:
        frame_records_pp, pp_stats = postprocess_sequence(
            frame_records,
            hi=context.hand_hi,
            max_gap=context.pp_max_gap,
            smoother=context.pp_smoother,
            only_anchors=context.pp_only_anchors,
            world_coords=context.world_coords,
        )

    eval_metrics = None
    if context.eval_mode:
        missing_frames_1 = sum(1 for fr in frame_records if fr.hand_1.landmarks is None)
        missing_frames_2 = sum(1 for fr in frame_records if fr.hand_2.landmarks is None)
        occluded_frames_1 = sum(1 for fr in frame_records if diag(fr, "occluded_1"))
        occluded_frames_2 = sum(1 for fr in frame_records if diag(fr, "occluded_2"))
        swap_frames = sum(1 for fr in frame_records if diag(fr, "swap_applied"))
        dedup_trigger_frames = sum(1 for fr in frame_records if diag(fr, "dedup_triggered"))
        sp_attempt_frames_1 = sum(1 for fr in frame_records if diag(fr, "hand_1_sp_attempted"))
        sp_attempt_frames_2 = sum(1 for fr in frame_records if diag(fr, "hand_2_sp_attempted"))
        sp_recovered_frames_1 = sum(1 for fr in frame_records if diag(fr, "hand_1_sp_recovered"))
        sp_recovered_frames_2 = sum(1 for fr in frame_records if diag(fr, "hand_2_sp_recovered"))
        hold_frames_1 = sum(1 for fr in frame_records if diag(fr, "hand_1_track_ok"))
        hold_frames_2 = sum(1 for fr in frame_records if diag(fr, "hand_2_track_ok"))
        sanity_reject_frames_1 = sum(1 for fr in frame_records if is_sanity_reject(fr.hand_1.reject_reason))
        sanity_reject_frames_2 = sum(1 for fr in frame_records if is_sanity_reject(fr.hand_2.reject_reason))

        missing_gap_p50_1, missing_gap_p90_1, missing_gap_max_1 = gap_stats(
            frame_records, lambda fr: fr.hand_1.landmarks is None
        )
        missing_gap_p50_2, missing_gap_p90_2, missing_gap_max_2 = gap_stats(
            frame_records, lambda fr: fr.hand_2.landmarks is None
        )
        occluded_gap_p50_1, occluded_gap_p90_1, occluded_gap_max_1 = gap_stats(
            frame_records, lambda fr: diag(fr, "occluded_1")
        )
        occluded_gap_p50_2, occluded_gap_p90_2, occluded_gap_max_2 = gap_stats(
            frame_records, lambda fr: diag(fr, "occluded_2")
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
        if context.postprocess:
            eval_metrics.update({
                "pp_filled_left": int(pp_stats.get("pp_filled_left", 0)),
                "pp_filled_right": int(pp_stats.get("pp_filled_right", 0)),
                "pp_gaps_filled_left": int(pp_stats.get("pp_gaps_filled_left", 0)),
                "pp_gaps_filled_right": int(pp_stats.get("pp_gaps_filled_right", 0)),
                "pp_smoothing_delta_left": float(pp_stats.get("pp_smoothing_delta_left", 0.0)),
                "pp_smoothing_delta_right": float(pp_stats.get("pp_smoothing_delta_right", 0.0)),
            })

    quality_score = (
        0.4 * both_cov
        + 0.2 * (left_cov + right_cov) / 2.0
        + 0.2 * pose_cov
        + 0.2 * (left_med + right_med) / 2.0
    )

    manifest_entry = VideoMeta(
        sample_id=str(context.sample_id),
        slug=str(context.sample_id),
        source_video=str(context.source_video).replace("\\", "/"),
        num_frames=len(frame_records),
        fps=context.fps,
        hands_frames=context.hands_present,
        hands_coverage=round(context.hands_present / frames_total, 4),
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
        pp_filled_left=int(pp_stats.get("pp_filled_left", 0)),
        pp_filled_right=int(pp_stats.get("pp_filled_right", 0)),
        pp_gaps_filled_left=int(pp_stats.get("pp_gaps_filled_left", 0)),
        pp_gaps_filled_right=int(pp_stats.get("pp_gaps_filled_right", 0)),
        pp_smoothing_delta_left=float(pp_stats.get("pp_smoothing_delta_left", 0.0)),
        pp_smoothing_delta_right=float(pp_stats.get("pp_smoothing_delta_right", 0.0)),
    )

    mean_gap_len_hand_1 = mean_gap_len(frame_records, 1)
    mean_gap_len_hand_2 = mean_gap_len(frame_records, 2)
    sp_recovered_count_1 = sum(1 for fr in frame_records if diag(fr, "hand_1_sp_recovered"))
    sp_recovered_count_2 = sum(1 for fr in frame_records if diag(fr, "hand_2_sp_recovered"))
    sp_attempt_count_1 = sum(1 for fr in frame_records if diag(fr, "hand_1_sp_attempted"))
    sp_attempt_count_2 = sum(1 for fr in frame_records if diag(fr, "hand_2_sp_attempted"))
    occl_saved_1 = sum(1 for fr in frame_records if diag(fr, "occlusion_saved_1"))
    occl_saved_2 = sum(1 for fr in frame_records if diag(fr, "occlusion_saved_2"))
    miss_pre_1 = sum(1 for fr in frame_records if diag(fr, "missing_pre_occ_1"))
    miss_pre_2 = sum(1 for fr in frame_records if diag(fr, "missing_pre_occ_2"))
    sp_recovery_rate_1 = sp_recovered_count_1 / frames_total
    sp_recovery_rate_2 = sp_recovered_count_2 / frames_total
    sp_attempt_rate_1 = sp_attempt_count_1 / frames_total
    sp_attempt_rate_2 = sp_attempt_count_2 / frames_total
    sp_success_given_attempt_1 = (sp_recovered_count_1 / sp_attempt_count_1) if sp_attempt_count_1 > 0 else 0.0
    sp_success_given_attempt_2 = (sp_recovered_count_2 / sp_attempt_count_2) if sp_attempt_count_2 > 0 else 0.0
    sp_rescue_rate_1 = (sp_recovered_count_1 / context.sp_missing_left_pre) if context.sp_missing_left_pre > 0 else 0.0
    sp_rescue_rate_2 = (sp_recovered_count_2 / context.sp_missing_right_pre) if context.sp_missing_right_pre > 0 else 0.0
    occlusion_saved_left_frac = occl_saved_1 / frames_total
    occlusion_saved_right_frac = occl_saved_2 / frames_total
    occlusion_recall_1 = occl_saved_1 / max(1, miss_pre_1)
    occlusion_recall_2 = occl_saved_2 / max(1, miss_pre_2)
    swap_rate = sum(1 for fr in frame_records if diag(fr, "swap_applied")) / frames_total
    dedup_trigger_count = sum(1 for fr in frame_records if diag(fr, "dedup_triggered"))
    dedup_removed_hand_1_count = sum(1 for fr in frame_records if diag(fr, "dedup_removed") == "hand_1")
    dedup_removed_hand_2_count = sum(1 for fr in frame_records if diag(fr, "dedup_removed") == "hand_2")
    dedup_trigger_rate = dedup_trigger_count / frames_total
    dedup_removed_rate = (dedup_removed_hand_1_count + dedup_removed_hand_2_count) / frames_total
    dedup_iou_vals = [diag(fr, "dedup_iou") for fr in frame_records if diag(fr, "dedup_triggered") and diag(fr, "dedup_iou") is not None]
    dedup_dist_vals = [diag(fr, "dedup_dist_norm") for fr in frame_records if diag(fr, "dedup_triggered") and diag(fr, "dedup_dist_norm") is not None]
    dedup_avg_iou_on_trigger = float(np.mean(dedup_iou_vals)) if dedup_iou_vals else 0.0
    dedup_avg_dist_norm_on_trigger = float(np.mean(dedup_dist_vals)) if dedup_dist_vals else 0.0
    occlusion_rate_1 = sum(1 for fr in frame_records if diag(fr, "occluded_1")) / frames_total
    occlusion_rate_2 = sum(1 for fr in frame_records if diag(fr, "occluded_2")) / frames_total
    occlusion_samples_count = sum(1 for fr in frame_records if diag(fr, "occlusion_samples_ok"))
    occlusion_iou_vals = [diag(fr, "occlusion_iou") for fr in frame_records if diag(fr, "occlusion_iou") is not None]
    occlusion_zdiff_vals = [diag(fr, "occlusion_z_diff") for fr in frame_records if diag(fr, "occlusion_z_diff") is not None]
    occlusion_behind_vals = [diag(fr, "occlusion_behind_diff") for fr in frame_records if diag(fr, "occlusion_behind_diff") is not None]
    occlusion_iou_p50 = float(np.percentile(occlusion_iou_vals, 50)) if occlusion_samples_count > 0 and occlusion_iou_vals else None
    occlusion_iou_p90 = float(np.percentile(occlusion_iou_vals, 90)) if occlusion_samples_count > 0 and occlusion_iou_vals else None
    occlusion_zdiff_p50 = float(np.percentile(occlusion_zdiff_vals, 50)) if occlusion_zdiff_vals else 0.0
    occlusion_zdiff_p90 = float(np.percentile(occlusion_zdiff_vals, 90)) if occlusion_zdiff_vals else 0.0
    occlusion_behind_p50 = float(np.percentile(occlusion_behind_vals, 50)) if occlusion_behind_vals else None
    occlusion_behind_p90 = float(np.percentile(occlusion_behind_vals, 90)) if occlusion_behind_vals else None
    occlusion_iou_not_occ_vals = [
        diag(fr, "occlusion_iou")
        for fr in frame_records
        if diag(fr, "occlusion_samples_ok") and not diag(fr, "occluded_1") and not diag(fr, "occluded_2") and diag(fr, "occlusion_iou") is not None
    ]
    occlusion_iou_occ_vals = [
        diag(fr, "occlusion_iou")
        for fr in frame_records
        if (diag(fr, "occluded_1") or diag(fr, "occluded_2")) and diag(fr, "occlusion_iou") is not None
    ]
    occlusion_behind_not_occ_vals = [
        diag(fr, "occlusion_behind_diff")
        for fr in frame_records
        if diag(fr, "occlusion_samples_ok") and not diag(fr, "occluded_1") and not diag(fr, "occluded_2") and diag(fr, "occlusion_behind_diff") is not None
    ]
    occlusion_behind_occ_vals = [
        diag(fr, "occlusion_behind_diff")
        for fr in frame_records
        if diag(fr, "occlusion_samples_ok") and (diag(fr, "occluded_1") or diag(fr, "occluded_2")) and diag(fr, "occlusion_behind_diff") is not None
    ]
    occlusion_iou_not_occ_p90 = float(np.percentile(occlusion_iou_not_occ_vals, 90)) if occlusion_iou_not_occ_vals else None
    occlusion_iou_occ_p50 = float(np.percentile(occlusion_iou_occ_vals, 50)) if occlusion_iou_occ_vals else None
    occlusion_behind_not_occ_p90 = float(np.percentile(occlusion_behind_not_occ_vals, 90)) if occlusion_behind_not_occ_vals else None
    occlusion_behind_occ_p50 = float(np.percentile(occlusion_behind_occ_vals, 50)) if occlusion_behind_occ_vals else None
    both_present_count = sum(1 for fr in frame_records if fr.hand_1.landmarks is not None and fr.hand_2.landmarks is not None)
    occlusion_rate_when_both_present_1 = (
        sum(1 for fr in frame_records if fr.hand_1.landmarks is not None and fr.hand_2.landmarks is not None and diag(fr, "occluded_1"))
        / both_present_count if both_present_count > 0 else 0.0
    )
    occlusion_rate_when_both_present_2 = (
        sum(1 for fr in frame_records if fr.hand_1.landmarks is not None and fr.hand_2.landmarks is not None and diag(fr, "occluded_2"))
        / both_present_count if both_present_count > 0 else 0.0
    )
    occlusion_toggle_count_1 = 0
    occlusion_toggle_count_2 = 0
    if frame_records:
        prev_occ_1 = bool(diag(frame_records[0], "occluded_1"))
        prev_occ_2 = bool(diag(frame_records[0], "occluded_2"))
        for fr in frame_records[1:]:
            cur_occ_1 = bool(diag(fr, "occluded_1"))
            cur_occ_2 = bool(diag(fr, "occluded_2"))
            if cur_occ_1 != prev_occ_1:
                occlusion_toggle_count_1 += 1
                prev_occ_1 = cur_occ_1
            if cur_occ_2 != prev_occ_2:
                occlusion_toggle_count_2 += 1
                prev_occ_2 = cur_occ_2

    summary_metrics = {
        "video": context.video_name,
        "sample_id": str(context.sample_id),
        "frames": len(frame_records),
        "processing": {
            "duration_sec": round(context.processing_elapsed, 4),
            "hand_runtime_sec": round(context.hand_runtime, 4),
            "pose_runtime_sec": round(context.pose_runtime, 4),
            "second_pass_runtime_sec": round(context.second_pass_runtime, 4),
            "effective_fps": round(len(frame_records) / max(context.processing_elapsed, 1e-6), 3),
        },
        "mp_backend": context.backend,
        "mp_tasks_delegate": context.tasks_delegate,
        "second_pass_enabled": bool(context.second_pass_enabled),
        "sp_recovered_left": int(context.sp_rec_left),
        "sp_recovered_right": int(context.sp_rec_right),
        "track_recovered_left": int(track_recovered_count_1),
        "track_recovered_right": int(track_recovered_count_2),
        "ndjson_written": bool(context.ndjson_path),
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

    manifest_dict = manifest_entry.__dict__.copy()
    if eval_metrics:
        manifest_dict.update(eval_metrics)

    return ReportingResult(
        frame_records_pp=frame_records_pp,
        pp_stats=pp_stats,
        manifest_entry=manifest_entry,
        manifest_dict=manifest_dict,
        summary_metrics=summary_metrics,
    )
