from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from kp_export.algos.postprocess import postprocess_sequence
from kp_export.algos.sanity import check_hand_sanity
from kp_export.algos.tracking import smooth_tracks
from kp_export.config import (
    DebugConfig,
    ExtractorConfig,
    LoggingConfig,
    MediaPipeConfig,
    OcclusionConfig,
    OutputConfig,
    PoseConfig,
    PostprocessConfig,
    RuntimeConfig,
    SanityConfig,
    ScoreConfig,
    SecondPassConfig,
    TrackingConfig,
    VideoConfig,
)
from kp_export.output.schema import FRAME_PARQUET_COLUMNS, normalize_row
from kp_export.output.staging import load_staged_payload, write_staged_payload
from kp_export.process.contracts import FrameDiagnostics, FrameRecord, HandObservation, PoseObservation, SamplePayload, SecondPassResult
from kp_export.process.pipeline.recover import HandFrameState, apply_occlusion_transition, update_or_track_hand
from kp_export.process.pipeline.second_pass import _execute_second_pass
from kp_export.process.reporting import ReportingContext, finalize_records
from kp_export.process.state import HandRuntime, classify_hand_state
from kp_export.tasks import TaskSpec


def _config() -> ExtractorConfig:
    return ExtractorConfig(
        video=VideoConfig(in_dir="in", out_dir="out", pattern="*.mp4", stride=2, short_side=256, ts_source="auto"),
        pose=PoseConfig(keep_pose_indices=[0, 9], world_coords=False, pose_every=2, pose_complexity=1, pose_ema=0.1),
        score=ScoreConfig(min_det=0.4, min_track=0.35, min_hand_score=0.1, hand_score_lo=0.4, hand_score_hi=0.8),
        second_pass=SecondPassConfig(enabled=True, debug_roi=True),
        tracking=TrackingConfig(interp_hold=7, write_hand_mask=False, max_gap=20, score_decay=0.93, reset_ms=300),
        occlusion=OcclusionConfig(hyst_frames=20, return_k=1.3),
        sanity=SanityConfig(enabled=True, scale_range=(0.7, 1.35), wrist_k=2.0, bone_tol=0.3),
        postprocess=PostprocessConfig(enabled=True, max_gap=20, smoother="ema", only_anchors=True),
        mediapipe=MediaPipeConfig(backend="tasks", hand_task="hand.task", pose_task="pose.task", tasks_delegate="cpu"),
        debug=DebugConfig(ndjson="debug/frames.ndjson", debug_video=""),
        output=OutputConfig(stage_dir="stage"),
        runtime=RuntimeConfig(jobs=4, seed=123, video_count=8),
        logging=LoggingConfig(log_dir="logs", log_level="INFO"),
    )


class RuntimeContractsTests(unittest.TestCase):
    class _FakeTracker:
        def __init__(self, tracked=None, last_score=0.9):
            self._tracked = tracked
            self.last_score = last_score
            self.last_valid_ts = 0.0
            self.reset_calls = 0
            self.update_calls = 0

        def reset(self):
            self.reset_calls += 1

        def update(self, landmarks, ts, rgb, score=1.0):
            self.update_calls += 1
            self.last_score = score
            self.last_valid_ts = ts

        def track(self, rgb, ts):
            self.last_valid_ts = ts
            return self._tracked

    def _frame_record(
        self,
        *,
        frame_idx: int,
        ts_ms: int,
        left=None,
        right=None,
        left_score=None,
        right_score=None,
        left_source=None,
        right_source=None,
        left_state=None,
        right_state=None,
        left_anchor=False,
        right_anchor=False,
    ) -> FrameRecord:
        return FrameRecord(
            frame_idx=frame_idx,
            ts_ms=ts_ms,
            dt_ms=33,
            hand_1=HandObservation(
                landmarks=left,
                score=left_score,
                source=left_source,
                state=left_state,
                is_anchor=left_anchor,
            ),
            hand_2=HandObservation(
                landmarks=right,
                score=right_score,
                source=right_source,
                state=right_state,
                is_anchor=right_anchor,
            ),
            pose=PoseObservation(landmarks=None, visibility=None, interpolated=False),
            both_hands=bool(left is not None and right is not None),
            diagnostics=FrameDiagnostics(
                values={
                    "hand_1_source": left_source,
                    "hand_1_state": left_state,
                    "hand_1_is_anchor": bool(left_anchor),
                    "hand_2_source": right_source,
                    "hand_2_state": right_state,
                    "hand_2_is_anchor": bool(right_anchor),
                }
            ),
        )

    def test_config_roundtrip_and_process_kwargs(self) -> None:
        cfg = _config()
        restored = ExtractorConfig.from_dict(cfg.to_dict())
        kwargs = restored.to_process_video_kwargs()
        self.assertEqual(kwargs["pose_every"], 2)
        self.assertEqual(kwargs["mp_backend"], "tasks")
        self.assertEqual(kwargs["track_max_gap"], 20)
        self.assertEqual(kwargs["sanity_scale_range"], (0.7, 1.35))

    def test_task_spec_roundtrip(self) -> None:
        task = TaskSpec(
            sample_id="sample",
            slug="sample",
            source_video="/tmp/sample.mp4",
            config_dict=_config().to_dict(),
            frame_start=10,
            frame_end=20,
            segment_meta={"seg_uid": "sample", "split": "train"},
            ndjson_path="/tmp/sample.ndjson",
        )
        restored = TaskSpec.from_payload(task.to_payload())
        self.assertEqual(restored.sample_id, "sample")
        self.assertEqual(restored.frame_start, 10)
        self.assertEqual(restored.segment_meta["split"], "train")

    def test_staging_roundtrip_uses_directory_payload(self) -> None:
        payload = SamplePayload(
            sample_id="sample",
            slug="sample",
            source_video="/tmp/sample.mp4",
            sample_attrs={"sample_id": "sample", "coords_mode": "image"},
            video_row={"sample_id": "sample", "slug": "sample", "source_video": "/tmp/sample.mp4"},
            frame_rows=[{"frame_idx": 0, "ts_ms": 0, "dt_ms": 0, "hand_1_present": False, "hand_2_present": False, "pose_present": False, "both_hands": False}],
            raw_arrays={"ts_ms": np.asarray([0], dtype=np.int64), "left_xyz": np.zeros((1, 21, 3), dtype=np.float32)},
            pp_arrays=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            staged = write_staged_payload(tmp, "sample", payload)
            self.assertTrue(Path(staged).is_dir())
            loaded = load_staged_payload(staged)
            self.assertEqual(loaded["sample_id"], "sample")
            self.assertEqual(int(loaded["raw_arrays"]["ts_ms"][0]), 0)

    def test_normalize_row_rejects_unknown_fields(self) -> None:
        with self.assertRaises(KeyError):
            normalize_row({"sample_id": "x", "unknown": 1}, FRAME_PARQUET_COLUMNS)

    def test_sanity_returns_structured_result(self) -> None:
        pts = [{"x": 0.1, "y": 0.2, "z": 0.0, "visibility": 1.0} for _ in range(21)]
        result = check_hand_sanity(pts)
        self.assertTrue(result.ok)
        self.assertIsInstance(result.reason_codes, list)

    def test_execute_second_pass_without_detector_returns_result_object(self) -> None:
        result, elapsed = _execute_second_pass("left", None, None, hands_sp=None)
        self.assertIsInstance(result, SecondPassResult)
        self.assertFalse(result.recovered)
        self.assertGreaterEqual(elapsed, 0.0)

    def test_smooth_tracks_preserves_visibility_or_none(self) -> None:
        frames = [
            {"hand 1": [{"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.25}] * 21},
            {"hand 1": [{"x": 1.0, "y": 1.0, "z": 1.0}] * 21},
        ]
        smooth_tracks(frames, window_size=3)
        self.assertEqual(frames[0]["hand 1"][0]["visibility"], 0.25)
        self.assertIsNone(frames[1]["hand 1"][0]["visibility"])

    def test_smooth_tracks_supports_frame_records(self) -> None:
        frames = [
            self._frame_record(
                frame_idx=0,
                ts_ms=0,
                left=[{"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.25}] * 21,
            ),
            self._frame_record(
                frame_idx=1,
                ts_ms=33,
                left=[{"x": 1.0, "y": 1.0, "z": 1.0}] * 21,
            ),
        ]
        smooth_tracks(frames, window_size=3)
        self.assertEqual(frames[0].hand_1.landmarks[0]["visibility"], 0.25)
        self.assertIsNone(frames[1].hand_1.landmarks[0]["visibility"])

    def test_postprocess_sequence_supports_frame_records(self) -> None:
        anchor_left_a = [{"x": 0.1, "y": 0.2, "z": 0.0}] * 21
        anchor_left_b = [{"x": 0.3, "y": 0.4, "z": 0.0}] * 21
        frames = [
            self._frame_record(
                frame_idx=0,
                ts_ms=0,
                left=anchor_left_a,
                left_score=0.95,
                left_source="pass1",
                left_state="observed",
                left_anchor=True,
            ),
            self._frame_record(frame_idx=1, ts_ms=33),
            self._frame_record(
                frame_idx=2,
                ts_ms=66,
                left=anchor_left_b,
                left_score=0.96,
                left_source="pass1",
                left_state="observed",
                left_anchor=True,
            ),
        ]
        frames_pp, stats = postprocess_sequence(
            frames,
            hi=0.8,
            max_gap=2,
            smoother="none",
            only_anchors=True,
            world_coords=False,
        )
        self.assertIsInstance(frames_pp[1], FrameRecord)
        self.assertIsNotNone(frames_pp[1].hand_1.landmarks)
        self.assertEqual(frames_pp[1].hand_1.source, "interp")
        self.assertEqual(frames_pp[1].hand_1.state, "predicted")
        self.assertEqual(frames_pp[1].diagnostics.get("hand_1_source"), "interp")
        self.assertEqual(stats["pp_filled_left"], 1)

    def test_finalize_records_builds_reporting_outputs(self) -> None:
        frames = [
            self._frame_record(
                frame_idx=0,
                ts_ms=0,
                left=[{"x": 0.1, "y": 0.2, "z": 0.0}] * 21,
                left_score=0.9,
                left_source="pass1",
                left_state="observed",
                left_anchor=True,
            ),
            self._frame_record(
                frame_idx=1,
                ts_ms=33,
                right=[{"x": 0.7, "y": 0.8, "z": 0.0}] * 21,
                right_score=0.85,
                right_source="pass1",
                right_state="observed",
                right_anchor=True,
            ),
        ]
        result = finalize_records(
            frame_records=frames,
            context=ReportingContext(
                sample_id="sample",
                video_name="sample.mp4",
                source_video="/tmp/sample.mp4",
                fps=30.0,
                backend="tasks",
                tasks_delegate="cpu",
                processing_elapsed=0.5,
                hand_runtime=0.1,
                pose_runtime=0.2,
                second_pass_runtime=0.0,
                second_pass_enabled=True,
                hands_present=2,
                sp_rec_left=0,
                sp_rec_right=0,
                sp_missing_left_pre=1,
                sp_missing_right_pre=1,
                ndjson_path=None,
                eval_mode=True,
                postprocess=False,
                pp_max_gap=10,
                pp_smoother="ema",
                pp_only_anchors=True,
                hand_hi=0.8,
                world_coords=False,
            ),
        )
        self.assertEqual(result.manifest_entry.num_frames, 2)
        self.assertEqual(result.manifest_dict["sample_id"], "sample")
        self.assertEqual(result.summary_metrics["frames"], 2)
        self.assertTrue(result.summary_metrics["second_pass_enabled"])
        self.assertIsNone(result.frame_records_pp)

    def test_hand_runtime_tracks_history_export_and_anchor(self) -> None:
        pts = [{"x": 0.1, "y": 0.2, "z": 0.0}] * 21
        state = HandRuntime(side="left")
        state.note_observation(
            5,
            landmarks=pts,
            source="pass1",
            overlap_ambiguous=False,
            side_ok=True,
            overlap_guard=False,
        )
        self.assertIs(state.previous_observation(6), pts)
        self.assertIsNone(state.previous_observation(8))
        self.assertTrue(
            state.maybe_export(
                landmarks=pts,
                score=0.9,
                source="pass1",
                overlap_ambiguous=False,
                side_ok=True,
                overlap_guard=False,
                cur_px=pts,
                cur_img=pts,
            )
        )
        self.assertEqual(state.last_export_score, 0.9)
        self.assertIs(state.last_good_img, pts)
        self.assertTrue(
            state.maybe_anchor(
                5,
                landmarks=pts,
                score=0.95,
                source="pass1",
                anchor_score=0.85,
                pose_ok=True,
                overlap_ambiguous=False,
                side_ok=True,
                overlap_guard=False,
            )
        )
        self.assertIs(state.anchor_for_sanity(12, max_gap=10), pts)
        self.assertIsNone(state.anchor_for_sanity(20, max_gap=10))

    def test_classify_hand_state_uses_source_priority(self) -> None:
        pts = [{"x": 0.1, "y": 0.2, "z": 0.0}] * 21
        self.assertEqual(classify_hand_state(None, None), "missing")
        self.assertEqual(classify_hand_state(pts, "occluded"), "occluded")
        self.assertEqual(classify_hand_state(pts, "tracked"), "predicted")
        self.assertEqual(classify_hand_state(pts, "pass1"), "observed")

    def test_apply_occlusion_transition_restores_last_export(self) -> None:
        result = apply_occlusion_transition(
            HandFrameState(
                landmarks=None,
                score=None,
                source=None,
                reject_reason=None,
                cur_img=None,
                cur_px=None,
            ),
            side="left",
            occluded=True,
            occ_ttl=3,
            occ_freeze_age=0,
            hold=5,
            overlap_guard=False,
            overlap_freeze_side=None,
            score_source="handedness",
            hand_hi=0.8,
            anchor_score_eff=0.85,
            score_gate=None,
            pose_ok=True,
            side_ok_accept=True,
            det_img=None,
            last_good_img=None,
            last_export=[{"x": 0.1, "y": 0.2, "z": 0.0}] * 21,
            last_export_score=0.9,
            occ_freeze_max_frames=4,
            occ_return_k=1.2,
            world_coords=False,
            proc_w=256,
            proc_h=256,
            pose_img_landmarks=None,
            pose_world_full=None,
            last_pose_world_full=None,
        )
        self.assertTrue(result.occluded)
        self.assertEqual(result.hold, 0)
        self.assertIsNotNone(result.hand.landmarks)
        self.assertEqual(result.hand.source, "occluded")
        self.assertEqual(result.hand.score, 0.9)
        self.assertTrue(result.occlusion_saved)
        self.assertEqual(result.occ_freeze_age, 1)

    def test_update_or_track_hand_recovers_missing_hand(self) -> None:
        tracked = [{"x": 0.2, "y": 0.3, "z": 0.0}] * 21
        tracker = self._FakeTracker(tracked=tracked, last_score=0.8)
        result = update_or_track_hand(
            HandFrameState(
                landmarks=None,
                score=None,
                source=None,
                reject_reason=None,
                cur_img=None,
                cur_px=None,
            ),
            tracker=tracker,
            tracker_ready=True,
            track_age=0,
            hold=2,
            world_coords=False,
            overlap_ambiguous=False,
            side_ok=True,
            overlap_guard=False,
            pose_ok=True,
            block_track=False,
            tracker_init_score_eff=0.8,
            tracker_update_score_eff=None,
            score_gate=None,
            ts=33.0,
            dt=33.0,
            rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            track_reset_ms=300,
            track_max_gap=20,
            track_score_decay=0.93,
        )
        self.assertTrue(result.track_ok)
        self.assertFalse(result.track_reset)
        self.assertEqual(result.track_recovered_inc, 1)
        self.assertEqual(result.hand.source, "tracked")
        self.assertIsNotNone(result.hand.landmarks)
        self.assertAlmostEqual(result.hand.score, 0.8 * 0.93, places=6)
        self.assertEqual(result.track_age, 1)


if __name__ == "__main__":
    unittest.main()
