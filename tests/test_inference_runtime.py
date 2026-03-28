from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from bio.core.model import BioModelConfig, BioTagger
from bio.core.preprocessing import (
    BIO_PREPROCESSING_VERSION_V2,
    BIO_PREPROCESSING_VERSION_V3,
    BioPreprocessConfig,
    init_bio_preprocess_state,
    preprocess_frame_v3,
    preprocess_sequence,
)
from bio.runtime import BioDecoderConfig, BioSegmentDecoder, BioSegmenter, export_bio_runtime_bundle
from bio.pipeline.prelabel import PrelabelConfig, prepare_model_pts
from bio import runtime_commands as bio_runtime_commands
from msagcn.data.config import DSConfig
from msagcn.data.dataset import MultiStreamGestureDataset
from msagcn.models.agcn import MultiStreamAGCN
from msagcn import cli as msagcn_cli
from msagcn.runtime import MSAGCNClassifier, export_msagcn_runtime_bundle
from pipeline.app import InferencePipeline, InferencePipelineConfig, build_review_session, run_video_pipeline
from pipeline import cli as pipeline_cli
from desktop_review.session import load_review_session
from runtime.mediapipe_hands import _assign_detected_hands
from runtime.skeleton import canonicalize_sequence, combine_hands, frames_to_canonical_sequence, save_skeleton_sequence_npz
from runtime.sentence import SentenceBuilder


def _hand_points(base: float) -> list[dict[str, float]]:
    return [
        {
            "x": float(base + 0.01 * idx),
            "y": float(base + 0.02 * idx),
            "z": float(-0.005 * idx),
        }
        for idx in range(21)
    ]


def _pose_points(base: float) -> list[dict[str, float]]:
    return [
        {
            "x": float(base + 0.005 * idx),
            "y": float(base + 0.004 * idx),
            "z": float(-0.002 * idx),
        }
        for idx in range(33)
    ]


def _write_msagcn_fixture(root: Path, *, include_pose: bool = False) -> tuple[Path, Path, list[dict[str, object]], dict[str, object]]:
    json_dir = root / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in range(6):
        row = {
            "ts": float(t * 33.3333),
            "hand 1": _hand_points(0.1 + 0.01 * t),
            "hand 1_score": 0.99,
            "hand 2": _hand_points(0.6 + 0.01 * t),
            "hand 2_score": 0.98,
        }
        if include_pose:
            row["pose"] = _pose_points(0.2 + 0.01 * t)
            row["pose_vis"] = [0.95] * 33
        frames.append(row)
    meta = {"coords": "image", "fps": 30.0}
    if include_pose:
        meta["pose_indices"] = "all"
    (json_dir / "clip1.json").write_text(json.dumps({"frames": frames, "meta": meta}, ensure_ascii=False), encoding="utf-8")
    csv_path = root / "ann.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["attachment_id", "text", "begin", "end", "split"])
        writer.writeheader()
        writer.writerow({"attachment_id": "clip1", "text": "hello", "begin": 0, "end": 5, "split": "train"})
    return json_dir, csv_path, frames, meta


class _FakePrediction:
    def __init__(self, label: str = "hello", confidence: float = 0.9) -> None:
        self.label = label
        self.confidence = confidence

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "class_id": 0,
            "confidence": self.confidence,
            "topk": [{"rank": 1, "class_id": 0, "label": self.label, "prob": self.confidence}],
            "logits": [1.0],
            "probs": [self.confidence],
            "family_class_id": 7,
            "family_label": "family_7",
            "family_confidence": 0.7,
            "family_topk": [{"rank": 1, "class_id": 7, "label": "family_7", "prob": 0.7}],
            "family_logits": [1.0],
            "family_probs": [0.7],
            "meta": {},
        }


class _FakeDecoder:
    def flush(self, force: bool = False, eos_policy: str | None = None) -> list[object]:
        del force, eos_policy
        return []


class _FakeBioSegmenter:
    def __init__(self) -> None:
        self.decoder = _FakeDecoder()
        self._frame = 0
        self.threshold = 0.8
        self.metadata = {"config_resolution_source": "bundle", "decoder_version": "fake", "preprocessing_version": "test"}

    def step(self, pts, mask, *, ts_ms: float) -> dict[str, object]:
        del pts, mask, ts_ms
        frame = self._frame
        self._frame += 1
        segments = []
        if frame == 4:
            segments.append(
                {
                    "segment_id": 0,
                    "start_frame": 1,
                    "end_frame_exclusive": 4,
                    "start_time_ms": 33.3333,
                    "end_time_ms": 133.3332,
                    "boundary_score": 0.95,
                    "mean_inside_score": 0.88,
                    "threshold_used": 0.8,
                    "decoder_version": "fake",
                }
            )
        return {
            "frame_index": frame,
            "label": 0,
            "logits": [1.0, 0.0, 0.0],
            "probs": [0.8, 0.1, 0.1],
            "threshold": 0.8,
            "segments": segments,
        }


class _FakeMSAGCNClassifier:
    def __init__(self) -> None:
        self.ds_cfg = DSConfig(include_pose=False)
        self.metadata = {"config_resolution_source": "bundle", "preprocessing_version": "test"}

    def predict_sequence(self, seq):
        del seq
        return _FakePrediction()


class InferenceRuntimeTests(unittest.TestCase):
    def test_sentence_builder_duplicate_suppression(self) -> None:
        sb = SentenceBuilder()
        first = sb.add_prediction(segment_id=1, label="hello", confidence=0.9, start_time_ms=0.0, end_time_ms=500.0)
        second = sb.add_prediction(segment_id=2, label="hello", confidence=0.92, start_time_ms=600.0, end_time_ms=900.0)
        self.assertTrue(first.accepted)
        self.assertFalse(second.accepted)
        self.assertEqual(sb.sentence, "hello")

    def test_runtime_skeleton_handles_missing_hand(self) -> None:
        frames = [
            {"ts": 0.0, "hand 2": _hand_points(0.5)},
            {"ts": 33.0, "hand 1": _hand_points(0.1), "hand 2": _hand_points(0.6)},
        ]
        seq = frames_to_canonical_sequence(frames, meta={"fps": 30.0})
        self.assertEqual(seq.pts.shape, (2, 42, 3))
        self.assertEqual(float(seq.mask[0, :21].sum()), 0.0)
        self.assertGreater(float(seq.mask[0, 21:].sum()), 0.0)

    def test_combine_hands_places_left_and_right_in_fixed_slots(self) -> None:
        left = np.stack([np.array([100 + i, 200 + i, 300 + i], dtype=np.float32) for i in range(21)], axis=0)
        right = np.stack([np.array([400 + i, 500 + i, 600 + i], dtype=np.float32) for i in range(21)], axis=0)
        pts, mask = combine_hands(left, right)
        self.assertTrue(np.allclose(pts[:21], left))
        self.assertTrue(np.allclose(pts[21:], right))
        self.assertEqual(int(mask[:21].sum()), 21)
        self.assertEqual(int(mask[21:].sum()), 21)

    def test_assign_detected_hands_respects_handedness_labels(self) -> None:
        left = np.ones((21, 3), dtype=np.float32)
        right = np.full((21, 3), 2.0, dtype=np.float32)
        left_out, right_out, meta = _assign_detected_hands(
            [
                ("right", 0.9, right),
                ("left", 0.8, left),
            ]
        )
        self.assertTrue(np.allclose(left_out, left))
        self.assertTrue(np.allclose(right_out, right))
        self.assertEqual(meta["num_detections"], 2)

    def test_assign_detected_hands_unknown_fallback_fills_left_then_right(self) -> None:
        a = np.ones((21, 3), dtype=np.float32)
        b = np.full((21, 3), 2.0, dtype=np.float32)
        left_out, right_out, _meta = _assign_detected_hands(
            [
                ("", 0.9, a),
                ("", 0.8, b),
            ]
        )
        self.assertTrue(np.allclose(left_out, a))
        self.assertTrue(np.allclose(right_out, b))

    def test_runtime_skeleton_roundtrip_with_pose_sidecar(self) -> None:
        seq = canonicalize_sequence(
            np.zeros((3, 42, 3), dtype=np.float32),
            np.ones((3, 42, 1), dtype=np.float32),
            np.arange(3, dtype=np.float32) * 33.0,
            pose_xyz=np.zeros((3, 33, 3), dtype=np.float32),
            pose_vis=np.ones((3, 33), dtype=np.float32),
            pose_indices=list(range(33)),
            meta={"fps": 30.0},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seq.npz"
            save_skeleton_sequence_npz(path, seq)
            loaded = canonicalize_sequence(**{
                "pts": np.load(path)["pts"],
                "mask": np.load(path)["mask"],
                "ts_ms": np.load(path)["ts_ms"],
                "pose_xyz": np.load(path)["pose_xyz"],
                "pose_vis": np.load(path)["pose_vis"],
                "pose_indices": np.load(path)["pose_indices"].tolist(),
                "meta": {"fps": 30.0},
            })
        self.assertIsNotNone(loaded.pose_xyz)
        self.assertEqual(loaded.pose_xyz.shape, (3, 33, 3))

    def test_frames_to_canonical_sequence_reorders_pose_by_pose_indices(self) -> None:
        pose_a = _pose_points(0.1)
        pose_b = list(reversed(pose_a))
        frame0 = {
            "ts": 0.0,
            "hand 1": _hand_points(0.1),
            "pose": pose_a,
            "pose_vis": [0.9] * 33,
            "pose_indices": list(range(33)),
        }
        frame1 = {
            "ts": 33.0,
            "hand 1": _hand_points(0.2),
            "pose": pose_b,
            "pose_vis": [0.9] * 33,
            "pose_indices": list(reversed(range(33))),
        }
        seq = frames_to_canonical_sequence([frame0, frame1], meta={"fps": 30.0})
        self.assertIsNotNone(seq.pose_xyz)
        self.assertTrue(np.allclose(seq.pose_xyz[0, 0], seq.pose_xyz[1, 0], atol=1e-6))
        self.assertEqual(seq.pose_indices, tuple(range(33)))

    def test_bio_decoder_emits_segment_after_gap(self) -> None:
        decoder = BioSegmentDecoder(
            BioDecoderConfig(
                start_threshold=0.8,
                min_segment_frames=2,
                min_gap_frames=2,
                max_idle_inside_segment=4,
                cooldown_frames=0,
                stream_window=8,
            )
        )
        seq = [
            [0.9, 0.05, 0.05],
            [0.1, 0.85, 0.05],
            [0.1, 0.2, 0.7],
            [0.1, 0.15, 0.75],
            [0.8, 0.1, 0.1],
            [0.85, 0.08, 0.07],
        ]
        out = []
        for i, probs in enumerate(seq):
            out.extend(decoder.step(probs, ts_ms=float(i * 33.0)))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].start_frame, 1)
        self.assertEqual(out[0].end_frame_exclusive, 4)
        self.assertEqual(out[0].end_reason, "gap_closed")

    def test_bio_decoder_closes_valid_segment_on_eos(self) -> None:
        decoder = BioSegmentDecoder(
            BioDecoderConfig(
                start_threshold=0.8,
                min_segment_frames=2,
                min_gap_frames=2,
                cooldown_frames=0,
                eos_policy="close_open_segment_on_eos",
            )
        )
        decoder.step([0.1, 0.9, 0.0], ts_ms=0.0)
        decoder.step([0.1, 0.1, 0.8], ts_ms=33.0)
        flushed = decoder.flush(force=False, eos_policy="close_open_segment_on_eos")
        self.assertEqual(len(flushed), 1)
        self.assertEqual(flushed[0].end_reason, "eos_closed")

    def test_bio_decoder_drops_partial_segment_on_eos_in_live_mode(self) -> None:
        decoder = BioSegmentDecoder(
            BioDecoderConfig(
                start_threshold=0.8,
                min_segment_frames=4,
                min_gap_frames=2,
                cooldown_frames=0,
                eos_policy="drop_partial_on_eos",
            )
        )
        decoder.step([0.1, 0.9, 0.0], ts_ms=0.0)
        decoder.step([0.1, 0.1, 0.8], ts_ms=33.0)
        flushed = decoder.flush(force=False, eos_policy="drop_partial_on_eos")
        self.assertEqual(flushed, [])

    def test_bio_decoder_uses_explicit_continue_threshold(self) -> None:
        decoder = BioSegmentDecoder(
            BioDecoderConfig(
                start_threshold=0.8,
                continue_threshold=0.72,
                continue_threshold_policy="fixed_ratio",
            )
        )
        self.assertAlmostEqual(decoder._continue_threshold(), 0.72, places=6)

    def test_bio_decoder_guard_blocks_start_without_visible_hands(self) -> None:
        decoder = BioSegmentDecoder(
            BioDecoderConfig(
                start_threshold=0.8,
                cooldown_frames=0,
                require_hand_presence_to_start=True,
                min_visible_hand_frames_to_start=2,
                min_valid_hand_joints_to_start=8,
                allow_one_hand_to_start=True,
            )
        )
        out0 = decoder.step([0.1, 0.9, 0.0], ts_ms=0.0, left_valid_joints=0, right_valid_joints=0)
        out1 = decoder.step([0.1, 0.88, 0.02], ts_ms=33.0, left_valid_joints=0, right_valid_joints=0)
        self.assertEqual(out0, [])
        self.assertEqual(out1, [])
        self.assertIsNone(decoder._active)
        self.assertTrue(decoder.last_step_debug["start_blocked_by_hand_guard"])
        self.assertFalse(decoder.last_step_debug["hand_presence_ok"])

    def test_bio_decoder_guard_allows_start_after_stable_hand_presence(self) -> None:
        decoder = BioSegmentDecoder(
            BioDecoderConfig(
                start_threshold=0.8,
                cooldown_frames=0,
                require_hand_presence_to_start=True,
                min_visible_hand_frames_to_start=2,
                min_valid_hand_joints_to_start=8,
                allow_one_hand_to_start=True,
            )
        )
        decoder.step([0.95, 0.02, 0.03], ts_ms=0.0, left_valid_joints=0, right_valid_joints=10)
        decoder.step([0.1, 0.85, 0.05], ts_ms=33.0, left_valid_joints=0, right_valid_joints=10)
        self.assertIsNotNone(decoder._active)
        self.assertTrue(decoder.last_step_debug["hand_presence_ok"])
        self.assertFalse(decoder.last_step_debug["start_blocked_by_hand_guard"])

    def test_bio_decoder_guard_disabled_preserves_existing_start_behavior(self) -> None:
        decoder = BioSegmentDecoder(
            BioDecoderConfig(
                start_threshold=0.8,
                cooldown_frames=0,
                require_hand_presence_to_start=False,
            )
        )
        decoder.step([0.1, 0.9, 0.0], ts_ms=0.0, left_valid_joints=0, right_valid_joints=0)
        self.assertIsNotNone(decoder._active)

    def test_bio_decoder_signness_gate_blocks_low_active_start(self) -> None:
        decoder = BioSegmentDecoder(
            BioDecoderConfig(
                start_threshold=0.8,
                cooldown_frames=0,
                use_signness_gate=True,
                signness_start_threshold=0.55,
            )
        )
        decoder.step([0.1, 0.9, 0.0], ts_ms=0.0, active_prob=0.20, left_valid_joints=10, right_valid_joints=0)
        self.assertIsNone(decoder._active)
        self.assertFalse(decoder.last_step_debug["signness_gate_ok"])

    def test_bio_decoder_signness_gate_allows_start_when_active_is_high(self) -> None:
        decoder = BioSegmentDecoder(
            BioDecoderConfig(
                start_threshold=0.8,
                cooldown_frames=0,
                use_signness_gate=True,
                signness_start_threshold=0.55,
            )
        )
        decoder.step([0.1, 0.9, 0.0], ts_ms=0.0, active_prob=0.80, left_valid_joints=10, right_valid_joints=0)
        self.assertIsNotNone(decoder._active)
        self.assertTrue(decoder.last_step_debug["signness_gate_ok"])

    def test_bio_decoder_onset_gate_allows_start_when_pb_is_low(self) -> None:
        decoder = BioSegmentDecoder(
            BioDecoderConfig(
                start_threshold=0.8,
                cooldown_frames=0,
                use_signness_gate=True,
                use_onset_gate=True,
                active_start_threshold=0.25,
                onset_start_threshold=0.45,
            )
        )
        decoder.step(
            [0.2, 0.3, 0.5],
            ts_ms=0.0,
            active_prob=0.80,
            onset_prob=0.90,
            left_valid_joints=10,
            right_valid_joints=0,
        )
        self.assertIsNotNone(decoder._active)
        self.assertAlmostEqual(float(decoder.last_step_debug["p_onset"]), 0.90, places=6)
        self.assertTrue(bool(decoder.last_step_debug["clip_hit_candidate"]))

    def test_bio_segmenter_step_exposes_hand_guard_debug_fields(self) -> None:
        model = BioTagger(BioModelConfig()).eval()
        segmenter = BioSegmenter.from_model(
            model,
            threshold=0.8,
            decoder_cfg=BioDecoderConfig(
                require_hand_presence_to_start=True,
                min_visible_hand_frames_to_start=2,
                min_valid_hand_joints_to_start=8,
            ),
            device="cpu",
        )
        out = segmenter.step(
            np.zeros((42, 3), dtype=np.float32),
            np.zeros((42, 1), dtype=np.float32),
            ts_ms=0.0,
        )
        self.assertIn("left_valid_joints", out)
        self.assertIn("right_valid_joints", out)
        self.assertIn("total_valid_hand_joints", out)
        self.assertIn("hand_presence_ok", out)
        self.assertIn("start_blocked_by_hand_guard", out)
        self.assertEqual(int(out["total_valid_hand_joints"]), 0)

    def test_bio_segmenter_with_aux_head_emits_p_active(self) -> None:
        model = BioTagger(BioModelConfig(use_signness_head=True)).eval()
        segmenter = BioSegmenter.from_model(
            model,
            threshold=0.8,
            decoder_cfg=BioDecoderConfig(),
            device="cpu",
        )
        out = segmenter.step(
            np.zeros((42, 3), dtype=np.float32),
            np.ones((42, 1), dtype=np.float32),
            ts_ms=0.0,
        )
        self.assertIn("p_active", out)
        self.assertIsNotNone(out["p_active"])
        self.assertIn("signness_gate_ok", out)

    def test_bio_segmenter_with_onset_head_emits_p_onset(self) -> None:
        model = BioTagger(BioModelConfig(use_signness_head=True, use_onset_head=True)).eval()
        segmenter = BioSegmenter.from_model(
            model,
            threshold=0.8,
            decoder_cfg=BioDecoderConfig(use_onset_gate=True),
            device="cpu",
        )
        out = segmenter.step(
            np.zeros((42, 3), dtype=np.float32),
            np.ones((42, 1), dtype=np.float32),
            ts_ms=0.0,
        )
        self.assertIn("p_onset", out)
        self.assertIsNotNone(out["p_onset"])
        self.assertIn("clip_hit_candidate", out)

    def test_bio_forward_with_aux_returns_expected_shapes(self) -> None:
        model = BioTagger(BioModelConfig(use_signness_head=True)).eval()
        pts = torch.zeros((2, 5, 42, 3), dtype=torch.float32)
        mask = torch.ones((2, 5, 42, 1), dtype=torch.float32)
        logits, signness_logits, hN = model.forward_with_aux(pts, mask)
        self.assertEqual(tuple(logits.shape), (2, 5, 3))
        self.assertIsNotNone(signness_logits)
        self.assertEqual(tuple(signness_logits.shape), (2, 5, 1))
        self.assertEqual(int(hN.shape[1]), 2)

    def test_bio_forward_with_heads_returns_expected_shapes(self) -> None:
        model = BioTagger(BioModelConfig(use_signness_head=True, use_onset_head=True)).eval()
        pts = torch.zeros((2, 5, 42, 3), dtype=torch.float32)
        mask = torch.ones((2, 5, 42, 1), dtype=torch.float32)
        logits, signness_logits, onset_logits, hN = model.forward_with_heads(pts, mask)
        self.assertEqual(tuple(logits.shape), (2, 5, 3))
        self.assertIsNotNone(signness_logits)
        self.assertEqual(tuple(signness_logits.shape), (2, 5, 1))
        self.assertIsNotNone(onset_logits)
        self.assertEqual(tuple(onset_logits.shape), (2, 5, 1))
        self.assertEqual(int(hN.shape[1]), 2)

    def test_bio_forward_matches_streaming_logits(self) -> None:
        torch.manual_seed(0)
        cfg = BioModelConfig(conv_layers=2, conv_kernel=5)
        model = BioTagger(cfg).eval()
        pts = np.random.randn(8, 42, 3).astype(np.float32)
        mask = np.ones((8, 42, 1), dtype=np.float32)
        seq = canonicalize_sequence(pts, mask, np.arange(8, dtype=np.float32) * 33.0)
        segmenter = BioSegmenter.from_model(
            model,
            threshold=0.5,
            decoder_cfg=BioDecoderConfig(stream_window=32, cooldown_frames=0),
            device="cpu",
        )
        forward_logits = segmenter.forward_logits(seq)
        stream_logits = segmenter.stream_logits(seq)
        self.assertTrue(np.allclose(forward_logits, stream_logits, atol=1e-5))

    def test_bio_preprocess_v3_offline_matches_streaming(self) -> None:
        pts = np.zeros((5, 42, 3), dtype=np.float32)
        mask = np.zeros((5, 42, 1), dtype=np.float32)
        for t, base in enumerate((0.10, 0.15, None, None, 0.20)):
            if base is None:
                continue
            for j in range(21):
                pts[t, 21 + j, 0] = float(base + 0.01 * j)
                pts[t, 21 + j, 1] = float(0.2 + 0.005 * j)
                mask[t, 21 + j, 0] = 1.0
        cfg = BioPreprocessConfig(version=BIO_PREPROCESSING_VERSION_V3)
        offline, _debug = preprocess_sequence(pts, mask, cfg=cfg)
        state = init_bio_preprocess_state(dtype=torch.float32)
        rows = []
        for idx in range(pts.shape[0]):
            frame, state, _step_debug = preprocess_frame_v3(pts[idx], mask[idx], state, cfg=cfg)
            rows.append(frame.detach().cpu().numpy())
        streaming = np.stack(rows, axis=0)
        self.assertTrue(np.allclose(np.asarray(offline), streaming, atol=1e-6))

    def test_bio_preprocess_v3_no_hand_gap_keeps_state(self) -> None:
        pts = np.zeros((42, 3), dtype=np.float32)
        mask = np.zeros((42, 1), dtype=np.float32)
        for j in range(21):
            pts[21 + j, 0] = float(0.3 + 0.01 * j)
            pts[21 + j, 1] = float(0.4 + 0.02 * j)
            mask[21 + j, 0] = 1.0
        cfg = BioPreprocessConfig(version=BIO_PREPROCESSING_VERSION_V3)
        state = init_bio_preprocess_state(dtype=torch.float32)
        _frame0, state, debug0 = preprocess_frame_v3(pts, mask, state, cfg=cfg)
        center0 = state.center.clone()
        scale0 = state.scale.clone()
        no_hand_pts = np.zeros((42, 3), dtype=np.float32)
        no_hand_mask = np.zeros((42, 1), dtype=np.float32)
        frame1, state, debug1 = preprocess_frame_v3(no_hand_pts, no_hand_mask, state, cfg=cfg)
        self.assertFalse(debug1["updated_center"])
        self.assertFalse(debug1["updated_scale"])
        self.assertTrue(torch.allclose(center0, state.center))
        self.assertTrue(torch.allclose(scale0, state.scale))
        self.assertEqual(float(frame1.abs().sum().item()), 0.0)

    def test_prepare_model_pts_v3_matches_shared_preprocess(self) -> None:
        pts = np.zeros((4, 42, 3), dtype=np.float32)
        mask = np.zeros((4, 42, 1), dtype=np.float32)
        for t in range(4):
            for j in range(21):
                pts[t, 21 + j, 0] = float(0.2 + 0.01 * (t + j))
                pts[t, 21 + j, 1] = float(0.3 + 0.005 * j)
                mask[t, 21 + j, 0] = 1.0
        prelabel_cfg = PrelabelConfig(preprocessing_version=BIO_PREPROCESSING_VERSION_V3)
        prepared = prepare_model_pts(pts, mask, np.arange(4, dtype=np.float32), prelabel_cfg)
        shared, _debug = preprocess_sequence(pts, mask, cfg=BioPreprocessConfig(version=BIO_PREPROCESSING_VERSION_V3))
        self.assertTrue(np.allclose(prepared, np.asarray(shared), atol=1e-6))

    def test_msagcn_runtime_feature_builder_matches_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_dir, csv_path, frames, meta = _write_msagcn_fixture(Path(tmp))
            cfg = DSConfig(
                max_frames=4,
                temporal_crop="resample",
                use_streams=("joints", "bones", "velocity"),
                include_pose=False,
                center=False,
                normalize=False,
                augment=False,
                boundary_jitter_prob=0.0,
                speed_perturb_prob=0.0,
            )
            ds = MultiStreamGestureDataset(json_dir, csv_path, split="train", cfg=cfg)
            item = ds[0]
            seq = frames_to_canonical_sequence(frames, meta=meta)
            builder = MSAGCNClassifier.from_model(
                MultiStreamAGCN(num_classes=1, V=42, A=ds.build_adjacency(normalize=False), streams=cfg.use_streams),
                label2idx={"hello": 0},
                ds_cfg=cfg,
                device="cpu",
            ).builder
            built = builder.prepare_sequence(seq)
            for key in ("joints", "bones", "velocity", "mask"):
                self.assertTrue(torch.allclose(item[key], built[key][0], atol=1e-5), key)

    def test_msagcn_runtime_feature_builder_matches_dataset_with_pose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_dir, csv_path, frames, meta = _write_msagcn_fixture(Path(tmp), include_pose=True)
            cfg = DSConfig(
                max_frames=4,
                temporal_crop="resample",
                use_streams=("joints", "bones", "velocity"),
                include_pose=True,
                connect_cross_edges=True,
                center=False,
                normalize=False,
                augment=False,
                boundary_jitter_prob=0.0,
                speed_perturb_prob=0.0,
            )
            ds = MultiStreamGestureDataset(json_dir, csv_path, split="train", cfg=cfg)
            item = ds[0]
            seq = frames_to_canonical_sequence(frames, meta=meta)
            builder = MSAGCNClassifier.from_model(
                MultiStreamAGCN(num_classes=1, V=ds.V, A=ds.build_adjacency(normalize=False), streams=cfg.use_streams),
                label2idx={"hello": 0},
                ds_cfg=cfg,
                device="cpu",
            ).builder
            built = builder.prepare_sequence(seq)
            for key in ("joints", "bones", "velocity", "mask"):
                self.assertTrue(torch.allclose(item[key], built[key][0], atol=1e-5), key)

    def test_runtime_bundle_export_and_load_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bio_ckpt = root / "bio.pt"
            bio_cfg = BioModelConfig()
            bio_model = BioTagger(bio_cfg)
            torch.save(
                {
                    "model_state": bio_model.state_dict(),
                    "ema_state": bio_model.state_dict(),
                    "cfg": asdict(bio_cfg),
                    "best_balanced_metrics": {"selection_threshold": 0.8},
                    "best_boundary_metrics": {"selection_threshold": 0.75},
                },
                bio_ckpt,
            )
            bio_manifest = export_bio_runtime_bundle(bio_ckpt, root / "bio_bundle")
            bio_seg = BioSegmenter.from_bundle(bio_manifest.parent, device="cpu")
            self.assertAlmostEqual(bio_seg.threshold, 0.8, places=6)

            ds_cfg = DSConfig(max_frames=4, temporal_crop="resample", use_streams=("joints", "bones", "velocity"), include_pose=False)
            builder = MSAGCNClassifier.from_model(
                MultiStreamAGCN(num_classes=2, V=42, A=torch.zeros((42, 42)), streams=ds_cfg.use_streams),
                label2idx={"hello": 0, "bye": 1},
                ds_cfg=ds_cfg,
                device="cpu",
            )
            ms_ckpt = root / "best.ckpt"
            torch.save(
                {
                    "model_state": builder.model.state_dict(),
                    "ema_state": builder.model.state_dict(),
                    "label2idx": {"hello": 0, "bye": 1},
                    "ds_cfg": asdict(ds_cfg),
                    "args": {},
                },
                ms_ckpt,
            )
            ms_manifest = export_msagcn_runtime_bundle(ms_ckpt, root / "ms_bundle")
            clf = MSAGCNClassifier.from_bundle(ms_manifest.parent, device="cpu")
            self.assertEqual(clf.idx2label[0], "hello")

    def test_bio_bundle_manifest_threshold_is_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bio_ckpt = root / "bio.pt"
            bio_cfg = BioModelConfig()
            bio_model = BioTagger(bio_cfg)
            torch.save(
                {
                    "model_state": bio_model.state_dict(),
                    "ema_state": bio_model.state_dict(),
                    "cfg": asdict(bio_cfg),
                    "best_balanced_metrics": {"selection_threshold": 0.2},
                },
                bio_ckpt,
            )
            bio_manifest = export_bio_runtime_bundle(bio_ckpt, root / "bio_bundle")
            payload = json.loads(bio_manifest.read_text(encoding="utf-8"))
            payload["threshold"] = 0.9
            payload["decoder_config"]["min_segment_frames"] = 7
            bio_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            bio_seg = BioSegmenter.from_bundle(bio_manifest.parent, device="cpu")
            self.assertAlmostEqual(bio_seg.threshold, 0.9, places=6)
            self.assertEqual(bio_seg.decoder.cfg.min_segment_frames, 7)

    def test_bio_decoder_flush_respects_emit_partial_segments(self) -> None:
        decoder = BioSegmentDecoder(BioDecoderConfig(emit_partial_segments=False, min_segment_frames=2))
        decoder._open_segment(0, 0.0, 0.9, 0.9)
        self.assertEqual(decoder.flush(force=False), [])

    def test_msagcn_bundle_and_raw_checkpoint_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds_cfg = DSConfig(max_frames=4, temporal_crop="resample", use_streams=("joints", "bones", "velocity"), include_pose=False)
            model = MultiStreamAGCN(num_classes=2, V=42, A=torch.zeros((42, 42)), streams=ds_cfg.use_streams)
            ckpt = root / "best.ckpt"
            payload = {
                "model_state": model.state_dict(),
                "ema_state": model.state_dict(),
                "label2idx": {"hello": 0, "bye": 1},
                "ds_cfg": asdict(ds_cfg),
                "args": {},
            }
            torch.save(payload, ckpt)
            bundle = export_msagcn_runtime_bundle(ckpt, root / "bundle")
            seq = canonicalize_sequence(
                np.random.randn(4, 42, 3).astype(np.float32),
                np.ones((4, 42, 1), dtype=np.float32),
                np.arange(4, dtype=np.float32) * 33.0,
            )
            raw = MSAGCNClassifier.from_checkpoint(ckpt, device="cpu")
            bundled = MSAGCNClassifier.from_bundle(bundle.parent, device="cpu")
            raw_pred = raw.predict_sequence(seq)
            bundle_pred = bundled.predict_sequence(seq)
            self.assertEqual(raw_pred.class_id, bundle_pred.class_id)
            self.assertTrue(np.allclose(np.asarray(raw_pred.probs), np.asarray(bundle_pred.probs), atol=1e-6))

    def test_msagcn_family_head_reconstruction_infers_num_families(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds_cfg = DSConfig(max_frames=4, temporal_crop="resample", use_streams=("joints", "bones", "velocity"), include_pose=False)
            model = MultiStreamAGCN(
                num_classes=2,
                V=42,
                A=torch.zeros((42, 42)),
                streams=ds_cfg.use_streams,
                use_family_head=True,
                num_families=3,
            )
            ckpt = root / "best.ckpt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "ema_state": model.state_dict(),
                    "label2idx": {"hello": 0, "bye": 1},
                    "ds_cfg": asdict(ds_cfg),
                    "args": {"use_family_head": True, "num_families": 0},
                },
                ckpt,
            )
            clf = MSAGCNClassifier.from_checkpoint(ckpt, device="cpu")
            self.assertTrue(clf.model.use_family_head)
            self.assertEqual(clf.model.num_families, 3)

    def test_msagcn_prediction_exposes_family_outputs(self) -> None:
        ds_cfg = DSConfig(max_frames=4, temporal_crop="resample", use_streams=("joints", "bones", "velocity"), include_pose=False)
        model = MultiStreamAGCN(
            num_classes=2,
            V=42,
            A=torch.zeros((42, 42)),
            streams=ds_cfg.use_streams,
            use_family_head=True,
            num_families=3,
        )
        clf = MSAGCNClassifier.from_model(model, label2idx={"hello": 0, "bye": 1}, ds_cfg=ds_cfg, device="cpu")
        seq = canonicalize_sequence(
            np.random.randn(4, 42, 3).astype(np.float32),
            np.ones((4, 42, 1), dtype=np.float32),
            np.arange(4, dtype=np.float32) * 33.0,
        )
        pred = clf.predict_sequence(seq)
        self.assertIsNotNone(pred.family_probs)
        self.assertEqual(len(pred.family_probs or []), 3)
        self.assertIsNotNone(pred.family_topk)

    def test_pipeline_waits_for_post_context_before_classifying(self) -> None:
        class _Bio:
            def __init__(self) -> None:
                self.threshold = 0.8
                self.metadata = {"config_resolution_source": "checkpoint", "decoder_version": "fake", "preprocessing_version": "test"}
                self.decoder = _FakeDecoder()
                self.idx = 0

            def step(self, pts, mask, *, ts_ms: float):
                del pts, mask, ts_ms
                cur = self.idx
                self.idx += 1
                segments = []
                if cur == 1:
                    segments.append(
                        {
                            "segment_id": 1,
                            "start_frame": 0,
                            "end_frame_exclusive": 2,
                            "start_time_ms": 0.0,
                            "end_time_ms": 33.0,
                            "boundary_score": 0.9,
                            "mean_inside_score": 0.8,
                            "threshold_used": 0.8,
                            "decoder_version": "fake",
                        }
                    )
                return {"segments": segments}

        pipeline = InferencePipeline(
            _Bio(),
            _FakeMSAGCNClassifier(),
            async_classification=False,
        )
        pipeline.bridge.post_context_frames = 2
        frame = np.zeros((42, 3), dtype=np.float32)
        mask = np.ones((42, 1), dtype=np.float32)
        out1 = pipeline.process_frame(pts=frame, mask=mask, ts_ms=0.0)
        out2 = pipeline.process_frame(pts=frame, mask=mask, ts_ms=33.0)
        out3 = pipeline.process_frame(pts=frame, mask=mask, ts_ms=66.0)
        out4 = pipeline.process_frame(pts=frame, mask=mask, ts_ms=99.0)
        self.assertEqual(len(out1["completed_predictions"]), 0)
        self.assertEqual(len(out2["completed_predictions"]), 0)
        self.assertEqual(len(out3["completed_predictions"]), 0)
        self.assertEqual(len(out4["completed_predictions"]), 1)

    def test_bridge_trim_protection_respects_pending_segment(self) -> None:
        class _Bio:
            def __init__(self) -> None:
                self.threshold = 0.8
                self.metadata = {"config_resolution_source": "checkpoint", "decoder_version": "fake", "preprocessing_version": "test"}
                self.decoder = _FakeDecoder()
                self.idx = 0

            def step(self, pts, mask, *, ts_ms: float):
                del pts, mask, ts_ms
                cur = self.idx
                self.idx += 1
                if cur == 0:
                    return {
                        "segments": [
                            {
                                "segment_id": 1,
                                "start_frame": 0,
                                "end_frame_exclusive": 1,
                                "start_time_ms": 0.0,
                                "end_time_ms": 0.0,
                                "boundary_score": 0.9,
                                "mean_inside_score": 0.8,
                                "threshold_used": 0.8,
                                "decoder_version": "fake",
                                "end_reason": "gap_closed",
                            }
                        ]
                    }
                return {"segments": []}

        pipeline = InferencePipeline(
            _Bio(),
            _FakeMSAGCNClassifier(),
            async_classification=False,
        )
        pipeline.bridge.max_buffer_frames = 4
        pipeline.bridge.post_context_frames = 10
        frame = np.zeros((42, 3), dtype=np.float32)
        mask = np.ones((42, 1), dtype=np.float32)
        for idx in range(6):
            pipeline.process_frame(pts=frame, mask=mask, ts_ms=float(idx * 33.0))
        self.assertEqual(pipeline.bridge.protected_from_global, 0)
        self.assertEqual(pipeline.bridge._base_index, 0)

    def test_console_json_output_is_ascii_safe(self) -> None:
        payload = {"sentence": "привет", "segments": 1, "predictions": 1}
        with mock.patch("sys.stdout", new=io.StringIO()) as buf:
            pipeline_cli._emit_result(payload, console_format="json")
            self.assertIn("\\u043f", buf.getvalue())
        pred_payload = {
            "label": "слово",
            "confidence": 0.9,
            "input": "x",
            "family_label": "семья",
            "family_confidence": 0.7,
        }
        with mock.patch("sys.stdout", new=io.StringIO()) as buf:
            msagcn_cli._emit_prediction(pred_payload, console_format="json")
            self.assertIn("\\u0441", buf.getvalue())
        with mock.patch("sys.stdout", new=io.StringIO()) as buf:
            bio_runtime_commands._emit_payload({"segments": [], "threshold": 0.8}, console_format="json")
            self.assertIn("\"threshold\"", buf.getvalue())

    def test_offline_video_pipeline_writes_sentence_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq = canonicalize_sequence(
                np.zeros((6, 42, 3), dtype=np.float32),
                np.ones((6, 42, 1), dtype=np.float32),
                np.arange(6, dtype=np.float32) * 33.3333,
                meta={"fps": 30.0},
            )
            frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(6)]
            cfg = InferencePipelineConfig(bio_bundle="bio_bundle", msagcn_bundle="ms_bundle")
            out_dir = root / "offline_out"
            with mock.patch("pipeline.app.extract_video_sequence", return_value=(seq, frames)):
                with mock.patch("pipeline.app.BioSegmenter.from_bundle", return_value=_FakeBioSegmenter()):
                    with mock.patch("pipeline.app.MSAGCNClassifier.from_bundle", return_value=_FakeMSAGCNClassifier()):
                        result = run_video_pipeline("fake.mp4", cfg=cfg, out_dir=out_dir, write_preview=True)
            self.assertEqual(result["sentence_builder"]["sentence"], "hello")
            self.assertTrue((out_dir / "segments.json").exists())
            self.assertTrue((out_dir / "predictions.json").exists())
            self.assertTrue((out_dir / "sentence.txt").exists())
            self.assertEqual((out_dir / "sentence.txt").read_text(encoding="utf-8"), "hello")

    def test_build_review_session_writes_bundle_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq = canonicalize_sequence(
                np.zeros((6, 42, 3), dtype=np.float32),
                np.ones((6, 42, 1), dtype=np.float32),
                np.arange(6, dtype=np.float32) * 33.3333,
                meta={"fps": 30.0, "extractor_mode": "hands_only", "extractor": "test"},
            )
            frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(6)]
            cfg = InferencePipelineConfig(bio_bundle="bio_bundle", msagcn_bundle="ms_bundle")
            out_dir = root / "review_out"
            with mock.patch("pipeline.app.extract_video_sequence", return_value=(seq, frames)):
                with mock.patch("pipeline.app.BioSegmenter.from_bundle", return_value=_FakeBioSegmenter()):
                    with mock.patch("pipeline.app.MSAGCNClassifier.from_bundle", return_value=_FakeMSAGCNClassifier()):
                        result = build_review_session("fake.mp4", cfg=cfg, out_dir=out_dir)
            self.assertTrue((out_dir / "session.json").exists())
            self.assertTrue((out_dir / "canonical_sequence.npz").exists())
            self.assertTrue((out_dir / "frame_debug.jsonl").exists())
            self.assertTrue((out_dir / "timeline_tracks.json").exists())
            self.assertIn("session_path", result)

    def test_review_session_loader_reads_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq = canonicalize_sequence(
                np.zeros((6, 42, 3), dtype=np.float32),
                np.ones((6, 42, 1), dtype=np.float32),
                np.arange(6, dtype=np.float32) * 33.3333,
                meta={"fps": 30.0, "extractor_mode": "hands_only", "extractor": "test"},
            )
            frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(6)]
            cfg = InferencePipelineConfig(bio_bundle="bio_bundle", msagcn_bundle="ms_bundle")
            out_dir = root / "review_out"
            with mock.patch("pipeline.app.extract_video_sequence", return_value=(seq, frames)):
                with mock.patch("pipeline.app.BioSegmenter.from_bundle", return_value=_FakeBioSegmenter()):
                    with mock.patch("pipeline.app.MSAGCNClassifier.from_bundle", return_value=_FakeMSAGCNClassifier()):
                        build_review_session("fake.mp4", cfg=cfg, out_dir=out_dir)
            session = load_review_session(out_dir)
            self.assertEqual(session.frame_count, 6)
            self.assertEqual(session.sentence, "hello")
            self.assertEqual(len(session.frame_rows), 6)
            self.assertGreaterEqual(len(session.combined_segment_rows()), 1)

    def test_review_session_summary_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq = canonicalize_sequence(
                np.zeros((6, 42, 3), dtype=np.float32),
                np.ones((6, 42, 1), dtype=np.float32),
                np.arange(6, dtype=np.float32) * 33.3333,
                meta={"fps": 30.0, "extractor_mode": "hands_only", "extractor": "test"},
            )
            frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(6)]
            cfg = InferencePipelineConfig(bio_bundle="bio_bundle", msagcn_bundle="ms_bundle")
            out_dir = root / "review_out"
            with mock.patch("pipeline.app.extract_video_sequence", return_value=(seq, frames)):
                with mock.patch("pipeline.app.BioSegmenter.from_bundle", return_value=_FakeBioSegmenter()):
                    with mock.patch("pipeline.app.MSAGCNClassifier.from_bundle", return_value=_FakeMSAGCNClassifier()):
                        build_review_session("fake.mp4", cfg=cfg, out_dir=out_dir)
            session = load_review_session(out_dir)
            self.assertEqual(session.session_name, "fake.mp4")
            self.assertEqual(session.accepted_segment_count, 1)
            self.assertEqual(session.rejected_segment_count, 0)
            self.assertGreater(session.left_hand_present_ratio, 0.0)
            self.assertEqual(session.right_hand_present_ratio, 1.0)

    def test_desktop_review_window_loads_session_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq = canonicalize_sequence(
                np.zeros((6, 42, 3), dtype=np.float32),
                np.ones((6, 42, 1), dtype=np.float32),
                np.arange(6, dtype=np.float32) * 33.3333,
                meta={"fps": 30.0, "extractor_mode": "hands_only", "extractor": "test"},
            )
            frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(6)]
            cfg = InferencePipelineConfig(bio_bundle="bio_bundle", msagcn_bundle="ms_bundle")
            out_dir = root / "review_out"
            with mock.patch("pipeline.app.extract_video_sequence", return_value=(seq, frames)):
                with mock.patch("pipeline.app.BioSegmenter.from_bundle", return_value=_FakeBioSegmenter()):
                    with mock.patch("pipeline.app.MSAGCNClassifier.from_bundle", return_value=_FakeMSAGCNClassifier()):
                        build_review_session("fake.mp4", cfg=cfg, out_dir=out_dir)
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            from PySide6.QtWidgets import QApplication
            from desktop_review.qt_app import ReviewWindow

            app = QApplication.instance() or QApplication([])
            window = ReviewWindow()
            try:
                window._load_session(out_dir)
                self.assertEqual(window.segment_table.rowCount(), 1)
                self.assertIn("fake.mp4", window.session_title_label.text())
                self.assertIn("hello", window.sentence_value_label.text())
                window.segment_filter_combo.setCurrentText("Accepted")
                self.assertEqual(window.segment_table.rowCount(), 1)
                window.view_mode_combo.setCurrentText("Overlay")
                self.assertTrue(window.original_group.isHidden())
                self.assertFalse(window.overlay_group.isHidden())
            finally:
                window.close()

    def test_actual_target_checkpoints_load_when_present(self) -> None:
        bio_ckpt = Path("outputs/runs/bio_v2_run/best_balanced_model.pt")
        ms_ckpt = Path("outputs/families/agcn_supcon_cb_oof/family_finetune/best_rebalance.ckpt")
        if not bio_ckpt.exists() or not ms_ckpt.exists():
            self.skipTest("Target deployment checkpoints are not available in this workspace")
        bio = BioSegmenter.from_checkpoint(bio_ckpt, device="cpu")
        self.assertGreater(bio.threshold, 0.0)
        ms = MSAGCNClassifier.from_checkpoint(ms_ckpt, device="cpu")
        self.assertTrue(ms.ds_cfg.include_pose)
        seq = canonicalize_sequence(
            np.random.randn(8, 42, 3).astype(np.float32),
            np.ones((8, 42, 1), dtype=np.float32),
            np.arange(8, dtype=np.float32) * 33.0,
            pose_xyz=np.random.randn(8, 33, 3).astype(np.float32),
            pose_vis=np.ones((8, 33), dtype=np.float32),
            pose_indices=list(range(33)),
            meta={"fps": 30.0, "coords": "image"},
        )
        pred = ms.predict_sequence(seq)
        self.assertIsNotNone(pred.family_probs)
        self.assertEqual(len(pred.family_probs or []), int(ms.model.num_families))
