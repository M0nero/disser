from __future__ import annotations

import csv
import json
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from bio.runtime import BIO_RUNTIME_PREPROCESSING_VERSION, BioDecoderConfig, BioSegmenter
from msagcn.data.topology import HAND_EDGES_ONE, POSE_EDGE_PAIRS_ABS, hand_edges_42
from msagcn.runtime import MSAGCN_RUNTIME_PREPROCESSING_VERSION, MSAGCNClassifier
from runtime.bridge import SegmentBridge, SegmentClip
from runtime.manifest import load_runtime_manifest
from runtime.mediapipe_hands import (
    MediaPipeHandsConfig,
    MediaPipeHandsTracker,
    MediaPipeHolisticConfig,
    MediaPipeHolisticTracker,
    extract_video_sequence,
)
from runtime.sentence import SentenceBuilder, SentenceBuilderConfig
from runtime.skeleton import CanonicalSkeletonSequence, save_skeleton_sequence_npz


@dataclass
class InferencePipelineConfig:
    bio_bundle: str = ""
    bio_checkpoint: str = ""
    bio_selection: str = "best_balanced"
    bio_decoder_config_json: str = ""
    bio_threshold: float | None = None
    msagcn_bundle: str = ""
    msagcn_checkpoint: str = ""
    msagcn_label_map: str = ""
    msagcn_ds_config: str = ""
    device: str = ""
    pre_context_frames: int = 4
    post_context_frames: int = 4
    max_buffer_frames: int = 4096
    extractor_mode: str = "auto"
    max_pending_segments: int = 8
    classifier_worker_count: int = 1
    drop_or_block_policy: str = "block_on_classifier_queue"
    sentence_min_confidence: float = 0.5
    sentence_duplicate_gap_ms: float = 1200.0


def _load_decoder_cfg_json(path: str | Path | None) -> BioDecoderConfig | None:
    if not path:
        return None
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"BIO decoder config must be a JSON object: {source}")
    return BioDecoderConfig(**{k: v for k, v in payload.items() if k in BioDecoderConfig.__dataclass_fields__})


def _build_holistic_cfg(cfg: MediaPipeHandsConfig | None) -> MediaPipeHolisticConfig:
    if cfg is None:
        return MediaPipeHolisticConfig()
    return MediaPipeHolisticConfig(
        static_image_mode=bool(cfg.static_image_mode),
        model_complexity=int(cfg.model_complexity),
        min_detection_confidence=float(cfg.min_detection_confidence),
        min_tracking_confidence=float(cfg.min_tracking_confidence),
    )


@dataclass
class _PendingSegment:
    segment: Dict[str, Any]
    ready_after_frame: int


@dataclass
class _InflightPrediction:
    future: Future[Dict[str, Any]]
    protected_from_global: int


class InferencePipeline:
    def __init__(
        self,
        bio_segmenter: BioSegmenter,
        classifier: MSAGCNClassifier,
        *,
        bridge: Optional[SegmentBridge] = None,
        sentence_builder: Optional[SentenceBuilder] = None,
        async_classification: bool = False,
        max_pending_segments: int = 8,
        classifier_worker_count: int = 1,
        drop_or_block_policy: str = "block_on_classifier_queue",
        extractor_mode: str = "auto",
    ) -> None:
        self.bio = bio_segmenter
        self.classifier = classifier
        self.bridge = bridge or SegmentBridge()
        self.sentence_builder = sentence_builder or SentenceBuilder()
        self.async_classification = bool(async_classification)
        self.max_pending_segments = max(1, int(max_pending_segments))
        self.classifier_worker_count = max(1, int(classifier_worker_count))
        self.drop_or_block_policy = str(drop_or_block_policy or "block_on_classifier_queue")
        self.extractor_mode = str(extractor_mode or "auto")
        self._executor = (
            ThreadPoolExecutor(max_workers=self.classifier_worker_count, thread_name_prefix="segment_classifier")
            if self.async_classification
            else None
        )
        self._pending_segments: List[_PendingSegment] = []
        self._inflight: List[_InflightPrediction] = []
        self.segments: List[Dict[str, Any]] = []
        self.predictions: List[Dict[str, Any]] = []
        self.runtime_summary = {
            "bio_config_source": str(self.bio.metadata.get("config_resolution_source", "checkpoint")),
            "msagcn_config_source": str(self.classifier.metadata.get("config_resolution_source", "checkpoint")),
            "pose_enabled": bool(self.classifier.ds_cfg.include_pose),
            "decoder_version": str(self.bio.metadata.get("decoder_version", "bio_segment_decoder_v1")),
            "selected_threshold": float(self.bio.threshold),
            "preprocessing_version": {
                "bio": str(self.bio.metadata.get("preprocessing_version", BIO_RUNTIME_PREPROCESSING_VERSION)),
                "msagcn": str(self.classifier.metadata.get("preprocessing_version", MSAGCN_RUNTIME_PREPROCESSING_VERSION)),
            },
            "extractor_mode": self.extractor_mode,
            "pending_segments": 0,
            "inflight_jobs": 0,
            "max_queue_depth_seen": 0,
            "frames_protected_from_trim": 0,
            "drop_or_block_policy": self.drop_or_block_policy,
            "classifier_worker_count": self.classifier_worker_count,
        }

    @classmethod
    def from_config(cls, cfg: InferencePipelineConfig, *, async_classification: bool = False) -> "InferencePipeline":
        bio = cls._load_bio(cfg)
        msagcn = cls._load_msagcn(cfg)
        bridge = SegmentBridge(
            pre_context_frames=int(cfg.pre_context_frames),
            post_context_frames=int(cfg.post_context_frames),
            max_buffer_frames=int(cfg.max_buffer_frames),
        )
        sentence = SentenceBuilder(
            SentenceBuilderConfig(
                min_confidence=float(cfg.sentence_min_confidence),
                duplicate_gap_ms=float(cfg.sentence_duplicate_gap_ms),
            )
        )
        return cls(
            bio,
            msagcn,
            bridge=bridge,
            sentence_builder=sentence,
            async_classification=async_classification,
            max_pending_segments=int(cfg.max_pending_segments),
            classifier_worker_count=int(cfg.classifier_worker_count),
            drop_or_block_policy=str(cfg.drop_or_block_policy or "block_on_classifier_queue"),
            extractor_mode=str(cfg.extractor_mode or "auto"),
        )

    @staticmethod
    def _load_bio(cfg: InferencePipelineConfig) -> BioSegmenter:
        device = cfg.device or None
        if cfg.bio_bundle:
            return BioSegmenter.from_bundle(cfg.bio_bundle, device=device)
        if not cfg.bio_checkpoint:
            raise ValueError("Provide either bio_bundle or bio_checkpoint")
        decoder_cfg = _load_decoder_cfg_json(cfg.bio_decoder_config_json)
        return BioSegmenter.from_checkpoint(
            cfg.bio_checkpoint,
            selection=str(cfg.bio_selection or "best_balanced"),
            device=device,
            decoder_cfg=decoder_cfg,
            threshold=cfg.bio_threshold,
        )

    @staticmethod
    def _load_msagcn(cfg: InferencePipelineConfig) -> MSAGCNClassifier:
        device = cfg.device or None
        if cfg.msagcn_bundle:
            return MSAGCNClassifier.from_bundle(cfg.msagcn_bundle, device=device)
        if not cfg.msagcn_checkpoint:
            raise ValueError("Provide either msagcn_bundle or msagcn_checkpoint")
        return MSAGCNClassifier.from_checkpoint(
            cfg.msagcn_checkpoint,
            device=device,
            label_map_path=(cfg.msagcn_label_map or None),
            ds_config_path=(cfg.msagcn_ds_config or None),
        )

    @property
    def requires_pose(self) -> bool:
        return bool(self.classifier.ds_cfg.include_pose)

    def _segment_protected_from_global(self, segment: Dict[str, Any]) -> int:
        return max(0, int(segment.get("start_frame", 0)) - int(self.bridge.pre_context_frames))

    def _update_trim_protection(self) -> None:
        desired_keep = max(0, int(self.bridge.total_frames) - int(self.bridge.max_buffer_frames))
        protected_candidates = [self._segment_protected_from_global(item.segment) for item in self._pending_segments]
        protected_candidates.extend(int(job.protected_from_global) for job in self._inflight)
        protected_from = min(protected_candidates) if protected_candidates else None
        self.bridge.set_protected_from_global(protected_from)
        if protected_from is None:
            protected_frames = 0
        else:
            protected_frames = max(0, desired_keep - int(protected_from))
        self.runtime_summary["pending_segments"] = int(len(self._pending_segments))
        self.runtime_summary["inflight_jobs"] = int(len(self._inflight))
        self.runtime_summary["max_queue_depth_seen"] = max(
            int(self.runtime_summary.get("max_queue_depth_seen", 0)),
            int(len(self._pending_segments) + len(self._inflight)),
        )
        self.runtime_summary["frames_protected_from_trim"] = int(protected_frames)

    def _predict_clip(self, segment: Dict[str, Any], clip: SegmentClip) -> Dict[str, Any]:
        pred = self.classifier.predict_sequence(clip.as_sequence()).to_dict()
        return {
            "segment": dict(segment),
            "clip": {
                "start_frame": int(clip.start_frame),
                "end_frame_exclusive": int(clip.end_frame_exclusive),
                "start_time_ms": float(clip.start_time_ms),
                "end_time_ms": float(clip.end_time_ms),
            },
            "prediction": pred,
        }

    def _consume_prediction(self, row: Dict[str, Any]) -> Dict[str, Any]:
        seg = dict(row.get("segment", {}) or {})
        pred = dict(row.get("prediction", {}) or {})
        accepted = self.sentence_builder.add_prediction(
            segment_id=int(seg.get("segment_id", 0)),
            label=str(pred.get("label", "")),
            confidence=float(pred.get("confidence", 0.0)),
            start_time_ms=float(seg.get("start_time_ms", 0.0)),
            end_time_ms=float(seg.get("end_time_ms", 0.0)),
            meta={
                "segment": dict(seg),
                "classifier": {
                    "class_id": int(pred.get("class_id", -1)),
                    "top1_confidence": float(pred.get("confidence", 0.0)),
                },
            },
        )
        full_row = {
            **row,
            "sentence_decision": accepted.to_dict() if hasattr(accepted, "to_dict") else asdict(accepted),
        }
        self.segments.append(dict(seg))
        self.predictions.append(full_row)
        return full_row

    def _collect_completed(self, *, block: bool = False) -> List[Dict[str, Any]]:
        if not self._inflight:
            return []
        completed: List[Dict[str, Any]] = []
        remaining: List[_InflightPrediction] = []
        for job in self._inflight:
            future = job.future
            if block:
                row = future.result()
                completed.append(self._consume_prediction(row))
            elif future.done():
                row = future.result()
                completed.append(self._consume_prediction(row))
            else:
                remaining.append(job)
        self._inflight = remaining if not block else []
        self._update_trim_protection()
        return completed

    def _wait_for_one_inflight(self) -> List[Dict[str, Any]]:
        if not self._inflight:
            return []
        job = self._inflight.pop(0)
        row = job.future.result()
        completed = [self._consume_prediction(row)]
        self._update_trim_protection()
        return completed

    def _enqueue_segments(self, segments: List[Dict[str, Any]]) -> None:
        for seg in segments:
            ready_after = int(seg.get("end_frame_exclusive", 0)) + int(self.bridge.post_context_frames)
            self._pending_segments.append(_PendingSegment(segment=dict(seg), ready_after_frame=ready_after))
        self._update_trim_protection()

    def _drain_pending(self, *, end_of_stream: bool = False) -> List[Dict[str, Any]]:
        completed: List[Dict[str, Any]] = []
        pending: List[_PendingSegment] = []
        current_frames = int(self.bridge.total_frames)
        for item in self._pending_segments:
            if (not end_of_stream) and current_frames < int(item.ready_after_frame):
                pending.append(item)
                continue
            clip = self.bridge.clip_for_segment(item.segment)
            if self._executor is not None:
                while len(self._inflight) >= self.max_pending_segments:
                    if self.drop_or_block_policy == "drop_segment":
                        self.segments.append(dict(item.segment) | {"dropped": True, "drop_reason": "classifier_queue_full"})
                        break
                    completed.extend(self._wait_for_one_inflight())
                else:
                    self._inflight.append(
                        _InflightPrediction(
                            future=self._executor.submit(self._predict_clip, item.segment, clip),
                            protected_from_global=int(clip.start_frame),
                        )
                    )
            else:
                completed.append(self._consume_prediction(self._predict_clip(item.segment, clip)))
        self._pending_segments = pending
        self._update_trim_protection()
        completed.extend(self._collect_completed(block=False))
        return completed

    def process_frame(self, *, pts, mask, ts_ms: float, pose_xyz=None, pose_vis=None, pose_indices=None) -> Dict[str, Any]:
        self.bridge.append_frame(pts, mask, ts_ms, pose_xyz=pose_xyz, pose_vis=pose_vis, pose_indices=pose_indices)
        out = self.bio.step(pts, mask, ts_ms=ts_ms)
        if out["segments"]:
            self._enqueue_segments(out["segments"])
        completed = self._drain_pending(end_of_stream=False)
        self._update_trim_protection()
        return {
            "bio": out,
            "completed_predictions": completed,
            "sentence": self.sentence_builder.sentence,
            "runtime_summary": dict(self.runtime_summary),
        }

    def finalize(self, *, eos_policy: str = "drop_partial_on_eos") -> Dict[str, Any]:
        flush_segments = [seg.to_dict() for seg in self.bio.decoder.flush(force=False, eos_policy=eos_policy)]
        if flush_segments:
            self._enqueue_segments(flush_segments)
        completed = self._drain_pending(end_of_stream=True)
        completed.extend(self._collect_completed(block=True))
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
        self._update_trim_protection()
        return {
            "segments": list(self.segments),
            "predictions": list(self.predictions),
            "sentence_builder": self.sentence_builder.to_dict(),
            "completed_predictions": completed,
            "runtime_summary": dict(self.runtime_summary),
        }

    def run_sequence(self, seq: CanonicalSkeletonSequence, *, eos_policy: str = "close_open_segment_on_eos") -> Dict[str, Any]:
        for idx in range(seq.length):
            self.process_frame(
                pts=seq.pts[idx],
                mask=seq.mask[idx],
                ts_ms=float(seq.ts_ms[idx]),
                pose_xyz=(seq.pose_xyz[idx] if seq.pose_xyz is not None else None),
                pose_vis=(seq.pose_vis[idx] if seq.pose_vis is not None else None),
                pose_indices=seq.pose_indices,
            )
        final = self.finalize(eos_policy=eos_policy)
        final["sequence_meta"] = dict(seq.meta)
        return final


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_timeline_csv(path: Path, predictions: List[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "segment_id",
                "label",
                "confidence",
                "start_time_ms",
                "end_time_ms",
                "start_frame",
                "end_frame_exclusive",
            ],
        )
        writer.writeheader()
        for row in predictions:
            seg = dict(row.get("segment", {}) or {})
            pred = dict(row.get("prediction", {}) or {})
            writer.writerow(
                {
                    "segment_id": int(seg.get("segment_id", 0)),
                    "label": str(pred.get("label", "")),
                    "confidence": float(pred.get("confidence", 0.0)),
                    "start_time_ms": float(seg.get("start_time_ms", 0.0)),
                    "end_time_ms": float(seg.get("end_time_ms", 0.0)),
                    "start_frame": int(seg.get("start_frame", 0)),
                    "end_frame_exclusive": int(seg.get("end_frame_exclusive", 0)),
                }
            )
    return path


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _bio_label_name(label_id: int) -> str:
    labels = ("O", "B", "I")
    if 0 <= int(label_id) < len(labels):
        return labels[int(label_id)]
    return str(int(label_id))


def _segment_prediction_maps(predictions: List[Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    prediction_by_segment: Dict[int, Dict[str, Any]] = {}
    decision_by_segment: Dict[int, Dict[str, Any]] = {}
    for row in predictions:
        seg = dict(row.get("segment", {}) or {})
        seg_id = int(seg.get("segment_id", -1))
        if seg_id < 0:
            continue
        prediction_by_segment[seg_id] = dict(row.get("prediction", {}) or {})
        decision_by_segment[seg_id] = dict(row.get("sentence_decision", {}) or {})
    return prediction_by_segment, decision_by_segment


def _frame_segment_ids(frame_count: int, segments: List[Dict[str, Any]]) -> List[int | None]:
    out: List[int | None] = [None] * int(frame_count)
    for seg in segments:
        seg_id = int(seg.get("segment_id", -1))
        if seg_id < 0:
            continue
        start = max(0, int(seg.get("start_frame", 0)))
        end_excl = min(int(frame_count), max(start, int(seg.get("end_frame_exclusive", start))))
        for frame_idx in range(start, end_excl):
            out[frame_idx] = seg_id
    return out


def _frame_warning_flags(*, left_valid_frac: float, right_valid_frac: float, pose_valid_frac: float, pose_enabled: bool) -> List[str]:
    flags: List[str] = []
    if left_valid_frac <= 0.05:
        flags.append("left_hand_missing")
    if right_valid_frac <= 0.05:
        flags.append("right_hand_missing")
    if pose_enabled and pose_valid_frac <= 0.25:
        flags.append("pose_unreliable")
    return flags


def _build_frame_debug_rows(
    seq: CanonicalSkeletonSequence,
    *,
    frame_outputs: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    threshold: float,
) -> List[Dict[str, Any]]:
    frame_count = int(seq.length)
    active_segment_ids = _frame_segment_ids(frame_count, segments)
    prediction_by_segment, decision_by_segment = _segment_prediction_maps(predictions)
    rows: List[Dict[str, Any]] = []
    pose_enabled = bool(seq.pose_xyz is not None and seq.pose_vis is not None)
    for idx in range(frame_count):
        out = dict(frame_outputs[idx] if idx < len(frame_outputs) else {})
        probs = list(out.get("probs", []) or [])
        probs = [float(probs[i]) if i < len(probs) else 0.0 for i in range(3)]
        label_id = int(out.get("label", int(np.argmax(np.asarray(probs, dtype=np.float32))) if probs else 0))
        seg_id = active_segment_ids[idx]
        pred = prediction_by_segment.get(int(seg_id), {}) if seg_id is not None else {}
        decision = decision_by_segment.get(int(seg_id), {}) if seg_id is not None else {}
        left_valid_joints = int(seq.mask[idx, :21, 0].sum())
        right_valid_joints = int(seq.mask[idx, 21:, 0].sum())
        left_valid_frac = float(seq.mask[idx, :21, 0].mean())
        right_valid_frac = float(seq.mask[idx, 21:, 0].mean())
        pose_valid_joints = int(seq.pose_vis[idx].sum()) if pose_enabled and seq.pose_vis is not None else 0
        pose_valid_frac = float(seq.pose_vis[idx].mean()) if pose_enabled and seq.pose_vis is not None else 0.0
        rows.append(
            {
                "frame_index": int(idx),
                "ts_ms": float(seq.ts_ms[idx]),
                "bio_label_id": int(label_id),
                "bio_label": _bio_label_name(label_id),
                "pO": float(probs[0]),
                "pB": float(probs[1]),
                "pI": float(probs[2]),
                "threshold": float(out.get("threshold", threshold)),
                "active_segment_id": (None if seg_id is None else int(seg_id)),
                "predicted_label": str(pred.get("label", "")),
                "predicted_confidence": float(pred.get("confidence", 0.0) or 0.0),
                "family_label": str(pred.get("family_label", "")),
                "family_confidence": float(pred.get("family_confidence", 0.0) or 0.0),
                "prediction_accepted": bool(decision.get("accepted", False)),
                "sentence_decision_reason": str(decision.get("reason", "")),
                "left_valid_joints": left_valid_joints,
                "right_valid_joints": right_valid_joints,
                "left_valid_frac": left_valid_frac,
                "right_valid_frac": right_valid_frac,
                "pose_valid_joints": pose_valid_joints,
                "pose_valid_frac": pose_valid_frac,
                "warnings": _frame_warning_flags(
                    left_valid_frac=left_valid_frac,
                    right_valid_frac=right_valid_frac,
                    pose_valid_frac=pose_valid_frac,
                    pose_enabled=pose_enabled,
                ),
            }
        )
    return rows


def _build_review_warnings(
    seq: CanonicalSkeletonSequence,
    *,
    frame_rows: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    sentence_builder: Dict[str, Any],
) -> List[Dict[str, Any]]:
    warnings: List[Dict[str, Any]] = []
    frame_count = max(1, int(seq.length))
    left_present_ratio = float(sum(1 for row in frame_rows if float(row.get("left_valid_frac", 0.0)) > 0.0)) / frame_count
    right_present_ratio = float(sum(1 for row in frame_rows if float(row.get("right_valid_frac", 0.0)) > 0.0)) / frame_count
    pose_enabled = bool(seq.pose_vis is not None)
    pose_valid_mean = float(np.mean(seq.pose_vis)) if pose_enabled and seq.pose_vis is not None else 0.0
    if pose_enabled and pose_valid_mean < 0.35:
        warnings.append(
            {
                "id": "pose_reliability_low",
                "severity": "warning",
                "message": "Pose reliability is low for this clip; holistic pose may be unstable.",
                "pose_valid_mean": pose_valid_mean,
            }
        )
    if left_present_ratio < 0.25:
        warnings.append(
            {
                "id": "left_hand_sparse",
                "severity": "warning",
                "message": "Left hand is missing for most of the clip.",
                "present_ratio": left_present_ratio,
            }
        )
    if right_present_ratio < 0.25:
        warnings.append(
            {
                "id": "right_hand_sparse",
                "severity": "warning",
                "message": "Right hand is missing for most of the clip.",
                "present_ratio": right_present_ratio,
            }
        )
    if not segments:
        warnings.append(
            {
                "id": "no_segments",
                "severity": "warning",
                "message": "BIO emitted no segments for this clip.",
            }
        )
    elif predictions and not list(sentence_builder.get("words", []) or []):
        warnings.append(
            {
                "id": "all_predictions_rejected",
                "severity": "warning",
                "message": "All segment predictions were rejected by the sentence builder.",
                "rejected_count": int(len(list(sentence_builder.get("rejected_predictions", []) or []))),
            }
        )
    return warnings


def _build_timeline_tracks(
    *,
    seq: CanonicalSkeletonSequence,
    frame_rows: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prediction_by_segment, decision_by_segment = _segment_prediction_maps(predictions)
    segment_spans: List[Dict[str, Any]] = []
    prediction_spans: List[Dict[str, Any]] = []
    for seg in segments:
        seg_id = int(seg.get("segment_id", -1))
        pred = prediction_by_segment.get(seg_id, {})
        decision = decision_by_segment.get(seg_id, {})
        span = {
            "segment_id": seg_id,
            "start_frame": int(seg.get("start_frame", 0)),
            "end_frame_exclusive": int(seg.get("end_frame_exclusive", 0)),
            "start_time_ms": float(seg.get("start_time_ms", 0.0)),
            "end_time_ms": float(seg.get("end_time_ms", 0.0)),
            "end_reason": str(seg.get("end_reason", "")),
            "boundary_score": float(seg.get("boundary_score", 0.0)),
            "mean_inside_score": float(seg.get("mean_inside_score", 0.0)),
            "label": str(pred.get("label", "")),
            "confidence": float(pred.get("confidence", 0.0) or 0.0),
            "family_label": str(pred.get("family_label", "")),
            "family_confidence": float(pred.get("family_confidence", 0.0) or 0.0),
            "accepted": bool(decision.get("accepted", False)),
            "decision_reason": str(decision.get("reason", "")),
        }
        segment_spans.append(span)
        if pred:
            prediction_spans.append(dict(span))
    return {
        "version": 1,
        "frame_count": int(seq.length),
        "fps": float(seq.meta.get("fps", 0.0) or 0.0),
        "duration_ms": float(seq.ts_ms[-1] if seq.length else 0.0),
        "bio": {
            "threshold": float(frame_rows[0]["threshold"]) if frame_rows else 0.0,
            "label_id": [int(row.get("bio_label_id", 0)) for row in frame_rows],
            "label": [str(row.get("bio_label", "O")) for row in frame_rows],
            "pO": [float(row.get("pO", 0.0)) for row in frame_rows],
            "pB": [float(row.get("pB", 0.0)) for row in frame_rows],
            "pI": [float(row.get("pI", 0.0)) for row in frame_rows],
        },
        "hands": {
            "left_valid_frac": [float(row.get("left_valid_frac", 0.0)) for row in frame_rows],
            "right_valid_frac": [float(row.get("right_valid_frac", 0.0)) for row in frame_rows],
        },
        "pose": {
            "enabled": bool(seq.pose_vis is not None),
            "valid_frac": [float(row.get("pose_valid_frac", 0.0)) for row in frame_rows],
        },
        "segment_ids": [row.get("active_segment_id") for row in frame_rows],
        "prediction": {
            "label": [str(row.get("predicted_label", "")) for row in frame_rows],
            "confidence": [float(row.get("predicted_confidence", 0.0)) for row in frame_rows],
            "accepted": [bool(row.get("prediction_accepted", False)) for row in frame_rows],
        },
        "frame_warnings": [list(row.get("warnings", []) or []) for row in frame_rows],
        "segments": segment_spans,
        "predictions": prediction_spans,
        "global_warnings": warnings,
    }


def _write_preview_video(path: Path, frames: List[Any], predictions: List[Dict[str, Any]], fps: float) -> Optional[Path]:
    if not frames:
        return None
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(float(fps), 1.0), (width, height))
    if not writer.isOpened():
        return None
    labels_by_frame: Dict[int, List[str]] = {}
    for row in predictions:
        seg = dict(row.get("segment", {}) or {})
        pred = dict(row.get("prediction", {}) or {})
        label = str(pred.get("label", ""))
        for frame_idx in range(int(seg.get("start_frame", 0)), int(seg.get("end_frame_exclusive", 0))):
            labels_by_frame.setdefault(frame_idx, []).append(label)
    for idx, frame in enumerate(frames):
        canvas = frame.copy()
        labels = labels_by_frame.get(idx, [])
        if labels:
            cv2.putText(canvas, " ".join(labels[-3:]), (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2)
        writer.write(canvas)
    writer.release()
    return path


def _coord_to_px(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    return int(round(float(np.clip(x, 0.0, 1.0)) * max(width - 1, 1))), int(round(float(np.clip(y, 0.0, 1.0)) * max(height - 1, 1)))


def _draw_overlay_badge(
    frame: Any,
    lines: List[str],
    *,
    anchor: str,
    fill_bgr: tuple[int, int, int] = (24, 24, 28),
    text_bgr: tuple[int, int, int] = (245, 245, 245),
) -> Any:
    if frame is None or not lines:
        return frame
    canvas = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1
    pad_x = 10
    pad_y = 8
    line_h = 20
    sizes = [cv2.getTextSize(str(line), font, font_scale, thickness)[0] for line in lines]
    width = max((size[0] for size in sizes), default=0) + 2 * pad_x
    height = line_h * len(lines) + 2 * pad_y - 4
    frame_h, frame_w = canvas.shape[:2]
    margin = 12
    if anchor == "top_left":
        x0, y0 = margin, margin
    elif anchor == "top_right":
        x0, y0 = max(margin, frame_w - width - margin), margin
    elif anchor == "bottom_left":
        x0, y0 = margin, max(margin, frame_h - height - margin)
    else:
        x0, y0 = max(margin, frame_w - width - margin), max(margin, frame_h - height - margin)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), fill_bgr, -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0.0, canvas)
    y = y0 + pad_y + 12
    for line in lines:
        cv2.putText(canvas, str(line), (x0 + pad_x, y), font, font_scale, text_bgr, thickness, cv2.LINE_AA)
        y += line_h
    return canvas


def _draw_hand21_overlay(frame: Any, pts21, mask21, *, color: tuple[int, int, int]) -> Any:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    pts_arr = np.asarray(pts21, dtype=np.float32)
    mask_arr = np.asarray(mask21, dtype=np.float32).reshape(-1)
    for a, b in HAND_EDGES_ONE:
        if mask_arr[a] > 0.0 and mask_arr[b] > 0.0:
            p0 = _coord_to_px(pts_arr[a, 0], pts_arr[a, 1], width, height)
            p1 = _coord_to_px(pts_arr[b, 0], pts_arr[b, 1], width, height)
            cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)
    for idx in range(int(pts_arr.shape[0])):
        if mask_arr[idx] <= 0.0:
            continue
        cv2.circle(canvas, _coord_to_px(pts_arr[idx, 0], pts_arr[idx, 1], width, height), 3, color, -1, cv2.LINE_AA)
    return canvas


def _draw_hand_overlay(frame: Any, pts, mask, *, color: tuple[int, int, int]) -> Any:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    pts_arr = np.asarray(pts, dtype=np.float32)
    mask_arr = np.asarray(mask, dtype=np.float32).reshape(-1)
    for a, b in hand_edges_42():
        if mask_arr[a] > 0.0 and mask_arr[b] > 0.0:
            p0 = _coord_to_px(pts_arr[a, 0], pts_arr[a, 1], width, height)
            p1 = _coord_to_px(pts_arr[b, 0], pts_arr[b, 1], width, height)
            cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)
    for idx in range(int(pts_arr.shape[0])):
        if mask_arr[idx] <= 0.0:
            continue
        cv2.circle(canvas, _coord_to_px(pts_arr[idx, 0], pts_arr[idx, 1], width, height), 3, color, -1, cv2.LINE_AA)
    return canvas


def _draw_pose_overlay(frame: Any, pose_xyz, pose_vis, pose_indices) -> Any:
    if pose_xyz is None or pose_vis is None or pose_indices is None:
        return frame
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    pts_arr = np.asarray(pose_xyz, dtype=np.float32)
    vis_arr = np.asarray(pose_vis, dtype=np.float32).reshape(-1)
    pos_map = {int(abs_idx): i for i, abs_idx in enumerate(pose_indices)}
    for a_abs, b_abs in POSE_EDGE_PAIRS_ABS:
        a = pos_map.get(int(a_abs), -1)
        b = pos_map.get(int(b_abs), -1)
        if a < 0 or b < 0 or vis_arr[a] <= 0.0 or vis_arr[b] <= 0.0:
            continue
        p0 = _coord_to_px(pts_arr[a, 0], pts_arr[a, 1], width, height)
        p1 = _coord_to_px(pts_arr[b, 0], pts_arr[b, 1], width, height)
        cv2.line(canvas, p0, p1, (180, 180, 255), 1, cv2.LINE_AA)
    for idx in range(int(pts_arr.shape[0])):
        if vis_arr[idx] <= 0.0:
            continue
        cv2.circle(canvas, _coord_to_px(pts_arr[idx, 0], pts_arr[idx, 1], width, height), 2, (180, 180, 255), -1, cv2.LINE_AA)
    return canvas


def _write_mediapipe_overlay(path: Path, frames: List[Any], seq: CanonicalSkeletonSequence) -> Optional[Path]:
    if not frames:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    fps = float(seq.meta.get("fps", 30.0) or 30.0)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (width, height))
    if not writer.isOpened():
        return None
    try:
        for idx, frame in enumerate(frames):
            canvas = frame.copy()
            canvas = _draw_pose_overlay(
                canvas,
                (seq.pose_xyz[idx] if seq.pose_xyz is not None else None),
                (seq.pose_vis[idx] if seq.pose_vis is not None else None),
                seq.pose_indices,
            )
            canvas = _draw_hand_overlay(canvas, seq.pts[idx], seq.mask[idx], color=(80, 220, 80))
            cv2.putText(canvas, f"frame={idx}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()
    return path


def _write_review_overlay_video(
    path: Path,
    frames: List[Any],
    seq: CanonicalSkeletonSequence,
    frame_rows: List[Dict[str, Any]],
) -> Optional[Path]:
    if not frames:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    fps = float(seq.meta.get("fps", 30.0) or 30.0)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (width, height))
    if not writer.isOpened():
        return None
    try:
        for idx, frame in enumerate(frames):
            row = dict(frame_rows[idx] if idx < len(frame_rows) else {})
            canvas = frame.copy()
            canvas = _draw_pose_overlay(
                canvas,
                (seq.pose_xyz[idx] if seq.pose_xyz is not None else None),
                (seq.pose_vis[idx] if seq.pose_vis is not None else None),
                seq.pose_indices,
            )
            canvas = _draw_hand21_overlay(canvas, seq.pts[idx, :21, :], seq.mask[idx, :21, :], color=(80, 220, 80))
            canvas = _draw_hand21_overlay(canvas, seq.pts[idx, 21:, :], seq.mask[idx, 21:, :], color=(80, 180, 255))
            lines = [f"frame {idx}  t={float(row.get('ts_ms', 0.0)):.1f}ms", f"mode {seq.meta.get('extractor_mode', 'unknown')}", f"BIO {row.get('bio_label', 'O')}"]
            canvas = _draw_overlay_badge(canvas, lines, anchor="top_left")
            canvas = _draw_overlay_badge(
                canvas,
                [
                    (
                        f"pO {float(row.get('pO', 0.0)):.2f}  "
                        f"pB {float(row.get('pB', 0.0)):.2f}  "
                        f"pI {float(row.get('pI', 0.0)):.2f}"
                    ),
                    f"thr {float(row.get('threshold', 0.0)):.2f}",
                ],
                anchor="top_right",
            )
            bottom_left: List[str] = []
            seg_id = row.get("active_segment_id")
            if seg_id is not None:
                bottom_left.append(f"segment #{int(seg_id)}")
            pred_label = str(row.get("predicted_label", "")).strip()
            if pred_label:
                bottom_left.append(f"{pred_label}  {float(row.get('predicted_confidence', 0.0)):.2f}")
            family_label = str(row.get("family_label", "")).strip()
            if family_label:
                bottom_left.append(f"family {family_label}  {float(row.get('family_confidence', 0.0)):.2f}")
            if bottom_left:
                canvas = _draw_overlay_badge(canvas, bottom_left, anchor="bottom_left")
            warnings = list(row.get("warnings", []) or [])
            if warnings:
                canvas = _draw_overlay_badge(
                    canvas,
                    [f"warn {str(flag)}" for flag in warnings[:3]],
                    anchor="bottom_right",
                    fill_bgr=(54, 34, 24),
                )
            writer.write(canvas)
    finally:
        writer.release()
    return path


def _mediapipe_debug_summary(seq: CanonicalSkeletonSequence, *, extractor_mode: str) -> Dict[str, Any]:
    left_valid = seq.mask[:, :21, 0].sum(axis=1)
    right_valid = seq.mask[:, 21:, 0].sum(axis=1)
    pose_valid = seq.pose_vis.sum(axis=1) if seq.pose_vis is not None else np.zeros((seq.length,), dtype=np.float32)
    return {
        "video_path": str(seq.meta.get("source_video", "")),
        "frames": int(seq.length),
        "fps": float(seq.meta.get("fps", 0.0) or 0.0),
        "extractor_mode": str(extractor_mode),
        "extractor": str(seq.meta.get("extractor", "unknown")),
        "pose_enabled": bool(seq.pose_xyz is not None),
        "left_hand_present_frames": int((left_valid > 0).sum()),
        "right_hand_present_frames": int((right_valid > 0).sum()),
        "both_hands_present_frames": int(((left_valid > 0) & (right_valid > 0)).sum()),
        "pose_present_frames": int((pose_valid > 0).sum()),
        "mean_left_valid_joints": float(left_valid.mean() if left_valid.size else 0.0),
        "mean_right_valid_joints": float(right_valid.mean() if right_valid.size else 0.0),
        "mean_pose_valid_joints": float(pose_valid.mean() if pose_valid.size else 0.0),
        "preprocessing_version": BIO_RUNTIME_PREPROCESSING_VERSION,
    }


def _write_frame_stats_jsonl(path: Path, seq: CanonicalSkeletonSequence) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx in range(seq.length):
            row = {
                "frame_index": int(idx),
                "ts_ms": float(seq.ts_ms[idx]),
                "left_valid_joints": int(seq.mask[idx, :21, 0].sum()),
                "right_valid_joints": int(seq.mask[idx, 21:, 0].sum()),
                "left_valid_frac": float(seq.mask[idx, :21, 0].mean()),
                "right_valid_frac": float(seq.mask[idx, 21:, 0].mean()),
                "pose_valid_joints": int(seq.pose_vis[idx].sum()) if seq.pose_vis is not None else 0,
                "pose_valid_frac": float(seq.pose_vis[idx].mean()) if seq.pose_vis is not None else 0.0,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _load_runtime_metadata_from_config(cfg: InferencePipelineConfig) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "bio_config_source": ("bundle" if cfg.bio_bundle else "checkpoint"),
        "msagcn_config_source": ("bundle" if cfg.msagcn_bundle else "checkpoint"),
    }
    if cfg.bio_bundle:
        try:
            summary["bio_manifest"] = {k: v for k, v in load_runtime_manifest(cfg.bio_bundle).items() if not str(k).startswith("_")}
        except FileNotFoundError:
            summary["bio_manifest_missing"] = str(cfg.bio_bundle)
    if cfg.msagcn_bundle:
        try:
            summary["msagcn_manifest"] = {k: v for k, v in load_runtime_manifest(cfg.msagcn_bundle).items() if not str(k).startswith("_")}
        except FileNotFoundError:
            summary["msagcn_manifest_missing"] = str(cfg.msagcn_bundle)
    return summary


def _resolve_extractor_mode(cfg: InferencePipelineConfig, *, requires_pose: bool) -> str:
    requested = str(cfg.extractor_mode or "auto").strip().lower() or "auto"
    if requested == "auto":
        return "holistic_hands_pose" if requires_pose else "hands_only"
    if requires_pose and requested != "holistic_hands_pose":
        raise RuntimeError("This MSAGCN checkpoint requires pose. Use extractor_mode=holistic_hands_pose or leave it on auto.")
    return requested


def run_video_pipeline(
    video_path: str | Path,
    *,
    cfg: InferencePipelineConfig,
    out_dir: str | Path,
    tracker_cfg: Optional[MediaPipeHandsConfig] = None,
    max_frames: int = 0,
    write_preview: bool = False,
) -> Dict[str, Any]:
    pipeline = InferencePipeline.from_config(cfg, async_classification=False)
    extractor_mode = _resolve_extractor_mode(cfg, requires_pose=pipeline.requires_pose)
    require_pose = extractor_mode == "holistic_hands_pose"
    seq, frames = extract_video_sequence(
        video_path,
        tracker_cfg=tracker_cfg,
        holistic_cfg=_build_holistic_cfg(tracker_cfg),
        require_pose=require_pose,
        max_frames=int(max_frames),
        return_frames=bool(write_preview),
    )
    pipeline.runtime_summary["extractor_mode"] = str(seq.meta.get("extractor_mode", extractor_mode))
    pipeline.runtime_summary["extractor"] = str(seq.meta.get("extractor", "unknown"))
    result = pipeline.run_sequence(seq, eos_policy="close_open_segment_on_eos")
    result["video_path"] = str(Path(video_path).resolve())
    result["runtime_summary"] = {**result.get("runtime_summary", {}), **_load_runtime_metadata_from_config(cfg)}
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    _write_json(out_root / "segments.json", {"segments": result["segments"], "sequence_meta": result.get("sequence_meta", {}), "runtime_summary": result["runtime_summary"]})
    _write_json(out_root / "predictions.json", {"predictions": result["predictions"], "sentence_builder": result["sentence_builder"], "runtime_summary": result["runtime_summary"]})
    _write_timeline_csv(out_root / "timeline.csv", result["predictions"])
    (out_root / "sentence.txt").write_text(str(result["sentence_builder"]["sentence"]), encoding="utf-8")
    if write_preview:
        _write_preview_video(out_root / "preview_overlay.mp4", frames, result["predictions"], float(seq.meta.get("fps", 30.0) or 30.0))
    return result


def build_review_session(
    video_path: str | Path,
    *,
    cfg: InferencePipelineConfig,
    out_dir: str | Path,
    tracker_cfg: Optional[MediaPipeHandsConfig] = None,
    max_frames: int = 0,
) -> Dict[str, Any]:
    pipeline = InferencePipeline.from_config(cfg, async_classification=False)
    extractor_mode = _resolve_extractor_mode(cfg, requires_pose=pipeline.requires_pose)
    require_pose = extractor_mode == "holistic_hands_pose"
    seq, frames = extract_video_sequence(
        video_path,
        tracker_cfg=tracker_cfg,
        holistic_cfg=_build_holistic_cfg(tracker_cfg),
        require_pose=require_pose,
        max_frames=int(max_frames),
        return_frames=True,
    )
    pipeline.runtime_summary["extractor_mode"] = str(seq.meta.get("extractor_mode", extractor_mode))
    pipeline.runtime_summary["extractor"] = str(seq.meta.get("extractor", "unknown"))
    frame_outputs: List[Dict[str, Any]] = []
    for idx in range(seq.length):
        step_result = pipeline.process_frame(
            pts=seq.pts[idx],
            mask=seq.mask[idx],
            ts_ms=float(seq.ts_ms[idx]),
            pose_xyz=(seq.pose_xyz[idx] if seq.pose_xyz is not None else None),
            pose_vis=(seq.pose_vis[idx] if seq.pose_vis is not None else None),
            pose_indices=seq.pose_indices,
        )
        frame_outputs.append(dict(step_result.get("bio", {}) or {}))
    final = pipeline.finalize(eos_policy="close_open_segment_on_eos")
    final["sequence_meta"] = dict(seq.meta)
    runtime_summary = {**final.get("runtime_summary", {}), **_load_runtime_metadata_from_config(cfg)}
    frame_rows = _build_frame_debug_rows(
        seq,
        frame_outputs=frame_outputs,
        segments=list(final.get("segments", []) or []),
        predictions=list(final.get("predictions", []) or []),
        threshold=float(pipeline.bio.threshold),
    )
    warnings = _build_review_warnings(
        seq,
        frame_rows=frame_rows,
        segments=list(final.get("segments", []) or []),
        predictions=list(final.get("predictions", []) or []),
        sentence_builder=dict(final.get("sentence_builder", {}) or {}),
    )
    timeline_tracks = _build_timeline_tracks(
        seq=seq,
        frame_rows=frame_rows,
        segments=list(final.get("segments", []) or []),
        predictions=list(final.get("predictions", []) or []),
        warnings=warnings,
    )

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    canonical_path = save_skeleton_sequence_npz(out_root / "canonical_sequence.npz", seq)
    sequence_manifest_path = _write_json(out_root / "sequence_manifest.json", seq.to_manifest_dict())
    frame_stats_path = _write_frame_stats_jsonl(out_root / "frame_stats.jsonl", seq)
    frame_debug_path = _write_jsonl(out_root / "frame_debug.jsonl", frame_rows)
    segments_path = _write_json(
        out_root / "segments.json",
        {
            "segments": list(final.get("segments", []) or []),
            "sequence_meta": dict(seq.meta),
            "runtime_summary": runtime_summary,
        },
    )
    predictions_path = _write_json(
        out_root / "predictions.json",
        {
            "predictions": list(final.get("predictions", []) or []),
            "sentence_builder": dict(final.get("sentence_builder", {}) or {}),
            "runtime_summary": runtime_summary,
        },
    )
    timeline_csv_path = _write_timeline_csv(out_root / "timeline.csv", list(final.get("predictions", []) or []))
    timeline_tracks_path = _write_json(out_root / "timeline_tracks.json", timeline_tracks)
    sentence_path = out_root / "sentence.txt"
    sentence_path.write_text(str(dict(final.get("sentence_builder", {}) or {}).get("sentence", "")), encoding="utf-8")
    preview_path = _write_review_overlay_video(out_root / "preview_overlay.mp4", frames, seq, frame_rows)
    if preview_path is None:
        preview_path = _write_preview_video(
            out_root / "preview_overlay.mp4",
            frames,
            list(final.get("predictions", []) or []),
            float(seq.meta.get("fps", 30.0) or 30.0),
        )

    session_manifest = {
        "version": 1,
        "session_type": "desktop_review_session",
        "video_path": str(Path(video_path).expanduser().resolve()),
        "output_dir": str(out_root.resolve()),
        "extractor_mode": str(seq.meta.get("extractor_mode", extractor_mode)),
        "extractor": str(seq.meta.get("extractor", "unknown")),
        "pose_enabled": bool(seq.pose_xyz is not None),
        "runtime_summary": runtime_summary,
        "sequence_meta": dict(seq.meta),
        "decoder_config": (
            asdict(pipeline.bio.decoder.cfg)
            if getattr(getattr(pipeline.bio, "decoder", None), "cfg", None) is not None
            else {"start_threshold": float(getattr(pipeline.bio, "threshold", 0.0))}
        ),
        "sentence_builder_config": asdict(pipeline.sentence_builder.cfg),
        "config": asdict(cfg),
        "model_sources": {
            "bio": {
                "mode": ("bundle" if cfg.bio_bundle else "checkpoint"),
                "source": str(cfg.bio_bundle or cfg.bio_checkpoint),
                "selection": str(cfg.bio_selection or "best_balanced"),
            },
            "msagcn": {
                "mode": ("bundle" if cfg.msagcn_bundle else "checkpoint"),
                "source": str(cfg.msagcn_bundle or cfg.msagcn_checkpoint),
                "label_map": str(cfg.msagcn_label_map or ""),
                "ds_config": str(cfg.msagcn_ds_config or ""),
            },
        },
        "counts": {
            "frames": int(seq.length),
            "segments": int(len(list(final.get("segments", []) or []))),
            "predictions": int(len(list(final.get("predictions", []) or []))),
            "accepted_words": int(len(list(dict(final.get("sentence_builder", {}) or {}).get("words", []) or []))),
        },
        "warnings": warnings,
        "artifacts": {
            "canonical_sequence": str(canonical_path.relative_to(out_root)),
            "sequence_manifest": str(sequence_manifest_path.relative_to(out_root)),
            "frame_stats": str(frame_stats_path.relative_to(out_root)),
            "frame_debug": str(frame_debug_path.relative_to(out_root)),
            "segments": str(segments_path.relative_to(out_root)),
            "predictions": str(predictions_path.relative_to(out_root)),
            "timeline_csv": str(timeline_csv_path.relative_to(out_root)),
            "timeline_tracks": str(timeline_tracks_path.relative_to(out_root)),
            "sentence": str(sentence_path.relative_to(out_root)),
            "preview_overlay": (str(preview_path.relative_to(out_root)) if preview_path is not None else ""),
        },
    }
    session_path = _write_json(out_root / "session.json", session_manifest)
    return {
        "session_path": str(session_path.resolve()),
        "session": session_manifest,
        "sentence": str(dict(final.get("sentence_builder", {}) or {}).get("sentence", "")),
        "segments": int(len(list(final.get("segments", []) or []))),
        "predictions": int(len(list(final.get("predictions", []) or []))),
        "warnings": warnings,
    }


def run_camera_pipeline(
    source: int | str,
    *,
    cfg: InferencePipelineConfig,
    tracker_cfg: Optional[MediaPipeHandsConfig] = None,
    out_dir: str | Path | None = None,
    max_frames: int = 0,
    display: bool = False,
) -> Dict[str, Any]:
    pipeline = InferencePipeline.from_config(cfg, async_classification=True)
    extractor_mode = _resolve_extractor_mode(cfg, requires_pose=pipeline.requires_pose)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera/video source: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
    dt_ms = 1000.0 / fps
    tracker: MediaPipeHandsTracker | MediaPipeHolisticTracker
    if extractor_mode == "holistic_hands_pose":
        tracker = MediaPipeHolisticTracker(cfg=_build_holistic_cfg(tracker_cfg))
        pipeline.runtime_summary["extractor"] = "mediapipe.solutions.holistic"
    else:
        tracker = MediaPipeHandsTracker(cfg=tracker_cfg or MediaPipeHandsConfig())
        pipeline.runtime_summary["extractor"] = "mediapipe.solutions.hands"
    pipeline.runtime_summary["extractor_mode"] = extractor_mode
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts_ms = float(frame_idx) * dt_ms
            item = tracker.process_bgr(frame, ts_ms=ts_ms)
            result = pipeline.process_frame(
                pts=item["pts"],
                mask=item["mask"],
                ts_ms=ts_ms,
                pose_xyz=item.get("pose_xyz"),
                pose_vis=item.get("pose_vis"),
                pose_indices=item.get("pose_indices"),
            )
            if display:
                cv2.putText(frame, result["sentence"] or "", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2)
                cv2.imshow("pipeline infer-camera", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            frame_idx += 1
            if int(max_frames) > 0 and frame_idx >= int(max_frames):
                break
    finally:
        tracker.close()
        cap.release()
        if display:
            cv2.destroyAllWindows()
    final = pipeline.finalize(eos_policy="drop_partial_on_eos")
    final["runtime_summary"] = {**final.get("runtime_summary", {}), **_load_runtime_metadata_from_config(cfg)}
    if out_dir is not None:
        out_root = Path(out_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        _write_json(out_root / "segments.json", {"segments": final["segments"], "runtime_summary": final["runtime_summary"]})
        _write_json(out_root / "predictions.json", {"predictions": final["predictions"], "sentence_builder": final["sentence_builder"], "runtime_summary": final["runtime_summary"]})
        (out_root / "sentence.txt").write_text(str(final["sentence_builder"]["sentence"]), encoding="utf-8")
    return final


def run_mediapipe_debug(
    video_path: str | Path,
    *,
    out_dir: str | Path,
    tracker_cfg: Optional[MediaPipeHandsConfig] = None,
    max_frames: int = 0,
    extractor_mode: str = "hands_only",
    write_overlay: bool = True,
) -> Dict[str, Any]:
    require_pose = str(extractor_mode or "hands_only") == "holistic_hands_pose"
    seq, frames = extract_video_sequence(
        video_path,
        tracker_cfg=tracker_cfg,
        holistic_cfg=_build_holistic_cfg(tracker_cfg),
        require_pose=require_pose,
        max_frames=int(max_frames),
        return_frames=bool(write_overlay),
    )
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    save_skeleton_sequence_npz(out_root / "canonical_sequence.npz", seq)
    (out_root / "sequence_manifest.json").write_text(
        json.dumps(seq.to_manifest_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_frame_stats_jsonl(out_root / "frame_stats.jsonl", seq)
    summary = _mediapipe_debug_summary(seq, extractor_mode=str(seq.meta.get("extractor_mode", extractor_mode)))
    _write_json(out_root / "summary.json", summary)
    overlay_path = None
    if write_overlay:
        overlay_path = _write_mediapipe_overlay(out_root / "overlay.mp4", frames, seq)
    return {
        "summary": summary,
        "output_dir": str(out_root.resolve()),
        "overlay_path": (str(overlay_path.resolve()) if overlay_path is not None else ""),
        "canonical_sequence": str((out_root / "canonical_sequence.npz").resolve()),
    }
