from __future__ import annotations

import json
from functools import cached_property
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from runtime.skeleton import CanonicalSkeletonSequence, load_skeleton_sequence


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _resolve_session_path(path: str | Path) -> Path:
    source = Path(path).expanduser()
    if source.is_dir():
        source = source / "session.json"
    if not source.exists():
        raise FileNotFoundError(source)
    return source.resolve()


@dataclass
class ReviewSession:
    root: Path
    manifest: Dict[str, Any]
    segments_payload: Dict[str, Any]
    predictions_payload: Dict[str, Any]
    timeline_tracks: Dict[str, Any]
    frame_rows: List[Dict[str, Any]]
    frame_stats_rows: List[Dict[str, Any]]
    sequence_manifest: Dict[str, Any]
    _sequence: CanonicalSkeletonSequence | None = None

    @property
    def session_name(self) -> str:
        video_name = Path(self.video_path).name if self.video_path else ""
        return video_name or self.root.name

    @property
    def sentence(self) -> str:
        return str(self.predictions_payload.get("sentence_builder", {}).get("sentence", ""))

    @property
    def warnings(self) -> List[Dict[str, Any]]:
        return list(self.manifest.get("warnings", []) or [])

    @property
    def frame_count(self) -> int:
        return int(self.manifest.get("counts", {}).get("frames", len(self.frame_rows)))

    @property
    def fps(self) -> float:
        return float(self.manifest.get("sequence_meta", {}).get("fps", self.timeline_tracks.get("fps", 0.0)) or 0.0)

    @property
    def video_path(self) -> str:
        return str(self.manifest.get("video_path", ""))

    @cached_property
    def segment_rows(self) -> List[Dict[str, Any]]:
        return self.combined_segment_rows()

    @property
    def accepted_segment_count(self) -> int:
        return int(sum(1 for row in self.segment_rows if bool(row.get("accepted", False))))

    @property
    def rejected_segment_count(self) -> int:
        return int(sum(1 for row in self.segment_rows if not bool(row.get("accepted", False))))

    @property
    def left_hand_present_ratio(self) -> float:
        rows = list(self.frame_rows or [])
        if not rows:
            return 0.0
        return float(sum(1 for row in rows if float(row.get("left_valid_frac", 0.0)) > 0.0)) / float(len(rows))

    @property
    def right_hand_present_ratio(self) -> float:
        rows = list(self.frame_rows or [])
        if not rows:
            return 0.0
        return float(sum(1 for row in rows if float(row.get("right_valid_frac", 0.0)) > 0.0)) / float(len(rows))

    @property
    def pose_valid_mean(self) -> float:
        rows = list(self.frame_rows or [])
        if not rows:
            return 0.0
        vals = [float(row.get("pose_valid_frac", 0.0)) for row in rows]
        return float(sum(vals) / max(len(vals), 1))

    def artifact_path(self, key: str) -> Path:
        rel = str(self.manifest.get("artifacts", {}).get(key, "") or "")
        if not rel:
            raise KeyError(f"Missing artifact path for {key!r}")
        return (self.root / rel).resolve()

    def load_sequence(self) -> CanonicalSkeletonSequence:
        if self._sequence is None:
            self._sequence = load_skeleton_sequence(self.artifact_path("canonical_sequence"))
        return self._sequence

    def frame_row(self, frame_index: int) -> Dict[str, Any]:
        idx = max(0, min(int(frame_index), max(0, len(self.frame_rows) - 1)))
        return dict(self.frame_rows[idx])

    def combined_segment_rows(self) -> List[Dict[str, Any]]:
        prediction_rows = list(self.predictions_payload.get("predictions", []) or [])
        pred_by_segment = {
            int(dict(row.get("segment", {}) or {}).get("segment_id", -1)): row
            for row in prediction_rows
            if int(dict(row.get("segment", {}) or {}).get("segment_id", -1)) >= 0
        }
        rows: List[Dict[str, Any]] = []
        for seg in list(self.segments_payload.get("segments", []) or []):
            seg = dict(seg or {})
            seg_id = int(seg.get("segment_id", -1))
            row = pred_by_segment.get(seg_id, {})
            pred = dict(row.get("prediction", {}) or {})
            decision = dict(row.get("sentence_decision", {}) or {})
            rows.append(
                {
                    "segment_id": seg_id,
                    "start_frame": int(seg.get("start_frame", 0)),
                    "end_frame_exclusive": int(seg.get("end_frame_exclusive", 0)),
                    "start_time_ms": float(seg.get("start_time_ms", 0.0)),
                    "end_time_ms": float(seg.get("end_time_ms", 0.0)),
                    "duration_ms": max(0.0, float(seg.get("end_time_ms", 0.0)) - float(seg.get("start_time_ms", 0.0))),
                    "end_reason": str(seg.get("end_reason", "")),
                    "boundary_score": float(seg.get("boundary_score", 0.0)),
                    "mean_inside_score": float(seg.get("mean_inside_score", 0.0)),
                    "label": str(pred.get("label", "")),
                    "confidence": float(pred.get("confidence", 0.0) or 0.0),
                    "family_label": str(pred.get("family_label", "")),
                    "family_confidence": float(pred.get("family_confidence", 0.0) or 0.0),
                    "accepted": bool(decision.get("accepted", False)),
                    "decision_reason": str(decision.get("reason", "")),
                    "prediction": pred,
                    "sentence_decision": decision,
                }
            )
        return rows


def load_review_session(path: str | Path) -> ReviewSession:
    session_path = _resolve_session_path(path)
    root = session_path.parent
    manifest = _load_json(session_path)
    artifacts = dict(manifest.get("artifacts", {}) or {})
    segments_payload = _load_json(root / str(artifacts.get("segments", "segments.json")))
    predictions_payload = _load_json(root / str(artifacts.get("predictions", "predictions.json")))
    timeline_tracks = _load_json(root / str(artifacts.get("timeline_tracks", "timeline_tracks.json")))
    frame_rows = _load_jsonl(root / str(artifacts.get("frame_debug", "frame_debug.jsonl")))
    frame_stats_rows = _load_jsonl(root / str(artifacts.get("frame_stats", "frame_stats.jsonl")))
    sequence_manifest = _load_json(root / str(artifacts.get("sequence_manifest", "sequence_manifest.json")))
    return ReviewSession(
        root=root,
        manifest=manifest,
        segments_payload=segments_payload,
        predictions_payload=predictions_payload,
        timeline_tracks=timeline_tracks,
        frame_rows=frame_rows,
        frame_stats_rows=frame_stats_rows,
        sequence_manifest=sequence_manifest,
    )
