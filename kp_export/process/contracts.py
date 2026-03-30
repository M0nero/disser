from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, TypedDict


class Landmark(TypedDict):
    x: float
    y: float
    z: float
    visibility: Optional[float]


class FrameData(TypedDict, total=False):
    ts: int
    dt: int
    hand_1: Optional[List[Dict[str, float]]]
    hand_1_score: Optional[float]
    hand_1_source: Optional[str]
    hand_1_state: Optional[str]
    hand_1_is_anchor: Optional[bool]
    hand_1_reject_reason: Optional[str]
    hand_2: Optional[List[Dict[str, float]]]
    hand_2_score: Optional[float]
    hand_2_source: Optional[str]
    hand_2_state: Optional[str]
    hand_2_is_anchor: Optional[bool]
    hand_2_reject_reason: Optional[str]
    pose: Optional[List[Dict[str, float]]]
    pose_vis: Optional[List[float]]
    pose_interpolated: bool
    hand_mask: Optional[List[int]]
    both_hands: Optional[int]
    hand_1_sp_roi_px: Optional[List[int]]
    hand_2_sp_roi_px: Optional[List[int]]


@dataclass
class SanityResult:
    ok: bool
    reason_codes: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        yield self.ok
        yield self.reason_codes


@dataclass
class SecondPassResult:
    landmarks: Optional[List[Dict[str, float]]]
    score: Optional[float]
    recovered: bool
    roi: Any = None
    label_match: Optional[bool] = None
    combined_score: Optional[float] = None
    debug: Optional[Dict[str, Any]] = None

    def __iter__(self):
        yield self.landmarks
        yield self.score
        yield self.recovered
        yield self.roi


@dataclass
class HandObservation:
    landmarks: Optional[List[Dict[str, float]]] = None
    score: Optional[float] = None
    score_gate: Optional[float] = None
    source: Optional[str] = None
    state: Optional[str] = None
    is_anchor: Optional[bool] = None
    reject_reason: Optional[str] = None
    pose_quality: Optional[float] = None
    wrist_z: Optional[float] = None
    track_age: Optional[int] = None
    track_reset: Optional[bool] = None
    tracker_ready: Optional[bool] = None
    tracker_last_score: Optional[float] = None
    tracker_last_ts: Optional[float] = None


@dataclass
class PoseObservation:
    landmarks: Optional[List[Dict[str, float]]] = None
    visibility: Optional[List[float]] = None
    interpolated: bool = False
    world_landmarks: Optional[List[Dict[str, float]]] = None
    image_landmarks: Optional[List[Dict[str, float]]] = None


@dataclass
class FrameDiagnostics:
    values: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.values)


@dataclass
class FrameRecord:
    frame_idx: int
    ts_ms: int
    dt_ms: int
    hand_1: HandObservation = field(default_factory=HandObservation)
    hand_2: HandObservation = field(default_factory=HandObservation)
    pose: PoseObservation = field(default_factory=PoseObservation)
    both_hands: bool = False
    diagnostics: FrameDiagnostics = field(default_factory=FrameDiagnostics)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleSummary:
    sample_id: str
    slug: str
    source_video: str
    sample_attrs: Dict[str, Any]
    video_row: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "slug": self.slug,
            "source_video": self.source_video,
            "sample_attrs": dict(self.sample_attrs),
            "video_row": dict(self.video_row),
        }


@dataclass
class SamplePayload:
    sample_id: str
    slug: str
    source_video: str
    sample_attrs: Dict[str, Any]
    video_row: Dict[str, Any]
    frame_rows: List[Dict[str, Any]]
    raw_arrays: Dict[str, Any]
    pp_arrays: Optional[Dict[str, Any]] = None
    runtime_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "slug": self.slug,
            "source_video": self.source_video,
            "sample_attrs": dict(self.sample_attrs),
            "video_row": dict(self.video_row),
            "frame_rows": [dict(row) for row in self.frame_rows],
            "raw_arrays": dict(self.raw_arrays),
            "pp_arrays": dict(self.pp_arrays) if self.pp_arrays is not None else None,
            "runtime_metrics": dict(self.runtime_metrics) if self.runtime_metrics is not None else None,
        }

    def to_json_ready_dict(self) -> Dict[str, Any]:
        return asdict(self)
