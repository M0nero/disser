from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TaskSpec:
    sample_id: str
    slug: str
    source_video: str
    config_dict: Dict[str, Any]
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None
    segment_meta: Dict[str, Any] = field(default_factory=dict)
    debug_video_path: str = ""
    ndjson_path: str = ""

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "TaskSpec":
        return cls(
            sample_id=str(payload["sample_id"]),
            slug=str(payload.get("slug") or payload["sample_id"]),
            source_video=str(payload["source_video"]),
            config_dict=dict(payload["config_dict"]),
            frame_start=payload.get("frame_start"),
            frame_end=payload.get("frame_end"),
            segment_meta=dict(payload.get("segment_meta") or {}),
            debug_video_path=str(payload.get("debug_video_path") or ""),
            ndjson_path=str(payload.get("ndjson_path") or ""),
        )

    @property
    def source_path(self) -> Path:
        return Path(self.source_video)

    @property
    def has_segment(self) -> bool:
        return bool(self.segment_meta)
