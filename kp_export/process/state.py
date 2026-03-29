from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


MIN_HISTORY_INDEX = -(10**9)


def _can_persist_detected_hand(
    landmarks: Optional[List[Dict[str, float]]],
    source: Optional[str],
    *,
    overlap_ambiguous: bool,
    side_ok: bool,
    overlap_guard: bool,
) -> bool:
    return (
        landmarks is not None
        and source in ("pass1", "pass2")
        and (not overlap_ambiguous or side_ok)
        and not overlap_guard
    )


@dataclass
class HandRuntime:
    side: str
    last_export: Optional[List[Dict[str, float]]] = None
    last_export_score: Optional[float] = None
    last_good_px: Optional[List[Dict[str, float]]] = None
    last_good_img: Optional[List[Dict[str, float]]] = None
    last_anchor: Optional[List[Dict[str, float]]] = None
    last_anchor_i: int = MIN_HISTORY_INDEX
    last_obs: Optional[List[Dict[str, float]]] = None
    last_obs_i: int = MIN_HISTORY_INDEX
    hold: int = 0
    occ_ttl: int = 0
    occ_freeze_age: int = 0
    tracker_ready: bool = False
    track_age: int = 0
    track_recovered: int = 0

    def previous_observation(self, frame_idx: int) -> Optional[List[Dict[str, float]]]:
        if frame_idx - self.last_obs_i == 1:
            return self.last_obs
        return None

    def anchor_for_sanity(
        self,
        frame_idx: int,
        *,
        max_gap: int,
    ) -> Optional[List[Dict[str, float]]]:
        if max_gap > 0 and (frame_idx - self.last_anchor_i) > max_gap:
            return None
        return self.last_anchor

    def note_observation(
        self,
        frame_idx: int,
        *,
        landmarks: Optional[List[Dict[str, float]]],
        source: Optional[str],
        overlap_ambiguous: bool,
        side_ok: bool,
        overlap_guard: bool,
    ) -> None:
        if _can_persist_detected_hand(
            landmarks,
            source,
            overlap_ambiguous=overlap_ambiguous,
            side_ok=side_ok,
            overlap_guard=overlap_guard,
        ):
            self.last_obs = landmarks
            self.last_obs_i = int(frame_idx)

    def maybe_export(
        self,
        *,
        landmarks: Optional[List[Dict[str, float]]],
        score: Optional[float],
        source: Optional[str],
        overlap_ambiguous: bool,
        side_ok: bool,
        overlap_guard: bool,
        cur_px: Optional[List[Dict[str, float]]],
        cur_img: Optional[List[Dict[str, float]]],
    ) -> bool:
        if not _can_persist_detected_hand(
            landmarks,
            source,
            overlap_ambiguous=overlap_ambiguous,
            side_ok=side_ok,
            overlap_guard=overlap_guard,
        ):
            return False
        if score is None:
            return False
        self.last_export = landmarks
        self.last_export_score = float(score)
        self.last_good_px = cur_px
        self.last_good_img = cur_img
        return True

    def maybe_anchor(
        self,
        frame_idx: int,
        *,
        landmarks: Optional[List[Dict[str, float]]],
        score: Optional[float],
        source: Optional[str],
        anchor_score: float,
        pose_ok: bool,
        overlap_ambiguous: bool,
        side_ok: bool,
        overlap_guard: bool,
    ) -> bool:
        is_anchor = bool(
            landmarks is not None
            and score is not None
            and score >= anchor_score
            and pose_ok
            and (not overlap_ambiguous or side_ok)
            and not overlap_guard
            and source in ("pass1", "pass2")
        )
        if is_anchor:
            self.last_anchor = landmarks
            self.last_anchor_i = int(frame_idx)
        return is_anchor


def classify_hand_state(
    landmarks: Optional[List[Dict[str, float]]],
    source: Optional[str],
) -> str:
    if landmarks is None:
        return "missing"
    if source == "occluded":
        return "occluded"
    if source == "tracked":
        return "predicted"
    return "observed"


@dataclass
class SampleRuntime:
    sample_id: str
    left: HandRuntime = field(default_factory=lambda: HandRuntime(side="left"))
    right: HandRuntime = field(default_factory=lambda: HandRuntime(side="right"))
    last_overlap_freeze_side: Optional[str] = None
    last_overlap_freeze_i: int = MIN_HISTORY_INDEX
    frame_start: int = 0
    frame_end: Optional[int] = None
    counters: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self.counters[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.counters.get(key, default)
