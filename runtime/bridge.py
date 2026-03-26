from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from .skeleton import CanonicalSkeletonSequence, canonicalize_sequence


@dataclass
class SegmentClip:
    segment_id: int
    start_frame: int
    end_frame_exclusive: int
    start_time_ms: float
    end_time_ms: float
    pts: np.ndarray
    mask: np.ndarray
    ts_ms: np.ndarray
    pose_xyz: np.ndarray | None
    pose_vis: np.ndarray | None
    pose_indices: tuple[int, ...] | None
    source: Dict[str, Any]

    def as_sequence(self) -> CanonicalSkeletonSequence:
        return canonicalize_sequence(
            self.pts,
            self.mask,
            self.ts_ms,
            pose_xyz=self.pose_xyz,
            pose_vis=self.pose_vis,
            pose_indices=self.pose_indices,
            meta=dict(self.source),
        )


class SegmentBridge:
    def __init__(
        self,
        *,
        pre_context_frames: int = 4,
        post_context_frames: int = 4,
        max_buffer_frames: int = 4096,
    ) -> None:
        self.pre_context_frames = max(0, int(pre_context_frames))
        self.post_context_frames = max(0, int(post_context_frames))
        self.max_buffer_frames = max(128, int(max_buffer_frames))
        self._pts: List[np.ndarray] = []
        self._mask: List[np.ndarray] = []
        self._ts_ms: List[float] = []
        self._pose_xyz: List[np.ndarray | None] = []
        self._pose_vis: List[np.ndarray | None] = []
        self._pose_indices: tuple[int, ...] | None = None
        self._base_index = 0
        self._protected_from_global: int | None = None

    @property
    def total_frames(self) -> int:
        return self._base_index + len(self._pts)

    @property
    def protected_from_global(self) -> int | None:
        return self._protected_from_global

    def append_frame(
        self,
        pts: np.ndarray,
        mask: np.ndarray,
        ts_ms: float,
        *,
        pose_xyz: np.ndarray | None = None,
        pose_vis: np.ndarray | None = None,
        pose_indices: Sequence[int] | None = None,
    ) -> None:
        self._pts.append(np.asarray(pts, dtype=np.float32))
        self._mask.append(np.asarray(mask, dtype=np.float32))
        self._ts_ms.append(float(ts_ms))
        if pose_xyz is not None:
            self._pose_xyz.append(np.asarray(pose_xyz, dtype=np.float32))
            if pose_vis is None:
                self._pose_vis.append(np.isfinite(np.asarray(pose_xyz, dtype=np.float32)).all(axis=1).astype(np.float32))
            else:
                self._pose_vis.append(np.asarray(pose_vis, dtype=np.float32))
            if pose_indices is not None:
                pose_idx = tuple(int(x) for x in pose_indices)
                if self._pose_indices is None:
                    self._pose_indices = pose_idx
                elif self._pose_indices != pose_idx:
                    raise ValueError("SegmentBridge pose_indices changed across frames")
        else:
            self._pose_xyz.append(None)
            self._pose_vis.append(None)
        self._trim_old_frames()

    def set_protected_from_global(self, frame_idx: int | None) -> None:
        self._protected_from_global = None if frame_idx is None else max(0, int(frame_idx))
        self._trim_old_frames()

    def _trim_old_frames(self, keep_from_global: int | None = None) -> None:
        if keep_from_global is None:
            keep_from_global = max(0, self.total_frames - self.max_buffer_frames)
        if self._protected_from_global is not None:
            keep_from_global = min(int(keep_from_global), int(self._protected_from_global))
        keep_from_global = max(int(self._base_index), int(keep_from_global))
        while self._pts and self._base_index < keep_from_global:
            self._pts.pop(0)
            self._mask.pop(0)
            self._ts_ms.pop(0)
            self._pose_xyz.pop(0)
            self._pose_vis.pop(0)
            self._base_index += 1

    def clip_for_segment(self, segment: Dict[str, Any]) -> SegmentClip:
        seg_id = int(segment.get("segment_id", 0))
        start = int(segment.get("start_frame", 0))
        end_excl = int(segment.get("end_frame_exclusive", 0))
        clip_start = max(self._base_index, start - self.pre_context_frames)
        clip_end = min(self.total_frames, end_excl + self.post_context_frames)
        if clip_end <= clip_start:
            raise RuntimeError(f"Segment clip is empty: segment_id={seg_id}")
        local_start = clip_start - self._base_index
        local_end = clip_end - self._base_index
        pts = np.stack(self._pts[local_start:local_end], axis=0)
        mask = np.stack(self._mask[local_start:local_end], axis=0)
        ts_ms = np.asarray(self._ts_ms[local_start:local_end], dtype=np.float32)
        pose_slice = self._pose_xyz[local_start:local_end]
        vis_slice = self._pose_vis[local_start:local_end]
        pose_xyz = None
        pose_vis = None
        if self._pose_indices is not None and pose_slice and all(item is not None for item in pose_slice):
            pose_xyz = np.stack([np.asarray(item, dtype=np.float32) for item in pose_slice if item is not None], axis=0)
            pose_vis = np.stack([np.asarray(item, dtype=np.float32) for item in vis_slice if item is not None], axis=0)
        source = {
            "segment_id": seg_id,
            "segment_start_frame": start,
            "segment_end_frame_exclusive": end_excl,
            "clip_start_frame": clip_start,
            "clip_end_frame_exclusive": clip_end,
        }
        return SegmentClip(
            segment_id=seg_id,
            start_frame=clip_start,
            end_frame_exclusive=clip_end,
            start_time_ms=float(ts_ms[0]),
            end_time_ms=float(ts_ms[-1]) if ts_ms.size > 0 else 0.0,
            pts=pts,
            mask=mask,
            ts_ms=ts_ms,
            pose_xyz=pose_xyz,
            pose_vis=pose_vis,
            pose_indices=self._pose_indices,
            source=source,
        )

    def clips_for_segments(self, segments: Sequence[Dict[str, Any]]) -> List[SegmentClip]:
        out = [self.clip_for_segment(seg) for seg in segments]
        if out:
            self._trim_old_frames(keep_from_global=max(0, int(out[-1].end_frame_exclusive) - self.pre_context_frames))
        return out
