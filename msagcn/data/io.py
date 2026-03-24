from __future__ import annotations

import json
import mmap
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import orjson as _fastjson

    def _loads_bytes(b: bytes):
        return _fastjson.loads(b)
except Exception:  # pragma: no cover - optional dependency

    def _loads_bytes(b: bytes):
        return json.loads(b.decode("utf-8"))


def read_video_file_nocache(path_str: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read per-video JSON -> (frames, meta), no global cache."""
    p = Path(path_str)
    with p.open("rb") as f:
        blob = _loads_bytes(f.read())
    if isinstance(blob, dict) and "frames" in blob:
        return blob["frames"], blob.get("meta", {})
    # legacy: file is a list of frames
    return blob, {}


def frames_from_combined(skel: Dict[str, Any], vid: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if "videos" in skel:
        v = skel["videos"].get(vid, {})
        return v.get("frames", []), v.get("meta", {})
    return skel.get(vid, []), {}


def build_resolved_video_map(base_dir: str | Path, prefer_pp: bool = True) -> Dict[str, Path]:
    root = Path(base_dir)
    selected: Dict[str, Path] = {}
    if not root.is_dir():
        return selected
    for path in sorted(root.glob("*.json")):
        stem = path.stem
        is_pp = stem.endswith("_pp")
        vid = stem[:-3] if is_pp else stem
        prev = selected.get(vid)
        if prev is None:
            selected[vid] = path
            continue
        prev_is_pp = prev.stem.endswith("_pp")
        if prefer_pp:
            if is_pp and not prev_is_pp:
                selected[vid] = path
        else:
            if (not is_pp) and prev_is_pp:
                selected[vid] = path
    return selected


def _extract_meta_from_raw(raw: bytes) -> Dict[str, Any]:
    blob = _loads_bytes(raw)
    if isinstance(blob, dict) and "frames" in blob:
        return blob.get("meta", {})
    return {}


_HAND_JOINTS = 21
_POSE_ABS_COUNT = 25
_FLOAT32_BYTES = 4
_DECODED_RECORD_FLOATS = 1 + (_HAND_JOINTS * 3) + (_HAND_JOINTS * 3) + 1 + 1 + (_POSE_ABS_COUNT * 3) + _POSE_ABS_COUNT
_DECODED_RECORD_BYTES = _DECODED_RECORD_FLOATS * _FLOAT32_BYTES


@dataclass(frozen=True)
class DecodedVideoArrays:
    ts: np.ndarray
    left_xyz: np.ndarray
    right_xyz: np.ndarray
    left_score: np.ndarray
    right_score: np.ndarray
    pose_xyz: np.ndarray
    pose_vis: np.ndarray
    meta: Dict[str, Any]


def _pose_abs_order(meta: Dict[str, Any], pose_len: int) -> List[int]:
    order = meta.get("pose_indices", None)
    if order == "all":
        return list(range(pose_len))
    if isinstance(order, list):
        out: List[int] = []
        for value in order[:pose_len]:
            try:
                out.append(int(value))
            except Exception:
                out.append(-1)
        return out
    if pose_len == _POSE_ABS_COUNT:
        return list(range(_POSE_ABS_COUNT))
    if pose_len == 11:
        return [0, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24]
    if pose_len == 9:
        return [0, 9, 10, 11, 12, 13, 14, 23, 24]
    return list(range(pose_len))


def _point_xyz(point: Any) -> tuple[float, float, float]:
    if isinstance(point, dict):
        return (
            float(point.get("x", 0.0)),
            float(point.get("y", 0.0)),
            float(point.get("z", 0.0)),
        )
    return 0.0, 0.0, 0.0


def _decode_video_arrays_from_raw(raw: bytes) -> tuple[DecodedVideoArrays, Dict[str, Any]]:
    blob = _loads_bytes(raw)
    if isinstance(blob, dict) and "frames" in blob:
        frames = blob.get("frames", [])
        meta = blob.get("meta", {})
    else:
        frames = blob
        meta = {}
    n = len(frames)
    ts = np.zeros((n,), dtype=np.float32)
    left_xyz = np.zeros((n, _HAND_JOINTS, 3), dtype=np.float32)
    right_xyz = np.zeros((n, _HAND_JOINTS, 3), dtype=np.float32)
    left_score = np.full((n,), -1.0, dtype=np.float32)
    right_score = np.full((n,), -1.0, dtype=np.float32)
    pose_xyz = np.zeros((n, _POSE_ABS_COUNT, 3), dtype=np.float32)
    pose_vis = np.zeros((n, _POSE_ABS_COUNT), dtype=np.float32)

    for i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        ts[i] = float(frame.get("ts", 0.0) or 0.0)

        left = frame.get("hand 1")
        if isinstance(left, list):
            limit = min(len(left), _HAND_JOINTS)
            for j in range(limit):
                left_xyz[i, j] = _point_xyz(left[j])
        left_raw_score = frame.get("hand 1_score")
        if left_raw_score is not None:
            left_score[i] = float(left_raw_score)

        right = frame.get("hand 2")
        if isinstance(right, list):
            limit = min(len(right), _HAND_JOINTS)
            for j in range(limit):
                right_xyz[i, j] = _point_xyz(right[j])
        right_raw_score = frame.get("hand 2_score")
        if right_raw_score is not None:
            right_score[i] = float(right_raw_score)

        pose = frame.get("pose")
        if isinstance(pose, list) and pose:
            pose_order = _pose_abs_order(meta if isinstance(meta, dict) else {}, len(pose))
            pose_vis_row = frame.get("pose_vis")
            for src_idx, abs_idx in enumerate(pose_order):
                if not (0 <= abs_idx < _POSE_ABS_COUNT):
                    continue
                pose_xyz[i, abs_idx] = _point_xyz(pose[src_idx])
                if isinstance(pose_vis_row, list) and src_idx < len(pose_vis_row):
                    pose_vis[i, abs_idx] = float(pose_vis_row[src_idx] or 0.0)

    meta_out = dict(meta) if isinstance(meta, dict) else {}
    meta_out["coords"] = str(meta_out.get("coords", "image"))
    try:
        meta_out["fps"] = float(meta_out.get("fps", 0.0) or 0.0)
    except Exception:
        meta_out["fps"] = 0.0
    return (
        DecodedVideoArrays(
            ts=ts,
            left_xyz=left_xyz,
            right_xyz=right_xyz,
            left_score=left_score,
            right_score=right_score,
            pose_xyz=pose_xyz,
            pose_vis=pose_vis,
            meta=meta_out,
        ),
        meta_out,
    )


def _decoded_arrays_to_bytes(arrays: DecodedVideoArrays) -> bytes:
    return b"".join(
        (
            arrays.ts.astype(np.float32, copy=False).tobytes(order="C"),
            arrays.left_xyz.astype(np.float32, copy=False).tobytes(order="C"),
            arrays.right_xyz.astype(np.float32, copy=False).tobytes(order="C"),
            arrays.left_score.astype(np.float32, copy=False).tobytes(order="C"),
            arrays.right_score.astype(np.float32, copy=False).tobytes(order="C"),
            arrays.pose_xyz.astype(np.float32, copy=False).tobytes(order="C"),
            arrays.pose_vis.astype(np.float32, copy=False).tobytes(order="C"),
        )
    )


class PackedVideoStore:
    """
    Sidecar packed store for per-video JSON files:
    - `data.bin` keeps the original JSON bytes contiguously.
    - `index.json` stores offset/length/meta/source stat per video.

    Each worker re-opens the mmap locally, so the OS page cache can be shared
    across workers without duplicating a Python-level cache.
    """

    VERSION = 1
    INDEX_NAME = "index.json"
    DATA_NAME = "data.bin"

    def __init__(self, cache_dir: str | Path, payload: Dict[str, Any]):
        self.cache_dir = Path(cache_dir)
        self.index_path = self.cache_dir / self.INDEX_NAME
        self.data_path = self.cache_dir / self.DATA_NAME
        self.payload = payload
        self.entries: Dict[str, Dict[str, Any]] = dict(payload.get("entries", {}))
        self.source_dir = Path(payload.get("source_dir", ""))
        self.prefer_pp = bool(payload.get("prefer_pp", True))
        self._fh = None
        self._mm = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_fh"] = None
        state["_mm"] = None
        return state

    def __del__(self):
        self.close()

    def close(self) -> None:
        try:
            if self._mm is not None:
                self._mm.close()
        finally:
            self._mm = None
            if self._fh is not None:
                self._fh.close()
            self._fh = None

    def _ensure_mmap(self) -> None:
        if self._mm is not None:
            return
        self._fh = self.data_path.open("rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def has_video(self, vid: str) -> bool:
        return str(vid) in self.entries

    def get_meta(self, vid: str) -> Dict[str, Any]:
        entry = self.entries.get(str(vid))
        if not entry:
            return {}
        meta = entry.get("meta", {})
        return dict(meta) if isinstance(meta, dict) else {}

    def read_video(self, vid: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        entry = self.entries.get(str(vid))
        if entry is None:
            raise KeyError(f"Video {vid} not found in packed skeleton cache")
        self._ensure_mmap()
        offset = int(entry["offset"])
        length = int(entry["length"])
        raw = self._mm[offset : offset + length]
        blob = _loads_bytes(raw)
        if isinstance(blob, dict) and "frames" in blob:
            return blob["frames"], blob.get("meta", {})
        return blob, {}

    @classmethod
    def open_or_build(
        cls,
        *,
        source_dir: str | Path,
        cache_dir: str | Path,
        prefer_pp: bool = True,
        rebuild: bool = False,
        vids: List[str] | None = None,
        context_label: str = "",
    ) -> "PackedVideoStore":
        cache_dir = Path(cache_dir)
        source_dir = Path(source_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        index_path = cache_dir / cls.INDEX_NAME
        data_path = cache_dir / cls.DATA_NAME
        selected = build_resolved_video_map(source_dir, prefer_pp=prefer_pp)

        def _load_existing() -> Dict[str, Any] | None:
            if rebuild or (not index_path.exists()) or (not data_path.exists()):
                return None
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception:
                return None
            if int(payload.get("version", -1)) != cls.VERSION:
                return None
            if Path(payload.get("source_dir", "")) != source_dir:
                return None
            if bool(payload.get("prefer_pp", True)) != bool(prefer_pp):
                return None
            entries = payload.get("entries", {})
            if set(entries.keys()) != set(selected.keys()):
                return None
            for vid, path in selected.items():
                entry = entries.get(vid)
                if not isinstance(entry, dict):
                    return None
                try:
                    st = path.stat()
                except OSError:
                    return None
                rel_path = os.path.relpath(str(path), str(source_dir))
                if entry.get("source_relpath") != rel_path:
                    return None
                if int(entry.get("source_size", -1)) != int(st.st_size):
                    return None
                if int(entry.get("source_mtime_ns", -1)) != int(st.st_mtime_ns):
                    return None
            return payload

        payload = _load_existing()
        prefix = f"[{context_label}] " if context_label else ""
        if payload is None:
            print(f"{prefix}Building packed skeleton cache in {cache_dir} ...")
            tmp_data = cache_dir / f"{cls.DATA_NAME}.tmp"
            tmp_index = cache_dir / f"{cls.INDEX_NAME}.tmp"
            entries: Dict[str, Dict[str, Any]] = {}
            with tmp_data.open("wb") as fout:
                for vid, path in sorted(selected.items()):
                    raw = path.read_bytes()
                    offset = fout.tell()
                    fout.write(raw)
                    st = path.stat()
                    entries[vid] = {
                        "offset": int(offset),
                        "length": int(len(raw)),
                        "meta": _extract_meta_from_raw(raw),
                        "source_relpath": os.path.relpath(str(path), str(source_dir)),
                        "source_size": int(st.st_size),
                        "source_mtime_ns": int(st.st_mtime_ns),
                    }
            payload = {
                "version": cls.VERSION,
                "source_dir": str(source_dir),
                "prefer_pp": bool(prefer_pp),
                "entries": entries,
            }
            tmp_index.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp_data, data_path)
            os.replace(tmp_index, index_path)
            print(f"{prefix}Packed skeleton cache ready: {len(entries)} videos | data={data_path}")
        else:
            print(f"{prefix}Using packed skeleton cache: {cache_dir}")
        if vids is not None:
            subset_entries = {str(vid): payload["entries"][str(vid)] for vid in vids if str(vid) in payload["entries"]}
            payload = dict(payload)
            payload["entries"] = subset_entries
        return cls(cache_dir, payload)


class DecodedVideoStore:
    """
    Sidecar store for already-decoded skeletal arrays.

    Unlike `PackedVideoStore`, this removes per-sample JSON parsing and most frame-dict
    traversal from `Dataset.__getitem__`. The payload is stored as contiguous float32
    arrays inside a single mmap-backed `data.bin`, while `index.json` keeps offsets and
    source-file metadata for invalidation.
    """

    VERSION = 1
    INDEX_NAME = "index.json"
    DATA_NAME = "data.bin"

    def __init__(self, cache_dir: str | Path, payload: Dict[str, Any]):
        self.cache_dir = Path(cache_dir)
        self.index_path = self.cache_dir / self.INDEX_NAME
        self.data_path = self.cache_dir / self.DATA_NAME
        self.payload = payload
        self.entries: Dict[str, Dict[str, Any]] = dict(payload.get("entries", {}))
        self.source_dir = Path(payload.get("source_dir", ""))
        self.prefer_pp = bool(payload.get("prefer_pp", True))
        self._fh = None
        self._mm = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_fh"] = None
        state["_mm"] = None
        return state

    def __del__(self):
        self.close()

    def close(self) -> None:
        try:
            if self._mm is not None:
                self._mm.close()
        finally:
            self._mm = None
            if self._fh is not None:
                self._fh.close()
            self._fh = None

    def _ensure_mmap(self) -> None:
        if self._mm is not None:
            return
        self._fh = self.data_path.open("rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def has_video(self, vid: str) -> bool:
        return str(vid) in self.entries

    def get_meta(self, vid: str) -> Dict[str, Any]:
        entry = self.entries.get(str(vid))
        if not entry:
            return {}
        meta = entry.get("meta", {})
        return dict(meta) if isinstance(meta, dict) else {}

    def read_video(self, vid: str) -> DecodedVideoArrays:
        entry = self.entries.get(str(vid))
        if entry is None:
            raise KeyError(f"Video {vid} not found in decoded skeleton cache")
        self._ensure_mmap()
        frames = int(entry["frames"])
        offset = int(entry["offset"])
        base = offset

        ts = np.frombuffer(self._mm, dtype=np.float32, count=frames, offset=base)
        base += frames * _FLOAT32_BYTES

        left_xyz = np.frombuffer(self._mm, dtype=np.float32, count=frames * _HAND_JOINTS * 3, offset=base).reshape(
            frames, _HAND_JOINTS, 3
        )
        base += frames * _HAND_JOINTS * 3 * _FLOAT32_BYTES

        right_xyz = np.frombuffer(self._mm, dtype=np.float32, count=frames * _HAND_JOINTS * 3, offset=base).reshape(
            frames, _HAND_JOINTS, 3
        )
        base += frames * _HAND_JOINTS * 3 * _FLOAT32_BYTES

        left_score = np.frombuffer(self._mm, dtype=np.float32, count=frames, offset=base)
        base += frames * _FLOAT32_BYTES

        right_score = np.frombuffer(self._mm, dtype=np.float32, count=frames, offset=base)
        base += frames * _FLOAT32_BYTES

        pose_xyz = np.frombuffer(self._mm, dtype=np.float32, count=frames * _POSE_ABS_COUNT * 3, offset=base).reshape(
            frames, _POSE_ABS_COUNT, 3
        )
        base += frames * _POSE_ABS_COUNT * 3 * _FLOAT32_BYTES

        pose_vis = np.frombuffer(self._mm, dtype=np.float32, count=frames * _POSE_ABS_COUNT, offset=base).reshape(
            frames, _POSE_ABS_COUNT
        )

        return DecodedVideoArrays(
            ts=ts,
            left_xyz=left_xyz,
            right_xyz=right_xyz,
            left_score=left_score,
            right_score=right_score,
            pose_xyz=pose_xyz,
            pose_vis=pose_vis,
            meta=self.get_meta(vid),
        )

    @classmethod
    def open_or_build(
        cls,
        *,
        source_dir: str | Path,
        cache_dir: str | Path,
        prefer_pp: bool = True,
        rebuild: bool = False,
        vids: List[str] | None = None,
        context_label: str = "",
    ) -> "DecodedVideoStore":
        cache_dir = Path(cache_dir)
        source_dir = Path(source_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        index_path = cache_dir / cls.INDEX_NAME
        data_path = cache_dir / cls.DATA_NAME
        selected = build_resolved_video_map(source_dir, prefer_pp=prefer_pp)

        def _load_existing() -> Dict[str, Any] | None:
            if rebuild or (not index_path.exists()) or (not data_path.exists()):
                return None
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception:
                return None
            if int(payload.get("version", -1)) != cls.VERSION:
                return None
            if Path(payload.get("source_dir", "")) != source_dir:
                return None
            if bool(payload.get("prefer_pp", True)) != bool(prefer_pp):
                return None
            entries = payload.get("entries", {})
            if set(entries.keys()) != set(selected.keys()):
                return None
            for vid, path in selected.items():
                entry = entries.get(vid)
                if not isinstance(entry, dict):
                    return None
                try:
                    st = path.stat()
                except OSError:
                    return None
                rel_path = os.path.relpath(str(path), str(source_dir))
                if entry.get("source_relpath") != rel_path:
                    return None
                if int(entry.get("source_size", -1)) != int(st.st_size):
                    return None
                if int(entry.get("source_mtime_ns", -1)) != int(st.st_mtime_ns):
                    return None
            return payload

        payload = _load_existing()
        prefix = f"[{context_label}] " if context_label else ""
        if payload is None:
            print(f"{prefix}Building decoded skeleton cache in {cache_dir} ...")
            tmp_data = cache_dir / f"{cls.DATA_NAME}.tmp"
            tmp_index = cache_dir / f"{cls.INDEX_NAME}.tmp"
            entries: Dict[str, Dict[str, Any]] = {}
            with tmp_data.open("wb") as fout:
                for vid, path in sorted(selected.items()):
                    raw = path.read_bytes()
                    arrays, meta = _decode_video_arrays_from_raw(raw)
                    payload_bytes = _decoded_arrays_to_bytes(arrays)
                    offset = fout.tell()
                    fout.write(payload_bytes)
                    st = path.stat()
                    entries[vid] = {
                        "offset": int(offset),
                        "frames": int(arrays.ts.shape[0]),
                        "length": int(len(payload_bytes)),
                        "meta": meta,
                        "source_relpath": os.path.relpath(str(path), str(source_dir)),
                        "source_size": int(st.st_size),
                        "source_mtime_ns": int(st.st_mtime_ns),
                    }
            payload = {
                "version": cls.VERSION,
                "source_dir": str(source_dir),
                "prefer_pp": bool(prefer_pp),
                "entries": entries,
            }
            tmp_index.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp_data, data_path)
            os.replace(tmp_index, index_path)
            data_size_gb = float(data_path.stat().st_size) / (1024.0 ** 3)
            print(f"{prefix}Decoded skeleton cache ready: {len(entries)} videos | data={data_path} | size={data_size_gb:.2f} GB")
        else:
            print(f"{prefix}Using decoded skeleton cache: {cache_dir}")
        if vids is not None:
            subset_entries = {str(vid): payload["entries"][str(vid)] for vid in vids if str(vid) in payload["entries"]}
            payload = dict(payload)
            payload["entries"] = subset_entries
        return cls(cache_dir, payload)
