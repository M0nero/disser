from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from msagcn.data.io import read_video_file_nocache


HAND_JOINTS = 21
NUM_HAND_NODES = 42


@dataclass(frozen=True)
class RuntimeSkeletonSpec:
    version: int = 1
    joint_layout: str = "left21_right21"
    num_joints: int = NUM_HAND_NODES
    coord_dim: int = 3
    mask_dim: int = 1
    coords: str = "image"
    handedness_policy: str = "label_then_score"
    missing_hand_policy: str = "zero_fill_mask0"
    pose_sidecar: bool = True
    pose_layout: str = "mediapipe_pose_optional"


CANONICAL_SKELETON_SPEC = RuntimeSkeletonSpec()


@dataclass
class CanonicalSkeletonSequence:
    pts: np.ndarray
    mask: np.ndarray
    ts_ms: np.ndarray
    meta: Dict[str, Any]
    pose_xyz: np.ndarray | None = None
    pose_vis: np.ndarray | None = None
    pose_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        self.pts = np.asarray(self.pts, dtype=np.float32)
        self.mask = np.asarray(self.mask, dtype=np.float32)
        self.ts_ms = np.asarray(self.ts_ms, dtype=np.float32).reshape(-1)
        self.meta = dict(self.meta or {})
        if self.pts.ndim != 3 or self.pts.shape[1:] != (NUM_HAND_NODES, 3):
            raise ValueError(f"pts must be (T,{NUM_HAND_NODES},3), got {tuple(self.pts.shape)}")
        if self.mask.ndim != 3 or self.mask.shape[1:] != (NUM_HAND_NODES, 1):
            raise ValueError(f"mask must be (T,{NUM_HAND_NODES},1), got {tuple(self.mask.shape)}")
        if self.mask.shape[0] != self.pts.shape[0]:
            raise ValueError("pts/mask length mismatch")
        if self.ts_ms.shape[0] not in (0, self.pts.shape[0]):
            raise ValueError("ts_ms length mismatch")
        if self.ts_ms.shape[0] == 0:
            self.ts_ms = np.arange(self.pts.shape[0], dtype=np.float32) * float(self.meta.get("frame_dt_ms", 33.3333))
        if self.pose_xyz is None:
            self.pose_vis = None
            self.pose_indices = None
        else:
            self.pose_xyz = np.asarray(self.pose_xyz, dtype=np.float32)
            if self.pose_xyz.ndim != 3 or self.pose_xyz.shape[0] != self.pts.shape[0] or self.pose_xyz.shape[2] != 3:
                raise ValueError(f"pose_xyz must be (T,P,3), got {tuple(self.pose_xyz.shape)}")
            if self.pose_vis is None:
                self.pose_vis = np.isfinite(self.pose_xyz).all(axis=2).astype(np.float32)
            else:
                self.pose_vis = np.asarray(self.pose_vis, dtype=np.float32)
                if self.pose_vis.ndim != 2 or self.pose_vis.shape != self.pose_xyz.shape[:2]:
                    raise ValueError(f"pose_vis must be (T,P), got {tuple(self.pose_vis.shape)}")
                self.pose_vis = np.nan_to_num(self.pose_vis, nan=0.0, posinf=0.0, neginf=0.0)
            if self.pose_indices is None:
                self.pose_indices = tuple(int(i) for i in range(int(self.pose_xyz.shape[1])))
            else:
                self.pose_indices = tuple(int(i) for i in self.pose_indices)
            if len(self.pose_indices) != int(self.pose_xyz.shape[1]):
                raise ValueError("pose_indices length mismatch")
            self.pose_xyz = np.nan_to_num(self.pose_xyz, nan=0.0, posinf=0.0, neginf=0.0)
            # Pose visibility is a confidence sidecar, not a binary mask. Keep
            # coordinates intact and let downstream runtimes threshold
            # `pose_vis` explicitly when they build model inputs.

    @property
    def length(self) -> int:
        return int(self.pts.shape[0])

    def to_manifest_dict(self) -> Dict[str, Any]:
        return {
            "spec": asdict(CANONICAL_SKELETON_SPEC),
            "length": int(self.length),
            "meta": dict(self.meta),
            "pose_joints": int(self.pose_xyz.shape[1]) if self.pose_xyz is not None else 0,
            "pose_indices": list(self.pose_indices or []),
        }


def _sanitize_pts_mask(pts: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 3 or pts.shape[1:] != (NUM_HAND_NODES, 3):
        raise ValueError(f"pts must be (T,{NUM_HAND_NODES},3), got {tuple(pts.shape)}")
    if mask is None:
        mask_arr = np.isfinite(pts).all(axis=2, keepdims=True).astype(np.float32)
    else:
        mask_arr = np.asarray(mask, dtype=np.float32)
        if mask_arr.ndim == 2:
            mask_arr = mask_arr[..., None]
        if mask_arr.ndim != 3 or mask_arr.shape[1:] != (NUM_HAND_NODES, 1):
            raise ValueError(f"mask must be (T,{NUM_HAND_NODES},1), got {tuple(mask_arr.shape)}")
        if mask_arr.shape[0] != pts.shape[0]:
            raise ValueError("pts/mask length mismatch")
        mask_arr = (mask_arr > 0.0).astype(np.float32)
    pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)
    pts *= mask_arr
    return pts.astype(np.float32, copy=False), mask_arr.astype(np.float32, copy=False)


def canonicalize_sequence(
    pts: np.ndarray,
    mask: Optional[np.ndarray] = None,
    ts_ms: Optional[Sequence[float]] = None,
    *,
    pose_xyz: Optional[np.ndarray] = None,
    pose_vis: Optional[np.ndarray] = None,
    pose_indices: Optional[Sequence[int]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> CanonicalSkeletonSequence:
    pts_arr, mask_arr = _sanitize_pts_mask(pts, mask)
    if ts_ms is None:
        ts_arr = np.zeros((pts_arr.shape[0],), dtype=np.float32)
    else:
        ts_arr = np.asarray(ts_ms, dtype=np.float32).reshape(-1)
        if ts_arr.shape[0] != pts_arr.shape[0]:
            raise ValueError("ts_ms length mismatch")
    meta_out = dict(meta or {})
    if pts_arr.shape[0] > 1 and ts_arr.shape[0] == pts_arr.shape[0]:
        diffs = np.diff(ts_arr)
        finite = diffs[np.isfinite(diffs) & (diffs > 0)]
        if finite.size > 0:
            meta_out.setdefault("frame_dt_ms", float(np.median(finite)))
    pose_xyz_arr: np.ndarray | None = None
    pose_vis_arr: np.ndarray | None = None
    pose_idx_out: tuple[int, ...] | None = None
    if pose_xyz is not None:
        pose_xyz_arr = np.asarray(pose_xyz, dtype=np.float32)
        if pose_xyz_arr.ndim != 3 or pose_xyz_arr.shape[0] != pts_arr.shape[0] or pose_xyz_arr.shape[2] != 3:
            raise ValueError(f"pose_xyz must be (T,P,3), got {tuple(pose_xyz_arr.shape)}")
        if pose_vis is None:
            pose_vis_arr = np.isfinite(pose_xyz_arr).all(axis=2).astype(np.float32)
        else:
            pose_vis_arr = np.asarray(pose_vis, dtype=np.float32)
            if pose_vis_arr.ndim != 2 or pose_vis_arr.shape != pose_xyz_arr.shape[:2]:
                raise ValueError(f"pose_vis must be (T,P), got {tuple(pose_vis_arr.shape)}")
            pose_vis_arr = np.nan_to_num(pose_vis_arr, nan=0.0, posinf=0.0, neginf=0.0)
        if pose_indices is None:
            pose_idx_out = tuple(int(i) for i in range(int(pose_xyz_arr.shape[1])))
        else:
            pose_idx_out = tuple(int(i) for i in pose_indices)
        if len(pose_idx_out) != int(pose_xyz_arr.shape[1]):
            raise ValueError("pose_indices length mismatch")
        pose_xyz_arr = np.nan_to_num(pose_xyz_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return CanonicalSkeletonSequence(
        pts=pts_arr,
        mask=mask_arr,
        ts_ms=ts_arr,
        meta=meta_out,
        pose_xyz=pose_xyz_arr,
        pose_vis=pose_vis_arr,
        pose_indices=pose_idx_out,
    )


def save_skeleton_sequence_npz(path: str | Path, seq: CanonicalSkeletonSequence) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "pts": seq.pts.astype(np.float32, copy=False),
        "mask": seq.mask.astype(np.float32, copy=False),
        "ts_ms": seq.ts_ms.astype(np.float32, copy=False),
        "meta_json": np.asarray(json.dumps(seq.meta, ensure_ascii=False), dtype=np.unicode_),
    }
    if seq.pose_xyz is not None:
        payload["pose_xyz"] = seq.pose_xyz.astype(np.float32, copy=False)
        payload["pose_vis"] = seq.pose_vis.astype(np.float32, copy=False)
        payload["pose_indices"] = np.asarray(list(seq.pose_indices or []), dtype=np.int32)
    np.savez_compressed(out, **payload)
    return out


def _point_xyz(point: Any) -> tuple[float, float, float]:
    if isinstance(point, dict):
        return (
            float(point.get("x", 0.0)),
            float(point.get("y", 0.0)),
            float(point.get("z", 0.0)),
        )
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return float(point[0]), float(point[1]), float(point[2] if len(point) >= 3 else 0.0)
    return 0.0, 0.0, 0.0


def _points_to_array(points: Any) -> Optional[np.ndarray]:
    if points is None:
        return None
    if isinstance(points, np.ndarray):
        arr = np.asarray(points, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None
        if arr.shape[1] == 2:
            arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=arr.dtype)], axis=1)
        return arr[:, :3]
    if not isinstance(points, list):
        return None
    arr = np.full((len(points), 3), np.nan, dtype=np.float32)
    for idx, point in enumerate(points):
        try:
            arr[idx] = np.asarray(_point_xyz(point), dtype=np.float32)
        except Exception:
            continue
    return arr


def _pick_points_from_hand_obj(hand_obj: Any) -> Any:
    if isinstance(hand_obj, dict):
        for key in ("landmarks", "hand_landmarks", "keypoints", "points", "pts", "xyz"):
            if key in hand_obj:
                return hand_obj[key]
        if "x" in hand_obj or "X" in hand_obj:
            return [hand_obj]
    return hand_obj


def _to_hand_array(hand_obj: Any) -> Optional[np.ndarray]:
    arr = _points_to_array(_pick_points_from_hand_obj(hand_obj))
    if arr is None:
        return None
    if arr.shape[0] < HAND_JOINTS:
        out = np.full((HAND_JOINTS, 3), np.nan, dtype=np.float32)
        out[: arr.shape[0]] = arr
        return out
    if arr.shape[0] > HAND_JOINTS:
        return arr[:HAND_JOINTS].astype(np.float32, copy=False)
    return arr.astype(np.float32, copy=False)


def _hand_side_from_obj(hand_obj: Any) -> str:
    if not isinstance(hand_obj, dict):
        return ""
    for key in ("handedness", "handedness_label", "label", "type", "side", "hand"):
        val = hand_obj.get(key)
        if isinstance(val, dict):
            val = val.get("label") or val.get("type")
        if val is None:
            continue
        s = str(val).strip().lower()
        if "left" in s:
            return "left"
        if "right" in s:
            return "right"
    return ""


def _extract_hand_landmarks(frame: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not isinstance(frame, dict):
        return None, None
    if ("hand 1" in frame) or ("hand 2" in frame):
        return _to_hand_array(frame.get("hand 1")), _to_hand_array(frame.get("hand 2"))
    if ("left_hand" in frame) or ("right_hand" in frame):
        return _to_hand_array(frame.get("left_hand")), _to_hand_array(frame.get("right_hand"))

    hands = frame.get("hands")
    if isinstance(hands, dict):
        return _to_hand_array(hands.get("left") or hands.get("left_hand")), _to_hand_array(hands.get("right") or hands.get("right_hand"))
    if isinstance(hands, list):
        left = right = None
        for hand in hands:
            side = _hand_side_from_obj(hand)
            pts = _to_hand_array(hand)
            if side == "left" and left is None:
                left = pts
            elif side == "right" and right is None:
                right = pts
            elif left is None:
                left = pts
            elif right is None:
                right = pts
        return left, right

    for key in ("hand_landmarks", "landmarks"):
        if key not in frame:
            continue
        val = frame.get(key)
        if isinstance(val, dict):
            return _to_hand_array(val.get("left") or val.get("left_hand")), _to_hand_array(val.get("right") or val.get("right_hand"))
        if isinstance(val, list):
            if len(val) == 2 and all(isinstance(v, list) for v in val):
                return _to_hand_array(val[0]), _to_hand_array(val[1])
            if len(val) == 1 and isinstance(val[0], list):
                return _to_hand_array(val[0]), None
            if len(val) == HAND_JOINTS:
                return _to_hand_array(val), None
    return None, None


def _extract_pose_landmarks(frame: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[int]]]:
    if not isinstance(frame, dict):
        return None, None, None
    pose_obj = None
    pose_vis = frame.get("pose_vis")
    pose_indices = frame.get("pose_indices")
    for key in ("pose_landmarks", "pose", "pose_xyz"):
        if key in frame:
            pose_obj = frame.get(key)
            break
    if pose_obj is None:
        return None, None, None
    if isinstance(pose_obj, dict):
        pose_vis = pose_obj.get("visibility") or pose_obj.get("pose_vis") or pose_vis
        pose_indices = pose_obj.get("indices") or pose_obj.get("pose_indices") or pose_indices
        pose_obj = pose_obj.get("landmarks") or pose_obj.get("points") or pose_obj.get("xyz") or pose_obj
    arr = _points_to_array(pose_obj)
    if arr is None:
        return None, None, None
    if pose_vis is None:
        vis_arr = np.ones((arr.shape[0],), dtype=np.float32)
    else:
        vis_arr = np.asarray(pose_vis, dtype=np.float32).reshape(-1)
        if vis_arr.shape[0] < arr.shape[0]:
            pad = np.zeros((arr.shape[0] - vis_arr.shape[0],), dtype=np.float32)
            vis_arr = np.concatenate([vis_arr, pad], axis=0)
        vis_arr = vis_arr[: arr.shape[0]]
    if pose_indices is None:
        idx_out = list(range(int(arr.shape[0])))
    else:
        idx_out = [int(x) for x in pose_indices][: int(arr.shape[0])]
        if len(idx_out) < int(arr.shape[0]):
            idx_out.extend(list(range(len(idx_out), int(arr.shape[0]))))
    return arr.astype(np.float32, copy=False), vis_arr.astype(np.float32, copy=False), idx_out


def combine_hands(left: Optional[np.ndarray], right: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.full((NUM_HAND_NODES, 3), np.nan, dtype=np.float32)
    mask = np.zeros((NUM_HAND_NODES, 1), dtype=np.float32)
    if left is not None:
        pts[:HAND_JOINTS] = left
        mask[:HAND_JOINTS, 0] = np.isfinite(left).all(axis=1).astype(np.float32)
    if right is not None:
        pts[HAND_JOINTS:] = right
        mask[HAND_JOINTS:, 0] = np.isfinite(right).all(axis=1).astype(np.float32)
    pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)
    return pts, mask


def frames_to_canonical_sequence(frames: Sequence[Dict[str, Any]], *, meta: Optional[Dict[str, Any]] = None) -> CanonicalSkeletonSequence:
    frames = list(frames)
    pts = np.zeros((len(frames), NUM_HAND_NODES, 3), dtype=np.float32)
    mask = np.zeros((len(frames), NUM_HAND_NODES, 1), dtype=np.float32)
    ts = np.zeros((len(frames),), dtype=np.float32)
    pose_rows: List[np.ndarray] = []
    pose_vis_rows: List[np.ndarray] = []
    pose_indices: Optional[List[int]] = None
    meta_out = dict(meta or {})
    fps = float(meta_out.get("fps", 0.0) or 0.0)
    dt_ms = 1000.0 / fps if fps > 0.0 else 33.3333
    for i, frame in enumerate(frames):
        left, right = _extract_hand_landmarks(frame)
        cur_pts, cur_mask = combine_hands(left, right)
        pts[i] = cur_pts
        mask[i] = cur_mask
        pose_xyz, pose_vis, cur_pose_indices = _extract_pose_landmarks(frame)
        if pose_xyz is not None:
            if pose_indices is None:
                pose_indices = list(cur_pose_indices or [])
            if cur_pose_indices and list(cur_pose_indices) != list(pose_indices):
                pos_map = {abs_idx: j for j, abs_idx in enumerate(cur_pose_indices)}
                reordered = np.zeros((len(pose_indices), 3), dtype=np.float32)
                reordered_vis = np.zeros((len(pose_indices),), dtype=np.float32)
                for j, abs_idx in enumerate(pose_indices):
                    src = pos_map.get(abs_idx, -1)
                    if src >= 0:
                        reordered[j] = pose_xyz[src]
                        reordered_vis[j] = pose_vis[src]
                pose_xyz = reordered
                pose_vis = reordered_vis
            pose_rows.append(np.asarray(pose_xyz, dtype=np.float32))
            pose_vis_rows.append(np.asarray(pose_vis, dtype=np.float32))
        elif pose_indices is not None:
            pose_rows.append(np.zeros((len(pose_indices), 3), dtype=np.float32))
            pose_vis_rows.append(np.zeros((len(pose_indices),), dtype=np.float32))
        raw_ts = frame.get("ts", None) if isinstance(frame, dict) else None
        if raw_ts is None:
            ts[i] = float(i) * dt_ms
        else:
            ts[i] = float(raw_ts)
    meta_out.setdefault("coords", str(meta_out.get("coords", "image")))
    meta_out.setdefault("fps", fps)
    meta_out.setdefault("frame_dt_ms", dt_ms)
    pose_xyz_arr = np.stack(pose_rows, axis=0) if pose_rows else None
    pose_vis_arr = np.stack(pose_vis_rows, axis=0) if pose_vis_rows else None
    if pose_indices is not None:
        meta_out.setdefault("pose_indices", list(pose_indices))
    return canonicalize_sequence(
        pts,
        mask,
        ts,
        pose_xyz=pose_xyz_arr,
        pose_vis=pose_vis_arr,
        pose_indices=pose_indices,
        meta=meta_out,
    )


def _load_jsonl_frames(path: Path) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"JSONL frame must be an object: {path}")
        frames.append(item)
    return frames


def load_skeleton_sequence(path: str | Path) -> CanonicalSkeletonSequence:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as z:
            pts = z["pts"]
            mask = z["mask"] if "mask" in z else None
            ts = z["ts_ms"] if "ts_ms" in z else (z["ts"] if "ts" in z else None)
            pose_xyz = z["pose_xyz"] if "pose_xyz" in z else None
            pose_vis = z["pose_vis"] if "pose_vis" in z else None
            pose_indices = z["pose_indices"].tolist() if "pose_indices" in z else None
            meta: Dict[str, Any] = {}
            if "meta_json" in z:
                try:
                    meta = json.loads(str(z["meta_json"].item()))
                except Exception:
                    meta = {}
        return canonicalize_sequence(pts, mask, ts, pose_xyz=pose_xyz, pose_vis=pose_vis, pose_indices=pose_indices, meta=meta)
    if suffix == ".json":
        frames, meta = read_video_file_nocache(str(source))
        return frames_to_canonical_sequence(frames, meta=meta)
    if suffix == ".jsonl":
        return frames_to_canonical_sequence(_load_jsonl_frames(source), meta={"coords": "image"})
    raise ValueError(f"Unsupported skeleton sequence format: {source}")
