from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bio.core.config_utils import load_config_section, write_dataset_manifest, write_run_config
from bio.core.preprocessing import (
    BIO_PREPROCESSING_VERSION_V2,
    BIO_PREPROCESSING_VERSION_V3,
    BioPreprocessConfig,
    preprocess_sequence,
)

NUM_HAND_JOINTS = 21
NUM_HAND_NODES = 42
POSE_KEEP_DEFAULT = [0, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24]


try:
    import orjson as _fastjson

    def _loads_bytes(b: bytes):
        return _fastjson.loads(b)
except Exception:

    def _loads_bytes(b: bytes):
        return json.loads(b.decode("utf-8"))


def _read_video_file_nocache(path_str: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    p = Path(path_str)
    with p.open("rb") as f:
        blob = _loads_bytes(f.read())
    if isinstance(blob, dict) and "frames" in blob:
        return blob["frames"], blob.get("meta", {})
    return blob, {}


def _frames_from_combined(skel: Dict[str, Any], vid: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if "videos" in skel:
        v = skel["videos"].get(vid, {})
        return v.get("frames", []), v.get("meta", {})
    return skel.get(vid, []), {}


@dataclass
class PrelabelConfig:
    include_pose: bool = False
    pose_keep: Tuple[int, ...] = tuple(POSE_KEEP_DEFAULT)
    pose_vis_thr: float = 0.5
    hand_score_thr: float = 0.45
    hand_score_thr_fallback: float = 0.35
    thr_tune_steps: int = 6
    thr_tune_step: float = 0.05
    center: bool = True
    center_mode: str = "masked_mean"
    normalize: bool = True
    norm_method: str = "p95"
    norm_scale: float = 1.0
    smooth_win: int = 7
    motion_percentile: float = 70.0
    hysteresis: float = 0.6
    pad: int = 4
    min_len: int = 8
    fallback_len: int = 64
    min_motion: float = 1e-4
    min_valid_frames: int = 8
    file_cache_size: int = 0
    prefer_pp: bool = True
    trimmed_mode: bool = False
    preprocessing_version: str = BIO_PREPROCESSING_VERSION_V3
    preprocessing_center_alpha: float = 0.2
    preprocessing_scale_alpha: float = 0.1
    preprocessing_min_scale: float = 1e-3
    preprocessing_min_visible_joints_for_scale: int = 4


@dataclass(frozen=True)
class CsvSample:
    vid: str
    label_str: str
    begin: int
    end: int
    split: str
    dataset: str
    signer_id: str
    source_group: str
    sample_id: str
    row_num: int


@dataclass(frozen=True)
class CsvParseResult:
    rows: Tuple[CsvSample, ...]
    rejected: Tuple[Dict[str, Any], ...]
    skipped_split: int
    total_rows: int
    columns: Tuple[str, ...]


class VideoStore:
    def __init__(self, skeletons_path: str | Path, cache_size: int = 0, prefer_pp: bool = True) -> None:
        self.path = Path(skeletons_path)
        self.is_dir = self.path.is_dir()
        self.cache_size = max(0, int(cache_size))
        self.prefer_pp = bool(prefer_pp)
        self._cache: "OrderedDict[str, Tuple[List[Dict[str, Any]], Dict[str, Any]]]" = OrderedDict()
        self._skel: Optional[Dict[str, Any]] = None
        self._path_cache: Dict[str, str] = {}
        if not self.is_dir:
            with self.path.open("rb") as f:
                blob = _loads_bytes(f.read())
            if not isinstance(blob, dict):
                raise ValueError("Combined skeletons JSON must be a dict.")
            self._skel = blob

    def _resolve_path(self, vid: str) -> str:
        cached = self._path_cache.get(vid)
        if cached is not None:
            return cached
        base = self.path / f"{vid}.json"
        pp = self.path / f"{vid}_pp.json"
        if self.prefer_pp and pp.exists():
            chosen = pp
        elif base.exists():
            chosen = base
        elif pp.exists():
            chosen = pp
        else:
            chosen = base
        path = str(chosen)
        self._path_cache[vid] = path
        return path

    def get_frames(self, vid: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if self.is_dir:
            path = self._resolve_path(vid)
            if self.cache_size > 0:
                hit = self._cache.get(path)
                if hit is not None:
                    self._cache.move_to_end(path)
                    return hit
            if not Path(path).exists():
                return [], {}
            frames, meta = _read_video_file_nocache(path)
            if self.cache_size > 0:
                self._cache[path] = (frames, meta)
                if len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
            return frames, meta
        if self._skel is None:
            return [], {}
        return _frames_from_combined(self._skel, vid)


def _as_bool(s: str) -> Optional[bool]:
    if s is None:
        return None
    v = s.strip().lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    return None


def _bool_or_default(raw: Any, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        parsed = _as_bool(raw)
        if parsed is not None:
            return parsed
    return default


def _csv_or_default(raw: Any, default: str) -> str:
    if raw is None:
        return default
    if isinstance(raw, (list, tuple)):
        return ",".join(str(x) for x in raw)
    return str(raw)


def _is_missing(raw: Any) -> bool:
    if raw is None:
        return True
    if isinstance(raw, str) and not raw.strip():
        return True
    return False


def _slugify(text: str, *, max_len: int = 80) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
    clean = clean.strip("._-")
    if not clean:
        clean = "sample"
    return clean[:max_len]


def _stable_sample_id(vid: str, label: str, begin: int, end: int) -> str:
    digest = hashlib.sha1(f"{vid}|{begin}|{end}|{label}".encode("utf-8")).hexdigest()[:10]
    return f"{_slugify(vid)}__{int(begin)}_{int(end)}__{digest}"


def _norm_split_name(raw: Any) -> str:
    val = str(raw or "").strip().lower()
    if val == "test":
        return "val"
    return val


def _parse_required_int(raw: Any, field: str) -> int:
    if _is_missing(raw):
        raise ValueError(f"missing_{field}")
    try:
        return int(float(raw))
    except Exception as exc:
        raise ValueError(f"invalid_{field}") from exc


def _resolve_row_split(row: Dict[str, Any], split: str) -> Tuple[bool, Optional[str]]:
    split = _norm_split_name(split)
    raw_split = _norm_split_name(row.get("split"))
    raw_train = row.get("train") or row.get("is_train")
    if raw_split:
        return raw_split == split, None
    if raw_train is not None and str(raw_train).strip():
        bt = _as_bool(str(raw_train))
        if bt is None:
            return str(raw_train).strip().lower() == split, None
        return (bt if split == "train" else (not bt)), None
    return False, "missing_split_value"


def _validate_csv_schema(fieldnames: Sequence[str] | None) -> None:
    fields = {str(name or "").strip() for name in (fieldnames or [])}
    required = {"attachment_id", "text", "begin", "end"}
    missing = sorted(x for x in required if x not in fields)
    if missing:
        raise RuntimeError(f"CSV is missing required columns: {', '.join(missing)}")
    if "split" not in fields and "train" not in fields and "is_train" not in fields:
        raise RuntimeError("CSV must contain either 'split' or 'train'/'is_train' columns.")


def parse_csv(csv_path: Path, split: str) -> CsvParseResult:
    rows: List[CsvSample] = []
    rejected: List[Dict[str, Any]] = []
    skipped_split = 0
    total_rows = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        except csv.Error:
            class _D(csv.Dialect):
                delimiter = "\t" if ("\t" in sample and "," not in sample) else ","
                quotechar = '"'
                doublequote = True
                skipinitialspace = False
                lineterminator = "\n"
                quoting = csv.QUOTE_MINIMAL
            dialect = _D()

        rdr = csv.DictReader(f, dialect=dialect)
        _validate_csv_schema(rdr.fieldnames)
        columns = tuple(str(x) for x in (rdr.fieldnames or []))
        seen_sample_ids: set[str] = set()

        for row_num, row in enumerate(rdr, start=2):
            total_rows += 1
            use, split_reason = _resolve_row_split(row, split)
            if split_reason is not None:
                rejected.append(
                    {
                        "row_num": int(row_num),
                        "reason": split_reason,
                        "attachment_id": row.get("attachment_id", ""),
                        "text": row.get("text", ""),
                    }
                )
                continue
            if not use:
                skipped_split += 1
                continue

            vid_raw = (row.get("attachment_id") or "").strip()
            if not vid_raw:
                rejected.append(
                    {
                        "row_num": int(row_num),
                        "reason": "missing_attachment_id",
                        "text": row.get("text", ""),
                    }
                )
                continue
            vid = Path(vid_raw).stem
            if not vid:
                rejected.append(
                    {
                        "row_num": int(row_num),
                        "reason": "invalid_attachment_id",
                        "attachment_id": vid_raw,
                    }
                )
                continue

            label = (row.get("text") or "").strip()
            if not label:
                rejected.append(
                    {
                        "row_num": int(row_num),
                        "reason": "missing_label",
                        "attachment_id": vid_raw,
                    }
                )
                continue

            try:
                begin = _parse_required_int(row.get("begin"), "begin")
                end = _parse_required_int(row.get("end"), "end")
            except ValueError as exc:
                rejected.append(
                    {
                        "row_num": int(row_num),
                        "reason": str(exc),
                        "attachment_id": vid_raw,
                        "text": label,
                        "begin": row.get("begin"),
                        "end": row.get("end"),
                    }
                )
                continue
            if begin < 0 or end < 0:
                rejected.append(
                    {
                        "row_num": int(row_num),
                        "reason": "negative_clip_bounds",
                        "attachment_id": vid_raw,
                        "text": label,
                        "begin": begin,
                        "end": end,
                    }
                )
                continue
            if end <= begin:
                rejected.append(
                    {
                        "row_num": int(row_num),
                        "reason": "invalid_clip_range",
                        "attachment_id": vid_raw,
                        "text": label,
                        "begin": begin,
                        "end": end,
                    }
                )
                continue
            signer_id = str(
                row.get("user_id")
                or row.get("signer_id")
                or row.get("user")
                or row.get("signer")
                or ""
            ).strip()
            if not signer_id:
                signer_id = vid

            sample_id = _stable_sample_id(vid, label, begin, end)
            if sample_id in seen_sample_ids:
                rejected.append(
                    {
                        "row_num": int(row_num),
                        "reason": "duplicate_sample_id",
                        "attachment_id": vid_raw,
                        "text": label,
                        "begin": begin,
                        "end": end,
                        "sample_id": sample_id,
                    }
                )
                continue
            seen_sample_ids.add(sample_id)

            rows.append(
                CsvSample(
                    vid=vid,
                    label_str=label,
                    begin=begin,
                    end=end,
                    split=split,
                    dataset="slovo",
                    signer_id=signer_id,
                    source_group=signer_id,
                    sample_id=sample_id,
                    row_num=int(row_num),
                )
            )

    if not rows:
        raise RuntimeError(
            "No samples for split after CSV filtering. "
            "Check delimiter, column names, split/train flags, and rejected_rows.json."
        )
    return CsvParseResult(
        rows=tuple(rows),
        rejected=tuple(rejected),
        skipped_split=int(skipped_split),
        total_rows=int(total_rows),
        columns=columns,
    )


def build_mirror_idx(include_pose: bool, pose_keep: Sequence[int]) -> np.ndarray:
    idx = list(range(21, 42)) + list(range(0, 21))
    if include_pose and pose_keep:
        pairs = {(9, 10), (11, 12), (13, 14), (15, 16), (23, 24)}
        pos_map = {abs_i: i for i, abs_i in enumerate(pose_keep)}
        for abs_i in pose_keep:
            if abs_i == 0:
                idx.append(NUM_HAND_NODES + pos_map[0])
                continue
            pair_abs = None
            for a, b in pairs:
                if abs_i == a:
                    pair_abs = b
                elif abs_i == b:
                    pair_abs = a
            if pair_abs is None or pair_abs not in pos_map:
                idx.append(NUM_HAND_NODES + pos_map[abs_i])
            else:
                idx.append(NUM_HAND_NODES + pos_map[pair_abs])
    return np.array(idx, dtype=np.int64)


def compute_pose_reorder(meta: Dict[str, Any], pose_keep: Sequence[int]) -> Optional[List[int]]:
    if not pose_keep:
        return None
    order = meta.get("pose_indices", None)
    if order == "all":
        return list(pose_keep)
    if isinstance(order, list):
        order_list: List[Any] = []
        for v in order:
            try:
                order_list.append(int(v))
            except Exception:
                order_list.append(v)
        idxs: List[int] = []
        for want in pose_keep:
            try:
                idxs.append(order_list.index(want))
            except ValueError:
                idxs.append(-1)
        return idxs
    return list(range(len(pose_keep)))


def _build_mask_for_frame(fr: Dict[str, Any], thr: float) -> Tuple[List[int], List[int]]:
    mL = [0] * NUM_HAND_JOINTS
    mR = [0] * NUM_HAND_JOINTS
    L = fr.get("hand 1")
    R = fr.get("hand 2")
    Ls = fr.get("hand 1_score")
    Rs = fr.get("hand 2_score")
    try:
        Ls = float(Ls) if Ls is not None else 0.0
    except Exception:
        Ls = 0.0
    try:
        Rs = float(Rs) if Rs is not None else 0.0
    except Exception:
        Rs = 0.0
    if L is not None and Ls >= thr:
        mL = [1] * NUM_HAND_JOINTS
    if R is not None and Rs >= thr:
        mR = [1] * NUM_HAND_JOINTS
    return mL, mR


def choose_thr_for_video(frames: Sequence[Dict[str, Any]], cfg: PrelabelConfig) -> Tuple[float, float]:
    thresholds: List[float] = []
    for s in range(max(0, int(cfg.thr_tune_steps)) + 1):
        thresholds.append(max(0.0, float(cfg.hand_score_thr) - s * float(cfg.thr_tune_step)))
    thresholds.append(float(cfg.hand_score_thr_fallback))
    seen = []
    for t in thresholds:
        if t not in seen:
            seen.append(t)
    best_thr = seen[0] if seen else float(cfg.hand_score_thr)
    best_cov = -1.0
    total = len(frames)
    for thr in seen:
        valid = 0
        for fr in frames:
            L = fr.get("hand 1")
            R = fr.get("hand 2")
            Ls = fr.get("hand 1_score") or 0.0
            Rs = fr.get("hand 2_score") or 0.0
            try:
                Ls = float(Ls)
            except Exception:
                Ls = 0.0
            try:
                Rs = float(Rs)
            except Exception:
                Rs = 0.0
            vL = (L is not None) and (Ls >= thr)
            vR = (R is not None) and (Rs >= thr)
            if vL or vR:
                valid += 1
        cov = valid / float(total) if total > 0 else 0.0
        if (cov > best_cov) or (abs(cov - best_cov) < 1e-9 and thr > best_thr):
            best_cov = cov
            best_thr = thr
    return best_thr, best_cov


def frames_to_raw_arrays(
    frames: Sequence[Dict[str, Any]],
    cfg: PrelabelConfig,
    thr: float,
    meta: Dict[str, Any],
    mirror_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = len(frames)
    P = len(cfg.pose_keep) if cfg.include_pose else 0
    V = NUM_HAND_NODES + P
    pts = np.zeros((T, V, 3), dtype=np.float32)
    mask = np.zeros((T, V, 1), dtype=np.float32)
    ts = np.zeros((T,), dtype=np.float32)

    reorder = compute_pose_reorder(meta, cfg.pose_keep) if cfg.include_pose else None
    wrist_left_out = cfg.pose_keep.index(15) if cfg.include_pose and 15 in cfg.pose_keep else -1
    wrist_right_out = cfg.pose_keep.index(16) if cfg.include_pose and 16 in cfg.pose_keep else -1
    coords_tag = str(meta.get("coords", "image")).lower()
    allow_pose_wrist = (
        cfg.include_pose
        and P > 0
        and (wrist_left_out >= 0 or wrist_right_out >= 0)
        and coords_tag != "world"
    )

    for t, fr in enumerate(frames):
        try:
            ts[t] = float(fr.get("ts", 0.0))
        except Exception:
            ts[t] = 0.0
        mL, mR = _build_mask_for_frame(fr, thr)
        L = fr.get("hand 1") if (mL[0] == 1) else None
        R = fr.get("hand 2") if (mR[0] == 1) else None
        if L is not None:
            for j in range(min(NUM_HAND_JOINTS, len(L))):
                p = L[j]
                pts[t, j, 0] = float(p.get("x", 0.0))
                pts[t, j, 1] = float(p.get("y", 0.0))
                pts[t, j, 2] = float(p.get("z", 0.0))
                mask[t, j, 0] = 1.0
        if R is not None:
            for j in range(min(NUM_HAND_JOINTS, len(R))):
                p = R[j]
                pts[t, 21 + j, 0] = float(p.get("x", 0.0))
                pts[t, 21 + j, 1] = float(p.get("y", 0.0))
                pts[t, 21 + j, 2] = float(p.get("z", 0.0))
                mask[t, 21 + j, 0] = 1.0

        pose_wrist_left = None
        pose_wrist_right = None
        pose_wrist_left_ok = False
        pose_wrist_right_ok = False
        if cfg.include_pose and P > 0:
            pose = fr.get("pose")
            pose_vis = fr.get("pose_vis")
            if isinstance(pose, list) and reorder is not None:
                for k_out, k_in in enumerate(reorder):
                    v_idx = NUM_HAND_NODES + k_out
                    if k_in < 0 or k_in >= len(pose):
                        continue
                    pk = pose[k_in]
                    pts[t, v_idx, 0] = float(pk.get("x", 0.0))
                    pts[t, v_idx, 1] = float(pk.get("y", 0.0))
                    pts[t, v_idx, 2] = float(pk.get("z", 0.0))
                    if isinstance(pose_vis, list) and k_in < len(pose_vis):
                        if float(pose_vis[k_in]) >= cfg.pose_vis_thr:
                            mask[t, v_idx, 0] = 1.0
                            if allow_pose_wrist and k_out == wrist_left_out:
                                pose_wrist_left_ok = True
                            if allow_pose_wrist and k_out == wrist_right_out:
                                pose_wrist_right_ok = True
                    else:
                        mask[t, v_idx, 0] = 1.0
                        if allow_pose_wrist and k_out == wrist_left_out:
                            pose_wrist_left_ok = True
                        if allow_pose_wrist and k_out == wrist_right_out:
                            pose_wrist_right_ok = True
                    if allow_pose_wrist and k_out == wrist_left_out:
                        pose_wrist_left = pk
                    if allow_pose_wrist and k_out == wrist_right_out:
                        pose_wrist_right = pk

        if allow_pose_wrist:
            if pose_wrist_left is not None and mask[t, 0, 0] == 0.0 and pose_wrist_left_ok:
                pts[t, 0, 0] = float(pose_wrist_left.get("x", 0.0))
                pts[t, 0, 1] = float(pose_wrist_left.get("y", 0.0))
                pts[t, 0, 2] = float(pose_wrist_left.get("z", 0.0))
                mask[t, 0, 0] = 1.0
            if pose_wrist_right is not None and mask[t, 21, 0] == 0.0 and pose_wrist_right_ok:
                pts[t, 21, 0] = float(pose_wrist_right.get("x", 0.0))
                pts[t, 21, 1] = float(pose_wrist_right.get("y", 0.0))
                pts[t, 21, 2] = float(pose_wrist_right.get("z", 0.0))
                mask[t, 21, 0] = 1.0

    if T > 0:
        m = torch.from_numpy(mask).permute(1, 2, 0)
        m = F.avg_pool1d(m, kernel_size=3, stride=1, padding=1)
        m = (m >= 0.5).float()
        mask = m.permute(2, 0, 1).contiguous().numpy()

        left_cnt = float(mask[:, 0:21, :].sum())
        right_cnt = float(mask[:, 21:42, :].sum())
        if left_cnt < 0.1 * T * 21 and right_cnt >= 0.6 * T * 21:
            pts = pts[:, mirror_idx, :]
            mask = mask[:, mirror_idx, :]

    return pts, mask, ts


def _shared_preprocess_cfg(cfg: PrelabelConfig) -> BioPreprocessConfig:
    return BioPreprocessConfig(
        version=str(cfg.preprocessing_version or BIO_PREPROCESSING_VERSION_V3),
        center_alpha=float(cfg.preprocessing_center_alpha),
        scale_alpha=float(cfg.preprocessing_scale_alpha),
        min_scale=float(cfg.preprocessing_min_scale),
        min_visible_joints_for_scale=int(cfg.preprocessing_min_visible_joints_for_scale),
    )


def prepare_model_pts(
    pts_raw: np.ndarray,
    mask: np.ndarray,
    ts: np.ndarray,
    cfg: PrelabelConfig,
) -> np.ndarray:
    version = str(cfg.preprocessing_version or BIO_PREPROCESSING_VERSION_V3)
    if version == BIO_PREPROCESSING_VERSION_V2:
        pts_t = torch.from_numpy(pts_raw)
        mask_t = torch.from_numpy(mask)
        pts_t = torch.nan_to_num(pts_t, nan=0.0, posinf=0.0, neginf=0.0)
        return center_norm(pts_t, mask_t, cfg).numpy()
    pts_v3, _debug = preprocess_sequence(
        pts_raw,
        mask,
        cfg=_shared_preprocess_cfg(cfg),
        dtype=torch.float32,
    )
    return np.asarray(pts_v3, dtype=np.float32)


def center_norm(pts: torch.Tensor, mask: torch.Tensor, cfg: PrelabelConfig) -> torch.Tensor:
    if cfg.center:
        if cfg.center_mode == "wrists":
            wr = []
            if float(mask[:, 0, :].sum()) > 0:
                wr.append(pts[:, 0:1, :])
            if mask.shape[1] > 21 and float(mask[:, 21, :].sum()) > 0:
                wr.append(pts[:, 21:22, :])
            if wr:
                c = torch.cat(wr, dim=1).mean(dim=1, keepdim=True)
                pts = pts - c
            else:
                denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
                mean = (pts * mask).sum(dim=1, keepdim=True) / denom
                pts = pts - mean
        else:
            denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
            mean = (pts * mask).sum(dim=1, keepdim=True) / denom
            pts = pts - mean
    if cfg.normalize:
        if cfg.norm_method == "p95":
            flat = pts.abs().reshape(-1)
            k = max(1, int(flat.numel() * 0.95))
            span = flat.kthvalue(k)[0]
        elif cfg.norm_method == "mad":
            med = pts.median().values
            mad = (pts - med).abs().median().values * 1.4826
            span = torch.clamp(mad, min=1e-6)
        else:
            span = pts.abs().amax()
        if float(span) > 1e-6:
            scale = span / max(1e-6, float(cfg.norm_scale))
            pts = pts / (scale + 1e-6)
    return pts


def compute_motion(
    pts: np.ndarray,
    mask: np.ndarray,
    ts: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    T = pts.shape[0]
    if T == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32), False
    vel = pts[1:] - pts[:-1]
    used_dt = False
    if ts is not None and ts.size == T:
        dt = np.diff(ts.astype(np.float32))
        if dt.size > 0 and np.all(dt > 0):
            dt = np.clip(dt, 1e-6, None)
            vel = vel / dt[:, None, None]
            used_dt = True
    valid = (mask[1:, :NUM_HAND_NODES, 0] > 0.0) & (mask[:-1, :NUM_HAND_NODES, 0] > 0.0)
    speed = np.linalg.norm(vel[:, :NUM_HAND_NODES, :], axis=2)
    speed[~valid] = 0.0
    valid_counts = valid.sum(axis=1)
    motion = np.zeros((T,), dtype=np.float32)
    if speed.shape[0] > 0:
        denom = np.clip(valid_counts, 1, None).astype(np.float32)
        motion[1:] = speed.sum(axis=1) / denom
        motion[1:][valid_counts == 0] = 0.0
    valid_counts_full = np.zeros((T,), dtype=np.int32)
    valid_counts_full[1:] = valid_counts
    return motion, valid_counts_full, used_dt


def smooth_motion(motion: np.ndarray, win: int) -> np.ndarray:
    if motion.size == 0 or win <= 1 or win > motion.size:
        return motion.copy()
    kernel = np.ones((win,), dtype=np.float32) / float(win)
    return np.convolve(motion, kernel, mode="same").astype(np.float32)


def best_motion_window(motion: np.ndarray, window: int) -> int:
    if motion.size == 0:
        return 0
    if window <= 0 or motion.size <= window:
        return 0
    kernel = np.ones((window,), dtype=np.float32)
    scores = np.convolve(motion, kernel, mode="valid")
    return int(np.argmax(scores))


def find_active_segment(
    motion: np.ndarray,
    valid_mask: Optional[np.ndarray],
    cfg: PrelabelConfig,
) -> Tuple[Optional[Tuple[int, int]], Dict[str, Any]]:
    if motion.size == 0:
        return None, {"thr_on": 0.0, "thr_off": 0.0, "peak": 0}
    if valid_mask is not None and valid_mask.size == motion.size:
        base = motion[valid_mask]
    else:
        base = motion
    if base.size == 0:
        base = motion[motion > 0]
    if base.size == 0:
        thr_on = 0.0
    else:
        thr_on = float(np.percentile(base, cfg.motion_percentile))
    thr_on = max(thr_on, float(cfg.min_motion), 1e-6)
    thr_off = float(cfg.hysteresis) * thr_on
    active = motion >= thr_on
    if not active.any():
        return None, {"thr_on": thr_on, "thr_off": thr_off, "peak": 0}
    peak = int(np.argmax(motion))
    left = peak
    right = peak
    while left > 0 and active[left - 1]:
        left -= 1
    while right < motion.size - 1 and active[right + 1]:
        right += 1
    while left > 0 and motion[left - 1] >= thr_off:
        left -= 1
    while right < motion.size - 1 and motion[right + 1] >= thr_off:
        right += 1
    start = max(0, left - int(cfg.pad))
    end = min(motion.size - 1, right + int(cfg.pad))
    return (start, end), {"thr_on": thr_on, "thr_off": thr_off, "peak": peak}


def make_bio_labels(
    label_str: str,
    motion: np.ndarray,
    valid_frames: int,
    valid_mask: Optional[np.ndarray],
    cfg: PrelabelConfig,
) -> Tuple[np.ndarray, int, int, Dict[str, Any]]:
    T = motion.size
    bio = np.zeros((T,), dtype=np.uint8)
    is_no_event = label_str.strip().lower() == "no_event"
    if is_no_event or T == 0:
        return bio, -1, -1, {"used_fallback": False}

    segment, seg_stats = find_active_segment(motion, valid_mask, cfg)
    motion_max = float(motion.max()) if T > 0 else 0.0
    min_valid_frames = cfg.min_valid_frames if cfg.min_valid_frames > 0 else cfg.min_len

    used_fallback = False
    start_idx = -1
    end_idx = -1

    if (
        valid_frames < min_valid_frames
        or motion_max < float(cfg.min_motion)
        or segment is None
    ):
        used_fallback = True
    else:
        start_idx, end_idx = segment
        if (end_idx - start_idx + 1) < cfg.min_len:
            used_fallback = True

    if used_fallback:
        if T <= int(cfg.fallback_len):
            peak = int(np.argmax(motion)) if T > 0 else 0
            start_idx = max(0, peak - int(cfg.fallback_len // 2))
        else:
            start_idx = int(best_motion_window(motion, cfg.fallback_len))
        end_idx = min(int(start_idx + cfg.fallback_len - 1), T - 1)

    if start_idx >= 0 and end_idx >= start_idx:
        bio[start_idx] = 1
        if end_idx > start_idx:
            bio[start_idx + 1 : end_idx + 1] = 2

    stats = {"used_fallback": used_fallback}
    stats.update(seg_stats)
    return bio, start_idx, end_idx, stats


def make_trimmed_gold_bio(
    label_str: str,
    *,
    total_frames: int,
    begin_hint: int,
    end_hint: int,
) -> Tuple[np.ndarray, int, int, Dict[str, Any]]:
    bio = np.zeros((max(0, int(total_frames)),), dtype=np.uint8)
    if int(total_frames) <= 0:
        return bio, -1, -1, {"used_fallback": False, "label_source": "trimmed_csv", "end_semantics": "half_open"}
    is_no_event = str(label_str or "").strip().lower() == "no_event"
    if is_no_event:
        return bio, -1, -1, {"used_fallback": False, "label_source": "trimmed_csv", "end_semantics": "half_open"}
    start_idx = int(np.clip(int(begin_hint), 0, int(total_frames) - 1))
    end_exclusive = int(np.clip(int(end_hint), start_idx + 1, int(total_frames)))
    end_idx = int(end_exclusive - 1)
    bio[start_idx] = 1
    if end_idx > start_idx:
        bio[start_idx + 1 : end_idx + 1] = 2
    return bio, start_idx, end_idx, {"used_fallback": False, "label_source": "trimmed_csv", "end_semantics": "half_open"}


def save_npz(
    out_path: Path,
    pts: np.ndarray,
    mask: np.ndarray,
    ts: np.ndarray,
    bio: np.ndarray,
    label_str: str,
    is_no_event: bool,
    start_idx: int,
    end_idx: int,
    meta: Dict[str, Any],
    extra_arrays: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    meta_json = json.dumps(meta, ensure_ascii=True)
    payload: Dict[str, Any] = {
        "pts": pts.astype(np.float32, copy=False),
        "mask": mask.astype(np.float32, copy=False),
        "ts": ts.astype(np.float32, copy=False),
        "bio": bio.astype(np.uint8, copy=False),
        "label_str": np.array(label_str, dtype=np.unicode_),
        "is_no_event": np.array(bool(is_no_event)),
        "start_idx": np.array(int(start_idx), dtype=np.int64),
        "end_idx": np.array(int(end_idx), dtype=np.int64),
        "meta": np.array(meta_json, dtype=np.unicode_),
    }
    for key, value in dict(extra_arrays or {}).items():
        if value is None:
            continue
        payload[str(key)] = np.asarray(value)
    np.savez(out_path, **payload)


def save_debug_log(out_dir: Path, sample_id: str, payload: Dict[str, Any]) -> None:
    path = out_dir / f"debug_{sample_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def save_debug_plot(
    out_dir: Path,
    sample_id: str,
    motion: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    path = out_dir / f"debug_{sample_id}.png"
    x = np.arange(motion.size)
    plt.figure(figsize=(10, 4))
    plt.plot(x, motion, label="motion")
    if start_idx >= 0:
        plt.axvline(start_idx, color="g", linestyle="--", label="start")
    if end_idx >= 0:
        plt.axvline(end_idx, color="r", linestyle="--", label="end")
    if start_idx >= 0 and end_idx >= start_idx:
        plt.axvspan(start_idx, end_idx, color="g", alpha=0.1)
    plt.xlabel("frame")
    plt.ylabel("motion")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return True


def process_sample(
    task: Dict[str, Any],
    cfg: PrelabelConfig,
    store: VideoStore,
    mirror_idx: np.ndarray,
    out_dir: Path,
) -> Dict[str, Any]:
    vid = task["vid"]
    sample_id = task["sample_id"]
    label_str = task["label_str"]
    begin_hint = int(task["begin"])
    end_hint = int(task["end"])
    out_path = Path(task["out_path"])
    split = str(task.get("split", ""))
    dataset = str(task.get("dataset", "slovo"))
    source_group = str(task.get("source_group", vid))
    signer_id = str(task.get("signer_id", source_group or vid))

    frames, meta = store.get_frames(vid)
    if not frames:
        P = len(cfg.pose_keep) if cfg.include_pose else 0
        V = NUM_HAND_NODES + P
        pts = np.zeros((0, V, 3), dtype=np.float32)
        pts_raw = np.zeros((0, V, 3), dtype=np.float32)
        mask = np.zeros((0, V, 1), dtype=np.float32)
        ts = np.zeros((0,), dtype=np.float32)
        bio = np.zeros((0,), dtype=np.uint8)
        meta_out = {
            "error": "no_frames",
            "vid": vid,
            "sample_id": sample_id,
            "split": split,
            "dataset": dataset,
            "source_group": source_group,
            "signer_id": signer_id,
            "begin_hint": begin_hint,
            "end_hint": end_hint,
            "preprocessing_version": str(cfg.preprocessing_version),
            "preprocessing_config": asdict(_shared_preprocess_cfg(cfg)),
            "coords": "image",
        }
        save_npz(
            out_path,
            pts,
            mask,
            ts,
            bio,
            label_str,
            label_str.lower() == "no_event",
            -1,
            -1,
            meta_out,
            extra_arrays={"pts_raw": pts_raw},
        )
        return {
            "index": {
                "vid": sample_id,
                "label_str": label_str,
                "path_to_npz": str(out_path.name),
                "T_total": 0,
                "start_idx": -1,
                "end_idx": -1,
                "is_no_event": bool(label_str.strip().lower() == "no_event"),
                "split": split,
                "dataset": dataset,
                "source_group": source_group,
                "signer_id": signer_id,
            },
            "stats": {
                "segment_len": 0,
                "used_fallback": False,
                "is_no_event": bool(label_str.strip().lower() == "no_event"),
                "valid_frames": 0,
                "T_total": 0,
            },
        }

    total_frames = len(frames)
    if bool(cfg.trimmed_mode):
        begin = 0
        end = total_frames
        seg = frames
    else:
        begin = max(0, begin_hint)
        end = end_hint if end_hint > begin else total_frames
        end = min(end, total_frames)
        seg = frames[begin:end] if end > begin else frames
    T_total = len(seg)

    thr_used, coverage_pre = choose_thr_for_video(seg, cfg)
    pts_raw, mask, ts = frames_to_raw_arrays(seg, cfg, thr_used, meta, mirror_idx)
    pts = prepare_model_pts(pts_raw, mask, ts, cfg)

    valid_frames = int(np.any(mask[:, :NUM_HAND_NODES, 0] > 0.0, axis=1).sum()) if T_total > 0 else 0
    coverage_post = valid_frames / float(T_total) if T_total > 0 else 0.0

    motion_raw, valid_counts, motion_dt_used = compute_motion(pts, mask, ts)
    motion = smooth_motion(motion_raw, cfg.smooth_win)
    valid_mask = (valid_counts > 0) if valid_counts.size == motion.size else None

    motion_start_idx = -1
    motion_end_idx = -1
    bio, start_idx, end_idx, debug_stats = make_bio_labels(label_str, motion, valid_frames, valid_mask, cfg)
    if bool(cfg.trimmed_mode):
        motion_start_idx = int(start_idx)
        motion_end_idx = int(end_idx)
        bio, start_idx, end_idx, debug_stats = make_trimmed_gold_bio(
            label_str,
            total_frames=T_total,
            begin_hint=begin_hint,
            end_hint=end_hint,
        )

    is_no_event = label_str.strip().lower() == "no_event"
    seg_len = int(end_idx - start_idx + 1) if start_idx >= 0 else 0
    motion_max = float(motion.max()) if motion.size > 0 else 0.0
    motion_mean = float(motion.mean()) if motion.size > 0 else 0.0
    motion_p95 = float(np.percentile(motion, 95)) if motion.size > 0 else 0.0
    motion_p99 = float(np.percentile(motion, 99)) if motion.size > 0 else 0.0

    meta_out = {
        "vid": vid,
        "sample_id": sample_id,
        "thr_used": float(thr_used),
        "coverage": float(coverage_pre),
        "coverage_post": float(coverage_post),
        "valid_frames": int(valid_frames),
        "motion_max": motion_max,
        "motion_mean": motion_mean,
        "motion_p95": motion_p95,
        "motion_p99": motion_p99,
        "thr_on": float(debug_stats.get("thr_on", 0.0)),
        "thr_off": float(debug_stats.get("thr_off", 0.0)),
        "motion_peak": int(debug_stats.get("peak", 0)),
        "motion_dt_used": bool(motion_dt_used),
        "motion_valid_frames": int(valid_mask.sum()) if valid_mask is not None else 0,
        "segment_len": int(seg_len),
        "used_fallback": bool(debug_stats.get("used_fallback", False)),
        "begin_hint": int(begin_hint),
        "end_hint": int(end_hint),
        "clip_begin": int(begin),
        "clip_end": int(end),
        "trimmed_mode": bool(cfg.trimmed_mode),
        "label_source": str(debug_stats.get("label_source", "motion")),
        "annotation_end_semantics": str(debug_stats.get("end_semantics", "unknown")),
        "motion_start_idx": int(motion_start_idx),
        "motion_end_idx": int(motion_end_idx),
        "split": split,
        "dataset": dataset,
        "source_group": source_group,
        "signer_id": signer_id,
        "coords": meta.get("coords", ""),
        "include_pose": bool(cfg.include_pose),
        "pose_keep": list(cfg.pose_keep),
        "preprocessing_version": str(cfg.preprocessing_version),
        "preprocessing_config": asdict(_shared_preprocess_cfg(cfg)),
        "raw_pts_key": "pts_raw",
        "preview_pts_key": "pts",
    }
    if sample_id != vid:
        meta_out["orig_vid"] = vid

    extra_arrays: Dict[str, np.ndarray] = {"pts_raw": pts_raw}
    if str(cfg.preprocessing_version) == BIO_PREPROCESSING_VERSION_V3:
        extra_arrays["pts_preview_v3"] = pts
    save_npz(out_path, pts, mask, ts, bio, label_str, is_no_event, start_idx, end_idx, meta_out, extra_arrays=extra_arrays)

    if task.get("debug"):
        debug_payload = {
            "vid": vid,
            "sample_id": sample_id,
            "label_str": label_str,
            "T_total": int(T_total),
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "thr_used": float(thr_used),
            "coverage": float(coverage_pre),
            "coverage_post": float(coverage_post),
            "valid_frames": int(valid_frames),
            "motion_valid_frames": int(valid_mask.sum()) if valid_mask is not None else 0,
            "used_fallback": bool(debug_stats.get("used_fallback", False)),
            "label_source": str(debug_stats.get("label_source", "motion")),
            "trimmed_mode": bool(cfg.trimmed_mode),
            "motion_stats": {
                "max": motion_max,
                "mean": motion_mean,
                "p95": motion_p95,
                "p99": motion_p99,
                "thr_on": float(debug_stats.get("thr_on", 0.0)),
                "thr_off": float(debug_stats.get("thr_off", 0.0)),
                "peak": int(debug_stats.get("peak", 0)),
                "dt_used": bool(motion_dt_used),
            },
            "motion": motion.tolist(),
        }
        save_debug_log(out_dir, sample_id, debug_payload)
        save_debug_plot(out_dir, sample_id, motion, start_idx, end_idx)

    return {
        "index": {
            "vid": sample_id,
            "label_str": label_str,
            "path_to_npz": str(out_path.name),
            "T_total": int(T_total),
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "is_no_event": bool(is_no_event),
            "split": split,
            "dataset": dataset,
            "source_group": source_group,
            "signer_id": signer_id,
        },
        "stats": {
            "segment_len": seg_len,
            "used_fallback": bool(debug_stats.get("used_fallback", False)),
            "is_no_event": bool(is_no_event),
            "valid_frames": int(valid_frames),
            "T_total": int(T_total),
        },
    }


_WORKER_CFG: Optional[PrelabelConfig] = None
_WORKER_STORE: Optional[VideoStore] = None
_WORKER_MIRROR_IDX: Optional[np.ndarray] = None
_WORKER_OUT_DIR: Optional[Path] = None


def _init_worker(cfg: PrelabelConfig, skeletons_path: str, out_dir: str) -> None:
    global _WORKER_CFG, _WORKER_STORE, _WORKER_MIRROR_IDX, _WORKER_OUT_DIR
    _WORKER_CFG = cfg
    _WORKER_STORE = VideoStore(skeletons_path, cache_size=cfg.file_cache_size, prefer_pp=cfg.prefer_pp)
    _WORKER_MIRROR_IDX = build_mirror_idx(cfg.include_pose, cfg.pose_keep)
    _WORKER_OUT_DIR = Path(out_dir)


def _process_task(task: Dict[str, Any]) -> Dict[str, Any]:
    if _WORKER_CFG is None or _WORKER_STORE is None or _WORKER_MIRROR_IDX is None or _WORKER_OUT_DIR is None:
        raise RuntimeError("Worker not initialized.")
    return process_sample(task, _WORKER_CFG, _WORKER_STORE, _WORKER_MIRROR_IDX, _WORKER_OUT_DIR)


def write_index_files(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    index_json = out_dir / "index.json"
    with index_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=True, indent=2)

    index_csv = out_dir / "index.csv"
    if rows:
        fields = ["vid", "label_str", "path_to_npz", "T_total", "start_idx", "end_idx", "is_no_event", "split", "dataset", "source_group", "signer_id"]
        with index_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fields})


def write_csv_audit(out_dir: Path, parsed: CsvParseResult) -> Dict[str, Any]:
    reason_counts = Counter(str(item.get("reason", "unknown")) for item in parsed.rejected)
    audit = {
        "total_rows": int(parsed.total_rows),
        "accepted_rows": int(len(parsed.rows)),
        "skipped_split": int(parsed.skipped_split),
        "rejected_rows": int(len(parsed.rejected)),
        "rejected_reason_counts": dict(sorted(reason_counts.items())),
        "columns": list(parsed.columns),
    }
    (out_dir / "csv_audit.json").write_text(json.dumps(audit, ensure_ascii=True, indent=2), encoding="utf-8")
    (out_dir / "rejected_rows.json").write_text(
        json.dumps(list(parsed.rejected), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return audit


def build_len_hist(lengths: List[int]) -> Dict[str, int]:
    bins = [1, 8, 16, 32, 64, 128, 256, 512, 1024]
    hist: Dict[str, int] = {}
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1] - 1
        key = f"{lo}-{hi}"
        hist[key] = sum(1 for x in lengths if lo <= x <= hi)
    hist[f"{bins[-1]}+"] = sum(1 for x in lengths if x >= bins[-1])
    return hist


def write_summary(
    out_dir: Path,
    stats: List[Dict[str, Any]],
    *,
    split: str,
    dataset: str,
    csv_audit: Dict[str, Any],
) -> Dict[str, Any]:
    total = len(stats)
    no_event = sum(1 for s in stats if s["is_no_event"])
    fallback = sum(1 for s in stats if s["used_fallback"])
    no_valid = sum(1 for s in stats if s["valid_frames"] == 0)
    lengths = [s["segment_len"] for s in stats if s["segment_len"] > 0]
    summary: Dict[str, Any] = {
        "dataset": dataset,
        "split": split,
        "total": total,
        "no_event": {"count": no_event, "frac": no_event / total if total else 0.0},
        "fallback": {"count": fallback, "frac": fallback / total if total else 0.0},
        "no_valid_hands": {"count": no_valid, "frac": no_valid / total if total else 0.0},
        "csv_audit": csv_audit,
        "segment_len": {
            "count": len(lengths),
            "mean": float(np.mean(lengths)) if lengths else 0.0,
            "median": float(np.median(lengths)) if lengths else 0.0,
            "min": int(np.min(lengths)) if lengths else 0,
            "max": int(np.max(lengths)) if lengths else 0,
            "p25": float(np.percentile(lengths, 25)) if lengths else 0.0,
            "p75": float(np.percentile(lengths, 75)) if lengths else 0.0,
            "hist": build_len_hist(lengths),
        },
    }
    path = out_dir / "summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, _ = pre.parse_known_args(argv)
    defaults = load_config_section(pre_args.config, "prelabel")

    p = argparse.ArgumentParser(description="BIO prelabeling from MediaPipe keypoints.")
    p.add_argument("--config", type=str, default=pre_args.config, help="Path to config JSON (section: prelabel).")
    p.add_argument(
        "--skeletons",
        default=defaults.get("skeletons"),
        required=_is_missing(defaults.get("skeletons")),
        help="Path to combined JSON or directory with per-video JSON.",
    )
    p.add_argument(
        "--csv",
        default=defaults.get("csv"),
        required=_is_missing(defaults.get("csv")),
        help="CSV with attachment_id/text/begin/end/split.",
    )
    p.add_argument("--split", default=defaults.get("split", "train"))
    p.add_argument(
        "--out",
        default=defaults.get("out"),
        required=_is_missing(defaults.get("out")),
        help="Output directory.",
    )
    p.add_argument("--include_pose", action="store_true", default=_bool_or_default(defaults.get("include_pose"), False))
    p.add_argument("--pose_keep", type=str, default=_csv_or_default(defaults.get("pose_keep"), "0,9,10,11,12,13,14,15,16,23,24"))
    p.add_argument("--pose_vis_thr", type=float, default=float(defaults.get("pose_vis_thr", 0.5)))
    p.add_argument("--hand_thr", type=float, default=float(defaults.get("hand_thr", 0.45)))
    p.add_argument("--hand_thr_fallback", type=float, default=float(defaults.get("hand_thr_fallback", 0.35)))
    p.add_argument("--thr_tune_steps", type=int, default=int(defaults.get("thr_tune_steps", 6)))
    p.add_argument("--thr_tune_step", type=float, default=float(defaults.get("thr_tune_step", 0.05)))
    p.add_argument("--percentile", type=float, default=float(defaults.get("percentile", 70.0)))
    p.add_argument("--pad", type=int, default=int(defaults.get("pad", 4)))
    p.add_argument("--min_len", type=int, default=int(defaults.get("min_len", 8)))
    p.add_argument("--fallback_len", type=int, default=int(defaults.get("fallback_len", 64)))
    p.add_argument("--num_workers", type=int, default=int(defaults.get("num_workers", 0)))
    p.add_argument("--debug_vid", type=str, default=defaults.get("debug_vid"))
    p.add_argument("--smooth_win", type=int, default=int(defaults.get("smooth_win", 7)))
    p.add_argument("--hysteresis", type=float, default=float(defaults.get("hysteresis", 0.6)))
    p.add_argument("--min_motion", type=float, default=float(defaults.get("min_motion", 1e-4)))
    p.add_argument("--min_valid_frames", type=int, default=int(defaults.get("min_valid_frames", 0)))
    p.add_argument("--file_cache", type=int, default=int(defaults.get("file_cache", 0)))
    prefer_pp_default = _bool_or_default(defaults.get("prefer_pp"), True)
    p.add_argument(
        "--prefer_pp",
        dest="prefer_pp",
        action="store_true",
        default=prefer_pp_default,
        help="Prefer *_pp.json when using per-video skeletons (fallback to raw .json).",
    )
    p.add_argument("--no_prefer_pp", dest="prefer_pp", action="store_false")
    trimmed_mode_default = _bool_or_default(defaults.get("trimmed_mode"), False)
    p.add_argument("--trimmed_mode", dest="trimmed_mode", action="store_true", default=trimmed_mode_default)
    p.add_argument("--no_trimmed_mode", dest="trimmed_mode", action="store_false")
    center_default = _bool_or_default(defaults.get("center"), True)
    p.add_argument("--center", dest="center", action="store_true", default=center_default)
    p.add_argument("--no_center", dest="center", action="store_false")
    p.add_argument("--center_mode", type=str, default=defaults.get("center_mode", "masked_mean"), choices=["masked_mean", "wrists"])
    normalize_default = _bool_or_default(defaults.get("normalize"), True)
    p.add_argument("--normalize", dest="normalize", action="store_true", default=normalize_default)
    p.add_argument("--no_normalize", dest="normalize", action="store_false")
    p.add_argument("--norm_method", type=str, default=defaults.get("norm_method", "p95"), choices=["p95", "mad", "max"])
    p.add_argument("--norm_scale", type=float, default=float(defaults.get("norm_scale", 1.0)))
    p.add_argument(
        "--preprocessing_version",
        type=str,
        default=str(defaults.get("preprocessing_version", BIO_PREPROCESSING_VERSION_V3)),
        choices=[BIO_PREPROCESSING_VERSION_V2, BIO_PREPROCESSING_VERSION_V3],
    )
    p.add_argument("--preprocessing_center_alpha", type=float, default=float(defaults.get("preprocessing_center_alpha", 0.2)))
    p.add_argument("--preprocessing_scale_alpha", type=float, default=float(defaults.get("preprocessing_scale_alpha", 0.1)))
    p.add_argument("--preprocessing_min_scale", type=float, default=float(defaults.get("preprocessing_min_scale", 1e-3)))
    p.add_argument(
        "--preprocessing_min_visible_joints_for_scale",
        type=int,
        default=int(defaults.get("preprocessing_min_visible_joints_for_scale", 4)),
    )
    p.add_argument("--log_every", type=int, default=int(defaults.get("log_every", 50)))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    split = _norm_split_name(args.split)
    skeletons_path = Path(args.skeletons)
    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not skeletons_path.exists():
        raise FileNotFoundError(skeletons_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    pose_keep = tuple(int(x) for x in args.pose_keep.split(",") if x.strip())
    min_len = max(1, int(args.min_len))
    fallback_len = max(1, int(args.fallback_len))
    cfg = PrelabelConfig(
        include_pose=args.include_pose,
        pose_keep=pose_keep,
        pose_vis_thr=args.pose_vis_thr,
        hand_score_thr=args.hand_thr,
        hand_score_thr_fallback=args.hand_thr_fallback,
        thr_tune_steps=args.thr_tune_steps,
        thr_tune_step=args.thr_tune_step,
        center=args.center,
        center_mode=args.center_mode,
        normalize=args.normalize,
        norm_method=args.norm_method,
        norm_scale=args.norm_scale,
        smooth_win=args.smooth_win,
        motion_percentile=args.percentile,
        hysteresis=args.hysteresis,
        pad=args.pad,
        min_len=min_len,
        fallback_len=fallback_len,
        min_motion=args.min_motion,
        min_valid_frames=args.min_valid_frames if args.min_valid_frames > 0 else min_len,
        file_cache_size=args.file_cache,
        prefer_pp=args.prefer_pp,
        trimmed_mode=args.trimmed_mode,
        preprocessing_version=str(args.preprocessing_version),
        preprocessing_center_alpha=float(args.preprocessing_center_alpha),
        preprocessing_scale_alpha=float(args.preprocessing_scale_alpha),
        preprocessing_min_scale=float(args.preprocessing_min_scale),
        preprocessing_min_visible_joints_for_scale=int(args.preprocessing_min_visible_joints_for_scale),
    )

    parsed = parse_csv(csv_path, split)
    if args.debug_vid:
        debug_rows = [s for s in parsed.rows if s.vid == args.debug_vid or s.source_group == args.debug_vid or s.signer_id == args.debug_vid]
        if not debug_rows:
            raise RuntimeError(f"No samples found for debug_vid={args.debug_vid}")
        parsed = CsvParseResult(
            rows=tuple(debug_rows),
            rejected=parsed.rejected,
            skipped_split=parsed.skipped_split,
            total_rows=parsed.total_rows,
            columns=parsed.columns,
        )

    csv_audit = write_csv_audit(out_dir, parsed)
    write_run_config(
        out_dir,
        args,
        config_path=args.config,
        section="prelabel",
        extra={"prelabel_cfg": asdict(cfg), "csv_audit": csv_audit},
    )

    store = VideoStore(skeletons_path, cache_size=cfg.file_cache_size, prefer_pp=cfg.prefer_pp)
    mirror_idx = build_mirror_idx(cfg.include_pose, cfg.pose_keep)

    is_dir = skeletons_path.is_dir()
    if args.num_workers > 0 and not is_dir:
        print("Combined JSON detected; forcing num_workers=0 to avoid RAM blowups.")
        args.num_workers = 0
    if args.debug_vid and args.num_workers > 0:
        print("Debug mode enabled; forcing num_workers=0 for deterministic logging.")
        args.num_workers = 0

    tasks: List[Dict[str, Any]] = []
    for sample in parsed.rows:
        vid = sample.vid
        label_str = sample.label_str
        begin = sample.begin
        end = sample.end
        sample_id = sample.sample_id
        out_path = out_dir / f"{sample_id}.npz"
        tasks.append(
            {
                "vid": vid,
                "sample_id": sample_id,
                "label_str": label_str,
                "begin": begin,
                "end": end,
                "split": sample.split,
                "dataset": sample.dataset,
                "signer_id": sample.signer_id,
                "source_group": sample.source_group,
                "out_path": str(out_path),
                "debug": bool(args.debug_vid),
            }
        )

    index_rows: List[Dict[str, Any]] = []
    stats_rows: List[Dict[str, Any]] = []

    if args.num_workers > 0:
        import multiprocessing as mp

        with mp.Pool(
            processes=args.num_workers,
            initializer=_init_worker,
            initargs=(cfg, str(skeletons_path), str(out_dir)),
        ) as pool:
            for i, result in enumerate(pool.imap_unordered(_process_task, tasks), 1):
                index_rows.append(result["index"])
                stats_rows.append(result["stats"])
                if args.log_every and (i % args.log_every == 0):
                    print(f"Processed {i}/{len(tasks)}")
    else:
        for i, task in enumerate(tasks, 1):
            result = process_sample(task, cfg, store, mirror_idx, out_dir)
            index_rows.append(result["index"])
            stats_rows.append(result["stats"])
            if args.log_every and (i % args.log_every == 0):
                print(f"Processed {i}/{len(tasks)}")

    index_rows.sort(key=lambda item: str(item.get("path_to_npz", "")))
    write_index_files(out_dir, index_rows)
    summary = write_summary(out_dir, stats_rows, split=split, dataset="slovo", csv_audit=csv_audit)
    write_dataset_manifest(
        out_dir,
        stage="prelabel",
        args=args,
        config_path=args.config,
        section="prelabel",
        inputs={
            "skeletons": str(skeletons_path),
            "csv": str(csv_path),
        },
        counts={
            "accepted_rows": int(len(parsed.rows)),
            "rejected_rows": int(len(parsed.rejected)),
            "skipped_split": int(parsed.skipped_split),
            "written_samples": int(len(index_rows)),
            "split": split,
        },
        extra={
            "dataset": "slovo",
            "csv_audit": csv_audit,
            "summary": summary,
        },
    )

    print("Done.")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
