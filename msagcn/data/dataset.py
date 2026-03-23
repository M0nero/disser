from __future__ import annotations

import csv
import json
import math
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from .config import DSConfig
from .io import DecodedVideoArrays, DecodedVideoStore, PackedVideoStore, frames_from_combined, read_video_file_nocache
from .topology import (
    CROSS_EDGE_PAIRS_ABS,
    NUM_HAND_JOINTS,
    NUM_HAND_NODES,
    POSE_EDGE_PAIRS_ABS,
    hand_edges_42,
)

__all__ = ["DSConfig", "MultiStreamGestureDataset"]


# --------------------------- Helpers -----------------------------------------

def _build_mask_for_frame(fr: Dict[str, Any], thr: float) -> Tuple[List[int], List[int]]:
    mL = [0] * NUM_HAND_JOINTS
    mR = [0] * NUM_HAND_JOINTS
    L, Ls = fr.get("hand 1"), fr.get("hand 1_score")
    R, Rs = fr.get("hand 2"), fr.get("hand 2_score")
    # Treat None-scores as 0.0 (invalid), require score >= thr
    Ls = float(Ls) if Ls is not None else 0.0
    Rs = float(Rs) if Rs is not None else 0.0
    if L is not None and (Ls >= thr):
        mL = [1] * NUM_HAND_JOINTS
    if R is not None and (Rs >= thr):
        mR = [1] * NUM_HAND_JOINTS
    return mL, mR


def _window_ok(frames: Sequence[Dict[str, Any]], t0: int, T: int, thr: float, ratio: float) -> bool:
    valid = 0
    end = min(t0 + T, len(frames))
    for k in range(t0, end):
        f = frames[k]
        h1s = f.get("hand 1_score")
        h2s = f.get("hand 2_score")
        vL = (f.get("hand 1") is not None) and (float(h1s or 0.0) >= thr)
        vR = (f.get("hand 2") is not None) and (float(h2s or 0.0) >= thr)
        if vL or vR:
            valid += 1
    return valid >= int(ratio * T)


def _best_coverage_window(frames: Sequence[Dict[str, Any]], T: int, thr: float) -> Tuple[int, int]:
    best_t0, best_valid = 0, -1
    for t0 in range(0, max(1, len(frames) - T + 1)):
        v = 0
        for k in range(t0, t0 + T):
            f = frames[k]
            h1s = f.get("hand 1_score")
            h2s = f.get("hand 2_score")
            vL = (f.get("hand 1") is not None) and (float(h1s or 0.0) >= thr)
            vR = (f.get("hand 2") is not None) and (float(h2s or 0.0) >= thr)
            v += 1 if (vL or vR) else 0
        if v > best_valid:
            best_valid = v
            best_t0 = t0
    return best_t0, best_valid


def _best_motion_window(frames: Sequence[Dict[str, Any]], T: int, thr: float) -> int:
    import numpy as np

    def wrist(fr):
        L, Ls = fr.get("hand 1"), fr.get("hand 1_score") or 0.0
        R, Rs = fr.get("hand 2"), fr.get("hand 2_score") or 0.0
        if L is not None and Ls >= thr:
            return L[0]
        if R is not None and Rs >= thr:
            return R[0]
        return None

    coords = []
    for fr in frames:
        w = wrist(fr)
        coords.append(None if w is None else (float(w["x"]), float(w["y"]), float(w["z"])))
    diffs = [0.0]
    for i in range(1, len(coords)):
        a, b = coords[i - 1], coords[i]
        if a is None or b is None:
            diffs.append(0.0)
        else:
            diffs.append(math.dist(a, b))
    diffs = np.array(diffs)
    # moving average to find contiguous motion
    L = 4
    if len(diffs) < L:
        return 0
    csum = np.convolve(diffs, np.ones(L), mode="valid")
    return int(np.argmax(csum))


# --------------------------- Main Dataset ------------------------------------


class MultiStreamGestureDataset(Dataset):
    """
    Gesture dataset with multi-stream (joints, bones, velocity) skeletal data.
    - Supports lazy loading from per-video JSON or combined JSON.
    - Handles pose subset, mask generation, normalization, augmentation, etc.
    """

    def __init__(
        self,
        skeletons_json: str | Path,  # path to combined.json OR directory with per-video *.json
        csv_path: str | Path,
        split: str = "train",
        cfg: Optional[DSConfig] = None,
        label2idx: Optional[Dict[str, int]] = None,
        *,
        use_packed_cache: bool = False,
        packed_cache_dir: Optional[str | Path] = None,
        packed_cache_rebuild: bool = False,
        use_decoded_cache: bool = False,
        decoded_cache_dir: Optional[str | Path] = None,
        decoded_cache_rebuild: bool = False,
    ) -> None:
        super().__init__()
        self.split = split.lower().strip()
        self.cfg = cfg or DSConfig()
        self.skeletons_json = Path(skeletons_json)
        self.csv_path = Path(csv_path)
        if not self.skeletons_json.exists():
            raise FileNotFoundError(self.skeletons_json)
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)

        # 1) CSV samples
        self.samples: List[Tuple[str, str, int, Optional[int]]] = self._parse_csv(self.csv_path, self.split)

        # 2) label map
        if label2idx is None:
            uniq_labels = sorted({lbl for _, lbl, _, _ in self.samples})
            self.label2idx = {l: i for i, l in enumerate(uniq_labels)}
        else:
            self.label2idx = dict(label2idx)
            before = len(self.samples)
            self.samples = [s for s in self.samples if s[1] in self.label2idx]
            if len(self.samples) == 0:
                raise RuntimeError("No samples left after filtering by provided label2idx. Check splits/labels.")
            if len(self.samples) < before:
                drop = before - len(self.samples)
                print(f"[{self.split}] dropped {drop} samples with labels not in train label2idx")

        # 3) Storage mode & meta (do this before pose reorder)
        self._is_dir = self.skeletons_json.is_dir()
        self._skel: Optional[Dict[str, Any]] = None
        self._meta_by_vid: Dict[str, Dict[str, Any]] = {}
        self._file_cache: "OrderedDict[str, Tuple[List[Dict[str, Any]], Dict[str, Any]]]" = OrderedDict()
        self._file_by_vid: Dict[str, Path] = {}
        self._packed_store: Optional[PackedVideoStore] = None
        self._decoded_store: Optional[DecodedVideoStore] = None

        if self._is_dir:
            seen = set()
            for vid, _, _, _ in self.samples:
                if vid in seen:
                    continue
                seen.add(vid)
                f = self._resolve_vid_path(vid)
                self._file_by_vid[vid] = f
            if use_decoded_cache:
                if decoded_cache_dir:
                    cache_dir = Path(decoded_cache_dir)
                else:
                    suffix = "pp" if self.cfg.prefer_pp else "raw"
                    cache_dir = self.skeletons_json.parent / f"{self.skeletons_json.name}__decoded_cache_{suffix}"
                self._decoded_store = DecodedVideoStore.open_or_build(
                    source_dir=self.skeletons_json,
                    cache_dir=cache_dir,
                    prefer_pp=self.cfg.prefer_pp,
                    rebuild=decoded_cache_rebuild,
                    vids=list(self._file_by_vid.keys()),
                )
            elif use_packed_cache:
                if packed_cache_dir:
                    cache_dir = Path(packed_cache_dir)
                else:
                    suffix = "pp" if self.cfg.prefer_pp else "raw"
                    cache_dir = self.skeletons_json.parent / f"{self.skeletons_json.name}__packed_cache_{suffix}"
                self._packed_store = PackedVideoStore.open_or_build(
                    source_dir=self.skeletons_json,
                    cache_dir=cache_dir,
                    prefer_pp=self.cfg.prefer_pp,
                    rebuild=packed_cache_rebuild,
                    vids=list(self._file_by_vid.keys()),
                )
            for vid, path in self._file_by_vid.items():
                if self._decoded_store is not None and self._decoded_store.has_video(vid):
                    self._meta_by_vid[vid] = self._decoded_store.get_meta(vid)
                    continue
                if self._packed_store is not None and self._packed_store.has_video(vid):
                    self._meta_by_vid[vid] = self._packed_store.get_meta(vid)
                    continue
                if not path.exists():
                    self._meta_by_vid[vid] = {}
                    continue
                try:
                    _, meta = read_video_file_nocache(str(path))
                    self._meta_by_vid[vid] = meta
                except Exception:
                    self._meta_by_vid[vid] = {}
        else:
            with self.skeletons_json.open("r", encoding="utf-8") as f:
                self._skel = json.load(f)
            if "videos" in self._skel:
                for vid, blob in self._skel["videos"].items():
                    self._meta_by_vid[vid] = blob.get("meta", {})

        # 4) Pose subset bookkeeping
        self.pose_keep = list(self.cfg.pose_keep)
        self.P = len(self.pose_keep) if self.cfg.include_pose else 0
        if self.cfg.include_pose and self.P > 0:
            self._pose_wrist_out = (
                self.pose_keep.index(15) if 15 in self.pose_keep else -1,
                self.pose_keep.index(16) if 16 in self.pose_keep else -1,
            )
        else:
            self._pose_wrist_out = (-1, -1)
        self.V = NUM_HAND_NODES + self.P

        # 5) Graph & helpers
        self._edges = self._build_edges()
        self._parent_idx = torch.tensor(self._build_parent_map(self._edges), dtype=torch.long)
        self._child_mask = self._parent_idx >= 0
        self._child_idx = torch.where(self._child_mask)[0]
        self._par_idx = self._parent_idx[self._child_idx]
        self._mirror_idx = self._build_mirror_idx()
        self._pose_reorder_by_vid: Dict[str, List[int]] = self._precompute_pose_reorder()

    @staticmethod
    def _parse_csv(csv_path: Path, split: str) -> List[Tuple[str, str, int, Optional[int]]]:
        def _as_bool(s: str) -> Optional[bool]:
            if s is None:
                return None
            v = s.strip().lower()
            if v in ("true", "1", "yes", "y"):
                return True
            if v in ("false", "0", "no", "n"):
                return False
            return None

        rows: List[Tuple[str, str, int, int]] = []
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
            for row in rdr:
                # split select
                use = True
                raw_split = (row.get("split") or "").strip()
                raw_train = row.get("train") or row.get("is_train")

                if raw_split:
                    use = raw_split.strip().lower() == split
                elif raw_train is not None:
                    bt = _as_bool(str(raw_train))
                    if bt is None:
                        use = str(raw_train).strip().lower() == split  # "train"/"val"
                    else:
                        use = bt if split == "train" else (not bt)

                if not use:
                    continue

                vid = (row.get("attachment_id") or "").strip()
                vid = Path(vid).stem
                if not vid:
                    continue

                label = (row.get("text") or "").strip()
                if not label:
                    continue

                # Skip functional background class
                if label.strip().lower() == "no_event":
                    continue
                raw_begin = row.get("begin")
                raw_end = row.get("end")
                try:
                    begin = int(float(raw_begin)) if raw_begin is not None and str(raw_begin).strip() != "" else 0
                except Exception:
                    begin = 0
                try:
                    end = int(float(raw_end)) if raw_end is not None and str(raw_end).strip() != "" else None
                except Exception:
                    end = None
                # Treat end<=0 as "missing end" (common in some annotation exports)
                if end is not None and end <= 0:
                    end = None

                rows.append((vid, label, begin, end))

        if not rows:
            raise RuntimeError(
                "No samples for split after CSV filtering. "
                "Check delimiter (TSV/CSV), column names, and split/train flags."
            )
        return rows

    # --------------------- Graph/mirror ------------------
    def _build_edges(self) -> List[Tuple[int, int]]:
        edges = hand_edges_42()
        if not self.cfg.include_pose or self.P == 0:
            return edges
        pos_map: Dict[int, int] = {abs_idx: i for i, abs_idx in enumerate(self.pose_keep)}
        for a, b in POSE_EDGE_PAIRS_ABS:
            if a in pos_map and b in pos_map:
                ai = NUM_HAND_NODES + pos_map[a]
                bi = NUM_HAND_NODES + pos_map[b]
                edges.append((ai, bi))
        if self.cfg.connect_cross_edges:
            for hand_tag, pose_abs in CROSS_EDGE_PAIRS_ABS:
                if pose_abs not in pos_map:
                    continue
                pi = NUM_HAND_NODES + pos_map[pose_abs]
                hi = 0 if hand_tag == "LWRIST" else 21
                edges.append((hi, pi))
        return edges

    def _build_parent_map(self, edges: List[Tuple[int, int]]) -> List[int]:
        parent = [-1] * self.V
        for p, c in edges:
            parent[c] = p
        return parent

    def _build_mirror_idx(self) -> torch.Tensor:
        idx = list(range(21, 42)) + list(range(0, 21))
        if self.cfg.include_pose and self.P > 0:
            pairs = {(9, 10), (11, 12), (13, 14), (15, 16), (23, 24)}
            pos_map = {abs_i: i for i, abs_i in enumerate(self.pose_keep)}
            for abs_i in self.pose_keep:
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
        return torch.tensor(idx, dtype=torch.long)

    def _precompute_pose_reorder(self) -> Dict[str, List[int]]:
        mapping: Dict[str, List[int]] = {}
        if not (self.cfg.include_pose and self.P > 0):
            return mapping
        keep = list(self.pose_keep)
        default = list(range(self.P))
        for vid in {v for v, _, _, _ in self.samples}:
            meta = self._meta_by_vid.get(vid, {})
            order = meta.get("pose_indices", None)
            if order == "all":
                mapping[vid] = keep[:]  # k_out -> abs_id
            elif isinstance(order, list):
                idxs: List[int] = []
                for want in keep:
                    try:
                        idxs.append(order.index(want))
                    except ValueError:
                        idxs.append(-1)
                mapping[vid] = idxs
            else:
                mapping[vid] = default[:]
        return mapping

    # --------------------- core fetch --------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        mode = "dir" if self._is_dir else "combined"
        return f"MSGDataset(split={self.split}, mode={mode}, samples={len(self)}, V={self.V}, pose={self.cfg.include_pose})"

    def _resolve_vid_path(self, vid: str) -> Path:
        base = self.skeletons_json / f"{vid}.json"
        pp = self.skeletons_json / f"{vid}_pp.json"
        if self.cfg.prefer_pp and pp.exists():
            return pp
        if base.exists():
            return base
        if pp.exists():
            return pp
        return base

    def _get_frames(self, vid: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if self._is_dir:
            if self._packed_store is not None and self._packed_store.has_video(vid):
                return self._packed_store.read_video(vid)
            path_obj = self._file_by_vid.get(vid)
            if path_obj is None:
                path_obj = self._resolve_vid_path(vid)
                self._file_by_vid[vid] = path_obj
            path = str(path_obj)
            # small per-dataset LRU (disabled if size=0)
            if self.cfg.file_cache_size > 0:
                hit = self._file_cache.get(path)
                if hit is not None:
                    self._file_cache.move_to_end(path)
                    return hit
                frames, meta = read_video_file_nocache(path)
                self._file_cache[path] = (frames, meta)
                if len(self._file_cache) > self.cfg.file_cache_size:
                    self._file_cache.popitem(last=False)
                return frames, meta
            return read_video_file_nocache(path)
        return frames_from_combined(self._skel, vid)

    def _get_decoded_video(self, vid: str) -> Optional[DecodedVideoArrays]:
        if self._is_dir and self._decoded_store is not None and self._decoded_store.has_video(vid):
            return self._decoded_store.read_video(vid)
        return None

    @staticmethod
    def _decoded_valid_any(video: DecodedVideoArrays, t0: int, t1: int, thr: float) -> np.ndarray:
        left_valid = video.left_score[t0:t1] >= float(thr)
        right_valid = video.right_score[t0:t1] >= float(thr)
        return np.logical_or(left_valid, right_valid)

    @staticmethod
    def _best_coverage_window_decoded(valid_any: np.ndarray, T: int) -> Tuple[int, int]:
        if valid_any.size <= 0:
            return 0, 0
        if valid_any.size <= T:
            return 0, int(valid_any.sum())
        kernel = np.ones((T,), dtype=np.int32)
        wins = np.convolve(valid_any.astype(np.int32), kernel, mode="valid")
        best_t0 = int(np.argmax(wins))
        return best_t0, int(wins[best_t0])

    def _best_motion_window_decoded(self, video: DecodedVideoArrays, t0: int, t1: int, T: int, thr: float) -> int:
        left_valid = video.left_score[t0:t1] >= float(thr)
        right_valid = video.right_score[t0:t1] >= float(thr)
        wrists = np.zeros((t1 - t0, 3), dtype=np.float32)
        have = np.zeros((t1 - t0,), dtype=bool)
        if left_valid.any():
            wrists[left_valid] = video.left_xyz[t0:t1, 0, :][left_valid]
            have[left_valid] = True
        use_right = np.logical_and(~have, right_valid)
        if use_right.any():
            wrists[use_right] = video.right_xyz[t0:t1, 0, :][use_right]
            have[use_right] = True
        diffs = np.zeros((t1 - t0,), dtype=np.float32)
        if diffs.size <= 1:
            return 0
        consecutive = np.logical_and(have[1:], have[:-1])
        if consecutive.any():
            delta = wrists[1:] - wrists[:-1]
            diffs[1:][consecutive] = np.linalg.norm(delta[consecutive], axis=1)
        kernel = np.ones((4,), dtype=np.float32)
        if diffs.size < kernel.size:
            return 0
        smoothed = np.convolve(diffs, kernel, mode="valid")
        return int(np.argmax(smoothed))

    def _select_window_decoded(self, video: DecodedVideoArrays, seg_begin: int, seg_end: int, T: int, thr: float, ratio: float) -> Tuple[int, float, float]:
        seg_len = max(0, int(seg_end) - int(seg_begin))
        if seg_len >= T:
            mode = str(getattr(self.cfg, "temporal_crop", "random")).strip().lower()
            if self.split != "train":
                mode = "best" if mode == "random" else mode
            if mode == "center":
                return max(0, (seg_len - T) // 2), thr, ratio

            valid_any = self._decoded_valid_any(video, seg_begin, seg_end, thr)
            if mode == "random" and self.split == "train":
                max_t0 = max(0, seg_len - T)
                for _ in range(8):
                    tt = random.randint(0, max_t0)
                    if int(valid_any[tt : tt + T].sum()) >= int(ratio * T):
                        return tt, thr, ratio

            t_cov, valid = self._best_coverage_window_decoded(valid_any, T)
            if valid >= int(ratio * T):
                return t_cov, thr, ratio

            best_t0, best_thr, best_cov = t_cov, thr, valid / float(T)
            steps = max(0, int(getattr(self.cfg, "thr_tune_steps", 0)))
            step_size = float(getattr(self.cfg, "thr_tune_step", 0.05))
            for s in range(1, steps + 1):
                thr_i = max(0.0, thr - s * step_size)
                valid_i = self._decoded_valid_any(video, seg_begin, seg_end, thr_i)
                t_i, v_i = self._best_coverage_window_decoded(valid_i, T)
                cov_i = v_i / float(T)
                if (cov_i > best_cov) or (abs(cov_i - best_cov) < 1e-6 and thr_i > best_thr):
                    best_t0, best_thr, best_cov = t_i, thr_i, cov_i
                if v_i >= int(ratio * T):
                    return t_i, thr_i, ratio

            valid_fallback = self._decoded_valid_any(video, seg_begin, seg_end, self.cfg.hand_score_thr_fallback)
            t_cov2, valid2 = self._best_coverage_window_decoded(valid_fallback, T)
            if valid2 >= int(self.cfg.window_valid_ratio_fallback * T):
                return t_cov2, self.cfg.hand_score_thr_fallback, self.cfg.window_valid_ratio_fallback

            if best_cov > 0:
                return best_t0, best_thr, ratio

            t_mov = self._best_motion_window_decoded(video, seg_begin, seg_end, T, thr)
            return t_mov, thr, ratio
        return 0, thr, ratio

    def _select_thr_for_segment_decoded(self, video: DecodedVideoArrays, seg_begin: int, seg_end: int, thr: float, ratio: float) -> Tuple[float, float]:
        valid_any = self._decoded_valid_any(video, seg_begin, seg_end, thr)
        cov0 = float(valid_any.mean()) if valid_any.size > 0 else 0.0
        if cov0 >= ratio:
            return thr, ratio

        best_thr, best_cov = thr, cov0
        steps = max(0, int(getattr(self.cfg, "thr_tune_steps", 0)))
        step_size = float(getattr(self.cfg, "thr_tune_step", 0.05))
        for s in range(1, steps + 1):
            thr_i = max(0.0, thr - s * step_size)
            valid_i = self._decoded_valid_any(video, seg_begin, seg_end, thr_i)
            cov_i = float(valid_i.mean()) if valid_i.size > 0 else 0.0
            if (cov_i > best_cov) or (abs(cov_i - best_cov) < 1e-6 and thr_i > best_thr):
                best_thr, best_cov = thr_i, cov_i
            if cov_i >= ratio:
                return thr_i, ratio

        valid_fallback = self._decoded_valid_any(video, seg_begin, seg_end, self.cfg.hand_score_thr_fallback)
        cov_f = float(valid_fallback.mean()) if valid_fallback.size > 0 else 0.0
        if cov_f >= self.cfg.window_valid_ratio_fallback:
            return self.cfg.hand_score_thr_fallback, self.cfg.window_valid_ratio_fallback
        return best_thr, ratio

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        vid, label_str, begin_hint, end_hint = self.samples[idx]
        decoded_video = self._get_decoded_video(vid)
        if decoded_video is not None:
            return self._getitem_from_decoded(idx, vid, label_str, begin_hint, end_hint, decoded_video)
        frames, vmeta = self._get_frames(vid)
        if not frames:
            raise IndexError(f"No frames for {vid}")

        T_total = len(frames)
        begin = int(begin_hint)
        end_hint_val = None if end_hint is None else int(end_hint)
        if end_hint_val is not None and end_hint_val < 0:
            end_hint_val = None

        end_incl = None
        if end_hint_val is not None:
            end_incl = end_hint_val if self.cfg.end_inclusive else (end_hint_val - 1)
            if end_incl < begin:
                end_incl = begin
            seg_len = end_incl - begin + 1

            if seg_len >= T_total:
                begin = 0
                end_incl = T_total - 1
            else:
                if begin < 0:
                    begin = 0
                    end_incl = begin + seg_len - 1
                if end_incl >= T_total:
                    end_incl = T_total - 1
                    begin = end_incl - seg_len + 1
                    if begin < 0:
                        begin = 0
                if end_incl < begin:
                    end_incl = begin

            end_excl = end_incl + 1
        else:
            # No end provided -> use from begin to end-of-video.
            if begin < 0:
                begin = 0
            if begin >= T_total:
                begin = 0
            end_excl = T_total

        # --- Small boundary jitter (train only) ---
        if (self.split == "train") and (end_hint_val is not None):
            p = float(getattr(self.cfg, "boundary_jitter_prob", 0.0))
            jmax = int(getattr(self.cfg, "boundary_jitter_max", 0))
            if jmax > 0 and random.random() < p:
                jb = random.randint(-jmax, jmax)
                je = random.randint(-jmax, jmax)
                begin = begin + jb
                end_excl = end_excl + je

        begin = max(0, min(int(begin), T_total - 1))
        end_excl = max(begin + 1, min(int(end_excl), T_total))
        seg = frames[begin:end_excl]

        T = self.cfg.max_frames
        mode = str(getattr(self.cfg, "temporal_crop", "random")).strip().lower()

        # "resample" keeps the whole segment and time-resamples it to exactly T frames.
        if mode == "resample":
            thr, ratio = self._select_thr_for_segment(seg, self.cfg.hand_score_thr, self.cfg.window_valid_ratio)
            idxs = self._resample_indices(len(seg), T)

            # --- Mild speed perturbation (train only) ---
            if self.split == "train":
                sp = float(getattr(self.cfg, "speed_perturb_prob", 0.0))
                kmin = int(getattr(self.cfg, "speed_perturb_kmin", T))
                kmax = int(getattr(self.cfg, "speed_perturb_kmax", T))
                if sp > 0.0 and random.random() < sp:
                    lo, hi = (kmin, kmax) if kmin <= kmax else (kmax, kmin)
                    K = random.randint(max(2, lo), max(2, hi))
                    idxK = self._resample_indices(len(seg), K)
                    mapT = self._resample_indices(K, T)
                    idxs = [idxK[j] for j in mapT]

            window = [seg[i] for i in idxs]
            # for meta: window is derived from the entire [begin,end) segment
            t0 = 0
            t1_src_excl = len(seg)
        else:
            t0, thr, ratio = self._select_window(seg, T, self.cfg.hand_score_thr, self.cfg.window_valid_ratio)
            window = self._slice_with_pad(seg, t0, T)
            t1_src_excl = min(len(seg), t0 + T)

        # allocate
        pts = torch.zeros((T, self.V, 3), dtype=torch.float32)
        mask = torch.zeros((T, self.V, 1), dtype=torch.float32)
        ts = torch.zeros((T,), dtype=torch.float32)

        # pose reorder for this video
        reorder = self._pose_reorder_by_vid.get(vid, None)
        wrist_left_out, wrist_right_out = self._pose_wrist_out
        coords_tag = str(vmeta.get("coords", "image")).lower()
        allow_pose_wrist = (
            self.cfg.include_pose
            and self.P > 0
            and (wrist_left_out >= 0 or wrist_right_out >= 0)
            and coords_tag != "world"
        )

        for t, fr in enumerate(window):
            ts[t] = float(fr.get("ts", 0.0))
            pose_wrist_left = None
            pose_wrist_right = None
            pose_wrist_left_ok = False
            pose_wrist_right_ok = False
            # hands
            mL, mR = _build_mask_for_frame(fr, thr)
            L = fr.get("hand 1") if (mL[0] == 1) else None
            R = fr.get("hand 2") if (mR[0] == 1) else None
            if L is not None:
                for j in range(NUM_HAND_JOINTS):
                    p = L[j]
                    pts[t, j, 0] = float(p["x"])
                    pts[t, j, 1] = float(p["y"])
                    pts[t, j, 2] = float(p["z"])
                    mask[t, j, 0] = 1.0
            if R is not None:
                for j in range(NUM_HAND_JOINTS):
                    p = R[j]
                    pts[t, 21 + j, 0] = float(p["x"])
                    pts[t, 21 + j, 1] = float(p["y"])
                    pts[t, 21 + j, 2] = float(p["z"])
                    mask[t, 21 + j, 0] = 1.0

            # pose (optional)
            if self.cfg.include_pose and self.P > 0:
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
                            if float(pose_vis[k_in]) >= self.cfg.pose_vis_thr:
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

            # pose -> wrist fallback when hand is missing
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

        # temporal de-flicker for masks
        m = mask.permute(1, 2, 0)  # (V,1,T)
        m = F.avg_pool1d(m, kernel_size=3, stride=1, padding=1)
        m = (m >= 0.5).float()
        mask = m.permute(2, 0, 1).contiguous()

        # single-hand canonicalization (right->left) if strongly one-sided
        left_cnt = float(mask[:, 0:21, :].sum())
        right_cnt = float(mask[:, 21:42, :].sum())
        T_frames = mask.shape[0]
        if left_cnt < 0.1 * T_frames * 21 and right_cnt >= 0.6 * T_frames * 21:
            pts = pts[:, self._mirror_idx, :]
            mask = mask[:, self._mirror_idx, :]

        # augment
        if self.cfg.augment and self.split == "train":
            coords = vmeta.get("coords", "world")
            pts, mask = self._apply_spatial_aug(pts, mask, coords_world=(coords == "world"))
            # time-dropout
            if self.cfg.time_drop_prob > 0.0 and random.random() < self.cfg.time_drop_prob:
                L = random.randint(2, max(2, int(0.1 * pts.shape[0])))
                s = random.randint(0, pts.shape[0] - L)
                pts[s : s + L] = 0.0
                mask[s : s + L] = 0.0
            # hand-dropout (whole-hand)
            if self.cfg.hand_drop_prob > 0.0 and random.random() < self.cfg.hand_drop_prob:
                if random.random() < 0.5:
                    pts[:, 0:21, :] = 0.0
                    mask[:, 0:21, :] = 0.0
                else:
                    pts[:, 21:42, :] = 0.0
                    mask[:, 21:42, :] = 0.0

        # guard + center + normalize
        pts = torch.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)
        pts = self._center_norm(pts, mask)

        out: Dict[str, Tensor] = {}
        # joints
        if "joints" in self.cfg.use_streams:
            out["joints"] = pts.permute(2, 1, 0).contiguous()  # (C,V,T)

        # velocity (dt-aware)
        if "velocity" in self.cfg.use_streams:
            vel = torch.zeros_like(pts)
            use_ts = ts.numel() > 1 and torch.isfinite(ts).all() and float((ts[1:] - ts[:-1]).abs().sum()) > 0.0
            if use_ts:
                dt_ms = torch.zeros_like(ts)
                dt_ms[1:] = torch.clamp(ts[1:] - ts[:-1], min=1.0)
                dt_s = dt_ms / 1000.0
            else:
                fps = float(vmeta.get("fps", 0.0) or 0.0)
                if fps <= 0.0:
                    fps = 30.0
                dt_s = torch.full_like(ts, 1.0 / max(fps, 1e-6))
            vel[1:] = (pts[1:] - pts[:-1]) / dt_s[1:].view(-1, 1, 1).clamp_min(1e-6)
            out["velocity"] = vel.permute(2, 1, 0).contiguous()

        # bones (vectorized)
        if "bones" in self.cfg.use_streams:
            bones = torch.zeros_like(pts)
            cidx = self._child_idx
            pidx = self._par_idx
            bones[:, cidx, :] = pts[:, cidx, :] - pts[:, pidx, :]
            out["bones"] = bones.permute(2, 1, 0).contiguous()

        out["mask"] = mask.permute(2, 1, 0).contiguous()
        out["label"] = torch.tensor(self.label2idx[label_str], dtype=torch.long)

        achieved = float(mask.mean())
        out["meta"] = {
            "video": vid,
            "clip_begin": int(begin),
            "clip_end_excl": int(end_excl),
            "temporal": mode,
            # t0/t1 are *source* indices within the original video timeline
            "t0": int(begin + (t0 if mode != "resample" else 0)),
            "t1": int(begin + (t1_src_excl if mode == "resample" else (t0 + T))),
            "V": int(self.V),
            "thr_used": float(thr),
            "target_ratio": float(ratio),
            "achieved_ratio": achieved,
            "src_len": int(len(seg)),
            "out_len": int(T),
        }
        return out

    def _getitem_from_decoded(
        self,
        idx: int,
        vid: str,
        label_str: str,
        begin_hint: int,
        end_hint: Optional[int],
        video: DecodedVideoArrays,
    ) -> Dict[str, Any]:
        del idx
        vmeta = video.meta
        T_total = int(video.ts.shape[0])
        if T_total <= 0:
            raise IndexError(f"No frames for {vid}")

        begin = int(begin_hint)
        end_hint_val = None if end_hint is None else int(end_hint)
        if end_hint_val is not None and end_hint_val < 0:
            end_hint_val = None

        end_incl = None
        if end_hint_val is not None:
            end_incl = end_hint_val if self.cfg.end_inclusive else (end_hint_val - 1)
            if end_incl < begin:
                end_incl = begin
            seg_len = end_incl - begin + 1

            if seg_len >= T_total:
                begin = 0
                end_incl = T_total - 1
            else:
                if begin < 0:
                    begin = 0
                    end_incl = begin + seg_len - 1
                if end_incl >= T_total:
                    end_incl = T_total - 1
                    begin = end_incl - seg_len + 1
                    if begin < 0:
                        begin = 0
                if end_incl < begin:
                    end_incl = begin
            end_excl = end_incl + 1
        else:
            if begin < 0:
                begin = 0
            if begin >= T_total:
                begin = 0
            end_excl = T_total

        if (self.split == "train") and (end_hint_val is not None):
            p = float(getattr(self.cfg, "boundary_jitter_prob", 0.0))
            jmax = int(getattr(self.cfg, "boundary_jitter_max", 0))
            if jmax > 0 and random.random() < p:
                begin = begin + random.randint(-jmax, jmax)
                end_excl = end_excl + random.randint(-jmax, jmax)

        begin = max(0, min(int(begin), T_total - 1))
        end_excl = max(begin + 1, min(int(end_excl), T_total))
        seg_len = int(end_excl - begin)

        T = self.cfg.max_frames
        mode = str(getattr(self.cfg, "temporal_crop", "random")).strip().lower()
        if mode == "resample":
            thr, ratio = self._select_thr_for_segment_decoded(video, begin, end_excl, self.cfg.hand_score_thr, self.cfg.window_valid_ratio)
            src_indices = np.asarray(self._resample_indices(seg_len, T), dtype=np.int64)
            if self.split == "train":
                sp = float(getattr(self.cfg, "speed_perturb_prob", 0.0))
                kmin = int(getattr(self.cfg, "speed_perturb_kmin", T))
                kmax = int(getattr(self.cfg, "speed_perturb_kmax", T))
                if sp > 0.0 and random.random() < sp:
                    lo, hi = (kmin, kmax) if kmin <= kmax else (kmax, kmin)
                    K = random.randint(max(2, lo), max(2, hi))
                    idx_k = self._resample_indices(seg_len, K)
                    map_t = self._resample_indices(K, T)
                    src_indices = np.asarray([idx_k[j] for j in map_t], dtype=np.int64)
            src_indices = src_indices + int(begin)
            t0 = 0
            t1_src_excl = seg_len
        else:
            t0, thr, ratio = self._select_window_decoded(video, begin, end_excl, T, self.cfg.hand_score_thr, self.cfg.window_valid_ratio)
            max_src = end_excl - 1
            src_indices = np.asarray([min(begin + t0 + i, max_src) for i in range(T)], dtype=np.int64)
            t1_src_excl = min(seg_len, t0 + T)

        pts = torch.zeros((T, self.V, 3), dtype=torch.float32)
        mask = torch.zeros((T, self.V, 1), dtype=torch.float32)

        ts_np = np.ascontiguousarray(video.ts[src_indices], dtype=np.float32)
        ts = torch.from_numpy(ts_np)

        left_xyz_np = np.ascontiguousarray(video.left_xyz[src_indices], dtype=np.float32)
        right_xyz_np = np.ascontiguousarray(video.right_xyz[src_indices], dtype=np.float32)
        left_scores_np = np.ascontiguousarray(video.left_score[src_indices], dtype=np.float32)
        right_scores_np = np.ascontiguousarray(video.right_score[src_indices], dtype=np.float32)

        left_xyz = torch.from_numpy(left_xyz_np)
        right_xyz = torch.from_numpy(right_xyz_np)
        left_valid = torch.from_numpy((left_scores_np >= float(thr)).astype(np.float32))
        right_valid = torch.from_numpy((right_scores_np >= float(thr)).astype(np.float32))

        pts[:, 0:21, :] = left_xyz * left_valid.view(T, 1, 1)
        pts[:, 21:42, :] = right_xyz * right_valid.view(T, 1, 1)
        mask[:, 0:21, 0] = left_valid.view(T, 1)
        mask[:, 21:42, 0] = right_valid.view(T, 1)

        wrist_left_out, wrist_right_out = self._pose_wrist_out
        coords_tag = str(vmeta.get("coords", "image")).lower()
        allow_pose_wrist = (
            self.cfg.include_pose
            and self.P > 0
            and (wrist_left_out >= 0 or wrist_right_out >= 0)
            and coords_tag != "world"
        )

        if self.cfg.include_pose and self.P > 0:
            pose_xyz_np = np.ascontiguousarray(video.pose_xyz[src_indices][:, self.pose_keep, :], dtype=np.float32)
            pose_vis_np = np.ascontiguousarray(video.pose_vis[src_indices][:, self.pose_keep], dtype=np.float32)
            pose_xyz = torch.from_numpy(pose_xyz_np)
            pose_valid = torch.from_numpy((pose_vis_np >= float(self.cfg.pose_vis_thr)).astype(np.float32))
            pts[:, NUM_HAND_NODES : NUM_HAND_NODES + self.P, :] = pose_xyz
            mask[:, NUM_HAND_NODES : NUM_HAND_NODES + self.P, 0] = pose_valid

            if allow_pose_wrist:
                if wrist_left_out >= 0:
                    left_pose_valid = pose_valid[:, wrist_left_out] > 0.0
                    missing_left = mask[:, 0, 0] == 0.0
                    use_left = torch.logical_and(left_pose_valid, missing_left)
                    if bool(use_left.any()):
                        pts[use_left, 0, :] = pose_xyz[use_left, wrist_left_out, :]
                        mask[use_left, 0, 0] = 1.0
                if wrist_right_out >= 0:
                    right_pose_valid = pose_valid[:, wrist_right_out] > 0.0
                    missing_right = mask[:, 21, 0] == 0.0
                    use_right = torch.logical_and(right_pose_valid, missing_right)
                    if bool(use_right.any()):
                        pts[use_right, 21, :] = pose_xyz[use_right, wrist_right_out, :]
                        mask[use_right, 21, 0] = 1.0

        m = mask.permute(1, 2, 0)
        m = F.avg_pool1d(m, kernel_size=3, stride=1, padding=1)
        m = (m >= 0.5).float()
        mask = m.permute(2, 0, 1).contiguous()

        left_cnt = float(mask[:, 0:21, :].sum())
        right_cnt = float(mask[:, 21:42, :].sum())
        T_frames = mask.shape[0]
        if left_cnt < 0.1 * T_frames * 21 and right_cnt >= 0.6 * T_frames * 21:
            pts = pts[:, self._mirror_idx, :]
            mask = mask[:, self._mirror_idx, :]

        if self.cfg.augment and self.split == "train":
            coords = vmeta.get("coords", "world")
            pts, mask = self._apply_spatial_aug(pts, mask, coords_world=(coords == "world"))
            if self.cfg.time_drop_prob > 0.0 and random.random() < self.cfg.time_drop_prob:
                L = random.randint(2, max(2, int(0.1 * pts.shape[0])))
                s = random.randint(0, pts.shape[0] - L)
                pts[s : s + L] = 0.0
                mask[s : s + L] = 0.0
            if self.cfg.hand_drop_prob > 0.0 and random.random() < self.cfg.hand_drop_prob:
                if random.random() < 0.5:
                    pts[:, 0:21, :] = 0.0
                    mask[:, 0:21, :] = 0.0
                else:
                    pts[:, 21:42, :] = 0.0
                    mask[:, 21:42, :] = 0.0

        pts = torch.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)
        pts = self._center_norm(pts, mask)

        out: Dict[str, Tensor] = {}
        if "joints" in self.cfg.use_streams:
            out["joints"] = pts.permute(2, 1, 0).contiguous()

        if "velocity" in self.cfg.use_streams:
            vel = torch.zeros_like(pts)
            use_ts = ts.numel() > 1 and torch.isfinite(ts).all() and float((ts[1:] - ts[:-1]).abs().sum()) > 0.0
            if use_ts:
                dt_ms = torch.zeros_like(ts)
                dt_ms[1:] = torch.clamp(ts[1:] - ts[:-1], min=1.0)
                dt_s = dt_ms / 1000.0
            else:
                fps = float(vmeta.get("fps", 0.0) or 0.0)
                if fps <= 0.0:
                    fps = 30.0
                dt_s = torch.full_like(ts, 1.0 / max(fps, 1e-6))
            vel[1:] = (pts[1:] - pts[:-1]) / dt_s[1:].view(-1, 1, 1).clamp_min(1e-6)
            out["velocity"] = vel.permute(2, 1, 0).contiguous()

        if "bones" in self.cfg.use_streams:
            bones = torch.zeros_like(pts)
            cidx = self._child_idx
            pidx = self._par_idx
            bones[:, cidx, :] = pts[:, cidx, :] - pts[:, pidx, :]
            out["bones"] = bones.permute(2, 1, 0).contiguous()

        out["mask"] = mask.permute(2, 1, 0).contiguous()
        out["label"] = torch.tensor(self.label2idx[label_str], dtype=torch.long)

        achieved = float(mask.mean())
        out["meta"] = {
            "video": vid,
            "clip_begin": int(begin),
            "clip_end_excl": int(end_excl),
            "temporal": mode,
            "t0": int(begin + (t0 if mode != "resample" else 0)),
            "t1": int(begin + (t1_src_excl if mode == "resample" else (t0 + T))),
            "V": int(self.V),
            "thr_used": float(thr),
            "target_ratio": float(ratio),
            "achieved_ratio": achieved,
            "src_len": int(seg_len),
            "out_len": int(T),
        }
        return out

    # --------------------- internals --------------------
    def _select_window(self, seg: List[Dict[str, Any]], T: int, thr: float, ratio: float) -> Tuple[int, float, float]:
        if len(seg) >= T:
            mode = str(getattr(self.cfg, "temporal_crop", "random")).strip().lower()
            # Never use randomness in eval splits to keep metrics stable.
            if self.split != "train":
                mode = "best" if mode == "random" else mode

            # 0) Center crop (deterministic)
            if mode == "center":
                t_center = max(0, (len(seg) - T) // 2)
                return t_center, thr, ratio

            # Fast path: try a few random windows (train)
            if mode == "random" and self.split == "train":
                for _ in range(8):
                    tt = random.randint(0, len(seg) - T)
                    if _window_ok(seg, tt, T, thr, ratio):
                        return tt, thr, ratio

            # 1) Best window at base threshold
            t_cov, valid = _best_coverage_window(seg, T, thr)
            if valid >= int(ratio * T):
                return t_cov, thr, ratio

            # 2) Threshold tuning downward to try to meet target ratio
            best_t0, best_thr, best_cov = t_cov, thr, valid / float(T)
            steps = max(0, int(getattr(self.cfg, "thr_tune_steps", 0)))
            step_size = float(getattr(self.cfg, "thr_tune_step", 0.05))
            for s in range(1, steps + 1):
                thr_i = max(0.0, thr - s * step_size)
                t_i, v_i = _best_coverage_window(seg, T, thr_i)
                cov_i = v_i / float(T)
                # keep best coverage; prefer higher threshold on ties
                if (cov_i > best_cov) or (abs(cov_i - best_cov) < 1e-6 and thr_i > best_thr):
                    best_t0, best_thr, best_cov = t_i, thr_i, cov_i
                if v_i >= int(ratio * T):
                    return t_i, thr_i, ratio

            # 3) Explicit fallback threshold/ratio
            t_cov2, valid2 = _best_coverage_window(seg, T, self.cfg.hand_score_thr_fallback)
            if valid2 >= int(self.cfg.window_valid_ratio_fallback * T):
                return t_cov2, self.cfg.hand_score_thr_fallback, self.cfg.window_valid_ratio_fallback

            # 4) If we saw any valid frames during tuning - take the best we found
            if best_cov > 0:
                return best_t0, best_thr, ratio

            # 5) Motion-based fallback (may yield 0 valid)
            t_mov = _best_motion_window(seg, T, thr)
            return t_mov, thr, ratio
        return 0, thr, ratio

    def _segment_coverage(self, seg: List[Dict[str, Any]], thr: float) -> float:
        """Fraction of frames in seg that have at least one valid hand at threshold thr."""
        if not seg:
            return 0.0
        valid = 0
        for f in seg:
            h1s = f.get("hand 1_score")
            h2s = f.get("hand 2_score")
            vL = (f.get("hand 1") is not None) and (float(h1s or 0.0) >= thr)
            vR = (f.get("hand 2") is not None) and (float(h2s or 0.0) >= thr)
            valid += 1 if (vL or vR) else 0
        return valid / float(max(1, len(seg)))

    def _select_thr_for_segment(self, seg: List[Dict[str, Any]], thr: float, ratio: float) -> Tuple[float, float]:
        """Pick hand-score threshold for *whole segment* (used by temporal_crop='resample')."""
        cov0 = self._segment_coverage(seg, thr)
        if cov0 >= ratio:
            return thr, ratio

        best_thr, best_cov = thr, cov0
        steps = max(0, int(getattr(self.cfg, "thr_tune_steps", 0)))
        step_size = float(getattr(self.cfg, "thr_tune_step", 0.05))
        for s in range(1, steps + 1):
            thr_i = max(0.0, thr - s * step_size)
            cov_i = self._segment_coverage(seg, thr_i)
            if (cov_i > best_cov) or (abs(cov_i - best_cov) < 1e-6 and thr_i > best_thr):
                best_thr, best_cov = thr_i, cov_i
            if cov_i >= ratio:
                return thr_i, ratio

        # explicit fallback threshold/ratio
        cov_f = self._segment_coverage(seg, self.cfg.hand_score_thr_fallback)
        if cov_f >= self.cfg.window_valid_ratio_fallback:
            return self.cfg.hand_score_thr_fallback, self.cfg.window_valid_ratio_fallback

        return best_thr, ratio

    @staticmethod
    def _resample_indices(L: int, T: int) -> List[int]:
        """Uniformly sample T indices from [0..L-1] (inclusive ends), with repetition if L < T."""
        if T <= 0:
            return []
        if L <= 1:
            return [0] * T
        if T == 1:
            return [0]
        # integer linspace with inclusive endpoints
        out = [int(round(i * (L - 1) / float(T - 1))) for i in range(T)]
        out[0] = 0
        out[-1] = L - 1
        # clamp just in case of rounding edge-cases
        out = [min(max(x, 0), L - 1) for x in out]
        return out

    def _slice_with_pad(self, seg: List[Dict[str, Any]], t0: int, T: int) -> List[Dict[str, Any]]:
        if t0 + T <= len(seg):
            return seg[t0 : t0 + T]
        win = list(seg[t0:]) or [seg[-1]]
        while len(win) < T:
            win.append(win[-1])
        return win

    def _apply_spatial_aug(self, pts: Tensor, mask: Tensor, coords_world: bool) -> Tuple[Tensor, Tensor]:
        # mirror
        if random.random() < self.cfg.mirror_prob:
            pts = pts[:, self._mirror_idx, :]
            mask = mask[:, self._mirror_idx, :]
            if (not coords_world) and (not self.cfg.mirror_swap_only):
                pts[..., 0] = -pts[..., 0]
        # rotation around Z (XY)
        ang = math.radians(random.uniform(-self.cfg.rot_deg, self.cfg.rot_deg))
        cos, sin = math.cos(ang), math.sin(ang)
        xy = pts[..., :2].reshape(-1, 2)
        R = torch.tensor([[cos, -sin], [sin, cos]], dtype=pts.dtype, device=pts.device)
        xy = (R @ xy.T).T.reshape(pts.shape[0], pts.shape[1], 2)
        pts = torch.cat([xy, pts[..., 2:3]], dim=2)
        # scale
        s = random.uniform(1 - self.cfg.scale_jitter, 1 + self.cfg.scale_jitter)
        pts = pts * s
        # noise
        if self.cfg.noise_sigma > 0:
            pts = pts + torch.randn_like(pts) * self.cfg.noise_sigma
        return pts, mask

    def _center_norm(self, pts: Tensor, mask: Tensor) -> Tensor:
        if self.cfg.center:
            if self.cfg.center_mode == "wrists":
                wr = []
                if float(mask[:, 0, :].sum()) > 0:
                    wr.append(pts[:, 0:1, :])
                if mask.shape[1] > 21 and float(mask[:, 21, :].sum()) > 0:
                    wr.append(pts[:, 21:22, :])
                if wr:
                    c = torch.cat(wr, dim=1).mean(dim=1, keepdim=True)  # (T,1,3)
                    pts = pts - c
                else:
                    denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
                    mean = (pts * mask).sum(dim=1, keepdim=True) / denom
                    pts = pts - mean
            else:
                denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
                mean = (pts * mask).sum(dim=1, keepdim=True) / denom
                pts = pts - mean
        if self.cfg.normalize:
            if self.cfg.norm_method == "p95":
                flat = pts.abs().reshape(-1)
                k = max(1, int(flat.numel() * 0.95))
                span = flat.kthvalue(k)[0]
            elif self.cfg.norm_method == "mad":
                med = pts.median().values
                mad = (pts - med).abs().median().values * 1.4826
                span = torch.clamp(mad, min=1e-6)
            else:  # 'max'
                span = pts.abs().amax()
            if float(span) > 1e-6:
                scale = span / max(1e-6, float(self.cfg.norm_scale))
                pts = pts / (scale + 1e-6)
        return pts

    # --------------------- collate & adjacency -----------
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]):
        ys = torch.stack([b["label"] for b in batch], dim=0)
        metas = [b["meta"] for b in batch]
        out: Dict[str, Tensor] = {}
        for k in ("joints", "bones", "velocity", "mask"):
            if k in batch[0]:
                out[k] = torch.stack([b[k] for b in batch], dim=0)
        return out, ys, metas

    def build_adjacency(self, normalize: bool | str = True) -> torch.Tensor:
        V = self.V
        A = torch.zeros((V, V), dtype=torch.float32)
        for (p, c) in self._edges:
            A[p, c] = 1.0
            A[c, p] = 1.0
        # normalize can be: True/"row" -> row-normalize, "sym" -> D^{-1/2} A D^{-1/2}, False -> raw
        if normalize is True or normalize == "row":
            d = A.sum(dim=1, keepdim=True).clamp_min(1e-6)
            A = A / d
        elif normalize in ("sym", "symmetric", "symm"):
            d = A.sum(dim=1).clamp_min(1e-6)
            D_inv_sqrt = torch.diag(torch.pow(d, -0.5))
            A = D_inv_sqrt @ A @ D_inv_sqrt
        # else: False -> raw adjacency
        return A
