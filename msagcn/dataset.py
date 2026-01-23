"""dataset.py — GestureDataset v7 (legacy CSV + fully vectorized augs)
════════════════════════════════════════════════════════════════

*Поддерживается только старая аннотация*  
`attachment_id, text, train, begin, end`

Главное: аугментации теперь полностью на тензорах **без циклов** —
DataLoader‑воркеры готовят батч, пока GPU считает, и загрузка не «пилит».
Сохранил *всё*, что было в v5: center/normalize, velocity‑канал, time‑resample.
"""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

__all__ = ["GestureDataset"]

# mirror index: swap руки (0‑20) ↔ (21‑41) --------------------------------------
_MIRROR_IDX = torch.tensor(list(range(21, 42)) + list(range(21)))

# -----------------------------------------------------------------------------
# Main dataset class
# -----------------------------------------------------------------------------

class GestureDataset(Dataset):
    """ST‑GCN dataset c legacy CSV, векторными аугментациями и cache‑ing."""

    _JSON_CACHE: Dict[Path, Dict[str, Any]] = {}

    # ---------------------
    # ctor / CSV parse
    # ---------------------
    def __init__(
        self,
        json_path: str | Path,
        csv_path: str | Path,
        split: str = "train",
        delimiter: str = "\t",
        max_frames: int = 64,
        center: bool = False,
        normalize: bool = False,
        augment: bool = False,
        rot_deg: float = 15.0,
        scale_jitter: float = 0.1,
        noise_sigma: float = 0.01,
        mirror_prob: float = 0.5,
        add_vel: bool = False,
    ) -> None:
        super().__init__()
        self.json_path = Path(json_path)
        self.csv_path = Path(csv_path)
        self.split = split.lower()
        self.delimiter = delimiter
        self.max_frames = max_frames
        self.center = center
        self.normalize = normalize
        self.augment = augment and (self.split == "train")
        self.rot_deg = rot_deg
        self.scale_jitter = scale_jitter
        self.noise_sigma = noise_sigma
        self.mirror_prob = mirror_prob
        self.add_vel = add_vel
        self.num_nodes = 42

        if not self.json_path.exists():
            raise FileNotFoundError(self.json_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)

        self._load_json()
        self._parse_csv()

    # --------------------- JSON (cached across workers) ----------------------
    def _load_json(self) -> None:
        if self.json_path not in self._JSON_CACHE:
            with self.json_path.open("r", encoding="utf-8") as f:
                self._JSON_CACHE[self.json_path] = json.load(f)
        self._seqs: Dict[str, List[Dict[str, Any]]] = self._JSON_CACHE[self.json_path]

    # --------------------- CSV ----------------------------------------------
    def _parse_csv(self) -> None:
        rows: List[Tuple[str, str, int, int]] = []
        with self.csv_path.open("r", encoding="utf-8-sig") as f:
            rdr = csv.DictReader(f, delimiter=self.delimiter)
            for row in rdr:
                is_train = row.get("train", "").strip().lower() == "true"
                if (self.split == "train" and not is_train) or (self.split == "val" and is_train):
                    continue
                uuid = row.get("attachment_id", "").strip()
                label = row.get("text", "").strip()
                seq = self._seqs.get(uuid)
                if not seq:
                    continue
                begin = int(row.get("begin") or 0)
                end = int(row.get("end") or len(seq))
                end = min(max(begin, end), len(seq))
                rows.append((uuid, label, begin, end))
        if not rows:
            raise RuntimeError("GestureDataset: no samples after parsing CSV. Check column names & split.")

        uniq_labels = sorted({lbl for _, lbl, _, _ in rows})
        self.label2idx = {l: i for i, l in enumerate(uniq_labels)}
        self.samples: List[Tuple[str, int, int, int]] = [
            (u, self.label2idx[lbl], b, e) for u, lbl, b, e in rows
        ]

    # --------------------- util: frames → tensor -----------------------------
    def _frames_to_tensor(self, frames: List[Dict[str, Any]]) -> Tensor:
        T = len(frames)
        pts = torch.zeros((T, self.num_nodes, 3), dtype=torch.float32)
        for t, fr in enumerate(frames):
            merged = self._merge_hands(fr)
            pts[t, :, 0] = torch.tensor([p["x"] for p in merged])
            pts[t, :, 1] = torch.tensor([p["y"] for p in merged])
            pts[t, :, 2] = torch.tensor([p["z"] for p in merged])
        return pts

    # --------------------- temporal resample --------------------------------
    def _time_resample(self, pts: Tensor) -> Tensor:
        """Resize sequence to self.max_frames with 1D linear interpolation."""
        T_old, V, C = pts.shape
        if T_old == self.max_frames:
            return pts
        # (T,V,3) → (C,V,T)
        perm = pts.permute(2, 1, 0).contiguous()
        # flatten spatial dims → (1, C*V, T_old)
        flat = perm.view(1, -1, T_old)
        # 1D linear interpolation
        resized = F.interpolate(flat, size=self.max_frames, mode='linear', align_corners=True)
        # reshape back → (C,V,T_new)
        reshaped = resized.view(C, V, self.max_frames)
        # back to (T_new,V,3)
        pts_new = reshaped.permute(2, 1, 0).contiguous()
        if self.augment:
            shift = random.randint(0, self.max_frames - 1)
            pts_new = torch.roll(pts_new, shifts=shift, dims=0)
        return pts_new

    # --------------------- augmentation --------------------------------
    def _apply_spatial_aug(self, pts: Tensor) -> Tensor:
        """Vectorized mirroring, rotation, scale and noise on (T,V,3)."""
        # mirror
        if random.random() < self.mirror_prob:
            pts = pts[:, _MIRROR_IDX, :]
        # rotation in XY
        ang = math.radians(random.uniform(-self.rot_deg, self.rot_deg))
        cos, sin = math.cos(ang), math.sin(ang)
        xy = pts[..., :2].reshape(-1, 2)
        R = torch.tensor([[cos, -sin], [sin, cos]], dtype=pts.dtype, device=pts.device)
        xy = (R @ xy.T).T.reshape(pts.shape[0], pts.shape[1], 2)
        pts = torch.cat([xy, pts[..., 2:3]], dim=2)
        # scale
        factor = random.uniform(1 - self.scale_jitter, 1 + self.scale_jitter)
        pts = pts * factor
        # noise
        pts = pts + torch.randn_like(pts) * self.noise_sigma
        return pts

    # --------------------- center / normalize -------------------------------
    def _center_norm(self, pts: Tensor) -> Tensor:
        if self.center:
            pts = pts - pts[:, :1, :]
        if self.normalize:
            span = pts.abs().max()
            pts = pts / (span + 1e-6)
        return pts

    # --------------------- public API ---------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        uuid, label_idx, begin, end = self.samples[idx]
        frames = self._seqs[uuid][begin:end]
        pts = self._frames_to_tensor(frames)  # (T,V,3)
        pts = self._time_resample(pts)
        if self.augment:
            pts = self._apply_spatial_aug(pts)
        
        pts = self._center_norm(pts)

        if self.add_vel:
            vel = torch.zeros_like(pts)
            vel[1:] = pts[1:] - pts[:-1]
            pts = torch.cat([pts, vel], dim=2)

        # (T,V,C) → (C,V,T)
        tensor = pts.permute(2, 1, 0).contiguous()
        return tensor, label_idx

    # --------------------- collate (vector aug) -----------------------------
    @staticmethod
    def collate_fn(batch: List[Tuple[Tensor, int]]) -> Tuple[Tensor, Tensor]:  # noqa: D401
        X, Y = zip(*batch)
        return torch.stack(X), torch.tensor(Y, dtype=torch.long)

    # --------------------- merge both hands ---------------------------------
    @staticmethod
    def _merge_hands(fr: Dict[str, Any]) -> List[Dict[str, float]]:
        pts: List[Dict[str, float]] = [dict(x=0.0, y=0.0, z=0.0) for _ in range(42)]
        for idx, pt in enumerate((fr.get("hand 1") or [])[:21]):
            pts[idx] = pt
        for idx, pt in enumerate((fr.get("hand 2") or [])[:21]):
            pts[21 + idx] = pt
        return pts

    # --------------------- pickling (multiprocessing) ------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        # _seqs is large → drop, will reattach in worker
        state["_seqs"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load_json()
