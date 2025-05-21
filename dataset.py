from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

class GestureDataset(Dataset[Tuple[str, int, int, int]]):
    """Lazy‑loading skeleton dataset with Windows‑friendly pickling.

    *   Возвращает образцы в виде кортежей `(uuid, label_idx, begin, end)`;
    *   Сборка батча происходит в собственном `collate_fn`, который
        формирует тензор `X: (B, 3, 42, T_max)` и метки `Y: (B,)`.
    *   JSON‑словарь со всеми последовательностями кэшируется на уровне класса,
        а при сериализации в worker не копируется (см. `__getstate__/__setstate__`).
    """

    _CACHE: Dict[Path, Dict[str, Any]] = {}

    def __init__(
        self,
        json_path: str,
        csv_path: str,
        split: str = 'train',
        delimiter: str = '\t',
        center: bool = False,
        normalize: bool = False,
        max_frames: int = 256,
    ) -> None:
        self.json_path = Path(json_path)
        self.center = center
        self.normalize = normalize
        self.max_frames = max_frames
        self.num_nodes = 42

        # Verify files
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load / cache JSON index
        if self.json_path not in self._CACHE:
            with self.json_path.open('r', encoding='utf-8') as f:
                self._CACHE[self.json_path] = json.load(f)
        self._sequences: Dict[str, List[Dict[str, Any]]] = self._CACHE[self.json_path]

        # Parse CSV and build sample list
        entries: List[Tuple[str, str, int, int]] = []
        with csv_path.open('r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                is_train = row.get('train', '').strip().lower() == 'true'
                if (split == 'train' and not is_train) or (split == 'val' and is_train):
                    continue

                uuid = row.get('attachment_id', '').strip()
                label = row.get('text', '').strip()
                seq = self._sequences.get(uuid)
                if not seq:
                    continue

                begin = int(row.get('begin') or 0)
                end = int(row.get('end') or len(seq))
                end = min(max(begin, end), len(seq))
                entries.append((uuid, label, begin, end))

        if not entries:
            raise RuntimeError(f"No samples found for split '{split}' in CSV {csv_path}")

        # Label mapping
        unique_labels = sorted({label for _, label, *_ in entries})
        self.label2idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}

        # Final sample list: (uuid, label_idx, begin, end)
        self.samples: List[Tuple[str, int, int, int]] = [
            (uuid, self.label2idx[label], begin, end) for uuid, label, begin, end in entries
        ]

    # --------------------------------------------------
    # Standard Dataset API
    # --------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, int, int, int]:
        return self.samples[idx]

    # --------------------------------------------------
    # Collate & helpers
    # --------------------------------------------------
    def collate_fn(
        self,
        batch: List[Tuple[str, int, int, int]],
    ) -> Tuple[Tensor, Tensor]:
        """Собирает список сэмплов в батч‑тензоры X и Y.

        X: (B, 3, 42, T_max)
        Y: (B,)
        """
        sequences: List[List[Dict[str, Any]]] = []
        labels: List[int] = []
        lengths: List[int] = []

        # Собираем и при необходимости даунсемплируем кадры
        for uuid, label_idx, begin, end in batch:
            frames = self._sequences[uuid][begin:end]
            if len(frames) > self.max_frames:
                step = len(frames) / self.max_frames
                frames = [frames[int(i * step)] for i in range(self.max_frames)]
            sequences.append(frames)
            labels.append(label_idx)
            lengths.append(len(frames))

        B = len(batch)
        T_max = max(lengths)
        C, N = 3, self.num_nodes

        X = torch.zeros((B, C, N, T_max), dtype=torch.float32)
        Y = torch.tensor(labels, dtype=torch.long)

        for i, frames in enumerate(sequences):
            for t, frame in enumerate(frames):
                pts = self._merge_hands(frame)
                X[i, 0, :, t] = torch.tensor([pt['x'] for pt in pts])
                X[i, 1, :, t] = torch.tensor([pt['y'] for pt in pts])
                X[i, 2, :, t] = torch.tensor([pt['z'] for pt in pts])

            if self.center:
                X[i] -= X[i, :, 0:1, :]
            if self.normalize:
                span = X[i].max() - X[i].min()
                X[i] /= (span + 1e-6)

        return X, Y

    @staticmethod
    def _merge_hands(frame: Dict[str, Any]) -> List[Dict[str, float]]:
        """Объединяет до двух кистей в фиксированный список из 42 точек."""
        pts: List[Dict[str, float]] = [dict(x=0.0, y=0.0, z=0.0) for _ in range(42)]
        hand1 = frame.get('hand 1') or []
        hand2 = frame.get('hand 2') or []

        for idx, pt in enumerate(hand1[:21]):
            pts[idx] = pt
        for idx, pt in enumerate(hand2[:21]):
            pts[21 + idx] = pt
        return pts

    # --------------------------------------------------
    # Pickle hooks to avoid RAM explosion with num_workers>0 on Windows
    # --------------------------------------------------
    def __getstate__(self) -> dict:  # noqa: D401 (short description OK)
        state = self.__dict__.copy()
        state['_sequences'] = None  # heavy field — drop before pickling
        return state

    def __setstate__(self, state: dict) -> None:  # noqa: D401
        self.__dict__.update(state)
        if self.json_path not in self._CACHE:
            with self.json_path.open('r', encoding='utf-8') as f:
                self._CACHE[self.json_path] = json.load(f)
        self._sequences = self._CACHE[self.json_path]
