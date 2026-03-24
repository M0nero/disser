from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from torch.utils.data import Sampler


def build_sample_weights(
    samples: Sequence[Tuple[str, str, int, Optional[int]]],
    label2idx: Dict[str, int],
    meta_by_vid: Dict[str, Dict[str, Any]],
    quality_floor: float = 0.4,
    quality_power: float = 1.0,
    cover_key: str = "both_coverage",
    cover_floor: float = 0.3,
) -> List[float]:
    freq: Dict[str, int] = {lbl: 0 for lbl in label2idx.keys()}
    for _, lbl, _, _ in samples:
        freq[lbl] = freq.get(lbl, 0) + 1

    weights: List[float] = []
    for vid, lbl, _, _ in samples:
        w_class = 1.0 / max(1, freq[lbl])
        meta = meta_by_vid.get(vid, {})
        q = max(float(meta.get("quality_score", 1.0)), quality_floor)
        cov = max(float(meta.get(cover_key, 0.0)), cover_floor)
        wq = q**quality_power
        wc = 0.5 + 0.5 * cov
        weights.append(w_class * wq * wc)
    return weights


class ClassBalancedBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that yields exactly:
      classes_per_batch unique labels x samples_per_class indices.

    Intended for batch-local SupCon experiments where ordinary random shuffle
    produces too few same-label pairs inside a batch.
    """

    def __init__(
        self,
        samples: Sequence[Tuple[str, str, int, Optional[int]]] | Sequence[int],
        label2idx: Dict[str, int] | None,
        *,
        classes_per_batch: int,
        samples_per_class: int,
        seed: int = 42,
    ) -> None:
        if int(classes_per_batch) <= 0:
            raise ValueError("classes_per_batch must be > 0")
        if int(samples_per_class) <= 0:
            raise ValueError("samples_per_class must be > 0")
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.batch_size = int(self.classes_per_batch * self.samples_per_class)
        self.seed = int(seed)
        self._epoch = 0

        label_to_indices: dict[int, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            if isinstance(sample, (tuple, list)):
                if len(sample) < 2:
                    raise ValueError("sample tuples must contain at least (vid, label, ...)")
                label = sample[1]
                if label2idx is None:
                    raise ValueError("label2idx is required when sampler is built from dataset samples")
                label_id = int(label2idx[str(label)])
            else:
                label_id = int(sample)
            label_to_indices[label_id].append(int(index))

        self._indices_by_label = {
            int(label_id): [int(i) for i in indices]
            for label_id, indices in sorted(label_to_indices.items(), key=lambda item: int(item[0]))
            if indices
        }
        self._label_ids = list(self._indices_by_label.keys())
        if len(self._label_ids) < self.classes_per_batch:
            raise ValueError(
                "ClassBalancedBatchSampler requires at least "
                f"{self.classes_per_batch} unique labels, got {len(self._label_ids)}."
            )
        self._num_batches = len(samples) // self.batch_size

    def __len__(self) -> int:
        return int(self._num_batches)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _reshuffle_pool(self, label_id: int, rng: random.Random) -> dict[str, Any]:
        pool = list(self._indices_by_label[label_id])
        rng.shuffle(pool)
        return {"pool": pool, "pos": 0}

    def _draw_class_indices(
        self,
        *,
        label_id: int,
        state: dict[str, Any],
        rng: random.Random,
    ) -> list[int]:
        all_indices = self._indices_by_label[label_id]
        allow_duplicates = len(all_indices) < self.samples_per_class
        batch: list[int] = []
        seen: set[int] = set()
        while len(batch) < self.samples_per_class:
            if int(state["pos"]) >= len(state["pool"]):
                state.update(self._reshuffle_pool(label_id, rng))
            idx = int(state["pool"][int(state["pos"])])
            state["pos"] = int(state["pos"]) + 1
            if not allow_duplicates and idx in seen:
                if len(seen) >= len(all_indices):
                    allow_duplicates = True
                else:
                    continue
            batch.append(idx)
            seen.add(idx)
        return batch

    def __iter__(self) -> Iterator[List[int]]:
        iter_epoch = int(self._epoch)
        rng = random.Random(self.seed + iter_epoch)
        class_states = {
            int(label_id): self._reshuffle_pool(int(label_id), rng)
            for label_id in self._label_ids
        }
        for _ in range(self._num_batches):
            batch: list[int] = []
            for label_id in rng.sample(self._label_ids, self.classes_per_batch):
                batch.extend(
                    self._draw_class_indices(
                        label_id=int(label_id),
                        state=class_states[int(label_id)],
                        rng=rng,
                    )
                )
            yield batch
        self._epoch = iter_epoch + 1


class HybridSupConBatchSampler(Sampler[List[int]]):
    """
    Hybrid SupCon-oriented batch sampler:
      repeated_classes_per_batch labels x repeated_samples_per_class samples
      + the remaining slots filled with singleton labels.

    This keeps some same-label collisions for batch-local SupCon while preserving
    higher class diversity than a fully class-balanced N x K batch.
    """

    def __init__(
        self,
        samples: Sequence[Tuple[str, str, int, Optional[int]]] | Sequence[int],
        label2idx: Dict[str, int] | None,
        *,
        batch_size: int,
        repeated_classes_per_batch: int,
        repeated_samples_per_class: int = 2,
        seed: int = 42,
    ) -> None:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be > 0")
        if int(repeated_classes_per_batch) <= 0:
            raise ValueError("repeated_classes_per_batch must be > 0")
        if int(repeated_samples_per_class) < 2:
            raise ValueError("repeated_samples_per_class must be >= 2 for SupCon-positive pairs")
        self.batch_size = int(batch_size)
        self.repeated_classes_per_batch = int(repeated_classes_per_batch)
        self.repeated_samples_per_class = int(repeated_samples_per_class)
        self.repeated_total = int(self.repeated_classes_per_batch * self.repeated_samples_per_class)
        if self.repeated_total > self.batch_size:
            raise ValueError(
                "HybridSupConBatchSampler requires repeated_classes_per_batch * repeated_samples_per_class <= batch_size "
                f"(got {self.repeated_classes_per_batch} * {self.repeated_samples_per_class} = {self.repeated_total}, "
                f"batch_size={self.batch_size})."
            )
        self.singleton_slots = int(self.batch_size - self.repeated_total)
        self.seed = int(seed)
        self._epoch = 0

        label_to_indices: dict[int, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            if isinstance(sample, (tuple, list)):
                if len(sample) < 2:
                    raise ValueError("sample tuples must contain at least (vid, label, ...)")
                label = sample[1]
                if label2idx is None:
                    raise ValueError("label2idx is required when sampler is built from dataset samples")
                label_id = int(label2idx[str(label)])
            else:
                label_id = int(sample)
            label_to_indices[label_id].append(int(index))

        self._indices_by_label = {
            int(label_id): [int(i) for i in indices]
            for label_id, indices in sorted(label_to_indices.items(), key=lambda item: int(item[0]))
            if indices
        }
        self._label_ids = list(self._indices_by_label.keys())
        if len(self._label_ids) < self.repeated_classes_per_batch:
            raise ValueError(
                "HybridSupConBatchSampler requires at least "
                f"{self.repeated_classes_per_batch} unique labels, got {len(self._label_ids)}."
            )
        min_required_unique = int(self.repeated_classes_per_batch + self.singleton_slots)
        if len(self._label_ids) < min_required_unique:
            raise ValueError(
                "HybridSupConBatchSampler requires enough unique labels to fill repeated and singleton slots "
                f"(need {min_required_unique}, got {len(self._label_ids)})."
            )
        self._num_batches = len(samples) // self.batch_size

    def __len__(self) -> int:
        return int(self._num_batches)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _reshuffle_pool(self, label_id: int, rng: random.Random) -> dict[str, Any]:
        pool = list(self._indices_by_label[label_id])
        rng.shuffle(pool)
        return {"pool": pool, "pos": 0}

    def _draw_class_indices(
        self,
        *,
        label_id: int,
        draw_count: int,
        state: dict[str, Any],
        rng: random.Random,
    ) -> list[int]:
        all_indices = self._indices_by_label[label_id]
        allow_duplicates = len(all_indices) < int(draw_count)
        batch: list[int] = []
        seen: set[int] = set()
        while len(batch) < int(draw_count):
            if int(state["pos"]) >= len(state["pool"]):
                state.update(self._reshuffle_pool(label_id, rng))
            idx = int(state["pool"][int(state["pos"])])
            state["pos"] = int(state["pos"]) + 1
            if not allow_duplicates and idx in seen:
                if len(seen) >= len(all_indices):
                    allow_duplicates = True
                else:
                    continue
            batch.append(idx)
            seen.add(idx)
        return batch

    def __iter__(self) -> Iterator[List[int]]:
        iter_epoch = int(self._epoch)
        rng = random.Random(self.seed + iter_epoch)
        class_states = {
            int(label_id): self._reshuffle_pool(int(label_id), rng)
            for label_id in self._label_ids
        }
        for _ in range(self._num_batches):
            batch: list[int] = []
            repeated_labels = rng.sample(self._label_ids, self.repeated_classes_per_batch)
            repeated_set = {int(label_id) for label_id in repeated_labels}
            for label_id in repeated_labels:
                batch.extend(
                    self._draw_class_indices(
                        label_id=int(label_id),
                        draw_count=int(self.repeated_samples_per_class),
                        state=class_states[int(label_id)],
                        rng=rng,
                    )
                )
            if self.singleton_slots > 0:
                singleton_labels = rng.sample(
                    [int(label_id) for label_id in self._label_ids if int(label_id) not in repeated_set],
                    self.singleton_slots,
                )
                for label_id in singleton_labels:
                    batch.extend(
                        self._draw_class_indices(
                            label_id=int(label_id),
                            draw_count=1,
                            state=class_states[int(label_id)],
                            rng=rng,
                        )
                    )
            yield batch
        self._epoch = iter_epoch + 1
