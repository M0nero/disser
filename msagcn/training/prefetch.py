from __future__ import annotations

from torch.utils.data import DataLoader
import torch

from .utils import move_to_device


class PrefetchLoader:
    """Async transfer of dict-of-tensors batches to GPU."""

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None

    def __iter__(self):
        self.iter = iter(self.loader)
        if self.stream is not None:
            self._preload()
        return self

    def __next__(self):
        if self.stream is None:
            return next(self.iter)
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        X, y, metas = batch
        for t in X.values():
            if isinstance(t, torch.Tensor):
                t.record_stream(torch.cuda.current_stream())
        y.record_stream(torch.cuda.current_stream())
        self._preload()
        return X, y, metas

    def _preload(self):
        try:
            X, y, metas = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return
        if self.stream is None:
            self.next_batch = (X, y, metas)
            return
        with torch.cuda.stream(self.stream):
            X = move_to_device(X, self.device)
            y = y.to(self.device, non_blocking=True)
            self.next_batch = (X, y, metas)

    def __len__(self):
        return len(self.loader)

