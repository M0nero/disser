from __future__ import annotations

import torch
import torch.nn as nn

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

torch.backends.cudnn.benchmark = True


def _best_gn_groups(ch: int) -> int:
    for g in [32, 16, 8, 4, 2, 1]:
        if ch % g == 0:
            return g
    return 1


class DropPath(nn.Module):
    """Stochastic Depth (per-sample)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div(keep)
        return x * mask

