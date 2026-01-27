from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineClassifier(nn.Module):
    """AM-Softmax / CosFace head. If y=None -> plain cosine-softmax (m=0)."""

    def __init__(self, feat_dim: int, num_classes: int, s: float = 30.0, m: float = 0.20):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_classes, feat_dim))
        nn.init.xavier_normal_(self.W)
        self.s = float(s)
        self.m = float(m)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = F.normalize(x, dim=1)  # (B,C)
        W_norm = F.normalize(self.W, dim=1)  # (K,C)
        logits = self.s * (x_norm @ W_norm.t())  # (B,K)
        if y is None:
            return logits
        with torch.no_grad():
            target = torch.zeros_like(logits)
            target.scatter_(1, y.view(-1, 1), 1.0)
        return logits - self.s * self.m * target

