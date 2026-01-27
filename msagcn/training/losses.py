from __future__ import annotations

import torch
import torch.nn as nn


class LogitAdjustedCrossEntropyLoss(nn.Module):
    """Cross-entropy with logit adjustment based on class frequencies (Menon et al., 2021)."""

    def __init__(self, class_freq: torch.Tensor, reduction: str = "mean"):
        super().__init__()
        freq = class_freq.clone().detach().float()
        freq = torch.clamp(freq, min=1.0)  # avoid log(0)
        log_freq = torch.log(freq)
        self.register_buffer("log_freq", log_freq)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_freq = self.log_freq.to(logits.device, logits.dtype)
        adjusted_logits = logits - log_freq
        loss = nn.functional.cross_entropy(adjusted_logits, target, reduction=self.reduction)
        return loss

