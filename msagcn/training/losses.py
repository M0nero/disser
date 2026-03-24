from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SupervisedContrastiveLoss(nn.Module):
    """Batch-local supervised contrastive loss with sparse-positive safety."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = float(temperature)
        self.last_stats = {
            "valid_anchor_count": 0,
            "positive_pair_count": 0,
            "batch_size": 0,
            "valid_anchor_ratio": 0.0,
        }

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"features must have shape (B, C), got {tuple(features.shape)}")
        if labels.ndim != 1:
            raise ValueError(f"labels must have shape (B,), got {tuple(labels.shape)}")
        if features.size(0) != labels.size(0):
            raise ValueError("features and labels must have the same batch dimension")

        batch_size = int(features.size(0))
        device = features.device
        if batch_size < 2:
            self.last_stats = {
                "valid_anchor_count": 0,
                "positive_pair_count": 0,
                "batch_size": batch_size,
                "valid_anchor_ratio": 0.0,
            }
            return features.sum() * 0.0

        z = F.normalize(features.float(), dim=1)
        labels = labels.view(-1)
        sim = (z @ z.t()) / self.temperature

        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        logits_mask = ~eye_mask
        pos_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & logits_mask
        positive_counts = pos_mask.sum(dim=1)
        valid_mask = positive_counts > 0

        valid_anchor_count = int(valid_mask.sum().item())
        positive_pair_count = int(pos_mask.sum().item())
        self.last_stats = {
            "valid_anchor_count": valid_anchor_count,
            "positive_pair_count": positive_pair_count,
            "batch_size": batch_size,
            "valid_anchor_ratio": (float(valid_anchor_count) / float(batch_size)) if batch_size > 0 else 0.0,
        }
        if valid_anchor_count == 0:
            return features.sum() * 0.0

        sim = sim.masked_fill(~logits_mask, float("-inf"))
        log_denom = torch.logsumexp(sim, dim=1, keepdim=True)
        log_prob = sim - log_denom
        pos_log_prob = (log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1)) / positive_counts.clamp_min(1).float()
        loss = -pos_log_prob[valid_mask].mean()
        return loss.to(dtype=features.dtype)
