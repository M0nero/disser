from __future__ import annotations

import copy
import torch
import torch.nn as nn


class ModelEma:
    """Exponential Moving Average wrapper for more stable eval."""

    def __init__(self, model: nn.Module, decay: float = 0.999, device: torch.device | None = None):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        if device is not None:
            self.module.to(device)
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for k, v in ema_state.items():
            src = model_state[k]
            if not v.dtype.is_floating_point:
                v.copy_(src)
                continue
            v.mul_(self.decay).add_(src, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)

