from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn


class StreamAttention(nn.Module):
    """Attention-based fusion across streams, mask-aware global pooling."""

    def __init__(self, ch: int, streams: Sequence[str]):
        super().__init__()
        self.streams = list(streams)
        self.score = nn.ModuleDict({s: nn.Conv2d(ch, 1, 1) for s in self.streams})
        for m in self.score.values():
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, feats: Dict[str, torch.Tensor], mask: Optional[torch.Tensor]) -> torch.Tensor:
        B = next(iter(feats.values())).size(0)
        logits, xs = [], []
        for s in self.streams:
            if s not in feats:
                continue
            x = feats[s]
            xs.append(x)
            m = mask if mask is not None else x.new_ones(B, 1, x.size(2), x.size(3))
            w = (m > 0).to(dtype=x.dtype, device=x.device)
            denom = w.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
            g = (x * w).sum(dim=(2, 3), keepdim=True) / denom
            logits.append(self.score[s](g))  # (B,1,1,1)
        W = torch.softmax(torch.cat(logits, dim=1), dim=1)  # (B,S,1,1)
        out = 0
        for i in range(len(xs)):
            out = out + xs[i] * W[:, i : i + 1]
        return out


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels without destroying (B,C,V,T) layout."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.LayerNorm(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, V, T = x.shape
        y = x.permute(0, 2, 3, 1).contiguous()
        y = self.norm(y)
        return y.permute(0, 3, 1, 2).contiguous()


class NodeAttention(nn.Module):
    """Simple node attention along V (mask-aware, robust to empty masks)."""

    def __init__(self, ch: int):
        super().__init__()
        self.score = nn.Conv1d(ch, 1, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, C, V, T = x.shape
        if mask is None:
            x_avg = x.mean(-1)  # (B,C,V)
            mask_valid = None
        else:
            w = (mask > 0).to(dtype=x.dtype, device=x.device)  # (B,1,V,T)
            denom = w.sum(-1).clamp_min(1.0)  # (B,1,V)
            x_avg = (x * w).sum(-1) / denom  # (B,C,V)
            # valid if seen at least once over time
            mask_valid = (mask > 0).any(dim=-1).squeeze(1)  # (B,V)
            # if a sample has no valid nodes at all -> don't mask it
            none_valid = ~mask_valid.any(dim=-1)  # (B,)
            if none_valid.any():
                mask_valid = mask_valid.clone()
                mask_valid[none_valid] = True  # disable masking for those rows
        attn = self.score(x_avg).squeeze(1)  # (B,V)
        if mask_valid is not None:
            attn = attn.masked_fill(~mask_valid, float("-inf"))
        attn = attn.softmax(-1).view(B, 1, V, 1)
        return x * attn


class AttentionPoolVT(nn.Module):
    """Mask-aware attention pooling over VxT -> (B,C), robust to empty masks."""

    def __init__(self, ch: int):
        super().__init__()
        self.attn = nn.Conv1d(ch, 1, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, C, V, T = x.shape
        flat = x.view(B, C, V * T)
        scores = self.attn(flat)  # (B,1,V*T)
        if mask is None:
            w = scores.softmax(-1)
        else:
            m = (mask > 0).view(B, 1, V * T)  # (B,1,V*T)
            any_valid = m.any(dim=-1, keepdim=True)  # (B,1,1)
            masked = scores.masked_fill(~m, float("-inf"))
            # if no valid positions -> fall back to unmasked scores (no NaN)
            masked = torch.where(~any_valid, scores, masked)
            w = masked.softmax(-1)
        return (flat * w).sum(-1)  # (B,C)
