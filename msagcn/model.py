# ============================ model.py ============================
# -*- coding: utf-8 -*-
"""
model.py — Multi-Stream Attention-GCN for hand keypoints (21+21) + optional pose

Ключевое:
- Никаких prior'ов.
- Stream-dropout не убивает все потоки и не ломает фьюжн.
- Mask-aware stems/fusion/pooling.
- GCNBlock: правильный порядок BN→ReLU→TCN→SE, без in-place после Sigmoid.
- Опциональная AM-Softmax (CosFace) голова.
- FIX: устойчивые к пустым маскам NodeAttention/AttentionPoolVT (без NaN).

Совместимо с torch.compile, channels_last, AMP.
"""
from __future__ import annotations
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

torch.backends.cudnn.benchmark = True


# ----------------------------------------------------------------------------- #
# Utils
# ----------------------------------------------------------------------------- #
def _normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """Symmetric normalization with self-loop. (V,V) or (K,V,V) -> (K,V,V)"""
    if A.dim() == 2:
        A = A.unsqueeze(0)
    K, V, _ = A.shape
    I = torch.eye(V, device=A.device, dtype=A.dtype).expand(K, V, V)
    A = A + I
    d = A.sum(-1).clamp_min(1e-6).pow(-0.5)
    D = torch.diag_embed(d)
    return D @ A @ D


def _hand_adjacency_42(device=None, dtype=None) -> torch.Tensor:
    """(1,42,42) normalized adjacency for two hands (21+21) + wrist-to-wrist."""
    V = 42
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9,10), (10,11), (11,12),
        (0,13), (13,14), (14,15), (15,16),
        (0,17), (17,18), (18,19), (19,20),
    ]
    A = torch.zeros(V, V, device=device, dtype=dtype)
    for i, j in edges:
        A[i, j] = A[j, i] = 1
        A[i + 21, j + 21] = A[j + 21, i + 21] = 1
    A[0, 21] = A[21, 0] = 1
    return _normalize_adjacency(A)


def _best_gn_groups(ch: int) -> int:
    for g in [32, 16, 8, 4, 2, 1]:
        if ch % g == 0:
            return g
    return 1


class DropPath(nn.Module):
    """Stochastic Depth (per-sample)"""
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


# ----------------------------------------------------------------------------- #
# Building blocks
# ----------------------------------------------------------------------------- #
class MS_TCN(nn.Module):
    """Depthwise-separable multi-scale TCN with dilations (k=3/5)."""
    def __init__(self, ch: int, drop: float = 0.25, stride_t: int = 1):
        super().__init__()
        self.stride_t = stride_t
        cfg = [(3, 1), (3, 2), (5, 1), (5, 2)]
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, (1, k), dilation=(1, d),
                          padding=(0, ((k - 1) // 2) * d), groups=ch),
                nn.Conv2d(ch, ch, 1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ) for k, d in cfg
        ])
        self.fuse = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.BatchNorm2d(ch), nn.Dropout2d(drop))

    def forward(self, x):  # (B,C,V,T)
        out = None
        for b in self.branches:
            y = b(x)
            out = y if out is None else out + y
        out = self.fuse(out)
        if self.stride_t > 1:
            out = F.avg_pool2d(out, kernel_size=(1, self.stride_t), stride=(1, self.stride_t))
        return out


class SE(nn.Module):
    """Squeeze-Excite: y = x * sigmoid(MLP(GAP(x))). Без in-place последствий."""
    def __init__(self, ch: int, r: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w


class GCNBlock(nn.Module):
    """
    Graph + temporal block with learnable edge-importance and DropPath.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        A: torch.Tensor,
        *,
        temp_k: int = 9,
        stride_t: int = 1,
        drop: float = 0.25,
        droppath: float = 0.0,
        use_mstcn: bool = True,
        use_se: bool = True
    ):
        super().__init__()
        inter = out_ch // 2
        if A.dim() == 2:
            A = A.unsqueeze(0)
        self.register_buffer('A', A)                  # (K,V,V)
        self.edge = nn.Parameter(torch.ones(A.size(0), 1, 1))

        # GCN (per time step)
        self.theta = nn.Conv2d(in_ch, inter, 1)
        self.conv  = nn.Conv2d(inter, out_ch, 1)
        self.bn    = nn.BatchNorm2d(out_ch)

        # Temporal
        if use_mstcn and stride_t == 1: 
            self.tcn = MS_TCN(out_ch, drop, stride_t)
        else:
            pad = (temp_k - 1) // 2
            self.tcn = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, (1, temp_k), padding=(0, pad), stride=(1, stride_t)),
                nn.BatchNorm2d(out_ch),
            )

        # SE + regularization
        self.se = SE(out_ch) if use_se else nn.Identity()
        self.drop2d   = nn.Dropout2d(drop)
        self.droppath = DropPath(droppath) if droppath > 0.0 else nn.Identity()

        # Residual
        self.res = (
            nn.Identity()
            if (in_ch == out_ch and stride_t == 1)
            else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=(1, stride_t)), nn.BatchNorm2d(out_ch))
        )

        # ВАЖНО: никакого in-place после Sigmoid
        self.act = nn.ReLU(inplace=False)

    def forward(
        self,
        x: torch.Tensor,                        # (B,C,V,T)
        mask: Optional[torch.Tensor] = None,   # (B,1,V,T)
        A_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1) Effective adjacency
        if A_override is None:
            A_eff = (self.A * self.edge).sum(0)  # (V,V)
        else:
            if A_override.dim() == 3:
                K = A_override.size(0)
                edge = self.edge[:K] if self.edge.size(0) != K else self.edge
                A_eff = (A_override * edge).sum(0)
            elif A_override.dim() == 2:
                A_eff = A_override
            else:
                raise ValueError(f"Unexpected adjacency shape: {tuple(A_override.shape)}")

        # 2) Graph conv per time step
        y = self.theta(x)                                   # (B,C',V,T)
        if mask is not None:
            y = y * (mask > 0).to(y.dtype)                 # zero-out invalid nodes

        y = y.permute(0, 1, 3, 2).contiguous()             # (B,C',T,V)
        y = torch.matmul(y, A_eff.t())                     # (B,C',T,V)
        y = y.permute(0, 1, 3, 2).contiguous()             # (B,C',V,T)

        # 3) BN → ReLU → TCN → SE → Dropout2d → DropPath + Residual
        y = self.bn(self.conv(y))
        y = self.act(y)
        y = self.tcn(y)
        y = self.se(y)
        y = self.drop2d(y)
        y = self.droppath(y) + self.res(x)

        # 4) Final ReLU
        return self.act(y)


class StreamStem(nn.Module):
    """Light stem per stream: 1x1 conv -> GN -> ReLU -> depthwise temporal conv."""
    def __init__(self, in_ch: int, out_ch: int, use_groupnorm: bool = True, temp_k: int = 5):
        super().__init__()
        gn = nn.GroupNorm(_best_gn_groups(out_ch), out_ch) if use_groupnorm else nn.BatchNorm2d(out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1), gn, nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, (1, temp_k), padding=(0, (temp_k - 1) // 2), groups=out_ch),
            nn.Conv2d(out_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):  # (B,C,V,T)
        return self.net(x)


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
            x = feats[s]; xs.append(x)
            m = mask if mask is not None else x.new_ones(B, 1, x.size(2), x.size(3))
            w = (m > 0).to(dtype=x.dtype, device=x.device)
            denom = w.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
            g = (x * w).sum(dim=(2, 3), keepdim=True) / denom
            logits.append(self.score[s](g))     # (B,1,1,1)
        W = torch.softmax(torch.cat(logits, dim=1), dim=1)  # (B,S,1,1)
        out = 0
        for i in range(len(xs)):
            out = out + xs[i] * W[:, i:i+1]
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


class StreamFeatureEncoder(nn.Module):
    """Light per-stream GCN encoder before fusion (keeps modalities separate longer)."""
    def __init__(
        self,
        streams: Sequence[str],
        ch: int,
        A: torch.Tensor,
        drop: float,
        droppath: float,
    ):
        super().__init__()
        self.streams = list(streams)
        self.blocks = nn.ModuleDict({
            s: GCNBlock(
                ch, ch, A,
                temp_k=5, stride_t=1, drop=drop,
                droppath=droppath, use_mstcn=False, use_se=True,
            )
            for s in self.streams
        })

    def forward(
        self,
        feats: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor],
        A_override: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        encoded: Dict[str, torch.Tensor] = {}
        for name, block in self.blocks.items():
            if name not in feats:
                continue
            encoded[name] = block(feats[name], mask, A_override=A_override)
        # keep pass-through streams (e.g., if some were missing during init)
        for name, feat in feats.items():
            if name not in encoded:
                encoded[name] = feat
        return encoded


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
            w = (mask > 0).to(dtype=x.dtype, device=x.device)   # (B,1,V,T)
            denom = w.sum(-1).clamp_min(1.0)                    # (B,1,V)
            x_avg = (x * w).sum(-1) / denom                     # (B,C,V)
            # valid if seen at least once over time
            mask_valid = (mask > 0).any(dim=-1).squeeze(1)      # (B,V)
            # if a sample has no valid nodes at all → don't mask it
            none_valid = ~mask_valid.any(dim=-1)                # (B,)
            if none_valid.any():
                mask_valid = mask_valid.clone()
                mask_valid[none_valid] = True  # disable masking for those rows
        attn = self.score(x_avg).squeeze(1)                     # (B,V)
        if mask_valid is not None:
            attn = attn.masked_fill(~mask_valid, float("-inf"))
        attn = attn.softmax(-1).view(B, 1, V, 1)
        return x * attn


class AttentionPoolVT(nn.Module):
    """Mask-aware attention pooling over V×T -> (B,C), robust to empty masks."""
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
            m = (mask > 0).view(B, 1, V * T)                    # (B,1,V*T)
            any_valid = m.any(dim=-1, keepdim=True)             # (B,1,1)
            masked = scores.masked_fill(~m, float('-inf'))
            # if no valid positions -> fall back to unmasked scores (no NaN)
            masked = torch.where(~any_valid, scores, masked)
            w = masked.softmax(-1)
        return (flat * w).sum(-1)                                 # (B,C)


class CosineClassifier(nn.Module):
    """AM-Softmax / CosFace head. If y=None -> plain cosine-softmax (m=0)."""
    def __init__(self, feat_dim: int, num_classes: int, s: float = 30.0, m: float = 0.20):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_classes, feat_dim))
        nn.init.xavier_normal_(self.W)
        self.s = float(s)
        self.m = float(m)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = F.normalize(x, dim=1)                      # (B,C)
        W_norm = F.normalize(self.W, dim=1)                 # (K,C)
        logits = self.s * (x_norm @ W_norm.t())             # (B,K)
        if y is None:
            return logits
        with torch.no_grad():
            target = torch.zeros_like(logits)
            target.scatter_(1, y.view(-1, 1), 1.0)
        return logits - self.s * self.m * target


# ----------------------------------------------------------------------------- #
# Main model
# ----------------------------------------------------------------------------- #
class MultiStreamAGCN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        V: int = 42,
        A: Optional[torch.Tensor] = None,
        in_ch: int = 3,
        streams: Sequence[str] = ("joints", "bones", "velocity"),
        drop: float = 0.25,
        droppath: float = 0.1,
        depths: Sequence[int] = (64, 128, 256, 256),
        temp_ks: Sequence[int] = (9, 7, 5, 5),
        use_groupnorm_stem: bool = True,
        stream_drop_p: float = 0.15,
        use_cosine_head: bool = False,
        cosine_margin: float = 0.20,
        cosine_scale: float = 30.0,
    ):
        super().__init__()
        assert len(depths) == len(temp_ks) == 4
        self.streams = list(streams)
        self.stream_drop_p = float(stream_drop_p)
        self.V = int(V)

        # adjacency
        A = _hand_adjacency_42() if A is None else _normalize_adjacency(A)
        self.register_buffer('A', A)  # (K,V,V) or (1,V,V)

        # stems per stream
        stem_out = depths[0] // 2
        self.stems = nn.ModuleDict({s: StreamStem(in_ch, stem_out, use_groupnorm_stem, temp_k=5) for s in self.streams})

        # fusion + attention
        self.fuse = StreamAttention(stem_out, self.streams)
        self.node_attn = NodeAttention(stem_out)
        self.stream_encoder = (
            StreamFeatureEncoder(self.streams, stem_out, A, drop=drop * 0.5, droppath=droppath * 0.5)
            if len(self.streams) > 1 else None
        )
        self.pre_fuse_norm = ChannelLayerNorm(stem_out)

        # backbone blocks
        chs = [stem_out, *depths]
        blocks = []
        for i in range(4):
            stride_t = 2 if i == 0 else 1
            blocks.append(
                GCNBlock(
                    chs[i], chs[i+1], A,
                    temp_k=temp_ks[i], stride_t=stride_t, drop=drop,
                    droppath=droppath * (i + 1) / 4.0, use_mstcn=(i > 0), use_se=True,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # pooling
        self.pool = AttentionPoolVT(depths[-1])
        self.embed_norm = nn.LayerNorm(depths[-1])

        # head
        if use_cosine_head:
            self.head = CosineClassifier(depths[-1], num_classes, s=cosine_scale, m=cosine_margin)
            self._use_cos = True
        else:
            self.head = nn.Sequential(
                nn.Linear(depths[-1], depths[-1]),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(depths[-1], num_classes),
            )
            self._use_cos = False

    def _apply_stream_dropout(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Zero-out entire streams during training, keep keys; ensure ≥1 stays."""
        if not self.training or self.stream_drop_p <= 0.0 or len(feats) <= 1:
            return feats
        streams = list(feats.keys())
        dev = next(iter(feats.values())).device
        keep_mask = torch.rand(len(streams), device=dev) > self.stream_drop_p
        if not keep_mask.any():
            idx = torch.randint(0, len(streams), (1,), device=dev)
            keep_mask[idx] = True
        for i, s in enumerate(streams):
            if not bool(keep_mask[i]):
                feats[s] = feats[s] * 0.0
        return feats

    def forward(
        self,
        X: Dict[str, torch.Tensor],                  # streams -> (B,C,V,T)
        mask: Optional[torch.Tensor] = None,         # (B,1,V,T)
        A: Optional[torch.Tensor] = None,            # (V,V) or (K,V,V)
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert isinstance(X, dict) and len(X) > 0, "X must be dict of streams"
        X = {k: v for k, v in X.items() if k in self.streams}
        if len(X) == 0:
            raise ValueError("No known streams provided")

        feats = {s: self.stems[s](x) for s, x in X.items()}      # (B,c0,V,T)
        feats = self._apply_stream_dropout(feats)
        ref_feat = next(iter(feats.values()))
        A_eff = (
            _normalize_adjacency(A).to(ref_feat.device, ref_feat.dtype)
            if A is not None else self.A.to(ref_feat.device, ref_feat.dtype)
        )
        if self.stream_encoder is not None and len(feats) > 1:
            feats = self.stream_encoder(feats, mask, A_eff)

        cur_mask = mask
        if len(feats) == 1:
            y_feat = next(iter(feats.values()))
        else:
            y_feat = self.fuse(feats, cur_mask)
        y_feat = self.node_attn(y_feat, cur_mask)
        y_feat = self.pre_fuse_norm(y_feat)

        for blk in self.blocks:
            y_feat = blk(y_feat, cur_mask, A_override=A_eff)
            if cur_mask is not None and cur_mask.shape[-1] != y_feat.shape[-1]:
                cur_mask = F.interpolate(cur_mask.float(), size=(cur_mask.shape[2], y_feat.shape[-1]), mode="nearest")

        g = self.pool(y_feat, cur_mask)                            # (B,C)
        g = self.embed_norm(g)
        out = self.head(g, y) if self._use_cos else self.head(g)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ----------------------------------------------------------------------------- #
# Legacy helpers (оставил для обратной совместимости)
# ----------------------------------------------------------------------------- #
def build_hand_adjacency() -> torch.Tensor:
    return _hand_adjacency_42()


class SqueezeExcite(nn.Module):
    """Старый SE для legacy блоков (не используется в AGCN)."""
    def __init__(self, ch: int, r: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch // 4, ch, 1), nn.Sigmoid(),
        )
    def forward(self, x):  # (B,C,V,T)
        return x * self.net(x)


class MultiScaleTCN(nn.Module):
    def __init__(self, ch: int, drop: float = 0.25, stride: int = 1):
        super().__init__()
        self.stride = stride
        cfg = [(3, 1), (3, 2), (5, 1), (5, 2)]
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, (1, k), dilation=(1, d),
                          padding=(0, ((k - 1) // 2) * d), groups=ch),
                nn.Conv2d(ch, ch, 1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            ) for k, d in cfg
        ])
        self.fuse = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.BatchNorm2d(ch), nn.Dropout2d(drop))

    def forward(self, x):  # (B,C,V,T)
        out = None
        for b in self.branches:
            y = b(x)
            out = y if out is None else out + y
        out = self.fuse(out)
        if self.stride > 1:
            out = F.avg_pool2d(out, kernel_size=(1, self.stride), stride=(1, self.stride))
        return out


class STGCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor, *,
                 temp_k: int = 9, stride: int = 1, drop: float = 0.25,
                 use_mstcn: bool = True, se: bool = True):
        super().__init__()
        inter = out_ch // 2
        if A.dim() == 2:
            A = A.unsqueeze(0)
        self.register_buffer('A', A)
        self.edge = nn.Parameter(torch.ones(A.size(0), 1, 1))

        self.theta = nn.Conv2d(in_ch, inter, 1)
        self.phi   = nn.Conv2d(in_ch, inter, 1)
        self.conv1x1 = nn.Conv2d(inter, out_ch, 1)
        self.bn_gcn  = nn.BatchNorm2d(out_ch)

        if use_mstcn and stride == 1:
            self.tcn = MultiScaleTCN(out_ch, drop, stride=1)
        else:
            pad = (temp_k - 1) // 2
            self.tcn = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, (1, temp_k), padding=(0, pad), stride=(1, stride)),
                nn.BatchNorm2d(out_ch),
            )
        self.se   = SE(out_ch) if se else nn.Identity()
        self.drop = nn.Dropout2d(drop)
        self.res  = (
            nn.Identity() if (in_ch == out_ch and stride == 1) else
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=(1, stride)), nn.BatchNorm2d(out_ch))
        )
        self.act = nn.ReLU(inplace=False)  # безопасно

    def _gcn(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        x_t_v = x.permute(0, 1, 3, 2)              # (B,C,T,V)
        x_t_v = torch.matmul(x_t_v, A.t())         # (B,C,T,V)
        return x_t_v.permute(0, 1, 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,V,T)
        A_eff = (self.A * self.edge).sum(0)              # (V,V)
        y = self.phi(x)
        y = self._gcn(y, A_eff).permute(0, 1, 3, 2)
        y = self.bn_gcn(self.conv1x1(self.theta(x) + y))

        # BN→ReLU→TCN→SE→Dropout (без in-place)
        y = self.act(y)
        y = self.tcn(y)
        y = self.se(y)
        y = self.drop(y)
        return self.act(y + self.res(x))


class AttentionPool(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.attn = nn.Sequential(nn.Conv1d(ch, ch // 2, 1), nn.Tanh(), nn.Conv1d(ch // 2, 1, 1))

    def forward(self, x):  # (B,C,V,T)
        B, C, V, T = x.shape
        flat = x.view(B, C, V * T)
        w = self.attn(flat).softmax(-1)
        return (flat * w).sum(-1)  # (B,C)


class STGCN(nn.Module):
    def __init__(self, num_classes: int, *, in_ch: int = 3, add_vel: bool = True,
                 drop: float = 0.25, depths: Sequence[int] = (64, 128, 256, 256),
                 temp_ks: Sequence[int] = (9, 7, 5, 5)):
        super().__init__()
        assert len(depths) == len(temp_ks) == 4

        if add_vel:
            in_ch *= 2
        self.add_vel = add_vel
        self.data_bn = nn.BatchNorm1d(in_ch * 42)

        A = _hand_adjacency_42()
        self.layers = nn.ModuleList()
        ch = [in_ch] + list(depths)
        for i in range(4):
            self.layers.append(
                STGCNBlock(
                    ch[i], ch[i + 1], A,
                    temp_k=temp_ks[i], stride=2 if i == 0 else 1,
                    drop=drop, use_mstcn=(i > 0), se=True,
                )
            )

        self.pool = AttentionPool(depths[-1])
        self.head = nn.Sequential(
            nn.Linear(depths[-1], depths[-1]), nn.BatchNorm1d(depths[-1]), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear(depths[-1], num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,C,V,T), V<=42
        B, C, V, T = x.shape
        pad = x.new_zeros(B, C, 42, T)
        pad[:, :, :min(V, 42), :] = x[:, :, :min(V, 42), :]
        x = pad

        if self.add_vel:
            vel = torch.zeros_like(x)
            vel[:, :, :, 1:] = x[:, :, :, 1:] - x[:, :, :, :-1]
            x = torch.cat([x, vel], 1)

        x = self.data_bn(x.flatten(1, 2)).view(B, -1, 42, T)
        for blk in self.layers:
            x = blk(x)
        x = self.pool(x)
        out = self.head(x)
        # Direct output; avoid ops (nan_to_num/isfinite) unsupported by some exporters.
        return out
