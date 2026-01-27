from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DropPath, _best_gn_groups


class MS_TCN(nn.Module):
    """Depthwise-separable multi-scale TCN with dilations (k=3/5)."""

    def __init__(self, ch: int, drop: float = 0.25, stride_t: int = 1):
        super().__init__()
        self.stride_t = stride_t
        cfg = [(3, 1), (3, 2), (5, 1), (5, 2)]
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    ch,
                    ch,
                    (1, k),
                    dilation=(1, d),
                    padding=(0, ((k - 1) // 2) * d),
                    groups=ch,
                ),
                nn.Conv2d(ch, ch, 1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            )
            for k, d in cfg
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
    """Squeeze-Excite: y = x * sigmoid(MLP(GAP(x)))."""

    def __init__(self, ch: int, r: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w


class GCNBlock(nn.Module):
    """Graph + temporal block with learnable edge-importance and DropPath."""

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
        use_se: bool = True,
    ):
        super().__init__()
        inter = out_ch // 2
        if A.dim() == 2:
            A = A.unsqueeze(0)
        self.register_buffer("A", A)  # (K,V,V)
        self.edge = nn.Parameter(torch.ones(A.size(0), 1, 1))

        # GCN (per time step)
        self.theta = nn.Conv2d(in_ch, inter, 1)
        self.conv = nn.Conv2d(inter, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)

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
        self.drop2d = nn.Dropout2d(drop)
        self.droppath = DropPath(droppath) if droppath > 0.0 else nn.Identity()

        # Residual
        self.res = (
            nn.Identity()
            if (in_ch == out_ch and stride_t == 1)
            else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=(1, stride_t)), nn.BatchNorm2d(out_ch))
        )

        # no in-place after Sigmoid
        self.act = nn.ReLU(inplace=False)

    def forward(
        self,
        x: torch.Tensor,  # (B,C,V,T)
        mask: Optional[torch.Tensor] = None,  # (B,1,V,T)
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
        y = self.theta(x)  # (B,C',V,T)
        if mask is not None:
            y = y * (mask > 0).to(y.dtype)  # zero-out invalid nodes

        y = y.permute(0, 1, 3, 2).contiguous()  # (B,C',T,V)
        y = torch.matmul(y, A_eff.t())  # (B,C',T,V)
        y = y.permute(0, 1, 3, 2).contiguous()  # (B,C',V,T)

        # 3) BN->ReLU->TCN->SE->Dropout2d->DropPath + Residual
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
            nn.Conv2d(in_ch, out_ch, 1),
            gn,
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, (1, temp_k), padding=(0, (temp_k - 1) // 2), groups=out_ch),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # (B,C,V,T)
        return self.net(x)
