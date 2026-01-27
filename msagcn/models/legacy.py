from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adjacency import _hand_adjacency_42
from .blocks import SE


def build_hand_adjacency() -> torch.Tensor:
    return _hand_adjacency_42()


class SqueezeExcite(nn.Module):
    """Legacy SE for old blocks (not used in AGCN)."""

    def __init__(self, ch: int, r: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 4, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):  # (B,C,V,T)
        return x * self.net(x)


class MultiScaleTCN(nn.Module):
    def __init__(self, ch: int, drop: float = 0.25, stride: int = 1):
        super().__init__()
        self.stride = stride
        cfg = [(3, 1), (3, 2), (5, 1), (5, 2)]
        self.branches = nn.ModuleList(
            [
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
            ]
        )
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
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        A: torch.Tensor,
        *,
        temp_k: int = 9,
        stride: int = 1,
        drop: float = 0.25,
        use_mstcn: bool = True,
        se: bool = True,
    ):
        super().__init__()
        inter = out_ch // 2
        if A.dim() == 2:
            A = A.unsqueeze(0)
        self.register_buffer("A", A)
        self.edge = nn.Parameter(torch.ones(A.size(0), 1, 1))

        self.theta = nn.Conv2d(in_ch, inter, 1)
        self.phi = nn.Conv2d(in_ch, inter, 1)
        self.conv1x1 = nn.Conv2d(inter, out_ch, 1)
        self.bn_gcn = nn.BatchNorm2d(out_ch)

        if use_mstcn and stride == 1:
            self.tcn = MultiScaleTCN(out_ch, drop, stride=1)
        else:
            pad = (temp_k - 1) // 2
            self.tcn = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, (1, temp_k), padding=(0, pad), stride=(1, stride)),
                nn.BatchNorm2d(out_ch),
            )
        self.se = SE(out_ch) if se else nn.Identity()
        self.drop = nn.Dropout2d(drop)
        self.res = (
            nn.Identity()
            if (in_ch == out_ch and stride == 1)
            else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=(1, stride)), nn.BatchNorm2d(out_ch))
        )
        self.act = nn.ReLU(inplace=False)  # safe

    def _gcn(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        x_t_v = x.permute(0, 1, 3, 2)  # (B,C,T,V)
        x_t_v = torch.matmul(x_t_v, A.t())  # (B,C,T,V)
        return x_t_v.permute(0, 1, 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,V,T)
        A_eff = (self.A * self.edge).sum(0)  # (V,V)
        y = self.phi(x)
        y = self._gcn(y, A_eff).permute(0, 1, 3, 2)
        y = self.bn_gcn(self.conv1x1(self.theta(x) + y))

        # BN->ReLU->TCN->SE->Dropout (no in-place)
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
    def __init__(
        self,
        num_classes: int,
        *,
        in_ch: int = 3,
        add_vel: bool = True,
        drop: float = 0.25,
        depths: Sequence[int] = (64, 128, 256, 256),
        temp_ks: Sequence[int] = (9, 7, 5, 5),
    ):
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
                    ch[i],
                    ch[i + 1],
                    A,
                    temp_k=temp_ks[i],
                    stride=2 if i == 0 else 1,
                    drop=drop,
                    use_mstcn=(i > 0),
                    se=True,
                )
            )

        self.pool = AttentionPool(depths[-1])
        self.head = nn.Sequential(
            nn.Linear(depths[-1], depths[-1]),
            nn.BatchNorm1d(depths[-1]),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(depths[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,C,V,T), V<=42
        B, C, V, T = x.shape
        pad = x.new_zeros(B, C, 42, T)
        pad[:, :, : min(V, 42), :] = x[:, :, : min(V, 42), :]
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
