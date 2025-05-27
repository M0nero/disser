"""ST‑GCN для распознавания жестов (42 узла: две руки)
Версия 2025‑05‑27 — пирмидальные temporal‑kernels, SE‑Attention и Dropout2d.
Основные изменения против базовой реализации:
* 4 блока: 64‑128‑256‑256 (stride = 2 только в первом)
* kernels: 9‑7‑5‑5 (расписаны в конструкторе)
* Squeeze‑and‑Excite после TCN в каждом блоке
* Multi‑Scale TCN в блоках 2‑4 (dilations 1/2, k = 3/5)
* nn.Dropout2d вместо Dropout (surrogate Stochastic Depth)
* Граф‑свертка исправлена (adjacency действует по V, а не по T)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# -----------------------------------------------------------------------------
# Graph utilities
# -----------------------------------------------------------------------------

def _normalize(A: torch.Tensor) -> torch.Tensor:
    I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    A_hat = A + I
    D_inv_sqrt = (A_hat.sum(1).clamp(min=1e-6)).pow(-0.5)
    D = torch.diag(D_inv_sqrt)
    return D @ A_hat @ D  # DAD


def build_hand_adjacency() -> torch.Tensor:
    """42×42 нормализованный adjacency: 2 ладони по 21 точке + связь запястий."""
    V = 42
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # большой палец
        (0, 5), (5, 6), (6, 7), (7, 8),        # указательный
        (0, 9), (9, 10), (10, 11), (11, 12),   # средний
        (0, 13), (13, 14), (14, 15), (15, 16), # безымянный
        (0, 17), (17, 18), (18, 19), (19, 20), # мизинец
    ]
    A = torch.zeros(V, V)
    for i, j in edges:          # левая ладонь
        A[i, j] = A[j, i] = 1
        A[i + 21, j + 21] = A[j + 21, i + 21] = 1  # правая ладонь
    A[0, 21] = A[21, 0] = 1  # запястье‑к‑запястью
    return _normalize(A).unsqueeze(0)  # (1,42,42)

# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

class SqueezeExcite(nn.Module):
    def __init__(self, ch: int, r: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class MultiScaleTCN(nn.Module):
    """Depth‑wise separable multi‑scale TCN: k = 3/5, dilation = 1/2."""

    def __init__(self, ch: int, drop: float = 0.25, stride: int = 1):
        super().__init__()
        self.stride = stride
        cfg = [(3, 1), (3, 2), (5, 1), (5, 2)]
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, (1, k), dilation=(1, d), padding=(0, ((k - 1) // 2) * d), groups=ch),
                nn.Conv2d(ch, ch, 1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            ) for k, d in cfg
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(ch, ch, 1), nn.BatchNorm2d(ch), nn.Dropout2d(drop)
        )

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
        self.register_buffer('A', A)               # (K,42,42)
        self.edge = nn.Parameter(torch.ones(A.size(0), 1, 1))

        self.theta = nn.Conv2d(in_ch, inter, 1)
        self.phi   = nn.Conv2d(in_ch, inter, 1)
        self.conv1x1 = nn.Conv2d(inter, out_ch, 1)
        self.bn_gcn  = nn.BatchNorm2d(out_ch)

        # Temporal branch
        if use_mstcn and stride == 1:
            self.tcn = MultiScaleTCN(out_ch, drop, stride=1)
        else:
            pad = (temp_k - 1) // 2
            self.tcn = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, (1, temp_k), padding=(0, pad), stride=(1, stride)),
                nn.BatchNorm2d(out_ch),
            )

        self.se   = SqueezeExcite(out_ch) if se else nn.Identity()
        self.drop = nn.Dropout2d(drop)
        self.res  = (
            nn.Identity() if (in_ch == out_ch and stride == 1) else
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=(1, stride)), nn.BatchNorm2d(out_ch))
        )
        self.act = nn.ReLU(inplace=True)

    # ---------------------------------------------------------------------
    def _gcn(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Graph convolution on (B,C,V,T) with (V,V) adjacency."""
        # Convert to (B,C,T,V) to let einsum act over V
        x_t_v = x.permute(0, 1, 3, 2)              # (B,C,T,V)
        x_t_v = torch.einsum('vw,bctw->bctv', A, x_t_v)
        return x_t_v.permute(0, 1, 3, 2)            # (B,C,V,T)

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,V,T)
        A_eff = (self.A * self.edge).sum(0)              # (V,V)
        y = self.phi(x)                                  # (B,inter,V,T)
        y = self._gcn(y, A_eff)
        y = self.bn_gcn(self.conv1x1(self.theta(x) + y))
        y = self.drop(self.se(self.tcn(self.act(y))))
        return self.act(y + self.res(x))


class AttentionPool(nn.Module):
    """Боевой attention‑pool V×T → C."""

    def __init__(self, ch: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(ch, ch // 2, 1), nn.Tanh(), nn.Conv1d(ch // 2, 1, 1)
        )

    def forward(self, x):  # (B,C,V,T)
        B, C, V, T = x.shape
        flat = x.view(B, C, V * T)
        w = self.attn(flat).softmax(-1)
        return (flat * w).sum(-1)  # (B,C)

# -----------------------------------------------------------------------------
# Full network
# -----------------------------------------------------------------------------

class STGCN(nn.Module):
    def __init__(self, num_classes: int, *, in_ch: int = 3, add_vel: bool = True,
                 drop: float = 0.25,
                 depths: list[int] | tuple[int, ...] = (64, 128, 256, 256),
                 temp_ks: list[int] | tuple[int, ...] = (9, 7, 5, 5)):
        super().__init__()
        assert len(depths) == len(temp_ks) == 4, "depths & temp_ks должны иметь длину 4"

        if add_vel:
            in_ch *= 2
        self.add_vel = add_vel
        self.data_bn = nn.BatchNorm1d(in_ch * 42)

        A = build_hand_adjacency()
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

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,3,21|42,T)
        B, C, V, T = x.shape
        if V == 21:                                      # одна рука → допадим вторую
            pad = x.new_zeros(B, C, 42, T)
            pad[:, :, :21, :] = x
            x = pad
        elif V != 42:
            raise ValueError(f"Unexpected V={V}. Ожидалось 21 или 42 узла.")

        if self.add_vel:
            vel = torch.zeros_like(x)
            vel[:, :, :, 1:] = x[:, :, :, 1:] - x[:, :, :, :-1]
            x = torch.cat([x, vel], 1)

        x = self.data_bn(x.flatten(1, 2)).view(B, -1, 42, T)
        for blk in self.layers:
            x = blk(x)
        x = self.pool(x)
        return self.head(x)


__all__ = ["STGCN", "build_hand_adjacency"]
