# ============================ model.py ============================
"""ST-GCN, автоматически работающий с 1-й или 2-мя руками.
Если на входе 21 узел (одна рука), сеть допадит вторую руку нулями
и применит тот же 42×42 граф.
"""
from __future__ import annotations
import torch
import torch.nn as nn

torch.set_float32_matmul_precision('high')

# ------------------------------------------------------------------
# Graph utils
# ------------------------------------------------------------------
def _normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    A_hat = A + I
    D = A_hat.sum(1)
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0
    D_mat = torch.diag(D_inv_sqrt)
    return D_mat @ A_hat @ D_mat


def build_hand_adjacency() -> torch.Tensor:
    V = 42
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    A = torch.zeros((V, V))
    for i, j in edges:
        A[i, j] = A[j, i] = 1                # левая
        A[i+21, j+21] = A[j+21, i+21] = 1    # правая
    A[0, 21] = A[21, 0] = 1                  # запястье-к-запястью
    return _normalize_adjacency(A)

# ------------------------------------------------------------------
# ST-GCN блок
# ------------------------------------------------------------------
class STGCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor,
                 *, temp_k: int = 3, stride: int = 1, drop: float = 0.5):
        super().__init__()
        self.register_buffer("A", A)                 # (42, 42)
        self.gcn = nn.Conv2d(in_ch, out_ch, 1)
        pad = (temp_k - 1) // 2
        self.tcn = nn.Conv2d(
            out_ch, out_ch, (temp_k, 1),
            stride=(stride, 1), padding=(pad, 0)
        )
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.res  = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch)
            ) if in_ch != out_ch or stride != 1 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # (B,C,V,T)
        B, C, V, T = x.shape
        x_sp = x.permute(0, 3, 2, 1).reshape(B*T, V, C)           # (BT,V,C)
        x_sp = torch.matmul(self.A[:V, :V], x_sp)                 # V=21 или 42
        x_sp = x_sp.view(B, T, V, C).permute(0, 3, 2, 1)          # (B,C,V,T)

        x_sp = self.gcn(x_sp)
        x_tp = self.bn(self.tcn(x_sp))
        return self.drop(self.relu(x_tp + self.res(x)))

# ------------------------------------------------------------------
# Полная сеть
# ------------------------------------------------------------------
class STGCN(nn.Module):
    def __init__(self, num_classes: int, *,
                 in_ch: int = 3, temp_k: int = 3,
                 drop: float = 0.0, use_data_bn: bool = False):
        super().__init__()
        self.use_data_bn = use_data_bn
        if use_data_bn:
            self.data_bn = nn.BatchNorm1d(in_ch * 42)

        A = build_hand_adjacency()
        self.layers = nn.ModuleList([
            STGCNBlock(in_ch,  64, A, stride=2, temp_k=temp_k, drop=drop),
            STGCNBlock(64,    128, A, temp_k=temp_k, drop=drop),
            STGCNBlock(128,   256, A, temp_k=temp_k, drop=drop),
        ])
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # (B,C,V,T)
        B, C, V, T = x.shape
        if V == 21:                                              # одна рука
            pad = torch.zeros(B, C, 42, T, device=x.device, dtype=x.dtype)
            pad[:, :, :21, :] = x
            x, V = pad, 42
        elif V != 42:
            raise ValueError(f"Unexpected V={V}. Ожидалось 21 или 42 узла.")

        if self.use_data_bn:                                     # опц. BN
            x = self.data_bn(x.reshape(B, C*V, T)).reshape(B, C, V, T)

        for block in self.layers:
            x = block(x)
        x = x.mean(-1).mean(-1)                                  # pool T,V
        return self.fc(x)
