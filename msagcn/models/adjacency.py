from __future__ import annotations

import torch


def _normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """Symmetric normalization with self-loop. (V,V) or (K,V,V) -> (K,V,V)."""
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
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    A = torch.zeros(V, V, device=device, dtype=dtype)
    for i, j in edges:
        A[i, j] = A[j, i] = 1
        A[i + 21, j + 21] = A[j + 21, i + 21] = 1
    A[0, 21] = A[21, 0] = 1
    return _normalize_adjacency(A)

