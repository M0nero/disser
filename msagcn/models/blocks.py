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


class CTRHandRefine(nn.Module):
    """Group-wise hand-only topology correction on top of the static graph prior."""

    def __init__(
        self,
        ch: int,
        *,
        groups: int = 4,
        hand_nodes: int = 42,
        rel_channels: Optional[int] = None,
        alpha_init: float = 0.0,
    ):
        super().__init__()
        if groups <= 0:
            raise ValueError(f"ctr_groups must be positive, got {groups}")
        if hand_nodes <= 0:
            raise ValueError(f"ctr_hand_nodes must be positive, got {hand_nodes}")
        if ch % groups != 0:
            raise ValueError(
                f"CTR hand refine requires theta(x) channels divisible by ctr_groups; got {ch} and {groups}"
            )

        self.groups = int(groups)
        self.hand_nodes = int(hand_nodes)
        self.ch_per_group = ch // self.groups
        auto_rel = max(4, min(16, self.ch_per_group))
        self.rel_channels = int(rel_channels) if (rel_channels is not None and rel_channels > 0) else auto_rel

        self.q_proj = nn.Conv1d(ch, self.groups * self.rel_channels, 1, groups=self.groups, bias=False)
        self.k_proj = nn.Conv1d(ch, self.groups * self.rel_channels, 1, groups=self.groups, bias=False)
        self.rel_proj = nn.Conv2d(
            self.groups * self.rel_channels,
            self.groups,
            1,
            groups=self.groups,
            bias=False,
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.last_stats: dict[str, float] = {}

    def forward(self, y: torch.Tensor, A_hand_prior: torch.Tensor) -> torch.Tensor:
        B, C, V, T = y.shape
        H = min(V, self.hand_nodes, A_hand_prior.size(-1))
        if H <= 0:
            return y.new_zeros(B, C, 0, T)

        yh = y[:, :, :H, :]
        pooled = yh.mean(dim=-1)  # (B,C,H)

        q = self.q_proj(pooled).reshape(B, self.groups, self.rel_channels, H)
        k = self.k_proj(pooled).reshape(B, self.groups, self.rel_channels, H)

        # Pairwise group-wise relation features over hand nodes only.
        diff = torch.tanh(q.unsqueeze(-1) - k.unsqueeze(-2))  # (B,G,R,H,H)
        q_rel = self.rel_proj(diff.reshape(B, self.groups * self.rel_channels, H, H))
        q_rel = torch.tanh(q_rel)
        q_rel = 0.5 * (q_rel + q_rel.transpose(-1, -2))
        q_rel = torch.nan_to_num(q_rel, nan=0.0, posinf=0.0, neginf=0.0)

        hand_prior = A_hand_prior[:H, :H].to(dtype=y.dtype).unsqueeze(0).unsqueeze(0)
        alpha = self.alpha.to(dtype=y.dtype)
        refined = hand_prior + alpha * q_rel
        delta = refined - hand_prior  # residual correction relative to the static hand prior
        q_rel_abs = q_rel.abs()
        hand_prior_abs = hand_prior.abs()
        self.last_stats = {
            "alpha": float(self.alpha.detach().cpu().item()),
            "delta_abs_mean": float(delta.detach().abs().mean().cpu().item()),
            "delta_abs_max": float(delta.detach().abs().amax().cpu().item()),
            "delta_std": float(delta.detach().float().std(unbiased=False).cpu().item()),
            "delta_to_static_ratio": float(
                (delta.detach().abs().mean() / hand_prior_abs.mean().clamp_min(1e-6)).cpu().item()
            ),
            "q_norm": float(q.detach().float().norm(dim=2).mean().cpu().item()),
            "k_norm": float(k.detach().float().norm(dim=2).mean().cpu().item()),
            "q_saturation": float((q_rel_abs.detach() > 0.95).float().mean().cpu().item()),
            "refined_hand_density": float((refined.detach().abs() > 1e-6).float().mean().cpu().item()),
        }

        yh_group = yh.reshape(B, self.groups, self.ch_per_group, H, T).permute(0, 1, 2, 4, 3).contiguous()
        hand_delta = torch.einsum("bgctu,bgvu->bgctv", yh_group, delta)
        hand_delta = hand_delta.permute(0, 1, 2, 4, 3).reshape(B, C, H, T).contiguous()
        return torch.nan_to_num(hand_delta, nan=0.0, posinf=0.0, neginf=0.0)


def _build_ctr_hand_refine_preserve_rng(
    ch: int,
    *,
    groups: int,
    hand_nodes: int,
    rel_channels: Optional[int],
    alpha_init: float,
) -> CTRHandRefine:
    """
    Instantiate the optional CTR module without perturbing the RNG stream used by
    the legacy backbone initialization. This keeps the base model initialization
    aligned with runs where CTR was disabled, making alpha=0 a cleaner ablation.
    """
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    module = CTRHandRefine(
        ch,
        groups=groups,
        hand_nodes=hand_nodes,
        rel_channels=rel_channels,
        alpha_init=alpha_init,
    )
    torch.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    return module


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
        use_ctr_hand_refine: bool = False,
        ctr_groups: int = 4,
        ctr_hand_nodes: int = 42,
        ctr_rel_channels: Optional[int] = None,
        ctr_alpha_init: float = 0.0,
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
        self.ctr_hand_refine = (
            _build_ctr_hand_refine_preserve_rng(
                inter,
                groups=ctr_groups,
                hand_nodes=ctr_hand_nodes,
                rel_channels=ctr_rel_channels,
                alpha_init=ctr_alpha_init,
            )
            if use_ctr_hand_refine
            else None
        )
        self.last_topology_stats: dict[str, float] = {}

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

    @staticmethod
    def _apply_adjacency(y: torch.Tensor, A_eff: torch.Tensor) -> torch.Tensor:
        y = y.permute(0, 1, 3, 2).contiguous()  # (B,C,T,V)
        y = torch.matmul(y, A_eff.t())  # (B,C,T,V)
        return y.permute(0, 1, 3, 2).contiguous()  # (B,C,V,T)

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

        y_proj = y
        y = self._apply_adjacency(y_proj, A_eff)
        self.last_topology_stats = {}
        if self.ctr_hand_refine is not None:
            H = min(y.size(2), self.ctr_hand_refine.hand_nodes, A_eff.size(-1))
            if H > 0:
                hand_delta = self.ctr_hand_refine(y_proj, A_eff[:H, :H])
                y = y.clone()
                y[:, :, :H, :] = y[:, :, :H, :] + hand_delta
                self.last_topology_stats = dict(self.ctr_hand_refine.last_stats)
                self.last_topology_stats["hand_delta_abs_mean"] = float(hand_delta.detach().abs().mean().cpu().item())
                self.last_topology_stats["hand_nodes"] = float(H)

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
