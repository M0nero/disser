from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BioModelConfig:
    """
    Model config for streaming BIO tagging.

    Labels are assumed:
      O = 0
      B = 1
      I = 2
    """
    num_joints: int = 42  # V
    in_coords: int = 3    # xyz
    embed_dim: int = 128

    conv_kernel: int = 5
    conv_layers: int = 2
    conv_dropout: float = 0.10

    gru_hidden: int = 192
    gru_layers: int = 1
    gru_dropout: float = 0.0  # only used if gru_layers > 1

    head_dropout: float = 0.10
    signness_head_dropout: float = 0.10
    onset_head_dropout: float = 0.10

    # Feature toggles
    use_vel: bool = True
    use_acc: bool = True
    use_mask: bool = True
    use_aggs: bool = True

    # Hands layout: first 21 = left, next 21 = right (MediaPipe Hands default)
    assume_two_hands_21: bool = True
    use_signness_head: bool = False
    use_onset_head: bool = False


@dataclass
class BioStreamState:
    pts_buf: torch.Tensor   # (B,W,V,3)
    mask_buf: torch.Tensor  # (B,W,V,1)
    n: int                  # how many frames are valid (<= W)
    h: Optional[torch.Tensor]  # (num_layers,B,H) or None


class CausalConv1d(nn.Module):
    """
    1D causal conv over time axis.
    Input:  (B, C, T)
    Output: (B, C_out, T)
    """
    def __init__(self, c_in: int, c_out: int, kernel_size: int, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.pad_left = (self.kernel_size - 1) * self.dilation
        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        if self.pad_left > 0:
            x = F.pad(x, (self.pad_left, 0))
        return self.conv(x)


class CausalConvBlock(nn.Module):
    """
    Residual causal conv block.
    """
    def __init__(self, channels: int, kernel_size: int, dropout: float, dilation: int = 1):
        super().__init__()
        self.conv = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.act(y)
        y = self.drop(y)
        return x + y


class BioTagger(nn.Module):
    """
    Conv1D front-end (causal) + GRU + BIO head.
    Designed for:
      - Training on fixed windows (B, T, V, 3)
      - Streaming inference with a short temporal buffer via `stream_step`.
    """
    def __init__(self, cfg: BioModelConfig):
        super().__init__()
        self.cfg = cfg
        self.num_classes = 3  # O,B,I

        feat_dim = self._feature_dim(cfg)

        self.embed = nn.Sequential(
            nn.Linear(feat_dim, cfg.embed_dim),
            nn.ReLU(inplace=True),
        )

        # causal conv front-end over time, operating on embedding channels
        blocks = []
        for i in range(cfg.conv_layers):
            # keep dilation=1 here; the GRU provides longer memory.
            blocks.append(CausalConvBlock(cfg.embed_dim, kernel_size=cfg.conv_kernel, dropout=cfg.conv_dropout, dilation=1))
        self.conv_front = nn.Sequential(*blocks) if blocks else nn.Identity()

        self.gru = nn.GRU(
            input_size=cfg.embed_dim,
            hidden_size=cfg.gru_hidden,
            num_layers=cfg.gru_layers,
            dropout=(cfg.gru_dropout if cfg.gru_layers > 1 else 0.0),
            batch_first=True,
            bidirectional=False,
        )

        self.head = nn.Sequential(
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.gru_hidden, self.num_classes),
        )
        self.signness_head = (
            nn.Sequential(
                nn.Dropout(cfg.signness_head_dropout),
                nn.Linear(cfg.gru_hidden, 1),
            )
            if bool(cfg.use_signness_head)
            else None
        )
        self.onset_head = (
            nn.Sequential(
                nn.Dropout(cfg.onset_head_dropout),
                nn.Linear(cfg.gru_hidden, 1),
            )
            if bool(cfg.use_onset_head)
            else None
        )

    @staticmethod
    def _feature_dim(cfg: BioModelConfig) -> int:
        V = cfg.num_joints
        base = V * cfg.in_coords  # coords
        if cfg.use_vel:
            base += V * cfg.in_coords
        if cfg.use_acc:
            base += V * cfg.in_coords
        if cfg.use_mask:
            base += V
        if cfg.use_aggs:
            # a few scalars (depends on two-hands layout)
            base += 10
        return int(base)

    @staticmethod
    def _safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        return torch.sqrt(torch.clamp((x * x).sum(dim=dim), min=eps))

    def _build_features(self, pts: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Build per-frame feature vectors.

        pts:  (B, T, V, 3) float
        mask: (B, T, V, 1) float in {0,1} (or [0..1])
        out:  (B, T, D)
        """
        B, T, V, C = pts.shape
        assert V == self.cfg.num_joints and C == self.cfg.in_coords, (V, C, self.cfg.num_joints, self.cfg.in_coords)

        if mask is None:
            m = torch.ones((B, T, V, 1), device=pts.device, dtype=pts.dtype)
        else:
            m = mask

        # apply mask to coords
        coords = pts * m  # (B,T,V,3)

        feats = [coords.reshape(B, T, V * C)]

        # velocity / acceleration (dt assumed constant; ok because your data is already time-normalized-ish)
        vel = None
        acc = None
        if self.cfg.use_vel or self.cfg.use_acc:
            vel = torch.zeros_like(coords)
            vel[:, 1:] = coords[:, 1:] - coords[:, :-1]
            # vel[t] is valid only if both t and t-1 are valid
            valid_vel = m[:, 1:] * m[:, :-1]  # (B,T-1,V,1)
            valid_vel = F.pad(valid_vel, (0, 0, 0, 0, 1, 0))  # (B,T,V,1)
            vel = vel * valid_vel
            if self.cfg.use_vel:
                feats.append(vel.reshape(B, T, V * C))

        if self.cfg.use_acc:
            if vel is None:
                vel = torch.zeros_like(coords)
                vel[:, 1:] = coords[:, 1:] - coords[:, :-1]
            acc = torch.zeros_like(coords)
            acc[:, 1:] = vel[:, 1:] - vel[:, :-1]
            # acc[t] is valid only if t, t-1, t-2 are valid
            if T < 2:
                valid_acc = torch.zeros_like(m)
            else:
                valid_acc = m[:, 2:] * m[:, 1:-1] * m[:, :-2]  # (B,T-2,V,1)
                valid_acc = F.pad(valid_acc, (0, 0, 0, 0, 2, 0))  # (B,T,V,1)
            acc = acc * valid_acc
            feats.append(acc.reshape(B, T, V * C))

        if self.cfg.use_mask:
            feats.append(m.reshape(B, T, V))

        if self.cfg.use_aggs:
            # aggregated scalar features per frame (cheap & helps boundaries)
            # valid fraction
            valid_frac = m.mean(dim=(2, 3))  # (B,T)

            # mean speed / mean accel magnitude
            if self.cfg.use_vel:
                speed = self._safe_norm(vel, dim=-1)  # (B,T,V)
                speed = speed * m.squeeze(-1)
                denom = (m.squeeze(-1).sum(dim=-1).clamp(min=1.0))  # (B,T)
                mean_speed = speed.sum(dim=-1) / denom
            else:
                mean_speed = torch.zeros((B, T), device=pts.device, dtype=pts.dtype)

            if self.cfg.use_acc:
                accmag = self._safe_norm(acc, dim=-1)
                accmag = accmag * m.squeeze(-1)
                denom = (m.squeeze(-1).sum(dim=-1).clamp(min=1.0))
                mean_acc = accmag.sum(dim=-1) / denom
            else:
                mean_acc = torch.zeros((B, T), device=pts.device, dtype=pts.dtype)

            # two-hands features if layout matches MediaPipe Hands
            if self.cfg.assume_two_hands_21 and V >= 42:
                lw = 0
                rw = 21
                lw_xy = coords[:, :, lw, :]  # (B,T,3)
                rw_xy = coords[:, :, rw, :]
                lw_m = m[:, :, lw, 0]  # (B,T)
                rw_m = m[:, :, rw, 0]

                both = (lw_m * rw_m).clamp(0.0, 1.0)
                wrist_dist = self._safe_norm(lw_xy - rw_xy, dim=-1) * both  # (B,T)

                # spread: mean distance of landmarks to wrist for each hand
                # left hand joints [0..20], right [21..41]
                left = coords[:, :, 0:21, :]
                right = coords[:, :, 21:42, :]
                left_m = m[:, :, 0:21, 0]
                right_m = m[:, :, 21:42, 0]

                lw0 = coords[:, :, 0:1, :]  # (B,T,1,3)
                rw0 = coords[:, :, 21:22, :]

                ldist = self._safe_norm(left - lw0, dim=-1) * left_m
                rdist = self._safe_norm(right - rw0, dim=-1) * right_m

                lden = left_m.sum(dim=-1).clamp(min=1.0)
                rden = right_m.sum(dim=-1).clamp(min=1.0)
                lspread = ldist.sum(dim=-1) / lden
                rspread = rdist.sum(dim=-1) / rden

                # hand speeds (mean over joints)
                if self.cfg.use_vel:
                    lvel = vel[:, :, 0:21, :]
                    rvel = vel[:, :, 21:42, :]
                    ls = self._safe_norm(lvel, dim=-1) * left_m
                    rs = self._safe_norm(rvel, dim=-1) * right_m
                    lms = ls.sum(dim=-1) / lden
                    rms = rs.sum(dim=-1) / rden
                else:
                    lms = torch.zeros((B, T), device=pts.device, dtype=pts.dtype)
                    rms = torch.zeros((B, T), device=pts.device, dtype=pts.dtype)
            else:
                wrist_dist = torch.zeros((B, T), device=pts.device, dtype=pts.dtype)
                lspread = torch.zeros((B, T), device=pts.device, dtype=pts.dtype)
                rspread = torch.zeros((B, T), device=pts.device, dtype=pts.dtype)
                lms = torch.zeros((B, T), device=pts.device, dtype=pts.dtype)
                rms = torch.zeros((B, T), device=pts.device, dtype=pts.dtype)

            aggs = torch.stack(
                [
                    valid_frac,
                    mean_speed,
                    mean_acc,
                    wrist_dist,
                    lspread,
                    rspread,
                    lms,
                    rms,
                    (lms + rms) * 0.5,  # total hand motion proxy
                    (lspread + rspread) * 0.5,  # total spread proxy
                ],
                dim=-1,
            )  # (B,T,10)

            feats.append(aggs)

        return torch.cat(feats, dim=-1)

    def init_stream_state(
        self,
        batch_size: int = 1,
        window: int = 5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> BioStreamState:
        B = max(1, int(batch_size))
        W = max(1, int(window))
        V = int(self.cfg.num_joints)
        C = int(self.cfg.in_coords)

        if device is None or dtype is None:
            p = next(self.parameters(), None)
            if device is None:
                device = p.device if p is not None else torch.device("cpu")
            if dtype is None:
                dtype = p.dtype if p is not None else torch.float32

        pts_buf = torch.zeros((B, W, V, C), device=device, dtype=dtype)
        mask_buf = torch.zeros((B, W, V, 1), device=device, dtype=dtype)
        return BioStreamState(pts_buf=pts_buf, mask_buf=mask_buf, n=0, h=None)

    @torch.no_grad()
    def stream_step(
        self,
        pt: torch.Tensor,
        mask: Optional[torch.Tensor],
        state: BioStreamState,
    ) -> Tuple[torch.Tensor, BioStreamState]:
        """
        Streaming step over a fixed window with stateful GRU.
        pt:   (B,V,3) or (V,3)
        mask: (B,V,1) or (V,1) or None
        Returns:
          logits: (B, 3)
          state: updated BioStreamState
        """
        if pt.dim() == 2:
            pt = pt.unsqueeze(0)  # (1,V,3)
        if pt.dim() != 3:
            raise ValueError(f"pt must be (B,V,3) or (V,3), got {tuple(pt.shape)}")

        if mask is None:
            mask = torch.ones((pt.shape[0], pt.shape[1], 1), device=pt.device, dtype=pt.dtype)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() != 3:
            raise ValueError(f"mask must be (B,V,1) or (V,1), got {tuple(mask.shape)}")

        if pt.shape[0] != state.pts_buf.shape[0]:
            raise ValueError(f"batch mismatch: pt B={pt.shape[0]} vs state B={state.pts_buf.shape[0]}")
        if pt.shape[1] != state.pts_buf.shape[2]:
            raise ValueError(f"V mismatch: pt V={pt.shape[1]} vs state V={state.pts_buf.shape[2]}")

        if pt.device != state.pts_buf.device or mask.device != state.mask_buf.device:
            raise ValueError("pt/mask device must match stream state device")
        if pt.dtype != state.pts_buf.dtype or mask.dtype != state.mask_buf.dtype:
            raise ValueError("pt/mask dtype must match stream state dtype")

        if state.n < state.pts_buf.shape[1]:
            idx = state.n
            state.pts_buf[:, idx] = pt
            state.mask_buf[:, idx] = mask
            state.n += 1
        else:
            state.pts_buf[:, :-1] = state.pts_buf[:, 1:].clone()
            state.mask_buf[:, :-1] = state.mask_buf[:, 1:].clone()
            state.pts_buf[:, -1] = pt
            state.mask_buf[:, -1] = mask

        pts_w = state.pts_buf[:, : state.n]
        mask_w = state.mask_buf[:, : state.n]

        feats = self._build_features(pts_w, mask_w)  # (B,n,D)
        emb = self.embed(feats)  # (B,n,E)
        conv = self.conv_front(emb.transpose(1, 2)).transpose(1, 2)  # (B,n,E)
        conv_last = conv[:, -1, :]  # (B,E)

        y1, hN = self.gru(conv_last.unsqueeze(1), state.h)  # (B,1,H)
        logits = self.head(y1)  # (B,1,3)
        state.h = hN
        return logits[:, 0, :], state

    @torch.no_grad()
    def stream_step_with_aux(
        self,
        pt: torch.Tensor,
        mask: Optional[torch.Tensor],
        state: BioStreamState,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], BioStreamState]:
        if pt.dim() == 2:
            pt = pt.unsqueeze(0)
        if pt.dim() != 3:
            raise ValueError(f"pt must be (B,V,3) or (V,3), got {tuple(pt.shape)}")

        if mask is None:
            mask = torch.ones((pt.shape[0], pt.shape[1], 1), device=pt.device, dtype=pt.dtype)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() != 3:
            raise ValueError(f"mask must be (B,V,1) or (V,1), got {tuple(mask.shape)}")

        if pt.shape[0] != state.pts_buf.shape[0]:
            raise ValueError(f"batch mismatch: pt B={pt.shape[0]} vs state B={state.pts_buf.shape[0]}")
        if pt.shape[1] != state.pts_buf.shape[2]:
            raise ValueError(f"V mismatch: pt V={pt.shape[1]} vs state V={state.pts_buf.shape[2]}")

        if pt.device != state.pts_buf.device or mask.device != state.mask_buf.device:
            raise ValueError("pt/mask device must match stream state device")
        if pt.dtype != state.pts_buf.dtype or mask.dtype != state.mask_buf.dtype:
            raise ValueError("pt/mask dtype must match stream state dtype")

        if state.n < state.pts_buf.shape[1]:
            idx = state.n
            state.pts_buf[:, idx] = pt
            state.mask_buf[:, idx] = mask
            state.n += 1
        else:
            state.pts_buf[:, :-1] = state.pts_buf[:, 1:].clone()
            state.mask_buf[:, :-1] = state.mask_buf[:, 1:].clone()
            state.pts_buf[:, -1] = pt
            state.mask_buf[:, -1] = mask

        pts_w = state.pts_buf[:, : state.n]
        mask_w = state.mask_buf[:, : state.n]
        feats = self._build_features(pts_w, mask_w)
        emb = self.embed(feats)
        conv = self.conv_front(emb.transpose(1, 2)).transpose(1, 2)
        conv_last = conv[:, -1, :]
        y1, hN = self.gru(conv_last.unsqueeze(1), state.h)
        logits = self.head(y1)
        signness_logits = self.signness_head(y1) if self.signness_head is not None else None
        state.h = hN
        return logits[:, 0, :], (signness_logits[:, 0, :] if signness_logits is not None else None), state

    @torch.no_grad()
    def stream_step_with_heads(
        self,
        pt: torch.Tensor,
        mask: Optional[torch.Tensor],
        state: BioStreamState,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], BioStreamState]:
        if pt.dim() == 2:
            pt = pt.unsqueeze(0)
        if pt.dim() != 3:
            raise ValueError(f"pt must be (B,V,3) or (V,3), got {tuple(pt.shape)}")

        if mask is None:
            mask = torch.ones((pt.shape[0], pt.shape[1], 1), device=pt.device, dtype=pt.dtype)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() != 3:
            raise ValueError(f"mask must be (B,V,1) or (V,1), got {tuple(mask.shape)}")

        if pt.shape[0] != state.pts_buf.shape[0]:
            raise ValueError(f"batch mismatch: pt B={pt.shape[0]} vs state B={state.pts_buf.shape[0]}")
        if pt.shape[1] != state.pts_buf.shape[2]:
            raise ValueError(f"V mismatch: pt V={pt.shape[1]} vs state V={state.pts_buf.shape[2]}")

        if pt.device != state.pts_buf.device or mask.device != state.mask_buf.device:
            raise ValueError("pt/mask device must match stream state device")
        if pt.dtype != state.pts_buf.dtype or mask.dtype != state.mask_buf.dtype:
            raise ValueError("pt/mask dtype must match stream state dtype")

        if state.n < state.pts_buf.shape[1]:
            idx = state.n
            state.pts_buf[:, idx] = pt
            state.mask_buf[:, idx] = mask
            state.n += 1
        else:
            state.pts_buf[:, :-1] = state.pts_buf[:, 1:].clone()
            state.mask_buf[:, :-1] = state.mask_buf[:, 1:].clone()
            state.pts_buf[:, -1] = pt
            state.mask_buf[:, -1] = mask

        pts_w = state.pts_buf[:, : state.n]
        mask_w = state.mask_buf[:, : state.n]
        feats = self._build_features(pts_w, mask_w)
        emb = self.embed(feats)
        conv = self.conv_front(emb.transpose(1, 2)).transpose(1, 2)
        conv_last = conv[:, -1, :]
        y1, hN = self.gru(conv_last.unsqueeze(1), state.h)
        logits = self.head(y1)
        signness_logits = self.signness_head(y1) if self.signness_head is not None else None
        onset_logits = self.onset_head(y1) if self.onset_head is not None else None
        state.h = hN
        return (
            logits[:, 0, :],
            (signness_logits[:, 0, :] if signness_logits is not None else None),
            (onset_logits[:, 0, :] if onset_logits is not None else None),
            state,
        )

    def forward(self, pts: torch.Tensor, mask: Optional[torch.Tensor] = None, h0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        pts:  (B, T, V, 3)
        mask: (B, T, V, 1)
        h0:   (num_layers, B, H) initial GRU state
        returns:
          logits: (B, T, 3)
          hN:     (num_layers, B, H)
        """
        logits, _, hN = self.forward_with_aux(pts, mask, h0=h0)
        return logits, hN

    def forward_with_aux(
        self,
        pts: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        x = self._build_features(pts, mask)
        x = self.embed(x)
        y = x.transpose(1, 2)
        y = self.conv_front(y)
        y = y.transpose(1, 2)
        y, hN = self.gru(y, h0)
        logits = self.head(y)
        signness_logits = self.signness_head(y) if self.signness_head is not None else None
        return logits, signness_logits, hN

    def forward_with_heads(
        self,
        pts: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        x = self._build_features(pts, mask)
        x = self.embed(x)
        y = x.transpose(1, 2)
        y = self.conv_front(y)
        y = y.transpose(1, 2)
        y, hN = self.gru(y, h0)
        logits = self.head(y)
        signness_logits = self.signness_head(y) if self.signness_head is not None else None
        onset_logits = self.onset_head(y) if self.onset_head is not None else None
        return logits, signness_logits, onset_logits, hN


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = BioModelConfig(num_joints=42)
    model = BioTagger(cfg)

    B, T, V = 2, 8, cfg.num_joints
    pts = torch.randn(B, T, V, 3)
    mask = (torch.rand(B, T, V, 1) > 0.1).float()

    logits, _ = model(pts, mask)
    assert logits.shape == (B, T, 3), logits.shape

    state = model.init_stream_state(batch_size=B, window=5, device=pts.device, dtype=pts.dtype)
    for t in range(T):
        out, state = model.stream_step(pts[:, t], mask[:, t], state)
        assert out.shape == (B, 3), out.shape

    print("BioTagger self-check passed.")
