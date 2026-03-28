from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch


BIO_PREPROCESSING_VERSION_V2 = "canonical_hands42_v2"
BIO_PREPROCESSING_VERSION_V3 = "canonical_hands42_v3"

LEFT_HAND_SLICE = slice(0, 21)
RIGHT_HAND_SLICE = slice(21, 42)
LEFT_WRIST_INDEX = 0
RIGHT_WRIST_INDEX = 21


@dataclass(frozen=True)
class BioPreprocessConfig:
    version: str = BIO_PREPROCESSING_VERSION_V3
    center_alpha: float = 0.2
    scale_alpha: float = 0.1
    min_scale: float = 1e-3
    min_visible_joints_for_scale: int = 4


@dataclass
class BioPreprocessState:
    center: torch.Tensor
    scale: torch.Tensor
    has_center: bool = False
    has_scale: bool = False


def init_bio_preprocess_state(
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> BioPreprocessState:
    resolved_device = torch.device(device) if device is not None else torch.device("cpu")
    return BioPreprocessState(
        center=torch.zeros((3,), dtype=dtype, device=resolved_device),
        scale=torch.ones((), dtype=dtype, device=resolved_device),
        has_center=False,
        has_scale=False,
    )


def _ensure_torch_tensor(
    value: np.ndarray | torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _sanitize_frame_inputs(
    raw_pts: torch.Tensor,
    raw_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pts = torch.nan_to_num(raw_pts, nan=0.0, posinf=0.0, neginf=0.0)
    mask = torch.nan_to_num(raw_mask, nan=0.0, posinf=0.0, neginf=0.0)
    if pts.dim() != 2:
        raise ValueError(f"BIO preprocessing expects frame pts shape (V,3), got {tuple(pts.shape)}")
    if mask.dim() == 1:
        mask = mask.unsqueeze(-1)
    elif mask.dim() != 2:
        raise ValueError(f"BIO preprocessing expects frame mask shape (V,1) or (V,), got {tuple(mask.shape)}")
    if pts.shape[0] != mask.shape[0]:
        raise ValueError(f"BIO preprocessing pts/mask length mismatch: {tuple(pts.shape)} vs {tuple(mask.shape)}")
    if pts.shape[1] != 3:
        raise ValueError(f"BIO preprocessing expects xyz coords, got pts shape {tuple(pts.shape)}")
    return pts, mask


def _visible_mask(frame_mask: torch.Tensor) -> torch.Tensor:
    return frame_mask[..., 0] > 0.5


def _compute_center_candidate(frame_pts: torch.Tensor, visible: torch.Tensor) -> torch.Tensor | None:
    wrists = []
    for idx in (LEFT_WRIST_INDEX, RIGHT_WRIST_INDEX):
        if idx < int(visible.numel()) and bool(visible[idx]):
            wrists.append(frame_pts[idx])
    if wrists:
        return torch.stack(wrists, dim=0).mean(dim=0)
    if bool(visible.any()):
        return frame_pts[visible].mean(dim=0)
    return None


def _hand_scale_candidate(
    frame_pts: torch.Tensor,
    visible: torch.Tensor,
    *,
    hand_slice: slice,
    wrist_idx: int,
    min_visible_joints_for_scale: int,
) -> torch.Tensor | None:
    if wrist_idx >= int(visible.numel()) or not bool(visible[wrist_idx]):
        return None
    hand_visible = visible[hand_slice]
    if int(hand_visible.sum().item()) < int(max(2, min_visible_joints_for_scale)):
        return None
    hand_pts = frame_pts[hand_slice]
    wrist = frame_pts[wrist_idx]
    dists = torch.linalg.norm(hand_pts[hand_visible] - wrist.unsqueeze(0), dim=-1)
    dists = dists[dists > 1e-6]
    if int(dists.numel()) <= 0:
        return None
    return dists.mean()


def _compute_scale_candidate(
    frame_pts: torch.Tensor,
    visible: torch.Tensor,
    *,
    min_visible_joints_for_scale: int,
) -> torch.Tensor | None:
    candidates = []
    left = _hand_scale_candidate(
        frame_pts,
        visible,
        hand_slice=LEFT_HAND_SLICE,
        wrist_idx=LEFT_WRIST_INDEX,
        min_visible_joints_for_scale=min_visible_joints_for_scale,
    )
    if left is not None:
        candidates.append(left)
    right = _hand_scale_candidate(
        frame_pts,
        visible,
        hand_slice=RIGHT_HAND_SLICE,
        wrist_idx=RIGHT_WRIST_INDEX,
        min_visible_joints_for_scale=min_visible_joints_for_scale,
    )
    if right is not None:
        candidates.append(right)
    if not candidates:
        return None
    return torch.stack(candidates, dim=0).mean()


def preprocess_frame_v2(
    raw_pts: np.ndarray | torch.Tensor,
    raw_mask: np.ndarray | torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    pts = _ensure_torch_tensor(raw_pts, dtype=dtype, device=device)
    mask = _ensure_torch_tensor(raw_mask, dtype=dtype, device=device)
    pts, mask = _sanitize_frame_inputs(pts, mask)
    return pts * mask


def preprocess_frame_v3(
    raw_pts: np.ndarray | torch.Tensor,
    raw_mask: np.ndarray | torch.Tensor,
    state: BioPreprocessState,
    *,
    cfg: BioPreprocessConfig | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tuple[torch.Tensor, BioPreprocessState, Dict[str, Any]]:
    cfg = cfg or BioPreprocessConfig()
    pts = _ensure_torch_tensor(raw_pts, dtype=dtype, device=device or state.center.device)
    mask = _ensure_torch_tensor(raw_mask, dtype=dtype, device=device or state.center.device)
    pts, mask = _sanitize_frame_inputs(pts, mask)
    visible = _visible_mask(mask)

    center_candidate = _compute_center_candidate(pts, visible)
    if center_candidate is not None:
        if state.has_center:
            alpha_c = float(np.clip(float(cfg.center_alpha), 0.0, 1.0))
            state.center = (1.0 - alpha_c) * state.center + alpha_c * center_candidate
        else:
            state.center = center_candidate
            state.has_center = True

    scale_candidate = _compute_scale_candidate(
        pts,
        visible,
        min_visible_joints_for_scale=int(cfg.min_visible_joints_for_scale),
    )
    if scale_candidate is not None:
        scale_candidate = torch.clamp(scale_candidate, min=float(cfg.min_scale))
        if state.has_scale:
            alpha_s = float(np.clip(float(cfg.scale_alpha), 0.0, 1.0))
            state.scale = (1.0 - alpha_s) * state.scale + alpha_s * scale_candidate
        else:
            state.scale = scale_candidate
            state.has_scale = True

    center = state.center if state.has_center else torch.zeros_like(state.center)
    scale = torch.clamp(state.scale if state.has_scale else torch.ones_like(state.scale), min=float(cfg.min_scale))
    pts_norm = ((pts - center.unsqueeze(0)) / scale) * mask
    debug = {
        "visible_joints": int(visible.sum().item()),
        "center": center.detach().cpu().tolist(),
        "scale": float(scale.detach().cpu().item()),
        "updated_center": bool(center_candidate is not None),
        "updated_scale": bool(scale_candidate is not None),
    }
    return pts_norm, state, debug


def preprocess_sequence(
    raw_pts: np.ndarray | torch.Tensor,
    raw_mask: np.ndarray | torch.Tensor,
    *,
    cfg: BioPreprocessConfig | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[np.ndarray | torch.Tensor, Dict[str, Any]]:
    cfg = cfg or BioPreprocessConfig()
    input_is_torch = isinstance(raw_pts, torch.Tensor)
    pts = _ensure_torch_tensor(raw_pts, dtype=dtype, device=device)
    mask = _ensure_torch_tensor(raw_mask, dtype=dtype, device=device)
    if pts.dim() != 3:
        raise ValueError(f"BIO preprocessing expects sequence pts shape (T,V,3), got {tuple(pts.shape)}")
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    elif mask.dim() != 3:
        raise ValueError(f"BIO preprocessing expects sequence mask shape (T,V,1) or (T,V), got {tuple(mask.shape)}")
    if pts.shape[:2] != mask.shape[:2]:
        raise ValueError(f"BIO preprocessing sequence pts/mask mismatch: {tuple(pts.shape)} vs {tuple(mask.shape)}")

    if str(cfg.version) == BIO_PREPROCESSING_VERSION_V2:
        out = torch.stack([preprocess_frame_v2(pts[idx], mask[idx], dtype=dtype, device=pts.device) for idx in range(int(pts.shape[0]))], dim=0)
        debug = {"version": BIO_PREPROCESSING_VERSION_V2}
        if input_is_torch:
            return out, debug
        return out.detach().cpu().numpy(), debug

    state = init_bio_preprocess_state(device=pts.device, dtype=pts.dtype)
    rows = []
    centers = []
    scales = []
    for idx in range(int(pts.shape[0])):
        frame_out, state, step_debug = preprocess_frame_v3(pts[idx], mask[idx], state, cfg=cfg, dtype=pts.dtype, device=pts.device)
        rows.append(frame_out)
        centers.append(step_debug["center"])
        scales.append(step_debug["scale"])
    out = torch.stack(rows, dim=0) if rows else torch.zeros_like(pts)
    debug = {
        "version": str(cfg.version),
        "config": asdict(cfg),
        "center_trace": centers,
        "scale_trace": scales,
    }
    if input_is_torch:
        return out, debug
    return out.detach().cpu().numpy(), debug

