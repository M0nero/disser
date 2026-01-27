from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from msagcn.data import MultiStreamGestureDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def move_to_device(batch_x: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch_x.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
    return out


def build_mirror_idx(ds: MultiStreamGestureDataset) -> torch.LongTensor:
    V = ds.V
    idx = list(range(21, 42)) + list(range(0, 21))  # swap hands
    if V > 42:
        pose_keep = list(ds.pose_keep)
        pairs = {(9, 10), (11, 12), (13, 14), (15, 16), (23, 24)}
        pos_map = {abs_i: i for i, abs_i in enumerate(pose_keep)}
        for abs_i in pose_keep:
            if abs_i == 0:
                idx.append(42 + pos_map[0])
                continue
            pair_abs = None
            for a, b in pairs:
                if abs_i == a:
                    pair_abs = b
                elif abs_i == b:
                    pair_abs = a
            if pair_abs is None or pair_abs not in pos_map:
                idx.append(42 + pos_map[abs_i])
            else:
                idx.append(42 + pos_map[pair_abs])
    return torch.tensor(idx, dtype=torch.long)


def select_hist_params(
    model: nn.Module,
    preferred: Tuple[str, ...] = ("stems.", "head."),
    max_items: int = 2,
):
    selected = []
    selected_names = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(pat in name for pat in preferred):
            selected.append((name, param))
            selected_names.add(name)
        if len(selected) >= max_items:
            return selected
    if len(selected) < max_items:
        for name, param in model.named_parameters():
            if not param.requires_grad or name in selected_names:
                continue
            selected.append((name, param))
            if len(selected) >= max_items:
                break
    return selected

