from __future__ import annotations

from typing import Optional, Tuple

import torch

from .logging_utils import log
from .paths import ensure_project_root

ensure_project_root()

# Your project module
from msagcn.model import STGCN  # noqa: E402


def build_model(state: dict, num_classes: int) -> STGCN:
    model = STGCN(num_classes=num_classes)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log(f"Loaded state with {len(missing)} missing keys (ok for buffers / non-critical): e.g. {missing[:5]}")
    if unexpected:
        log(f"State had {len(unexpected)} unexpected keys (ignored): e.g. {unexpected[:5]}")
    model.eval()
    return model


def make_example(model: STGCN, V: int, T: int, channels: Optional[int]) -> Tuple[torch.Tensor, int]:
    # Infer channels C from model config (data_bn) to avoid mismatches.
    if channels is None:
        C = None
        if hasattr(model, "data_bn"):
            try:
                feats = int(model.data_bn.num_features)
                base = feats // V
                C = base // 2 if getattr(model, "add_vel", False) else base
            except Exception:
                C = None
        if C is None:
            C = 3  # sensible default (x,y,z) per joint
    else:
        C = int(channels)
    example = torch.randn(1, C, V, T)
    return example, C
