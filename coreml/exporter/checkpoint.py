from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .logging_utils import log


def load_state(ckpt_path: Path) -> dict:
    """Load a checkpoint and return a clean state_dict."""
    # Torch 2.6+ defaults weights_only=True which breaks older checkpoints.
    # Try safe load first, then allowlist numpy scalar if needed.
    try:
        obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except Exception:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])  # trusted local checkpoints
        obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    # common nesting patterns
    for key in ("state_dict", "model_state", "model", "net", "ema_state_dict"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            return obj[key]
    # some training loops save torch.nn.Module directly
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if not isinstance(obj, dict):
        raise ValueError("Unsupported checkpoint format: expected dict or object with state_dict()")
    return obj


def infer_num_classes(state: dict) -> int:
    """Infer num_classes from the last linear head weight (2D tensor)."""
    candidates = []
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim == 2 and ("head" in k or "classifier" in k or k.endswith(".fc.weight")):
            # v shape: (out_features, in_features)
            candidates.append((k, v.shape[0]))
    if not candidates:
        # fallback: pick the largest 2D weight (likely final FC)
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                candidates.append((k, v.shape[0]))
    if not candidates:
        raise ValueError("Could not infer num_classes: no 2D linear weights found in state_dict.")
    # prefer keys with 'head' / 'classifier' by ordering
    candidates.sort(key=lambda kv: (0 if ("head" in kv[0] or "classifier" in kv[0]) else 1, -kv[1]))
    chosen_key, num = candidates[0]
    log(f"Inferred num_classes={num} from '{chosen_key}'")
    return int(num)
