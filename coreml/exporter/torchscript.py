from __future__ import annotations

import torch

from .logging_utils import log


def try_torchscript(
    model: torch.nn.Module,
    example: torch.Tensor,
    prefer_script: bool = True,
) -> torch.jit.ScriptModule:
    with torch.no_grad():
        if prefer_script:
            try:
                log("Trying torch.jit.script()...")
                scripted = torch.jit.script(model)
                torch.jit.freeze(scripted)
                return scripted
            except Exception as exc:
                log(f"script() failed: {exc}")
        log("Falling back to torch.jit.trace()...")
        traced = torch.jit.trace(model, example, strict=False)
        torch.jit.freeze(traced)
        return traced
