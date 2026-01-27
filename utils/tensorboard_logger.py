from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _is_rank0() -> bool:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
    except Exception:
        return True
    return True


def _clean_hparam_value(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)):
        return value
    if value is None:
        return "None"
    return str(value)


class TensorboardLogger:
    def __init__(
        self,
        log_dir: str | Path,
        run_name: str | None = None,
        enabled: bool = True,
        flush_secs: int = 30,
    ) -> None:
        self.enabled = bool(enabled) and _is_rank0()
        self.flush_secs = int(flush_secs) if flush_secs else 30
        self.flush_every_steps = 100
        self._last_flush_step = 0
        self.writer = None

        self.log_dir = Path(log_dir)
        if run_name:
            self.log_dir = self.log_dir / str(run_name)

        if not self.enabled:
            return

        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:
            print(f"WARN: TensorBoard disabled (SummaryWriter import failed: {exc})")
            self.enabled = False
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=self.flush_secs)

    def _maybe_flush(self, step: Optional[int]) -> None:
        if self.writer is None or step is None:
            return
        if self.flush_every_steps <= 0:
            return
        if (step - self._last_flush_step) >= self.flush_every_steps:
            self.writer.flush()
            self._last_flush_step = int(step)

    def scalar(self, tag: str, value: float, step: Optional[int]) -> None:
        if self.writer is None:
            return
        self.writer.add_scalar(tag, value, step)
        self._maybe_flush(step)

    def scalars(self, values: Dict[str, float], step: Optional[int]) -> None:
        if self.writer is None:
            return
        for tag, value in values.items():
            self.writer.add_scalar(tag, value, step)
        self._maybe_flush(step)

    def histogram(self, tag: str, tensor: torch.Tensor, step: Optional[int]) -> None:
        if self.writer is None or tensor is None:
            return
        data = tensor.detach()
        if data.dtype.is_floating_point:
            data = data.float()
        data = data.cpu()
        self.writer.add_histogram(tag, data, step)
        self._maybe_flush(step)

    def image(self, tag: str, img, step: Optional[int], dataformats: str = "HWC") -> None:
        if self.writer is None:
            return
        self.writer.add_image(tag, img, step, dataformats=dataformats)
        self._maybe_flush(step)

    def text(self, tag: str, text: str, step: Optional[int]) -> None:
        if self.writer is None:
            return
        self.writer.add_text(tag, text, step)
        self._maybe_flush(step)

    def hparams(self, hparams_dict: Dict[str, Any], final_metrics_dict: Dict[str, Any]) -> None:
        if self.writer is None:
            return
        clean_hparams = {k: _clean_hparam_value(v) for k, v in hparams_dict.items()}
        clean_metrics = {k: _clean_hparam_value(v) for k, v in final_metrics_dict.items()}
        self.writer.add_hparams(clean_hparams, clean_metrics)
        self.writer.flush()

    def close(self) -> None:
        if self.writer is None:
            return
        self.writer.flush()
        self.writer.close()
