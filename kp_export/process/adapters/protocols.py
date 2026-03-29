from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Protocol, Tuple, runtime_checkable


@runtime_checkable
class HandDetectorProtocol(Protocol):
    def process(self, rgb: Any, timestamp_ms: Optional[int] = None) -> Any:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class PoseDetectorProtocol(Protocol):
    def process(self, rgb: Any, timestamp_ms: Optional[int] = None) -> Any:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class DetectorFactoryProtocol(Protocol):
    def create(
        self,
        *,
        backend: str,
        mp: Any,
        mp_solutions: Any,
        hand_model_path: Optional[Path],
        pose_model_path: Optional[Path],
        min_det: float,
        min_track: float,
        pose_complexity: int,
        tasks_delegate: Optional[str],
        second_pass: bool,
        world_coords: bool,
    ) -> Tuple[HandDetectorProtocol, PoseDetectorProtocol, Optional[HandDetectorProtocol]]:
        ...
