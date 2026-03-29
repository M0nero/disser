from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional


def combine_to_single_json(
    per_video_files: Iterable[Path] | Path | str,
    combined_path: Path | str,
    include_meta: bool = False,
    prefer_pp: bool = True,
    with_meta: Optional[bool] = None,
) -> None:
    raise RuntimeError(
        "Combined JSON export has been removed from kp_export. "
        "Use landmarks.zarr with videos.parquet, frames.parquet, and runs.parquet."
    )
