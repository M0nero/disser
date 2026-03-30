from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pyarrow.compute as pc
import pyarrow.parquet as pq
import zarr

from ..output.schema import OUTPUT_SCHEMA_NAME, OUTPUT_SCHEMA_VERSION


def _required_paths(root: Path) -> Dict[str, Path]:
    return {
        "zarr": root / "landmarks.zarr",
        "videos": root / "videos.parquet",
        "frames": root / "frames.parquet",
        "runs": root / "runs.parquet",
    }


def validate_artifact_root(root: str | Path) -> Tuple[Dict[str, Any], List[str]]:
    artifact_root = Path(root)
    paths = _required_paths(artifact_root)
    errors: List[str] = []
    summary: Dict[str, Any] = {
        "root": str(artifact_root),
        "sample_count_videos": 0,
        "sample_count_zarr": 0,
        "frame_rows": 0,
        "run_rows": 0,
        "frame_count_mismatches": 0,
        "schema_version": OUTPUT_SCHEMA_VERSION,
    }

    for label, path in paths.items():
        if not path.exists():
            errors.append(f"Missing {label}: {path}")
    if errors:
        return summary, errors

    videos = pq.read_table(paths["videos"])
    frames = pq.read_table(paths["frames"])
    runs = pq.read_table(paths["runs"])

    summary["frame_rows"] = len(frames)
    summary["run_rows"] = len(runs)

    required_video_columns = {"sample_id", "num_frames", "schema_name", "schema_version"}
    if not required_video_columns.issubset(set(videos.column_names)):
        errors.append("videos.parquet missing required columns")
        return summary, errors
    required_frame_columns = {"sample_id"}
    if not required_frame_columns.issubset(set(frames.column_names)):
        errors.append("frames.parquet missing required columns")
        return summary, errors

    if len(videos) > 0:
        schema_mask = pc.equal(videos["schema_version"], OUTPUT_SCHEMA_VERSION)
        videos = videos.filter(schema_mask)
    summary["sample_count_videos"] = len(videos)

    sample_ids = [str(x) for x in videos["sample_id"].to_pylist() if x is not None]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("Duplicate sample_id rows in videos.parquet")

    schema_names = {str(x) for x in videos["schema_name"].to_pylist() if x is not None}
    if schema_names and schema_names != {OUTPUT_SCHEMA_NAME}:
        errors.append(f"Unexpected schema_name values in videos.parquet: {sorted(schema_names)}")

    frame_counts: Dict[str, int] = {}
    for sample_id in frames["sample_id"].to_pylist():
        if sample_id is None:
            continue
        key = str(sample_id)
        frame_counts[key] = frame_counts.get(key, 0) + 1

    expected_counts = {
        str(sample_id): int(num_frames or 0)
        for sample_id, num_frames in zip(videos["sample_id"].to_pylist(), videos["num_frames"].to_pylist())
        if sample_id is not None
    }
    mismatches = [
        sample_id
        for sample_id, expected in expected_counts.items()
        if frame_counts.get(sample_id, -1) != expected
    ]
    if mismatches:
        summary["frame_count_mismatches"] = len(mismatches)
        errors.append(f"Frame row count mismatches for {len(mismatches)} sample(s)")

    root_group = zarr.open_group(str(paths["zarr"]), mode="r")
    summary["sample_count_zarr"] = len(list(root_group["samples"].group_keys()))
    zarr_ids = {str(x) for x in root_group["samples"].group_keys()}
    video_ids = set(expected_counts)
    if zarr_ids != video_ids:
        missing_zarr = sorted(video_ids - zarr_ids)
        extra_zarr = sorted(zarr_ids - video_ids)
        if missing_zarr:
            errors.append(f"Missing Zarr groups for {len(missing_zarr)} sample(s)")
        if extra_zarr:
            errors.append(f"Extra Zarr groups for {len(extra_zarr)} sample(s)")

    for sample_id in list(zarr_ids & video_ids)[: min(100, len(zarr_ids & video_ids))]:
        group = root_group["samples"][sample_id]
        if int(group.attrs.get("schema_version", -1)) != OUTPUT_SCHEMA_VERSION:
            errors.append(f"{sample_id}: schema_version mismatch in Zarr attrs")
            break
        if str(group.attrs.get("schema_name", "")) != OUTPUT_SCHEMA_NAME:
            errors.append(f"{sample_id}: schema_name mismatch in Zarr attrs")
            break
        if "raw" not in group:
            errors.append(f"{sample_id}: missing raw variant in Zarr")
            break

    return summary, errors
