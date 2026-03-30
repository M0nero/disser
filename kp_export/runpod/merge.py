from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from ..output.schema import FRAME_PARQUET_COLUMNS, RUN_PARQUET_COLUMNS, VIDEO_PARQUET_COLUMNS, normalize_row
from ..output.writer import ExtractorOutputWriter
from .validate import validate_artifact_root


def _field_to_arrow(name: str, kind: str):
    if kind == "string":
        return pa.field(name, pa.string())
    if kind == "int32":
        return pa.field(name, pa.int32())
    if kind == "int64":
        return pa.field(name, pa.int64())
    if kind == "float32":
        return pa.field(name, pa.float32())
    if kind == "bool":
        return pa.field(name, pa.bool_())
    raise ValueError(f"Unsupported field kind: {kind}")


def _read_and_validate_table(path: Path, columns: Sequence[tuple[str, str]]):
    table = pq.read_table(path)
    expected_schema = pa.schema([_field_to_arrow(name, kind) for name, kind in columns])
    if table.schema != expected_schema:
        raise RuntimeError(f"Schema mismatch at {path}")
    return table


def _copy_sample_group(source_group: Any, dest_samples: Any, sample_id: str) -> None:
    if sample_id in dest_samples:
        raise RuntimeError(f"Duplicate sample_id during merge: {sample_id}")
    dest_group = dest_samples.create_group(sample_id)
    dest_group.attrs.update(dict(source_group.attrs))
    for variant in ("raw", "pp"):
        if variant not in source_group:
            continue
        src_variant = source_group[variant]
        dst_variant = dest_group.create_group(variant)
        for array_name in src_variant.array_keys():
            src_array = src_variant[array_name]
            dst_variant.create_dataset(
                array_name,
                data=src_array[:],
                shape=src_array.shape,
                dtype=src_array.dtype,
                chunks=src_array.chunks,
                compressor=src_array.compressor,
                overwrite=True,
            )


def merge_shard_outputs(
    shard_roots: Iterable[str | Path],
    *,
    out_dir: str | Path,
    run_id: str | None = None,
) -> Dict[str, Any]:
    roots = [Path(root) for root in shard_roots]
    if not roots:
        raise RuntimeError("No shard roots provided for merge")

    for root in roots:
        _, errors = validate_artifact_root(root)
        if errors:
            raise RuntimeError(f"Shard validation failed for {root}: {errors[0]}")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    merged_run_id = str(run_id or f"merge_{uuid.uuid4().hex}")
    merged_at = datetime.now(timezone.utc).isoformat()

    all_video_rows: List[Dict[str, Any]] = []
    all_frame_rows: List[Dict[str, Any]] = []

    zarr_root = zarr.open_group(str(out_root / "landmarks.zarr"), mode="w")
    samples_group = zarr_root.require_group("samples")
    seen_ids = set()

    for shard_root in roots:
        video_table = _read_and_validate_table(shard_root / "videos.parquet", VIDEO_PARQUET_COLUMNS)
        frame_table = _read_and_validate_table(shard_root / "frames.parquet", FRAME_PARQUET_COLUMNS)
        shard_video_rows = video_table.to_pylist()
        shard_frame_rows = frame_table.to_pylist()
        shard_ids = [str(row["sample_id"]) for row in shard_video_rows]
        dup = seen_ids.intersection(shard_ids)
        if dup:
            raise RuntimeError(f"Duplicate sample_id across shards: {sorted(dup)[:5]}")
        seen_ids.update(shard_ids)
        all_video_rows.extend(shard_video_rows)
        all_frame_rows.extend(shard_frame_rows)

        source_zarr = zarr.open_group(str(shard_root / "landmarks.zarr"), mode="r")
        for sample_id in source_zarr["samples"].group_keys():
            _copy_sample_group(source_zarr["samples"][sample_id], samples_group, str(sample_id))

    video_schema = pa.schema([_field_to_arrow(name, kind) for name, kind in VIDEO_PARQUET_COLUMNS])
    frame_schema = pa.schema([_field_to_arrow(name, kind) for name, kind in FRAME_PARQUET_COLUMNS])
    pq.write_table(pa.Table.from_pylist(all_video_rows, schema=video_schema), out_root / "videos.parquet")
    pq.write_table(pa.Table.from_pylist(all_frame_rows, schema=frame_schema), out_root / "frames.parquet")

    aggregate = ExtractorOutputWriter._aggregate_video_rows(all_video_rows)
    run_row = normalize_row(
        {
            "run_id": merged_run_id,
            "created_at": merged_at,
            "schema_name": "kp_extract_landmarks",
            "schema_version": 1,
            "out_dir": str(out_root),
            "zarr_path": str(out_root / "landmarks.zarr"),
            "videos_parquet_path": str(out_root / "videos.parquet"),
            "frames_parquet_path": str(out_root / "frames.parquet"),
            "runs_parquet_path": str(out_root / "runs.parquet"),
            "args_json": json.dumps({"mode": "merge", "shard_roots": [str(p) for p in roots]}, ensure_ascii=False, sort_keys=True),
            "versions_json": json.dumps({}, ensure_ascii=False, sort_keys=True),
            "sample_count_existing": 0,
            "sample_count_scheduled": len(all_video_rows),
            "sample_count_processed": len(all_video_rows),
            "sample_count_skipped": 0,
            "sample_count_failed": 0,
            "segments_mode": any(bool(row.get("seg_uid")) for row in all_video_rows),
            "jobs": 0,
            "seed": 0,
            "mp_backend": "merged",
            "quality_score": aggregate.get("quality_score"),
            "swap_rate": aggregate.get("swap_rate"),
            "missing_rate": aggregate.get("missing_rate"),
            "outlier_rate": aggregate.get("outlier_rate"),
            "sanity_reject_rate": aggregate.get("sanity_reject_rate"),
            "pp_filled_frac": aggregate.get("pp_filled_frac"),
            "pp_smoothing_delta": aggregate.get("pp_smoothing_delta"),
        },
        RUN_PARQUET_COLUMNS,
    )
    run_schema = pa.schema([_field_to_arrow(name, kind) for name, kind in RUN_PARQUET_COLUMNS])
    pq.write_table(pa.Table.from_pylist([run_row], schema=run_schema), out_root / "runs.parquet")

    zarr_root.attrs.update(
        {
            "latest_run_id": merged_run_id,
            "latest_run_created_at": merged_at,
            "latest_run_summary": {
                "scheduled_count": len(all_video_rows),
                "skipped_count": 0,
                "failed_count": 0,
                "processed_count": len(all_video_rows),
                "segments_mode": any(bool(row.get("seg_uid")) for row in all_video_rows),
                "jobs": 0,
                "seed": 0,
                "mp_backend": "merged",
                "aggregate": dict(aggregate),
            },
        }
    )

    return {
        "out_dir": str(out_root),
        "run_id": merged_run_id,
        "samples": len(all_video_rows),
        "zarr_path": str(out_root / "landmarks.zarr"),
        "videos_parquet_path": str(out_root / "videos.parquet"),
        "frames_parquet_path": str(out_root / "frames.parquet"),
        "runs_parquet_path": str(out_root / "runs.parquet"),
    }
