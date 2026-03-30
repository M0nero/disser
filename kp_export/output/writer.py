from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

import numpy as np

from .schema import (
    FRAME_PARQUET_COLUMNS,
    OUTPUT_SCHEMA_NAME,
    OUTPUT_SCHEMA_VERSION,
    RUN_PARQUET_COLUMNS,
    VIDEO_PARQUET_COLUMNS,
    normalize_row,
    normalize_rows,
)
from .staging import load_staged_payload, remove_staged_payload


class ExtractorOutputWriter:
    def __init__(
        self,
        *,
        out_dir: str | Path,
        scratch_dir: str | Path | None = None,
        run_id: str,
        args_snapshot: Dict[str, Any],
        versions: Dict[str, Any],
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        scratch_root = Path(scratch_dir) if scratch_dir else (self.out_dir / ".staging")
        self.scratch_root = scratch_root
        self.stage_dir = scratch_root
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.current_run_dir = self.stage_dir / run_id
        if self.current_run_dir.exists():
            shutil.rmtree(self.current_run_dir)
        self.current_run_dir.mkdir(parents=True, exist_ok=True)

        self.zarr_path = self.out_dir / "landmarks.zarr"
        self.videos_parquet_path = self.out_dir / "videos.parquet"
        self.frames_parquet_path = self.out_dir / "frames.parquet"
        self.runs_parquet_path = self.out_dir / "runs.parquet"
        self.run_id = str(run_id)
        self.args_snapshot = dict(args_snapshot)
        self.versions = dict(versions)
        self.created_at = datetime.now(timezone.utc).isoformat()
        self._processed_ids: Set[str] = set()
        self._existing_ids = self._load_committed_ids()
        self._video_stage_path = self.current_run_dir / "videos.new.parquet"
        self._frame_stage_path = self.current_run_dir / "frames.new.parquet"
        self._video_writer = None
        self._frame_writer = None
        self._video_schema = None
        self._frame_schema = None

        import zarr  # local import so CLI help does not depend on zarr
        from numcodecs import Blosc

        self._zarr = zarr
        self._compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
        self._root = zarr.open_group(str(self.zarr_path), mode="a")
        self._samples = self._root.require_group("samples")
        self._root.attrs.update(
            {
                "schema_name": OUTPUT_SCHEMA_NAME,
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "created_at": self.created_at,
                "run_id": self.run_id,
                "args_snapshot": self.args_snapshot,
                "versions": self.versions,
            }
        )

    def _load_committed_ids(self) -> Set[str]:
        video_rows: Dict[str, int] = {}
        frame_counts: Dict[str, int] = {}
        group_ids: Set[str] = set()
        if self.videos_parquet_path.exists():
            try:
                import pyarrow.compute as pc
                import pyarrow.parquet as pq

                table = pq.read_table(self.videos_parquet_path)
                if {
                    "sample_id",
                    "schema_version",
                    "num_frames",
                }.issubset(set(table.column_names)):
                    mask = pc.equal(table["schema_version"], OUTPUT_SCHEMA_VERSION)
                    filtered = table.filter(mask)
                    sample_ids = filtered["sample_id"].to_pylist()
                    num_frames = filtered["num_frames"].to_pylist()
                    for sample_id, num_frames_value in zip(sample_ids, num_frames):
                        if sample_id is None or not str(sample_id).strip():
                            continue
                        video_rows[str(sample_id)] = int(num_frames_value or 0)
            except Exception:
                video_rows = {}

        if self.frames_parquet_path.exists():
            try:
                import pyarrow.parquet as pq

                table = pq.read_table(self.frames_parquet_path, columns=["sample_id"])
                for sample_id in table["sample_id"].to_pylist():
                    if sample_id is None or not str(sample_id).strip():
                        continue
                    key = str(sample_id)
                    frame_counts[key] = frame_counts.get(key, 0) + 1
            except Exception:
                frame_counts = {}

        if self.zarr_path.exists():
            try:
                import zarr

                root = zarr.open_group(str(self.zarr_path), mode="a")
                samples = root.require_group("samples")
                group_ids = {str(name) for name in samples.group_keys()}
            except Exception:
                group_ids = set()

        committed: Set[str] = set()
        for sample_id, expected_frames in video_rows.items():
            if sample_id in group_ids and frame_counts.get(sample_id, -1) == expected_frames:
                committed.add(sample_id)
        return committed

    @property
    def existing_sample_ids(self) -> Set[str]:
        return set(self._existing_ids)

    def is_sample_committed(self, sample_id: str) -> bool:
        sample = str(sample_id)
        return sample in self._existing_ids or sample in self._processed_ids

    def _ensure_parquet_writer(self, which: str):
        import pyarrow as pa
        import pyarrow.parquet as pq

        if which == "videos":
            if self._video_writer is not None:
                return self._video_writer
            schema = pa.schema([self._field_to_arrow(name, kind) for name, kind in VIDEO_PARQUET_COLUMNS])
            self._video_schema = schema
            self._video_writer = pq.ParquetWriter(str(self._video_stage_path), schema)
            return self._video_writer

        if which == "frames":
            if self._frame_writer is not None:
                return self._frame_writer
            schema = pa.schema([self._field_to_arrow(name, kind) for name, kind in FRAME_PARQUET_COLUMNS])
            self._frame_schema = schema
            self._frame_writer = pq.ParquetWriter(str(self._frame_stage_path), schema)
            return self._frame_writer

        raise ValueError(which)

    @staticmethod
    def _field_to_arrow(name: str, kind: str):
        import pyarrow as pa

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

    def _time_chunks(self, arr: np.ndarray, *, kind: str) -> tuple[int, ...]:
        if arr.ndim == 0:
            return ()
        limit = 1024 if kind in {"time", "score", "valid"} else 256
        t = max(1, min(limit, int(arr.shape[0])))
        if arr.ndim == 1:
            return (t,)
        return (t, *tuple(int(x) for x in arr.shape[1:]))

    def _write_variant_arrays(self, group: Any, arrays: Dict[str, np.ndarray], *, variant: str) -> None:
        variant_group = group.require_group(variant)
        for name, array in arrays.items():
            arr = np.asarray(array)
            if name in {"ts_ms", "left_score", "right_score"}:
                chunks = self._time_chunks(arr, kind="score" if name != "ts_ms" else "time")
            elif name in {"left_valid", "right_valid"}:
                chunks = self._time_chunks(arr, kind="valid")
            else:
                chunks = self._time_chunks(arr, kind="landmark")
            if name in variant_group:
                del variant_group[name]
            variant_group.create_dataset(
                name,
                data=arr,
                shape=arr.shape,
                dtype=arr.dtype,
                chunks=chunks,
                compressor=self._compressor,
                overwrite=True,
            )

    def _write_sample_group(self, payload: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        sample_id = str(payload["sample_id"])
        if sample_id in self._samples:
            del self._samples[sample_id]
        group = self._samples.create_group(sample_id)
        sample_attrs = dict(payload.get("sample_attrs", {}))
        sample_attrs["schema_name"] = OUTPUT_SCHEMA_NAME
        sample_attrs["schema_version"] = OUTPUT_SCHEMA_VERSION
        group.attrs.update(sample_attrs)
        raw_arrays = dict(payload.get("raw_arrays", {}))
        if not raw_arrays:
            raise RuntimeError(f"Sample {sample_id} has no raw arrays")
        self._write_variant_arrays(group, raw_arrays, variant="raw")
        pp_arrays = payload.get("pp_arrays") or None
        has_pp = bool(pp_arrays)
        if has_pp:
            self._write_variant_arrays(group, dict(pp_arrays), variant="pp")
        return f"samples/{sample_id}", {"has_pp": has_pp}

    def _video_row_from_payload(self, payload: Dict[str, Any], *, zarr_group: str, has_pp: bool) -> Dict[str, Any]:
        row = dict(payload.get("video_row", {}))
        row["run_id"] = self.run_id
        row["sample_id"] = str(payload["sample_id"])
        row.setdefault("slug", str(payload["sample_id"]))
        row.setdefault("source_video", str(payload.get("source_video", "")))
        row["zarr_group"] = zarr_group
        row["raw_group"] = f"{zarr_group}/raw"
        row["pp_group"] = f"{zarr_group}/pp" if has_pp else ""
        row["has_pp"] = bool(has_pp)
        row["schema_name"] = OUTPUT_SCHEMA_NAME
        row["schema_version"] = OUTPUT_SCHEMA_VERSION
        row.setdefault("coords_mode", str(payload.get("sample_attrs", {}).get("coords_mode", "")))
        row.setdefault("postprocess_enabled", bool(payload.get("sample_attrs", {}).get("postprocess_enabled", False)))
        return normalize_row(row, VIDEO_PARQUET_COLUMNS)

    def _frame_rows_from_payload(self, payload: Dict[str, Any], *, zarr_group: str, has_pp: bool) -> List[Dict[str, Any]]:
        rows = []
        raw_group = f"{zarr_group}/raw"
        pp_group = f"{zarr_group}/pp" if has_pp else ""
        for row in payload.get("frame_rows", []):
            out = dict(row)
            out["run_id"] = self.run_id
            out["sample_id"] = str(payload["sample_id"])
            out["zarr_group"] = zarr_group
            out["variant_raw_group"] = raw_group
            out["variant_pp_group"] = pp_group
            rows.append(out)
        return normalize_rows(rows, FRAME_PARQUET_COLUMNS)

    @staticmethod
    def _normalize_payload(payload: Any) -> Dict[str, Any]:
        if isinstance(payload, dict):
            return dict(payload)
        to_dict = getattr(payload, "to_dict", None)
        if callable(to_dict):
            normalized = to_dict()
            if isinstance(normalized, dict):
                return normalized
        raise TypeError(f"Unsupported payload type for commit_payload: {type(payload).__name__}")

    def commit_staged_sample(self, staged_path: str | Path) -> Dict[str, Any]:
        payload = load_staged_payload(staged_path)
        return self.commit_payload(payload, remove_stage_path=staged_path)

    def commit_payload(
        self,
        payload: Any,
        *,
        remove_stage_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        payload = self._normalize_payload(payload)
        sample_id = str(payload["sample_id"])
        zarr_group, info = self._write_sample_group(payload)
        has_pp = bool(info.get("has_pp"))

        import pyarrow as pa

        video_row = self._video_row_from_payload(payload, zarr_group=zarr_group, has_pp=has_pp)
        frame_rows = self._frame_rows_from_payload(payload, zarr_group=zarr_group, has_pp=has_pp)

        video_writer = self._ensure_parquet_writer("videos")
        video_table = pa.Table.from_pylist([video_row], schema=self._video_schema)
        video_writer.write_table(video_table)

        frame_writer = self._ensure_parquet_writer("frames")
        frame_table = pa.Table.from_pylist(frame_rows, schema=self._frame_schema)
        frame_writer.write_table(frame_table)

        self._processed_ids.add(sample_id)
        if remove_stage_path is not None:
            remove_staged_payload(remove_stage_path)
        return video_row

    def _read_table_or_empty(self, path: Path, columns: Sequence[tuple[str, str]]):
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pa.schema([self._field_to_arrow(name, kind) for name, kind in columns])
        if not path.exists():
            return pa.Table.from_pylist([], schema=schema)
        return pq.read_table(path)

    def _empty_table(self, columns: Sequence[tuple[str, str]]):
        import pyarrow as pa

        schema = pa.schema([self._field_to_arrow(name, kind) for name, kind in columns])
        return pa.Table.from_pylist([], schema=schema)

    def _merge_table(self, existing_path: Path, new_path: Path, final_path: Path, columns: Sequence[tuple[str, str]]) -> None:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        schema = pa.schema([self._field_to_arrow(name, kind) for name, kind in columns])
        existing = self._read_table_or_empty(existing_path, columns)
        new = self._read_table_or_empty(new_path, columns)
        if existing.schema != schema and len(existing) > 0:
            raise RuntimeError(f"Existing parquet schema mismatch at {existing_path}")
        if new.schema != schema and len(new) > 0:
            raise RuntimeError(f"Staged parquet schema mismatch at {new_path}")
        if existing.schema != schema:
            existing = self._empty_table(columns)
        if new.schema != schema:
            new = self._empty_table(columns)
        if len(new) == 0:
            if len(existing) == 0:
                empty = self._empty_table(columns)
                pq.write_table(empty, final_path)
                return empty
            pq.write_table(existing, final_path)
            return existing
        if "sample_id" in existing.column_names and len(existing) > 0:
            sample_ids = [str(x) for x in new["sample_id"].to_pylist() if x is not None]
            if sample_ids:
                mask = pc.is_in(existing["sample_id"], value_set=pa.array(sample_ids, type=pa.string()))
                existing = existing.filter(pc.invert(mask))
        merged = pa.concat_tables([existing, new], promote_options="default")
        pq.write_table(merged, final_path)
        return merged

    @staticmethod
    def _aggregate_video_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        frames_total = 0.0
        quality_sum = 0.0
        quality_frames = 0.0
        swap_total = 0.0
        missing_total = 0.0
        outlier_total = 0.0
        sanity_total = 0.0
        pp_filled_total = 0.0
        pp_filled_seen = False
        pp_delta_weighted_sum = 0.0
        pp_delta_weighted_frames = 0.0

        for row in rows:
            num_frames = float(int(row.get("num_frames", 0) or 0))
            if num_frames <= 0:
                continue
            frames_total += num_frames
            quality = row.get("quality_score")
            if quality is not None:
                quality_sum += float(quality) * num_frames
                quality_frames += num_frames
            swap_total += float(row.get("swap_frames", 0) or 0)
            missing_total += float(row.get("missing_frames_1", 0) or 0) + float(row.get("missing_frames_2", 0) or 0)
            outlier_total += float(row.get("outlier_frames_1", 0) or 0) + float(row.get("outlier_frames_2", 0) or 0)
            sanity_total += float(row.get("sanity_reject_frames_1", 0) or 0) + float(row.get("sanity_reject_frames_2", 0) or 0)
            if row.get("pp_filled_left") is not None or row.get("pp_filled_right") is not None:
                pp_filled_seen = True
                pp_filled_total += float(row.get("pp_filled_left", 0) or 0) + float(row.get("pp_filled_right", 0) or 0)
            pp_delta_vals = []
            if row.get("pp_smoothing_delta_left") is not None:
                pp_delta_vals.append(float(row.get("pp_smoothing_delta_left")))
            if row.get("pp_smoothing_delta_right") is not None:
                pp_delta_vals.append(float(row.get("pp_smoothing_delta_right")))
            if pp_delta_vals:
                pp_delta_weighted_sum += (sum(pp_delta_vals) / len(pp_delta_vals)) * num_frames
                pp_delta_weighted_frames += num_frames

        return {
            "quality_score": (quality_sum / quality_frames) if quality_frames > 0.0 else None,
            "swap_rate": (swap_total / frames_total) if frames_total > 0.0 else None,
            "missing_rate": (missing_total / (2.0 * frames_total)) if frames_total > 0.0 else None,
            "outlier_rate": (outlier_total / (2.0 * frames_total)) if frames_total > 0.0 else None,
            "sanity_reject_rate": (sanity_total / (2.0 * frames_total)) if frames_total > 0.0 else None,
            "pp_filled_frac": (pp_filled_total / (2.0 * frames_total)) if pp_filled_seen and frames_total > 0.0 else None,
            "pp_smoothing_delta": (pp_delta_weighted_sum / pp_delta_weighted_frames) if pp_delta_weighted_frames > 0.0 else None,
        }

    @classmethod
    def _snapshot_from_video_table(cls, table: Any) -> tuple[int, Dict[str, Any]]:
        try:
            rows = table.to_pylist()
        except Exception:
            rows = []
        return len(rows), cls._aggregate_video_rows(rows)

    def finalize(
        self,
        *,
        scheduled_count: int,
        skipped_count: int,
        failed_count: int,
        processed_count: int,
        segments_mode: bool,
        jobs: int,
        seed: int,
        mp_backend: str,
        aggregate: Dict[str, Any],
    ) -> Dict[str, str]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None
        if self._frame_writer is not None:
            self._frame_writer.close()
            self._frame_writer = None

        merged_videos = self._merge_table(
            self.videos_parquet_path,
            self._video_stage_path,
            self.videos_parquet_path,
            VIDEO_PARQUET_COLUMNS,
        )
        self._merge_table(
            self.frames_parquet_path,
            self._frame_stage_path,
            self.frames_parquet_path,
            FRAME_PARQUET_COLUMNS,
        )

        snapshot_processed_count, snapshot_aggregate = self._snapshot_from_video_table(merged_videos)
        snapshot_scheduled_count = int(snapshot_processed_count + max(0, int(failed_count)))

        run_row = {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "schema_name": OUTPUT_SCHEMA_NAME,
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "out_dir": str(self.out_dir),
            "zarr_path": str(self.zarr_path),
            "videos_parquet_path": str(self.videos_parquet_path),
            "frames_parquet_path": str(self.frames_parquet_path),
            "runs_parquet_path": str(self.runs_parquet_path),
            "args_json": json.dumps(self.args_snapshot, ensure_ascii=False, sort_keys=True, default=str),
            "versions_json": json.dumps(self.versions, ensure_ascii=False, sort_keys=True, default=str),
            "sample_count_existing": int(len(self._existing_ids)),
            "sample_count_scheduled": int(snapshot_scheduled_count),
            "sample_count_processed": int(snapshot_processed_count),
            "sample_count_skipped": int(skipped_count),
            "sample_count_failed": int(failed_count),
            "segments_mode": bool(segments_mode),
            "jobs": int(jobs),
            "seed": int(seed),
            "mp_backend": str(mp_backend),
            "quality_score": snapshot_aggregate.get("quality_score"),
            "swap_rate": snapshot_aggregate.get("swap_rate"),
            "missing_rate": snapshot_aggregate.get("missing_rate"),
            "outlier_rate": snapshot_aggregate.get("outlier_rate"),
            "sanity_reject_rate": snapshot_aggregate.get("sanity_reject_rate"),
            "pp_filled_frac": snapshot_aggregate.get("pp_filled_frac"),
            "pp_smoothing_delta": snapshot_aggregate.get("pp_smoothing_delta"),
        }
        run_schema = pa.schema([self._field_to_arrow(name, kind) for name, kind in RUN_PARQUET_COLUMNS])
        run_table = pa.Table.from_pylist([normalize_row(run_row, RUN_PARQUET_COLUMNS)], schema=run_schema)
        existing_runs = self._read_table_or_empty(self.runs_parquet_path, RUN_PARQUET_COLUMNS)
        if existing_runs.schema != run_schema:
            existing_runs = pa.Table.from_pylist([], schema=run_schema)
        merged_runs = pa.concat_tables([existing_runs, run_table], promote_options="default")
        pq.write_table(merged_runs, self.runs_parquet_path)

        self._root.attrs.update(
            {
                "latest_run_id": self.run_id,
                "latest_run_created_at": self.created_at,
                "latest_run_summary": {
                    "scheduled_count": int(snapshot_scheduled_count),
                    "skipped_count": int(skipped_count),
                    "failed_count": int(failed_count),
                    "processed_count": int(snapshot_processed_count),
                    "segments_mode": bool(segments_mode),
                    "jobs": int(jobs),
                    "seed": int(seed),
                    "mp_backend": str(mp_backend),
                    "aggregate": dict(snapshot_aggregate),
                },
            }
        )

        if self.current_run_dir.exists():
            shutil.rmtree(self.current_run_dir)

        return {
            "zarr_path": str(self.zarr_path),
            "videos_parquet_path": str(self.videos_parquet_path),
            "frames_parquet_path": str(self.frames_parquet_path),
            "runs_parquet_path": str(self.runs_parquet_path),
        }
