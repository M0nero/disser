from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def detect_parquet_engine() -> str | None:
    try:
        import pyarrow  # noqa: F401

        return "pyarrow"
    except Exception:
        pass
    try:
        import fastparquet  # noqa: F401

        return "fastparquet"
    except Exception:
        return None


def write_predictions_table(
    *,
    rows: list[dict[str, Any]],
    out_dir: Path,
    basename: str = "oof_predictions",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{basename}.csv"
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    engine = detect_parquet_engine()
    parquet_path = out_dir / f"{basename}.parquet"
    parquet_written = False
    parquet_error = ""
    if engine is not None:
        try:
            pd.DataFrame(rows).to_parquet(parquet_path, index=False, engine=engine)
            parquet_written = True
        except Exception as exc:
            parquet_error = f"{type(exc).__name__}: {exc}"
    return {
        "csv_path": str(csv_path.resolve()),
        "parquet_path": (str(parquet_path.resolve()) if parquet_written else ""),
        "parquet_engine": (engine or ""),
        "parquet_written": bool(parquet_written),
        "parquet_error": parquet_error,
    }


def read_predictions_table(oof_dir: str | Path, basename: str = "oof_predictions") -> list[dict[str, Any]]:
    base = Path(oof_dir)
    parquet_path = base / f"{basename}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path).to_dict(orient="records")
    csv_path = base / f"{basename}.csv"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise FileNotFoundError(f"Expected {parquet_path.name} or {csv_path.name} in {base}")


def write_sharded_array(
    *,
    out_dir: Path,
    name: str,
    sample_ids: list[str],
    array: np.ndarray,
    shard_size: int = 4096,
) -> dict[str, Any]:
    if int(array.shape[0]) != len(sample_ids):
        raise ValueError(f"{name}: sample_ids length {len(sample_ids)} does not match array rows {array.shape[0]}")
    shard_root = out_dir / f"{name}_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    for start in range(0, len(sample_ids), int(shard_size)):
        end = min(len(sample_ids), start + int(shard_size))
        shard_path = shard_root / f"part-{start:06d}-{end:06d}.npz"
        np.savez_compressed(
            shard_path,
            sample_id=np.asarray(sample_ids[start:end], dtype=object),
            data=np.asarray(array[start:end]),
        )
        files.append(
            {
                "path": str(shard_path.resolve()),
                "start": int(start),
                "end": int(end),
                "rows": int(end - start),
            }
        )
    manifest = {
        "version": 1,
        "name": str(name),
        "rows": int(array.shape[0]),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "shard_size": int(shard_size),
        "files": files,
    }
    manifest_path = shard_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return {
        "root": str(shard_root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "rows": int(array.shape[0]),
        "shards": int(len(files)),
    }


def read_sharded_array(oof_dir: str | Path, name: str) -> np.ndarray:
    shard_root = Path(oof_dir) / f"{name}_shards"
    manifest_path = shard_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Sharded manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    parts: list[np.ndarray] = []
    for file_info in manifest.get("files", []):
        path = Path(file_info["path"])
        payload = np.load(path, allow_pickle=True)
        parts.append(np.asarray(payload["data"]))
    return np.concatenate(parts, axis=0) if parts else np.empty((0,), dtype=np.float32)
