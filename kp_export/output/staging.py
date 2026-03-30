from __future__ import annotations

import json
import gzip
import os
import pickle
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..process.contracts import SamplePayload


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _payload_to_dict(payload: Dict[str, Any] | SamplePayload) -> Dict[str, Any]:
    if isinstance(payload, SamplePayload):
        return payload.to_dict()
    return dict(payload)


def write_staged_payload(stage_dir: str | Path, sample_id: str, payload: Dict[str, Any] | SamplePayload) -> Path:
    root = Path(stage_dir)
    root.mkdir(parents=True, exist_ok=True)
    safe_id = str(sample_id).replace(os.sep, "__")
    final_path = root / f"{safe_id}.{uuid.uuid4().hex}.payload"
    tmp_path = root / f"{final_path.name}.tmp"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    payload_dict = _payload_to_dict(payload)
    meta = {
        "sample_id": payload_dict["sample_id"],
        "slug": payload_dict["slug"],
        "source_video": payload_dict["source_video"],
        "sample_attrs": _json_ready(payload_dict.get("sample_attrs", {})),
        "video_row": _json_ready(payload_dict.get("video_row", {})),
        "runtime_metrics": _json_ready(payload_dict.get("runtime_metrics")),
    }
    (tmp_path / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    with (tmp_path / "frame_rows.jsonl").open("w", encoding="utf-8") as f:
        for row in payload_dict.get("frame_rows", []):
            f.write(json.dumps(_json_ready(row), ensure_ascii=False, sort_keys=True) + "\n")
    np.savez_compressed(tmp_path / "raw.npz", **payload_dict.get("raw_arrays", {}))
    if payload_dict.get("pp_arrays") is not None:
        np.savez_compressed(tmp_path / "pp.npz", **payload_dict["pp_arrays"])

    os.replace(tmp_path, final_path)
    return final_path


def load_staged_payload(path: str | Path) -> Dict[str, Any]:
    payload_path = Path(path)
    if payload_path.is_dir():
        meta = json.loads((payload_path / "meta.json").read_text(encoding="utf-8"))
        frame_rows = []
        with (payload_path / "frame_rows.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    frame_rows.append(json.loads(line))
        with np.load(payload_path / "raw.npz", allow_pickle=False) as raw_npz:
            raw_arrays = {key: raw_npz[key] for key in raw_npz.files}
        pp_arrays = None
        pp_path = payload_path / "pp.npz"
        if pp_path.exists():
            with np.load(pp_path, allow_pickle=False) as pp_npz:
                pp_arrays = {key: pp_npz[key] for key in pp_npz.files}
        return {
            "sample_id": meta["sample_id"],
            "slug": meta["slug"],
            "source_video": meta["source_video"],
            "sample_attrs": meta.get("sample_attrs", {}),
            "video_row": meta.get("video_row", {}),
            "frame_rows": frame_rows,
            "raw_arrays": raw_arrays,
            "pp_arrays": pp_arrays,
            "runtime_metrics": meta.get("runtime_metrics"),
        }
    with gzip.open(payload_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid staged payload at {payload_path}: expected dict")
    return payload


def remove_staged_payload(path: str | Path) -> None:
    payload_path = Path(path)
    try:
        if payload_path.is_dir():
            shutil.rmtree(payload_path)
        else:
            payload_path.unlink()
    except FileNotFoundError:
        return
