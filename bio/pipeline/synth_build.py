from __future__ import annotations

"""
Step 2 (offline): build synthetic "continuous" BIO sequences from Step-1 prelabels.

This script creates a new dataset directory with:
- shards/*.npz (or samples/*.npz)
- index.json + index.csv describing the generated sequences
- stats.json with pool + generation statistics

The generator consumes Step-1 artifacts from `bio/pipeline/prelabel.py`:
- prelabel_dir/index.json or index.csv
- prelabel_dir/<sample_id>.npz with keys: pts, mask, bio, label_str, is_no_event, start_idx, end_idx

Usage (example):
  python -m bio synth-build \
      --prelabel_dir /path/to/prelabels_train \
      --out_dir /path/to/synth_train \
      --num_samples 200000 \
      --seq_len 256 \
      --shard_size 256 \
      --seed 1337

If you want "infinite" synthetic data, you can skip this script and use
`SyntheticContinuousDataset` from bio/core/datasets/synth_dataset.py directly during training.
"""

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bio.core.config_utils import load_config_section, write_dataset_manifest, write_run_config
from bio.core.datasets.synth_dataset import (
    SynthConfig,
    load_prelabel_index,
    summarize_pools,
    SyntheticContinuousDataset,
)


@dataclass
class BuildStats:
    num_samples: int
    seq_len: int
    shard_size: int
    seed: int
    config: Dict[str, Any]
    pools: Dict[str, Any]
    generated: Dict[str, Any]
    dataset_signature: Dict[str, Any]


_WORKER_DATASET: Optional[SyntheticContinuousDataset] = None
_WORKER_SHARDS_DIR: Optional[Path] = None
_WORKER_SEQ_LEN: int = 0


def _synth_auto_workers_cache_path() -> Path:
    if platform.system() == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "bio" / "synth_auto_workers_v1.json"
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "bio" / "synth_auto_workers_v1.json"


def _load_synth_auto_workers_cache() -> Dict[str, Any]:
    path = _synth_auto_workers_cache_path()
    if not path.exists():
        return {"version": 1, "records": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "records": {}}
    if not isinstance(payload, dict):
        return {"version": 1, "records": {}}
    records = payload.get("records")
    if not isinstance(records, dict):
        records = {}
    return {"version": 1, "records": records}


def _save_synth_auto_workers_record(fingerprint_hash: str, record: Dict[str, Any]) -> Path:
    path = _synth_auto_workers_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _load_synth_auto_workers_cache()
    payload.setdefault("records", {})[str(fingerprint_hash)] = record
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _path_str(path: Path) -> str:
    return path.as_posix()


def _length_stats(values: Sequence[int]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.int32)
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }


def _count_histogram(counts: Sequence[int]) -> Dict[str, int]:
    if not counts:
        return {}
    bins = [
        ("1", 1, 1),
        ("2", 2, 2),
        ("3", 3, 3),
        ("4", 4, 4),
        ("5-9", 5, 9),
        ("10-19", 10, 19),
        ("20+", 20, None),
    ]
    out: Dict[str, int] = {}
    for name, lo, hi in bins:
        total = 0
        for value in counts:
            if hi is None:
                ok = int(value) >= lo
            else:
                ok = lo <= int(value) <= hi
            if ok:
                total += 1
        out[name] = int(total)
    return out


def _exact_histogram(values: Sequence[int]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        key = str(int(value))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True)
    except Exception:
        return ""
    return out.strip()


def _config_sha(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _summarize_prelabel_source(path: Path) -> Dict[str, Any]:
    rows = load_prelabel_index(path)
    datasets: Dict[str, int] = {}
    splits: Dict[str, int] = {}
    for row in rows:
        dataset = str(row.dataset or "")
        split = str(row.split or "")
        datasets[dataset or ""] = datasets.get(dataset or "", 0) + 1
        splits[split or ""] = splits.get(split or "", 0) + 1
    return {
        "dir": _path_str(path),
        "summary": _load_json_if_exists(path / "summary.json"),
        "dataset_manifest": _load_json_if_exists(path / "dataset_manifest.json"),
        "rows": int(len(rows)),
        "datasets": datasets,
        "splits": splits,
    }


def _write_index(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    (out_dir / "index.json").write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")

    fields = ["id", "path_to_npz", "seq_len", "V"]
    with (out_dir / "index.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def _save_shard(path: Path, pts: np.ndarray, mask: np.ndarray, bio: np.ndarray, metas: List[str]) -> None:
    # pts: (N,L,V,3), mask: (N,L,V,1), bio: (N,L)
    np.savez(
        path,
        pts=pts.astype(np.float32, copy=False),
        mask=mask.astype(np.float32, copy=False),
        bio=bio.astype(np.uint8, copy=False),
        meta=np.asarray(metas, dtype=np.unicode_),
    )


def _shard_name(shard_id: int) -> str:
    return f"shard_{int(shard_id):06d}.npz"


def _shard_path(shards_dir: Path, shard_id: int) -> Path:
    return shards_dir / _shard_name(shard_id)


def _shard_sidecar_path(shards_dir: Path, shard_id: int) -> Path:
    return shards_dir / f"shard_{int(shard_id):06d}.stats.json"


def _write_shard_sidecar(shards_dir: Path, shard_result: Dict[str, Any]) -> None:
    payload = {
        "shard_id": int(shard_result["shard_id"]),
        "index_row": dict(shard_result["index_row"]),
        "stats": dict(shard_result["stats"]),
        "num_samples": int(shard_result.get("num_samples", 0)),
    }
    _shard_sidecar_path(shards_dir, int(shard_result["shard_id"])).write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _load_shard_result_from_disk(shards_dir: Path, shard_id: int) -> Optional[Dict[str, Any]]:
    sidecar_path = _shard_sidecar_path(shards_dir, shard_id)
    if sidecar_path.is_file():
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        return {
            "shard_id": int(payload["shard_id"]),
            "index_row": dict(payload["index_row"]),
            "stats": dict(payload["stats"]),
            "num_samples": int(payload.get("num_samples", int(payload["index_row"].get("num_samples", 0)))),
        }

    shard_path = _shard_path(shards_dir, shard_id)
    if not shard_path.is_file():
        return None

    with np.load(shard_path, allow_pickle=True) as data:
        bio = np.asarray(data["bio"], dtype=np.uint8)
        metas_raw = data["meta"]
        seq_len = int(bio.shape[1]) if bio.ndim >= 2 else 0
        V = int(data["pts"].shape[2]) if "pts" in data and data["pts"].ndim >= 3 else 0
        num_samples = int(bio.shape[0]) if bio.ndim >= 1 else 0
        acc = _new_generated_accumulator()
        for idx in range(num_samples):
            meta_item = metas_raw[idx]
            meta = json.loads(str(meta_item))
            _accumulate_sample_stats(
                acc,
                pts=np.zeros((seq_len, max(1, V), 3), dtype=np.float32),
                bio=np.asarray(bio[idx], dtype=np.uint8),
                meta=meta,
            )
    shard_result = {
        "shard_id": int(shard_id),
        "index_row": {
            "id": f"shard_{int(shard_id):06d}",
            "path_to_npz": (Path("shards") / _shard_name(shard_id)).as_posix(),
            "seq_len": int(seq_len),
            "V": int(V),
            "num_samples": int(num_samples),
        },
        "stats": acc,
        "num_samples": int(num_samples),
    }
    _write_shard_sidecar(shards_dir, shard_result)
    return shard_result


def _candidate_synth_workers(auto_workers_max: int) -> List[int]:
    upper = max(1, int(auto_workers_max))
    ladder = [1, 2, 4, 8, 12, 16, upper]
    return sorted({max(1, int(x)) for x in ladder if int(x) <= upper} | {1, upper})


def _build_synth_auto_workers_fingerprint(
    *,
    prelabel_dir: Path,
    preferred_dirs: Sequence[Path],
    extra_dirs: Sequence[Path],
    cfg: SynthConfig,
    requested_workers: int,
    auto_workers_max: int,
    probe_samples: int,
    shard_size: int,
    seed: int,
    min_sign_len: int,
) -> Tuple[Dict[str, Any], str]:
    payload = {
        "version": 1,
        "platform": {
            "system": platform.system(),
            "python": platform.python_version(),
            "cpu_count": int(os.cpu_count() or 1),
        },
        "inputs": {
            "prelabel_dir": str(prelabel_dir.resolve()),
            "preferred_noev_prelabel_dirs": [str(Path(x).resolve()) for x in preferred_dirs],
            "extra_noev_prelabel_dirs": [str(Path(x).resolve()) for x in extra_dirs],
        },
        "generation": {
            "requested_workers": int(requested_workers),
            "auto_workers_max": int(auto_workers_max),
            "probe_samples": int(probe_samples),
            "shard_size": int(shard_size),
            "seed": int(seed),
            "min_sign_len": int(min_sign_len),
            "cfg": asdict(cfg),
        },
    }
    blob = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return payload, hashlib.sha1(blob).hexdigest()


def _new_generated_accumulator() -> Dict[str, Any]:
    return {
        "generated_b_count": 0,
        "generated_i_count": 0,
        "generated_o_count": 0,
        "generation_mode_counts": {},
        "samples_with_b": 0,
        "samples_with_leading_o_prefix": 0,
        "all_o_samples": 0,
        "first_b_frames": [],
        "first_b_frame_zero": 0,
        "corrupted_samples": 0,
        "startup_prefix_no_hand_frames": 0,
        "startup_prefix_total_frames": 0,
        "longest_no_hand_spans": [],
        "corruption_type_counts": {},
        "no_event_source_counts": {},
        "no_event_role_counts": {},
        "source_group_counts": {},
        "sign_label_counts": {},
        "sign_label_frame_counts": {},
        "sign_source_group_counts": {},
        "sign_source_group_frames": {},
        "dense_signer_samples": 0,
        "sparse_signer_samples": 0,
        "dense_signer_multi_sign_samples": 0,
        "sparse_signer_multi_sign_samples": 0,
        "gap_lens": [],
        "pad_lens": [],
        "tail_lens": [],
        "expanded_boundary_center_deltas": [],
        "expanded_internal_center_deltas": [],
        "expanded_boundary_scale_deltas": [],
        "expanded_internal_scale_deltas": [],
        "expanded_boundary_valid_joint_deltas": [],
        "expanded_internal_valid_joint_deltas": [],
        "expanded_boundary_wrist_deltas": [],
        "expanded_internal_wrist_deltas": [],
        "expanded_boundary_type_counts": {},
        "expanded_same_source_boundary_count": 0,
        "expanded_cross_source_boundary_count": 0,
        "semantic_boundary_center_deltas": [],
        "semantic_internal_center_deltas": [],
        "semantic_boundary_scale_deltas": [],
        "semantic_internal_scale_deltas": [],
        "semantic_boundary_valid_joint_deltas": [],
        "semantic_internal_valid_joint_deltas": [],
        "semantic_boundary_wrist_deltas": [],
        "semantic_internal_wrist_deltas": [],
        "semantic_boundary_type_counts": {},
        "semantic_same_source_boundary_count": 0,
        "semantic_cross_source_boundary_count": 0,
        "transition_frame_center_deltas": [],
        "transition_frame_scale_deltas": [],
        "transition_frame_wrist_deltas": [],
        "post_transition_internal_center_deltas": [],
        "post_transition_internal_scale_deltas": [],
        "post_transition_internal_wrist_deltas": [],
    }


def _merge_int_dict(dst: Dict[str, int], src: Dict[str, Any]) -> None:
    for key, value in dict(src or {}).items():
        dst[str(key)] = int(dst.get(str(key), 0)) + int(value)


def _accumulate_seam_scope(acc: Dict[str, Any], prefix: str, diag: Dict[str, Any]) -> None:
    if not diag:
        return
    for src_key, dst_key in (
        ("boundary_center_deltas", f"{prefix}_boundary_center_deltas"),
        ("internal_center_deltas", f"{prefix}_internal_center_deltas"),
        ("boundary_scale_deltas", f"{prefix}_boundary_scale_deltas"),
        ("internal_scale_deltas", f"{prefix}_internal_scale_deltas"),
        ("boundary_valid_joint_deltas", f"{prefix}_boundary_valid_joint_deltas"),
        ("internal_valid_joint_deltas", f"{prefix}_internal_valid_joint_deltas"),
        ("boundary_wrist_deltas", f"{prefix}_boundary_wrist_deltas"),
        ("internal_wrist_deltas", f"{prefix}_internal_wrist_deltas"),
    ):
        acc[dst_key].extend(float(v) for v in list(diag.get(src_key, []) or []))
    _merge_int_dict(acc[f"{prefix}_boundary_type_counts"], dict(diag.get("boundary_type_counts", {}) or {}))
    acc[f"{prefix}_same_source_boundary_count"] += int(diag.get("same_source_boundary_count", 0) or 0)
    acc[f"{prefix}_cross_source_boundary_count"] += int(diag.get("cross_source_boundary_count", 0) or 0)


def _accumulate_sample_stats(acc: Dict[str, Any], *, pts: np.ndarray, bio: np.ndarray, meta: Dict[str, Any]) -> None:
    acc["generated_b_count"] += int((bio == 1).sum())
    acc["generated_i_count"] += int((bio == 2).sum())
    acc["generated_o_count"] += int((bio == 0).sum())
    if (bio == 1).any():
        acc["samples_with_b"] += 1
    if bool(meta.get("has_leading_o_prefix", False)):
        acc["samples_with_leading_o_prefix"] += 1
    if bool(meta.get("is_all_noev", False)):
        acc["all_o_samples"] += 1
    generation_mode = str(meta.get("generation_mode", "") or "unknown")
    acc["generation_mode_counts"][generation_mode] = acc["generation_mode_counts"].get(generation_mode, 0) + 1
    is_dense_signer = bool(meta.get("sequence_is_dense_signer", False))
    num_signs = int(meta.get("num_signs", 0) or 0)
    if is_dense_signer:
        acc["dense_signer_samples"] += 1
        if num_signs > 1:
            acc["dense_signer_multi_sign_samples"] += 1
    else:
        acc["sparse_signer_samples"] += 1
        if num_signs > 1:
            acc["sparse_signer_multi_sign_samples"] += 1
    first_b_frame = meta.get("first_B_frame", None)
    if first_b_frame is not None:
        acc["first_b_frames"].append(int(first_b_frame))
        if int(first_b_frame) == 0:
            acc["first_b_frame_zero"] += 1
    if bool(meta.get("corruption_applied", False)):
        acc["corrupted_samples"] += 1
    startup_prefix_len = int(meta.get("leading_o_prefix_len", 0) or 0)
    if first_b_frame is None:
        startup_prefix_len = int(meta.get("seq_len", pts.shape[0]) or pts.shape[0])
    acc["startup_prefix_total_frames"] += max(0, startup_prefix_len)
    acc["startup_prefix_no_hand_frames"] += int(meta.get("startup_no_hand_frames_after_corruption", 0) or 0)
    acc["longest_no_hand_spans"].append(int(meta.get("longest_no_hand_span_after_corruption", 0) or 0))
    _merge_int_dict(acc["corruption_type_counts"], dict(meta.get("corruption_counts", {}) or {}))
    seam_diag = dict(meta.get("seam_diagnostics", {}) or {})
    _accumulate_seam_scope(acc, "expanded", dict(seam_diag.get("expanded", {}) or seam_diag))
    _accumulate_seam_scope(acc, "semantic", dict(seam_diag.get("semantic", {}) or {}))
    transition_diag = dict(seam_diag.get("transition_frame", {}) or {})
    acc["transition_frame_center_deltas"].extend(float(v) for v in list(transition_diag.get("center_deltas", []) or []))
    acc["transition_frame_scale_deltas"].extend(float(v) for v in list(transition_diag.get("scale_deltas", []) or []))
    acc["transition_frame_wrist_deltas"].extend(float(v) for v in list(transition_diag.get("wrist_deltas", []) or []))
    post_diag = dict(seam_diag.get("post_transition_internal", {}) or {})
    acc["post_transition_internal_center_deltas"].extend(float(v) for v in list(post_diag.get("center_deltas", []) or []))
    acc["post_transition_internal_scale_deltas"].extend(float(v) for v in list(post_diag.get("scale_deltas", []) or []))
    acc["post_transition_internal_wrist_deltas"].extend(float(v) for v in list(post_diag.get("wrist_deltas", []) or []))

    for part in meta.get("parts", []):
        length = int(part.get("length", 0) or 0)
        if length <= 0:
            continue
        role = str(part.get("role", ""))
        part_type = str(part.get("type", ""))
        source_type = str(part.get("source_type", ""))
        source_group = str(part.get("source_group", ""))
        if part_type == "sign":
            label = str(part.get("label_str", "") or "")
            acc["sign_label_counts"][label] = acc["sign_label_counts"].get(label, 0) + 1
            acc["sign_label_frame_counts"][label] = acc["sign_label_frame_counts"].get(label, 0) + length
            if source_group:
                acc["sign_source_group_counts"][source_group] = acc["sign_source_group_counts"].get(source_group, 0) + 1
                acc["sign_source_group_frames"][source_group] = acc["sign_source_group_frames"].get(source_group, 0) + length
            continue
        if part_type not in ("no_event", "blend"):
            continue
        if role == "gap":
            acc["gap_lens"].append(length)
        if role in ("pad_pre", "pad_post"):
            acc["pad_lens"].append(length)
        if source_type in ("tail_pre", "tail_post"):
            acc["tail_lens"].append(length)
        if part_type == "blend":
            key = "blend"
        elif role in ("pad_pre", "pad_post"):
            key = "pad"
        elif source_type:
            key = source_type
        else:
            key = part_type or "unknown"
        acc["no_event_source_counts"][key] = acc["no_event_source_counts"].get(key, 0) + length
        role_key = role or part_type or "unknown"
        acc["no_event_role_counts"][role_key] = acc["no_event_role_counts"].get(role_key, 0) + length
        if source_group:
            acc["source_group_counts"][source_group] = acc["source_group_counts"].get(source_group, 0) + length


def _merge_accumulators(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    scalar_keys = [
        "generated_b_count",
        "generated_i_count",
        "generated_o_count",
        "samples_with_b",
        "samples_with_leading_o_prefix",
        "all_o_samples",
        "first_b_frame_zero",
        "corrupted_samples",
        "startup_prefix_no_hand_frames",
        "startup_prefix_total_frames",
        "dense_signer_samples",
        "sparse_signer_samples",
        "dense_signer_multi_sign_samples",
        "sparse_signer_multi_sign_samples",
    ]
    for key in scalar_keys:
        dst[key] += int(src.get(key, 0))
    list_keys = ["first_b_frames", "longest_no_hand_spans", "gap_lens", "pad_lens", "tail_lens"]
    list_keys.extend(
        [
            "expanded_boundary_center_deltas",
            "expanded_internal_center_deltas",
            "expanded_boundary_scale_deltas",
            "expanded_internal_scale_deltas",
            "expanded_boundary_valid_joint_deltas",
            "expanded_internal_valid_joint_deltas",
            "expanded_boundary_wrist_deltas",
            "expanded_internal_wrist_deltas",
            "semantic_boundary_center_deltas",
            "semantic_internal_center_deltas",
            "semantic_boundary_scale_deltas",
            "semantic_internal_scale_deltas",
            "semantic_boundary_valid_joint_deltas",
            "semantic_internal_valid_joint_deltas",
            "semantic_boundary_wrist_deltas",
            "semantic_internal_wrist_deltas",
            "transition_frame_center_deltas",
            "transition_frame_scale_deltas",
            "transition_frame_wrist_deltas",
            "post_transition_internal_center_deltas",
            "post_transition_internal_scale_deltas",
            "post_transition_internal_wrist_deltas",
        ]
    )
    for key in list_keys:
        dst[key].extend(list(src.get(key, []) or []))
    dict_keys = [
        "generation_mode_counts",
        "corruption_type_counts",
        "no_event_source_counts",
        "no_event_role_counts",
        "source_group_counts",
        "sign_label_counts",
        "sign_label_frame_counts",
        "sign_source_group_counts",
        "sign_source_group_frames",
        "expanded_boundary_type_counts",
        "semantic_boundary_type_counts",
    ]
    for key in dict_keys:
        _merge_int_dict(dst[key], dict(src.get(key, {}) or {}))
    dst["expanded_same_source_boundary_count"] += int(src.get("expanded_same_source_boundary_count", 0))
    dst["expanded_cross_source_boundary_count"] += int(src.get("expanded_cross_source_boundary_count", 0))
    dst["semantic_same_source_boundary_count"] += int(src.get("semantic_same_source_boundary_count", 0))
    dst["semantic_cross_source_boundary_count"] += int(src.get("semantic_cross_source_boundary_count", 0))


def _delta_stats(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": 0.0, "p95": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p95": float(np.percentile(arr, 95)),
    }


def _finalize_seam_realism(
    acc: Dict[str, Any],
    *,
    prefix: str,
    targets: Dict[str, float],
) -> Dict[str, Any]:
    boundary_center_stats = _delta_stats(acc[f"{prefix}_boundary_center_deltas"])
    internal_center_stats = _delta_stats(acc[f"{prefix}_internal_center_deltas"])
    boundary_scale_stats = _delta_stats(acc[f"{prefix}_boundary_scale_deltas"])
    internal_scale_stats = _delta_stats(acc[f"{prefix}_internal_scale_deltas"])
    same_source_count = int(acc.get(f"{prefix}_same_source_boundary_count", 0))
    cross_source_count = int(acc.get(f"{prefix}_cross_source_boundary_count", 0))
    total_boundary_count = max(1, same_source_count + cross_source_count)
    return {
        "targets": dict(targets),
        "boundary_center_delta": boundary_center_stats,
        "internal_center_delta": internal_center_stats,
        "boundary_scale_delta": boundary_scale_stats,
        "internal_scale_delta": internal_scale_stats,
        "boundary_valid_joint_delta": _delta_stats(acc[f"{prefix}_boundary_valid_joint_deltas"]),
        "internal_valid_joint_delta": _delta_stats(acc[f"{prefix}_internal_valid_joint_deltas"]),
        "boundary_wrist_delta": _delta_stats(acc[f"{prefix}_boundary_wrist_deltas"]),
        "internal_wrist_delta": _delta_stats(acc[f"{prefix}_internal_wrist_deltas"]),
        "boundary_internal_center_jump_ratio": (
            float(boundary_center_stats["mean"]) / max(1e-6, float(internal_center_stats["mean"]))
            if int(boundary_center_stats.get("count", 0)) > 0 and int(internal_center_stats.get("count", 0)) > 0
            else 0.0
        ),
        "boundary_internal_scale_jump_ratio": (
            float(boundary_scale_stats["mean"]) / max(1e-6, float(internal_scale_stats["mean"]))
            if int(boundary_scale_stats.get("count", 0)) > 0 and int(internal_scale_stats.get("count", 0)) > 0
            else 0.0
        ),
        "boundary_type_counts": dict(sorted(acc[f"{prefix}_boundary_type_counts"].items())),
        "same_source_boundary_count": same_source_count,
        "cross_source_boundary_count": cross_source_count,
        "cross_source_boundary_frac": float(cross_source_count) / float(total_boundary_count),
    }


def _finalize_source_mixing(acc: Dict[str, Any], *, prefix: str) -> Dict[str, Any]:
    same_source_count = int(acc.get(f"{prefix}_same_source_boundary_count", 0))
    cross_source_count = int(acc.get(f"{prefix}_cross_source_boundary_count", 0))
    total_boundary_count = max(1, same_source_count + cross_source_count)
    return {
        "same_source_boundary_count": same_source_count,
        "cross_source_boundary_count": cross_source_count,
        "same_source_boundary_frac": float(same_source_count) / float(total_boundary_count),
        "cross_source_boundary_frac": float(cross_source_count) / float(total_boundary_count),
        "boundary_type_counts": dict(sorted(acc[f"{prefix}_boundary_type_counts"].items())),
    }


def _finalize_generated_stats(acc: Dict[str, Any], *, num_samples: int, seq_len: int, shard_count: int) -> Dict[str, Any]:
    total_o_frames = max(1, int(acc["generated_o_count"]))
    first_b_frames = list(acc["first_b_frames"])
    profile = str(acc.get("dataset_profile", "main_continuous") or "main_continuous")
    seam_targets = {
        "boundary_internal_center_jump_ratio_max": 2.2 if profile == "main_continuous" else 2.0,
        "boundary_internal_scale_jump_ratio_max": 2.5,
        "cross_source_boundary_frac_max": 0.0 if profile in ("main_continuous", "warmup_single_sign") else 0.01,
    }
    expanded_seam_realism = _finalize_seam_realism(acc, prefix="expanded", targets=seam_targets)
    semantic_seam_realism = _finalize_seam_realism(acc, prefix="semantic", targets=seam_targets)
    return {
        "dataset_profile": str(acc.get("dataset_profile", "main_continuous") or "main_continuous"),
        "total_samples": int(num_samples),
        "samples_total": int(num_samples),
        "total_frames": int(num_samples * seq_len),
        "bio_counts": {
            "B": int(acc["generated_b_count"]),
            "I": int(acc["generated_i_count"]),
            "O": int(acc["generated_o_count"]),
        },
        "bio_fracs": {
            "B": float(acc["generated_b_count"]) / float(num_samples * seq_len),
            "I": float(acc["generated_i_count"]) / float(num_samples * seq_len),
            "O": float(acc["generated_o_count"]) / float(num_samples * seq_len),
        },
        "samples_with_B_frac": float(acc["samples_with_b"]) / float(num_samples) if num_samples > 0 else 0.0,
        "samples_with_leading_o_prefix_frac": float(acc["samples_with_leading_o_prefix"]) / float(num_samples) if num_samples > 0 else 0.0,
        "all_o_samples_frac": float(acc["all_o_samples"]) / float(num_samples) if num_samples > 0 else 0.0,
        "samples_with_hand_corruption_frac": float(acc["corrupted_samples"]) / float(num_samples) if num_samples > 0 else 0.0,
        "generation_mode_counts": dict(sorted(acc["generation_mode_counts"].items())),
        "generation_mode_fracs": {
            k: float(v) / float(max(1, num_samples))
            for k, v in sorted(acc["generation_mode_counts"].items())
        },
        "signer_density_summary": {
            "dense_signer_samples": int(acc["dense_signer_samples"]),
            "sparse_signer_samples": int(acc["sparse_signer_samples"]),
            "dense_signer_multi_sign_samples": int(acc["dense_signer_multi_sign_samples"]),
            "sparse_signer_multi_sign_samples": int(acc["sparse_signer_multi_sign_samples"]),
        },
        "shards": int(shard_count),
        "no_event_source_counts": dict(sorted(acc["no_event_source_counts"].items())),
        "no_event_source_fracs": {
            k: float(v) / float(total_o_frames)
            for k, v in sorted(acc["no_event_source_counts"].items())
        },
        "no_event_role_counts": dict(sorted(acc["no_event_role_counts"].items())),
        "gap_stats": _length_stats(acc["gap_lens"]),
        "pad_stats": _length_stats(acc["pad_lens"]),
        "tail_len_stats": _length_stats(acc["tail_lens"]),
        "first_B_frame_stats": _length_stats(first_b_frames),
        "first_B_frame_distribution": _exact_histogram(first_b_frames),
        "first_B_frame_eq0_frac_over_total": float(acc["first_b_frame_zero"]) / float(num_samples) if num_samples > 0 else 0.0,
        "first_B_frame_eq0_frac_over_samples_with_B": (
            float(acc["first_b_frame_zero"]) / float(len(first_b_frames)) if first_b_frames else 0.0
        ),
        "startup_prefix_no_hand_frame_frac_after_corruption": (
            float(acc["startup_prefix_no_hand_frames"]) / float(acc["startup_prefix_total_frames"])
            if int(acc["startup_prefix_total_frames"]) > 0
            else 0.0
        ),
        "longest_no_hand_span_stats_after_corruption": _length_stats(acc["longest_no_hand_spans"]),
        "corruption_type_counts": dict(sorted(acc["corruption_type_counts"].items())),
        "seam_realism": semantic_seam_realism,
        "semantic_seam_realism": semantic_seam_realism,
        "expanded_seam_realism": expanded_seam_realism,
        "transition_frame_realism": {
            "center_delta": _delta_stats(acc["transition_frame_center_deltas"]),
            "scale_delta": _delta_stats(acc["transition_frame_scale_deltas"]),
            "wrist_delta": _delta_stats(acc["transition_frame_wrist_deltas"]),
        },
        "post_transition_internal_realism": {
            "center_delta": _delta_stats(acc["post_transition_internal_center_deltas"]),
            "scale_delta": _delta_stats(acc["post_transition_internal_scale_deltas"]),
            "wrist_delta": _delta_stats(acc["post_transition_internal_wrist_deltas"]),
        },
        "source_mixing_report": _finalize_source_mixing(acc, prefix="semantic"),
        "semantic_source_mixing_report": _finalize_source_mixing(acc, prefix="semantic"),
        "expanded_source_mixing_report": _finalize_source_mixing(acc, prefix="expanded"),
        "sign_label_counts": dict(sorted(acc["sign_label_counts"].items())),
        "sign_label_frame_counts": dict(sorted(acc["sign_label_frame_counts"].items())),
        "sign_source_group_coverage": {
            "unique_groups": int(len(acc["sign_source_group_counts"])),
            "reuse_histogram": _count_histogram(list(acc["sign_source_group_counts"].values())),
            "top_groups_by_segments": [
                {"source_group": key, "segments": int(value)}
                for key, value in sorted(acc["sign_source_group_counts"].items(), key=lambda item: (-item[1], item[0]))[:10]
            ],
            "top_groups_by_frames": [
                {"source_group": key, "frames": int(value)}
                for key, value in sorted(acc["sign_source_group_frames"].items(), key=lambda item: (-item[1], item[0]))[:10]
            ],
        },
        "source_group_coverage": {
            "unique_groups": int(len(acc["source_group_counts"])),
            "top_groups_by_frames": [
                {"source_group": key, "frames": int(value)}
                for key, value in sorted(acc["source_group_counts"].items(), key=lambda item: (-item[1], item[0]))[:10]
            ],
        },
    }


def evaluate_realism_gate(generated: Dict[str, Any]) -> Dict[str, Any]:
    generated = dict(generated or {})
    profile = str(generated.get("dataset_profile", "main_continuous") or "main_continuous")
    seam = dict(generated.get("semantic_seam_realism", generated.get("seam_realism", {})) or {})
    targets = dict(seam.get("targets", {}) or {})
    if profile == "stress":
        return {
            "profile": profile,
            "passed": True,
            "failures": [],
            "checks": {},
        }
    if profile == "warmup_single_sign":
        checks = {
            "cross_source_boundary_frac": {
                "value": float(seam.get("cross_source_boundary_frac", 0.0)),
                "target_max": 0.0,
            },
            "first_B_frame_eq0_frac_over_total": {
                "value": float(generated.get("first_B_frame_eq0_frac_over_total", 0.0)),
                "target_max": 0.01,
            },
            "samples_with_leading_o_prefix_frac": {
                "value": float(generated.get("samples_with_leading_o_prefix_frac", 0.0)),
                "target_min": 0.02,
            },
            "all_o_samples_frac": {
                "value": float(generated.get("all_o_samples_frac", 0.0)),
                "target_min": 0.02,
            },
        }
        failures: List[str] = []
        for key, row in checks.items():
            value = float(row.get("value", 0.0))
            if "target_max" in row and value > float(row["target_max"]):
                failures.append(f"{key}={value:.4f} > {float(row['target_max']):.4f}")
            if "target_min" in row and value < float(row["target_min"]):
                failures.append(f"{key}={value:.4f} < {float(row['target_min']):.4f}")
        return {
            "profile": profile,
            "passed": not failures,
            "failures": failures,
            "checks": checks,
        }
    checks = {
        "boundary_internal_center_jump_ratio": {
            "value": float(seam.get("boundary_internal_center_jump_ratio", 0.0)),
            "target_max": float(targets.get("boundary_internal_center_jump_ratio_max", 2.2)),
        },
        "boundary_internal_scale_jump_ratio": {
            "value": float(seam.get("boundary_internal_scale_jump_ratio", 0.0)),
            "target_max": float(targets.get("boundary_internal_scale_jump_ratio_max", 2.5)),
        },
        "cross_source_boundary_frac": {
            "value": float(seam.get("cross_source_boundary_frac", 0.0)),
            "target_max": float(targets.get("cross_source_boundary_frac_max", 0.0)),
        },
        "first_B_frame_eq0_frac_over_total": {
            "value": float(generated.get("first_B_frame_eq0_frac_over_total", 0.0)),
            "target_max": 0.01,
        },
        "samples_with_leading_o_prefix_frac": {
            "value": float(generated.get("samples_with_leading_o_prefix_frac", 0.0)),
            "target_min": 0.05,
        },
        "all_o_samples_frac": {
            "value": float(generated.get("all_o_samples_frac", 0.0)),
            "target_min": 0.05,
        },
    }
    failures: List[str] = []
    for key, row in checks.items():
        value = float(row.get("value", 0.0))
        if "target_max" in row and value > float(row["target_max"]):
            failures.append(f"{key}={value:.4f} > {float(row['target_max']):.4f}")
        if "target_min" in row and value < float(row["target_min"]):
            failures.append(f"{key}={value:.4f} < {float(row['target_min']):.4f}")
    return {
        "profile": profile,
        "passed": not failures,
        "failures": failures,
        "checks": checks,
    }


def _init_synth_worker(
    prelabel_dir: str,
    cfg_payload: Dict[str, Any],
    seed: int,
    min_sign_len: int,
    preferred_dirs: Sequence[str],
    extra_dirs: Sequence[str],
    shards_dir: str,
) -> None:
    global _WORKER_DATASET, _WORKER_SHARDS_DIR, _WORKER_SEQ_LEN
    rows = load_prelabel_index(Path(prelabel_dir))
    cfg = SynthConfig(**dict(cfg_payload))
    _WORKER_DATASET = SyntheticContinuousDataset(
        prelabel_dir=Path(prelabel_dir),
        rows=rows,
        cfg=cfg,
        epoch_size=1,
        seed=int(seed),
        min_sign_len=int(min_sign_len),
        preferred_noev_prelabel_dirs=[Path(x) for x in preferred_dirs],
        extra_noev_prelabel_dirs=[Path(x) for x in extra_dirs],
    )
    _WORKER_SHARDS_DIR = Path(shards_dir)
    _WORKER_SEQ_LEN = int(cfg.seq_len)


def _generate_shard_worker(task: Tuple[int, int, int]) -> Dict[str, Any]:
    global _WORKER_DATASET, _WORKER_SHARDS_DIR, _WORKER_SEQ_LEN
    if _WORKER_DATASET is None or _WORKER_SHARDS_DIR is None:
        raise RuntimeError("synth-build worker is not initialized")
    shard_id, start_idx, end_idx = int(task[0]), int(task[1]), int(task[2])
    ds = _WORKER_DATASET
    acc = _new_generated_accumulator()
    pts_buf: List[np.ndarray] = []
    mask_buf: List[np.ndarray] = []
    bio_buf: List[np.ndarray] = []
    meta_buf: List[str] = []
    for i in range(start_idx, end_idx):
        sample = ds[i]
        pts = sample["pts"]
        mask = sample["mask"]
        bio = sample["bio"]
        meta = sample["meta"]
        if int(pts.shape[0]) != int(_WORKER_SEQ_LEN):
            raise RuntimeError(f"BUG: seq_len mismatch (got {pts.shape[0]}, expected {_WORKER_SEQ_LEN})")
        pts_buf.append(pts[None, ...])
        mask_buf.append(mask[None, ...])
        bio_buf.append(bio[None, ...])
        meta_buf.append(json.dumps(meta, ensure_ascii=True))
        _accumulate_sample_stats(acc, pts=pts, bio=bio, meta=meta)
    pts_sh = np.concatenate(pts_buf, axis=0)
    mask_sh = np.concatenate(mask_buf, axis=0)
    bio_sh = np.concatenate(bio_buf, axis=0)
    shard_name = _shard_name(shard_id)
    shard_path = _WORKER_SHARDS_DIR / shard_name
    _save_shard(shard_path, pts_sh, mask_sh, bio_sh, meta_buf)
    shard_result = {
        "shard_id": int(shard_id),
        "index_row": {
            "id": f"shard_{shard_id:06d}",
            "path_to_npz": (Path("shards") / shard_name).as_posix(),
            "seq_len": int(_WORKER_SEQ_LEN),
            "V": int(pts_sh.shape[2]),
            "num_samples": int(pts_sh.shape[0]),
        },
        "stats": acc,
        "num_samples": int(pts_sh.shape[0]),
    }
    _write_shard_sidecar(_WORKER_SHARDS_DIR, shard_result)
    return shard_result


def _generate_shard_worker_serial(
    *,
    ds: SyntheticContinuousDataset,
    shards_dir: Path,
    seq_len: int,
    shard_id: int,
    start_idx: int,
    end_idx: int,
) -> Dict[str, Any]:
    acc = _new_generated_accumulator()
    pts_buf: List[np.ndarray] = []
    mask_buf: List[np.ndarray] = []
    bio_buf: List[np.ndarray] = []
    meta_buf: List[str] = []
    for i in range(int(start_idx), int(end_idx)):
        sample = ds[i]
        pts = sample["pts"]
        mask = sample["mask"]
        bio = sample["bio"]
        meta = sample["meta"]
        if int(pts.shape[0]) != int(seq_len):
            raise RuntimeError(f"BUG: seq_len mismatch (got {pts.shape[0]}, expected {seq_len})")
        pts_buf.append(pts[None, ...])
        mask_buf.append(mask[None, ...])
        bio_buf.append(bio[None, ...])
        meta_buf.append(json.dumps(meta, ensure_ascii=True))
        _accumulate_sample_stats(acc, pts=pts, bio=bio, meta=meta)
    pts_sh = np.concatenate(pts_buf, axis=0)
    mask_sh = np.concatenate(mask_buf, axis=0)
    bio_sh = np.concatenate(bio_buf, axis=0)
    shard_name = _shard_name(int(shard_id))
    shard_path = shards_dir / shard_name
    _save_shard(shard_path, pts_sh, mask_sh, bio_sh, meta_buf)
    shard_result = {
        "shard_id": int(shard_id),
        "index_row": {
            "id": f"shard_{int(shard_id):06d}",
            "path_to_npz": (Path("shards") / shard_name).as_posix(),
            "seq_len": int(seq_len),
            "V": int(pts_sh.shape[2]),
            "num_samples": int(pts_sh.shape[0]),
        },
        "stats": acc,
        "num_samples": int(pts_sh.shape[0]),
    }
    _write_shard_sidecar(shards_dir, shard_result)
    return shard_result


def _measure_synth_worker(task: Tuple[int, int]) -> int:
    global _WORKER_DATASET
    if _WORKER_DATASET is None:
        raise RuntimeError("synth-build worker is not initialized")
    start_idx, end_idx = int(task[0]), int(task[1])
    ds = _WORKER_DATASET
    produced = 0
    for i in range(start_idx, end_idx):
        _ = ds[i]
        produced += 1
    return int(produced)


def _measure_synth_generation_sps(
    *,
    prelabel_dir: Path,
    cfg: SynthConfig,
    preferred_dirs: Sequence[Path],
    extra_dirs: Sequence[Path],
    seed: int,
    min_sign_len: int,
    probe_samples: int,
    workers: int,
) -> Dict[str, Any]:
    workers = max(1, int(workers))
    probe_samples = max(1, int(probe_samples))
    started = time.perf_counter()
    if workers <= 1:
        rows = load_prelabel_index(prelabel_dir)
        ds = SyntheticContinuousDataset(
            prelabel_dir=prelabel_dir,
            rows=rows,
            cfg=cfg,
            epoch_size=probe_samples,
            seed=int(seed),
            min_sign_len=int(min_sign_len),
            preferred_noev_prelabel_dirs=[Path(x) for x in preferred_dirs],
            extra_noev_prelabel_dirs=[Path(x) for x in extra_dirs],
        )
        for i in range(probe_samples):
            _ = ds[i]
        elapsed = max(1e-6, time.perf_counter() - started)
        return {
            "workers": int(workers),
            "success": True,
            "samples_per_sec": float(probe_samples / elapsed),
            "probe_samples": int(probe_samples),
        }

    mp_ctx = mp.get_context("spawn")
    initargs = (
        str(prelabel_dir),
        asdict(cfg),
        int(seed),
        int(min_sign_len),
        [str(x) for x in preferred_dirs],
        [str(x) for x in extra_dirs],
        str(prelabel_dir),  # unused for measurement; initializer still expects a path-like string
    )
    chunk = max(1, int(ceil(probe_samples / float(workers))))
    tasks: List[Tuple[int, int]] = []
    start_idx = 0
    while start_idx < probe_samples:
        end_idx = min(probe_samples, start_idx + chunk)
        tasks.append((int(start_idx), int(end_idx)))
        start_idx = end_idx
    produced = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp_ctx,
        initializer=_init_synth_worker,
        initargs=initargs,
    ) as ex:
        futures = [ex.submit(_measure_synth_worker, task) for task in tasks]
        for future in as_completed(futures):
            produced += int(future.result())
    elapsed = max(1e-6, time.perf_counter() - started)
    return {
        "workers": int(workers),
        "success": bool(produced == probe_samples),
        "samples_per_sec": float(produced / elapsed),
        "probe_samples": int(probe_samples),
    }


def _resolve_synth_workers(
    *,
    prelabel_dir: Path,
    cfg: SynthConfig,
    preferred_dirs: Sequence[Path],
    extra_dirs: Sequence[Path],
    requested_workers: int,
    auto_workers: bool,
    auto_workers_max: int,
    auto_workers_rebench: bool,
    auto_workers_probe_samples: int,
    shard_size: int,
    seed: int,
    min_sign_len: int,
) -> Tuple[int, Dict[str, Any]]:
    requested_workers = max(1, int(requested_workers))
    auto_workers_max = max(requested_workers, int(auto_workers_max or min(int(os.cpu_count() or 1), 16)))
    info: Dict[str, Any] = {
        "enabled": bool(auto_workers),
        "requested_workers": int(requested_workers),
        "from_cache": False,
    }
    if not bool(auto_workers):
        return requested_workers, info

    fingerprint_payload, fingerprint_hash = _build_synth_auto_workers_fingerprint(
        prelabel_dir=prelabel_dir,
        preferred_dirs=preferred_dirs,
        extra_dirs=extra_dirs,
        cfg=cfg,
        requested_workers=requested_workers,
        auto_workers_max=auto_workers_max,
        probe_samples=int(auto_workers_probe_samples),
        shard_size=int(shard_size),
        seed=int(seed),
        min_sign_len=int(min_sign_len),
    )
    info["fingerprint_hash"] = str(fingerprint_hash)
    cached = None if bool(auto_workers_rebench) else _load_synth_auto_workers_cache().get("records", {}).get(str(fingerprint_hash))
    if isinstance(cached, dict):
        selected = max(1, int(cached.get("selected_workers", requested_workers)))
        info.update(
            {
                "from_cache": True,
                "cache_path": str(_synth_auto_workers_cache_path()),
                "candidates": list(cached.get("results", []) or []),
                "selection_reason": str(cached.get("selection_reason", "cache hit")),
                "benchmark_samples_per_sec": float(cached.get("selected_samples_per_sec", 0.0)),
                "fingerprint": fingerprint_payload,
            }
        )
        return selected, info

    results: List[Dict[str, Any]] = []
    for workers in _candidate_synth_workers(auto_workers_max):
        try:
            row = _measure_synth_generation_sps(
                prelabel_dir=prelabel_dir,
                cfg=cfg,
                preferred_dirs=preferred_dirs,
                extra_dirs=extra_dirs,
                seed=int(seed),
                min_sign_len=int(min_sign_len),
                probe_samples=int(auto_workers_probe_samples),
                workers=int(workers),
            )
        except Exception as exc:
            row = {"workers": int(workers), "success": False, "error": str(exc), "samples_per_sec": 0.0}
        results.append(row)
    successes = [row for row in results if bool(row.get("success"))]
    if successes:
        best = max(successes, key=lambda row: float(row.get("samples_per_sec", 0.0)))
        floor = float(best.get("samples_per_sec", 0.0)) * 0.95
        close = [row for row in successes if float(row.get("samples_per_sec", 0.0)) >= floor]
        close.sort(key=lambda row: (int(row.get("workers", 0)), -float(row.get("samples_per_sec", 0.0))))
        picked = close[0]
        selection_reason = (
            f"conservative pick within 5% of best throughput "
            f"({float(best.get('samples_per_sec', 0.0)):.1f} sps)"
        )
    else:
        picked = {"workers": requested_workers, "samples_per_sec": 0.0}
        selection_reason = "all synth auto-workers candidates failed; fell back to requested_workers"
    selected = max(1, int(picked.get("workers", requested_workers)))
    record = {
        "selected_workers": int(selected),
        "results": results,
        "selection_reason": str(selection_reason),
        "selected_samples_per_sec": float(picked.get("samples_per_sec", 0.0)),
        "fingerprint": fingerprint_payload,
    }
    cache_path = _save_synth_auto_workers_record(fingerprint_hash, record)
    info.update(
        {
            "from_cache": False,
            "cache_path": str(cache_path),
            "candidates": results,
            "selection_reason": str(selection_reason),
            "benchmark_samples_per_sec": float(picked.get("samples_per_sec", 0.0)),
            "fingerprint": fingerprint_payload,
        }
    )
    return selected, info


def build_offline(
    prelabel_dir: Path,
    out_dir: Path,
    cfg: SynthConfig,
    num_samples: int,
    shard_size: int,
    seed: int,
    min_sign_len: int,
    workers: int = 1,
    epoch_size: Optional[int] = None,
    preferred_noev_prelabel_dirs: Optional[List[Path]] = None,
    extra_noev_prelabel_dirs: Optional[List[Path]] = None,
    overlap_report_summary: Optional[Dict[str, Any]] = None,
    resume: bool = True,
) -> BuildStats:
    out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    rows = load_prelabel_index(prelabel_dir)
    extra_dirs = [Path(d) for d in (extra_noev_prelabel_dirs or [])]

    # Online dataset as a generator primitive
    ds = SyntheticContinuousDataset(
        prelabel_dir=prelabel_dir,
        rows=rows,
        cfg=cfg,
        epoch_size=int(epoch_size or num_samples),
        seed=seed,
        min_sign_len=min_sign_len,
        preferred_noev_prelabel_dirs=[Path(d) for d in (preferred_noev_prelabel_dirs or [])],
        extra_noev_prelabel_dirs=extra_dirs,
    )

    pools_summary = summarize_pools(
        ds.sign_pool,
        ds.noev_pool,
        noev_pool_primary=ds.noev_pool_primary,
        noev_pool_extra=ds.noev_pool_extra,
    )

    N = int(num_samples)
    shard_size = max(1, int(shard_size))
    L = int(cfg.seq_len)
    total_shards = max(1, (N + shard_size - 1) // shard_size)
    workers = max(1, int(workers))

    print(
        f"[synth-build] start out_dir={out_dir} samples={N} seq_len={L} shard_size={shard_size} total_shards={total_shards} workers={workers}",
        flush=True,
    )
    tasks: List[Tuple[int, int, int]] = []
    for shard_id in range(total_shards):
        start_idx = int(shard_id * shard_size)
        end_idx = int(min(N, start_idx + shard_size))
        tasks.append((int(shard_id), start_idx, end_idx))

    index_rows: List[Dict[str, Any]] = []
    generated_acc = _new_generated_accumulator()
    completed_ids: set[int] = set()
    tasks_to_run: List[Tuple[int, int, int]] = list(tasks)

    if bool(resume):
        resumed_shards = 0
        resumed_samples = 0
        remaining_tasks: List[Tuple[int, int, int]] = []
        for task in tasks:
            shard_result = _load_shard_result_from_disk(shards_dir, int(task[0]))
            if shard_result is None:
                remaining_tasks.append(task)
                continue
            completed_ids.add(int(task[0]))
            index_rows.append(dict(shard_result["index_row"]))
            _merge_accumulators(generated_acc, dict(shard_result["stats"]))
            resumed_shards += 1
            resumed_samples += int(shard_result.get("num_samples", 0))
        tasks_to_run = remaining_tasks
        if resumed_shards > 0:
            print(
                f"[synth-build] resume detected existing_shards={resumed_shards}/{total_shards} "
                f"samples={resumed_samples}/{N}",
                flush=True,
            )

    if workers <= 1:
        samples_done = sum(int(row.get("num_samples", 0) or 0) for row in index_rows)
        completed = len(completed_ids)
        for shard_id, start_idx, end_idx in tasks_to_run:
            shard_result = _generate_shard_worker_serial(
                ds=ds,
                shards_dir=shards_dir,
                seq_len=L,
                shard_id=shard_id,
                start_idx=start_idx,
                end_idx=end_idx,
            )
            completed_ids.add(int(shard_id))
            index_rows.append(dict(shard_result["index_row"]))
            _merge_accumulators(generated_acc, dict(shard_result["stats"]))
            completed += 1
            samples_done += int(shard_result.get("num_samples", 0))
            elapsed = time.time() - start_time
            print(
                f"[synth-build] wrote shard {completed}/{total_shards} samples={samples_done}/{N} "
                f"elapsed_sec={elapsed:.1f}",
                flush=True,
            )
    else:
        mp_ctx = mp.get_context("spawn")
        cfg_payload = asdict(cfg)
        initargs = (
            str(prelabel_dir),
            cfg_payload,
            int(seed),
            int(min_sign_len),
            [str(x) for x in (preferred_noev_prelabel_dirs or [])],
            [str(x) for x in (extra_noev_prelabel_dirs or [])],
            str(shards_dir),
        )
        completed = len(completed_ids)
        samples_done = sum(int(row.get("num_samples", 0) or 0) for row in index_rows)
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp_ctx,
                initializer=_init_synth_worker,
                initargs=initargs,
            ) as ex:
                futures = {ex.submit(_generate_shard_worker, task): task for task in tasks_to_run}
                for future in as_completed(futures):
                    task = futures[future]
                    shard_result = future.result()
                    completed_ids.add(int(task[0]))
                    index_rows.append(dict(shard_result["index_row"]))
                    _merge_accumulators(generated_acc, dict(shard_result["stats"]))
                    completed += 1
                    samples_done += int(shard_result.get("num_samples", 0))
                    elapsed = time.time() - start_time
                    print(
                        f"[synth-build] wrote shard {completed}/{total_shards} samples={samples_done}/{N} "
                        f"elapsed_sec={elapsed:.1f}",
                        flush=True,
                    )
        except BrokenProcessPool as exc:
            remaining_tasks: List[Tuple[int, int, int]] = []
            for task in tasks_to_run:
                if int(task[0]) in completed_ids:
                    continue
                shard_result = _load_shard_result_from_disk(shards_dir, int(task[0]))
                if shard_result is not None:
                    completed_ids.add(int(task[0]))
                    index_rows.append(dict(shard_result["index_row"]))
                    _merge_accumulators(generated_acc, dict(shard_result["stats"]))
                    completed += 1
                    samples_done += int(shard_result.get("num_samples", 0))
                    continue
                remaining_tasks.append(task)
            print(
                f"[synth-build] worker pool broke; falling back to serial for remaining_shards={len(remaining_tasks)} "
                f"(completed={completed}/{total_shards} samples={samples_done}/{N}) reason={exc}",
                flush=True,
            )
            for shard_id, start_idx, end_idx in remaining_tasks:
                shard_result = _generate_shard_worker_serial(
                    ds=ds,
                    shards_dir=shards_dir,
                    seq_len=L,
                    shard_id=shard_id,
                    start_idx=start_idx,
                    end_idx=end_idx,
                )
                completed_ids.add(int(shard_id))
                index_rows.append(dict(shard_result["index_row"]))
                _merge_accumulators(generated_acc, dict(shard_result["stats"]))
                completed += 1
                samples_done += int(shard_result.get("num_samples", 0))
                elapsed = time.time() - start_time
                print(
                    f"[synth-build] wrote shard {completed}/{total_shards} samples={samples_done}/{N} "
                    f"elapsed_sec={elapsed:.1f}",
                    flush=True,
                )

    index_rows.sort(key=lambda row: str(row.get("id", "")))

    _write_index(out_dir, index_rows)

    prelabel_sources = [_summarize_prelabel_source(prelabel_dir)]
    for path in preferred_noev_prelabel_dirs or []:
        prelabel_sources.append(_summarize_prelabel_source(path))
    for path in extra_noev_prelabel_dirs or []:
        prelabel_sources.append(_summarize_prelabel_source(path))
    signature_payload = {
        "prelabel_dir": _path_str(prelabel_dir),
        "preferred_noev_prelabel_dirs": [_path_str(x) for x in (preferred_noev_prelabel_dirs or [])],
        "extra_noev_prelabel_dirs": [_path_str(x) for x in (extra_noev_prelabel_dirs or [])],
        "cfg": asdict(cfg),
        "num_samples": int(N),
        "shard_size": int(shard_size),
        "seed": int(seed),
    }
    dataset_signature = {
        "git_sha": _git_sha(Path(__file__).resolve().parents[2]),
        "config_sha": _config_sha(signature_payload),
        "inputs": {
            "prelabel_dir": _path_str(prelabel_dir),
            "preferred_noev_prelabel_dirs": [_path_str(x) for x in (preferred_noev_prelabel_dirs or [])],
            "extra_noev_prelabel_dirs": [_path_str(x) for x in (extra_noev_prelabel_dirs or [])],
        },
        "prelabel_sources": prelabel_sources,
        "overlap_report_summary": overlap_report_summary or {},
    }

    # Save stats
    generated_acc["dataset_profile"] = str(getattr(cfg, "dataset_profile", "main_continuous") or "main_continuous")
    generated = _finalize_generated_stats(generated_acc, num_samples=N, seq_len=L, shard_count=len(index_rows))
    generated["acceptance"] = evaluate_realism_gate(generated)
    stats = BuildStats(
        num_samples=int(N),
        seq_len=int(L),
        shard_size=int(shard_size),
        seed=int(seed),
        config=asdict(cfg),
        pools=pools_summary,
        generated=generated,
        dataset_signature=dataset_signature,
    )
    (out_dir / "stats.json").write_text(json.dumps(asdict(stats), ensure_ascii=True, indent=2), encoding="utf-8")
    print(
        f"[synth-build] done out_dir={out_dir} shards={len(index_rows)} total_samples={N} elapsed_sec={time.time() - start_time:.1f}",
        flush=True,
    )
    write_dataset_manifest(
        out_dir,
        stage="synth_build",
        args={
            "prelabel_dir": _path_str(prelabel_dir),
            "out_dir": _path_str(out_dir),
            "cfg": asdict(cfg),
            "num_samples": int(N),
            "shard_size": int(shard_size),
            "seed": int(seed),
            "min_sign_len": int(min_sign_len),
            "preferred_noev_prelabel_dirs": [_path_str(x) for x in (preferred_noev_prelabel_dirs or [])],
            "extra_noev_prelabel_dirs": [_path_str(x) for x in (extra_noev_prelabel_dirs or [])],
        },
        section="synth_build",
        inputs=dataset_signature.get("inputs", {}),
        counts={
            "num_samples": int(N),
            "seq_len": int(L),
                "shards": int(len(index_rows)),
        },
        extra={
            "dataset_signature": dataset_signature,
            "generated": generated,
            "pools": pools_summary,
        },
    )
    return stats


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, _ = pre.parse_known_args(argv)
    defaults = load_config_section(pre_args.config, "synth_build")

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=pre_args.config, help="Path to config JSON (section: synth_build).")
    p.add_argument(
        "--prelabel_dir",
        type=str,
        default=defaults.get("prelabel_dir"),
        required=not bool(defaults.get("prelabel_dir")),
        help="Directory produced by prelabel step.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=defaults.get("out_dir"),
        required=not bool(defaults.get("out_dir")),
        help="Where to write synthetic continuous dataset",
    )

    # generation size
    p.add_argument("--num_samples", type=int, default=int(defaults.get("num_samples", 200_000)), help="How many synthetic sequences to generate")
    p.add_argument("--seq_len", type=int, default=int(defaults.get("seq_len", 256)), help="Frames per synthetic sequence")
    p.add_argument("--shard_size", type=int, default=int(defaults.get("shard_size", 256)), help="How many sequences per shard .npz")
    p.add_argument("--seed", type=int, default=int(defaults.get("seed", 1337)))

    # pools
    p.add_argument("--min_sign_len", type=int, default=int(defaults.get("min_sign_len", 8)), help="Discard sign segments shorter than this")
    preferred_default = defaults.get("preferred_noev_prelabel_dir", [])
    if isinstance(preferred_default, str):
        preferred_default = [preferred_default] if preferred_default.strip() else []
    if not isinstance(preferred_default, list):
        preferred_default = []
    p.add_argument(
        "--preferred_noev_prelabel_dir",
        action="append",
        default=None,
        help="Preferred canonical Step1 no_event dirs from the same domain (sampled before external negatives).",
    )
    extra_default = defaults.get("extra_noev_prelabel_dir", [])
    if isinstance(extra_default, str):
        extra_default = [extra_default] if extra_default.strip() else []
    if not isinstance(extra_default, list):
        extra_default = []
    p.add_argument(
        "--extra_noev_prelabel_dir",
        action="append",
        default=None,
        help="Additional canonical Step1 no_event dirs containing no_event samples only (e.g., IPNHand background).",
    )

    # composition
    p.add_argument("--min_signs", type=int, default=int(defaults.get("min_signs", 1)))
    p.add_argument("--max_signs", type=int, default=int(defaults.get("max_signs", 4)))
    p.add_argument("--gap_min", type=int, default=int(defaults.get("gap_min", 0)))
    p.add_argument("--gap_max", type=int, default=int(defaults.get("gap_max", 20)))
    p.add_argument("--leading_noev_prob", type=float, default=float(defaults.get("leading_noev_prob", 0.35)))
    p.add_argument("--leading_noev_min", type=int, default=int(defaults.get("leading_noev_min", 8)))
    p.add_argument("--leading_noev_max", type=int, default=int(defaults.get("leading_noev_max", 64)))
    p.add_argument("--all_noev_prob", type=float, default=float(defaults.get("all_noev_prob", 0.10)))
    p.add_argument("--all_noev_min_len", type=int, default=int(defaults.get("all_noev_min_len", 64)))
    p.add_argument("--all_noev_max_len", type=int, default=int(defaults.get("all_noev_max_len", 0)))
    p.add_argument("--single_hand_dropout_prob", type=float, default=float(defaults.get("single_hand_dropout_prob", 0.35)))
    p.add_argument("--single_hand_dropout_span_min", type=int, default=int(defaults.get("single_hand_dropout_span_min", 3)))
    p.add_argument("--single_hand_dropout_span_max", type=int, default=int(defaults.get("single_hand_dropout_span_max", 18)))
    p.add_argument("--both_hands_dropout_prob", type=float, default=float(defaults.get("both_hands_dropout_prob", 0.12)))
    p.add_argument("--both_hands_dropout_span_min", type=int, default=int(defaults.get("both_hands_dropout_span_min", 2)))
    p.add_argument("--both_hands_dropout_span_max", type=int, default=int(defaults.get("both_hands_dropout_span_max", 12)))
    p.add_argument("--mask_flicker_prob", type=float, default=float(defaults.get("mask_flicker_prob", 0.25)))
    p.add_argument("--mask_flicker_span_min", type=int, default=int(defaults.get("mask_flicker_span_min", 1)))
    p.add_argument("--mask_flicker_span_max", type=int, default=int(defaults.get("mask_flicker_span_max", 4)))
    p.add_argument("--joint_jitter_prob", type=float, default=float(defaults.get("joint_jitter_prob", 0.35)))
    p.add_argument("--joint_jitter_std", type=float, default=float(defaults.get("joint_jitter_std", 0.015)))
    p.add_argument("--sign_ctx_pre", type=int, default=int(defaults.get("sign_ctx_pre", 0)))
    p.add_argument("--sign_ctx_post", type=int, default=int(defaults.get("sign_ctx_post", 0)))
    p.add_argument("--include_sign_tails_as_noev", dest="include_sign_tails_as_noev", action="store_true", default=bool(defaults.get("include_sign_tails_as_noev", True)))
    p.add_argument("--no_include_sign_tails_as_noev", dest="include_sign_tails_as_noev", action="store_false")
    p.add_argument("--min_tail_len", type=int, default=int(defaults.get("min_tail_len", 4)))
    p.add_argument("--primary_noev_prob", type=float, default=float(defaults.get("primary_noev_prob", 0.90)))
    p.add_argument(
        "--source_sampling",
        type=str,
        default=str(defaults.get("source_sampling", "uniform_source")),
        choices=["uniform_segment", "uniform_source"],
        help="How to sample no_event clips inside each pool (uniform_source = source-group aware).",
    )
    p.add_argument(
        "--sign_sampling",
        type=str,
        default=str(defaults.get("sign_sampling", defaults.get("source_sampling", "uniform_source"))),
        choices=["uniform_segment", "uniform_source", "uniform_label_source"],
        help="How to sample sign clips (uniform_label_source = label-balanced, then source-group balanced).",
    )
    p.add_argument(
        "--stitch_noev_chunks",
        dest="stitch_noev_chunks",
        action="store_true",
        default=bool(defaults.get("stitch_noev_chunks", True)),
        help="Fill long no_event spans by stitching multiple clips instead of repeat-padding a single clip.",
    )
    p.add_argument("--no_stitch_noev_chunks", dest="stitch_noev_chunks", action="store_false")
    p.add_argument(
        "--overlap_report",
        type=str,
        default=str(defaults.get("overlap_report", "")),
        help="Optional overlap report JSON. If provided, a short summary is embedded into stats.json.",
    )

    # blending
    p.add_argument("--blend_prob", type=float, default=float(defaults.get("blend_prob", 0.25)))
    p.add_argument("--blend_k_min", type=int, default=int(defaults.get("blend_k_min", 2)))
    p.add_argument("--blend_k_max", type=int, default=int(defaults.get("blend_k_max", 6)))
    p.add_argument("--blend_only_when_gap_leq", type=int, default=int(defaults.get("blend_only_when_gap_leq", 2)))
    p.add_argument("--align_chunks", dest="align_chunks", action="store_true", default=bool(defaults.get("align_chunks", True)))
    p.add_argument("--no_align_chunks", dest="align_chunks", action="store_false")
    p.add_argument("--transition_all_boundaries", dest="transition_all_boundaries", action="store_true", default=bool(defaults.get("transition_all_boundaries", True)))
    p.add_argument("--no_transition_all_boundaries", dest="transition_all_boundaries", action="store_false")
    p.add_argument("--transition_k_min", type=int, default=int(defaults.get("transition_k_min", 4)))
    p.add_argument("--transition_k_max", type=int, default=int(defaults.get("transition_k_max", 12)))
    p.add_argument("--same_source_sequence_prob", type=float, default=float(defaults.get("same_source_sequence_prob", 0.85)))
    p.add_argument("--cross_source_boundary_prob", type=float, default=float(defaults.get("cross_source_boundary_prob", 0.05)))
    p.add_argument(
        "--dataset_profile",
        type=str,
        default=str(defaults.get("dataset_profile", "main_continuous")),
        choices=["warmup_single_sign", "main_continuous", "stress"],
    )
    p.add_argument("--dense_signer_min_clips", type=int, default=int(defaults.get("dense_signer_min_clips", 8)))
    p.add_argument("--continuous_mode_weight", type=float, default=float(defaults.get("continuous_mode_weight", 0.70)))
    p.add_argument("--hard_negative_mode_weight", type=float, default=float(defaults.get("hard_negative_mode_weight", 0.20)))
    p.add_argument("--stress_mode_weight", type=float, default=float(defaults.get("stress_mode_weight", 0.10)))
    p.add_argument("--continuous_stats_dir", type=str, default=str(defaults.get("continuous_stats_dir", "")))
    p.add_argument(
        "--sampling_profile",
        type=str,
        default=str(defaults.get("sampling_profile", "prelabel_empirical")),
        choices=["runtime_empirical", "prelabel_empirical"],
    )
    p.add_argument("--preprocessing_version", type=str, default=str(defaults.get("preprocessing_version", "canonical_hands42_v3")))
    p.add_argument("--preprocessing_center_alpha", type=float, default=float(defaults.get("preprocessing_center_alpha", 0.2)))
    p.add_argument("--preprocessing_scale_alpha", type=float, default=float(defaults.get("preprocessing_scale_alpha", 0.1)))
    p.add_argument("--preprocessing_min_scale", type=float, default=float(defaults.get("preprocessing_min_scale", 1e-3)))
    p.add_argument(
        "--preprocessing_min_visible_joints_for_scale",
        type=int,
        default=int(defaults.get("preprocessing_min_visible_joints_for_scale", 4)),
    )

    # crop/pad
    p.add_argument("--crop_mode", type=str, default=defaults.get("crop_mode", "random"), choices=["random", "start", "center"])
    p.add_argument("--pad_mode", type=str, default=defaults.get("pad_mode", "both_no_event"), choices=["end_no_event", "both_no_event"])

    # caching
    p.add_argument("--npz_cache_items", type=int, default=int(defaults.get("npz_cache_items", 32)))
    p.add_argument(
        "--workers",
        type=int,
        default=int(defaults.get("workers", 1)),
        help="Manual worker count for offline synth shard generation when auto-workers is disabled.",
    )
    p.add_argument("--auto_workers", action="store_true", default=bool(defaults.get("auto_workers", False)), help="Benchmark and cache an effective synth-build worker count.")
    p.add_argument("--no_auto_workers", dest="auto_workers", action="store_false")
    p.add_argument("--auto_workers_max", type=int, default=int(defaults.get("auto_workers_max", 0)), help="Upper bound for synth auto-worker benchmarking (0=heuristic default).")
    p.add_argument("--auto_workers_rebench", action="store_true", default=bool(defaults.get("auto_workers_rebench", False)), help="Ignore cached synth auto-worker decisions and benchmark again.")
    p.add_argument("--auto_workers_probe_samples", type=int, default=int(defaults.get("auto_workers_probe_samples", 512)), help="Number of synthetic samples used to benchmark synth auto-workers.")
    p.add_argument("--fail_on_realism_gate", action="store_true", default=bool(defaults.get("fail_on_realism_gate", False)), help="Exit non-zero if the generated synth dataset does not pass realism thresholds.")
    p.add_argument("--resume", dest="resume", action="store_true", default=bool(defaults.get("resume", True)), help="Reuse already written shard files in out_dir/shards and continue from the missing shards.")
    p.add_argument("--no_resume", dest="resume", action="store_false")

    args = p.parse_args(argv)
    if args.preferred_noev_prelabel_dir is None:
        args.preferred_noev_prelabel_dir = list(preferred_default)
    if args.extra_noev_prelabel_dir is None:
        args.extra_noev_prelabel_dir = list(extra_default)
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    prelabel_dir = Path(args.prelabel_dir)
    out_dir = Path(args.out_dir)
    preferred_dirs = list(dict.fromkeys(Path(x) for x in args.preferred_noev_prelabel_dir))
    extra_dirs = list(dict.fromkeys(Path(x) for x in args.extra_noev_prelabel_dir))
    write_run_config(out_dir, args, config_path=args.config, section="synth_build")

    cfg = SynthConfig(
        seq_len=int(args.seq_len),
        min_signs=int(args.min_signs),
        max_signs=int(args.max_signs),
        gap_min=int(args.gap_min),
        gap_max=int(args.gap_max),
        leading_noev_prob=float(args.leading_noev_prob),
        leading_noev_min=int(args.leading_noev_min),
        leading_noev_max=int(args.leading_noev_max),
        all_noev_prob=float(args.all_noev_prob),
        all_noev_min_len=int(args.all_noev_min_len),
        all_noev_max_len=int(args.all_noev_max_len),
        single_hand_dropout_prob=float(args.single_hand_dropout_prob),
        single_hand_dropout_span_min=int(args.single_hand_dropout_span_min),
        single_hand_dropout_span_max=int(args.single_hand_dropout_span_max),
        both_hands_dropout_prob=float(args.both_hands_dropout_prob),
        both_hands_dropout_span_min=int(args.both_hands_dropout_span_min),
        both_hands_dropout_span_max=int(args.both_hands_dropout_span_max),
        mask_flicker_prob=float(args.mask_flicker_prob),
        mask_flicker_span_min=int(args.mask_flicker_span_min),
        mask_flicker_span_max=int(args.mask_flicker_span_max),
        joint_jitter_prob=float(args.joint_jitter_prob),
        joint_jitter_std=float(args.joint_jitter_std),
        sign_ctx_pre=int(args.sign_ctx_pre),
        sign_ctx_post=int(args.sign_ctx_post),
        include_sign_tails_as_noev=bool(args.include_sign_tails_as_noev),
        min_tail_len=int(args.min_tail_len),
        primary_noev_prob=float(args.primary_noev_prob),
        source_sampling=str(args.source_sampling),
        sign_sampling=str(args.sign_sampling),
        stitch_noev_chunks=bool(args.stitch_noev_chunks),
        blend_prob=float(args.blend_prob),
        blend_k_min=int(args.blend_k_min),
        blend_k_max=int(args.blend_k_max),
        blend_only_when_gap_leq=int(args.blend_only_when_gap_leq),
        align_chunks=bool(args.align_chunks),
        transition_all_boundaries=bool(args.transition_all_boundaries),
        transition_k_min=int(args.transition_k_min),
        transition_k_max=int(args.transition_k_max),
        same_source_sequence_prob=float(args.same_source_sequence_prob),
        cross_source_boundary_prob=float(args.cross_source_boundary_prob),
        dataset_profile=str(args.dataset_profile),
        continuous_mode_weight=float(args.continuous_mode_weight),
        hard_negative_mode_weight=float(args.hard_negative_mode_weight),
        stress_mode_weight=float(args.stress_mode_weight),
        dense_signer_min_clips=int(args.dense_signer_min_clips),
        continuous_stats_dir=str(args.continuous_stats_dir or ""),
        sampling_profile=str(args.sampling_profile),
        preprocessing_version=str(args.preprocessing_version),
        preprocessing_center_alpha=float(args.preprocessing_center_alpha),
        preprocessing_scale_alpha=float(args.preprocessing_scale_alpha),
        preprocessing_min_scale=float(args.preprocessing_min_scale),
        preprocessing_min_visible_joints_for_scale=int(args.preprocessing_min_visible_joints_for_scale),
        crop_mode=args.crop_mode,
        pad_mode=args.pad_mode,
        npz_cache_items=int(args.npz_cache_items),
    )

    resolved_workers, auto_info = _resolve_synth_workers(
        prelabel_dir=prelabel_dir,
        cfg=cfg,
        preferred_dirs=preferred_dirs,
        extra_dirs=extra_dirs,
        requested_workers=max(1, int(args.workers)),
        auto_workers=bool(args.auto_workers),
        auto_workers_max=int(args.auto_workers_max),
        auto_workers_rebench=bool(args.auto_workers_rebench),
        auto_workers_probe_samples=int(args.auto_workers_probe_samples),
        shard_size=int(args.shard_size),
        seed=int(args.seed),
        min_sign_len=int(args.min_sign_len),
    )
    if bool(args.auto_workers):
        if bool(auto_info.get("from_cache", False)):
            print(
                f"[auto-workers:synth] cache hit selected=w{int(resolved_workers)} "
                f"sps={float(auto_info.get('benchmark_samples_per_sec', 0.0)):.1f}",
                flush=True,
            )
        else:
            rows = []
            for row in list(auto_info.get("candidates", []) or []):
                if bool(row.get("success")):
                    rows.append(f"w{int(row.get('workers', 0))}:{float(row.get('samples_per_sec', 0.0)):.1f}")
                else:
                    rows.append(f"w{int(row.get('workers', 0))}:ERR")
            print(
                f"[auto-workers:synth] benchmarked selected=w{int(resolved_workers)} "
                f"sps={float(auto_info.get('benchmark_samples_per_sec', 0.0)):.1f} "
                f"({str(auto_info.get('selection_reason', ''))})",
                flush=True,
            )
            if rows:
                print(f"  candidates {', '.join(rows)}", flush=True)

    stats = build_offline(
        prelabel_dir=prelabel_dir,
        out_dir=out_dir,
        cfg=cfg,
        num_samples=int(args.num_samples),
        shard_size=int(args.shard_size),
        seed=int(args.seed),
        min_sign_len=int(args.min_sign_len),
        workers=int(resolved_workers),
        preferred_noev_prelabel_dirs=preferred_dirs,
        extra_noev_prelabel_dirs=extra_dirs,
        overlap_report_summary=_load_json_if_exists(Path(args.overlap_report)) if args.overlap_report else None,
        resume=bool(args.resume),
    )
    acceptance = dict(stats.generated.get("acceptance", {}) or {})
    if acceptance:
        status = "passed" if bool(acceptance.get("passed", False)) else "failed"
        print(f"[synth-build] realism gate {status}", flush=True)
        for line in list(acceptance.get("failures", []) or []):
            print(f"  - {line}", flush=True)
    print(json.dumps(asdict(stats), ensure_ascii=True, indent=2))
    if bool(args.fail_on_realism_gate) and acceptance and not bool(acceptance.get("passed", False)):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
