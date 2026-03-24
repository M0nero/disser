from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


AUTO_WORKERS_VERSION = 1


@dataclass(frozen=True)
class LoaderProfile:
    workers: int
    persistent_workers: bool
    prefetch_factor: int | None
    file_cache_size: int
    pin_memory: bool
    pin_memory_device: str | None
    use_prefetch_loader: bool
    notes: tuple[str, ...] = field(default_factory=tuple)


def resolve_loader_profile(
    *,
    requested_workers: int,
    requested_prefetch: int,
    requested_file_cache_size: int,
    os_name: str,
    is_dir_mode: bool,
    cache_mode: str,
    device_type: str,
    no_prefetch: bool,
    cuda_index: int | None = None,
) -> LoaderProfile:
    workers = max(0, int(requested_workers))
    requested_file_cache_size = max(0, int(requested_file_cache_size))
    notes: list[str] = []

    if (os_name == "nt") and (not is_dir_mode) and workers > 0:
        notes.append("Windows + combined JSON detected -> forcing workers=0 to avoid RAM blowups.")
        workers = 0

    persistent_workers = workers > 0
    prefetch_factor = (max(2, int(requested_prefetch)) if workers > 0 else None)
    file_cache_size = requested_file_cache_size

    if (os_name == "nt") and is_dir_mode and workers > 0:
        if cache_mode == "decoded":
            if not persistent_workers:
                persistent_workers = True
            if prefetch_factor is not None and prefetch_factor > 2:
                notes.append(f"Windows + decoded skeleton cache -> capping prefetch_factor {prefetch_factor} -> 2.")
                prefetch_factor = 2
            if file_cache_size != 0:
                notes.append(f"Windows + decoded skeleton cache -> disabling per-worker file_cache {file_cache_size} -> 0.")
                file_cache_size = 0
            notes.append("Windows + decoded skeleton cache -> keeping persistent_workers=True for high-worker throughput.")
        elif cache_mode == "packed":
            if not persistent_workers:
                persistent_workers = True
            if prefetch_factor is not None and prefetch_factor > 2:
                notes.append(f"Windows + packed skeleton cache -> capping prefetch_factor {prefetch_factor} -> 2.")
                prefetch_factor = 2
            if file_cache_size != 0:
                notes.append(f"Windows + packed skeleton cache -> disabling per-worker file_cache {file_cache_size} -> 0.")
                file_cache_size = 0
            notes.append("Windows + packed skeleton cache -> keeping persistent_workers=True for high-worker throughput.")
        else:
            if workers > 8:
                if not persistent_workers:
                    persistent_workers = True
                if prefetch_factor is not None and prefetch_factor != 1:
                    notes.append(f"Windows + per-video JSON + high workers -> capping prefetch_factor {prefetch_factor} -> 1.")
                    prefetch_factor = 1
                if file_cache_size != 0:
                    notes.append(f"Windows + per-video JSON + high workers -> disabling per-worker file_cache {file_cache_size} -> 0.")
                    file_cache_size = 0
                notes.append("Windows + per-video JSON + high workers -> keeping persistent_workers=True.")
            else:
                if persistent_workers:
                    notes.append("Windows + per-video JSON -> disabling persistent_workers for loader stability.")
                    persistent_workers = False
                if prefetch_factor is not None and prefetch_factor > 2:
                    notes.append(f"Windows + per-video JSON -> capping prefetch_factor {prefetch_factor} -> 2.")
                    prefetch_factor = 2
                safe_file_cache = min(file_cache_size, 8)
                if safe_file_cache != file_cache_size:
                    notes.append(f"Windows + per-video JSON -> capping per-worker file_cache {file_cache_size} -> {safe_file_cache}.")
                    file_cache_size = safe_file_cache

    pin_memory = (device_type == "cuda")
    pin_memory_device = (f"cuda:{int(cuda_index)}" if pin_memory and cuda_index is not None else None)
    use_prefetch_loader = bool(device_type == "cuda" and not no_prefetch)

    return LoaderProfile(
        workers=int(workers),
        persistent_workers=bool(persistent_workers and workers > 0),
        prefetch_factor=(int(prefetch_factor) if (prefetch_factor is not None and workers > 0) else None),
        file_cache_size=int(file_cache_size),
        pin_memory=bool(pin_memory),
        pin_memory_device=pin_memory_device,
        use_prefetch_loader=bool(use_prefetch_loader),
        notes=tuple(notes),
    )


def build_worker_candidates(
    *,
    auto_workers_max: int,
    cpu_count: int,
    os_name: str,
    is_dir_mode: bool,
    cache_mode: str,
) -> list[int]:
    cpu_count = max(1, int(cpu_count))
    manual_max = max(0, int(auto_workers_max))
    if manual_max > 0:
        upper = manual_max
    elif (os_name == "nt") and (not is_dir_mode):
        upper = 0
    elif cache_mode in {"decoded", "packed"}:
        upper = min(cpu_count, 32)
    elif is_dir_mode:
        upper = min(cpu_count, 24)
    else:
        upper = min(cpu_count, 16)

    if upper <= 0:
        return [0]
    ladder = [0, 1, 2, 4, 8, 12, 16, 24, 32]
    candidates = sorted({int(x) for x in ladder if int(x) <= upper} | {int(upper)})
    return candidates


def choose_best_candidate(results: list[dict[str, Any]], *, tolerance: float = 0.05) -> dict[str, Any] | None:
    successes = [row for row in results if bool(row.get("success"))]
    if not successes:
        return None
    best = max(successes, key=lambda row: float(row.get("samples_per_sec", 0.0)))
    best_sps = float(best.get("samples_per_sec", 0.0))
    floor = best_sps * (1.0 - float(tolerance))
    close = [row for row in successes if float(row.get("samples_per_sec", 0.0)) >= floor]
    close.sort(key=lambda row: (int(row.get("workers", 0)), -float(row.get("samples_per_sec", 0.0))))
    return close[0] if close else best


def build_auto_workers_fingerprint(
    *,
    args,
    json_path: Path,
    dataset_mode: str,
    cache_mode: str,
    train_size: int,
    sampler_mode: str,
) -> tuple[dict[str, Any], str]:
    gpu_name = ""
    gpu_vram_gb = 0.0
    if torch.cuda.is_available():
        gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
        gpu_vram_gb = float(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024**3))

    payload = {
        "version": int(AUTO_WORKERS_VERSION),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "os_name": os.name,
            "cpu_count": int(os.cpu_count() or 1),
        },
        "cuda": {
            "available": bool(torch.cuda.is_available()),
            "gpu_name": gpu_name,
            "gpu_vram_gb": round(gpu_vram_gb, 3),
        },
        "data": {
            "json_path": str(json_path.expanduser().resolve()),
            "dataset_mode": str(dataset_mode),
            "cache_mode": str(cache_mode),
            "train_size": int(train_size),
            "max_frames": int(args.max_frames),
            "streams": str(args.streams),
            "include_pose": bool(args.include_pose),
            "augment": bool(args.augment),
            "prefer_pp": bool(args.prefer_pp),
        },
        "loader": {
            "batch": int(args.batch),
            "requested_workers": int(args.workers),
            "requested_prefetch": int(args.prefetch),
            "requested_file_cache": int(args.file_cache),
            "no_prefetch": bool(args.no_prefetch),
            "auto_workers_max": int(args.auto_workers_max),
            "warmup_batches": int(args.auto_workers_warmup_batches),
            "measure_batches": int(args.auto_workers_measure_batches),
            "sampler_mode": str(sampler_mode),
            "weighted_sampler": bool(args.weighted_sampler),
            "class_balanced_batch": bool(args.supcon_class_balanced_batch),
            "mixed_batch": bool(getattr(args, "supcon_mixed_batch", False)),
            "supcon_classes_per_batch": int(args.supcon_classes_per_batch),
            "supcon_samples_per_class": int(args.supcon_samples_per_class),
            "supcon_mixed_repeated_classes": int(getattr(args, "supcon_mixed_repeated_classes", 0)),
            "supcon_mixed_repeated_samples": int(getattr(args, "supcon_mixed_repeated_samples", 0)),
        },
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return payload, hashlib.sha1(blob).hexdigest()


def get_auto_workers_cache_path() -> Path:
    if platform.system() == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "msagcn" / "auto_workers_v1.json"
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "msagcn" / "auto_workers_v1.json"


def load_auto_workers_cache() -> dict[str, Any]:
    path = get_auto_workers_cache_path()
    if not path.exists():
        return {"version": int(AUTO_WORKERS_VERSION), "records": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": int(AUTO_WORKERS_VERSION), "records": {}}
    if not isinstance(payload, dict):
        return {"version": int(AUTO_WORKERS_VERSION), "records": {}}
    if int(payload.get("version", -1)) != int(AUTO_WORKERS_VERSION):
        return {"version": int(AUTO_WORKERS_VERSION), "records": {}}
    records = payload.get("records")
    if not isinstance(records, dict):
        return {"version": int(AUTO_WORKERS_VERSION), "records": {}}
    return {"version": int(AUTO_WORKERS_VERSION), "records": records}


def load_auto_workers_record(fingerprint_hash: str) -> dict[str, Any] | None:
    cache = load_auto_workers_cache()
    record = cache.get("records", {}).get(str(fingerprint_hash))
    return record if isinstance(record, dict) else None


def save_auto_workers_record(fingerprint_hash: str, record: dict[str, Any]) -> Path:
    path = get_auto_workers_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_auto_workers_cache()
    cache.setdefault("records", {})[str(fingerprint_hash)] = record
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def profile_to_dict(profile: LoaderProfile) -> dict[str, Any]:
    return asdict(profile)
