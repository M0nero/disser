from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from sklearn.metrics import confusion_matrix, f1_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bio.core.config_utils import load_config_section, write_dataset_manifest, write_run_config
from bio.core.datasets.shard_dataset import (
    ShardedBiosDataset,
    make_shard_aware_boundary_batch_sampler,
)
from bio.core.model import BioModelConfig, BioTagger
from utils.tensorboard_logger import TensorboardLogger


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _as_bool(raw: object, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        val = raw.strip().lower()
        if val in ("true", "1", "yes", "y"):
            return True
        if val in ("false", "0", "no", "n"):
            return False
    return default


def _is_missing(raw: object) -> bool:
    if raw is None:
        return True
    if isinstance(raw, str) and not raw.strip():
        return True
    return False


def _fmt_float(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{100.0 * float(value):.{digits}f}%"
    except Exception:
        return str(value)


def _human_loader_desc(payload: Dict[str, Any], key: str) -> str:
    raw = payload.get(key, {}) or {}
    if not isinstance(raw, dict) or not raw:
        return "n/a"
    workers = int(raw.get("workers", 0))
    parts = [f"w{workers}"]
    if bool(raw.get("use_prefetch_loader", False)):
        parts.append("prefetch")
    if bool(raw.get("pin_memory", False)):
        parts.append("pin")
    return ",".join(parts)


def _format_console_payload(payload: Dict[str, Any]) -> Optional[str]:
    event = str(payload.get("event", "") or "")
    if not event:
        return None
    if event == "startup_phase":
        phase = str(payload.get("phase", "") or "")
        status = str(payload.get("status", "") or "")
        if status == "started":
            return f"[startup] {phase} started"
        if status == "completed":
            return f"[startup] {phase} completed in {_fmt_float(payload.get('elapsed_sec', 0.0), 2)}s"
        if status == "failed":
            return f"[startup] {phase} failed after {_fmt_float(payload.get('elapsed_sec', 0.0), 2)}s: {payload.get('error', '')}"
    if event == "loader_profile":
        role = str(payload.get("dataset_role", "loader"))
        selected = int(payload.get("selected_workers", 0))
        from_cache = bool(payload.get("from_cache", False))
        auto_enabled = bool(payload.get("auto_workers_enabled", False))
        reason = str(payload.get("selection_reason", "") or "")
        if auto_enabled:
            header = (
                f"[auto-workers:{role}] "
                f"{'cache hit' if from_cache else 'benchmarked'} "
                f"selected=w{selected} "
                f"sps={_fmt_float(payload.get('selected_samples_per_sec', 0.0), 1)}"
            )
        else:
            header = f"[loader:{role}] manual profile selected=w{selected}"
        if reason:
            header += f" ({reason})"
        candidates = payload.get("candidates", [])
        if auto_enabled and isinstance(candidates, list) and candidates:
            rows = []
            for row in candidates:
                if not isinstance(row, dict):
                    continue
                if bool(row.get("success", False)):
                    rows.append(f"w{int(row.get('workers', 0))}:{_fmt_float(row.get('samples_per_sec', 0.0), 1)}")
                else:
                    rows.append(f"w{int(row.get('workers', 0))}:ERR")
            if rows:
                header += "\n  candidates " + ", ".join(rows)
        return header
    if event == "start":
        return (
            f"start device={payload.get('device')} amp={payload.get('amp_dtype')} "
            f"train={int(payload.get('train_samples', 0))} val={int(payload.get('val_samples', 0))} "
            f"batch={int(payload.get('batch_size', 0))} params={int(payload.get('num_params', 0))} "
            f"train_loader={_human_loader_desc(payload, 'train_loader_profile')} "
            f"val_loader={_human_loader_desc(payload, 'val_loader_profile')} "
            f"pred_artifacts_every={int(payload.get('prediction_artifacts_every', 0))}"
            "\n  note: throughput matters more than raw GPU % for this small model."
        )
    if event == "epoch_start":
        return (
            f"epoch {int(payload.get('epoch', 0))}/{int(payload.get('epochs', 0))} start "
            f"lr={_fmt_float(payload.get('lr', 0.0), 6)} "
            f"steps={int(payload.get('steps_in_epoch', 0))} "
            f"first_log=batch1 then every {int(payload.get('log_every', 0))}"
        )
    if event == "train_step":
        step_in_epoch = payload.get("step_in_epoch")
        steps_in_epoch = payload.get("steps_in_epoch")
        step_part = f"batch={int(step_in_epoch)}/{int(steps_in_epoch)} " if step_in_epoch is not None and steps_in_epoch is not None else ""
        out = (
            f"train e{int(payload.get('epoch', 0))} {step_part}"
            f"step={int(payload.get('step', 0))} "
            f"loss={_fmt_float(payload.get('loss', 0.0), 4)} "
            f"acc={_fmt_float(payload.get('acc', 0.0), 4)} "
            f"b_f1={_fmt_float(payload.get('b_f1_tol', 0.0), 4)} "
            f"lr={_fmt_float(payload.get('lr', 0.0), 6)} "
            f"sps={_fmt_float(payload.get('samples_per_sec', 0.0), 2)}"
        )
        if "mem_reserved_mb" in payload:
            out += f" mem={_fmt_float(payload.get('mem_reserved_mb', 0.0), 0)}MB"
        return out
    if event == "train_epoch":
        out = (
            f"train e{int(payload.get('epoch', 0))} done "
            f"loss={_fmt_float(payload.get('avg_loss', 0.0), 4)} "
            f"sps={_fmt_float(payload.get('samples_per_sec', 0.0), 2)} "
            f"data={_fmt_pct(payload.get('data_time_share', 0.0))} "
            f"compute={_fmt_pct(payload.get('compute_time_share', 0.0))} "
            f"batch_ms={_fmt_float(payload.get('avg_batch_time_ms', 0.0), 1)}"
        )
        if "mem_reserved_mb" in payload:
            out += f" mem={_fmt_float(payload.get('mem_reserved_mb', 0.0), 0)}MB"
        hint = str(payload.get("throughput_hint", "") or "")
        if hint:
            out += f"\n  hint: {hint}"
        return out
    if event == "val_epoch":
        return (
            f"val e{int(payload.get('epoch', 0))} "
            f"loss={_fmt_float(payload.get('loss', 0.0), 4)} "
            f"b_f1={_fmt_float(payload.get('b_f1_tol', 0.0), 4)} "
            f"balanced={_fmt_float(payload.get('balanced_score', 0.0), 4)} "
            f"thr_b={_fmt_float(payload.get('selected_boundary_threshold', 0.5), 2)}"
            f"({payload.get('selected_boundary_source', 'argmax')}) "
            f"thr_bal={_fmt_float(payload.get('selected_balanced_threshold', 0.5), 2)}"
            f"({payload.get('selected_balanced_source', 'argmax')})"
        )
    if event == "best_boundary":
        return (
            f"best_boundary updated e{int(payload.get('epoch', 0))} "
            f"thr={_fmt_float(payload.get('threshold', 0.5), 2)} "
            f"b_f1={_fmt_float(payload.get('b_f1_tol', 0.0), 4)} "
            f"b_err={_fmt_float(payload.get('b_err_mean', 0.0), 3)} "
            f"pred_B_ratio={_fmt_float(payload.get('pred_B_ratio', 0.0), 3)}"
        )
    if event == "best_balanced":
        return (
            f"best_balanced updated e{int(payload.get('epoch', 0))} "
            f"thr={_fmt_float(payload.get('threshold', 0.5), 2)} "
            f"score={_fmt_float(payload.get('balanced_score', 0.0), 4)} "
            f"b_f1={_fmt_float(payload.get('b_f1_tol', 0.0), 4)}"
        )
    if event == "early_stop":
        return (
            f"early_stop e{int(payload.get('epoch', 0))} "
            f"patience={int(payload.get('patience', 0))} "
            f"epochs_no_improve={int(payload.get('epochs_no_improve', 0))}"
        )
    if event == "done":
        return f"done step={int(payload.get('step', 0))}"
    return None


class JsonlLogger:
    def __init__(self, path: Optional[Path], *, console_format: str = "text") -> None:
        self.path = path
        self.console_format = str(console_format or "text").strip().lower()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def log(self, payload: Dict[str, object]) -> None:
        payload = dict(payload)
        payload["ts"] = time.time()
        line = json.dumps(payload, ensure_ascii=True)
        console_line = line
        if self.console_format != "json":
            console_line = _format_console_payload(payload) or line
        print(console_line, flush=True)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


@dataclass(frozen=True)
class LoaderProfile:
    workers: int
    persistent_workers: bool
    prefetch_factor: int | None
    pin_memory: bool
    use_prefetch_loader: bool
    notes: Tuple[str, ...] = field(default_factory=tuple)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


class ModelEma:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999, device: torch.device | None = None) -> None:
        self.decay = float(decay)
        self.module = copy.deepcopy(_unwrap_model(model)).eval()
        if device is not None:
            self.module.to(device)
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = _unwrap_model(model).state_dict()
        for key, value in ema_state.items():
            src = model_state[key]
            if not value.dtype.is_floating_point:
                value.copy_(src)
                continue
            value.mul_(self.decay).add_(src, alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.module.load_state_dict(state_dict)


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _optimizer_to_device(optim: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optim.state.values():
        if not isinstance(state, dict):
            continue
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def select_hist_params(
    model: torch.nn.Module,
    preferred: Tuple[str, ...] = ("embed.", "head."),
    max_items: int = 2,
):
    selected = []
    selected_names = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(pat in name for pat in preferred):
            selected.append((name, param))
            selected_names.add(name)
        if len(selected) >= max_items:
            return selected
    if len(selected) < max_items:
        for name, param in model.named_parameters():
            if not param.requires_grad or name in selected_names:
                continue
            selected.append((name, param))
            if len(selected) >= max_items:
                break
    return selected


def collate_drop_meta(batch):
    if not batch:
        return {}
    if isinstance(batch[0], dict):
        batch = [{k: v for k, v in item.items() if k != "meta"} for item in batch]
    return default_collate(batch)


def collate_keep_meta(batch):
    if not batch:
        return {}
    metas = [item.get("meta") for item in batch]
    stripped = [{k: v for k, v in item.items() if k != "meta"} for item in batch]
    collated = default_collate(stripped)
    collated["meta"] = metas
    return collated


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device, *, channels_last: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensor = value.to(device, non_blocking=True)
            if channels_last and tensor.ndim == 4 and device.type == "cuda":
                tensor = tensor.contiguous(memory_format=torch.channels_last)
            out[key] = tensor
        else:
            out[key] = value
    return out


class PrefetchLoader:
    def __init__(self, loader: DataLoader, device: torch.device, *, channels_last: bool = False) -> None:
        self.loader = loader
        self.device = device
        self.channels_last = bool(channels_last)
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None
        self.moves_to_device = True

    def __iter__(self):
        self.iter = iter(self.loader)
        if self.stream is not None:
            self._preload()
        return self

    def __next__(self):
        if self.stream is None:
            batch = next(self.iter)
            return _move_batch_to_device(batch, self.device, channels_last=self.channels_last)
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                value.record_stream(torch.cuda.current_stream())
        self._preload()
        return batch

    def _preload(self) -> None:
        try:
            batch = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = _move_batch_to_device(batch, self.device, channels_last=self.channels_last)

    def __len__(self) -> int:
        return len(self.loader)


def _batch_prepare_fn(loader: Any, device: torch.device, *, channels_last: bool = False):
    if bool(getattr(loader, "moves_to_device", False)):
        return lambda batch: batch
    return lambda batch: _move_batch_to_device(batch, device, channels_last=channels_last)


@contextmanager
def _startup_phase(logger: JsonlLogger, phase_times: Dict[str, float], phase: str):
    logger.log({"event": "startup_phase", "phase": str(phase), "status": "started"})
    started = time.perf_counter()
    try:
        yield
    except Exception as exc:
        elapsed = float(time.perf_counter() - started)
        phase_times[str(phase)] = elapsed
        logger.log(
            {
                "event": "startup_phase",
                "phase": str(phase),
                "status": "failed",
                "elapsed_sec": elapsed,
                "error": str(exc),
            }
        )
        raise
    elapsed = float(time.perf_counter() - started)
    phase_times[str(phase)] = elapsed
    logger.log({"event": "startup_phase", "phase": str(phase), "status": "completed", "elapsed_sec": elapsed})


def _auto_workers_cache_path() -> Path:
    if platform.system() == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "bio" / "auto_workers_v2.json"
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "bio" / "auto_workers_v2.json"


def _load_auto_workers_cache() -> Dict[str, Any]:
    path = _auto_workers_cache_path()
    if not path.exists():
        return {"version": 2, "records": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 2, "records": {}}
    if not isinstance(payload, dict):
        return {"version": 2, "records": {}}
    records = payload.get("records")
    if not isinstance(records, dict):
        records = {}
    return {"version": 2, "records": records}


def _save_auto_workers_record(fingerprint_hash: str, record: Dict[str, Any]) -> Path:
    path = _auto_workers_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _load_auto_workers_cache()
    payload.setdefault("records", {})[str(fingerprint_hash)] = record
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _build_auto_workers_fingerprint(
    *,
    args: argparse.Namespace,
    dataset_dir: Path,
    train_size: int,
    device: torch.device,
    requested_workers: int,
    dataset_role: str,
    collate_name: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    model_signature: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], str]:
    payload = {
        "version": 2,
        "platform": {
            "system": platform.system(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cpu_count": int(os.cpu_count() or 1),
        },
        "data": {
            "dataset_dir": str(dataset_dir.resolve()),
            "train_size": int(train_size),
        },
        "loader": {
            "batch_size": int(args.batch_size),
            "prefetch": int(args.prefetch),
            "device": str(device.type),
            "channels_last": bool(args.channels_last),
            "use_prefetch_loader": bool(args.use_prefetch_loader),
            "requested_workers": int(requested_workers),
            "dataset_role": str(dataset_role),
            "collate_name": str(collate_name),
            "use_amp": bool(use_amp),
            "amp_dtype": str(amp_dtype).replace("torch.", ""),
            "p_with_b": float(getattr(args, "p_with_b", 0.0)),
            "train_shard_cache_items": int(getattr(args, "train_shard_cache_items", 0)),
            "val_shard_cache_items": int(getattr(args, "val_shard_cache_items", 0)),
        },
        "model": dict(model_signature or {}),
    }
    blob = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return payload, hashlib.sha1(blob).hexdigest()


def _candidate_workers(auto_workers_max: int) -> List[int]:
    upper = max(0, int(auto_workers_max))
    if upper <= 0:
        upper = min(int(os.cpu_count() or 1), 16)
    ladder = [0, 1, 2, 4, 8, 12, 16, upper]
    return sorted({max(0, int(x)) for x in ladder if int(x) <= upper} | {0, upper})


def _measure_loader_runtime_sps(
    dataset: ShardedBiosDataset,
    *,
    args: argparse.Namespace,
    batch_size: int,
    workers: int,
    prefetch: int,
    pin_memory: bool,
    collate_fn,
    warmup_batches: int,
    measure_batches: int,
    model: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    channels_last: bool,
    use_prefetch_loader: bool,
    dataset_role: str,
) -> Dict[str, Any]:
    profile = LoaderProfile(
        workers=int(workers),
        persistent_workers=bool(int(workers) > 0),
        prefetch_factor=(max(2, int(prefetch)) if int(workers) > 0 else None),
        pin_memory=bool(pin_memory),
        use_prefetch_loader=bool(device.type == "cuda" and use_prefetch_loader),
    )
    wrapped_loader = _build_loader_for_role(
        dataset=dataset,
        dataset_role=dataset_role,
        args=args,
        profile=profile,
        collate_fn=collate_fn,
        device=device,
        channels_last=channels_last,
    )
    loader = wrapped_loader.loader if isinstance(wrapped_loader, PrefetchLoader) else wrapped_loader
    prepare_batch = _batch_prepare_fn(wrapped_loader, device, channels_last=channels_last)
    total_batches = max(1, int(warmup_batches) + int(measure_batches))
    seen = 0
    measured_batches = 0
    start = None
    iter_loader = iter(wrapped_loader)
    raw_model = _unwrap_model(model)
    was_training = bool(raw_model.training)
    raw_model.eval()
    try:
        with torch.no_grad():
            if int(warmup_batches) <= 0:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
            for batch_idx in range(1, total_batches + 1):
                try:
                    batch = prepare_batch(next(iter_loader))
                except StopIteration:
                    break
                with amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    logits, _ = model(batch["pts"], batch["mask"])
                    _ = logits[..., :1]
                batch_size_seen = int(batch["bio"].shape[0]) if isinstance(batch, dict) and "bio" in batch else 0
                if batch_idx == int(warmup_batches):
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    seen = 0
                    measured_batches = 0
                    continue
                if batch_idx > int(warmup_batches):
                    measured_batches += 1
                    seen += batch_size_seen
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    finally:
        raw_model.train(was_training)
        del wrapped_loader
        del loader
    elapsed = max(1e-6, (time.perf_counter() - start) if start is not None else 1e-6)
    return {
        "workers": int(workers),
        "measured_batches": int(measured_batches),
        "samples_per_sec": float(seen / elapsed),
        "success": bool(measured_batches > 0),
        "benchmark_kind": f"forward_only_runtime_{dataset_role}",
    }


def _resolve_auto_workers(
    *,
    args: argparse.Namespace,
    dataset: ShardedBiosDataset,
    dataset_dir: Path,
    device: torch.device,
    model: torch.nn.Module,
    use_amp: bool,
    amp_dtype: torch.dtype,
    collate_fn=collate_drop_meta,
    requested_workers: Optional[int] = None,
    dataset_role: str = "train",
    model_signature: Optional[Dict[str, Any]] = None,
) -> Tuple[LoaderProfile, Dict[str, Any]]:
    requested_workers = max(0, int(args.num_workers if requested_workers is None else requested_workers))
    requested_prefetch = max(2, int(args.prefetch))
    pin_memory = bool(device.type == "cuda")
    default_profile = LoaderProfile(
        workers=requested_workers,
        persistent_workers=bool(requested_workers > 0),
        prefetch_factor=(requested_prefetch if requested_workers > 0 else None),
        pin_memory=pin_memory,
        use_prefetch_loader=bool(device.type == "cuda" and args.use_prefetch_loader),
        notes=tuple(),
    )
    info: Dict[str, Any] = {
        "enabled": bool(args.auto_workers),
        "from_cache": False,
        "requested_workers": int(requested_workers),
        "requested_prefetch": int(requested_prefetch),
        "dataset_role": str(dataset_role),
    }
    if not bool(args.auto_workers):
        return default_profile, info

    fingerprint_payload, fingerprint_hash = _build_auto_workers_fingerprint(
        args=args,
        dataset_dir=dataset_dir,
        train_size=len(dataset),
        device=device,
        requested_workers=requested_workers,
        dataset_role=dataset_role,
        collate_name=getattr(collate_fn, "__name__", collate_fn.__class__.__name__),
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        model_signature=model_signature,
    )
    info["fingerprint_hash"] = str(fingerprint_hash)
    cached = None if bool(args.auto_workers_rebench) else _load_auto_workers_cache().get("records", {}).get(str(fingerprint_hash))
    if isinstance(cached, dict) and "profile" in cached:
        profile_raw = dict(cached.get("profile", {}))
        profile = LoaderProfile(
            workers=int(profile_raw.get("workers", requested_workers)),
            persistent_workers=bool(profile_raw.get("persistent_workers", int(profile_raw.get("workers", 0)) > 0)),
            prefetch_factor=(
                int(profile_raw["prefetch_factor"]) if profile_raw.get("prefetch_factor") is not None else None
            ),
            pin_memory=bool(profile_raw.get("pin_memory", pin_memory)),
            use_prefetch_loader=bool(profile_raw.get("use_prefetch_loader", device.type == "cuda" and args.use_prefetch_loader)),
            notes=tuple(str(x) for x in profile_raw.get("notes", []) if str(x)),
        )
        info.update({"from_cache": True, "cache_path": str(_auto_workers_cache_path()), "fingerprint": fingerprint_payload})
        info["candidates"] = cached.get("results", [])
        info["selection_reason"] = str(cached.get("selection_reason", "cache hit"))
        info["benchmark_samples_per_sec"] = float(cached.get("selected_samples_per_sec", 0.0))
        return profile, info

    results: List[Dict[str, Any]] = []
    for workers in _candidate_workers(int(args.auto_workers_max)):
        try:
            row = _measure_loader_runtime_sps(
                dataset,
                args=args,
                batch_size=int(args.batch_size),
                workers=int(workers),
                prefetch=int(args.prefetch),
                pin_memory=pin_memory,
                collate_fn=collate_fn,
                warmup_batches=int(args.auto_workers_warmup_batches),
                measure_batches=int(args.auto_workers_measure_batches),
                model=model,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                channels_last=bool(args.channels_last),
                use_prefetch_loader=bool(args.use_prefetch_loader),
                dataset_role=str(dataset_role),
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
        picked = {"workers": requested_workers, "samples_per_sec": 0.0, "success": False}
        selection_reason = "all auto-workers candidates failed; fell back to requested_workers"
    effective_workers = int(picked.get("workers", requested_workers))
    profile = LoaderProfile(
        workers=effective_workers,
        persistent_workers=bool(effective_workers > 0),
        prefetch_factor=(requested_prefetch if effective_workers > 0 else None),
        pin_memory=pin_memory,
        use_prefetch_loader=bool(device.type == "cuda" and args.use_prefetch_loader),
        notes=tuple([f"auto_workers_selected={effective_workers}"]),
    )
    record = {
        "profile": asdict(profile),
        "results": results,
        "fingerprint": fingerprint_payload,
        "selection_reason": str(selection_reason),
        "selected_samples_per_sec": float(picked.get("samples_per_sec", 0.0)),
    }
    cache_path = _save_auto_workers_record(fingerprint_hash, record)
    info.update(
        {
            "from_cache": False,
            "cache_path": str(cache_path),
            "candidates": results,
            "benchmark_samples_per_sec": float(picked.get("samples_per_sec", 0.0)),
            "fingerprint": fingerprint_payload,
            "selection_reason": str(selection_reason),
        }
    )
    return profile, info


def _loader_kwargs_from_profile(
    *,
    batch_size: int,
    profile: LoaderProfile,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "batch_size": int(batch_size),
        "num_workers": int(profile.workers),
        "pin_memory": bool(profile.pin_memory),
    }
    if int(profile.workers) > 0:
        kwargs["persistent_workers"] = bool(profile.persistent_workers)
        if profile.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(profile.prefetch_factor)
    return kwargs


def _build_train_batch_sampler(dataset: ShardedBiosDataset, args: argparse.Namespace) -> Any:
    return make_shard_aware_boundary_batch_sampler(
        dataset,
        batch_size=int(args.batch_size),
        p_with_b=float(args.p_with_b),
        drop_last=True,
        replacement=True,
        seed=int(args.seed),
    )


def _build_loader_for_role(
    *,
    dataset: ShardedBiosDataset,
    dataset_role: str,
    args: argparse.Namespace,
    profile: LoaderProfile,
    collate_fn,
    device: torch.device,
    channels_last: bool,
    batch_sampler: Any = None,
) -> Any:
    if dataset_role == "train":
        if batch_sampler is None:
            batch_sampler = _build_train_batch_sampler(dataset, args)
        kwargs: Dict[str, Any] = {
            "batch_sampler": batch_sampler,
            "collate_fn": collate_fn,
            "num_workers": int(profile.workers),
            "pin_memory": bool(profile.pin_memory),
        }
        if int(profile.workers) > 0:
            kwargs["persistent_workers"] = bool(profile.persistent_workers)
            if profile.prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(profile.prefetch_factor)
        loader = DataLoader(dataset, **kwargs)
        return _maybe_wrap_prefetch_loader(loader, profile=profile, device=device, channels_last=channels_last)
    loader = DataLoader(
        dataset,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        **_loader_kwargs_from_profile(batch_size=int(args.batch_size), profile=profile),
    )
    return _maybe_wrap_prefetch_loader(loader, profile=profile, device=device, channels_last=channels_last)


def _maybe_wrap_prefetch_loader(
    loader: DataLoader,
    *,
    profile: LoaderProfile,
    device: torch.device,
    channels_last: bool,
):
    if bool(profile.use_prefetch_loader) and device.type == "cuda":
        return PrefetchLoader(loader, device=device, channels_last=channels_last)
    return loader


def _compile_model_best_effort(model: torch.nn.Module, *, enabled: bool) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": False,
        "backend": "",
        "error": "",
    }
    if not bool(enabled):
        return model, info
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        info["error"] = "torch.compile unavailable"
        return model, info
    try:
        compiled = compile_fn(model)
    except Exception as exc:
        info["error"] = str(exc)
        return model, info
    info["enabled"] = True
    info["backend"] = "torch.compile"
    return compiled, info


def _loader_profile_event(dataset_role: str, profile: LoaderProfile, info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event": "loader_profile",
        "dataset_role": str(dataset_role),
        "auto_workers_enabled": bool(info.get("enabled", False)),
        "selected_workers": int(profile.workers),
        "selected_samples_per_sec": float(info.get("benchmark_samples_per_sec", 0.0)),
        "from_cache": bool(info.get("from_cache", False)),
        "selection_reason": str(info.get("selection_reason", "") or ""),
        "cache_path": str(info.get("cache_path", "") or ""),
        "fingerprint_hash": str(info.get("fingerprint_hash", "") or ""),
        "candidates": list(info.get("candidates", []) or []),
        "profile": asdict(profile),
    }


def _make_runtime_summary(
    *,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    compile_info: Dict[str, Any],
    train_profile: LoaderProfile,
    train_loader_info: Dict[str, Any],
    train_loader: Any,
    val_profile: Optional[LoaderProfile],
    val_loader_info: Dict[str, Any],
    val_loader: Any,
    train_dataset_signature: Dict[str, Any],
    val_dataset_signature: Dict[str, Any],
    score_weights: Dict[str, float],
    schedule_state: Dict[str, Any],
    startup_phase_times: Dict[str, float],
    startup_total_sec: float,
    meta_parsing_train: bool,
    meta_parsing_val: bool,
    train_shard_cache_items: int,
    val_shard_cache_items: int,
) -> Dict[str, Any]:
    return {
        "device": str(device),
        "amp": bool(use_amp),
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "tf32": bool(args.tf32),
        "channels_last": bool(args.channels_last),
        "compile": dict(compile_info),
        "train_loader_profile": asdict(train_profile),
        "train_loader_info": dict(train_loader_info),
        "val_loader_profile": asdict(val_profile) if val_profile is not None else {},
        "val_loader_info": dict(val_loader_info),
        "train_prefetch_wrapper": bool(isinstance(train_loader, PrefetchLoader)),
        "val_prefetch_wrapper": bool(isinstance(val_loader, PrefetchLoader)) if val_loader is not None else False,
        "train_dataset_signature": dict(train_dataset_signature),
        "val_dataset_signature": dict(val_dataset_signature),
        "score_weights": dict(score_weights),
        "schedule_state": dict(schedule_state),
        "startup_phase_times": {str(k): float(v) for k, v in startup_phase_times.items()},
        "startup_total_sec": float(startup_total_sec),
        "auto_workers_candidates": {
            "train": list(train_loader_info.get("candidates", []) or []),
            "val": list(val_loader_info.get("candidates", []) or []),
        },
        "auto_workers_selection_reason": {
            "train": str(train_loader_info.get("selection_reason", "") or ""),
            "val": str(val_loader_info.get("selection_reason", "") or ""),
        },
        "meta_parsing_train": bool(meta_parsing_train),
        "meta_parsing_val": bool(meta_parsing_val),
        "train_shard_cache_items": int(train_shard_cache_items),
        "val_shard_cache_items": int(val_shard_cache_items),
        "train_batching_mode": "shard_aware_boundary_batch_sampler",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cpu_count": int(os.cpu_count() or 1),
        },
    }


def _model_only_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "epoch": int(payload.get("epoch", 0)),
        "global_step": int(payload.get("global_step", 0)),
        "model_state": payload.get("model_state"),
        "ema_state": payload.get("ema_state"),
        "cfg": dict(payload.get("cfg", {}) or {}),
        "args": dict(payload.get("args", {}) or {}),
        "last_metrics": dict(payload.get("last_metrics", {}) or {}),
        "best_boundary_metrics": dict(payload.get("best_boundary_metrics", {}) or {}),
        "best_balanced_metrics": dict(payload.get("best_balanced_metrics", {}) or {}),
        "train_dataset_signature": dict(payload.get("train_dataset_signature", {}) or {}),
        "val_dataset_signature": dict(payload.get("val_dataset_signature", {}) or {}),
        "runtime_summary": dict(payload.get("runtime_summary", {}) or {}),
        "schedule_state": dict(payload.get("schedule_state", {}) or {}),
    }


def _save_model_only_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    torch.save(_model_only_payload(payload), path)


def compute_boundary_f1_tolerant(y_true: torch.Tensor, y_pred: torch.Tensor, tol: int = 2) -> Dict[str, float]:
    """
    Tolerant boundary F1 for 'B' events.
    Matches predicted B positions to true B positions within +/- tol frames.

    y_true, y_pred: (B,T) int labels with B=1.
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    tp = 0
    fp = 0
    fn = 0

    for t, p in zip(y_true, y_pred):
        gt = np.where(t == 1)[0].tolist()
        pr = np.where(p == 1)[0].tolist()

        used = set()
        for pi in pr:
            # find closest unmatched gt within tol
            best = None
            best_d = None
            for gi in gt:
                if gi in used:
                    continue
                d = abs(int(pi) - int(gi))
                if d <= tol and (best_d is None or d < best_d):
                    best_d = d
                    best = gi
            if best is not None:
                tp += 1
                used.add(best)
            else:
                fp += 1
        fn += (len(gt) - len(used))

    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {"b_prec_tol": float(prec), "b_rec_tol": float(rec), "b_f1_tol": float(f1)}


def _boundary_counts_tolerant_np(y_true: np.ndarray, y_pred: np.ndarray, tol: int = 2) -> Dict[str, int]:
    tp = 0
    fp = 0
    fn = 0
    for t, p in zip(y_true, y_pred):
        gt = np.where(np.asarray(t) == 1)[0].tolist()
        pr = np.where(np.asarray(p) == 1)[0].tolist()
        used = set()
        for pi in pr:
            best = None
            best_d = None
            for gi in gt:
                if gi in used:
                    continue
                d = abs(int(pi) - int(gi))
                if d <= tol and (best_d is None or d < best_d):
                    best_d = d
                    best = gi
            if best is not None:
                tp += 1
                used.add(best)
            else:
                fp += 1
        fn += (len(gt) - len(used))
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn)}


def _boundary_error_sums_np(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err_sum = 0.0
    matched = 0
    missing = 0
    for t, p in zip(y_true, y_pred):
        gt = np.where(np.asarray(t) == 1)[0]
        pr = np.where(np.asarray(p) == 1)[0]
        if gt.size == 0:
            continue
        if pr.size == 0:
            missing += int(gt.size)
            continue
        for gi in gt:
            err_sum += float(np.min(np.abs(pr - gi)))
            matched += 1
    return {
        "err_sum": float(err_sum),
        "matched": int(matched),
        "missing": int(missing),
    }


def frame_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    """
    Framewise accuracy + per-class F1 for labels 0/1/2.
    """
    y_true = y_true.detach().cpu().numpy().reshape(-1)
    y_pred = y_pred.detach().cpu().numpy().reshape(-1)

    acc = float((y_true == y_pred).mean())

    out = {"acc": acc}
    for cls, name in [(0, "O"), (1, "B"), (2, "I")]:
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        out[f"p_{name}"] = float(prec)
        out[f"r_{name}"] = float(rec)
        out[f"f1_{name}"] = float(f1)
    return out


def _frame_metrics_from_confusion(cm: np.ndarray) -> Dict[str, float]:
    cm = np.asarray(cm, dtype=np.int64).reshape(3, 3)
    total = int(cm.sum())
    acc = float(np.trace(cm) / max(1, total))
    out = {"acc": acc}
    for cls, name in [(0, "O"), (1, "B"), (2, "I")]:
        tp = int(cm[cls, cls])
        fp = int(cm[:, cls].sum() - tp)
        fn = int(cm[cls, :].sum() - tp)
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        out[f"p_{name}"] = float(prec)
        out[f"r_{name}"] = float(rec)
        out[f"f1_{name}"] = float(f1)
    return out


def boundary_error_mean(y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float, int, int]:
    """
    Mean absolute distance (in frames) from each true B to nearest predicted B.
    Returns (mean_error, matched_count, missing_count).
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    errors: List[int] = []
    matched = 0
    missing = 0
    for t, p in zip(y_true, y_pred):
        gt = np.where(t == 1)[0]
        pr = np.where(p == 1)[0]
        if len(gt) == 0:
            continue
        if len(pr) == 0:
            missing += len(gt)
            continue
        for gi in gt:
            errors.append(int(np.min(np.abs(pr - gi))))
            matched += 1
    mean_err = float(np.mean(errors)) if errors else 0.0
    return mean_err, matched, missing


def parse_bio_segments_strict(seq: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(seq, dtype=np.uint8).reshape(-1)
    lengths: List[int] = []
    in_seg = False
    cur = 0
    invalid_i = 0
    for raw in arr:
        v = int(raw)
        if v == 1:  # B
            if in_seg and cur > 0:
                lengths.append(cur)
            in_seg = True
            cur = 1
        elif v == 2:  # I
            if in_seg:
                cur += 1
            else:
                invalid_i += 1
        else:  # O
            if in_seg and cur > 0:
                lengths.append(cur)
            in_seg = False
            cur = 0
    if in_seg and cur > 0:
        lengths.append(cur)
    total = int(arr.size)
    violation_rate = float(invalid_i / total) if total > 0 else 0.0
    return {
        "lengths": lengths,
        "invalid_i_count": int(invalid_i),
        "violation_count": int(invalid_i),
        "violation_rate": float(violation_rate),
    }


def avg_segment_length(seq: np.ndarray) -> List[int]:
    return list(parse_bio_segments_strict(seq).get("lengths", []))


def transition_rate(seq: np.ndarray) -> float:
    if seq.size < 2:
        return 0.0
    return float((seq[1:] != seq[:-1]).sum() / max(1, (seq.size - 1)))


def build_confusion_image(cm: np.ndarray, labels: List[str]) -> np.ndarray:
    cm = cm.astype(np.float32)
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    cm = cm / row_sum

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3, 3), dpi=140)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    ax.set_title("Confusion")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.canvas.draw()
    canvas = fig.canvas
    if hasattr(canvas, "buffer_rgba"):
        rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        img = np.ascontiguousarray(rgba[..., :3])
    else:
        w, h = canvas.get_width_height()
        if hasattr(canvas, "tostring_rgb"):
            img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        elif hasattr(canvas, "tostring_argb"):
            argb = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
            img = np.ascontiguousarray(argb[..., 1:4])
        else:
            plt.close(fig)
            raise RuntimeError(f"Unsupported matplotlib canvas type: {type(canvas).__name__}")
    plt.close(fig)
    return img


def _bio_seq_to_str(seq: np.ndarray, max_len: int = 120) -> str:
    mapping = {0: "O", 1: "B", 2: "I"}
    s = "".join(mapping.get(int(x), "?") for x in seq.tolist())
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"...(len={len(s)})"


def _boundary_error_sample(y_seq: np.ndarray, p_seq: np.ndarray) -> float:
    gt = np.where(y_seq == 1)[0]
    pr = np.where(p_seq == 1)[0]
    if gt.size == 0:
        return 0.0
    if pr.size == 0:
        return float("inf")
    return float(np.mean([np.min(np.abs(pr - gi)) for gi in gt]))


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    if conf.size == 0:
        return 0.0
    conf = np.clip(conf.astype(np.float32, copy=False), 0.0, 1.0)
    correct = correct.astype(np.float32, copy=False)
    edges = np.linspace(0.0, 1.0, int(max(2, bins)) + 1, dtype=np.float32)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi >= 1.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        acc = float(correct[mask].mean())
        avg_conf = float(conf[mask].mean())
        ece += abs(acc - avg_conf) * float(mask.mean())
    return float(ece)


def binary_calibration_metrics(prob: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    if prob.size == 0:
        return {"ece": 0.0, "brier": 0.0}
    p = np.clip(prob.astype(np.float32, copy=False), 0.0, 1.0)
    y = target.astype(np.float32, copy=False)
    return {
        "ece": expected_calibration_error(p, y),
        "brier": float(np.mean((p - y) ** 2)),
    }


def balanced_model_score(
    metrics: Dict[str, float],
    *,
    lambda_len: float = 0.10,
    lambda_trans: float = 0.75,
    lambda_berr: float = 0.05,
    lambda_bio_violation: float = 1.0,
) -> float:
    b_f1 = float(metrics.get("b_f1_tol", 0.0) or 0.0)
    seg_ratio = float(metrics.get("avg_seg_len_ratio", segment_length_ratio(metrics)) or 0.0)
    if seg_ratio <= 0.0:
        len_penalty = 1.5
    else:
        len_penalty = float(abs(np.log(seg_ratio)))
    trans_penalty = float(metrics.get("transition_rate_abs_err", 0.0) or 0.0)
    avg_true_seg = max(1.0, float(metrics.get("avg_seg_len_true", 1.0) or 1.0))
    b_err_penalty = float(metrics.get("b_err_mean", 0.0) or 0.0) / avg_true_seg
    bio_violation_penalty = float(metrics.get("bio_violation_abs_err", metrics.get("bio_violation_rate_pred", 0.0)) or 0.0)
    score = (
        b_f1
        - float(lambda_len) * len_penalty
        - float(lambda_trans) * trans_penalty
        - float(lambda_berr) * b_err_penalty
        - float(lambda_bio_violation) * bio_violation_penalty
    )
    return float(score)


def _score_weights_from_args(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "lambda_len": float(getattr(args, "balanced_lambda_len", 0.10)),
        "lambda_trans": float(getattr(args, "balanced_lambda_trans", 0.75)),
        "lambda_berr": float(getattr(args, "balanced_lambda_berr", 0.05)),
        "lambda_bio_violation": float(getattr(args, "balanced_lambda_bio_violation", 1.0)),
    }


def _selection_candidate(
    base_metrics: Dict[str, float],
    selected_row: Optional[Dict[str, Any]],
    *,
    selection_source: str,
) -> Dict[str, float]:
    merged = {k: float(v) for k, v in base_metrics.items()}
    if selected_row:
        for key, value in selected_row.items():
            if key == "threshold":
                continue
            merged[str(key)] = float(value)
        merged["selection_threshold"] = float(selected_row.get("threshold", 0.5))
    else:
        merged["selection_threshold"] = 0.5
    merged["selection_source"] = 1.0 if selection_source == "threshold_sweep" else 0.0
    merged["argmax_b_f1_tol"] = float(base_metrics.get("b_f1_tol", 0.0))
    merged["argmax_balanced_score"] = float(base_metrics.get("balanced_score", 0.0))
    return merged


def save_checkpoint(
    out_dir: Path,
    epoch: int,
    last_payload: Dict[str, Any],
    save_every_epochs: int,
    *,
    periodic_payload: Optional[Dict[str, Any]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(last_payload, out_dir / "last.pt")
    _save_model_only_checkpoint(out_dir / "last_model.pt", last_payload)
    if int(epoch) % max(1, int(save_every_epochs)) == 0:
        torch.save(periodic_payload if periodic_payload is not None else last_payload, out_dir / f"epoch_{int(epoch):04d}.pt")


def _structural_bio_stats(y_seq: np.ndarray, p_seq: np.ndarray) -> Dict[str, float]:
    true_segs: List[int] = []
    pred_segs: List[int] = []
    trans_true: List[float] = []
    trans_pred: List[float] = []
    true_violations = 0
    pred_violations = 0
    true_frames = 0
    pred_frames = 0
    for true_item, pred_item in zip(y_seq, p_seq):
        true_stats = parse_bio_segments_strict(true_item)
        pred_stats = parse_bio_segments_strict(pred_item)
        true_segs.extend(true_stats["lengths"])
        pred_segs.extend(pred_stats["lengths"])
        trans_true.append(transition_rate(true_item))
        trans_pred.append(transition_rate(pred_item))
        true_violations += int(true_stats["violation_count"])
        pred_violations += int(pred_stats["violation_count"])
        true_frames += int(np.asarray(true_item).size)
        pred_frames += int(np.asarray(pred_item).size)
    avg_true_seg = float(np.mean(true_segs)) if true_segs else 0.0
    avg_pred_seg = float(np.mean(pred_segs)) if pred_segs else 0.0
    tr_rate_true = float(np.mean(trans_true)) if trans_true else 0.0
    tr_rate_pred = float(np.mean(trans_pred)) if trans_pred else 0.0
    violation_rate_true = float(true_violations / max(1, true_frames))
    violation_rate_pred = float(pred_violations / max(1, pred_frames))
    return {
        "avg_seg_len_true": avg_true_seg,
        "avg_seg_len_pred": avg_pred_seg,
        "avg_seg_len_ratio": float(segment_length_ratio({"avg_seg_len_true": avg_true_seg, "avg_seg_len_pred": avg_pred_seg})),
        "transition_rate_true": tr_rate_true,
        "transition_rate_pred": tr_rate_pred,
        "transition_rate_abs_err": float(abs(tr_rate_pred - tr_rate_true)),
        "segment_length_abs_err": float(abs(avg_pred_seg - avg_true_seg)),
        "bio_violation_count_true": float(true_violations),
        "bio_violation_count_pred": float(pred_violations),
        "bio_violation_rate_true": violation_rate_true,
        "bio_violation_rate_pred": violation_rate_pred,
        "bio_violation_abs_err": float(abs(violation_rate_pred - violation_rate_true)),
    }


def _compute_warmup_epochs(epochs: int, warmup_frac: float) -> int:
    warmup_epochs = int(max(0, round(float(warmup_frac) * int(epochs))))
    if warmup_epochs >= int(epochs):
        warmup_epochs = max(0, int(epochs) - 1)
    return int(warmup_epochs)


def _make_schedule_state(base_lr: float, epochs: int, warmup_frac: float) -> Dict[str, Any]:
    return {
        "type": "warmup_cosine_epoch",
        "base_lr": float(base_lr),
        "epochs": int(epochs),
        "warmup_frac": float(warmup_frac),
        "warmup_epochs": int(_compute_warmup_epochs(int(epochs), float(warmup_frac))),
        "completed_epochs": 0,
    }


def _lr_for_epoch(schedule_state: Dict[str, Any], epoch: int) -> float:
    total_epochs = max(1, int(schedule_state.get("epochs", 1)))
    base_lr = float(schedule_state.get("base_lr", 0.0))
    warmup_epochs = int(schedule_state.get("warmup_epochs", 0))
    epoch_idx = max(1, int(epoch))
    if warmup_epochs > 0 and epoch_idx <= warmup_epochs:
        factor = 0.1 + 0.9 * float(epoch_idx - 1) / float(max(1, warmup_epochs))
        return float(base_lr * factor)
    main_epochs = max(1, total_epochs - warmup_epochs)
    main_step = min(max(0, epoch_idx - warmup_epochs - 1), main_epochs)
    cosine_factor = 0.5 * (1.0 + float(np.cos(np.pi * float(main_step) / float(main_epochs))))
    return float(base_lr * cosine_factor)


def _set_optimizer_lr(optim: torch.optim.Optimizer, lr: float) -> None:
    lr_val = float(lr)
    for group in optim.param_groups:
        group["lr"] = lr_val


def _checkpoint_schedule_state(schedule_state: Dict[str, Any], *, completed_epochs: int) -> Dict[str, Any]:
    out = dict(schedule_state)
    out["completed_epochs"] = int(completed_epochs)
    return out


def _schedule_matches(current: Dict[str, Any], saved: Dict[str, Any]) -> bool:
    keys = ("type", "base_lr", "epochs", "warmup_frac", "warmup_epochs")
    for key in keys:
        current_value = current.get(key)
        saved_value = saved.get(key)
        if isinstance(current_value, float) or isinstance(saved_value, float):
            if not np.isclose(float(current_value), float(saved_value), atol=1e-12, rtol=0.0):
                return False
        else:
            if current_value != saved_value:
                return False
    return True


def train_one_epoch(
    model: BioTagger,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
    class_weights: torch.Tensor,
    grad_clip: float,
    log_every: int,
    step0: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    epoch: int,
    logger: JsonlLogger,
    tb_logger: TensorboardLogger | None,
    tb_log_every: int,
    *,
    label_smoothing: float,
    ema: Optional[ModelEma],
    channels_last: bool,
) -> Tuple[int, Dict[str, float]]:
    model.train()
    step = step0
    prepare_batch = _batch_prepare_fn(loader, device, channels_last=channels_last)
    steps_in_epoch = len(loader)
    t_load_sum = 0.0
    t_fwd_sum = 0.0
    steps = 0
    total_loss = 0.0
    total_samples = 0
    nonfinite_batches = 0
    skipped_updates = 0
    grad_norm_sum = 0.0
    grad_norm_count = 0
    epoch_start = time.time()
    last_log_time = epoch_start
    last_log_samples = 0
    tb_enabled = tb_logger is not None and tb_logger.enabled and tb_log_every > 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    loader_iter = iter(loader)
    i = 0
    while True:
        t_fetch = time.time()
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        t_load_sum += (time.time() - t_fetch)

        t0 = time.time()
        batch = prepare_batch(batch)
        pts = batch["pts"]
        mask = batch["mask"]
        bio = batch["bio"]
        t_load_sum += (time.time() - t0)

        t1 = time.time()
        with amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits, _ = model(pts, mask)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            loss = F.cross_entropy(
                logits.reshape(-1, 3),
                bio.reshape(-1),
                weight=class_weights,
                label_smoothing=float(max(0.0, label_smoothing)),
            )

        optim.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            print("WARN: non-finite loss encountered → batch skipped")
            nonfinite_batches += 1
            continue

        if tb_enabled and ((step + 1) % tb_log_every == 0):
            lr = float(optim.param_groups[0]["lr"]) if optim.param_groups else 0.0
            tb_logger.scalar("train/loss", float(loss.item()), step + 1)
            tb_logger.scalar("train/lr", lr, step + 1)
            tb_logger.scalar("train/amp_scale", float(scaler.get_scale()), step + 1)
        scaler.scale(loss).backward()
        grad_norm = None
        if grad_clip > 0:
            scaler.unscale_(optim)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scale_before = float(scaler.get_scale())
        scaler.step(optim)
        scaler.update()
        scale_after = float(scaler.get_scale())
        update_skipped = bool(use_amp and scale_after < scale_before)
        if update_skipped:
            skipped_updates += 1
        elif ema is not None:
            ema.update(model)
        t_fwd_sum += (time.time() - t1)

        bs = int(bio.size(0))
        total_samples += bs
        total_loss += float(loss.item()) * bs
        if grad_norm is not None and torch.isfinite(torch.as_tensor(grad_norm)):
            grad_norm_sum += float(grad_norm)
            grad_norm_count += 1

        if i == 0 or (step + 1) % log_every == 0:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                m1 = frame_metrics(bio, pred)
                m2 = compute_boundary_f1_tolerant(bio, pred, tol=2)
                f1_macro = (m1["f1_O"] + m1["f1_B"] + m1["f1_I"]) / 3.0
            now = time.time()
            elapsed = max(1e-6, now - last_log_time)
            sps = (total_samples - last_log_samples) / elapsed
            lr = float(optim.param_groups[0]["lr"]) if optim.param_groups else 0.0
            if tb_enabled:
                tb_logger.scalar("train/acc", float(m1["acc"]), step + 1)
                tb_logger.scalar("train/f1_macro", float(f1_macro), step + 1)
                tb_logger.scalar("train/b_f1_tol", float(m2["b_f1_tol"]), step + 1)
                tb_logger.scalar("train/b_prec_tol", float(m2["b_prec_tol"]), step + 1)
                tb_logger.scalar("train/b_rec_tol", float(m2["b_rec_tol"]), step + 1)
                tb_logger.scalar("train/p_O", float(m1["p_O"]), step + 1)
                tb_logger.scalar("train/r_O", float(m1["r_O"]), step + 1)
                tb_logger.scalar("train/f1_O", float(m1["f1_O"]), step + 1)
                tb_logger.scalar("train/p_B", float(m1["p_B"]), step + 1)
                tb_logger.scalar("train/r_B", float(m1["r_B"]), step + 1)
                tb_logger.scalar("train/f1_B", float(m1["f1_B"]), step + 1)
                tb_logger.scalar("train/p_I", float(m1["p_I"]), step + 1)
                tb_logger.scalar("train/r_I", float(m1["r_I"]), step + 1)
                tb_logger.scalar("train/f1_I", float(m1["f1_I"]), step + 1)
                tb_logger.scalar("train/samples_per_sec", float(sps), step + 1)
                tb_logger.scalar("train/skipped_updates", float(skipped_updates), step + 1)
                tb_logger.scalar("train/nonfinite_batches", float(nonfinite_batches), step + 1)
                if grad_norm is not None:
                    tb_logger.scalar("train/grad_norm", float(grad_norm), step + 1)
                if device.type == "cuda":
                    tb_logger.scalar(
                        "train/mem_alloc_mb",
                        float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
                        step + 1,
                    )
            payload = {
                "event": "train_step",
                "epoch": epoch,
                "step": step + 1,
                "step_in_epoch": i + 1,
                "steps_in_epoch": int(steps_in_epoch),
                "loss": float(loss.item()),
                "acc": float(m1["acc"]),
                "f1_macro": float(f1_macro),
                "b_f1_tol": float(m2["b_f1_tol"]),
                "lr": lr,
                "samples_per_sec": float(sps),
                "amp_scale": float(scaler.get_scale()),
                "skipped_updates": int(skipped_updates),
                "nonfinite_batches": int(nonfinite_batches),
            }
            if grad_norm is not None:
                payload["grad_norm"] = float(grad_norm)
            if device.type == "cuda":
                payload["mem_alloc_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
                payload["mem_reserved_mb"] = float(torch.cuda.max_memory_reserved() / (1024 ** 2))
            logger.log(payload)
            last_log_time = now
            last_log_samples = total_samples

        step += 1
        steps += 1
        i += 1

    epoch_metrics: Dict[str, float] = {
        "epoch": float(epoch),
        "step": float(step),
        "avg_loss": float(total_loss / max(1, total_samples)) if total_samples > 0 else 0.0,
        "samples": float(total_samples),
        "steps": float(steps),
        "dataloader_time_sec": float(t_load_sum),
        "fwd_bwd_time_sec": float(t_fwd_sum),
        "nonfinite_batches": float(nonfinite_batches),
        "skipped_updates": float(skipped_updates),
        "lr": float(optim.param_groups[0]["lr"]) if optim.param_groups else 0.0,
        "amp_scale": float(scaler.get_scale()),
    }
    if steps > 0:
        epoch_time = max(1e-6, time.time() - epoch_start)
        avg_loss = total_loss / max(1, total_samples)
        lr = float(optim.param_groups[0]["lr"]) if optim.param_groups else 0.0
        avg_batch_time_ms = float((epoch_time / max(1, steps)) * 1000.0)
        data_time_share = float(min(1.0, t_load_sum / epoch_time))
        compute_time_share = float(min(1.0, t_fwd_sum / epoch_time))
        payload = {
            "event": "train_epoch",
            "epoch": epoch,
            "step": step,
            "avg_loss": float(avg_loss),
            "samples": int(total_samples),
            "steps": int(steps),
            "epoch_time_sec": float(epoch_time),
            "samples_per_sec": float(total_samples / epoch_time),
            "dataloader_time_sec": float(t_load_sum),
            "fwd_bwd_time_sec": float(t_fwd_sum),
            "data_time_share": float(data_time_share),
            "compute_time_share": float(compute_time_share),
            "avg_batch_time_ms": float(avg_batch_time_ms),
            "lr": lr,
            "nonfinite_batches": int(nonfinite_batches),
            "skipped_updates": int(skipped_updates),
            "amp_scale": float(scaler.get_scale()),
        }
        epoch_metrics["epoch_time_sec"] = float(epoch_time)
        epoch_metrics["samples_per_sec"] = float(total_samples / epoch_time)
        epoch_metrics["data_time_share"] = float(data_time_share)
        epoch_metrics["compute_time_share"] = float(compute_time_share)
        epoch_metrics["avg_batch_time_ms"] = float(avg_batch_time_ms)
        if grad_norm_count > 0:
            payload["grad_norm_mean"] = float(grad_norm_sum / grad_norm_count)
            epoch_metrics["grad_norm_mean"] = float(grad_norm_sum / grad_norm_count)
        if device.type == "cuda":
            payload["mem_alloc_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            payload["mem_reserved_mb"] = float(torch.cuda.max_memory_reserved() / (1024 ** 2))
            epoch_metrics["mem_alloc_mb"] = float(payload["mem_alloc_mb"])
            epoch_metrics["mem_reserved_mb"] = float(payload["mem_reserved_mb"])
            if float(payload["mem_reserved_mb"]) < 1024.0 and compute_time_share >= max(0.20, data_time_share * 1.2):
                payload["throughput_hint"] = "GPU memory headroom is high; try a larger batch_size."
            elif data_time_share >= 0.35:
                payload["throughput_hint"] = "Input pipeline share is high; rebench workers or reduce artifact cadence."
        logger.log(payload)
        if tb_enabled:
            tb_logger.scalar("train/epoch_loss", float(avg_loss), step)
            tb_logger.scalar("train/samples_per_sec", float(total_samples / epoch_time), step)
            tb_logger.scalar("train/dataloader_time_sec", float(t_load_sum), step)
            tb_logger.scalar("train/fwd_bwd_time_sec", float(t_fwd_sum), step)
            tb_logger.scalar("train/data_time_share", float(data_time_share), step)
            tb_logger.scalar("train/compute_time_share", float(compute_time_share), step)
            tb_logger.scalar("train/avg_batch_time_ms", float(avg_batch_time_ms), step)
            tb_logger.scalar("train/epoch_time_sec", float(epoch_time), step)
            tb_logger.scalar("train/nonfinite_batches", float(nonfinite_batches), step)
            tb_logger.scalar("train/skipped_updates", float(skipped_updates), step)
    return step, epoch_metrics


@torch.no_grad()
def eval_epoch(
    model: BioTagger,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    class_weights: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    collect_examples: bool = False,
    examples_k: int = 5,
    collect_predictions: bool = False,
    threshold_sweep_points: Optional[Sequence[float]] = None,
    channels_last: bool = False,
    score_weights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], np.ndarray, List[Dict[str, object]], Dict[str, Any]]:
    model.eval()
    prepare_batch = _batch_prepare_fn(loader, device, channels_last=channels_last)
    total_loss = 0.0
    total_n = 0
    ys = []
    ps = []
    metas_all: List[Any] = []
    conf_all: List[np.ndarray] = []
    correct_all: List[np.ndarray] = []
    b_prob_all: List[np.ndarray] = []
    b_target_all: List[np.ndarray] = []
    collect_threshold_inputs = bool(threshold_sweep_points)
    collect_prediction_rows = bool(collect_predictions)
    threshold_state = _init_threshold_sweep_state(threshold_sweep_points or []) if collect_threshold_inputs else None
    for batch in loader:
        batch = prepare_batch(batch)
        pts = batch["pts"]
        mask = batch["mask"]
        bio = batch["bio"]
        meta_batch = batch.get("meta", [None] * int(bio.size(0)))

        with amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits, _ = model(pts, mask)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            loss = F.cross_entropy(
                logits.reshape(-1, 3),
                bio.reshape(-1),
                weight=class_weights,
                label_smoothing=float(max(0.0, label_smoothing)),
            )
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        probs_cpu = probs.detach().float().cpu()
        bio_cpu = bio.detach().cpu()
        ys.append(bio)
        ps.append(pred)
        conf_all.append(probs_cpu.max(dim=-1).values.numpy().reshape(-1))
        correct_all.append((pred == bio).detach().cpu().numpy().reshape(-1).astype(np.float32))
        b_prob_all.append(probs_cpu[..., 1].numpy().reshape(-1))
        b_target_all.append((bio_cpu == 1).numpy().reshape(-1).astype(np.float32))
        if collect_threshold_inputs:
            _update_threshold_sweep_state(
                threshold_state or {},
                bio_cpu.numpy().astype(np.uint8, copy=False),
                probs_cpu.numpy().astype(np.float32, copy=False),
            )
        if collect_prediction_rows:
            metas_all.extend(meta_batch if isinstance(meta_batch, list) else [None] * int(bio.size(0)))
        bs = int(bio.size(0))
        total_loss += float(loss.item()) * bs
        total_n += bs

    y = torch.cat(ys, dim=0)
    p = torch.cat(ps, dim=0)

    m1 = frame_metrics(y, p)
    m2 = compute_boundary_f1_tolerant(y, p, tol=2)
    avg_loss = total_loss / max(1, total_n)

    y_np = y.detach().cpu().numpy().reshape(-1)
    p_np = p.detach().cpu().numpy().reshape(-1)
    support = np.bincount(y_np, minlength=3)
    f1_micro = float(f1_score(y_np, p_np, average="micro"))
    f1_weighted = float(f1_score(y_np, p_np, average="weighted"))
    b_err, b_matched, b_missing = boundary_error_mean(y, p)
    true_b = float((y_np == 1).sum())
    pred_b = float((p_np == 1).sum())
    ratio_b = float(pred_b / (true_b + 1e-9))

    y_seq = y.detach().cpu().numpy()
    p_seq = p.detach().cpu().numpy()
    structural = _structural_bio_stats(y_seq, p_seq)

    cm = confusion_matrix(y_np, p_np, labels=[0, 1, 2])

    metrics = {
        "loss": float(avg_loss),
        **m1,
        **m2,
        "support_O": float(support[0]),
        "support_B": float(support[1]),
        "support_I": float(support[2]),
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
        "b_err_mean": float(b_err),
        "b_err_matched": float(b_matched),
        "b_err_missing": float(b_missing),
        "pred_B_ratio": ratio_b,
        **structural,
        "balanced_score": float(balanced_model_score({
            "b_f1_tol": float(m2.get("b_f1_tol", 0.0)),
            "avg_seg_len_ratio": float(structural.get("avg_seg_len_ratio", 0.0)),
            "transition_rate_abs_err": float(structural.get("transition_rate_abs_err", 0.0)),
            "b_err_mean": float(b_err),
            "avg_seg_len_true": float(structural.get("avg_seg_len_true", 0.0)),
            "bio_violation_abs_err": float(structural.get("bio_violation_abs_err", 0.0)),
            "bio_violation_rate_pred": float(structural.get("bio_violation_rate_pred", 0.0)),
        }, **(score_weights or {}))),
    }
    top1_conf = np.concatenate(conf_all, axis=0) if conf_all else np.zeros((0,), dtype=np.float32)
    top1_correct = np.concatenate(correct_all, axis=0) if correct_all else np.zeros((0,), dtype=np.float32)
    b_prob = np.concatenate(b_prob_all, axis=0) if b_prob_all else np.zeros((0,), dtype=np.float32)
    b_target = np.concatenate(b_target_all, axis=0) if b_target_all else np.zeros((0,), dtype=np.float32)
    calib_b = binary_calibration_metrics(b_prob, b_target)
    metrics["ece_top1"] = float(expected_calibration_error(top1_conf, top1_correct))
    metrics["ece_B"] = float(calib_b["ece"])
    metrics["brier_B"] = float(calib_b["brier"])
    examples: List[Dict[str, object]] = []
    analysis: Dict[str, Any] = {}
    if collect_examples:
        per_f1 = []
        for i in range(y_seq.shape[0]):
            yi = y_seq[i]
            pi = p_seq[i]
            f1_i = float(f1_score(yi, pi, average="macro", labels=[0, 1, 2], zero_division=0))
            per_f1.append(f1_i)
            examples.append({
                "f1": f1_i,
                "b_err": _boundary_error_sample(yi, pi),
                "true_seq": yi,
                "pred_seq": pi,
            })
        # keep worst by f1
        examples.sort(key=lambda x: x.get("f1", 1.0))
        examples = examples[: max(1, int(examples_k))]
    if collect_prediction_rows:
        analysis["prediction_rows"] = _collect_sample_diagnostics(y_seq, p_seq, metas_all)
    if collect_threshold_inputs and threshold_state is not None:
        analysis["threshold_sweep"] = _finalize_threshold_sweep_state(threshold_state, score_weights=score_weights)
    return metrics, cm, examples, analysis


def segment_length_ratio(metrics: Dict[str, float]) -> float:
    true_len = float(metrics.get("avg_seg_len_true", 0.0) or 0.0)
    pred_len = float(metrics.get("avg_seg_len_pred", 0.0) or 0.0)
    if true_len <= 0.0:
        return 1.0 if pred_len <= 0.0 else float("inf")
    return float(pred_len / true_len)


def passes_balanced_guardrails(metrics: Dict[str, float]) -> bool:
    pred_b_ratio = float(metrics.get("pred_B_ratio", 0.0) or 0.0)
    seg_ratio = segment_length_ratio(metrics)
    return (0.85 <= pred_b_ratio <= 1.15) and (0.80 <= seg_ratio <= 1.25)


def _is_better_balanced(
    candidate: Dict[str, float],
    best: Optional[Dict[str, float]],
    *,
    score_weights: Optional[Dict[str, float]] = None,
) -> bool:
    if best is None:
        return True
    weights = dict(score_weights or {})
    cand_score = float(candidate.get("balanced_score", balanced_model_score(candidate, **weights)))
    best_score = float(best.get("balanced_score", balanced_model_score(best, **weights)))
    if cand_score > best_score:
        return True
    if cand_score < best_score:
        return False
    cand_f1 = float(candidate.get("b_f1_tol", -1.0))
    best_f1 = float(best.get("b_f1_tol", -1.0))
    if cand_f1 > best_f1:
        return True
    if cand_f1 < best_f1:
        return False
    return float(candidate.get("b_err_mean", float("inf"))) < float(best.get("b_err_mean", float("inf")))


def _boundary_pred_ratio_distance(metrics: Dict[str, Any]) -> float:
    return float(abs(float(metrics.get("pred_B_ratio", 0.0)) - 1.0))


def _boundary_tie_break_tuple(metrics: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    return (
        float(metrics.get("b_err_mean", float("inf"))),
        _boundary_pred_ratio_distance(metrics),
        float(metrics.get("transition_rate_abs_err", float("inf"))),
        float(metrics.get("bio_violation_abs_err", float("inf"))),
        float(metrics.get("selection_threshold", metrics.get("threshold", 0.5))),
    )


def _is_better_boundary(candidate: Dict[str, Any], best: Optional[Dict[str, Any]]) -> bool:
    if best is None:
        return True
    cand_f1 = float(candidate.get("b_f1_tol", -1.0))
    best_f1 = float(best.get("b_f1_tol", -1.0))
    if cand_f1 > best_f1:
        return True
    if cand_f1 < best_f1:
        return False
    return _boundary_tie_break_tuple(candidate) < _boundary_tie_break_tuple(best)


def _empty_threshold_accumulator(threshold: float) -> Dict[str, Any]:
    return {
        "threshold": float(threshold),
        "frame_cm": np.zeros((3, 3), dtype=np.int64),
        "boundary_tp": 0,
        "boundary_fp": 0,
        "boundary_fn": 0,
        "boundary_err_sum": 0.0,
        "boundary_err_matched": 0,
        "boundary_err_missing": 0,
        "true_b": 0,
        "pred_b": 0,
        "true_seg_sum": 0.0,
        "true_seg_count": 0,
        "pred_seg_sum": 0.0,
        "pred_seg_count": 0,
        "transition_true_sum": 0.0,
        "transition_pred_sum": 0.0,
        "sequence_count": 0,
        "true_violation_count": 0,
        "pred_violation_count": 0,
        "true_frames": 0,
        "pred_frames": 0,
        "support": np.zeros((3,), dtype=np.int64),
    }


def _init_threshold_sweep_state(thresholds: Sequence[float]) -> Dict[str, Any]:
    threshold_list = [float(x) for x in thresholds]
    return {
        "thresholds": threshold_list,
        "accumulators": [_empty_threshold_accumulator(thr) for thr in threshold_list],
    }


def _update_threshold_sweep_state(state: Dict[str, Any], y_batch: np.ndarray, probs_batch: np.ndarray) -> None:
    if not state.get("accumulators"):
        return
    y_arr = np.asarray(y_batch, dtype=np.uint8)
    probs_arr = np.asarray(probs_batch, dtype=np.float32)
    if y_arr.size == 0 or probs_arr.size == 0:
        return
    alt_pred = np.where(probs_arr[..., 0] >= probs_arr[..., 2], 0, 2).astype(np.uint8)
    flat_true = y_arr.reshape(-1)
    support = np.bincount(flat_true.astype(np.int64), minlength=3)
    true_b = int((flat_true == 1).sum())
    true_seg_sum = 0.0
    true_seg_count = 0
    transition_true_sum = 0.0
    true_violation_count = 0
    true_frames = 0
    for yi in y_arr:
        true_stats = parse_bio_segments_strict(yi)
        true_seg_sum += float(sum(int(v) for v in true_stats["lengths"]))
        true_seg_count += int(len(true_stats["lengths"]))
        transition_true_sum += float(transition_rate(yi))
        true_violation_count += int(true_stats["violation_count"])
        true_frames += int(np.asarray(yi).size)
    for acc in state.get("accumulators", []):
        thr = float(acc["threshold"])
        pred_thr = np.where(probs_arr[..., 1] >= thr, 1, alt_pred).astype(np.uint8)
        flat_pred = pred_thr.reshape(-1)
        np.add.at(acc["frame_cm"], (flat_true.astype(np.int64), flat_pred.astype(np.int64)), 1)
        boundary_counts = _boundary_counts_tolerant_np(y_arr, pred_thr, tol=2)
        boundary_errors = _boundary_error_sums_np(y_arr, pred_thr)
        acc["boundary_tp"] += int(boundary_counts["tp"])
        acc["boundary_fp"] += int(boundary_counts["fp"])
        acc["boundary_fn"] += int(boundary_counts["fn"])
        acc["boundary_err_sum"] += float(boundary_errors["err_sum"])
        acc["boundary_err_matched"] += int(boundary_errors["matched"])
        acc["boundary_err_missing"] += int(boundary_errors["missing"])
        acc["true_b"] += int(true_b)
        acc["pred_b"] += int((flat_pred == 1).sum())
        acc["true_seg_sum"] += float(true_seg_sum)
        acc["true_seg_count"] += int(true_seg_count)
        acc["transition_true_sum"] += float(transition_true_sum)
        acc["true_violation_count"] += int(true_violation_count)
        acc["true_frames"] += int(true_frames)
        acc["sequence_count"] += int(y_arr.shape[0])
        acc["support"] += support
        pred_seg_sum = 0.0
        pred_seg_count = 0
        transition_pred_sum = 0.0
        pred_violation_count = 0
        pred_frames = 0
        for pi in pred_thr:
            pred_stats = parse_bio_segments_strict(pi)
            pred_seg_sum += float(sum(int(v) for v in pred_stats["lengths"]))
            pred_seg_count += int(len(pred_stats["lengths"]))
            transition_pred_sum += float(transition_rate(pi))
            pred_violation_count += int(pred_stats["violation_count"])
            pred_frames += int(np.asarray(pi).size)
        acc["pred_seg_sum"] += float(pred_seg_sum)
        acc["pred_seg_count"] += int(pred_seg_count)
        acc["transition_pred_sum"] += float(transition_pred_sum)
        acc["pred_violation_count"] += int(pred_violation_count)
        acc["pred_frames"] += int(pred_frames)


def _finalize_threshold_metrics(acc: Dict[str, Any], *, score_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    cm = np.asarray(acc.get("frame_cm", np.zeros((3, 3), dtype=np.int64)), dtype=np.int64)
    frame_out = _frame_metrics_from_confusion(cm)
    tp = int(acc.get("boundary_tp", 0))
    fp = int(acc.get("boundary_fp", 0))
    fn = int(acc.get("boundary_fn", 0))
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    avg_true_seg = float(acc.get("true_seg_sum", 0.0)) / max(1, int(acc.get("true_seg_count", 0)))
    avg_pred_seg = float(acc.get("pred_seg_sum", 0.0)) / max(1, int(acc.get("pred_seg_count", 0)))
    transition_true = float(acc.get("transition_true_sum", 0.0)) / max(1, int(acc.get("sequence_count", 0)))
    transition_pred = float(acc.get("transition_pred_sum", 0.0)) / max(1, int(acc.get("sequence_count", 0)))
    true_frames = max(1, int(acc.get("true_frames", 0)))
    pred_frames = max(1, int(acc.get("pred_frames", 0)))
    true_b = float(acc.get("true_b", 0))
    pred_b = float(acc.get("pred_b", 0))
    metrics = {
        **frame_out,
        "b_prec_tol": float(prec),
        "b_rec_tol": float(rec),
        "b_f1_tol": float(f1),
        "support_O": float(int(acc.get("support", np.zeros((3,), dtype=np.int64))[0])),
        "support_B": float(int(acc.get("support", np.zeros((3,), dtype=np.int64))[1])),
        "support_I": float(int(acc.get("support", np.zeros((3,), dtype=np.int64))[2])),
        "b_err_mean": float(float(acc.get("boundary_err_sum", 0.0)) / max(1, int(acc.get("boundary_err_matched", 0)))),
        "b_err_matched": float(int(acc.get("boundary_err_matched", 0))),
        "b_err_missing": float(int(acc.get("boundary_err_missing", 0))),
        "pred_B_ratio": float(pred_b / (true_b + 1e-9)),
        "avg_seg_len_true": float(avg_true_seg),
        "avg_seg_len_pred": float(avg_pred_seg),
        "avg_seg_len_ratio": float(segment_length_ratio({"avg_seg_len_true": avg_true_seg, "avg_seg_len_pred": avg_pred_seg})),
        "transition_rate_true": float(transition_true),
        "transition_rate_pred": float(transition_pred),
        "transition_rate_abs_err": float(abs(transition_pred - transition_true)),
        "segment_length_abs_err": float(abs(avg_pred_seg - avg_true_seg)),
        "bio_violation_count_true": float(int(acc.get("true_violation_count", 0))),
        "bio_violation_count_pred": float(int(acc.get("pred_violation_count", 0))),
        "bio_violation_rate_true": float(int(acc.get("true_violation_count", 0)) / true_frames),
        "bio_violation_rate_pred": float(int(acc.get("pred_violation_count", 0)) / pred_frames),
    }
    metrics["bio_violation_abs_err"] = float(abs(metrics["bio_violation_rate_pred"] - metrics["bio_violation_rate_true"]))
    metrics["balanced_guardrails_passed"] = 1.0 if passes_balanced_guardrails(metrics) else 0.0
    metrics["balanced_score"] = float(balanced_model_score(metrics, **(score_weights or {})))
    return {k: float(v) for k, v in metrics.items()}


def _finalize_threshold_sweep_state(
    state: Dict[str, Any],
    *,
    score_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    accumulators = list(state.get("accumulators", []))
    if not accumulators:
        return {"thresholds": [], "best_b_f1_tol": {}, "best_balanced": {}}
    threshold_rows: List[Dict[str, Any]] = []
    best_f1: Optional[Dict[str, Any]] = None
    best_balanced: Optional[Dict[str, Any]] = None
    for acc in accumulators:
        metrics = _finalize_threshold_metrics(acc, score_weights=score_weights)
        row = {"threshold": float(acc.get("threshold", 0.5)), **metrics}
        threshold_rows.append(row)
        if _is_better_boundary(row, best_f1):
            best_f1 = dict(row)
        if row["balanced_guardrails_passed"] > 0.0:
            if best_balanced is None or _is_better_balanced(row, best_balanced, score_weights=score_weights):
                best_balanced = dict(row)
    return {
        "thresholds": threshold_rows,
        "best_b_f1_tol": best_f1 or {},
        "best_balanced": best_balanced or {},
    }


def _compress_meta(meta: Any) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in ("seq_len", "V", "source_sampling", "sign_sampling", "stitch_noev_chunks"):
        if key in meta:
            out[key] = meta.get(key)
    parts = meta.get("parts", [])
    if isinstance(parts, list):
        out["parts"] = [
            {
                "type": str(part.get("type", "")),
                "role": str(part.get("role", "")),
                "label_str": str(part.get("label_str", "")),
                "source_type": str(part.get("source_type", "")),
                "source_group": str(part.get("source_group", "")),
                "dataset": str(part.get("dataset", "")),
                "length": int(part.get("length", 0) or 0),
            }
            for part in parts
        ]
    return out


def _collect_sample_diagnostics(
    y_seq: np.ndarray,
    p_seq: np.ndarray,
    metas: Sequence[Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx in range(y_seq.shape[0]):
        yi = y_seq[idx]
        pi = p_seq[idx]
        true_stats = parse_bio_segments_strict(yi)
        pred_stats = parse_bio_segments_strict(pi)
        rows.append(
            {
                "sample_index": int(idx),
                "f1_macro": float(f1_score(yi, pi, average="macro", labels=[0, 1, 2], zero_division=0)),
                "b_err": float(_boundary_error_sample(yi, pi)),
                "true_bio_violation_count": int(true_stats["violation_count"]),
                "pred_bio_violation_count": int(pred_stats["violation_count"]),
                "true_bio_violation_rate": float(true_stats["violation_rate"]),
                "pred_bio_violation_rate": float(pred_stats["violation_rate"]),
                "bio_violation_abs_err": float(abs(float(pred_stats["violation_rate"]) - float(true_stats["violation_rate"]))),
                "true_B": np.where(yi == 1)[0].astype(int).tolist(),
                "pred_B": np.where(pi == 1)[0].astype(int).tolist(),
                "true_seq": _bio_seq_to_str(yi, max_len=max(256, yi.shape[0])),
                "pred_seq": _bio_seq_to_str(pi, max_len=max(256, pi.shape[0])),
                "meta": _compress_meta(metas[idx] if idx < len(metas) else None),
            }
        )
    return rows


def _threshold_sweep(
    y_seq: np.ndarray,
    probs: np.ndarray,
    thresholds: Sequence[float],
    *,
    score_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    if y_seq.size == 0 or probs.size == 0:
        return {"thresholds": [], "best_b_f1_tol": {}, "best_balanced": {}}
    state = _init_threshold_sweep_state(thresholds)
    _update_threshold_sweep_state(state, y_seq, probs)
    return _finalize_threshold_sweep_state(state, score_weights=score_weights)


def _build_checkpoint_payload(
    *,
    epoch: int,
    global_step: int,
    model: BioTagger,
    optim: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    ema: Optional[ModelEma],
    cfg: BioModelConfig,
    args: argparse.Namespace,
    last_metrics: Dict[str, Any],
    best_boundary_metrics: Optional[Dict[str, float]],
    best_balanced_metrics: Optional[Dict[str, float]],
    history: Sequence[Dict[str, Any]],
    epochs_no_improve: int,
    schedule_state: Dict[str, Any],
    train_dataset_signature: Dict[str, Any],
    val_dataset_signature: Dict[str, Any],
    balanced_tracking_started: bool,
    runtime_summary: Optional[Dict[str, Any]] = None,
    include_history: bool = True,
    include_runtime_summary: bool = True,
) -> Dict[str, Any]:
    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state": model.state_dict(),
        "optimizer_state": optim.state_dict(),
        "scaler_state": scaler.state_dict(),
        "ema_state": ema.state_dict() if ema is not None else None,
        "schedule_state": _checkpoint_schedule_state(schedule_state, completed_epochs=epoch),
        "cfg": asdict(cfg),
        "args": dict(vars(args)),
        "last_metrics": dict(last_metrics),
        "best_boundary_metrics": dict(best_boundary_metrics or {}),
        "best_balanced_metrics": dict(best_balanced_metrics or {}),
        "history_epoch": int(epoch),
        "epoch_summary": dict(last_metrics),
        "epochs_no_improve": int(epochs_no_improve),
        "balanced_tracking_started": bool(balanced_tracking_started),
        "train_dataset_signature": dict(train_dataset_signature or {}),
        "val_dataset_signature": dict(val_dataset_signature or {}),
    }
    if include_history:
        payload["history"] = list(history)
    if include_runtime_summary:
        payload["runtime_summary"] = dict(runtime_summary or {})
    return payload


def _write_analysis_artifacts(
    out_dir: Path,
    epoch: int,
    prediction_rows: Sequence[Dict[str, Any]],
    threshold_sweep: Optional[Dict[str, Any]],
    *,
    write_predictions: bool,
) -> None:
    analysis_dir = out_dir / "analysis"
    predictions_dir = analysis_dir / "predictions"
    errors_dir = analysis_dir / "errors"
    thresholds_dir = analysis_dir / "threshold_sweeps"
    pred_rows = list(prediction_rows)
    if write_predictions and pred_rows:
        pred_jsonl = predictions_dir / f"predictions_ep{int(epoch):03d}.jsonl"
        _write_jsonl(pred_jsonl, pred_rows)
        csv_rows: List[Dict[str, Any]] = []
        for row in pred_rows:
            csv_rows.append(
                {
                    "sample_index": int(row.get("sample_index", 0)),
                    "f1_macro": float(row.get("f1_macro", 0.0)),
                    "b_err": row.get("b_err"),
                    "true_bio_violation_count": int(row.get("true_bio_violation_count", 0)),
                    "pred_bio_violation_count": int(row.get("pred_bio_violation_count", 0)),
                    "true_bio_violation_rate": float(row.get("true_bio_violation_rate", 0.0)),
                    "pred_bio_violation_rate": float(row.get("pred_bio_violation_rate", 0.0)),
                    "bio_violation_abs_err": float(row.get("bio_violation_abs_err", 0.0)),
                    "true_B": json.dumps(row.get("true_B", []), ensure_ascii=False),
                    "pred_B": json.dumps(row.get("pred_B", []), ensure_ascii=False),
                    "true_seq": str(row.get("true_seq", "")),
                    "pred_seq": str(row.get("pred_seq", "")),
                    "meta": json.dumps(row.get("meta", {}), ensure_ascii=False),
                }
            )
        _write_csv(
            predictions_dir / f"predictions_ep{int(epoch):03d}.csv",
            csv_rows,
            [
                "sample_index",
                "f1_macro",
                "b_err",
                "true_bio_violation_count",
                "pred_bio_violation_count",
                "true_bio_violation_rate",
                "pred_bio_violation_rate",
                "bio_violation_abs_err",
                "true_B",
                "pred_B",
                "true_seq",
                "pred_seq",
                "meta",
            ],
        )
        errors = sorted(
            pred_rows,
            key=lambda item: (
                float(item.get("f1_macro", 1.0)),
                float(item.get("b_err", float("inf")) if item.get("b_err", float("inf")) != float("inf") else 1e9),
            ),
        )[:50]
        _write_json(errors_dir / f"errors_ep{int(epoch):03d}.json", errors)
    if threshold_sweep is not None:
        _write_json(thresholds_dir / f"threshold_sweep_ep{int(epoch):03d}.json", threshold_sweep)
        best_payload = {
            "epoch": int(epoch),
            "best_b_f1_tol": dict(threshold_sweep.get("best_b_f1_tol", {})),
            "best_balanced": dict(threshold_sweep.get("best_balanced", {})),
        }
        _write_json(thresholds_dir / f"best_threshold_ep{int(epoch):03d}.json", best_payload)
        _write_json(thresholds_dir / "best_threshold_latest.json", best_payload)


def _restore_history_from_checkpoint(ckpt: Dict[str, Any], resume_path: Path) -> List[Dict[str, Any]]:
    if isinstance(ckpt.get("history"), list):
        return list(ckpt.get("history", []) or [])
    epoch_limit = int(ckpt.get("epoch", 0))
    history_path = resume_path.parent / "history.json"
    if history_path.exists():
        try:
            raw = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                out: List[Dict[str, Any]] = []
                for row in raw:
                    if isinstance(row, dict) and int(row.get("epoch", 0) or 0) <= epoch_limit:
                        out.append(dict(row))
                if out:
                    return out
        except Exception:
            pass
    epoch_summary = ckpt.get("epoch_summary")
    if isinstance(epoch_summary, dict) and epoch_summary:
        return [dict(epoch_summary)]
    last_metrics = ckpt.get("last_metrics")
    if isinstance(last_metrics, dict) and last_metrics:
        return [dict(last_metrics)]
    return []


def _selection_context(selection: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not isinstance(selection, dict):
        return {}
    return {
        "selection_threshold": float(selection.get("selection_threshold", selection.get("threshold", 0.5))),
        "b_err_mean": float(selection.get("b_err_mean", 0.0)),
        "pred_B_ratio": float(selection.get("pred_B_ratio", 0.0)),
        "transition_rate_abs_err": float(selection.get("transition_rate_abs_err", 0.0)),
        "bio_violation_abs_err": float(selection.get("bio_violation_abs_err", 0.0)),
    }


def _raise_invalid_val_split_no_boundaries(val_dir: str, dataset_signature: Dict[str, Any]) -> None:
    sig = json.dumps(dataset_signature or {}, ensure_ascii=True, sort_keys=True)
    raise RuntimeError(
        "Validation split contains no true B events; boundary selection is undefined. "
        f"val_dir={val_dir!r} dataset_signature={sig}. "
        "Rebuild or fix the validation dataset for the boundary detector."
    )


def _should_write_prediction_artifacts(
    *,
    enabled: bool,
    epoch: int,
    every: int,
    total_epochs: int,
    boundary_improved: bool,
    balanced_improved: bool,
) -> bool:
    if not enabled:
        return False
    cadence_hit = (int(epoch) % max(1, int(every)) == 0)
    is_final = int(epoch) >= int(total_epochs)
    return bool(cadence_hit or boundary_improved or balanced_improved or is_final)


def _update_early_stop_counter(
    current: int,
    *,
    enabled: bool,
    epoch: int,
    armed_epoch: int,
    balanced_tracking_started: bool,
    balanced_improved: bool,
) -> int:
    if not enabled or int(epoch) < int(armed_epoch) or not balanced_tracking_started:
        return int(current)
    if balanced_improved:
        return 0
    return int(current) + 1


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, _ = pre.parse_known_args(argv)
    defaults = load_config_section(pre_args.config, "train")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=pre_args.config, help="Path to config JSON (section: train).")
    ap.add_argument(
        "--train_dir",
        type=str,
        default=defaults.get("train_dir"),
        required=_is_missing(defaults.get("train_dir")),
        help="Step2 synth train dir (has index.json and shards/)",
    )
    ap.add_argument("--val_dir", type=str, default=defaults.get("val_dir", ""), help="Step2 synth val dir (optional)")
    ap.add_argument(
        "--out_dir",
        type=str,
        default=defaults.get("out_dir"),
        required=_is_missing(defaults.get("out_dir")),
    )

    ap.add_argument("--epochs", type=int, default=int(defaults.get("epochs", 5)))
    ap.add_argument("--batch_size", type=int, default=int(defaults.get("batch_size", 64)))
    ap.add_argument("--num_workers", type=int, default=int(defaults.get("num_workers", 4)))
    ap.add_argument("--train_shard_cache_items", type=int, default=int(defaults.get("train_shard_cache_items", 8)), help="Train dataset shard cache size.")
    ap.add_argument("--val_shard_cache_items", type=int, default=int(defaults.get("val_shard_cache_items", 2)), help="Validation dataset shard cache size.")
    ap.add_argument("--prefetch", type=int, default=int(defaults.get("prefetch", 2)))
    ap.add_argument("--auto_workers", action="store_true", default=_as_bool(defaults.get("auto_workers"), False), help="Benchmark and cache an effective DataLoader worker profile.")
    ap.add_argument("--no_auto_workers", dest="auto_workers", action="store_false", help="Disable auto-worker benchmarking and use the requested num_workers profile directly.")
    ap.add_argument("--auto_workers_max", type=int, default=int(defaults.get("auto_workers_max", 0)), help="Upper bound for auto-worker benchmarking (0=heuristic default).")
    ap.add_argument("--auto_workers_rebench", action="store_true", default=_as_bool(defaults.get("auto_workers_rebench"), False), help="Ignore cached auto-worker decisions and benchmark again.")
    ap.add_argument("--auto_workers_warmup_batches", type=int, default=int(defaults.get("auto_workers_warmup_batches", 2)))
    ap.add_argument("--auto_workers_measure_batches", type=int, default=int(defaults.get("auto_workers_measure_batches", 8)))
    ap.add_argument("--use_prefetch_loader", action="store_true", default=_as_bool(defaults.get("use_prefetch_loader"), True), help="Use async GPU batch prefetch when CUDA is active.")
    ap.add_argument("--no_use_prefetch_loader", dest="use_prefetch_loader", action="store_false")
    ap.add_argument("--channels_last", action="store_true", default=_as_bool(defaults.get("channels_last"), False), help="Use channels_last memory format for 4D tensors on CUDA.")
    ap.add_argument("--compile", action="store_true", default=_as_bool(defaults.get("compile"), False), help="Best-effort torch.compile for the training/eval graph.")
    ap.add_argument("--lr", type=float, default=float(defaults.get("lr", 2e-3)))
    ap.add_argument("--warmup_frac", type=float, default=float(defaults.get("warmup_frac", 0.10)))
    ap.add_argument("--ema_decay", type=float, default=float(defaults.get("ema_decay", 0.0)))
    ap.add_argument("--label_smoothing", type=float, default=float(defaults.get("label_smoothing", 0.0)))
    ap.add_argument("--balanced_lambda_len", type=float, default=float(defaults.get("balanced_lambda_len", 0.10)))
    ap.add_argument("--balanced_lambda_trans", type=float, default=float(defaults.get("balanced_lambda_trans", 0.75)))
    ap.add_argument("--balanced_lambda_berr", type=float, default=float(defaults.get("balanced_lambda_berr", 0.05)))
    ap.add_argument("--balanced_lambda_bio_violation", type=float, default=float(defaults.get("balanced_lambda_bio_violation", 1.0)))
    ap.add_argument("--weight_decay", type=float, default=float(defaults.get("weight_decay", 1e-4)))
    ap.add_argument("--grad_clip", type=float, default=float(defaults.get("grad_clip", 1.0)))
    ap.add_argument("--no_amp", action="store_true", default=_as_bool(defaults.get("no_amp"), False), help="Disable AMP even on CUDA.")
    ap.add_argument("--tf32", action="store_true", default=_as_bool(defaults.get("tf32"), False), help="Enable TF32 on CUDA (matmul + cudnn).")
    ap.add_argument("--resume", type=str, default=defaults.get("resume", ""), help="Resume full training state from checkpoint.")
    ap.add_argument("--resume_model_only", action="store_true", default=_as_bool(defaults.get("resume_model_only"), False), help="Load model/EMA weights only and start optimizer/scheduler fresh.")
    ap.add_argument("--early_stop_patience", type=int, default=int(defaults.get("early_stop_patience", 0)), help="Epoch patience for best_balanced improvement (0 disables).")
    ap.add_argument("--early_stop_min_delta", type=float, default=float(defaults.get("early_stop_min_delta", 0.0)), help="Minimum best_balanced improvement to reset early-stop patience.")
    ap.add_argument("--early_stop_armed_epoch", type=int, default=int(defaults.get("early_stop_armed_epoch", 0)), help="Epoch after which early stopping becomes active (0=after warmup).")

    default_device = defaults.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=int(defaults.get("seed", 1337)))
    ap.add_argument("--device", type=str, default=default_device)

    # boundary-aware sampler: target fraction of samples that contain at least one B
    ap.add_argument("--p_with_b", type=float, default=float(defaults.get("p_with_b", 0.85)))

    # class weights for O,B,I (B should be big)
    ap.add_argument("--wO", type=float, default=float(defaults.get("wO", 1.5)))
    ap.add_argument("--wB", type=float, default=float(defaults.get("wB", 25.0)))
    ap.add_argument("--wI", type=float, default=float(defaults.get("wI", 1.0)))

    # model config
    ap.add_argument("--embed_dim", type=int, default=int(defaults.get("embed_dim", 128)))
    ap.add_argument("--conv_kernel", type=int, default=int(defaults.get("conv_kernel", 5)))
    ap.add_argument("--conv_layers", type=int, default=int(defaults.get("conv_layers", 2)))
    ap.add_argument("--gru_hidden", type=int, default=int(defaults.get("gru_hidden", 192)))
    ap.add_argument("--gru_layers", type=int, default=int(defaults.get("gru_layers", 1)))
    ap.add_argument("--drop_conv", type=float, default=float(defaults.get("drop_conv", 0.10)))
    ap.add_argument("--drop_head", type=float, default=float(defaults.get("drop_head", 0.10)))

    ap.add_argument("--log_every", type=int, default=int(defaults.get("log_every", 100)))
    ap.add_argument("--console_log_format", type=str, default=str(defaults.get("console_log_format", "text")), choices=["text", "json"], help="Stdout log format; train_log.jsonl remains structured JSONL.")
    ap.add_argument("--save_every_epochs", type=int, default=int(defaults.get("save_every_epochs", defaults.get("save_every", 1))), help="Write periodic epoch-boundary snapshots every N epochs.")
    ap.add_argument("--save_every", dest="save_every_legacy", type=int, default=None, help="Deprecated alias for --save_every_epochs.")
    ap.add_argument(
        "--log_jsonl",
        type=str,
        default=defaults.get("log_jsonl", ""),
        help="Write JSONL logs to this path (default: <out_dir>/train_log.jsonl).",
    )
    ap.add_argument("--tensorboard", action="store_true", default=_as_bool(defaults.get("tensorboard"), False), help="Enable TensorBoard logging.")
    ap.add_argument("--logdir", type=str, default=defaults.get("logdir", "runs"), help="TensorBoard base log dir.")
    ap.add_argument("--run_name", type=str, default=defaults.get("run_name", ""), help="TensorBoard run name (default: out dir name).")
    ap.add_argument("--flush_secs", type=int, default=int(defaults.get("flush_secs", 30)), help="TensorBoard flush_secs.")
    ap.add_argument("--log_every_steps", type=int, default=int(defaults.get("log_every_steps", 1)), help="TensorBoard step logging frequency.")
    ap.add_argument("--tb_log_examples", action="store_true", default=_as_bool(defaults.get("tb_log_examples"), False), help="Log BIO examples to TensorBoard.")
    ap.add_argument("--tb_examples_k", type=int, default=int(defaults.get("tb_examples_k", 5)), help="Number of examples to log.")
    ap.add_argument("--tb_examples_every", type=int, default=int(defaults.get("tb_examples_every", 5)), help="Log examples every N epochs.")
    ap.add_argument("--save_analysis_artifacts", action="store_true", default=_as_bool(defaults.get("save_analysis_artifacts"), False), help="Write val threshold-sweep artifacts every epoch and prediction/error artifacts on the configured cadence.")
    ap.add_argument("--no_save_analysis_artifacts", dest="save_analysis_artifacts", action="store_false")
    ap.add_argument("--prediction_artifacts_every", type=int, default=int(defaults.get("prediction_artifacts_every", 5)), help="Write full prediction/error artifacts every N epochs when analysis artifacts are enabled.")

    args = ap.parse_args(argv)
    args.save_every_deprecated_used = args.save_every_legacy is not None
    if args.save_every_deprecated_used:
        args.save_every_epochs = int(args.save_every_legacy)
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if bool(getattr(args, "save_every_deprecated_used", False)):
        warnings.warn("--save_every is deprecated; use --save_every_epochs.", DeprecationWarning, stacklevel=2)

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available → falling back to CPU")
        device = torch.device("cpu")

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.float32
    use_amp = (device.type == "cuda") and (not args.no_amp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_jsonl).resolve() if args.log_jsonl else (out_dir / "train_log.jsonl")
    logger = JsonlLogger(log_path, console_format=str(args.console_log_format))
    startup_phase_times: Dict[str, float] = {}
    startup_started = time.perf_counter()

    run_name = args.run_name.strip() or out_dir.name
    tb_logger = TensorboardLogger(
        log_dir=args.logdir,
        run_name=run_name,
        enabled=bool(args.tensorboard),
        flush_secs=int(args.flush_secs),
    )
    if tb_logger.enabled:
        print(f"TensorBoard logs: {tb_logger.log_dir}")
        print(f"Run: tensorboard --logdir {args.logdir}")

    train_ds: ShardedBiosDataset
    train_stats: Dict[str, Any]
    train_dataset_signature: Dict[str, Any]
    val_loader = None
    val_profile: Optional[LoaderProfile] = None
    val_loader_info: Dict[str, Any] = {}
    val_ds: Optional[ShardedBiosDataset] = None
    val_dataset_signature: Dict[str, Any] = {}
    train_loader = None
    train_profile: LoaderProfile
    train_loader_info: Dict[str, Any]
    train_loader_base = None
    val_loader_base = None
    cfg: BioModelConfig
    model: BioTagger
    train_model: torch.nn.Module
    compile_info: Dict[str, Any]
    hist_params = []
    num_params = 0
    train_batch_sampler = None

    with _startup_phase(logger, startup_phase_times, "dataset_scan"):
        train_ds = ShardedBiosDataset(args.train_dir, shard_cache_items=int(args.train_shard_cache_items), parse_meta=False)
        train_stats = _load_json_if_exists(Path(args.train_dir) / "stats.json")
        train_dataset_signature = train_stats.get("dataset_signature", {}) if isinstance(train_stats.get("dataset_signature", {}), dict) else {}
        if args.val_dir:
            val_ds = ShardedBiosDataset(args.val_dir, shard_cache_items=int(args.val_shard_cache_items), parse_meta=True)
            val_stats = _load_json_if_exists(Path(args.val_dir) / "stats.json")
            val_dataset_signature = val_stats.get("dataset_signature", {}) if isinstance(val_stats.get("dataset_signature", {}), dict) else {}

    step = 0
    start_epoch = 1
    best_boundary_metrics: Optional[Dict[str, float]] = None
    best_balanced_metrics: Optional[Dict[str, float]] = None
    history: List[Dict[str, Any]] = []
    epochs_no_improve = 0
    balanced_tracking_started = False
    threshold_points = [round(float(x), 2) for x in np.linspace(0.20, 0.80, 13)]
    with _startup_phase(logger, startup_phase_times, "model_init"):
        cfg = BioModelConfig(
            num_joints=42,
            in_coords=3,
            embed_dim=int(args.embed_dim),
            conv_kernel=int(args.conv_kernel),
            conv_layers=int(args.conv_layers),
            conv_dropout=float(args.drop_conv),
            gru_hidden=int(args.gru_hidden),
            gru_layers=int(args.gru_layers),
            head_dropout=float(args.drop_head),
            use_vel=True,
            use_acc=True,
            use_mask=True,
            use_aggs=True,
        )
        model = BioTagger(cfg).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        schedule_state = _make_schedule_state(float(args.lr), int(args.epochs), float(args.warmup_frac))
        warmup_epochs = int(schedule_state.get("warmup_epochs", 0))
        scaler = amp.GradScaler(enabled=use_amp)
        ema = ModelEma(model, decay=float(args.ema_decay), device=device) if float(args.ema_decay) > 0.0 else None

        if args.resume:
            resume_path = Path(args.resume).expanduser()
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            ckpt_train_sig = ckpt.get("train_dataset_signature", {})
            ckpt_val_sig = ckpt.get("val_dataset_signature", {})
            if (not args.resume_model_only) and train_dataset_signature and ckpt_train_sig and ckpt_train_sig != train_dataset_signature:
                raise ValueError("Resume checkpoint train dataset signature does not match current train_dir. Use matching data or --resume_model_only.")
            if (not args.resume_model_only) and args.val_dir and val_dataset_signature and ckpt_val_sig and ckpt_val_sig != val_dataset_signature:
                raise ValueError("Resume checkpoint val dataset signature does not match current val_dir. Use matching data or --resume_model_only.")
            ckpt_schedule_state = dict(ckpt.get("schedule_state", {}) or {})
            if args.resume_model_only:
                missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
                if missing or unexpected:
                    print(
                        "Model-only resume loaded with non-strict state_dict matching. "
                        f"missing={len(missing)} unexpected={len(unexpected)}"
                    )
                if ema is not None:
                    if ckpt.get("ema_state") is not None:
                        ema.load_state_dict(ckpt["ema_state"])
                    else:
                        ema.module.load_state_dict(model.state_dict())
                print(f"Loaded model weights from {resume_path} (model-only resume).")
            else:
                model.load_state_dict(ckpt["model_state"], strict=True)
                resumed_epoch = int(ckpt.get("epoch", 0))
                if ckpt_schedule_state and not _schedule_matches(schedule_state, ckpt_schedule_state):
                    raise ValueError(
                        "Resume checkpoint schedule_state does not match current lr/epochs/warmup_frac. "
                        "Use matching settings or --resume_model_only."
                    )
                if "optimizer_state" in ckpt:
                    optim.load_state_dict(ckpt["optimizer_state"])
                    _optimizer_to_device(optim, device)
                if not ckpt_schedule_state:
                    ckpt_schedule_state = _make_schedule_state(float(args.lr), int(args.epochs), float(args.warmup_frac))
                if "scaler_state" in ckpt:
                    try:
                        scaler.load_state_dict(ckpt["scaler_state"])
                    except Exception as exc:
                        print(f"Warning: failed to restore GradScaler state: {exc}")
                if ema is not None:
                    if ckpt.get("ema_state") is not None:
                        ema.load_state_dict(ckpt["ema_state"])
                    else:
                        ema.module.load_state_dict(model.state_dict())
                step = int(ckpt.get("global_step", 0))
                start_epoch = resumed_epoch + 1
                best_boundary_metrics = dict(ckpt.get("best_boundary_metrics", {}) or {}) or None
                best_balanced_metrics = dict(ckpt.get("best_balanced_metrics", {}) or {}) or None
                history = _restore_history_from_checkpoint(ckpt, resume_path)
                epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
                balanced_tracking_started = bool(ckpt.get("balanced_tracking_started", bool(best_balanced_metrics)))
                print(f"Resumed full training state from {resume_path} at epoch {start_epoch}.")
        schedule_state["completed_epochs"] = int(max(0, start_epoch - 1))
        if start_epoch <= int(args.epochs):
            _set_optimizer_lr(optim, _lr_for_epoch(schedule_state, start_epoch))
        train_model, compile_info = _compile_model_best_effort(model, enabled=bool(args.compile))
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        hist_params = select_hist_params(model, ("embed.", "head."))

    class_weights = torch.tensor([args.wO, args.wB, args.wI], dtype=torch.float32, device=device)
    score_weights = _score_weights_from_args(args)
    model_signature = {
        "model": "BioTagger",
        "num_params": int(num_params),
        "cfg": asdict(cfg),
    }
    early_stop_armed_epoch = int(args.early_stop_armed_epoch) if int(args.early_stop_armed_epoch) > 0 else int(warmup_epochs + 1)

    with _startup_phase(logger, startup_phase_times, "auto_workers"):
        train_batch_sampler = _build_train_batch_sampler(train_ds, args)
        train_profile, train_loader_info = _resolve_auto_workers(
            args=args,
            dataset=train_ds,
            dataset_dir=Path(args.train_dir),
            device=device,
            model=model,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            collate_fn=collate_drop_meta,
            requested_workers=int(args.num_workers),
            dataset_role="train",
            model_signature=model_signature,
        )
        train_loader = _build_loader_for_role(
            dataset=train_ds,
            dataset_role="train",
            args=args,
            profile=train_profile,
            collate_fn=collate_drop_meta,
            device=device,
            channels_last=bool(args.channels_last),
            batch_sampler=train_batch_sampler,
        )
        logger.log(_loader_profile_event("train", train_profile, train_loader_info))
        if val_ds is not None:
            val_profile, val_loader_info = _resolve_auto_workers(
                args=args,
                dataset=val_ds,
                dataset_dir=Path(args.val_dir),
                device=device,
                model=model,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                collate_fn=collate_keep_meta,
                requested_workers=max(0, int(args.num_workers) // 2),
                dataset_role="val",
                model_signature=model_signature,
            )
            val_loader = _build_loader_for_role(
                dataset=val_ds,
                dataset_role="val",
                args=args,
                profile=val_profile,
                collate_fn=collate_keep_meta,
                device=device,
                channels_last=bool(args.channels_last),
            )
            logger.log(_loader_profile_event("val", val_profile, val_loader_info))

    runtime_summary: Dict[str, Any] = {}
    with _startup_phase(logger, startup_phase_times, "runtime_summary_write"):
        runtime_summary = _make_runtime_summary(
            args=args,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            compile_info=compile_info,
            train_profile=train_profile,
            train_loader_info=train_loader_info,
            train_loader=train_loader,
            val_profile=val_profile,
            val_loader_info=val_loader_info,
            val_loader=val_loader,
            train_dataset_signature=train_dataset_signature,
            val_dataset_signature=val_dataset_signature,
            score_weights=score_weights,
            schedule_state=schedule_state,
            startup_phase_times=startup_phase_times,
            startup_total_sec=float(time.perf_counter() - startup_started),
            meta_parsing_train=bool(train_ds.parse_meta),
            meta_parsing_val=bool(val_ds.parse_meta) if val_ds is not None else False,
            train_shard_cache_items=int(args.train_shard_cache_items),
            val_shard_cache_items=int(args.val_shard_cache_items),
        )
        write_run_config(
            out_dir,
            args,
            config_path=args.config,
            section="train",
            extra={
                "model": asdict(cfg),
                "train_dataset_signature": train_dataset_signature,
                "val_dataset_signature": val_dataset_signature,
                "score_weights": score_weights,
                "schedule_state": schedule_state,
                "runtime_summary": runtime_summary,
            },
        )
        _write_json(out_dir / "analysis" / "runtime_summary.json", runtime_summary)
        write_dataset_manifest(
            out_dir,
            stage="train",
            args=args,
            config_path=args.config,
            section="train",
            inputs={
                "train_dir": str(Path(args.train_dir).resolve()),
                "val_dir": str(Path(args.val_dir).resolve()) if args.val_dir else "",
            },
            counts={
                "train_samples": int(len(train_ds)),
                "val_samples": int(len(val_ds)) if val_ds is not None else 0,
                "warmup_epochs": int(warmup_epochs),
            },
            extra={
                "model": asdict(cfg),
                "train_dataset_signature": train_dataset_signature,
                "val_dataset_signature": val_dataset_signature,
                "runtime_summary": runtime_summary,
                "score_weights": score_weights,
                "schedule_state": schedule_state,
            },
        )

    with _startup_phase(logger, startup_phase_times, "training_loop_start"):
        logger.log({
            "event": "start",
            "device": str(device),
            "amp": bool(use_amp),
            "amp_dtype": str(amp_dtype).replace("torch.", ""),
            "train_samples": int(len(train_ds)),
            "val_samples": int(len(val_ds)) if args.val_dir and val_ds is not None else 0,
            "batch_size": int(args.batch_size),
            "requested_num_workers": int(args.num_workers),
            "p_with_b": float(args.p_with_b),
            "num_params": int(num_params),
            "warmup_epochs": int(warmup_epochs),
            "ema_decay": float(args.ema_decay),
            "resume": str(args.resume or ""),
            "resume_model_only": bool(args.resume_model_only),
            "analysis_artifacts": bool(args.save_analysis_artifacts),
            "prediction_artifacts_every": int(args.prediction_artifacts_every),
            "channels_last": bool(args.channels_last),
            "compile_requested": bool(args.compile),
            "compile_enabled": bool(compile_info.get("enabled", False)),
            "train_loader_profile": asdict(train_profile),
            "val_loader_profile": asdict(val_profile) if val_profile is not None else {},
            "train_loader_from_cache": bool(train_loader_info.get("from_cache", False)),
            "val_loader_from_cache": bool(val_loader_info.get("from_cache", False)),
            "train_batching_mode": "shard_aware_boundary_batch_sampler",
            "train_shard_cache_items": int(args.train_shard_cache_items),
            "val_shard_cache_items": int(args.val_shard_cache_items),
            "score_weights": dict(score_weights),
            "save_every_epochs": int(args.save_every_epochs),
        })

    runtime_summary = _make_runtime_summary(
        args=args,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        compile_info=compile_info,
        train_profile=train_profile,
        train_loader_info=train_loader_info,
        train_loader=train_loader,
        val_profile=val_profile,
        val_loader_info=val_loader_info,
        val_loader=val_loader,
        train_dataset_signature=train_dataset_signature,
        val_dataset_signature=val_dataset_signature,
        score_weights=score_weights,
        schedule_state=schedule_state,
        startup_phase_times=startup_phase_times,
        startup_total_sec=float(time.perf_counter() - startup_started),
        meta_parsing_train=bool(train_ds.parse_meta),
        meta_parsing_val=bool(val_ds.parse_meta) if val_ds is not None else False,
        train_shard_cache_items=int(args.train_shard_cache_items),
        val_shard_cache_items=int(args.val_shard_cache_items),
    )
    write_run_config(
        out_dir,
        args,
        config_path=args.config,
        section="train",
        extra={
            "model": asdict(cfg),
            "train_dataset_signature": train_dataset_signature,
            "val_dataset_signature": val_dataset_signature,
            "score_weights": score_weights,
            "schedule_state": schedule_state,
            "runtime_summary": runtime_summary,
        },
    )
    _write_json(out_dir / "analysis" / "runtime_summary.json", runtime_summary)
    write_dataset_manifest(
        out_dir,
        stage="train",
        args=args,
        config_path=args.config,
        section="train",
        inputs={
            "train_dir": str(Path(args.train_dir).resolve()),
            "val_dir": str(Path(args.val_dir).resolve()) if args.val_dir else "",
        },
        counts={
            "train_samples": int(len(train_ds)),
            "val_samples": int(len(val_ds)) if val_ds is not None else 0,
            "warmup_epochs": int(warmup_epochs),
        },
        extra={
            "model": asdict(cfg),
            "train_dataset_signature": train_dataset_signature,
            "val_dataset_signature": val_dataset_signature,
            "runtime_summary": runtime_summary,
            "score_weights": score_weights,
            "schedule_state": schedule_state,
        },
    )

    for epoch in range(int(start_epoch), int(args.epochs) + 1):
        epoch_lr = _lr_for_epoch(schedule_state, epoch)
        _set_optimizer_lr(optim, epoch_lr)
        if train_batch_sampler is not None and hasattr(train_batch_sampler, "set_epoch"):
            train_batch_sampler.set_epoch(int(epoch))
        logger.log(
            {
                "event": "epoch_start",
                "epoch": int(epoch),
                "epochs": int(args.epochs),
                "step": int(step),
                "lr": float(epoch_lr),
                "steps_in_epoch": int(len(train_loader)),
                "log_every": int(args.log_every),
                "prediction_artifacts_every": int(args.prediction_artifacts_every),
            }
        )
        step, train_epoch_metrics = train_one_epoch(
            model=train_model,
            loader=train_loader,
            optim=optim,
            scaler=scaler,
            device=device,
            class_weights=class_weights,
            grad_clip=float(args.grad_clip),
            log_every=int(args.log_every),
            step0=step,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            epoch=epoch,
            logger=logger,
            tb_logger=tb_logger,
            tb_log_every=int(args.log_every_steps),
            label_smoothing=float(args.label_smoothing),
            ema=ema,
            channels_last=bool(args.channels_last),
        )

        epoch_summary: Dict[str, Any] = {
            "epoch": int(epoch),
            "global_step": int(step),
            "train": {k: float(v) for k, v in train_epoch_metrics.items()},
            "lr_before_sched": float(epoch_lr),
        }
        balanced_improved = False
        boundary_improved = False
        if val_loader is not None:
            collect_examples = bool(tb_logger.enabled and args.tb_log_examples and (epoch % max(1, int(args.tb_examples_every)) == 0))
            planned_prediction_artifacts = _should_write_prediction_artifacts(
                enabled=bool(args.save_analysis_artifacts),
                epoch=int(epoch),
                every=int(args.prediction_artifacts_every),
                total_epochs=int(args.epochs),
                boundary_improved=False,
                balanced_improved=False,
            )
            eval_model = ema.module if ema is not None else train_model
            metrics, cm, examples, analysis = eval_epoch(
                eval_model,
                val_loader,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                class_weights=class_weights,
                label_smoothing=float(args.label_smoothing),
                collect_examples=collect_examples,
                examples_k=int(args.tb_examples_k),
                collect_predictions=bool(planned_prediction_artifacts),
                threshold_sweep_points=threshold_points,
                channels_last=bool(args.channels_last),
                score_weights=score_weights,
            )
            if float(metrics.get("support_B", 0.0)) <= 0.0:
                _raise_invalid_val_split_no_boundaries(str(args.val_dir), val_dataset_signature)
            metrics["balanced_guardrails_passed"] = 1.0 if passes_balanced_guardrails(metrics) else 0.0
            metrics["balanced_tracking_started"] = 1.0 if balanced_tracking_started else 0.0
            epoch_summary["val"] = {k: float(v) for k, v in metrics.items()}
            threshold_sweep = analysis.get("threshold_sweep", {}) if isinstance(analysis, dict) else {}
            best_boundary_row = threshold_sweep.get("best_b_f1_tol", {}) if isinstance(threshold_sweep, dict) else {}
            best_balanced_row = threshold_sweep.get("best_balanced", {}) if isinstance(threshold_sweep, dict) else {}
            boundary_candidate = _selection_candidate(
                metrics,
                best_boundary_row if isinstance(best_boundary_row, dict) and best_boundary_row else None,
                selection_source="threshold_sweep" if best_boundary_row else "argmax",
            )
            balanced_candidate: Optional[Dict[str, float]] = None
            if isinstance(best_balanced_row, dict) and best_balanced_row:
                balanced_candidate = _selection_candidate(metrics, best_balanced_row, selection_source="threshold_sweep")
            elif passes_balanced_guardrails(metrics):
                balanced_candidate = _selection_candidate(metrics, None, selection_source="argmax")
            balanced_candidate_present = balanced_candidate is not None
            if balanced_candidate_present:
                balanced_tracking_started = True
            metrics["balanced_tracking_started"] = 1.0 if balanced_tracking_started else 0.0
            metrics["balanced_candidate_present"] = 1.0 if balanced_candidate_present else 0.0
            epoch_summary["val"] = {k: float(v) for k, v in metrics.items()}
            epoch_summary["selection"] = {
                "best_boundary": {
                    "threshold": float(boundary_candidate.get("selection_threshold", 0.5)),
                    "source": "threshold_sweep" if best_boundary_row else "argmax",
                    "b_f1_tol": float(boundary_candidate.get("b_f1_tol", 0.0)),
                    **_selection_context(boundary_candidate),
                },
                "best_balanced": (
                    {
                        "threshold": float(balanced_candidate.get("selection_threshold", 0.5)),
                        "source": "threshold_sweep" if best_balanced_row else "argmax",
                        "balanced_score": float(balanced_candidate.get("balanced_score", 0.0)),
                        **_selection_context(balanced_candidate),
                    }
                    if balanced_candidate is not None else {}
                ),
            }
            logger.log({
                "event": "val_epoch",
                "epoch": epoch,
                "step": step,
                **{k: float(v) for k, v in metrics.items()},
                "selected_boundary_threshold": float(boundary_candidate.get("selection_threshold", 0.5)),
                "selected_boundary_b_f1_tol": float(boundary_candidate.get("b_f1_tol", 0.0)),
                "selected_boundary_b_err_mean": float(boundary_candidate.get("b_err_mean", 0.0)),
                "selected_boundary_pred_B_ratio": float(boundary_candidate.get("pred_B_ratio", 0.0)),
                "selected_boundary_source": "threshold_sweep" if best_boundary_row else "argmax",
                "selected_balanced_threshold": float(balanced_candidate.get("selection_threshold", 0.5)) if balanced_candidate is not None else 0.5,
                "selected_balanced_score": float(balanced_candidate.get("balanced_score", 0.0)) if balanced_candidate is not None else float(metrics.get("balanced_score", 0.0)),
                "selected_balanced_source": ("threshold_sweep" if best_balanced_row else "argmax") if balanced_candidate is not None else "argmax",
                "balanced_tracking_started": 1.0 if balanced_tracking_started else 0.0,
                "balanced_candidate_present": 1.0 if balanced_candidate_present else 0.0,
            })
            if tb_logger.enabled:
                tb_logger.scalars({f"val/{k}": float(v) for k, v in metrics.items()}, step)
                img = build_confusion_image(cm, ["O", "B", "I"])
                tb_logger.image("val/confusion", img, step, dataformats="HWC")
                if collect_examples and examples:
                    lines = ["BIO examples (worst by F1):"]
                    for i, ex in enumerate(examples, start=1):
                        y_seq = ex["true_seq"]
                        p_seq = ex["pred_seq"]
                        y_str = _bio_seq_to_str(y_seq)
                        p_str = _bio_seq_to_str(p_seq)
                        y_b = np.where(y_seq == 1)[0].tolist()
                        y_i = np.where(y_seq == 2)[0].tolist()
                        p_b = np.where(p_seq == 1)[0].tolist()
                        p_i = np.where(p_seq == 2)[0].tolist()
                        f1_i = float(ex.get("f1", 0.0))
                        b_err = ex.get("b_err", 0.0)
                        b_err_s = "inf" if b_err == float("inf") else f"{float(b_err):.2f}"
                        lines.append(f"\n#{i} f1={f1_i:.3f} b_err={b_err_s}")
                        lines.append(f"true_B={y_b} true_I={y_i}")
                        lines.append(f"pred_B={p_b} pred_I={p_i}")
                        lines.append(f"true: {y_str}")
                        lines.append(f"pred: {p_str}")
                    tb_logger.text("val/examples", "\n".join(lines), step)
            if _is_better_boundary(boundary_candidate, best_boundary_metrics):
                best_boundary_metrics = dict(boundary_candidate)
                boundary_improved = True
                boundary_payload = _build_checkpoint_payload(
                    epoch=epoch,
                    global_step=step,
                    model=model,
                    optim=optim,
                    scaler=scaler,
                    ema=ema,
                    cfg=cfg,
                    args=args,
                    last_metrics=epoch_summary,
                    best_boundary_metrics=best_boundary_metrics,
                    best_balanced_metrics=best_balanced_metrics,
                    history=history,
                    epochs_no_improve=epochs_no_improve,
                    schedule_state=schedule_state,
                    train_dataset_signature=train_dataset_signature,
                    val_dataset_signature=val_dataset_signature,
                    balanced_tracking_started=balanced_tracking_started,
                    runtime_summary=runtime_summary,
                    include_history=False,
                    include_runtime_summary=False,
                )
                torch.save(boundary_payload, out_dir / "best_boundary.pt")
                _save_model_only_checkpoint(out_dir / "best_boundary_model.pt", boundary_payload)
                torch.save(boundary_payload, out_dir / "best.pt")
                _save_model_only_checkpoint(out_dir / "best_model.pt", boundary_payload)
                logger.log(
                    {
                        "event": "best_boundary",
                        "epoch": int(epoch),
                        "step": int(step),
                        "threshold": float(best_boundary_metrics.get("selection_threshold", 0.5)),
                        "b_f1_tol": float(best_boundary_metrics.get("b_f1_tol", 0.0)),
                        "b_err_mean": float(best_boundary_metrics.get("b_err_mean", 0.0)),
                        "pred_B_ratio": float(best_boundary_metrics.get("pred_B_ratio", 0.0)),
                        "transition_rate_abs_err": float(best_boundary_metrics.get("transition_rate_abs_err", 0.0)),
                        "bio_violation_abs_err": float(best_boundary_metrics.get("bio_violation_abs_err", 0.0)),
                    }
                )
            if balanced_candidate is not None:
                min_delta = float(max(0.0, args.early_stop_min_delta))
                better_than_best = best_balanced_metrics is None
                if best_balanced_metrics is not None:
                    cand_score = float(balanced_candidate.get("balanced_score", balanced_model_score(balanced_candidate, **score_weights)))
                    best_score = float(best_balanced_metrics.get("balanced_score", balanced_model_score(best_balanced_metrics, **score_weights)))
                    better_than_best = cand_score > best_score + min_delta
                    if not better_than_best and abs(cand_score - best_score) <= min_delta:
                        better_than_best = _is_better_balanced(
                            balanced_candidate,
                            best_balanced_metrics,
                            score_weights=score_weights,
                        )
                balanced_improved = bool(better_than_best)
            if balanced_candidate is not None and balanced_improved:
                best_balanced_metrics = dict(balanced_candidate)
                balanced_payload = _build_checkpoint_payload(
                    epoch=epoch,
                    global_step=step,
                    model=model,
                    optim=optim,
                    scaler=scaler,
                    ema=ema,
                    cfg=cfg,
                    args=args,
                    last_metrics=epoch_summary,
                    best_boundary_metrics=best_boundary_metrics,
                    best_balanced_metrics=best_balanced_metrics,
                    history=history,
                    epochs_no_improve=epochs_no_improve,
                    schedule_state=schedule_state,
                    train_dataset_signature=train_dataset_signature,
                    val_dataset_signature=val_dataset_signature,
                    balanced_tracking_started=balanced_tracking_started,
                    runtime_summary=runtime_summary,
                    include_history=False,
                    include_runtime_summary=False,
                )
                torch.save(balanced_payload, out_dir / "best_balanced.pt")
                _save_model_only_checkpoint(out_dir / "best_balanced_model.pt", balanced_payload)
                logger.log(
                    {
                        "event": "best_balanced",
                        "epoch": int(epoch),
                        "step": int(step),
                        "threshold": float(best_balanced_metrics.get("selection_threshold", 0.5)),
                        "balanced_score": float(best_balanced_metrics.get("balanced_score", 0.0)),
                        "b_f1_tol": float(best_balanced_metrics.get("b_f1_tol", 0.0)),
                        "b_err_mean": float(best_balanced_metrics.get("b_err_mean", 0.0)),
                        "pred_B_ratio": float(best_balanced_metrics.get("pred_B_ratio", 0.0)),
                    }
                )
            write_prediction_rows = _should_write_prediction_artifacts(
                enabled=bool(args.save_analysis_artifacts),
                epoch=int(epoch),
                every=int(args.prediction_artifacts_every),
                total_epochs=int(args.epochs),
                boundary_improved=bool(boundary_improved),
                balanced_improved=bool(balanced_improved),
            )
            if args.save_analysis_artifacts:
                if write_prediction_rows and "prediction_rows" not in analysis:
                    _pred_metrics, _pred_cm, _pred_examples, pred_analysis = eval_epoch(
                        eval_model,
                        val_loader,
                        device=device,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        class_weights=class_weights,
                        label_smoothing=float(args.label_smoothing),
                        collect_examples=False,
                        examples_k=int(args.tb_examples_k),
                        collect_predictions=True,
                        threshold_sweep_points=None,
                        channels_last=bool(args.channels_last),
                        score_weights=score_weights,
                    )
                    analysis["prediction_rows"] = pred_analysis.get("prediction_rows", [])
                _write_analysis_artifacts(
                    out_dir,
                    epoch,
                    analysis.get("prediction_rows", []),
                    analysis.get("threshold_sweep"),
                    write_predictions=write_prediction_rows,
                )
            epochs_no_improve = _update_early_stop_counter(
                epochs_no_improve,
                enabled=int(args.early_stop_patience) > 0,
                epoch=int(epoch),
                armed_epoch=int(early_stop_armed_epoch),
                balanced_tracking_started=bool(balanced_tracking_started),
                balanced_improved=bool(balanced_improved),
            )
            epoch_summary["balanced_tracking_started"] = bool(balanced_tracking_started)
            epoch_summary["balanced_candidate_present"] = bool(balanced_candidate_present)

        if tb_logger.enabled:
            for name, param in hist_params:
                tb_logger.histogram(f"weights/{name}", param, step)
                if param.grad is not None:
                    tb_logger.histogram(f"grads/{name}", param.grad, step)

        schedule_state["completed_epochs"] = int(epoch)
        epoch_summary["lr_after_sched"] = float(_lr_for_epoch(schedule_state, epoch + 1))
        epoch_summary["epochs_no_improve"] = int(epochs_no_improve)
        history.append(epoch_summary)
        _write_json(out_dir / "history.json", history)
        ckpt_payload = _build_checkpoint_payload(
            epoch=epoch,
            global_step=step,
            model=model,
            optim=optim,
            scaler=scaler,
            ema=ema,
            cfg=cfg,
            args=args,
            last_metrics=epoch_summary,
            best_boundary_metrics=best_boundary_metrics,
            best_balanced_metrics=best_balanced_metrics,
            history=history,
            epochs_no_improve=epochs_no_improve,
            schedule_state=schedule_state,
            train_dataset_signature=train_dataset_signature,
            val_dataset_signature=val_dataset_signature,
            balanced_tracking_started=balanced_tracking_started,
            runtime_summary=runtime_summary,
            include_history=True,
            include_runtime_summary=True,
        )
        periodic_ckpt_payload = _build_checkpoint_payload(
            epoch=epoch,
            global_step=step,
            model=model,
            optim=optim,
            scaler=scaler,
            ema=ema,
            cfg=cfg,
            args=args,
            last_metrics=epoch_summary,
            best_boundary_metrics=best_boundary_metrics,
            best_balanced_metrics=best_balanced_metrics,
            history=history,
            epochs_no_improve=epochs_no_improve,
            schedule_state=schedule_state,
            train_dataset_signature=train_dataset_signature,
            val_dataset_signature=val_dataset_signature,
            balanced_tracking_started=balanced_tracking_started,
            runtime_summary=runtime_summary,
            include_history=False,
            include_runtime_summary=False,
        )
        save_checkpoint(
            out_dir,
            epoch,
            ckpt_payload,
            int(args.save_every_epochs),
            periodic_payload=periodic_ckpt_payload,
        )
        if (
            int(args.early_stop_patience) > 0
            and epoch >= int(early_stop_armed_epoch)
            and balanced_tracking_started
            and epochs_no_improve >= int(args.early_stop_patience)
        ):
            logger.log(
                {
                    "event": "early_stop",
                    "epoch": int(epoch),
                    "step": int(step),
                    "epochs_no_improve": int(epochs_no_improve),
                    "patience": int(args.early_stop_patience),
                    "balanced_tracking_started": bool(balanced_tracking_started),
                }
            )
            break

    done_payload: Dict[str, object] = {"event": "done", "step": step}
    if best_boundary_metrics is not None:
        done_payload["best_boundary"] = {
            "b_f1_tol": float(best_boundary_metrics.get("b_f1_tol", 0.0)),
            "pred_B_ratio": float(best_boundary_metrics.get("pred_B_ratio", 0.0)),
            "avg_seg_len_ratio": float(best_boundary_metrics.get("avg_seg_len_ratio", 0.0)),
            "b_err_mean": float(best_boundary_metrics.get("b_err_mean", 0.0)),
            "threshold": float(best_boundary_metrics.get("selection_threshold", 0.5)),
        }
    if best_balanced_metrics is not None:
        done_payload["best_balanced"] = {
            "b_f1_tol": float(best_balanced_metrics.get("b_f1_tol", 0.0)),
            "pred_B_ratio": float(best_balanced_metrics.get("pred_B_ratio", 0.0)),
            "avg_seg_len_ratio": float(best_balanced_metrics.get("avg_seg_len_ratio", 0.0)),
            "b_err_mean": float(best_balanced_metrics.get("b_err_mean", 0.0)),
            "threshold": float(best_balanced_metrics.get("selection_threshold", 0.5)),
        }
    logger.log(done_payload)
    if tb_logger.enabled:
        tb_logger.hparams(
            {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "p_with_b": float(args.p_with_b),
                "embed_dim": int(args.embed_dim),
                "conv_layers": int(args.conv_layers),
                "gru_hidden": int(args.gru_hidden),
            },
            {
                "best_boundary_b_f1_tol": float(best_boundary_metrics.get("b_f1_tol", 0.0)) if best_boundary_metrics else 0.0,
                "best_balanced_b_f1_tol": float(best_balanced_metrics.get("b_f1_tol", 0.0)) if best_balanced_metrics else 0.0,
            },
        )
        tb_logger.scalar("meta/warmup_epochs", float(warmup_epochs), 0)
        tb_logger.scalar("meta/ema_decay", float(args.ema_decay), 0)
        tb_logger.close()
    print("Done. last.pt saved in", str(out_dir))
    if best_boundary_metrics is not None:
        print(
            "best_boundary.pt:",
            json.dumps(
                {
                    "b_f1_tol": float(best_boundary_metrics.get("b_f1_tol", 0.0)),
                    "pred_B_ratio": float(best_boundary_metrics.get("pred_B_ratio", 0.0)),
                    "avg_seg_len_ratio": float(best_boundary_metrics.get("avg_seg_len_ratio", 0.0)),
                    "b_err_mean": float(best_boundary_metrics.get("b_err_mean", 0.0)),
                    "threshold": float(best_boundary_metrics.get("selection_threshold", 0.5)),
                },
                ensure_ascii=True,
            ),
        )
    if best_balanced_metrics is not None:
        print(
            "best_balanced.pt:",
            json.dumps(
                {
                    "b_f1_tol": float(best_balanced_metrics.get("b_f1_tol", 0.0)),
                    "pred_B_ratio": float(best_balanced_metrics.get("pred_B_ratio", 0.0)),
                    "avg_seg_len_ratio": float(best_balanced_metrics.get("avg_seg_len_ratio", 0.0)),
                    "b_err_mean": float(best_balanced_metrics.get("b_err_mean", 0.0)),
                    "threshold": float(best_balanced_metrics.get("selection_threshold", 0.5)),
                },
                ensure_ascii=True,
            ),
        )


if __name__ == "__main__":
    main()
