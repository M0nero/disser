from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, Optional

from .config import ExtractorConfig
from .core.logging_utils import configure_logging, log_metrics, redirect_native_stderr, track_runtime
from .output.staging import write_staged_payload
from .process import process_task
from .tasks import TaskSpec

_WORKER_CPU_AFFINITY: tuple[int, ...] = ()


def normalize_sample_id(value: str) -> str:
    sample_id = str(value or "").strip().replace("\\", "__").replace("/", "__")
    if not sample_id:
        raise ValueError("sample_id must not be empty")
    return sample_id


def slug_for(v: Path, base: Optional[Path]) -> str:
    try:
        if base:
            rel = v.resolve().relative_to(base.resolve())
            return "__".join(rel.with_suffix("").parts)
    except Exception:
        pass
    return v.stem


def _resolve_debug_video_path(config: ExtractorConfig, slug: str) -> Optional[Path]:
    raw = (config.debug.debug_video or "").strip()
    if not raw:
        return None
    target = Path(raw)
    if target.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        return target / f"{slug}.mp4"
    if target.suffix.lower() == ".mp4":
        video_count = int(config.runtime.video_count or 1)
        if video_count <= 1:
            target.parent.mkdir(parents=True, exist_ok=True)
            return target
        debug_dir = target.with_suffix("")
        if debug_dir.exists() and not debug_dir.is_dir():
            debug_dir = target.with_name(f"{target.stem}_debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        return debug_dir / f"{slug}.mp4"
    target.mkdir(parents=True, exist_ok=True)
    return target / f"{slug}.mp4"


def detect_available_cpu_ids() -> list[int]:
    get_affinity = getattr(os, "sched_getaffinity", None)
    if callable(get_affinity):
        try:
            return sorted(int(cpu) for cpu in get_affinity(0))
        except Exception:
            pass
    cpu_count = int(os.cpu_count() or 1)
    return list(range(max(1, cpu_count)))


def plan_cpu_affinity_assignments(available_cpus: list[int], workers: int) -> list[tuple[int, ...]]:
    cpus = [int(cpu) for cpu in available_cpus]
    if not cpus:
        return []
    worker_count = max(1, min(int(workers), len(cpus)))
    base = len(cpus) // worker_count
    remainder = len(cpus) % worker_count
    assignments: list[tuple[int, ...]] = []
    start = 0
    for idx in range(worker_count):
        size = base + (1 if idx < remainder else 0)
        stop = start + size
        chunk = tuple(cpus[start:stop])
        if not chunk:
            chunk = (cpus[idx % len(cpus)],)
        assignments.append(chunk)
        start = stop
    return assignments


def _apply_cpu_affinity(cpu_ids: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    assigned = tuple(sorted({int(cpu) for cpu in cpu_ids}))
    if not assigned:
        return ()
    set_affinity = getattr(os, "sched_setaffinity", None)
    if not callable(set_affinity):
        return ()
    try:
        set_affinity(0, set(assigned))
    except Exception:
        return ()
    return assigned


def init_process_worker(affinity_queue: Any = None) -> None:
    global _WORKER_CPU_AFFINITY
    _WORKER_CPU_AFFINITY = ()
    if affinity_queue is None:
        return
    try:
        cpu_ids = affinity_queue.get_nowait()
    except Exception:
        return
    _WORKER_CPU_AFFINITY = _apply_cpu_affinity(tuple(int(cpu) for cpu in cpu_ids))


def process_worker(payload: dict) -> Dict[str, Any]:
    task_spec = TaskSpec.from_payload(payload)
    config = ExtractorConfig.from_dict(task_spec.config_dict)
    vpath = task_spec.source_path
    segment = task_spec.segment_meta or {}

    logger = configure_logging(
        config.logging.log_dir,
        config.logging.log_level,
        console=bool(config.logging.worker_console),
    )
    native_stderr_dir = Path(config.output.stage_dir) / "native_stderr"
    native_stderr_path = native_stderr_dir / f"worker_{os.getpid()}.stderr.log"

    in_dir = Path(config.video.in_dir) if config.video.in_dir else None
    out_dir = Path(config.video.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(config.output.stage_dir) / f"worker_{os.getpid()}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    slug = str(task_spec.slug or segment.get("seg_uid") or slug_for(vpath, in_dir))
    sample_id = normalize_sample_id(str(task_spec.sample_id or slug))
    debug_video_path = _resolve_debug_video_path(config, slug)
    seed = int(config.runtime.seed)
    frame_start = task_spec.frame_start
    frame_end = task_spec.frame_end

    ndjson_path = Path(task_spec.ndjson_path) if task_spec.ndjson_path else None

    log_metrics(logger, "process_worker.start", {
        "video": vpath.name,
        "slug": slug,
        "sample_id": str(sample_id),
        "stage_dir": str(stage_dir),
        "ndjson": str(ndjson_path) if ndjson_path else None,
        "debug_video": str(debug_video_path) if debug_video_path else None,
        "eval_mode": True,
        "seed": seed,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "seg_uid": segment.get("seg_uid") if segment else None,
        "cpu_affinity": list(_WORKER_CPU_AFFINITY),
    })

    try:
        with redirect_native_stderr(native_stderr_path):
            with track_runtime(logger, "process_worker", video=vpath.name, slug=slug):
                worker_task = TaskSpec(
                    sample_id=str(sample_id),
                    slug=str(slug),
                    source_video=str(vpath),
                    config_dict=config.to_dict(),
                    frame_start=frame_start,
                    frame_end=frame_end,
                    segment_meta=dict(segment),
                    debug_video_path=str(debug_video_path) if debug_video_path else "",
                    ndjson_path=str(ndjson_path) if ndjson_path else "",
                )
                sample_payload = process_task(worker_task)
                staged_path = write_staged_payload(stage_dir, str(sample_id), sample_payload)
                video_row = dict(sample_payload.video_row)
                runtime_metrics = dict(sample_payload.runtime_metrics or {})

        log_metrics(logger, "process_worker.result", {
            "video": vpath.name,
            "slug": slug,
            "sample_id": str(sample_id),
            "frames": video_row.get("num_frames", 0),
            "hands_coverage": video_row.get("hands_coverage"),
            "quality_score": video_row.get("quality_score"),
            "staged_path": str(staged_path),
        })
        return {
            "sample_id": str(sample_id),
            "slug": str(slug),
            "staged_path": str(staged_path),
            "num_frames": int(video_row.get("num_frames", 0) or 0),
            "hands_coverage": float(video_row.get("hands_coverage", 0.0) or 0.0),
            "quality_score": float(video_row.get("quality_score", 0.0) or 0.0),
            "runtime_metrics": runtime_metrics,
        }

    except Exception as exc:
        log_metrics(logger, "process_worker.error", {
            "video": vpath.name,
            "slug": slug,
            "sample_id": str(sample_id),
            "error": str(exc),
        })
        raise
