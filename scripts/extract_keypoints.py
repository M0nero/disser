from __future__ import annotations
import argparse, csv, json, os, sys, traceback, uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional, Tuple
from time import perf_counter
import subprocess

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kp_export.core.utils import parse_keep_indices
from kp_export.annotations import load_skip_ids
from kp_export.config import (
    DebugConfig,
    ExtractorConfig,
    LoggingConfig,
    MediaPipeConfig,
    OcclusionConfig,
    OutputConfig,
    PoseConfig,
    PostprocessConfig,
    RuntimeConfig,
    SanityConfig,
    ScoreConfig,
    SecondPassConfig,
    TrackingConfig,
    VideoConfig,
)
from kp_export.parallel import normalize_sample_id, process_worker, slug_for
from kp_export.process import process_video
from kp_export.process.adapters import MediaPipeGpuSession
from kp_export.mp.mp_utils import resolve_task_model_path, try_import_mediapipe
from kp_export.core.logging_utils import configure_logging, log_metrics, track_runtime
from kp_export.runpod.status import ShardStatusReporter
from kp_export.task_manifest import filter_tasks_for_shard, load_task_manifest, write_task_manifest
from kp_export.tasks import TaskSpec
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool


def _parse_range(raw: str) -> tuple[float, float]:
    try:
        parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError("expected two numbers")
        lo = float(parts[0])
        hi = float(parts[1])
    except Exception as exc:
        raise SystemExit(
            f"Invalid --sanity-scale-range '{raw}'. Expected format: '0.70,1.35'."
        ) from exc
    if lo <= 0.0 or hi <= 0.0 or lo >= hi:
        raise SystemExit(
            f"Invalid --sanity-scale-range '{raw}'. Expected positive lo<hi."
        )
    return (lo, hi)


_SEGMENT_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _sniff_csv_dialect(sample: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        class _D(csv.Dialect):
            delimiter = "\t" if ("\t" in sample and "," not in sample) else ","
            quotechar = '"'
            doublequote = True
            skipinitialspace = False
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL
        return _D()


def _read_segments_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Segments manifest not found: {path}")
    suffix = path.suffix.lower()
    rows: List[Dict[str, Any]] = []
    if suffix == ".jsonl":
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise SystemExit(f"Invalid JSONL at line {idx}: {path}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
        if not rows:
            raise SystemExit(f"Segments manifest is empty: {path}")
        return rows
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            dialect = _sniff_csv_dialect(sample)
            rdr = csv.DictReader(f, dialect=dialect)
            rows = [dict(r) for r in rdr]
        if not rows:
            raise SystemExit(f"Segments manifest is empty: {path}")
        return rows
    raise SystemExit(f"Unsupported segments manifest format: {path}")


def _parse_segment_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    errors: List[str] = []

    def _norm_row(row: Dict[str, Any]) -> Dict[str, Any]:
        return {str(k).strip().lower(): v for k, v in row.items()}

    def _req_str(row: Dict[str, Any], key: str) -> str:
        raw = row.get(key)
        val = str(raw).strip() if raw is not None else ""
        if not val:
            raise ValueError(f"missing '{key}'")
        return val

    def _req_int(row: Dict[str, Any], key: str) -> int:
        raw = row.get(key)
        if raw is None or str(raw).strip() == "":
            raise ValueError(f"missing '{key}'")
        try:
            return int(raw)
        except Exception:
            try:
                return int(float(str(raw)))
            except Exception as exc:
                raise ValueError(f"invalid '{key}': {raw}") from exc

    for idx, row in enumerate(rows, start=1):
        row_norm = _norm_row(row)
        try:
            video_id = _req_str(row_norm, "video_id")
            seg_uid = _req_str(row_norm, "seg_uid")
            start = _req_int(row_norm, "start")
            end = _req_int(row_norm, "end")
            if start < 0:
                raise ValueError("start must be >= 0")
            if end <= start:
                raise ValueError("end must be > start")
            seg: Dict[str, Any] = {
                "video_id": video_id,
                "seg_uid": seg_uid,
                "start": int(start),
                "end": int(end),
            }
            for key in ("split", "dataset", "label"):
                val = row_norm.get(key)
                val = str(val).strip() if val is not None else ""
                if val:
                    seg[key] = val
            if "length" in row_norm and str(row_norm.get("length")).strip() != "":
                try:
                    seg["length"] = int(float(str(row_norm.get("length"))))
                except Exception:
                    seg["length"] = row_norm.get("length")
            segments.append(seg)
        except Exception as exc:
            errors.append(f"row {idx}: {exc}")

    if errors:
        sample = "\n".join(errors[:5])
        raise SystemExit(f"Invalid segments manifest (first errors):\n{sample}")
    if not segments:
        raise SystemExit("Segments manifest has no valid segments.")
    return segments


def _index_videos_by_id(in_dir: Path) -> Tuple[List[Path], Dict[str, List[Path]], Dict[str, List[Path]]]:
    files = [
        p for p in in_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _SEGMENT_VIDEO_EXTS
    ]
    files = sorted(files, key=lambda p: str(p))
    by_name: Dict[str, List[Path]] = {}
    by_stem: Dict[str, List[Path]] = {}
    for p in files:
        by_name.setdefault(p.name.lower(), []).append(p)
        by_stem.setdefault(p.stem.lower(), []).append(p)
    return files, by_name, by_stem


def _resolve_video_path(
    video_id: str,
    by_name: Dict[str, List[Path]],
    by_stem: Dict[str, List[Path]],
) -> Tuple[Optional[Path], Optional[str]]:
    raw = str(video_id or "").strip()
    if not raw:
        return None, None
    name_key = Path(raw).name.lower()
    stem_key = Path(raw).stem.lower()
    matches = by_name.get(name_key)
    match_type = "name" if matches else "stem"
    if not matches:
        matches = by_stem.get(stem_key)
    if not matches:
        return None, None
    matches = sorted(matches, key=lambda p: str(p))
    picked = matches[0]
    if len(matches) > 1:
        return picked, match_type
    return picked, None


def _normalize_ndjson_base(out_dir: Path, raw: str) -> str:
    raw_text = str(raw or "").strip()
    if not raw_text:
        return ""
    base = Path(raw_text)
    name = base.name or "frames.ndjson"
    if not Path(name).suffix:
        name = f"{name}.ndjson"
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return str((debug_dir / name).resolve())


def _versions_snapshot() -> Dict[str, Any]:
    versions: Dict[str, Any] = {"python": sys.version}
    try:
        import cv2

        versions["cv2"] = cv2.__version__
    except Exception:
        versions["cv2"] = None
    try:
        import mediapipe as mp

        versions["mediapipe"] = getattr(mp, "__version__", None)
    except Exception:
        versions["mediapipe"] = None
    return versions


def _mean(values: List[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def _safe_rate(num: Any, denom: Any) -> Optional[float]:
    if denom and denom > 0:
        return float(num) / float(denom)
    return None


def _shutdown_process_pool_fast(executor: Optional[ProcessPoolExecutor]) -> None:
    if executor is None:
        return
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass

    processes = getattr(executor, "_processes", None) or {}
    for proc in list(processes.values()):
        if proc is None:
            continue
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass

    for proc in list(processes.values()):
        if proc is None:
            continue
        try:
            proc.join(timeout=1.0)
        except Exception:
            pass
        try:
            if proc.is_alive():
                kill = getattr(proc, "kill", None)
                if callable(kill):
                    kill()
                else:
                    proc.terminate()
        except Exception:
            pass


def _aggregate_video_rows(rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
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
        q = row.get("quality_score")
        if q is not None:
            quality_sum += float(q) * num_frames
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


def _format_duration_hms(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    try:
        total = max(0, int(round(float(seconds))))
    except Exception:
        return "?"
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _maybe_report_progress(
    logger,
    *,
    started_at: float,
    total: int,
    ok_count: int,
    failed_count: int,
    state: Dict[str, Any],
    reporter: Optional[ShardStatusReporter] = None,
    console_mode: str = "lines",
    progress_every_items: int = 25,
    progress_every_sec: float = 15.0,
    last_sample_id: str = "",
    force: bool = False,
) -> None:
    done_count = int(ok_count) + int(failed_count)
    if total <= 0 or done_count <= 0:
        return

    now = perf_counter()
    last_log_at = float(state.get("last_log_at", started_at))
    last_done = int(state.get("last_done", 0))
    done_delta = done_count - last_done
    should_log = (
        force
        or done_count == 1
        or done_count == total
        or done_delta >= max(1, int(progress_every_items))
        or (now - last_log_at) >= max(1.0, float(progress_every_sec))
    )
    if not should_log:
        return

    elapsed = max(0.0, now - started_at)
    rate = (done_count / elapsed) if elapsed > 0.0 else None
    remaining = max(0, total - done_count)
    eta_sec = (remaining / rate) if rate and rate > 0.0 else None
    avg_sec = (elapsed / done_count) if done_count > 0 else None

    payload = {
        "done": int(done_count),
        "total": int(total),
        "ok": int(ok_count),
        "failed": int(failed_count),
        "remaining": int(remaining),
        "elapsed_sec": round(elapsed, 3),
        "avg_sec_per_video": round(avg_sec, 3) if avg_sec is not None else None,
        "videos_per_sec": round(rate, 4) if rate is not None else None,
        "eta_sec": round(eta_sec, 3) if eta_sec is not None else None,
        "last_sample_id": str(last_sample_id or ""),
    }
    log_metrics(logger, "extract_keypoints.progress", payload)
    line = (
        "[PROGRESS] "
        f"{done_count}/{total} "
        f"| ok={ok_count} failed={failed_count} remaining={remaining} "
        f"| elapsed={_format_duration_hms(elapsed)} eta={_format_duration_hms(eta_sec)}"
    )
    if console_mode == "compact" and sys.stdout.isatty():
        end = "\n" if force or done_count == total else "\r"
        print(line.ljust(120), end=end, flush=True)
    else:
        print(line)
    if reporter is not None:
        reporter.update(
            state="running",
            processed=int(ok_count),
            failed=int(failed_count),
            remaining=int(remaining),
            videos_per_sec=rate,
            avg_sec_per_video=avg_sec,
            eta_sec=eta_sec,
            last_sample_id=str(last_sample_id or ""),
        )
    state["last_log_at"] = now
    state["last_done"] = done_count


def _load_only_list(path: str) -> set[str]:
    allow = set()
    if not path:
        return allow
    allow_path = Path(path)
    for line in allow_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            allow.add(Path(line).stem)
    return allow


def _probe_gpu_delegate() -> None:
    if not sys.platform.startswith("linux"):
        raise SystemExit("GPU delegate requires Linux/Ubuntu runtime.")
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit("nvidia-smi not found; cannot validate GPU delegate runtime.") from exc
    if result.returncode != 0 or not result.stdout.strip():
        stderr = result.stderr.strip() or "no GPUs visible"
        raise SystemExit(f"GPU delegate requested but NVIDIA runtime is not healthy: {stderr}")


def _detect_visible_gpu_count() -> Optional[int]:
    raw = os.environ.get("RUNPOD_GPU_COUNT", "").strip()
    if raw.isdigit():
        return int(raw)
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    lines = [line for line in (result.stdout or "").splitlines() if line.strip().startswith("GPU ")]
    return len(lines) if lines else None


def _resolve_execution_mode(
    requested_mode: str,
    *,
    mp_backend: str,
    mp_tasks_delegate: str,
) -> str:
    requested = str(requested_mode or "auto").strip().lower()
    backend = str(mp_backend or "solutions").strip().lower()
    delegate = str(mp_tasks_delegate or "auto").strip().lower()
    if requested not in {"auto", "cpu_pool", "gpu_single"}:
        raise SystemExit(f"Unsupported --execution-mode: {requested_mode}")
    if requested == "auto":
        if backend == "tasks" and delegate == "gpu":
            return "gpu_single"
        return "cpu_pool"
    if requested == "cpu_pool" and backend == "tasks" and delegate == "gpu":
        raise SystemExit("--execution-mode cpu_pool is unsupported with --mp-backend tasks --mp-tasks-delegate gpu")
    if requested == "gpu_single" and not (backend == "tasks" and delegate == "gpu"):
        raise SystemExit("--execution-mode gpu_single requires --mp-backend tasks --mp-tasks-delegate gpu")
    return requested


@contextmanager
def _redirect_native_stderr(path: Optional[Path]):
    if path is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stderr.flush()
    saved_fd = os.dup(2)
    target = open(path, "ab", buffering=0)
    try:
        os.dup2(target.fileno(), 2)
        yield
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        target.close()


def _accumulate_runtime_metrics(target: Dict[str, float], runtime_metrics: Optional[Dict[str, Any]]) -> None:
    if not runtime_metrics:
        return
    for key in (
        "processing_elapsed",
        "decode_runtime",
        "detector_init_runtime",
        "hand_runtime",
        "pose_runtime",
        "second_pass_runtime",
        "writer_runtime",
    ):
        value = runtime_metrics.get(key)
        if value is None:
            continue
        try:
            target[key] = float(target.get(key, 0.0) or 0.0) + float(value)
        except Exception:
            continue


def _summarize_native_stderr(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    counts: Dict[str, int] = {}
    try:
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            counts[line] = counts.get(line, 0) + 1
    except Exception:
        return []
    return [
        {"message": message, "count": count}
        for message, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]
    ]


def _emit_runtime_summary(
    logger,
    *,
    execution_mode: str,
    processed_rows: List[Dict[str, Any]],
    runtime_totals: Dict[str, float],
    started_at: float,
    native_warnings: Optional[List[Dict[str, Any]]] = None,
) -> None:
    elapsed = max(0.0, perf_counter() - started_at)
    videos = len(processed_rows)
    frames = sum(int(row.get("num_frames", 0) or 0) for row in processed_rows)
    videos_per_sec = _safe_rate(videos, elapsed)
    frames_per_sec = _safe_rate(frames, elapsed)
    stage_keys = [
        "decode_runtime",
        "detector_init_runtime",
        "hand_runtime",
        "pose_runtime",
        "second_pass_runtime",
        "writer_runtime",
    ]
    stage_shares = {
        f"{key}_share": round(runtime_totals.get(key, 0.0) / elapsed, 4) if elapsed > 0.0 else None
        for key in stage_keys
    }
    payload = {
        "execution_mode": str(execution_mode),
        "videos": int(videos),
        "frames": int(frames),
        "elapsed_sec": round(elapsed, 3),
        "videos_per_sec": round(videos_per_sec, 4) if videos_per_sec is not None else None,
        "frames_per_sec": round(frames_per_sec, 4) if frames_per_sec is not None else None,
        **{key: round(float(runtime_totals.get(key, 0.0) or 0.0), 3) for key in stage_keys},
        **stage_shares,
    }
    if native_warnings:
        payload["native_warning_count"] = int(sum(int(item.get("count", 0) or 0) for item in native_warnings))
        payload["native_warnings_top"] = native_warnings
    log_metrics(logger, "extract_keypoints.runtime_summary", payload)
    print(
        "[SUMMARY] "
        f"mode={execution_mode} videos={videos} frames={frames} "
        f"elapsed={_format_duration_hms(elapsed)} "
        f"videos/s={videos_per_sec:.3f} frames/s={frames_per_sec:.2f}"
        if videos_per_sec is not None and frames_per_sec is not None
        else f"[SUMMARY] mode={execution_mode} videos={videos} frames={frames} elapsed={_format_duration_hms(elapsed)}"
    )
    if native_warnings:
        top = native_warnings[0]
        print(
            "[SUMMARY] "
            f"native_stderr={sum(int(item.get('count', 0) or 0) for item in native_warnings)} lines "
            f"top={top['count']}x {top['message'][:120]}"
        )


def _ensure_writable_dir(path: Path, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".{label}.write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        raise SystemExit(f"{label} is not writable: {path}") from exc


def _build_tasks(
    *,
    args: argparse.Namespace,
    config: ExtractorConfig,
    writer,
    skip_ids: set[str],
    logger,
    ndjson_base: str,
) -> tuple[List[TaskSpec], int, int]:
    tasks: List[TaskSpec] = []
    total_items = 0
    skipped_existing = 0

    if args.task_manifest:
        manifest_path = Path(args.task_manifest)
        tasks = load_task_manifest(manifest_path)
        total_items = len(tasks)
        log_metrics(logger, "extract_keypoints.task_manifest_loaded", {
            "path": str(manifest_path),
            "tasks": len(tasks),
        })
        if args.skip_existing:
            filtered_tasks = []
            for task in tasks:
                if writer.is_sample_committed(task.sample_id):
                    skipped_existing += 1
                    continue
                filtered_tasks.append(task)
            tasks = filtered_tasks
    else:
        segments_manifest = Path(args.segments_manifest) if args.segments_manifest else None
        skip_ids_lower = {str(s).lower() for s in skip_ids}
        allow = _load_only_list(args.only_list)
        if segments_manifest:
            rows = _read_segments_manifest(segments_manifest)
            segments = _parse_segment_rows(rows)
            total_items = len(segments)
            video_files, by_name, by_stem = _index_videos_by_id(Path(args.in_dir))
            log_metrics(logger, "extract_keypoints.segments_manifest_loaded", {
                "manifest": str(segments_manifest),
                "segments": len(segments),
            })
            log_metrics(logger, "extract_keypoints.segment_videos_indexed", {
                "in_dir": str(args.in_dir),
                "files": len(video_files),
            })
            missing_videos: List[str] = []
            dup_video_matches = 0
            seen_seg_uids = set()
            seen_sample_ids = set()
            for seg in segments:
                video_id = str(seg.get("video_id", "")).strip()
                seg_uid = str(seg.get("seg_uid", "")).strip()
                if not video_id or not seg_uid:
                    continue
                if allow and Path(video_id).stem not in allow:
                    continue
                if skip_ids_lower and Path(video_id).stem.lower() in skip_ids_lower:
                    continue
                if seg_uid in seen_seg_uids:
                    continue
                vpath, dup_type = _resolve_video_path(video_id, by_name, by_stem)
                if vpath is None:
                    missing_videos.append(video_id)
                    continue
                if dup_type:
                    dup_video_matches += 1
                sample_id = normalize_sample_id(seg_uid)
                if sample_id in seen_sample_ids:
                    continue
                if args.skip_existing and writer.is_sample_committed(sample_id):
                    skipped_existing += 1
                    continue
                seen_seg_uids.add(seg_uid)
                seen_sample_ids.add(sample_id)
                task_ndjson = ""
                if ndjson_base:
                    base = Path(ndjson_base)
                    task_ndjson = str(base.with_name(f"{base.stem}.{sample_id}{base.suffix}"))
                tasks.append(TaskSpec(
                    sample_id=sample_id,
                    slug=sample_id,
                    source_video=str(vpath),
                    config_dict=config.to_dict(),
                    frame_start=seg.get("start"),
                    frame_end=seg.get("end"),
                    segment_meta=dict(seg),
                    ndjson_path=task_ndjson,
                ))
            if missing_videos:
                missing_set = sorted(set(missing_videos))
                log_metrics(logger, "extract_keypoints.error", {
                    "reason": "missing_videos",
                    "missing_count": len(missing_set),
                    "first_missing": ",".join(missing_set[:10]),
                })
                raise SystemExit(
                    "Missing videos for some segments. "
                    f"First missing: {', '.join(missing_set[:10])}"
                )
            if dup_video_matches:
                log_metrics(logger, "extract_keypoints.segment_duplicate_video_matches", {
                    "count": dup_video_matches,
                })
        else:
            videos = sorted(Path(args.in_dir).glob(args.pattern))
            if allow:
                videos = [v for v in videos if v.stem in allow]
                log_metrics(logger, "extract_keypoints.whitelist_loaded", {
                    "from": str(args.only_list),
                    "allow_count": len(allow),
                    "videos_after_filter": len(videos),
                })
            log_metrics(logger, "extract_keypoints.videos_discovered", {
                "count": len(videos),
                "pattern": args.pattern,
            })
            if not videos:
                log_metrics(logger, "extract_keypoints.error", {
                    "reason": "no_videos_found",
                    "in_dir": str(args.in_dir),
                    "pattern": args.pattern,
                })
                raise SystemExit(f"No videos found: {args.in_dir}/{args.pattern}")
            seen_slugs = set()
            total_items = len(videos)
            in_dir = Path(config.video.in_dir)
            for v in videos:
                if v.stem in skip_ids:
                    continue
                slug = slug_for(v, in_dir)
                if slug in seen_slugs:
                    continue
                seen_slugs.add(slug)
                sample_id = normalize_sample_id(slug)
                if args.skip_existing and writer.is_sample_committed(sample_id):
                    skipped_existing += 1
                    continue
                task_ndjson = ""
                if ndjson_base:
                    base = Path(ndjson_base)
                    task_ndjson = str(base.with_name(f"{base.stem}.{sample_id}{base.suffix}"))
                tasks.append(TaskSpec(
                    sample_id=sample_id,
                    slug=sample_id,
                    source_video=str(v),
                    config_dict=config.to_dict(),
                    ndjson_path=task_ndjson,
                ))
    return tasks, total_items, skipped_existing


def _write_failures_line(path: Optional[Path], sample_id: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{sample_id}\n")


def _error_key(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc).strip()}"


def run_pipeline(argv: list[str]) -> dict:
    return main(argv, _print_errors=False)


def main(argv: Optional[List[str]] = None, *, _print_errors: bool = True) -> Dict[str, Any]:
    out_dir: Optional[Path] = None
    ap = argparse.ArgumentParser(
        "Extract Pose+Hand landmarks to Zarr + Parquet (parallel, memory-safe)."
    )
    ap.add_argument("--in-dir", type=str, default="")
    ap.add_argument("--pattern", type=str, default="*.mp4")
    ap.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Extractor artifact root. Writes landmarks.zarr, videos.parquet, frames.parquet, and runs.parquet here.",
    )
    ap.add_argument("--task-manifest", type=str, default="",
                    help="Optional JSONL task manifest. When set, task discovery from --in-dir/--pattern is skipped.")
    ap.add_argument("--only-list", type=str, default="",
                    help="Optional path to a txt file with video names (one per line, with or without extension) to process from in-dir.")
    ap.add_argument("--segments-manifest", type=str, default="",
                    help="Process segments from a JSONL/CSV manifest (one segment per line).")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Deterministically partition selected tasks into this many shards.")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="Shard index to execute within --num-shards.")
    ap.add_argument("--write-task-manifest", type=str, default="",
                    help="Optional path to write the selected TaskSpec JSONL manifest before processing.")
    ap.add_argument("--prepare-only", action="store_true",
                    help="Only enumerate tasks and optionally write a task manifest; do not process videos.")
    ap.add_argument("--scratch-dir", type=str, default="",
                    help="Local scratch root for staging/temp files. Defaults to <out-dir>/.staging.")
    ap.add_argument("--status-path", type=str, default="",
                    help="Optional shard status JSON path.")
    ap.add_argument("--events-path", type=str, default="",
                    help="Optional structured events JSONL path.")
    ap.add_argument("--failures-path", type=str, default="",
                    help="Optional text file where failed sample ids are appended.")
    ap.add_argument("--progress-every-sec", type=float, default=15.0,
                    help="Emit progress at least this often while processing.")
    ap.add_argument("--progress-every-items", type=int, default=25,
                    help="Emit progress whenever this many new items complete.")
    ap.add_argument("--console-progress-mode", type=str, default="lines", choices=["lines", "compact"],
                    help="Render progress as one line per update or as a compact in-place status line.")
    ap.add_argument("--worker-console", action="store_true",
                    help="Allow worker processes to write logs to stderr/stdout. Off by default for cleaner parent logs.")
    ap.add_argument("--execution-mode", type=str, default="auto", choices=["auto", "cpu_pool", "gpu_single"],
                    help="Execution planner. 'auto' selects gpu_single for Tasks+GPU, otherwise cpu_pool.")
    ap.add_argument("--gpu-prefetch-frames", type=int, default=32,
                    help="Bounded frame prefetch depth for gpu_single mode.")

    ap.add_argument("--keep-pose-indices", type=str, default="0,9,10,11,12,13,14,15,16,23,24")
    ap.add_argument("--world-coords", action="store_true")
    ap.add_argument("--image-coords", action="store_true")

    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--short-side", type=int, default=0)
    ap.add_argument("--min-det", type=float, default=0.5)
    ap.add_argument("--min-track", type=float, default=0.5)
    ap.add_argument("--pose-every", type=int, default=1)
    ap.add_argument("--pose-complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--pose-ema", type=float, default=0.0)
    ap.add_argument("--min-hand-score", type=float, default=0.0)
    ap.add_argument("--hand-score-lo", type=float, default=0.55)
    ap.add_argument("--hand-score-hi", type=float, default=0.90)
    ap.add_argument("--hand-score-source", type=str, choices=["handedness", "presence"], default="handedness")
    ap.add_argument("--tracker-init-score", type=float, default=-1.0)
    ap.add_argument("--anchor-score", type=float, default=-1.0)
    ap.add_argument("--pose-dist-qual-min", type=float, default=0.50)
    ap.add_argument("--tracker-update-score", type=float, default=-1.0)
    ap.add_argument("--pose-side-reassign-ratio", type=float, default=0.85)
    ap.add_argument("--ts-source", type=str, default="auto", choices=["auto", "pos_msec", "frame_fps"])
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--ndjson",
        type=str,
        default="",
        help="Debug-only NDJSON base name. Files are written under <out-dir>/debug/ and are not part of the main output contract.",
    )
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--log-dir", type=str, default="outputs/logs",
                    help="Directory where structured kp_export logs will be written.")
    ap.add_argument("--log-level", type=str, default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    help="Verbosity for kp_export logging output.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug-video", type=str, default="")

    ap.add_argument("--mp-backend", type=str, default="solutions",
                    choices=["solutions", "tasks"],
                    help="Select MediaPipe API backend: classic solutions or Tasks")
    ap.add_argument("--hand-task", type=str, default="",
                    help="Path to hand_landmarker.task (used when --mp-backend=tasks)")
    ap.add_argument("--pose-task", type=str, default="",
                    help="Path to pose_landmarker*.task (used when --mp-backend=tasks)")
    ap.add_argument("--mp-tasks-delegate", type=str, default="cpu",
                    choices=["auto", "cpu", "gpu"],
                    help="Execution delegate for MediaPipe Tasks backend (GPU requires compatible drivers).")

    ap.add_argument("--annotations-csv", type=str, default="")
    ap.add_argument("--skip-labels", type=str, default="no_event")

    ap.add_argument("--second-pass", action="store_true")
    ap.add_argument("--sp-trigger-below", type=float, default=0.50)
    ap.add_argument("--sp-roi-frac", type=float, default=0.25)
    ap.add_argument("--sp-margin", type=float, default=0.35)
    ap.add_argument("--sp-escalate-step", type=float, default=0.25)
    ap.add_argument("--sp-escalate-max", type=float, default=2.0)
    ap.add_argument("--sp-hands-up-only", action="store_true")
    ap.add_argument("--sp-jitter-px", type=int, default=0)
    ap.add_argument("--sp-jitter-rings", type=int, default=1)
    ap.add_argument("--sp-center-penalty", type=float, default=0.30)
    ap.add_argument("--sp-label-relax", type=float, default=0.15)
    ap.add_argument("--sp-overlap-iou", type=float, default=0.15)
    ap.add_argument("--sp-overlap-shrink", type=float, default=0.70)
    ap.add_argument("--sp-overlap-penalty-mult", type=float, default=2.0)
    ap.add_argument("--sp-overlap-require-label", action="store_true")
    ap.add_argument(
        "--sp-debug-roi",
        action="store_true",
        help="Store ROI windows used by the second pass for later visualization.",
    )

    # extras
    ap.add_argument("--interp-hold", type=int, default=0)
    ap.add_argument("--write-hand-mask", action="store_true")
    ap.add_argument("--track-max-gap", type=int, default=15)
    ap.add_argument("--track-score-decay", type=float, default=0.90)
    ap.add_argument("--track-reset-ms", type=int, default=250)
    ap.add_argument("--occ-hyst-frames", type=int, default=15)
    ap.add_argument("--occ-return-k", type=float, default=1.2)
    ap.add_argument("--sanity-scale-range", type=str, default="0.70,1.35")
    ap.add_argument("--sanity-wrist-k", type=float, default=2.0)
    ap.add_argument("--sanity-bone-tol", type=float, default=0.30)
    ap.add_argument("--sanity-pass2", action="store_true")
    ap.add_argument("--sanity-anchor-max-gap", type=int, default=0)
    ap.add_argument("--no-sanity", action="store_true")
    ap.add_argument("--sanitize-rejects", dest="sanitize_rejects", action="store_true", default=True)
    ap.add_argument("--no-sanitize-rejects", dest="sanitize_rejects", action="store_false")
    ap.add_argument("--postprocess", action="store_true")
    ap.add_argument("--pp-max-gap", type=int, default=15)
    ap.add_argument("--pp-smoother", type=str, choices=["none", "ema", "rts"], default="ema")
    ap.add_argument("--pp-only-anchors", action="store_true", default=True)

    try:
        args = ap.parse_args(argv)
        log_level = (args.log_level or "INFO").upper()
        logger = configure_logging(args.log_dir or "outputs/logs", log_level)

        if args.task_manifest:
            task_manifest_path = Path(args.task_manifest)
            if not task_manifest_path.exists():
                raise SystemExit(f"Task manifest not found: {task_manifest_path}")
        elif not args.in_dir:
            raise SystemExit("Specify --in-dir unless --task-manifest is used.")
        if not args.out_dir:
            raise SystemExit("Specify --out-dir.")
        if int(args.num_shards) < 1:
            raise SystemExit("--num-shards must be >= 1")
        if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
            raise SystemExit("--shard-index must be in [0, num_shards)")

        mp_backend = (args.mp_backend or "solutions").lower()
        hand_task_path = args.hand_task or ""
        pose_task_path = args.pose_task or ""
        mp_tasks_delegate = args.mp_tasks_delegate or "auto"
        if mp_backend == "tasks":
            hand_resolved = resolve_task_model_path(hand_task_path or None, "hand_landmarker.task")
            pose_resolved = resolve_task_model_path(pose_task_path or None, "pose_landmarker_full.task")
            if hand_resolved is None:
                raise SystemExit(
                    "hand_landmarker.task not found. Place it under kp_export/mp/tasks/ or pass --hand-task"
                )
            if pose_resolved is None:
                raise SystemExit(
                    "pose_landmarker_full.task not found. Place it under kp_export/mp/tasks/ or pass --pose-task"
                )
            hand_task_path = str(hand_resolved)
            pose_task_path = str(pose_resolved)
            if str(mp_tasks_delegate).lower() == "gpu":
                _probe_gpu_delegate()

        world_coords = bool(args.world_coords) and not bool(args.image_coords)
        if not world_coords and not args.image_coords:
            raise SystemExit("Specify exactly one of --world-coords or --image-coords")

        in_dir = Path(args.in_dir)
        out_dir = Path(args.out_dir)
        _ensure_writable_dir(out_dir, "out_dir")
        scratch_dir = Path(args.scratch_dir) if args.scratch_dir else None
        if scratch_dir is not None:
            _ensure_writable_dir(scratch_dir, "scratch_dir")
        ndjson_base = _normalize_ndjson_base(out_dir, args.ndjson)

        keep = parse_keep_indices(args.keep_pose_indices)
        sanity_scale_range = _parse_range(args.sanity_scale_range)
        execution_mode = _resolve_execution_mode(
            args.execution_mode,
            mp_backend=mp_backend,
            mp_tasks_delegate=mp_tasks_delegate,
        )
        jobs = int(args.jobs)
        if execution_mode == "gpu_single" and jobs != 1:
            print("[WARN] gpu_single mode ignores --jobs and forces one extractor worker per GPU.")
            jobs = 1
        if execution_mode == "gpu_single":
            print("[WARN] gpu_single uses one in-process GPU worker; utilization may remain bursty by design.")
        if execution_mode == "gpu_single" and bool(args.second_pass):
            print("[WARN] gpu_single keeps second-pass parity enabled; GPU utilization may still be bursty by design.")
        args.jobs = jobs
        args.execution_mode = execution_mode
        args.gpu_prefetch_frames = max(1, int(args.gpu_prefetch_frames))

        config = ExtractorConfig(
            video=VideoConfig(
                in_dir=str(in_dir),
                out_dir=str(out_dir),
                pattern=str(args.pattern),
                stride=max(1, args.stride),
                short_side=int(args.short_side),
                ts_source=str(args.ts_source),
            ),
            pose=PoseConfig(
                keep_pose_indices=keep,
                world_coords=bool(world_coords),
                pose_every=max(1, args.pose_every),
                pose_complexity=int(args.pose_complexity),
                pose_ema=float(args.pose_ema),
            ),
            score=ScoreConfig(
                min_det=float(args.min_det),
                min_track=float(args.min_track),
                min_hand_score=float(args.min_hand_score),
                hand_score_lo=float(args.hand_score_lo),
                hand_score_hi=float(args.hand_score_hi),
                hand_score_source=str(args.hand_score_source),
                tracker_init_score=float(args.tracker_init_score),
                anchor_score=float(args.anchor_score),
                tracker_update_score=float(args.tracker_update_score),
                pose_dist_qual_min=float(args.pose_dist_qual_min),
                pose_side_reassign_ratio=float(args.pose_side_reassign_ratio),
            ),
            second_pass=SecondPassConfig(
                enabled=bool(args.second_pass),
                trigger_below=float(args.sp_trigger_below),
                roi_frac=float(args.sp_roi_frac),
                margin=float(args.sp_margin),
                escalate_step=float(args.sp_escalate_step),
                escalate_max=float(args.sp_escalate_max),
                hands_up_only=bool(args.sp_hands_up_only),
                jitter_px=int(args.sp_jitter_px),
                jitter_rings=int(args.sp_jitter_rings),
                center_penalty=float(args.sp_center_penalty),
                label_relax=float(args.sp_label_relax),
                overlap_iou=float(args.sp_overlap_iou),
                overlap_shrink=float(args.sp_overlap_shrink),
                overlap_penalty_mult=float(args.sp_overlap_penalty_mult),
                overlap_require_label=bool(args.sp_overlap_require_label),
                debug_roi=bool(args.sp_debug_roi),
            ),
            tracking=TrackingConfig(
                interp_hold=int(args.interp_hold),
                write_hand_mask=bool(args.write_hand_mask),
                max_gap=int(args.track_max_gap),
                score_decay=float(args.track_score_decay),
                reset_ms=int(args.track_reset_ms),
            ),
            occlusion=OcclusionConfig(
                hyst_frames=int(args.occ_hyst_frames),
                return_k=float(args.occ_return_k),
            ),
            sanity=SanityConfig(
                enabled=not bool(args.no_sanity),
                scale_range=sanity_scale_range,
                wrist_k=float(args.sanity_wrist_k),
                bone_tol=float(args.sanity_bone_tol),
                pass2=bool(args.sanity_pass2),
                anchor_max_gap=int(args.sanity_anchor_max_gap),
                sanitize_rejects=bool(args.sanitize_rejects),
            ),
            postprocess=PostprocessConfig(
                enabled=bool(args.postprocess),
                max_gap=int(args.pp_max_gap),
                smoother=str(args.pp_smoother),
                only_anchors=bool(args.pp_only_anchors),
            ),
            mediapipe=MediaPipeConfig(
                backend=str(mp_backend),
                hand_task=str(hand_task_path),
                pose_task=str(pose_task_path),
                tasks_delegate=str(mp_tasks_delegate),
            ),
            debug=DebugConfig(
                ndjson=str(ndjson_base),
                debug_video=str(args.debug_video) if args.debug_video else "",
            ),
            output=OutputConfig(stage_dir=""),
            runtime=RuntimeConfig(
                jobs=int(jobs),
                seed=int(args.seed),
                video_count=1,
                execution_mode=str(execution_mode),
                gpu_prefetch_frames=int(args.gpu_prefetch_frames),
            ),
            logging=LoggingConfig(
                log_dir=str(Path(args.log_dir or "outputs/logs")),
                log_level=str(log_level),
                worker_console=bool(args.worker_console),
            ),
        )

        skip_labels = [s.strip().lower() for s in (args.skip_labels or "").split(",") if s.strip()]
        skip_ids = set()
        if args.annotations_csv:
            skip_ids = load_skip_ids(Path(args.annotations_csv), skip_labels)
            log_metrics(logger, "extract_keypoints.skip_ids_loaded", {
                "from": str(args.annotations_csv),
                "labels": ",".join(skip_labels),
                "skip_count": len(skip_ids),
            })

        versions = _versions_snapshot()
        run_id = uuid.uuid4().hex
        try:
            from kp_export.output.writer import ExtractorOutputWriter
        except Exception as exc:
            raise SystemExit(
                "Zarr/Parquet output dependencies are missing. Install 'zarr', 'numcodecs', and 'pyarrow'."
            ) from exc

        writer = ExtractorOutputWriter(
            out_dir=out_dir,
            scratch_dir=scratch_dir,
            run_id=run_id,
            args_snapshot=vars(args),
            versions=versions,
        )
        config = config.with_stage_dir(
            str(writer.current_run_dir.resolve()),
            scratch_dir=str(scratch_dir.resolve()) if scratch_dir is not None else "",
        )

        tasks, total_items, skipped_existing = _build_tasks(
            args=args,
            config=config,
            writer=writer,
            skip_ids=skip_ids,
            logger=logger,
            ndjson_base=ndjson_base,
        )

        if not tasks and skipped_existing <= 0:
            log_metrics(logger, "extract_keypoints.error", {
                "reason": "no_tasks",
                "segments_mode": bool(args.segments_manifest),
            })
            raise SystemExit("No videos/segments to process after filtering.")

        if int(args.num_shards) > 1:
            pre_filter_count = len(tasks)
            tasks = filter_tasks_for_shard(
                tasks,
                num_shards=int(args.num_shards),
                shard_index=int(args.shard_index),
            )
            log_metrics(logger, "extract_keypoints.shard_filter", {
                "before": int(pre_filter_count),
                "after": int(len(tasks)),
                "num_shards": int(args.num_shards),
                "shard_index": int(args.shard_index),
            })

        if args.write_task_manifest:
            write_task_manifest(args.write_task_manifest, tasks)
            log_metrics(logger, "extract_keypoints.task_manifest_written", {
                "path": str(args.write_task_manifest),
                "tasks": len(tasks),
            })
            print(f"[OK] Task manifest written to {args.write_task_manifest}")
            if args.prepare_only:
                return {
                    "ok": True,
                    "out_dir": str(out_dir),
                    "task_manifest_path": str(args.write_task_manifest),
                    "selected": int(total_items),
                    "scheduled": int(len(tasks)),
                    "skipped_existing": int(skipped_existing),
                    "code": 0,
                }
        elif args.prepare_only:
            raise SystemExit("--prepare-only requires --write-task-manifest.")

        config = config.with_video_count(len(tasks))
        tasks = [
            TaskSpec(
                sample_id=task.sample_id,
                slug=task.slug,
                source_video=task.source_video,
                config_dict=config.to_dict(),
                frame_start=task.frame_start,
                frame_end=task.frame_end,
                segment_meta=dict(task.segment_meta),
                debug_video_path=task.debug_video_path,
                ndjson_path=task.ndjson_path,
            )
            for task in tasks
        ]

        log_metrics(logger, "extract_keypoints.tasks_prepared", {
            "scheduled": len(tasks),
            "skipped": max(0, total_items - len(tasks)),
            "skipped_existing": int(skipped_existing),
            "skip_existing": bool(args.skip_existing),
            "segments_mode": bool(args.segments_manifest),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "execution_mode": str(execution_mode),
        })
        print(
            "[INFO] "
            f"selected={total_items} scheduled={len(tasks)} "
            f"skipped_existing={int(skipped_existing)}"
        )

        failures_path = Path(args.failures_path) if args.failures_path else None
        reporter = ShardStatusReporter(
            run_id=run_id,
            total=len(tasks),
            shard_index=int(args.shard_index),
            num_shards=int(args.num_shards),
            status_path=args.status_path,
            events_path=args.events_path,
            failures_path=args.failures_path,
            pod_id=os.environ.get("RUNPOD_POD_ID", ""),
            gpu_count=int(os.environ["RUNPOD_GPU_COUNT"]) if os.environ.get("RUNPOD_GPU_COUNT", "").isdigit() else None,
        )
        reporter.emit_event(
            "run_started",
            selected=int(total_items),
            scheduled=int(len(tasks)),
            skipped_existing=int(skipped_existing),
            out_dir=str(out_dir),
        )
        reporter.update(
            state="running",
            processed=0,
            failed=0,
            remaining=len(tasks),
        )
        visible_gpu_count = _detect_visible_gpu_count() if execution_mode == "gpu_single" else None
        capability_payload = {
            "mp_backend": str(mp_backend),
            "mp_tasks_delegate": str(mp_tasks_delegate),
            "execution_mode": str(execution_mode),
            "visible_gpus": int(visible_gpu_count) if visible_gpu_count is not None else None,
            "worker_count": int(jobs),
            "gpu_prefetch_frames": int(config.runtime.gpu_prefetch_frames) if execution_mode == "gpu_single" else 0,
        }
        log_metrics(logger, "extract_keypoints.capabilities", capability_payload)
        reporter.emit_event("run_capabilities", **capability_payload)
        print(
            "[INFO] "
            f"backend={mp_backend} delegate={mp_tasks_delegate} mode={execution_mode} "
            f"visible_gpus={visible_gpu_count if visible_gpu_count is not None else '?'} "
            f"workers={jobs} prefetch={capability_payload['gpu_prefetch_frames']}"
        )

        processed_rows: List[Dict[str, Any]] = []
        failed_count = 0
        error_counts: Dict[str, int] = {}
        last_sample_id = ""
        progress_started_at = perf_counter()
        progress_state: Dict[str, Any] = {"last_log_at": progress_started_at, "last_done": 0}
        gpu_process_kwargs = config.to_process_video_kwargs() if execution_mode == "gpu_single" else None
        runtime_totals: Dict[str, float] = {
            "processing_elapsed": 0.0,
            "decode_runtime": 0.0,
            "detector_init_runtime": 0.0,
            "hand_runtime": 0.0,
            "pose_runtime": 0.0,
            "second_pass_runtime": 0.0,
            "writer_runtime": 0.0,
        }
        native_stderr_path = writer.current_run_dir / "gpu_native_stderr.log" if execution_mode == "gpu_single" else None
        with track_runtime(logger, "extract_keypoints.processing", videos=len(tasks), jobs=jobs):
            stderr_context = _redirect_native_stderr(native_stderr_path) if execution_mode == "gpu_single" else nullcontext()
            with stderr_context:
                if execution_mode == "gpu_single" and tasks:
                    gpu_session: Optional[MediaPipeGpuSession] = None
                    try:
                        mp, mp_solutions = try_import_mediapipe()
                        warmup_started = perf_counter()
                        gpu_session = MediaPipeGpuSession(
                            backend=str(mp_backend),
                            mp=mp,
                            mp_solutions=mp_solutions,
                            hand_model_path=Path(hand_task_path) if hand_task_path else None,
                            pose_model_path=Path(pose_task_path) if pose_task_path else None,
                            min_det=float(args.min_det),
                            min_track=float(args.min_track),
                            pose_complexity=int(args.pose_complexity),
                            tasks_delegate=str(mp_tasks_delegate),
                            second_pass=bool(args.second_pass),
                            world_coords=bool(world_coords),
                        )
                        gpu_session.warmup()
                        warmup_elapsed = perf_counter() - warmup_started
                        runtime_totals["detector_init_runtime"] += warmup_elapsed
                        reporter.emit_event(
                            "gpu_session_ready",
                            warmup_sec=round(warmup_elapsed, 3),
                            prefetch_frames=int(config.runtime.gpu_prefetch_frames),
                        )
                        for t in tasks:
                            try:
                                sample_payload = process_video(
                                    path=t.source_path,
                                    sample_id=str(t.sample_id),
                                    ndjson_path=Path(t.ndjson_path) if t.ndjson_path else None,
                                    debug_video_path=Path(t.debug_video_path) if t.debug_video_path else None,
                                    frame_start=t.frame_start,
                                    frame_end=t.frame_end,
                                    segment_meta=(t.segment_meta or None),
                                    detector_session=gpu_session,
                                    **dict(gpu_process_kwargs or {}),
                                )
                                writer_started = perf_counter()
                                video_row = writer.commit_payload(sample_payload)
                                writer_elapsed = perf_counter() - writer_started
                                processed_rows.append(video_row)
                                last_sample_id = str(t.sample_id)
                                runtime_metrics = dict(sample_payload.runtime_metrics or {})
                                runtime_metrics["writer_runtime"] = writer_elapsed
                                _accumulate_runtime_metrics(runtime_totals, runtime_metrics)
                                reporter.emit_event(
                                    "sample_committed",
                                    sample_id=str(t.sample_id),
                                    source_video=str(t.source_video),
                                    num_frames=int(video_row.get("num_frames", 0) or 0),
                                )
                            except Exception as e:
                                failed_count += 1
                                last_sample_id = str(t.sample_id)
                                key = _error_key(e)
                                error_counts[key] = error_counts.get(key, 0) + 1
                                _write_failures_line(failures_path, t.sample_id)
                                reporter.note_failure(t.sample_id, key)
                                if error_counts[key] == 1:
                                    print(f"[ERROR] {t.sample_id}: {e}")
                                elif error_counts[key] in {10, 50, 100}:
                                    print(f"[ERROR] repeated {error_counts[key]}x: {e}")
                                log_metrics(logger, "extract_keypoints.worker_error", {
                                    "error": str(e),
                                    "sample_id": str(t.sample_id),
                                    "error_count": int(error_counts[key]),
                                    "execution_mode": "gpu_single",
                                })
                            finally:
                                _maybe_report_progress(
                                    logger,
                                    started_at=progress_started_at,
                                    total=len(tasks),
                                    ok_count=len(processed_rows),
                                    failed_count=failed_count,
                                    state=progress_state,
                                    reporter=reporter,
                                    console_mode=str(args.console_progress_mode),
                                    progress_every_items=int(args.progress_every_items),
                                    progress_every_sec=float(args.progress_every_sec),
                                    last_sample_id=last_sample_id,
                                )
                    except KeyboardInterrupt:
                        print("\n[INTERRUPTED] Stopping GPU extraction...")
                        log_metrics(logger, "extract_keypoints.interrupted", {
                            "jobs": 1,
                            "processed_so_far": len(processed_rows),
                            "failed_so_far": int(failed_count),
                            "execution_mode": "gpu_single",
                        })
                        reporter.emit_event("run_interrupted", processed=len(processed_rows), failed=int(failed_count))
                        reporter.update(
                            state="interrupted",
                            processed=len(processed_rows),
                            failed=failed_count,
                            remaining=max(0, len(tasks) - len(processed_rows) - failed_count),
                            last_sample_id=last_sample_id,
                        )
                        raise SystemExit(130)
                    finally:
                        if gpu_session is not None:
                            gpu_session.close()
                elif jobs > 1 and tasks:
                    ex: Optional[ProcessPoolExecutor] = None
                    interrupted = False
                    broken_pool = False
                    try:
                        ex = ProcessPoolExecutor(max_workers=jobs)
                        fut_to_task = {ex.submit(process_worker, t.to_payload()): t for t in tasks}
                        for fut in as_completed(fut_to_task):
                            task = fut_to_task[fut]
                            try:
                                result = fut.result()
                                writer_started = perf_counter()
                                video_row = writer.commit_staged_sample(result["staged_path"])
                                writer_elapsed = perf_counter() - writer_started
                                processed_rows.append(video_row)
                                last_sample_id = str(task.sample_id)
                                runtime_metrics = dict(result.get("runtime_metrics") or {})
                                runtime_metrics["writer_runtime"] = writer_elapsed
                                _accumulate_runtime_metrics(runtime_totals, runtime_metrics)
                                reporter.emit_event(
                                    "sample_committed",
                                    sample_id=str(task.sample_id),
                                    source_video=str(task.source_video),
                                    num_frames=int(video_row.get("num_frames", 0) or 0),
                                )
                            except BrokenProcessPool as e:
                                failed_count += 1
                                broken_pool = True
                                last_sample_id = str(task.sample_id)
                                key = _error_key(e)
                                error_counts[key] = error_counts.get(key, 0) + 1
                                _write_failures_line(failures_path, task.sample_id)
                                reporter.note_failure(task.sample_id, key)
                                print(f"[ERROR] Worker pool crashed: {e}")
                                log_metrics(logger, "extract_keypoints.worker_pool_broken", {
                                    "error": str(e),
                                    "sample_id": str(task.sample_id),
                                })
                                if ex is not None:
                                    _shutdown_process_pool_fast(ex)
                                break
                            except Exception as e:
                                failed_count += 1
                                last_sample_id = str(task.sample_id)
                                key = _error_key(e)
                                error_counts[key] = error_counts.get(key, 0) + 1
                                _write_failures_line(failures_path, task.sample_id)
                                reporter.note_failure(task.sample_id, key)
                                if error_counts[key] == 1:
                                    print(f"[ERROR] {task.sample_id}: {e}")
                                elif error_counts[key] in {10, 50, 100}:
                                    print(f"[ERROR] repeated {error_counts[key]}x: {e}")
                                log_metrics(logger, "extract_keypoints.worker_error", {
                                    "error": str(e),
                                    "sample_id": str(task.sample_id),
                                    "error_count": int(error_counts[key]),
                                })
                            finally:
                                _maybe_report_progress(
                                    logger,
                                    started_at=progress_started_at,
                                    total=len(tasks),
                                    ok_count=len(processed_rows),
                                    failed_count=failed_count,
                                    state=progress_state,
                                    reporter=reporter,
                                    console_mode=str(args.console_progress_mode),
                                    progress_every_items=int(args.progress_every_items),
                                    progress_every_sec=float(args.progress_every_sec),
                                    last_sample_id=last_sample_id,
                                )
                        if broken_pool:
                            reporter.update(
                                state="failed",
                                processed=len(processed_rows),
                                failed=failed_count,
                                remaining=max(0, len(tasks) - len(processed_rows) - failed_count),
                                last_sample_id=last_sample_id,
                            )
                    except KeyboardInterrupt:
                        interrupted = True
                        print("\n[INTERRUPTED] Stopping workers...")
                        log_metrics(logger, "extract_keypoints.interrupted", {
                            "jobs": int(jobs),
                            "processed_so_far": len(processed_rows),
                            "failed_so_far": int(failed_count),
                        })
                        reporter.emit_event("run_interrupted", processed=len(processed_rows), failed=int(failed_count))
                        reporter.update(
                            state="interrupted",
                            processed=len(processed_rows),
                            failed=failed_count,
                            remaining=max(0, len(tasks) - len(processed_rows) - failed_count),
                            last_sample_id=last_sample_id,
                        )
                        if ex is not None:
                            _shutdown_process_pool_fast(ex)
                        raise SystemExit(130)
                    finally:
                        if ex is not None and not interrupted:
                            ex.shutdown(wait=True, cancel_futures=False)
                else:
                    try:
                        for t in tasks:
                            try:
                                result = process_worker(t.to_payload())
                                writer_started = perf_counter()
                                video_row = writer.commit_staged_sample(result["staged_path"])
                                writer_elapsed = perf_counter() - writer_started
                                processed_rows.append(video_row)
                                last_sample_id = str(t.sample_id)
                                runtime_metrics = dict(result.get("runtime_metrics") or {})
                                runtime_metrics["writer_runtime"] = writer_elapsed
                                _accumulate_runtime_metrics(runtime_totals, runtime_metrics)
                                reporter.emit_event(
                                    "sample_committed",
                                    sample_id=str(t.sample_id),
                                    source_video=str(t.source_video),
                                    num_frames=int(video_row.get("num_frames", 0) or 0),
                                )
                            except Exception as e:
                                failed_count += 1
                                last_sample_id = str(t.sample_id)
                                key = _error_key(e)
                                error_counts[key] = error_counts.get(key, 0) + 1
                                _write_failures_line(failures_path, t.sample_id)
                                reporter.note_failure(t.sample_id, key)
                                if error_counts[key] == 1:
                                    print(f"[ERROR] {t.sample_id}: {e}")
                                elif error_counts[key] in {10, 50, 100}:
                                    print(f"[ERROR] repeated {error_counts[key]}x: {e}")
                                log_metrics(logger, "extract_keypoints.worker_error", {
                                    "error": str(e),
                                    "sample_id": str(t.sample_id),
                                    "error_count": int(error_counts[key]),
                                })
                            finally:
                                _maybe_report_progress(
                                    logger,
                                    started_at=progress_started_at,
                                    total=len(tasks),
                                    ok_count=len(processed_rows),
                                    failed_count=failed_count,
                                    state=progress_state,
                                    reporter=reporter,
                                    console_mode=str(args.console_progress_mode),
                                    progress_every_items=int(args.progress_every_items),
                                    progress_every_sec=float(args.progress_every_sec),
                                    last_sample_id=last_sample_id,
                                )
                    except KeyboardInterrupt:
                        print("\n[INTERRUPTED] Stopping...")
                        log_metrics(logger, "extract_keypoints.interrupted", {
                            "jobs": 1,
                            "processed_so_far": len(processed_rows),
                            "failed_so_far": int(failed_count),
                        })
                        reporter.emit_event("run_interrupted", processed=len(processed_rows), failed=int(failed_count))
                        reporter.update(
                            state="interrupted",
                            processed=len(processed_rows),
                            failed=failed_count,
                            remaining=max(0, len(tasks) - len(processed_rows) - failed_count),
                            last_sample_id=last_sample_id,
                        )
                        raise SystemExit(130)

        _maybe_report_progress(
            logger,
            started_at=progress_started_at,
            total=len(tasks),
            ok_count=len(processed_rows),
            failed_count=failed_count,
            state=progress_state,
            reporter=reporter,
            console_mode=str(args.console_progress_mode),
            progress_every_items=int(args.progress_every_items),
            progress_every_sec=float(args.progress_every_sec),
            last_sample_id=last_sample_id,
            force=True,
        )

        aggregate = _aggregate_video_rows(processed_rows)
        output_paths = writer.finalize(
            scheduled_count=len(tasks),
            skipped_count=int(skipped_existing),
            failed_count=int(failed_count),
            processed_count=len(processed_rows),
            segments_mode=bool(args.segments_manifest),
            jobs=int(jobs),
            seed=int(args.seed),
            mp_backend=str(mp_backend),
            aggregate=aggregate,
        )
        _emit_runtime_summary(
            logger,
            execution_mode=str(execution_mode),
            processed_rows=processed_rows,
            runtime_totals=runtime_totals,
            started_at=progress_started_at,
            native_warnings=_summarize_native_stderr(native_stderr_path),
        )

        log_metrics(logger, "extract_keypoints.completed", {
            "videos_processed": len(processed_rows),
            "videos_failed": int(failed_count),
            "videos_skipped_existing": int(skipped_existing),
            "jobs_used": jobs,
            "execution_mode": str(execution_mode),
            "zarr_path": output_paths["zarr_path"],
            "videos_parquet_path": output_paths["videos_parquet_path"],
            "frames_parquet_path": output_paths["frames_parquet_path"],
            "runs_parquet_path": output_paths["runs_parquet_path"],
        })
        reporter.emit_event(
            "run_completed",
            processed=len(processed_rows),
            failed=int(failed_count),
            zarr_path=output_paths["zarr_path"],
            videos_parquet_path=output_paths["videos_parquet_path"],
            frames_parquet_path=output_paths["frames_parquet_path"],
            runs_parquet_path=output_paths["runs_parquet_path"],
        )
        reporter.update(
            state="completed" if failed_count == 0 else "completed_with_failures",
            processed=len(processed_rows),
            failed=failed_count,
            remaining=0,
            last_sample_id=last_sample_id,
        )
        print(f"[OK] Zarr written to {output_paths['zarr_path']}")
        print(f"[OK] Videos parquet written to {output_paths['videos_parquet_path']}")
        print(f"[OK] Frames parquet written to {output_paths['frames_parquet_path']}")
        print(f"[OK] Runs parquet written to {output_paths['runs_parquet_path']}")
        return {
            "ok": True,
            "out_dir": str(out_dir),
            "zarr_path": output_paths["zarr_path"],
            "videos_parquet_path": output_paths["videos_parquet_path"],
            "frames_parquet_path": output_paths["frames_parquet_path"],
            "runs_parquet_path": output_paths["runs_parquet_path"],
            "code": 0 if failed_count == 0 else 1,
        }
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        error = None
        if isinstance(exc.code, int):
            if exc.code not in (0, 2):
                error = str(exc)
        else:
            error = str(exc.code)
        if _print_errors and error:
            print(error, file=sys.stderr)
        return {
            "ok": False,
            "out_dir": str(out_dir) if out_dir else "",
            "zarr_path": str((out_dir / "landmarks.zarr")) if out_dir else "",
            "videos_parquet_path": str((out_dir / "videos.parquet")) if out_dir else "",
            "frames_parquet_path": str((out_dir / "frames.parquet")) if out_dir else "",
            "runs_parquet_path": str((out_dir / "runs.parquet")) if out_dir else "",
            "error": error,
            "code": int(code),
        }
    except Exception as exc:
        if _print_errors:
            traceback.print_exc()
        return {
            "ok": False,
            "out_dir": str(out_dir) if out_dir else "",
            "zarr_path": str((out_dir / "landmarks.zarr")) if out_dir else "",
            "videos_parquet_path": str((out_dir / "videos.parquet")) if out_dir else "",
            "frames_parquet_path": str((out_dir / "frames.parquet")) if out_dir else "",
            "runs_parquet_path": str((out_dir / "runs.parquet")) if out_dir else "",
            "error": str(exc),
            "code": 1,
        }


if __name__ == "__main__":
    # Windows-safe spawn
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    result = main()
    raise SystemExit(result.get("code", 1))
