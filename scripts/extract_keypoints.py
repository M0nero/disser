from __future__ import annotations
import argparse, csv, json, sys, traceback, uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from time import perf_counter

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
from kp_export.mp.mp_utils import resolve_task_model_path
from kp_export.core.logging_utils import configure_logging, log_metrics, track_runtime
from kp_export.tasks import TaskSpec
from concurrent.futures import ProcessPoolExecutor, as_completed


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
    force: bool = False,
) -> None:
    done_count = int(ok_count) + int(failed_count)
    if total <= 0 or done_count <= 0:
        return

    now = perf_counter()
    last_log_at = float(state.get("last_log_at", started_at))
    last_done = int(state.get("last_done", 0))
    done_delta = done_count - last_done
    should_log = force or done_count == 1 or done_count == total or done_delta >= 25 or (now - last_log_at) >= 15.0
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
    }
    log_metrics(logger, "extract_keypoints.progress", payload)
    print(
        "[PROGRESS] "
        f"{done_count}/{total} "
        f"| ok={ok_count} failed={failed_count} remaining={remaining} "
        f"| elapsed={_format_duration_hms(elapsed)} eta={_format_duration_hms(eta_sec)}"
    )
    state["last_log_at"] = now
    state["last_done"] = done_count


def run_pipeline(argv: list[str]) -> dict:
    return main(argv, _print_errors=False)


def main(argv: Optional[List[str]] = None, *, _print_errors: bool = True) -> Dict[str, Any]:
    out_dir: Optional[Path] = None
    ap = argparse.ArgumentParser(
        "Extract Pose+Hand landmarks to Zarr + Parquet (parallel, memory-safe)."
    )
    ap.add_argument("--in-dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="*.mp4")
    ap.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Extractor artifact root. Writes landmarks.zarr, videos.parquet, frames.parquet, and runs.parquet here.",
    )
    ap.add_argument("--only-list", type=str, default="",
                    help="Optional path to a txt file with video names (one per line, with or without extension) to process from in-dir.")
    ap.add_argument("--segments-manifest", type=str, default="",
                    help="Process segments from a JSONL/CSV manifest (one segment per line).")

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

        world_coords = bool(args.world_coords) and not bool(args.image_coords)
        if not world_coords and not args.image_coords:
            raise SystemExit("Specify exactly one of --world-coords or --image-coords")

        in_dir = Path(args.in_dir)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ndjson_base = _normalize_ndjson_base(out_dir, args.ndjson)

        keep = parse_keep_indices(args.keep_pose_indices)
        sanity_scale_range = _parse_range(args.sanity_scale_range)

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
                jobs=int(args.jobs),
                seed=int(args.seed),
                video_count=1,
            ),
            logging=LoggingConfig(
                log_dir=str(Path(args.log_dir or "outputs/logs")),
                log_level=str(log_level),
            ),
        )

        segments_manifest = Path(args.segments_manifest) if args.segments_manifest else None

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
            run_id=run_id,
            args_snapshot=vars(args),
            versions=versions,
        )
        config = config.with_stage_dir(str(writer.current_run_dir.resolve()))

        tasks: List[TaskSpec] = []
        total_items = 0
        skipped_existing = 0
        if segments_manifest:
            rows = _read_segments_manifest(segments_manifest)
            segments = _parse_segment_rows(rows)
            total_items = len(segments)
            video_files, by_name, by_stem = _index_videos_by_id(in_dir)

            log_metrics(logger, "extract_keypoints.segments_manifest_loaded", {
                "manifest": str(segments_manifest),
                "segments": len(segments),
            })
            log_metrics(logger, "extract_keypoints.segment_videos_indexed", {
                "in_dir": str(in_dir),
                "files": len(video_files),
            })

            allow = None
            if args.only_list:
                allow_path = Path(args.only_list)
                allow = set()
                for line in allow_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    allow.add(Path(line).stem)
                log_metrics(logger, "extract_keypoints.segment_allowlist_loaded", {
                    "from": str(allow_path),
                    "allow_count": len(allow),
                })

            skip_ids_lower = {s.lower() for s in skip_ids}
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
                    print(f"[SKIP by label] {video_id} ({seg_uid})")
                    continue
                if seg_uid in seen_seg_uids:
                    print(f"[WARN] duplicate seg_uid {seg_uid}, skipping duplicate")
                    continue
                vpath, dup_type = _resolve_video_path(video_id, by_name, by_stem)
                if vpath is None:
                    missing_videos.append(video_id)
                    continue
                if dup_type:
                    dup_video_matches += 1
                    print(f"[WARN] multiple video matches by {dup_type} for {video_id}, using {vpath}")
                sample_id = normalize_sample_id(seg_uid)
                if sample_id in seen_sample_ids:
                    print(f"[WARN] duplicate normalized sample_id {sample_id} from seg_uid {seg_uid}, skipping duplicate")
                    continue
                if args.skip_existing and writer.is_sample_committed(sample_id):
                    print(f"[SKIP] {sample_id}")
                    skipped_existing += 1
                    continue
                seen_seg_uids.add(seg_uid)
                seen_sample_ids.add(sample_id)
                ndjson_path = ""
                if config.debug.ndjson:
                    base = Path(config.debug.ndjson)
                    ndjson_path = str(base.with_name(f"{base.stem}.{sample_id}{base.suffix}"))
                tasks.append(TaskSpec(
                    sample_id=sample_id,
                    slug=sample_id,
                    source_video=str(vpath),
                    config_dict=config.to_dict(),
                    frame_start=seg.get("start"),
                    frame_end=seg.get("end"),
                    segment_meta=dict(seg),
                    ndjson_path=ndjson_path,
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
            # Discover input videos
            videos = sorted(in_dir.glob(args.pattern))

            # Optional whitelist by names/stems
            if args.only_list:
                allow_path = Path(args.only_list)
                allow = set()
                for line in allow_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    allow.add(Path(line).stem)
                videos = [v for v in videos if v.stem in allow]
                log_metrics(logger, "extract_keypoints.whitelist_loaded", {
                    "from": str(allow_path),
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
                    "in_dir": str(in_dir),
                    "pattern": args.pattern,
                })
                raise SystemExit(f"No videos found: {in_dir}/{args.pattern}")

            seen_slugs = set()
            for v in videos:
                if v.stem in skip_ids:
                    print(f"[SKIP by label] {v.name}")
                    continue
                slug = slug_for(v, in_dir)
                if slug in seen_slugs:
                    print(f"[WARN] duplicate slug for {v} -> {slug}, skipping duplicate")
                    continue
                seen_slugs.add(slug)
                sample_id = normalize_sample_id(slug)
                if args.skip_existing and writer.is_sample_committed(sample_id):
                    print(f"[SKIP] {sample_id}")
                    skipped_existing += 1
                    continue
                ndjson_path = ""
                if config.debug.ndjson:
                    base = Path(config.debug.ndjson)
                    ndjson_path = str(base.with_name(f"{base.stem}.{sample_id}{base.suffix}"))
                tasks.append(TaskSpec(
                    sample_id=sample_id,
                    slug=sample_id,
                    source_video=str(v),
                    config_dict=config.to_dict(),
                    ndjson_path=ndjson_path,
                ))

            total_items = len(videos)

        if not tasks and skipped_existing <= 0:
            log_metrics(logger, "extract_keypoints.error", {
                "reason": "no_tasks",
                "segments_mode": bool(segments_manifest),
            })
            raise SystemExit("No videos/segments to process after filtering.")

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
            "segments_mode": bool(segments_manifest),
        })
        print(
            "[INFO] "
            f"selected={total_items} scheduled={len(tasks)} "
            f"skipped_existing={int(skipped_existing)}"
        )

        processed_rows: List[Dict[str, Any]] = []
        failed_count = 0
        progress_started_at = perf_counter()
        progress_state: Dict[str, Any] = {"last_log_at": progress_started_at, "last_done": 0}
        with track_runtime(logger, "extract_keypoints.processing", videos=len(tasks), jobs=args.jobs):
            if args.jobs > 1 and tasks:
                ex: Optional[ProcessPoolExecutor] = None
                interrupted = False
                try:
                    ex = ProcessPoolExecutor(max_workers=args.jobs)
                    futs = [ex.submit(process_worker, t.to_payload()) for t in tasks]
                    for fut in as_completed(futs):
                        try:
                            result = fut.result()
                            video_row = writer.commit_staged_sample(result["staged_path"])
                            processed_rows.append(video_row)
                        except Exception as e:
                            failed_count += 1
                            print(f"[ERROR] {e}")
                            log_metrics(logger, "extract_keypoints.worker_error", {"error": str(e)})
                        finally:
                            _maybe_report_progress(
                                logger,
                                started_at=progress_started_at,
                                total=len(tasks),
                                ok_count=len(processed_rows),
                                failed_count=failed_count,
                                state=progress_state,
                            )
                except KeyboardInterrupt:
                    interrupted = True
                    print("\n[INTERRUPTED] Stopping workers...")
                    log_metrics(logger, "extract_keypoints.interrupted", {
                        "jobs": int(args.jobs),
                        "processed_so_far": len(processed_rows),
                        "failed_so_far": int(failed_count),
                    })
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
                            video_row = writer.commit_staged_sample(result["staged_path"])
                            processed_rows.append(video_row)
                        except Exception as e:
                            failed_count += 1
                            print(f"[ERROR] {e}")
                            log_metrics(logger, "extract_keypoints.worker_error", {"error": str(e)})
                        finally:
                            _maybe_report_progress(
                                logger,
                                started_at=progress_started_at,
                                total=len(tasks),
                                ok_count=len(processed_rows),
                                failed_count=failed_count,
                                state=progress_state,
                            )
                except KeyboardInterrupt:
                    print("\n[INTERRUPTED] Stopping...")
                    log_metrics(logger, "extract_keypoints.interrupted", {
                        "jobs": 1,
                        "processed_so_far": len(processed_rows),
                        "failed_so_far": int(failed_count),
                    })
                    raise SystemExit(130)

        _maybe_report_progress(
            logger,
            started_at=progress_started_at,
            total=len(tasks),
            ok_count=len(processed_rows),
            failed_count=failed_count,
            state=progress_state,
            force=True,
        )

        aggregate = _aggregate_video_rows(processed_rows)
        output_paths = writer.finalize(
            scheduled_count=len(tasks),
            skipped_count=int(skipped_existing),
            failed_count=int(failed_count),
            processed_count=len(processed_rows),
            segments_mode=bool(segments_manifest),
            jobs=int(args.jobs),
            seed=int(args.seed),
            mp_backend=str(mp_backend),
            aggregate=aggregate,
        )

        log_metrics(logger, "extract_keypoints.completed", {
            "videos_processed": len(processed_rows),
            "videos_failed": int(failed_count),
            "videos_skipped_existing": int(skipped_existing),
            "jobs_used": args.jobs,
            "zarr_path": output_paths["zarr_path"],
            "videos_parquet_path": output_paths["videos_parquet_path"],
            "frames_parquet_path": output_paths["frames_parquet_path"],
            "runs_parquet_path": output_paths["runs_parquet_path"],
        })
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
