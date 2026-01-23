from __future__ import annotations
import argparse, csv, json, sys, traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kp_export.core.utils import parse_keep_indices
from kp_export.annotations import load_skip_ids
from kp_export.core.io_utils import combine_to_single_json
from kp_export.parallel import process_worker, slug_for
from kp_export.mp.mp_utils import resolve_task_model_path
from kp_export.core.logging_utils import configure_logging, log_metrics, track_runtime
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


def run_pipeline(argv: list[str]) -> dict:
    return main(argv, _print_errors=False)


def main(argv: Optional[List[str]] = None, *, _print_errors: bool = True) -> Dict[str, Any]:
    out_dir: Optional[Path] = None
    manifest_path: Optional[Path] = None
    eval_report_path = ""
    ap = argparse.ArgumentParser(
        "Extract Pose+Hand landmarks to JSON (parallel, memory-safe)."
    )
    ap.add_argument("--in-dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="*.mp4")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--combined-json", type=str, default="")
    ap.add_argument("--combined-with-meta", action="store_true")
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
    ap.add_argument("--ndjson", type=str, default="")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--log-dir", type=str, default="outputs/logs",
                    help="Directory where structured kp_export logs will be written.")
    ap.add_argument("--log-level", type=str, default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    help="Verbosity for kp_export logging output.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-report", type=str, default="")
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

        keep = parse_keep_indices(args.keep_pose_indices)
        sanity_scale_range = _parse_range(args.sanity_scale_range)

        cfg: Dict[str, Any] = {
            "in_dir": str(in_dir),
            "out_dir": str(out_dir),
            "world_coords": world_coords,
            "keep_pose_indices": keep,
            "stride": max(1, args.stride),
            "short_side": args.short_side,
            "min_det": args.min_det,
            "min_track": args.min_track,
            "pose_every": max(1, args.pose_every),
            "pose_complexity": args.pose_complexity,
            "ts_source": args.ts_source,
            "min_hand_score": args.min_hand_score,
            "hand_score_lo": args.hand_score_lo,
            "hand_score_hi": args.hand_score_hi,
            "hand_score_source": str(args.hand_score_source),
            "tracker_init_score": float(args.tracker_init_score),
            "anchor_score": float(args.anchor_score),
            "pose_dist_qual_min": float(args.pose_dist_qual_min),
            "tracker_update_score": float(args.tracker_update_score),
            "pose_side_reassign_ratio": float(args.pose_side_reassign_ratio),
            "pose_ema": args.pose_ema,
            "ndjson": str(args.ndjson) if args.ndjson else "",
            "second_pass": bool(args.second_pass),
            "sp_trigger_below": args.sp_trigger_below,
            "sp_roi_frac": args.sp_roi_frac,
            "sp_margin": args.sp_margin,
            "sp_escalate_step": args.sp_escalate_step,
            "sp_escalate_max": args.sp_escalate_max,
            "sp_hands_up_only": bool(args.sp_hands_up_only),
            "sp_jitter_px": int(args.sp_jitter_px),
            "sp_jitter_rings": int(args.sp_jitter_rings),
            "sp_center_penalty": args.sp_center_penalty,
            "sp_label_relax": args.sp_label_relax,
            "sp_overlap_iou": args.sp_overlap_iou,
            "sp_overlap_shrink": args.sp_overlap_shrink,
            "sp_overlap_penalty_mult": args.sp_overlap_penalty_mult,
            "sp_overlap_require_label": bool(args.sp_overlap_require_label),
            "sp_debug_roi": bool(args.sp_debug_roi),
            "interp_hold": int(args.interp_hold),
            "write_hand_mask": bool(args.write_hand_mask),
            "track_max_gap": int(args.track_max_gap),
            "track_score_decay": float(args.track_score_decay),
            "track_reset_ms": int(args.track_reset_ms),
            "occ_hyst_frames": int(args.occ_hyst_frames),
            "occ_return_k": float(args.occ_return_k),
            "sanity_scale_range": sanity_scale_range,
            "sanity_wrist_k": float(args.sanity_wrist_k),
            "sanity_bone_tol": float(args.sanity_bone_tol),
            "sanity_pass2": bool(args.sanity_pass2),
            "sanity_anchor_max_gap": int(args.sanity_anchor_max_gap),
            "sanity_enable": not bool(args.no_sanity),
            "sanitize_rejects": bool(args.sanitize_rejects),
            "postprocess": bool(args.postprocess),
            "pp_max_gap": int(args.pp_max_gap),
            "pp_smoother": str(args.pp_smoother),
            "pp_only_anchors": bool(args.pp_only_anchors),
            "mp_backend": mp_backend,
            "hand_task": hand_task_path,
            "pose_task": pose_task_path,
            "mp_tasks_delegate": mp_tasks_delegate,
            "log_dir": str(Path(args.log_dir or "outputs/logs")),
            "log_level": log_level,
            "seed": int(args.seed),
            "eval_report": str(args.eval_report) if args.eval_report else "",
            "debug_video": str(args.debug_video) if args.debug_video else "",
        }

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

        cfg.update({
            "world_coords": world_coords,
            "stride": max(1, args.stride),
            "short_side": args.short_side,
            "jobs": args.jobs,
            "mp_backend": mp_backend,
            "log_dir": str(Path(args.log_dir or "outputs/logs")),
            "log_level": log_level,
        })

        tasks: List[Dict[str, Any]] = []
        total_items = 0
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
                out_path = out_dir / f"{seg_uid}.json"
                if args.skip_existing and out_path.exists():
                    print(f"[SKIP] {out_path.name}")
                    continue
                seen_seg_uids.add(seg_uid)
                tasks.append({
                    "vpath": str(vpath),
                    "cfg": cfg,
                    "slug": seg_uid,
                    "segment": seg,
                })

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
                out_path = out_dir / f"{slug}.json"
                if args.skip_existing and out_path.exists():
                    print(f"[SKIP] {out_path.name}")
                    continue
                tasks.append({"vpath": str(v), "cfg": cfg})

            total_items = len(videos)

        if not tasks:
            log_metrics(logger, "extract_keypoints.error", {
                "reason": "no_tasks",
                "segments_mode": bool(segments_manifest),
            })
            raise SystemExit("No videos/segments to process after filtering.")

        cfg["video_count"] = len(tasks)

        log_metrics(logger, "extract_keypoints.tasks_prepared", {
            "scheduled": len(tasks),
            "skipped": max(0, total_items - len(tasks)),
            "skip_existing": bool(args.skip_existing),
            "segments_mode": bool(segments_manifest),
        })

        manifest: List[Dict[str, Any]] = []
        with track_runtime(logger, "extract_keypoints.processing", videos=len(tasks), jobs=args.jobs):
            if args.jobs > 1:
                with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                    futs = [ex.submit(process_worker, t) for t in tasks]
                    for fut in as_completed(futs):
                        try:
                            manifest.append(fut.result())
                        except Exception as e:
                            print(f"[ERROR] {e}")
                            log_metrics(logger, "extract_keypoints.worker_error", {"error": str(e)})
            else:
                for t in tasks:
                    try:
                        manifest.append(process_worker(t))
                    except Exception as e:
                        print(f"[ERROR] {e}")
                        log_metrics(logger, "extract_keypoints.worker_error", {"error": str(e)})

        def _manifest_sort_key(item: Dict[str, Any]) -> tuple:
            seg_uid = item.get("seg_uid")
            if seg_uid:
                return (str(seg_uid), str(item.get("file", "")))
            slug = item.get("slug")
            if slug:
                return (str(slug), str(item.get("file", "")))
            return (str(item.get("id", "")), str(item.get("file", "")))

        manifest.sort(key=_manifest_sort_key)

        manifest_path = out_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[OK] Manifest written to {manifest_path}")

        if args.eval_report:
            eval_path = Path(args.eval_report)
            eval_path.parent.mkdir(parents=True, exist_ok=True)
            eval_report_path = str(eval_path)

            versions = {"python": sys.version}
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

            run_meta = {
                "seed": int(args.seed),
                "jobs": int(args.jobs),
                "mp_backend": mp_backend,
                "args": vars(args),
                "versions": versions,
            }

            slug_to_input = {}
            id_to_input = {}
            for t in tasks:
                vpath = Path(t["vpath"])
                slug = t.get("slug") or slug_for(vpath, in_dir)
                slug_to_input[slug] = str(vpath)
                seg = t.get("segment") or {}
                if seg.get("seg_uid"):
                    id_to_input.setdefault(str(seg.get("seg_uid")), str(vpath))
                else:
                    id_to_input.setdefault(vpath.stem, str(vpath))

            eval_keys = {
                "missing_frames_1",
                "missing_frames_2",
                "occluded_frames_1",
                "occluded_frames_2",
                "swap_frames",
                "dedup_trigger_frames",
                "sp_attempt_frames_1",
                "sp_attempt_frames_2",
                "sp_recovered_frames_1",
                "sp_recovered_frames_2",
                "hold_frames_1",
                "hold_frames_2",
                "outlier_frames_1",
                "outlier_frames_2",
                "sanity_reject_frames_1",
                "sanity_reject_frames_2",
                "missing_gap_p50_1",
                "missing_gap_p90_1",
                "missing_gap_max_1",
                "missing_gap_p50_2",
                "missing_gap_p90_2",
                "missing_gap_max_2",
                "occluded_gap_p50_1",
                "occluded_gap_p90_1",
                "occluded_gap_max_1",
                "occluded_gap_p50_2",
                "occluded_gap_p90_2",
                "occluded_gap_max_2",
                "pp_filled_left",
                "pp_filled_right",
                "pp_gaps_filled_left",
                "pp_gaps_filled_right",
                "pp_smoothing_delta_left",
                "pp_smoothing_delta_right",
            }

            def _safe_rate(num, denom):
                if denom and denom > 0:
                    return float(num) / float(denom)
                return None

            videos_report = []
            quality_scores = []
            swap_rates = []
            missing_rates = []
            outlier_rates = []
            sanity_rates = []
            pp_filled_fracs = []
            pp_smoothing_deltas = []

            for entry in manifest:
                entry_id = entry.get("id")
                output_path = entry.get("file", "")
                slug = entry.get("slug")
                if not slug and output_path:
                    slug = Path(output_path).stem

                input_path = ""
                if slug and slug in slug_to_input:
                    input_path = slug_to_input[slug]
                elif entry_id in id_to_input:
                    input_path = id_to_input[entry_id]

                eval_part = {k: entry[k] for k in eval_keys if k in entry}
                meta_part = {
                    k: v for k, v in entry.items()
                    if k not in eval_keys and k not in ("id", "file")
                }

                num_frames = entry.get("num_frames") or 0
                q = entry.get("quality_score")
                if q is not None:
                    quality_scores.append(float(q))
                if eval_part:
                    swap_rate = _safe_rate(eval_part.get("swap_frames"), num_frames)
                    missing_rate = _safe_rate(
                        (eval_part.get("missing_frames_1", 0) + eval_part.get("missing_frames_2", 0)),
                        2 * num_frames
                    )
                    outlier_rate = _safe_rate(
                        (eval_part.get("outlier_frames_1", 0) + eval_part.get("outlier_frames_2", 0)),
                        2 * num_frames
                    )
                    sanity_rate = _safe_rate(
                        (eval_part.get("sanity_reject_frames_1", 0) + eval_part.get("sanity_reject_frames_2", 0)),
                        2 * num_frames
                    )
                    if swap_rate is not None:
                        swap_rates.append(swap_rate)
                    if missing_rate is not None:
                        missing_rates.append(missing_rate)
                    if outlier_rate is not None:
                        outlier_rates.append(outlier_rate)
                    if sanity_rate is not None:
                        sanity_rates.append(sanity_rate)
                    pp_filled = (
                        eval_part.get("pp_filled_left", 0) + eval_part.get("pp_filled_right", 0)
                    )
                    pp_filled_frac = _safe_rate(pp_filled, 2 * num_frames)
                    if pp_filled_frac is not None:
                        pp_filled_fracs.append(pp_filled_frac)
                    pp_delta_vals = []
                    if eval_part.get("pp_smoothing_delta_left") is not None:
                        pp_delta_vals.append(float(eval_part.get("pp_smoothing_delta_left")))
                    if eval_part.get("pp_smoothing_delta_right") is not None:
                        pp_delta_vals.append(float(eval_part.get("pp_smoothing_delta_right")))
                    if pp_delta_vals:
                        pp_smoothing_deltas.append(sum(pp_delta_vals) / len(pp_delta_vals))

                video_item = {
                    "id": entry_id,
                    "slug": slug,
                    "input": input_path,
                    "output": output_path,
                    "meta": meta_part,
                }
                if eval_part:
                    video_item["eval"] = eval_part
                videos_report.append(video_item)

            videos_report.sort(key=lambda v: (str(v.get("slug") or v.get("id") or ""), str(v.get("output") or "")))

            def _mean(vals):
                return (sum(vals) / len(vals)) if vals else None

            aggregate = {
                "quality_score": _mean(quality_scores),
                "swap_rate": _mean(swap_rates),
                "missing_rate": _mean(missing_rates),
                "outlier_rate": _mean(outlier_rates),
                "sanity_reject_rate": _mean(sanity_rates),
                "pp_filled_frac": _mean(pp_filled_fracs),
                "pp_smoothing_delta": _mean(pp_smoothing_deltas),
            }

            report = {
                "run": run_meta,
                "videos": videos_report,
                "aggregate": aggregate,
            }
            with eval_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
            print(f"[OK] Eval report written to {eval_path}")

        if args.combined_json:
            combined_path = Path(args.combined_json)
            combine_to_single_json(
                out_dir, combined_path, with_meta=args.combined_with_meta
            )
            print(f"[OK] Combined JSON written to {combined_path}")

        log_metrics(logger, "extract_keypoints.completed", {
            "videos_processed": len(manifest),
            "jobs_used": args.jobs,
        })
        return {
            "ok": True,
            "out_dir": str(out_dir),
            "manifest_path": str(manifest_path),
            "eval_report_path": eval_report_path,
            "code": 0,
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
            "manifest_path": str(manifest_path) if manifest_path else "",
            "eval_report_path": eval_report_path,
            "error": error,
            "code": int(code),
        }
    except Exception as exc:
        if _print_errors:
            traceback.print_exc()
        return {
            "ok": False,
            "out_dir": str(out_dir) if out_dir else "",
            "manifest_path": str(manifest_path) if manifest_path else "",
            "eval_report_path": eval_report_path,
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
