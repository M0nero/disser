from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Optional
import os, json

from .process import process_video
from .core.logging_utils import configure_logging, log_metrics, track_runtime


def slug_for(v: Path, base: Optional[Path]) -> str:
    try:
        if base:
            rel = v.resolve().relative_to(base.resolve())
            return "__".join(rel.with_suffix("").parts)
    except Exception:
        pass
    return v.stem


def _resolve_debug_video_path(cfg: Dict[str, Any], slug: str) -> Optional[Path]:
    raw = (cfg.get("debug_video") or "").strip()
    if not raw:
        return None
    target = Path(raw)
    if target.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        return target / f"{slug}.mp4"
    if target.suffix.lower() == ".mp4":
        video_count = int(cfg.get("video_count") or 1)
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


def process_worker(payload: dict) -> Dict[str, Any]:
    vpath = Path(payload["vpath"])
    cfg = payload["cfg"]
    segment = payload.get("segment") or {}

    logger = configure_logging(
        cfg.get("log_dir", "outputs/logs"),
        cfg.get("log_level", "INFO"),
    )

    in_dir = Path(cfg.get("in_dir")) if cfg.get("in_dir") else None
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    slug = payload.get("slug") or segment.get("seg_uid") or slug_for(vpath, in_dir)
    out_name = payload.get("out_name") or slug
    out_path = out_dir / f"{out_name}.json"
    debug_video_path = _resolve_debug_video_path(cfg, slug)
    eval_mode = bool(cfg.get("eval_report"))
    seed = int(cfg.get("seed", 0))
    frame_start = segment.get("start") if segment else payload.get("frame_start")
    frame_end = segment.get("end") if segment else payload.get("frame_end")

    ndjson_path = None
    if cfg.get("ndjson"):
        base = Path(cfg["ndjson"])
        ndjson_path = base.with_name(f"{base.stem}.{slug}{base.suffix}")

    log_metrics(logger, "process_worker.start", {
        "video": vpath.name,
        "slug": slug,
        "out_path": str(out_path),
        "ndjson": str(ndjson_path) if ndjson_path else None,
        "debug_video": str(debug_video_path) if debug_video_path else None,
        "eval_mode": bool(eval_mode),
        "seed": seed,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "seg_uid": segment.get("seg_uid") if segment else None,
    })

    lock = out_path.with_suffix(".lock")
    claimed = False
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        claimed = True
    except FileExistsError:
        log_metrics(logger, "process_worker.skip_existing", {
            "video": vpath.name,
            "slug": slug,
            "out_path": str(out_path),
        })
        result = {
            "id": vpath.stem,
            "file": str(out_path),
            "num_frames": 0,
            "fps": 0.0,
            "hands_frames": 0,
            "hands_coverage": 0.0,
            "left_score_mean": 0.0,
            "right_score_mean": 0.0,
        }
        if segment:
            seg_uid = str(segment.get("seg_uid") or "")
            if seg_uid:
                result["id"] = seg_uid
                result["seg_uid"] = seg_uid
            for key in ("video_id", "start", "end", "split", "dataset", "label", "length"):
                if key in segment:
                    result[key] = segment.get(key)
        return result

    try:
        with track_runtime(logger, "process_worker", video=vpath.name, slug=slug):
            result = process_video(
                vpath,
                out_path,
                cfg["world_coords"],
                cfg["keep_pose_indices"],
                cfg["stride"],
                cfg["short_side"],
                cfg["min_det"],
                cfg["min_track"],
                cfg["pose_every"],
                cfg["pose_complexity"],
                cfg["ts_source"],
                cfg["min_hand_score"],
                cfg.get("hand_score_lo", 0.55),
                cfg.get("hand_score_hi", 0.90),
                cfg.get("hand_score_source", "handedness"),
                cfg.get("tracker_init_score", -1.0),
                cfg.get("anchor_score", -1.0),
                cfg.get("pose_dist_qual_min", 0.50),
                cfg.get("tracker_update_score", -1.0),
                cfg.get("pose_side_reassign_ratio", 0.85),
                cfg["pose_ema"],
                ndjson_path,
                cfg["second_pass"],
                cfg["sp_trigger_below"],
                cfg["sp_roi_frac"],
                cfg["sp_margin"],
                cfg["sp_escalate_step"],
                cfg["sp_escalate_max"],
                cfg["sp_hands_up_only"],
                cfg.get("sp_jitter_px", 0),
                cfg.get("sp_jitter_rings", 1),
                cfg.get("sp_center_penalty", 0.30),
                cfg.get("sp_label_relax", 0.15),
                cfg.get("sp_overlap_iou", 0.15),
                cfg.get("sp_overlap_shrink", 0.70),
                cfg.get("sp_overlap_penalty_mult", 2.0),
                cfg.get("sp_overlap_require_label", False),
                cfg.get("sp_debug_roi", False),
                cfg["interp_hold"],
                cfg.get("track_max_gap", 15),
                cfg.get("track_score_decay", 0.90),
                cfg.get("track_reset_ms", 250),
                cfg["write_hand_mask"],
                cfg.get("mp_backend", "solutions"),
                cfg.get("hand_task", ""),
                cfg.get("pose_task", ""),
                cfg.get("mp_tasks_delegate", ""),
                debug_video_path,
                bool(eval_mode),
                seed,
                cfg.get("occ_hyst_frames", 15),
                cfg.get("occ_return_k", 1.2),
                cfg.get("sanity_enable", True),
                cfg.get("sanity_scale_range", (0.70, 1.35)),
                cfg.get("sanity_wrist_k", 2.0),
                cfg.get("sanity_bone_tol", 0.30),
                cfg.get("sanity_pass2", False),
                cfg.get("sanity_anchor_max_gap", 0),
                cfg.get("sanitize_rejects", True),
                cfg.get("postprocess", False),
                cfg.get("pp_max_gap", 15),
                cfg.get("pp_smoother", "ema"),
                cfg.get("pp_only_anchors", True),
                frame_start=frame_start,
                frame_end=frame_end,
                segment_meta=segment if segment else None,
            )

        log_metrics(logger, "process_worker.result", {
            "video": vpath.name,
            "slug": slug,
            "frames": result.get("num_frames", 0),
            "hands_coverage": result.get("hands_coverage"),
            "quality_score": result.get("quality_score"),
        })
        if segment:
            seg_uid = str(segment.get("seg_uid") or "")
            if seg_uid:
                result["id"] = seg_uid
                result["seg_uid"] = seg_uid
            for key in ("video_id", "start", "end", "split", "dataset", "label", "length"):
                if key in segment:
                    result[key] = segment.get(key)
        return result

    except Exception as exc:
        log_metrics(logger, "process_worker.error", {
            "video": vpath.name,
            "slug": slug,
            "error": str(exc),
        })
        raise

    finally:
        if claimed:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
