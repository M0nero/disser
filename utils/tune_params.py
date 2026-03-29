from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import heapq
import math
import shutil
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from scripts.extract_keypoints import run_pipeline

DEFAULT_SCORE_WEIGHTS: Dict[str, float] = {
    "quality": 1.0,
    "swap_rate": -1.0,
    "outlier_rate": -0.6,
    "sanity_reject_rate": -0.4,
    "missing_rate": -0.3,
    "occluded_rate": 0.0,
    "missing_gap_p90_frac": 0.0,
    "occluded_gap_p90_frac": 0.0,
    "pp_filled_frac": 0.0,
    "pp_smoothing_delta": 0.0,
}

LOGGER = logging.getLogger(__name__)

RUNTIME_FLAGS = {
    "--debug-video",
    "--in-dir",
    "--out-dir",
}
POSITIONAL_KEY = "__positional__"
ID_IGNORE_FLAGS = set(RUNTIME_FLAGS) | {"--jobs"}


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return value


def normalize_args_dict(args_dict: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in args_dict.items():
        if raw_value is None:
            continue
        if raw_key == POSITIONAL_KEY:
            continue
        key = str(raw_key)
        if not key.startswith("-"):
            raise ValueError(f"Argument keys must start with '-' or '--': {key}")
        normalized[key] = _normalize_value(raw_value)
    return normalized


def _filter_args(
    args_dict: Dict[str, Any],
    *,
    drop_flags: Optional[set] = None,
    force_jobs: Optional[int] = None,
) -> Dict[str, Any]:
    drop = drop_flags or set()
    filtered = {
        k: v for k, v in args_dict.items()
        if k not in drop and k != POSITIONAL_KEY
    }
    if force_jobs is not None:
        filtered["--jobs"] = int(force_jobs)
    return filtered


def build_run_args(args_dict: Dict[str, Any], *, force_jobs: Optional[int] = None) -> Dict[str, Any]:
    normalized = normalize_args_dict(args_dict)
    return _filter_args(normalized, drop_flags=RUNTIME_FLAGS, force_jobs=force_jobs)


def config_id(args_dict: Dict[str, Any]) -> str:
    normalized = normalize_args_dict(args_dict)
    filtered = _filter_args(normalized, drop_flags=ID_IGNORE_FLAGS)
    stable_json = json.dumps(filtered, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(stable_json.encode("utf-8")).hexdigest()[:12]


def load_grid_config(
    path: str | Path,
) -> Tuple[Dict[str, Any], Dict[str, List[Any]], Dict[str, float], Dict[str, float], str]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    base_args = data.get("base_args") or {}
    grid = data.get("grid") or {}
    score_weights = data.get("score_weights")
    pp_score_weights = data.get("pp_score_weights")
    rank_by = str(data.get("rank_by") or "score").strip().lower()
    if score_weights is None:
        score_weights = dict(DEFAULT_SCORE_WEIGHTS)
    if pp_score_weights is None:
        pp_score_weights = dict(score_weights)
    return dict(base_args), dict(grid), dict(score_weights), dict(pp_score_weights), rank_by


def expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not grid:
        return [{}]
    items = list(grid.items())
    keys = [k for k, _ in items]
    values: List[List[Any]] = []
    for key, vals in items:
        if not isinstance(vals, list):
            raise ValueError(f"Grid values must be lists for {key}")
        if not vals:
            raise ValueError(f"Grid values must be non-empty for {key}")
        values.append(vals)
    out: List[Dict[str, Any]] = []
    for combo in product(*values):
        out.append(dict(zip(keys, combo)))
    return out


def merge_args(base_args: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base_args)
    merged.update(overrides)
    return merged


def args_dict_to_argv(args_dict: Dict[str, Any]) -> List[str]:
    positionals = args_dict.get(POSITIONAL_KEY) if isinstance(args_dict, dict) else None
    argv: List[str] = []
    for flag in sorted(k for k in args_dict.keys() if k != POSITIONAL_KEY):
        value = args_dict[flag]
        if value is None or value is False:
            continue
        if value is True:
            argv.append(str(flag))
            continue
        argv.append(str(flag))
        argv.append(str(value))
    if positionals:
        if isinstance(positionals, (list, tuple)):
            argv.extend(str(item) for item in positionals)
        else:
            argv.append(str(positionals))
    return argv


def build_arg_dicts(base_args: Dict[str, Any], grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    overrides_list = expand_grid(grid)
    return [merge_args(base_args, overrides) for overrides in overrides_list]


def build_run_dir(out_dir: str | Path, args_dict: Dict[str, Any]) -> Path:
    tmp_root = Path(out_dir) / "_tune_tmp"
    return tmp_root / config_id(args_dict)


def build_run_argv(in_dir: str | Path, out_dir: str | Path, args_dict: Dict[str, Any]) -> List[str]:
    run_dir = build_run_dir(out_dir, args_dict)
    return [
        "--in-dir", str(in_dir),
        "--out-dir", str(run_dir),
    ] + args_dict_to_argv(build_run_args(args_dict))


def _load_parquet_rows(path: Path) -> Optional[List[Dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    except Exception:
        return None


def _mean_numeric(entries: List[Dict[str, Any]]) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                val = float(value)
                if not math.isfinite(val):
                    continue
                sums[key] = sums.get(key, 0.0) + val
                counts[key] = counts.get(key, 0) + 1
    return {
        key: (sums[key] / counts[key])
        for key in counts
        if counts[key] > 0
    }


def _is_flag_token(token: str) -> bool:
    if not token.startswith("-") or token == "-":
        return False
    if token.startswith("--"):
        return True
    try:
        float(token)
    except ValueError:
        return True
    return False


def _argv_to_args_dict(argv: List[str]) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    positionals: List[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--":
            positionals.extend(str(item) for item in argv[i + 1:])
            break
        if _is_flag_token(token):
            if i + 1 >= len(argv) or _is_flag_token(argv[i + 1]):
                args[token] = True
                i += 1
                continue
            args[token] = argv[i + 1]
            i += 2
            continue
        positionals.append(str(token))
        i += 1
    if positionals:
        args[POSITIONAL_KEY] = positionals
    return args


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return None
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "y", "on"):
            return True
        if lowered in ("0", "false", "no", "n", "off"):
            return False
    return None


def _parse_sanity_scale_range(raw: Any) -> Optional[Tuple[float, float]]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        if len(raw) != 2:
            return None
        lo = _coerce_float(raw[0])
        hi = _coerce_float(raw[1])
    else:
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        if len(parts) != 2:
            return None
        lo = _coerce_float(parts[0])
        hi = _coerce_float(parts[1])
    if lo is None or hi is None:
        return None
    if lo <= 0.0 or hi <= 0.0 or lo >= hi:
        return None
    return float(lo), float(hi)


def validate_args_dict(args_dict: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    world_raw = args_dict.get("--world-coords")
    image_raw = args_dict.get("--image-coords")
    world = _coerce_bool(world_raw)
    image = _coerce_bool(image_raw)
    if world is None and world_raw is not None:
        errors.append("Invalid --world-coords value; expected boolean.")
    if image is None and image_raw is not None:
        errors.append("Invalid --image-coords value; expected boolean.")
    world_flag = bool(world) if world is not None else False
    image_flag = bool(image) if image is not None else False
    if world_flag == image_flag:
        errors.append("Exactly one of --world-coords or --image-coords must be true.")

    lo = _coerce_float(args_dict.get("--hand-score-lo"))
    hi = _coerce_float(args_dict.get("--hand-score-hi"))
    if lo is not None and hi is not None and lo > hi:
        errors.append("--hand-score-lo must be <= --hand-score-hi.")

    sanity_raw = args_dict.get("--sanity-scale-range")
    if sanity_raw is not None and _parse_sanity_scale_range(sanity_raw) is None:
        errors.append("Invalid --sanity-scale-range; expected '0.70,1.35' or two numeric values.")

    return errors


def _metrics_from_video_rows(rows: Optional[List[Dict[str, Any]]]) -> Dict[str, float]:
    if not rows:
        return {}

    metrics: Dict[str, float] = {}
    frames_total = 0.0
    quality_weighted_sum = 0.0
    quality_weighted_frames = 0.0
    miss_1_total = 0.0
    miss_2_total = 0.0
    out_1_total = 0.0
    out_2_total = 0.0
    san_1_total = 0.0
    san_2_total = 0.0
    missing_total = 0.0
    outlier_total = 0.0
    sanity_total = 0.0
    occluded_total = 0.0
    swap_total = 0.0
    pp_filled_total = 0.0
    pp_filled_seen = False
    pp_delta_weighted_sum = 0.0
    pp_delta_weighted_frames = 0.0
    missing_gap_p90_weighted_sum = 0.0
    missing_gap_p90_weighted_frames = 0.0
    occluded_gap_p90_weighted_sum = 0.0
    occluded_gap_p90_weighted_frames = 0.0

    for row in rows:
        if not isinstance(row, dict):
            continue
        num_frames = _coerce_float(row.get("num_frames"))
        if num_frames is None or num_frames <= 0:
            continue
        frames_total += num_frames
        q = _coerce_float(row.get("quality_score"))
        if q is not None:
            quality_weighted_sum += q * num_frames
            quality_weighted_frames += num_frames

        swap = _coerce_float(row.get("swap_frames"))
        if swap is not None:
            swap_total += swap

        missing_1 = _coerce_float(row.get("missing_frames_1")) or 0.0
        missing_2 = _coerce_float(row.get("missing_frames_2")) or 0.0
        outlier_1 = _coerce_float(row.get("outlier_frames_1")) or 0.0
        outlier_2 = _coerce_float(row.get("outlier_frames_2")) or 0.0
        sanity_1 = _coerce_float(row.get("sanity_reject_frames_1")) or 0.0
        sanity_2 = _coerce_float(row.get("sanity_reject_frames_2")) or 0.0
        occluded_1 = _coerce_float(row.get("occluded_frames_1")) or 0.0
        occluded_2 = _coerce_float(row.get("occluded_frames_2")) or 0.0
        pp_filled_left = _coerce_float(row.get("pp_filled_left")) or 0.0
        pp_filled_right = _coerce_float(row.get("pp_filled_right")) or 0.0

        miss_1_total += missing_1
        miss_2_total += missing_2
        out_1_total += outlier_1
        out_2_total += outlier_2
        san_1_total += sanity_1
        san_2_total += sanity_2
        missing_total += missing_1 + missing_2
        outlier_total += outlier_1 + outlier_2
        sanity_total += sanity_1 + sanity_2
        occluded_total += occluded_1 + occluded_2
        pp_filled_seen = pp_filled_seen or ("pp_filled_left" in row or "pp_filled_right" in row)
        pp_filled_total += pp_filled_left + pp_filled_right

        pp_delta_vals = []
        if row.get("pp_smoothing_delta_left") is not None:
            pp_delta_vals.append(float(row.get("pp_smoothing_delta_left")))
        if row.get("pp_smoothing_delta_right") is not None:
            pp_delta_vals.append(float(row.get("pp_smoothing_delta_right")))
        if pp_delta_vals:
            pp_delta_weighted_sum += (sum(pp_delta_vals) / len(pp_delta_vals)) * num_frames
            pp_delta_weighted_frames += num_frames

        missing_gap_vals = []
        if row.get("missing_gap_p90_1") is not None:
            missing_gap_vals.append(float(row.get("missing_gap_p90_1")))
        if row.get("missing_gap_p90_2") is not None:
            missing_gap_vals.append(float(row.get("missing_gap_p90_2")))
        if missing_gap_vals:
            missing_gap_p90_weighted_sum += (max(missing_gap_vals) / num_frames) * num_frames
            missing_gap_p90_weighted_frames += num_frames

        occluded_gap_vals = []
        if row.get("occluded_gap_p90_1") is not None:
            occluded_gap_vals.append(float(row.get("occluded_gap_p90_1")))
        if row.get("occluded_gap_p90_2") is not None:
            occluded_gap_vals.append(float(row.get("occluded_gap_p90_2")))
        if occluded_gap_vals:
            occluded_gap_p90_weighted_sum += (max(occluded_gap_vals) / num_frames) * num_frames
            occluded_gap_p90_weighted_frames += num_frames

    if quality_weighted_frames > 0.0:
        metrics["quality_score"] = float(quality_weighted_sum / quality_weighted_frames)
    if frames_total > 0.0:
        metrics["swap_rate"] = float(swap_total / frames_total)
        metrics["missing_rate_1"] = float(miss_1_total / frames_total)
        metrics["missing_rate_2"] = float(miss_2_total / frames_total)
        metrics["outlier_rate_1"] = float(out_1_total / frames_total)
        metrics["outlier_rate_2"] = float(out_2_total / frames_total)
        metrics["sanity_reject_rate_1"] = float(san_1_total / frames_total)
        metrics["sanity_reject_rate_2"] = float(san_2_total / frames_total)
        metrics["missing_rate"] = float(missing_total / (2.0 * frames_total))
        metrics["outlier_rate"] = float(outlier_total / (2.0 * frames_total))
        metrics["sanity_reject_rate"] = float(sanity_total / (2.0 * frames_total))
        metrics["occluded_rate"] = float(occluded_total / (2.0 * frames_total))
        if pp_filled_seen:
            metrics["pp_filled_frac"] = float(pp_filled_total / (2.0 * frames_total))
    if pp_delta_weighted_frames > 0.0:
        metrics["pp_smoothing_delta"] = float(pp_delta_weighted_sum / pp_delta_weighted_frames)
    if missing_gap_p90_weighted_frames > 0.0:
        metrics["missing_gap_p90_frac"] = float(missing_gap_p90_weighted_sum / missing_gap_p90_weighted_frames)
    if occluded_gap_p90_weighted_frames > 0.0:
        metrics["occluded_gap_p90_frac"] = float(occluded_gap_p90_weighted_sum / occluded_gap_p90_weighted_frames)
    return metrics


def _worker_run_one(
    config_id: str,
    args_dict: Dict[str, Any],
    in_dir: str,
    out_dir: str,
    resume: bool,
) -> Dict[str, Any]:
    run_dir = Path(out_dir) / "_tune_tmp" / config_id
    videos_parquet_path = run_dir / "videos.parquet"
    try:
        if resume and videos_parquet_path.exists():
            metrics = _metrics_from_video_rows(_load_parquet_rows(videos_parquet_path))
            return {
                "ok": True,
                "config_id": config_id,
                "run_dir": str(run_dir),
                "metrics": metrics,
                "args": args_dict,
                "skipped": True,
            }

        tmp_root = Path(out_dir) / "_tune_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)

        run_args = build_run_args(args_dict, force_jobs=1)
        argv = [
            "--in-dir", str(in_dir),
            "--out-dir", str(run_dir),
        ] + args_dict_to_argv(run_args)

        res = run_pipeline(argv)
        if not res.get("ok"):
            return {
                "ok": False,
                "config_id": config_id,
                "run_dir": str(run_dir),
                "error": res.get("error") or "run_pipeline failed",
            }

        metrics = _metrics_from_video_rows(_load_parquet_rows(videos_parquet_path))

        config_path = run_dir / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {"argv": argv, "args": args_dict, "run_args": run_args, "metrics": metrics}
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "ok": True,
            "config_id": config_id,
            "run_dir": str(run_dir),
            "metrics": metrics,
            "args": args_dict,
            "skipped": False,
        }
    except Exception as exc:
        return {
            "ok": False,
            "config_id": config_id,
            "run_dir": str(run_dir),
            "error": str(exc),
        }


def run_one_config(argv: List[str], run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    res = run_pipeline(argv)
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error") or "run_pipeline failed"}

    metrics = _metrics_from_video_rows(_load_parquet_rows(run_dir / "videos.parquet"))

    args_dict = _argv_to_args_dict(argv)
    run_args = _filter_args(args_dict, drop_flags=RUNTIME_FLAGS)
    config_path = run_dir / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {"argv": argv, "args": args_dict, "run_args": run_args, "metrics": metrics}
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"ok": True, "metrics": metrics}


def _compute_score_with_weights(
    metrics: Dict[str, Any],
    weights: Dict[str, float],
    *,
    log_missing: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    def _num(val: Any) -> Optional[float]:
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            num = float(val)
            if math.isfinite(num):
                return num
        return None

    def _fallback(weight: float) -> float:
        return 1.0 if weight < 0.0 else 0.0

    missing_fields: List[str] = []

    def _weight(primary: str, fallback: Optional[str] = None) -> float:
        if primary in weights:
            raw = weights.get(primary)
        elif fallback and fallback in weights:
            raw = weights.get(fallback)
        else:
            raw = 0.0
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return 0.0
        return val if math.isfinite(val) else 0.0

    wq = _weight("quality", "quality_score")
    ws = _weight("swap_rate")
    wo = _weight("outlier_rate")
    wsa = _weight("sanity_reject_rate")
    wm = _weight("missing_rate")
    wocc = _weight("occluded_rate")
    wmgap = _weight("missing_gap_p90_frac")
    wogap = _weight("occluded_gap_p90_frac")
    wpp_fill = _weight("pp_filled_frac")
    wpp_delta = _weight("pp_smoothing_delta")

    quality = _num(metrics.get("quality_score"))
    if quality is None:
        if wq != 0.0:
            missing_fields.append("quality_score")
        quality = _fallback(wq)

    swap = _num(metrics.get("swap_rate"))
    if swap is None:
        if ws != 0.0:
            missing_fields.append("swap_rate")
        swap = _fallback(ws)

    missing = _num(metrics.get("missing_rate"))
    if missing is None:
        m1 = _num(metrics.get("missing_rate_1"))
        m2 = _num(metrics.get("missing_rate_2"))
        if m1 is not None and m2 is not None:
            missing = 0.5 * (m1 + m2)
        else:
            if m1 is None and m2 is None and wm != 0.0:
                missing_fields.append("missing_rate")
            missing = _fallback(wm)

    outlier = _num(metrics.get("outlier_rate"))
    if outlier is None:
        o1 = _num(metrics.get("outlier_rate_1"))
        o2 = _num(metrics.get("outlier_rate_2"))
        if o1 is not None and o2 is not None:
            outlier = 0.5 * (o1 + o2)
        else:
            if o1 is None and o2 is None and wo != 0.0:
                missing_fields.append("outlier_rate")
            outlier = _fallback(wo)

    sanity = _num(metrics.get("sanity_reject_rate"))
    if sanity is None:
        s1 = _num(metrics.get("sanity_reject_rate_1"))
        s2 = _num(metrics.get("sanity_reject_rate_2"))
        if s1 is not None and s2 is not None:
            sanity = 0.5 * (s1 + s2)
        else:
            if s1 is None and s2 is None and wsa != 0.0:
                missing_fields.append("sanity_reject_rate")
            sanity = _fallback(wsa)

    occluded = _num(metrics.get("occluded_rate"))
    if occluded is None:
        if wocc != 0.0:
            missing_fields.append("occluded_rate")
        occluded = _fallback(wocc)

    missing_gap_p90 = _num(metrics.get("missing_gap_p90_frac"))
    if missing_gap_p90 is None:
        if wmgap != 0.0:
            missing_fields.append("missing_gap_p90_frac")
        missing_gap_p90 = _fallback(wmgap)

    occluded_gap_p90 = _num(metrics.get("occluded_gap_p90_frac"))
    if occluded_gap_p90 is None:
        if wogap != 0.0:
            missing_fields.append("occluded_gap_p90_frac")
        occluded_gap_p90 = _fallback(wogap)

    pp_filled = _num(metrics.get("pp_filled_frac"))
    if pp_filled is None:
        if wpp_fill != 0.0:
            missing_fields.append("pp_filled_frac")
        pp_filled = _fallback(wpp_fill)

    pp_delta = _num(metrics.get("pp_smoothing_delta"))
    if pp_delta is None:
        if wpp_delta != 0.0:
            missing_fields.append("pp_smoothing_delta")
        pp_delta = _fallback(wpp_delta)

    if missing_fields and log_missing:
        LOGGER.warning(
            "Missing metrics for scoring (%s), using fallback defaults",
            ", ".join(sorted(set(missing_fields))),
        )

    score = (
        wq * quality
        + ws * swap
        + wo * outlier
        + wsa * sanity
        + wm * missing
        + wocc * occluded
        + wmgap * missing_gap_p90
        + wogap * occluded_gap_p90
        + wpp_fill * pp_filled
        + wpp_delta * pp_delta
    )
    breakdown = {
        "quality": quality,
        "swap_rate": swap,
        "outlier_rate": outlier,
        "sanity_reject_rate": sanity,
        "missing_rate": missing,
        "occluded_rate": occluded,
        "missing_gap_p90_frac": missing_gap_p90,
        "occluded_gap_p90_frac": occluded_gap_p90,
        "pp_filled_frac": pp_filled,
        "pp_smoothing_delta": pp_delta,
        "weights": {
            "quality": wq,
            "swap_rate": ws,
            "outlier_rate": wo,
            "sanity_reject_rate": wsa,
            "missing_rate": wm,
            "occluded_rate": wocc,
            "missing_gap_p90_frac": wmgap,
            "occluded_gap_p90_frac": wogap,
            "pp_filled_frac": wpp_fill,
            "pp_smoothing_delta": wpp_delta,
        },
        "terms": {
            "quality": wq * quality,
            "swap_rate": ws * swap,
            "outlier_rate": wo * outlier,
            "sanity_reject_rate": wsa * sanity,
            "missing_rate": wm * missing,
            "occluded_rate": wocc * occluded,
            "missing_gap_p90_frac": wmgap * missing_gap_p90,
            "occluded_gap_p90_frac": wogap * occluded_gap_p90,
            "pp_filled_frac": wpp_fill * pp_filled,
            "pp_smoothing_delta": wpp_delta * pp_delta,
        },
        "missing_fields": sorted(set(missing_fields)),
    }
    return float(score), breakdown


def compute_score(
    metrics: Dict[str, Any],
    weights: Dict[str, float],
    pp_weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    score, breakdown = _compute_score_with_weights(metrics, weights, log_missing=True)
    pp_weights = pp_weights or weights
    pp_score, pp_breakdown = _compute_score_with_weights(metrics, pp_weights, log_missing=False)
    breakdown["overall_score"] = score
    breakdown["pp_score"] = pp_score
    breakdown["pp_weights"] = pp_breakdown.get("weights", {})
    breakdown["pp_terms"] = pp_breakdown.get("terms", {})
    breakdown["pp_missing_fields"] = pp_breakdown.get("missing_fields", [])
    return float(score), breakdown


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _safe_rmtree(path: Path, root: Path) -> bool:
    if not path.exists():
        return False
    if not _is_within(path, root):
        LOGGER.warning("Refusing to delete outside tmp_root: %s", path)
        return False
    shutil.rmtree(path)
    return True


def _safe_move(src: Path, dst: Path, root: Path) -> bool:
    if not src.exists():
        return False
    if not _is_within(src, root):
        LOGGER.warning("Refusing to move outside tmp_root: %s", src)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))
    return True


def run_grid_search(
    in_dir: str | Path,
    out_dir: str | Path,
    args_dicts: List[Dict[str, Any]],
    *,
    top_k: int = 5,
    resume: bool = False,
    score_weights: Optional[Dict[str, float]] = None,
    pp_score_weights: Optional[Dict[str, float]] = None,
    rank_by: str = "score",
    workers: int = 1,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    if top_k <= 0:
        return []
    workers = max(1, int(workers) if workers else 1)
    score_weights = score_weights or DEFAULT_SCORE_WEIGHTS
    pp_score_weights = pp_score_weights or score_weights
    tmp_root = Path(out_dir) / "_tune_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    kept: List[Tuple[float, str, Path, Dict[str, Any], Dict[str, Any]]] = []
    all_results: List[Dict[str, Any]] = []

    rank_by_norm = (rank_by or "score").strip().lower()
    if rank_by_norm in ("pp", "pp_score", "postprocess"):
        rank_by_norm = "pp"
    elif rank_by_norm in ("score", "overall", "total"):
        rank_by_norm = "score"
    else:
        LOGGER.warning("Unknown rank_by '%s', using overall score", rank_by)
        rank_by_norm = "score"

    configs: List[Tuple[str, Dict[str, Any]]] = []
    seen_ids: Dict[str, Dict[str, Any]] = {}
    for raw_args in args_dicts:
        try:
            normalized = normalize_args_dict(raw_args)
        except Exception as exc:
            if verbose:
                print(f"[SKIP] invalid args: {exc}")
            all_results.append({
                "config_id": "",
                "score": 0.0,
                "overall_score": 0.0,
                "pp_score": 0.0,
                "metrics": {},
                "breakdown": {},
                "args": raw_args,
                "error": str(exc),
            })
            continue
        errors = validate_args_dict(normalized)
        if errors:
            msg = "; ".join(errors)
            if verbose:
                print(f"[SKIP] invalid config: {msg}")
            all_results.append({
                "config_id": "",
                "score": 0.0,
                "overall_score": 0.0,
                "pp_score": 0.0,
                "metrics": {},
                "breakdown": {},
                "args": normalized,
                "error": msg,
            })
            continue
        cfg_id = config_id(normalized)
        if cfg_id in seen_ids:
            msg = f"Duplicate config_id {cfg_id}"
            if verbose:
                print(f"[SKIP] {msg}")
            all_results.append({
                "config_id": cfg_id,
                "score": 0.0,
                "overall_score": 0.0,
                "pp_score": 0.0,
                "metrics": {},
                "breakdown": {},
                "args": normalized,
                "error": msg,
            })
            continue
        seen_ids[cfg_id] = normalized
        configs.append((cfg_id, normalized))

    total = len(configs)
    finished = 0
    futures = {}
    if total:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for cfg_id, args_dict in configs:
                fut = ex.submit(_worker_run_one, cfg_id, args_dict, str(in_dir), str(out_dir), bool(resume))
                futures[fut] = (cfg_id, args_dict)

            for fut in as_completed(futures):
                finished += 1
                cfg_id, args_dict = futures[fut]
                run_dir = Path(out_dir) / "_tune_tmp" / cfg_id
                try:
                    res = fut.result()
                except Exception as exc:
                    removed = _safe_rmtree(run_dir, tmp_root)
                    if verbose:
                        print(f"[{finished}/{total}] {cfg_id} failed: {exc} (removed={removed})")
                    all_results.append({
                        "config_id": cfg_id,
                        "score": 0.0,
                        "overall_score": 0.0,
                        "pp_score": 0.0,
                        "metrics": {},
                        "breakdown": {},
                        "args": args_dict,
                        "error": str(exc),
                    })
                    continue

                if not res.get("ok"):
                    removed = _safe_rmtree(Path(res.get("run_dir") or run_dir), tmp_root)
                    if verbose:
                        print(f"[{finished}/{total}] {cfg_id} failed: {res.get('error')} (removed={removed})")
                    all_results.append({
                        "config_id": cfg_id,
                        "score": 0.0,
                        "overall_score": 0.0,
                        "pp_score": 0.0,
                        "metrics": {},
                        "breakdown": {},
                        "args": args_dict,
                        "error": res.get("error"),
                    })
                    continue

                metrics = res.get("metrics") or {}
                score, breakdown = compute_score(metrics, score_weights, pp_score_weights)
                pp_score = float(breakdown.get("pp_score") or 0.0)
                rank_score = pp_score if rank_by_norm == "pp" else float(score)
                status = "resume" if res.get("skipped") else "run"
                if verbose:
                    score_label = "pp_score" if rank_by_norm == "pp" else "score"
                    score_val = rank_score if rank_by_norm == "pp" else score
                    print(
                        f"[{finished}/{total}] {cfg_id} {score_label}={score_val:.4f} "
                        f"quality={breakdown['quality']:.4f} swap={breakdown['swap_rate']:.4f} "
                        f"missing_used={breakdown['missing_rate']:.4f} outlier_used={breakdown['outlier_rate']:.4f} "
                        f"sanity_used={breakdown['sanity_reject_rate']:.4f} ({status})"
                    )

                run_dir_path = Path(res.get("run_dir") or run_dir)
                if len(kept) < top_k:
                    heapq.heappush(kept, (rank_score, cfg_id, run_dir_path, metrics, breakdown))
                    if verbose:
                        print(f"  keep: {run_dir_path}")
                else:
                    if rank_score > kept[0][0]:
                        worst = heapq.heappop(kept)
                        removed = _safe_rmtree(worst[2], tmp_root)
                        if verbose:
                            print(f"  evict: {worst[1]} -> {worst[2]} (removed={removed})")
                        heapq.heappush(kept, (rank_score, cfg_id, run_dir_path, metrics, breakdown))
                        if verbose:
                            print(f"  keep: {run_dir_path}")
                    else:
                        removed = _safe_rmtree(run_dir_path, tmp_root)
                        if verbose:
                            print(f"  drop: {run_dir_path} (removed={removed})")

                all_results.append({
                    "config_id": cfg_id,
                    "score": float(rank_score),
                    "overall_score": float(score),
                    "pp_score": float(pp_score),
                    "metrics": metrics,
                    "breakdown": breakdown,
                    "args": args_dict,
                })

    kept_sorted = sorted(kept, key=lambda x: x[0], reverse=True)
    top_root = Path(out_dir) / "top_runs"
    for rank, (score, cfg_id, run_dir, _, _) in enumerate(kept_sorted, start=1):
        dest = top_root / f"top_{rank:02d}_{score:.4f}_{cfg_id}"
        moved = _safe_move(run_dir, dest, tmp_root)
        if verbose:
            print(f"  move: {run_dir} -> {dest} (moved={moved})")

    results_csv = Path(out_dir) / "tune_results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with results_csv.open("w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "config_id",
            "score",
            "overall_score",
            "pp_score",
            "quality",
            "swap_rate",
            "missing_rate",
            "outlier_rate",
            "sanity_reject_rate",
            "pp_filled_frac",
            "pp_smoothing_delta",
            "args_json",
            "error",
        ])
        for item in all_results:
            metrics = item.get("metrics") or {}
            args_json = json.dumps(item.get("args") or {}, sort_keys=True, ensure_ascii=False)
            writer.writerow([
                str(item.get("config_id") or ""),
                f"{float(item.get('score', 0.0)):.6f}",
                f"{float(item.get('overall_score', 0.0)):.6f}",
                f"{float(item.get('pp_score', 0.0)):.6f}",
                metrics.get("quality_score", ""),
                metrics.get("swap_rate", ""),
                metrics.get("missing_rate", ""),
                metrics.get("outlier_rate", ""),
                metrics.get("sanity_reject_rate", ""),
                metrics.get("pp_filled_frac", ""),
                metrics.get("pp_smoothing_delta", ""),
                args_json,
                str(item.get("error") or ""),
            ])

    top_configs_path = Path(out_dir) / "top_configs.json"
    top_configs = []
    for rank, item in enumerate(kept_sorted[:3], start=1):
        score, cfg_id, run_dir, metrics, breakdown = item
        top_dir = top_root / f"top_{rank:02d}_{score:.4f}_{cfg_id}"
        top_configs.append({
            "rank": rank,
            "score": float(score),
            "overall_score": float(breakdown.get("overall_score") or 0.0),
            "pp_score": float(breakdown.get("pp_score") or 0.0),
            "rank_by": rank_by_norm,
            "config_id": cfg_id,
            "run_dir": str(top_dir),
            "metrics": metrics,
            "breakdown": breakdown,
        })
    top_configs_path.write_text(
        json.dumps(top_configs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if verbose and kept_sorted:
        label = "pp_score" if rank_by_norm == "pp" else "score"
        print("Top runs:")
        for rank, (score, cfg_id, _, metrics, breakdown) in enumerate(kept_sorted, start=1):
            print(
                f"  {rank:02d}. {cfg_id} {label}={score:.4f} "
                f"quality={breakdown['quality']:.4f} swap={breakdown['swap_rate']:.4f} "
                f"missing_used={breakdown['missing_rate']:.4f} outlier_used={breakdown['outlier_rate']:.4f} "
                f"sanity_used={breakdown['sanity_reject_rate']:.4f}"
            )

    return [
        {
            "score": score,
            "overall_score": float(breakdown.get("overall_score") or 0.0),
            "pp_score": float(breakdown.get("pp_score") or 0.0),
            "rank_by": rank_by_norm,
            "config_id": cfg_id,
            "run_dir": str(top_root / f"top_{rank:02d}_{score:.4f}_{cfg_id}"),
            "metrics": metrics,
            "breakdown": breakdown,
        }
        for rank, (score, cfg_id, _, metrics, breakdown) in enumerate(kept_sorted, start=1)
    ]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Tune grid search parameters for keypoint extractor.")
    ap.add_argument("--in-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--grid", type=str, required=True)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    workers = max(1, int(args.workers) if args.workers else 1)
    print(f"workers={workers}")
    base_args, grid, score_weights, pp_score_weights, rank_by = load_grid_config(args.grid)
    args_dicts = build_arg_dicts(base_args, grid)
    if args.max_runs and args.max_runs > 0:
        args_dicts = args_dicts[: args.max_runs]
    run_grid_search(
        args.in_dir,
        args.out_dir,
        args_dicts,
        top_k=int(args.top_k),
        resume=bool(args.resume),
        score_weights=score_weights,
        pp_score_weights=pp_score_weights,
        rank_by=rank_by,
        workers=workers,
        verbose=True,
    )
    return 0


def prepare_runs(
    in_dir: str | Path,
    out_dir: str | Path,
    args_dicts: List[Dict[str, Any]],
    *,
    resume: bool = False,
) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for args_dict in args_dicts:
        normalized = normalize_args_dict(args_dict)
        run_dir = build_run_dir(out_dir, normalized)
        videos_parquet = run_dir / "videos.parquet"
        runs_parquet = run_dir / "runs.parquet"
        argv = build_run_argv(in_dir, out_dir, normalized)
        run_args = build_run_args(normalized)
        run = {
            "id": run_dir.name,
            "run_dir": run_dir,
            "videos_parquet_path": videos_parquet,
            "runs_parquet_path": runs_parquet,
            "argv": argv,
            "args": normalized,
            "run_args": run_args,
            "skipped": False,
            "video_metrics": None,
            "aggregate": None,
        }
        if resume and videos_parquet.exists():
            metrics = _metrics_from_video_rows(_load_parquet_rows(videos_parquet))
            if metrics:
                run["skipped"] = True
                run["video_metrics"] = metrics
                run["aggregate"] = metrics
        runs.append(run)
    return runs


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    raise SystemExit(main())
