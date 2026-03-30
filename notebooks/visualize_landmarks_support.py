from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr


def _artifact_paths(root: Path) -> dict[str, Path]:
    base = Path(root)
    if base.is_file():
        base = base.parent
    return {
        "root": base,
        "zarr": base / "landmarks.zarr",
        "videos": base / "videos.parquet",
        "frames": base / "frames.parquet",
    }


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def _row_to_clean_dict(row: pd.Series) -> dict[str, Any]:
    return {key: _clean_value(value) for key, value in row.to_dict().items()}


def _points_from_xyz(xyz: np.ndarray | None, valid: bool | None = True) -> list[dict[str, float]] | None:
    if xyz is None or valid is False:
        return None
    arr = np.asarray(xyz)
    if arr.ndim != 2 or arr.shape[-1] < 2:
        return None
    points: list[dict[str, float]] = []
    for point in arr:
        x = float(point[0])
        y = float(point[1])
        z = float(point[2]) if point.shape[0] > 2 else 0.0
        points.append({"x": x, "y": y, "z": z})
    return points


def _hand_changed(raw_xyz: np.ndarray, raw_valid: bool, pp_xyz: np.ndarray, pp_valid: bool, *, atol: float = 1e-6) -> bool:
    if bool(raw_valid) != bool(pp_valid):
        return True
    if not raw_valid and not pp_valid:
        return False
    return not np.allclose(np.asarray(raw_xyz), np.asarray(pp_xyz), atol=atol, rtol=0.0, equal_nan=True)


@functools.lru_cache(maxsize=8)
def _open_zarr_group(zarr_path: str):
    return zarr.open(zarr_path, mode="r")


def _read_parquet_filtered(path: Path, sample_id: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, filters=[("sample_id", "==", sample_id)])
    except Exception:
        df = pd.read_parquet(path)
        return df[df["sample_id"] == sample_id].copy()


@functools.lru_cache(maxsize=8)
def sample_catalog(root: str | Path) -> pd.DataFrame:
    paths = _artifact_paths(Path(root))
    df = pd.read_parquet(paths["videos"])
    cols = [c for c in ("sample_id", "source_video", "num_frames", "has_pp", "coords_mode", "fps", "fps_est") if c in df.columns]
    return df[cols].sort_values("sample_id").reset_index(drop=True)


def list_samples(root: str | Path) -> list[str]:
    try:
        return sample_catalog(str(root))["sample_id"].tolist()
    except Exception:
        return []


def guess_video_path(meta: dict, artifact_root: str | Path | None = None) -> Path | None:
    if isinstance(meta, dict):
        for key in ("video", "input", "input_file", "source_file", "source_video"):
            candidate = meta.get(key)
            if not candidate:
                continue
            p = Path(candidate)
            if p.exists():
                return p
            if artifact_root:
                base = Path(artifact_root)
                base_dir = base if base.is_dir() else base.parent
                anchors = [base_dir, *base_dir.parents]
                for anchor in anchors:
                    q = anchor / p
                    if q.exists():
                        return q
    return None


def _build_meta(video_row: pd.Series, frame_rows: pd.DataFrame, sample_attrs: dict[str, Any]) -> dict[str, Any]:
    frame0 = _row_to_clean_dict(frame_rows.iloc[0]) if not frame_rows.empty else {}
    meta = {key: _clean_value(value) for key, value in dict(sample_attrs).items()}
    meta.update(_row_to_clean_dict(video_row))
    meta["video"] = meta.get("source_video")
    meta["input_file"] = meta.get("source_video")
    meta["source_file"] = meta.get("source_video")
    meta["hand_score_gate"] = {
        "lo": frame0.get("hand_score_lo"),
        "hi": frame0.get("hand_score_hi"),
        "min_hand_score_legacy": frame0.get("min_hand_score"),
        "pose_dist_qual_min": frame0.get("pose_dist_qual_min"),
    }
    return meta


def _build_variant_frames(
    frame_rows: pd.DataFrame,
    variant_group,
    *,
    raw_group=None,
    variant_name: str,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    left_xyz = np.asarray(variant_group["left_xyz"])
    right_xyz = np.asarray(variant_group["right_xyz"])
    pose_xyz = np.asarray(variant_group["pose_xyz"])
    left_score = np.asarray(variant_group["left_score"])
    right_score = np.asarray(variant_group["right_score"])
    left_valid = np.asarray(variant_group["left_valid"])
    right_valid = np.asarray(variant_group["right_valid"])
    ts_ms = np.asarray(variant_group["ts_ms"])

    if raw_group is not None:
        raw_left_xyz = np.asarray(raw_group["left_xyz"])
        raw_right_xyz = np.asarray(raw_group["right_xyz"])
        raw_left_valid = np.asarray(raw_group["left_valid"])
        raw_right_valid = np.asarray(raw_group["right_valid"])
    else:
        raw_left_xyz = left_xyz
        raw_right_xyz = right_xyz
        raw_left_valid = left_valid
        raw_right_valid = right_valid

    for i, (_, row) in enumerate(frame_rows.iterrows()):
        fr = _row_to_clean_dict(row)
        fr["idx"] = int(fr.get("frame_idx", i))
        fr["i"] = fr["idx"]
        fr["frame"] = fr["idx"]
        fr["ts"] = int(ts_ms[i]) if i < len(ts_ms) else fr.get("ts_ms")
        fr["dt"] = fr.get("dt_ms")
        fr["hand_1"] = _points_from_xyz(left_xyz[i], valid=bool(left_valid[i]))
        fr["hand_2"] = _points_from_xyz(right_xyz[i], valid=bool(right_valid[i]))
        fr["pose"] = _points_from_xyz(pose_xyz[i], valid=bool(fr.get("pose_present", True)))
        fr["hand_1_score"] = float(left_score[i]) if i < len(left_score) else fr.get("hand_1_score")
        fr["hand_2_score"] = float(right_score[i]) if i < len(right_score) else fr.get("hand_2_score")

        if variant_name == "pp":
            left_changed = _hand_changed(raw_left_xyz[i], bool(raw_left_valid[i]), left_xyz[i], bool(left_valid[i]))
            right_changed = _hand_changed(raw_right_xyz[i], bool(raw_right_valid[i]), right_xyz[i], bool(right_valid[i]))
            fr["hand_1_pp_applied"] = bool(left_changed)
            fr["hand_2_pp_applied"] = bool(right_changed)
            fr["hand_1_pp_overrode"] = False
            fr["hand_2_pp_overrode"] = False
            fr["hand_1_pp_reason"] = "zarr_pp_diff" if left_changed else None
            fr["hand_2_pp_reason"] = "zarr_pp_diff" if right_changed else None
            fr["hand_1_pp_overrode_reason"] = None
            fr["hand_2_pp_overrode_reason"] = None
            if fr["hand_1"] is not None and not bool(raw_left_valid[i]):
                fr["hand_1_source"] = "interp"
                fr["hand_1_state"] = "predicted"
            if fr["hand_2"] is not None and not bool(raw_right_valid[i]):
                fr["hand_2_source"] = "interp"
                fr["hand_2_state"] = "predicted"
        else:
            fr["hand_1_pp_applied"] = False
            fr["hand_2_pp_applied"] = False
            fr["hand_1_pp_overrode"] = False
            fr["hand_2_pp_overrode"] = False
            fr["hand_1_pp_reason"] = None
            fr["hand_2_pp_reason"] = None
            fr["hand_1_pp_overrode_reason"] = None
            fr["hand_2_pp_overrode_reason"] = None

        frames.append(fr)
    return frames


def load_runs_from_artifact(root: str | Path, sample_id: str | None, *, use_pp: bool = True) -> dict[str, Any]:
    paths = _artifact_paths(Path(root))
    if not sample_id:
        samples = list_samples(paths["root"])
        sample_id = samples[0] if samples else None
    if not sample_id:
        return {"raw": {"path": paths["root"], "meta": {}, "frames": []}, "pp": None}

    video_rows = _read_parquet_filtered(paths["videos"], sample_id)
    frame_rows = _read_parquet_filtered(paths["frames"], sample_id).sort_values("frame_idx").reset_index(drop=True)
    if video_rows.empty:
        raise KeyError(f"sample_id not found in videos.parquet: {sample_id}")
    video_row = video_rows.iloc[0]

    root_group = _open_zarr_group(str(paths["zarr"]))
    sample_group = root_group["samples"][sample_id]
    meta = _build_meta(video_row, frame_rows, dict(sample_group.attrs))
    raw_frames = _build_variant_frames(frame_rows, sample_group["raw"], variant_name="raw")
    pp_run = None
    if use_pp and "pp" in sample_group:
        pp_frames = _build_variant_frames(frame_rows, sample_group["pp"], raw_group=sample_group["raw"], variant_name="pp")
        pp_run = {"path": paths["root"], "meta": dict(meta), "frames": pp_frames}
    return {
        "raw": {"path": paths["root"], "meta": dict(meta), "frames": raw_frames},
        "pp": pp_run,
    }
