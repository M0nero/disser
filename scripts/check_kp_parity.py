from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List

import pyarrow.parquet as pq
import zarr


def _load_table(path: Path) -> List[Dict]:
    if not path.exists():
        raise SystemExit(f"Missing parquet: {path}")
    return pq.read_table(path).to_pylist()


def _index_rows(rows: Iterable[Dict], key: str) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        out[str(value)] = dict(row)
    return out


def _float_equal(left, right, tol: float) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(float(left), float(right), rel_tol=tol, abs_tol=tol)


def _compare_video_rows(left_rows: Dict[str, Dict], right_rows: Dict[str, Dict], *, tol: float) -> List[str]:
    errors: List[str] = []
    if set(left_rows) != set(right_rows):
        missing_left = sorted(set(right_rows) - set(left_rows))
        missing_right = sorted(set(left_rows) - set(right_rows))
        if missing_left:
            errors.append(f"Missing in baseline: {', '.join(missing_left[:10])}")
        if missing_right:
            errors.append(f"Missing in candidate: {', '.join(missing_right[:10])}")
        return errors
    for sample_id in sorted(left_rows):
        left = left_rows[sample_id]
        right = right_rows[sample_id]
        for key in ("num_frames", "pose_joint_count", "coords_mode", "has_pp"):
            if left.get(key) != right.get(key):
                errors.append(f"{sample_id}: mismatch {key}: {left.get(key)} != {right.get(key)}")
        for key in (
            "quality_score",
            "hands_coverage",
            "left_coverage",
            "right_coverage",
            "both_coverage",
            "pose_coverage",
            "pose_interpolated_frac",
        ):
            if not _float_equal(left.get(key), right.get(key), tol):
                errors.append(f"{sample_id}: mismatch {key}: {left.get(key)} != {right.get(key)}")
    return errors


def _compare_zarr(left_path: Path, right_path: Path) -> List[str]:
    errors: List[str] = []
    left_root = zarr.open_group(str(left_path), mode="r")
    right_root = zarr.open_group(str(right_path), mode="r")
    left_samples = left_root["samples"]
    right_samples = right_root["samples"]
    left_ids = set(left_samples.group_keys())
    right_ids = set(right_samples.group_keys())
    if left_ids != right_ids:
        errors.append("Zarr sample ids differ")
        return errors
    for sample_id in sorted(left_ids):
        left_group = left_samples[sample_id]
        right_group = right_samples[sample_id]
        for variant in ("raw", "pp"):
            left_has = variant in left_group
            right_has = variant in right_group
            if left_has != right_has:
                errors.append(f"{sample_id}: variant presence differs for {variant}")
                continue
            if not left_has:
                continue
            for array_name in left_group[variant].array_keys():
                if array_name not in right_group[variant]:
                    errors.append(f"{sample_id}: missing array {variant}/{array_name} in candidate")
                    continue
                left_arr = left_group[variant][array_name]
                right_arr = right_group[variant][array_name]
                if left_arr.shape != right_arr.shape or str(left_arr.dtype) != str(right_arr.dtype):
                    errors.append(
                        f"{sample_id}: shape/dtype mismatch for {variant}/{array_name}: "
                        f"{left_arr.shape}/{left_arr.dtype} != {right_arr.shape}/{right_arr.dtype}"
                    )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("Compare two kp_export artifact roots for parity.")
    parser.add_argument("--baseline", required=True, type=str)
    parser.add_argument("--candidate", required=True, type=str)
    parser.add_argument("--float-tol", type=float, default=1e-5)
    args = parser.parse_args(argv)

    baseline = Path(args.baseline)
    candidate = Path(args.candidate)
    left_videos = _index_rows(_load_table(baseline / "videos.parquet"), "sample_id")
    right_videos = _index_rows(_load_table(candidate / "videos.parquet"), "sample_id")

    errors = []
    errors.extend(_compare_video_rows(left_videos, right_videos, tol=float(args.float_tol)))
    errors.extend(_compare_zarr(baseline / "landmarks.zarr", candidate / "landmarks.zarr"))

    left_frames = _load_table(baseline / "frames.parquet")
    right_frames = _load_table(candidate / "frames.parquet")
    if len(left_frames) != len(right_frames):
        errors.append(f"Frame row count mismatch: {len(left_frames)} != {len(right_frames)}")

    if errors:
        for error in errors[:50]:
            print(f"[PARITY-ERROR] {error}")
        return 1

    print("[PARITY-OK] artifacts match on sample ids, key metrics, frame counts, and zarr array shapes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
