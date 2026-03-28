from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bio.core.datasets.synth_dataset import _compute_sequence_geometry, load_prelabel_index


def _length_stats(values: Sequence[int]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.int32)
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }


def _iter_runs(flags: np.ndarray, *, value: int) -> Iterable[Tuple[int, int]]:
    start = None
    for idx, raw in enumerate(flags.tolist()):
        cur = int(raw)
        if cur == int(value):
            if start is None:
                start = idx
        elif start is not None:
            yield int(start), int(idx)
            start = None
    if start is not None:
        yield int(start), int(flags.shape[0])


def _low_motion_run_lengths(pts: np.ndarray, mask: np.ndarray, *, motion_epsilon: float) -> List[int]:
    if pts.shape[0] <= 1:
        return []
    geom = _compute_sequence_geometry(pts, mask)
    wrist_centers = np.asarray(geom["wrist_centers"], dtype=np.float32)
    wrist_valid = np.asarray(geom["wrist_valid"], dtype=np.uint8)
    deltas = np.zeros((pts.shape[0],), dtype=np.float32)
    for idx in range(1, int(pts.shape[0])):
        if bool(wrist_valid[idx]) and bool(wrist_valid[idx - 1]):
            deltas[idx] = float(np.linalg.norm(wrist_centers[idx] - wrist_centers[idx - 1]))
        else:
            deltas[idx] = np.inf
    quiet = (deltas <= float(max(0.0, motion_epsilon))).astype(np.uint8)
    out: List[int] = []
    for start, end in _iter_runs(quiet, value=1):
        length = int(end - start)
        if length > 0:
            out.append(length)
    return out


def _load_npz_sequence(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        pts = z["pts_raw"] if "pts_raw" in z else z["pts"]
        mask = z["mask"]
    return np.asarray(pts, dtype=np.float32), np.asarray(mask, dtype=np.float32)


def _collect_session_npz_paths(session_dirs: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    for root in session_dirs:
        path = root / "canonical_sequence.npz"
        if path.exists():
            out.append(path)
    return out


def _collect_prelabel_npz_paths(prelabel_dirs: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    for root in prelabel_dirs:
        rows = load_prelabel_index(root)
        for row in rows:
            npz_path = root / row.path_to_npz
            if npz_path.exists():
                out.append(npz_path)
    return out


def build_continuous_stats(
    *,
    session_dirs: Sequence[Path],
    prelabel_dirs: Sequence[Path],
    motion_epsilon: float,
) -> Dict[str, Any]:
    npz_paths = _collect_session_npz_paths(session_dirs) + _collect_prelabel_npz_paths(prelabel_dirs)
    if not npz_paths:
        raise RuntimeError("No canonical_sequence.npz or Step1 npz files found for continuous stats extraction.")

    sequence_lengths: List[int] = []
    startup_no_hand_lengths: List[int] = []
    no_hand_run_lengths: List[int] = []
    hand_visible_run_lengths: List[int] = []
    low_motion_run_lengths: List[int] = []
    center_delta_values: List[float] = []
    scale_delta_values: List[float] = []

    for path in npz_paths:
        pts, mask = _load_npz_sequence(path)
        if pts.ndim != 3 or mask.ndim != 3 or int(pts.shape[0]) <= 0:
            continue
        T = int(pts.shape[0])
        sequence_lengths.append(T)
        visible = (np.asarray(mask, dtype=np.float32)[..., 0].sum(axis=-1) > 0.5).astype(np.uint8)
        no_hand_runs = [int(end - start) for start, end in _iter_runs(visible, value=0)]
        hand_runs = [int(end - start) for start, end in _iter_runs(visible, value=1)]
        no_hand_run_lengths.extend(no_hand_runs)
        hand_visible_run_lengths.extend(hand_runs)
        if visible.size > 0 and int(visible[0]) == 0 and no_hand_runs:
            startup_no_hand_lengths.append(int(no_hand_runs[0]))
        geom = _compute_sequence_geometry(pts, mask)
        centers = np.asarray(geom["centers"], dtype=np.float32)
        center_valid = np.asarray(geom["center_valid"], dtype=np.uint8)
        scales = np.asarray(geom["scales"], dtype=np.float32)
        scale_valid = np.asarray(geom["scale_valid"], dtype=np.uint8)
        for idx in range(1, T):
            if bool(center_valid[idx]) and bool(center_valid[idx - 1]):
                center_delta_values.append(float(np.linalg.norm(centers[idx] - centers[idx - 1])))
            if bool(scale_valid[idx]) and bool(scale_valid[idx - 1]):
                scale_delta_values.append(float(abs(float(scales[idx]) - float(scales[idx - 1]))))
        low_motion_run_lengths.extend(_low_motion_run_lengths(pts, mask, motion_epsilon=motion_epsilon))

    gap_lengths = list(no_hand_run_lengths) + list(low_motion_run_lengths)
    payload = {
        "version": 1,
        "source": {
            "session_dirs": [str(x) for x in session_dirs],
            "prelabel_dirs": [str(x) for x in prelabel_dirs],
            "num_sequences": int(len(sequence_lengths)),
            "motion_epsilon": float(motion_epsilon),
        },
        "gap_lengths": [int(x) for x in gap_lengths if int(x) > 0],
        "leading_noev_lengths": [int(x) for x in startup_no_hand_lengths if int(x) > 0],
        "all_noev_lengths": [int(x) for x in sequence_lengths if int(x) > 0],
        "diagnostics": {
            "sequence_lengths": _length_stats(sequence_lengths),
            "startup_no_hand_lengths": _length_stats(startup_no_hand_lengths),
            "no_hand_run_lengths": _length_stats(no_hand_run_lengths),
            "hand_visible_run_lengths": _length_stats(hand_visible_run_lengths),
            "low_motion_run_lengths": _length_stats(low_motion_run_lengths),
            "center_delta_mean": float(np.mean(center_delta_values)) if center_delta_values else 0.0,
            "scale_delta_mean": float(np.mean(scale_delta_values)) if scale_delta_values else 0.0,
        },
    }
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract empirical continuous-video stats for synth-build.")
    ap.add_argument("--session_dir", action="append", default=[], help="Review session directory containing canonical_sequence.npz.")
    ap.add_argument("--prelabel_dir", action="append", default=[], help="Canonical Step1 directory with index.json/index.csv and npz files.")
    ap.add_argument("--out_dir", type=str, required=True, help="Directory where continuous_stats.json will be written.")
    ap.add_argument("--motion_epsilon", type=float, default=0.01, help="Wrist-center delta threshold for low-motion runs.")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_continuous_stats(
        session_dirs=[Path(x) for x in args.session_dir],
        prelabel_dirs=[Path(x) for x in args.prelabel_dir],
        motion_epsilon=float(args.motion_epsilon),
    )
    out_path = out_dir / "continuous_stats.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_path": str(out_path), "num_sequences": payload["source"]["num_sequences"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
