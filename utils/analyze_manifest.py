import argparse, json, math, statistics as stats, csv, os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

METRIC_KEYS = [
    "both_coverage", "left_coverage", "right_coverage",
    "pose_coverage", "pose_interpolated_frac",
    "left_score_median", "right_score_median",
    "hands_coverage", "fps_est", "dt_median_ms",
    "quality_score"
]

def _vec(videos: List[Dict[str,Any]], key: str, default: float=0.0) -> List[float]:
    out = []
    for v in videos:
        x = v.get(key, default)
        try:
            out.append(float(x))
        except Exception:
            out.append(default)
    return out

def _pct(x: List[float], p: float) -> float:
    if not x:
        return float('nan')
    return float(np.percentile(x, p))

def _hist(figpath: Path, values: List[float], title: str, bins: int=30):
    if not values:
        return
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()

def auto_recommend(videos: List[Dict[str,Any]]) -> Dict[str, Any]:
    rec = {}
    both = _vec(videos, "both_coverage")
    lmed = _vec(videos, "left_score_median")
    rmed = _vec(videos, "right_score_median")
    fpsv = _vec(videos, "fps_est")
    posei = _vec(videos, "pose_interpolated_frac")
    hands_cov = _vec(videos, "hands_coverage")

    # Basic centers
    med_both = np.nanmedian(both) if both else float('nan')
    med_l = np.nanmedian(lmed) if lmed else float('nan')
    med_r = np.nanmedian(rmed) if rmed else float('nan')
    med_score = np.nanmedian([x for x in lmed + rmed if not math.isnan(x)]) if (lmed or rmed) else float('nan')
    hi_fps_frac = np.mean([1.0 if x >= 50 else 0.0 for x in fpsv]) if fpsv else 0.0
    low_fps_frac = np.mean([1.0 if x <= 22 else 0.0 for x in fpsv]) if fpsv else 0.0
    med_posei = np.nanmedian(posei) if posei else float('nan')

    # Suggest min-hand-score ~ slightly below median score
    if not math.isnan(med_score) and med_score > 0.0:
        min_hand_score = max(0.2, min(0.45, med_score - 0.05))
    else:
        min_hand_score = 0.3

    # Suggest det/track and stride
    if not math.isnan(med_both) and med_both < 0.35:
        min_det = 0.5
        min_track = 0.5
    else:
        min_det = 0.6
        min_track = 0.6

    stride = 1
    if hi_fps_frac >= 0.4:
        stride = 2
    if low_fps_frac >= 0.6:
        stride = 1

    # Pose frequency & EMA
    pose_every = 1
    pose_ema = 0.0
    if med_posei > 0.35:
        # Many frames are interpolated -> compute pose more often or smooth
        pose_every = 1
        pose_ema = 0.4
    else:
        pose_every = 1
        pose_ema = 0.0

    rec["summary"] = {
        "median_both_coverage": round(float(med_both), 4) if not math.isnan(med_both) else None,
        "median_hand_score": round(float(med_score), 4) if not math.isnan(med_score) else None,
        "high_fps_fraction_(>=50fps)": round(float(hi_fps_frac), 3),
        "low_fps_fraction_(<=22fps)": round(float(low_fps_frac), 3),
        "median_pose_interpolated_frac": round(float(med_posei), 4) if not math.isnan(med_posei) else None
    }
    rec["recommended_flags"] = {
        "min_det": round(min_det,2),
        "min_track": round(min_track,2),
        "min_hand_score": round(min_hand_score,2),
        "stride": int(stride),
        "pose_every": int(pose_every),
        "pose_ema": round(pose_ema,2)
    }
    return rec

def _load_videos(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        return [dict(row) for row in pq.read_table(path).to_pylist()]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [dict(row) for row in data if isinstance(row, dict)]
    videos = data.get("videos", []) if isinstance(data, dict) else []
    return [dict(row) for row in videos if isinstance(row, dict)]


def main():
    ap = argparse.ArgumentParser("Analyze videos.parquet and suggest preprocessing flags")
    ap.add_argument("--manifest", type=str, required=True, help="Path to videos.parquet (or legacy manifest.json)")
    ap.add_argument("--out-dir", type=str, required=True, help="Directory to save the report")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = _load_videos(Path(args.manifest))
    if not videos:
        raise SystemExit("Empty input: no video/sample rows found.")

    # Save per-video CSV
    csv_path = out_dir / "metrics.csv"
    fieldnames = ["id","num_frames","fps","hands_frames","hands_coverage"] + METRIC_KEYS
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fieldnames)
        w.writeheader()
        for v in videos:
            row = {k: v.get(k, "") for k in fieldnames}
            w.writerow(row)

    # Global stats & histograms
    summary = {}
    for k in METRIC_KEYS:
        vec = [x for x in _vec(videos, k) if not math.isnan(x)]
        if not vec:
            continue
        summary[k] = {
            "count": len(vec),
            "mean": float(np.mean(vec)),
            "std": float(np.std(vec)),
            "p10": _pct(vec, 10),
            "p25": _pct(vec, 25),
            "p50": _pct(vec, 50),
            "p75": _pct(vec, 75),
            "p90": _pct(vec, 90),
        }
        # Plots
        _hist(out_dir / f"hist_{k}.png", vec, f"Histogram of {k}")

    # Recommendations
    rec = auto_recommend(videos)

    # Write summary.json
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fsum:
        json.dump({"metrics": summary, **rec}, fsum, ensure_ascii=False, indent=2)

    # Human-readable report.md
    md = []
    md.append("# Extractor Metrics Report\n")
    md.append(f"- Total videos: **{len(videos)}**\n")
    if "both_coverage" in summary:
        med = summary["both_coverage"]["p50"]
        md.append(f"- Median both_coverage: **{med:.3f}**\n")
    if "left_score_median" in summary and "right_score_median" in summary:
        lm = summary["left_score_median"]["p50"]
        rm = summary["right_score_median"]["p50"]
        md.append(f"- Median hand score (L/R): **{lm:.3f} / {rm:.3f}**\n")
    if "fps_est" in summary:
        md.append(f"- Median fps_est: **{summary['fps_est']['p50']:.2f}**\n")
    if "pose_interpolated_frac" in summary:
        md.append(f"- Median pose_interpolated_frac: **{summary['pose_interpolated_frac']['p50']:.3f}**\n")

    md.append("\n## Recommended flags\n")
    for k, v in rec["recommended_flags"].items():
        md.append(f"- **{k}**: `{v}`")
        md.append("\n")

    md.append("\n## Files\n")
    md.append("- `summary.json` — global stats & recommendations\n")
    md.append("- `metrics.csv` — per-video table for deep dives\n")
    md.append("- `hist_*.png` — histograms for key metrics\n")

    with open(out_dir / "report.md", "w", encoding="utf-8") as frep:
        frep.write("".join(md))

    print(f"Saved report to: {out_dir}")
    print("Recommended flags:", rec["recommended_flags"])

if __name__ == "__main__":
    main()
