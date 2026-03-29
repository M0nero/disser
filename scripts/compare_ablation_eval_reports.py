#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq


def _f(val: Any) -> float:
    try:
        if val is None:
            return float("nan")
        return float(val)
    except Exception:
        return float("nan")


def _safe_rate(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or den <= 0:
        return float("nan")
    return num / den


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _video_id(v: Dict[str, Any]) -> str:
    vid = str(v.get("sample_id") or "").strip()
    if vid:
        return vid
    slug = str(v.get("slug") or "").strip()
    if slug:
        return slug
    out = str(v.get("output") or "").strip()
    if out:
        return Path(out.replace("\\", "/")).stem
    return ""


def _extract_video_metrics(v: Dict[str, Any]) -> Optional[Dict[str, float]]:
    num_frames = _f(v.get("num_frames"))
    if not math.isfinite(num_frames) or num_frames <= 0:
        return None

    out: Dict[str, float] = {
        "num_frames": num_frames,
        "quality_score": _f(v.get("quality_score")),
        "hands_coverage": _f(v.get("hands_coverage")),
        "left_coverage": _f(v.get("left_coverage")),
        "right_coverage": _f(v.get("right_coverage")),
        "both_coverage": _f(v.get("both_coverage")),
        "sp_recovered_left_frac": _f(v.get("sp_recovered_left_frac")),
        "sp_recovered_right_frac": _f(v.get("sp_recovered_right_frac")),
        "track_recovered_left_frac": _f(v.get("track_recovered_left_frac")),
        "track_recovered_right_frac": _f(v.get("track_recovered_right_frac")),
    }

    missing = _f(v.get("missing_frames_1")) + _f(v.get("missing_frames_2"))
    occluded = _f(v.get("occluded_frames_1")) + _f(v.get("occluded_frames_2"))
    outlier = _f(v.get("outlier_frames_1")) + _f(v.get("outlier_frames_2"))
    sanity = _f(v.get("sanity_reject_frames_1")) + _f(v.get("sanity_reject_frames_2"))
    swap = _f(v.get("swap_frames"))
    pp_fill = _f(v.get("pp_filled_left")) + _f(v.get("pp_filled_right"))
    out.update(
        {
            "missing_rate": _safe_rate(missing, 2.0 * num_frames),
            "occluded_rate": _safe_rate(occluded, 2.0 * num_frames),
            "outlier_rate": _safe_rate(outlier, 2.0 * num_frames),
            "sanity_reject_rate": _safe_rate(sanity, 2.0 * num_frames),
            "swap_rate": _safe_rate(swap, num_frames),
            "pp_filled_frac": _safe_rate(pp_fill, 2.0 * num_frames),
        }
    )
    return out


def _build_map(videos: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for v in videos:
        if not isinstance(v, dict):
            continue
        vid = _video_id(v)
        if not vid:
            continue
        metrics = _extract_video_metrics(v)
        if metrics is None:
            continue
        out[vid] = metrics
    return out


def _mean(vals: List[float]) -> float:
    if not vals:
        return float("nan")
    return sum(vals) / float(len(vals))


def _median(vals: List[float]) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    n = len(s)
    m = n // 2
    if n % 2:
        return float(s[m])
    return (s[m - 1] + s[m]) / 2.0


def _fmt(v: float, nd: int = 6) -> str:
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{nd}f}"


def _compare_metric(
    metric: str,
    naive_map: Dict[str, Dict[str, float]],
    final_map: Dict[str, Dict[str, float]],
    common_ids: List[str],
    higher_is_better: bool,
) -> Dict[str, Any]:
    pairs: List[Tuple[float, float]] = []
    for vid in common_ids:
        a = _f(naive_map[vid].get(metric))
        b = _f(final_map[vid].get(metric))
        if math.isfinite(a) and math.isfinite(b):
            pairs.append((a, b))
    naive_vals = [p[0] for p in pairs]
    final_vals = [p[1] for p in pairs]
    naive_mean = _mean(naive_vals)
    final_mean = _mean(final_vals)
    naive_med = _median(naive_vals)
    final_med = _median(final_vals)
    delta = final_mean - naive_mean

    rel = float("nan")
    if math.isfinite(naive_mean) and abs(naive_mean) > 1e-12:
        if higher_is_better:
            rel = (final_mean - naive_mean) / abs(naive_mean)
        else:
            rel = (naive_mean - final_mean) / abs(naive_mean)

    return {
        "metric": metric,
        "n": len(pairs),
        "naive_mean": naive_mean,
        "final_mean": final_mean,
        "delta_mean": delta,
        "naive_median": naive_med,
        "final_median": final_med,
        "improvement_frac": rel,
        "higher_is_better": higher_is_better,
    }


def _as_markdown(
    naive_path: Path,
    final_path: Path,
    naive_map: Dict[str, Dict[str, float]],
    final_map: Dict[str, Dict[str, float]],
    rows: List[Dict[str, Any]],
    common_ids: List[str],
) -> str:
    lines: List[str] = []
    lines.append("# Ablation: pass-1 naive vs final fault-tolerant")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- naive parquet: `{naive_path}`")
    lines.append(f"- final parquet: `{final_path}`")
    lines.append(f"- videos with valid meta: naive={len(naive_map)}, final={len(final_map)}")
    lines.append(f"- matched by video id: {len(common_ids)}")
    lines.append("")
    lines.append("## Metric Comparison (matched subset)")
    lines.append("")
    lines.append("| Metric | n | naive mean | final mean | delta (final-naive) | improvement |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        imp = r["improvement_frac"]
        imp_txt = "nan" if not math.isfinite(imp) else f"{imp*100.0:+.2f}%"
        lines.append(
            f"| {r['metric']} | {r['n']} | {_fmt(r['naive_mean'])} | {_fmt(r['final_mean'])} | {_fmt(r['delta_mean'])} | {imp_txt} |"
        )
    lines.append("")
    lines.append("improvement convention:")
    lines.append("- higher-is-better metrics: positive means final is better")
    lines.append("- lower-is-better metrics: positive means final has lower error")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser("Compare naive vs final videos.parquet outputs for ablation.")
    ap.add_argument("--naive", default="outputs/Slovo_naive/videos.parquet")
    ap.add_argument("--final", default="outputs/Slovo/videos.parquet")
    ap.add_argument("--out-dir", default="outputs/ablation")
    args = ap.parse_args()

    naive_path = Path(args.naive)
    final_path = Path(args.final)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    naive_map = _build_map(_load_rows(naive_path))
    final_map = _build_map(_load_rows(final_path))
    common_ids = sorted(set(naive_map.keys()) & set(final_map.keys()))

    metric_specs = [
        ("quality_score", True),
        ("hands_coverage", True),
        ("left_coverage", True),
        ("right_coverage", True),
        ("both_coverage", True),
        ("missing_rate", False),
        ("occluded_rate", False),
        ("swap_rate", False),
        ("outlier_rate", False),
        ("sanity_reject_rate", False),
        ("pp_filled_frac", True),
        ("sp_recovered_left_frac", True),
        ("sp_recovered_right_frac", True),
        ("track_recovered_left_frac", True),
        ("track_recovered_right_frac", True),
    ]

    rows = [
        _compare_metric(metric, naive_map, final_map, common_ids, higher_is_better)
        for metric, higher_is_better in metric_specs
    ]

    summary = {
        "naive_report": str(naive_path),
        "final_report": str(final_path),
        "naive_videos": len(naive_map),
        "final_videos": len(final_map),
        "matched_videos": len(common_ids),
        "rows": rows,
    }

    out_json = out_dir / "pass1_vs_final.summary.json"
    out_md = out_dir / "pass1_vs_final.md"
    out_csv = out_dir / "pass1_vs_final.csv"

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(
        _as_markdown(naive_path, final_path, naive_map, final_map, rows, common_ids),
        encoding="utf-8",
    )
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "metric",
                "n",
                "naive_mean",
                "final_mean",
                "delta_mean",
                "naive_median",
                "final_median",
                "improvement_frac",
                "higher_is_better",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["metric"],
                    r["n"],
                    r["naive_mean"],
                    r["final_mean"],
                    r["delta_mean"],
                    r["naive_median"],
                    r["final_median"],
                    r["improvement_frac"],
                    r["higher_is_better"],
                ]
            )

    print(f"[OK] matched videos: {len(common_ids)}")
    print(f"[OK] summary json: {out_json}")
    print(f"[OK] summary md:   {out_md}")
    print(f"[OK] summary csv:  {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
