from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


def build_sample_weights(
    samples: Sequence[Tuple[str, str, int, Optional[int]]],
    label2idx: Dict[str, int],
    meta_by_vid: Dict[str, Dict[str, Any]],
    quality_floor: float = 0.4,
    quality_power: float = 1.0,
    cover_key: str = "both_coverage",
    cover_floor: float = 0.3,
) -> List[float]:
    freq: Dict[str, int] = {lbl: 0 for lbl in label2idx.keys()}
    for _, lbl, _, _ in samples:
        freq[lbl] = freq.get(lbl, 0) + 1

    weights: List[float] = []
    for vid, lbl, _, _ in samples:
        w_class = 1.0 / max(1, freq[lbl])
        meta = meta_by_vid.get(vid, {})
        q = max(float(meta.get("quality_score", 1.0)), quality_floor)
        cov = max(float(meta.get(cover_key, 0.0)), cover_floor)
        wq = q**quality_power
        wc = 0.5 + 0.5 * cov
        weights.append(w_class * wq * wc)
    return weights

