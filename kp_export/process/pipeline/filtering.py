from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ...algos.sanity import check_hand_sanity


def append_reject_reason(cur: Optional[str], reason: Optional[str]) -> Optional[str]:
    if not reason:
        return cur
    if cur:
        return f"{cur}; {reason}"
    return reason


def score_for_gate(score_raw: Optional[float], present: bool, score_source: str) -> Optional[float]:
    if not present:
        return None
    if score_source == "presence":
        return 1.0
    return score_raw


def apply_sanity_stage(
    pts,
    score,
    reject_reason: Optional[str],
    *,
    enabled: bool,
    prev_anchor,
    prev_pred,
    world_coords: bool,
    scale_range,
    wrist_k: float,
    bone_tol: float,
    stage: str,
):
    if pts is None or not enabled:
        return pts, score, reject_reason, None, None
    debug_out: Dict[str, Any] = {}
    result = check_hand_sanity(
        pts,
        prev_anchor=prev_anchor,
        prev_pred=prev_pred,
        world_coords=world_coords,
        scale_range=scale_range,
        wrist_k=wrist_k,
        bone_tol=bone_tol,
        debug_out=debug_out,
    )
    if not result.ok:
        reason = "sanity:" + "|".join(result.reason_codes)
        return None, None, append_reject_reason(reject_reason, reason), debug_out, stage
    return pts, score, reject_reason, debug_out, stage
