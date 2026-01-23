from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from time import perf_counter

from ..mp.roi import run_second_pass_for


def _execute_second_pass(hand, pts, score, *, debug_roi: bool = False, **kwargs):
    if kwargs.get("hands_sp") is None:
        return pts, score, False, 0.0, None
    t0 = perf_counter()
    pts, score, recovered, roi = run_second_pass_for(
        hand,
        cur_pts=pts,
        cur_score=score,
        debug_return_roi=debug_roi,
        **kwargs,
    )
    return pts, score, recovered, perf_counter() - t0, roi


def _apply_hold_if_needed(
    pts: Optional[List[Dict[str, float]]],
    last_export: Optional[List[Dict[str, float]]],
    hold_count: int,
    source: Optional[str],
    interp_hold: int,
) -> Tuple[Optional[List[Dict[str, float]]], Optional[str], int]:
    """
    Apply hold interpolation if hand is not detected.
    If pts is None and we have a previous export, use it for up to interp_hold frames.
    Note: When pts is not None (hand detected), hold_count is reset to 0.
    """
    if pts is None and last_export is not None and hold_count < max(0, interp_hold):
        return last_export, (source or "hold"), hold_count + 1
    return pts, source, 0  # Reset hold counter when hand is detected or hold limit exceeded
