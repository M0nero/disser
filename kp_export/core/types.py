
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

HandState = Literal["missing", "occluded", "predicted", "observed"]
HandSource = Literal["pass1", "pass2", "tracked", "occluded", "hold", None]

@dataclass
class VideoMeta:
    id: str
    file: str
    num_frames: int
    fps: float
    hands_frames: int
    hands_coverage: float
    left_score_mean: float
    right_score_mean: float
    left_coverage: float = 0.0
    right_coverage: float = 0.0
    both_coverage: float = 0.0
    pose_coverage: float = 0.0
    pose_interpolated_frac: float = 0.0
    dt_median_ms: float = 0.0
    fps_est: float = 0.0
    left_score_median: float = 0.0
    right_score_median: float = 0.0
    quality_score: float = 0.0
    sp_recovered_left_frac: float = 0.0
    sp_recovered_right_frac: float = 0.0
    track_recovered_left_frac: float = 0.0
    track_recovered_right_frac: float = 0.0
    file_pp: str = ""
    pp_filled_left: int = 0
    pp_filled_right: int = 0
    pp_gaps_filled_left: int = 0
    pp_gaps_filled_right: int = 0
    pp_smoothing_delta_left: float = 0.0
    pp_smoothing_delta_right: float = 0.0
