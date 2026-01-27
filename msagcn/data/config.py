from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .topology import POSE_KEEP_DEFAULT


@dataclass
class DSConfig:
    max_frames: int = 64
    # How to interpret CSV 'end' index: inclusive is common in begin/end annotations.
    end_inclusive: bool = True
    # Temporal strategy inside the annotated [begin,end] segment.
    #
    # 1) "random" / "best" / "center"  -> choose a contiguous window of length max_frames
    #    inside the segment (if the segment is longer than max_frames). This can *cut off* part
    #    of the gesture if it's long.
    #
    # 2) "resample" -> take the whole [begin,end] segment and time-resample it to exactly
    #    max_frames (nearest-neighbour over frame indices). This keeps both segment boundaries
    #    and avoids throwing away the start/end of the gesture.
    temporal_crop: str = "random"
    use_streams: Tuple[str, ...] = ("joints", "bones", "velocity")
    include_pose: bool = False
    pose_keep: Tuple[int, ...] = tuple(POSE_KEEP_DEFAULT)
    pose_vis_thr: float = 0.5
    connect_cross_edges: bool = False
    hand_score_thr: float = 0.45
    hand_score_thr_fallback: float = 0.35
    window_valid_ratio: float = 0.60
    window_valid_ratio_fallback: float = 0.50
    center: bool = False
    center_mode: str = "masked_mean"
    normalize: bool = False
    norm_method: str = "p95"
    norm_scale: float = 1.0
    augment: bool = False
    mirror_prob: float = 0.5
    rot_deg: float = 10.0
    scale_jitter: float = 0.10
    noise_sigma: float = 0.01
    mirror_swap_only: bool = False
    time_drop_prob: float = 0.0
    hand_drop_prob: float = 0.0
    # Small temporal augmentations (train only):
    # - boundary jitter slightly perturbs begin/end to be robust to annotation noise.
    boundary_jitter_prob: float = 0.3
    boundary_jitter_max: int = 2
    # - speed perturbation mildly warps temporal sampling by resampling the segment to K frames
    #   (K in [kmin, kmax]) and then uniformly mapping to max_frames. Keeps segment boundaries.
    speed_perturb_prob: float = 0.3
    speed_perturb_kmin: int = 60
    speed_perturb_kmax: int = 68
    file_cache_size: int = 64
    prefer_pp: bool = True
    thr_tune_steps: int = 6
    thr_tune_step: float = 0.05
    min_achieved_ratio_train: float = 0.50

