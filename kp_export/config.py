from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class VideoConfig:
    in_dir: str
    out_dir: str
    pattern: str = "*.mp4"
    stride: int = 1
    short_side: int = 0
    ts_source: str = "auto"


@dataclass(frozen=True)
class PoseConfig:
    keep_pose_indices: List[int]
    world_coords: bool
    pose_every: int = 1
    pose_complexity: int = 1
    pose_ema: float = 0.0


@dataclass(frozen=True)
class ScoreConfig:
    min_det: float = 0.5
    min_track: float = 0.5
    min_hand_score: float = 0.0
    hand_score_lo: float = 0.55
    hand_score_hi: float = 0.90
    hand_score_source: str = "handedness"
    tracker_init_score: float = -1.0
    anchor_score: float = -1.0
    tracker_update_score: float = -1.0
    pose_dist_qual_min: float = 0.5
    pose_side_reassign_ratio: float = 0.85


@dataclass(frozen=True)
class SecondPassConfig:
    enabled: bool = False
    trigger_below: float = 0.5
    roi_frac: float = 0.25
    margin: float = 0.35
    escalate_step: float = 0.25
    escalate_max: float = 2.0
    hands_up_only: bool = False
    jitter_px: int = 0
    jitter_rings: int = 1
    center_penalty: float = 0.30
    label_relax: float = 0.15
    overlap_iou: float = 0.15
    overlap_shrink: float = 0.70
    overlap_penalty_mult: float = 2.0
    overlap_require_label: bool = False
    debug_roi: bool = False


@dataclass(frozen=True)
class TrackingConfig:
    interp_hold: int = 0
    write_hand_mask: bool = False
    max_gap: int = 15
    score_decay: float = 0.90
    reset_ms: int = 250


@dataclass(frozen=True)
class OcclusionConfig:
    hyst_frames: int = 15
    return_k: float = 1.2


@dataclass(frozen=True)
class SanityConfig:
    enabled: bool = True
    scale_range: Tuple[float, float] = (0.70, 1.35)
    wrist_k: float = 2.0
    bone_tol: float = 0.30
    pass2: bool = False
    anchor_max_gap: int = 0
    sanitize_rejects: bool = True


@dataclass(frozen=True)
class PostprocessConfig:
    enabled: bool = False
    max_gap: int = 15
    smoother: str = "ema"
    only_anchors: bool = True


@dataclass(frozen=True)
class MediaPipeConfig:
    backend: str = "solutions"
    hand_task: str = ""
    pose_task: str = ""
    tasks_delegate: str = "auto"


@dataclass(frozen=True)
class DebugConfig:
    ndjson: str = ""
    debug_video: str = ""


@dataclass(frozen=True)
class OutputConfig:
    stage_dir: str = ""


@dataclass(frozen=True)
class RuntimeConfig:
    jobs: int = 1
    seed: int = 0
    video_count: int = 1


@dataclass(frozen=True)
class LoggingConfig:
    log_dir: str = "outputs/logs"
    log_level: str = "INFO"


@dataclass(frozen=True)
class ExtractorConfig:
    video: VideoConfig
    pose: PoseConfig
    score: ScoreConfig
    second_pass: SecondPassConfig
    tracking: TrackingConfig
    occlusion: OcclusionConfig
    sanity: SanityConfig
    postprocess: PostprocessConfig
    mediapipe: MediaPipeConfig
    debug: DebugConfig
    output: OutputConfig
    runtime: RuntimeConfig
    logging: LoggingConfig

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractorConfig":
        return cls(
            video=VideoConfig(**dict(data.get("video", {}))),
            pose=PoseConfig(**dict(data.get("pose", {}))),
            score=ScoreConfig(**dict(data.get("score", {}))),
            second_pass=SecondPassConfig(**dict(data.get("second_pass", {}))),
            tracking=TrackingConfig(**dict(data.get("tracking", {}))),
            occlusion=OcclusionConfig(**dict(data.get("occlusion", {}))),
            sanity=SanityConfig(**dict(data.get("sanity", {}))),
            postprocess=PostprocessConfig(**dict(data.get("postprocess", {}))),
            mediapipe=MediaPipeConfig(**dict(data.get("mediapipe", {}))),
            debug=DebugConfig(**dict(data.get("debug", {}))),
            output=OutputConfig(**dict(data.get("output", {}))),
            runtime=RuntimeConfig(**dict(data.get("runtime", {}))),
            logging=LoggingConfig(**dict(data.get("logging", {}))),
        )

    def with_stage_dir(self, stage_dir: str) -> "ExtractorConfig":
        return ExtractorConfig(
            video=self.video,
            pose=self.pose,
            score=self.score,
            second_pass=self.second_pass,
            tracking=self.tracking,
            occlusion=self.occlusion,
            sanity=self.sanity,
            postprocess=self.postprocess,
            mediapipe=self.mediapipe,
            debug=self.debug,
            output=OutputConfig(stage_dir=stage_dir),
            runtime=self.runtime,
            logging=self.logging,
        )

    def with_video_count(self, video_count: int) -> "ExtractorConfig":
        return ExtractorConfig(
            video=self.video,
            pose=self.pose,
            score=self.score,
            second_pass=self.second_pass,
            tracking=self.tracking,
            occlusion=self.occlusion,
            sanity=self.sanity,
            postprocess=self.postprocess,
            mediapipe=self.mediapipe,
            debug=self.debug,
            output=self.output,
            runtime=RuntimeConfig(
                jobs=self.runtime.jobs,
                seed=self.runtime.seed,
                video_count=int(video_count),
            ),
            logging=self.logging,
        )

    def to_process_video_kwargs(self) -> Dict[str, Any]:
        return {
            "world_coords": self.pose.world_coords,
            "keep_pose_indices": list(self.pose.keep_pose_indices),
            "stride": max(1, int(self.video.stride)),
            "short_side": int(self.video.short_side),
            "min_det": float(self.score.min_det),
            "min_track": float(self.score.min_track),
            "pose_every": max(1, int(self.pose.pose_every)),
            "pose_complexity": int(self.pose.pose_complexity),
            "ts_source": str(self.video.ts_source),
            "min_hand_score": float(self.score.min_hand_score),
            "hand_score_lo": float(self.score.hand_score_lo),
            "hand_score_hi": float(self.score.hand_score_hi),
            "hand_score_source": str(self.score.hand_score_source),
            "tracker_init_score": float(self.score.tracker_init_score),
            "anchor_score": float(self.score.anchor_score),
            "pose_dist_qual_min": float(self.score.pose_dist_qual_min),
            "tracker_update_score": float(self.score.tracker_update_score),
            "pose_side_reassign_ratio": float(self.score.pose_side_reassign_ratio),
            "pose_ema_alpha": float(self.pose.pose_ema),
            "second_pass": bool(self.second_pass.enabled),
            "sp_trigger_below": float(self.second_pass.trigger_below),
            "sp_roi_frac": float(self.second_pass.roi_frac),
            "sp_margin": float(self.second_pass.margin),
            "sp_escalate_step": float(self.second_pass.escalate_step),
            "sp_escalate_max": float(self.second_pass.escalate_max),
            "sp_hands_up_only": bool(self.second_pass.hands_up_only),
            "sp_jitter_px": int(self.second_pass.jitter_px),
            "sp_jitter_rings": int(self.second_pass.jitter_rings),
            "sp_center_penalty": float(self.second_pass.center_penalty),
            "sp_label_relax": float(self.second_pass.label_relax),
            "sp_overlap_iou": float(self.second_pass.overlap_iou),
            "sp_overlap_shrink": float(self.second_pass.overlap_shrink),
            "sp_overlap_penalty_mult": float(self.second_pass.overlap_penalty_mult),
            "sp_overlap_require_label": bool(self.second_pass.overlap_require_label),
            "sp_debug_roi": bool(self.second_pass.debug_roi),
            "interp_hold": int(self.tracking.interp_hold),
            "track_max_gap": int(self.tracking.max_gap),
            "track_score_decay": float(self.tracking.score_decay),
            "track_reset_ms": int(self.tracking.reset_ms),
            "write_hand_mask": bool(self.tracking.write_hand_mask),
            "mp_backend": str(self.mediapipe.backend),
            "hand_task": str(self.mediapipe.hand_task),
            "pose_task": str(self.mediapipe.pose_task),
            "mp_tasks_delegate": str(self.mediapipe.tasks_delegate),
            "eval_mode": True,
            "seed": int(self.runtime.seed),
            "occ_hyst_frames": int(self.occlusion.hyst_frames),
            "occ_return_k": float(self.occlusion.return_k),
            "sanity_enable": bool(self.sanity.enabled),
            "sanity_scale_range": tuple(self.sanity.scale_range),
            "sanity_wrist_k": float(self.sanity.wrist_k),
            "sanity_bone_tol": float(self.sanity.bone_tol),
            "sanity_pass2": bool(self.sanity.pass2),
            "sanity_anchor_max_gap": int(self.sanity.anchor_max_gap),
            "sanitize_rejects": bool(self.sanity.sanitize_rejects),
            "postprocess": bool(self.postprocess.enabled),
            "pp_max_gap": int(self.postprocess.max_gap),
            "pp_smoother": str(self.postprocess.smoother),
            "pp_only_anchors": bool(self.postprocess.only_anchors),
        }
