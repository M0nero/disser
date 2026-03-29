from __future__ import annotations

from time import perf_counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...algos.tracking import HandTracker
from ...config import ExtractorConfig
from ...tasks import TaskSpec
from ... import _env  # ensure caps before cv2 import
from ...core.logging_utils import get_logger, log_metrics
from ...mp.mp_utils import normalize_backend_name, normalize_tasks_delegate, try_import_mediapipe
from ..adapters import DetectorFactoryProtocol, MediaPipeDetectorFactory, _resolve_model_paths
from ..contracts import SamplePayload
from ..records.builder import build_frame_record, build_sample_payload
from ..records.legacy import legacy_frame_from_record
from ..reporting import ReportingContext, emit_ndjson_line, finalize_records
from ..state import SampleRuntime
from .decode import iter_decoded_frames, open_video_capture
from .detect import PoseRuntimeState
from .frame_step import FrameStepContext, process_frame_step

LOGGER = get_logger(__name__)


def process_video(
    path: Path,
    sample_id: str,
    world_coords: bool,
    keep_pose_indices: Optional[List[int]],
    stride: int,
    short_side: Optional[int],
    min_det: float,
    min_track: float,
    pose_every: int,
    pose_complexity: int,
    ts_source: str,
    min_hand_score: float = 0.0,
    hand_score_lo: float = 0.55,
    hand_score_hi: float = 0.90,
    hand_score_source: str = "handedness",
    tracker_init_score: float = -1.0,
    anchor_score: float = -1.0,
    pose_dist_qual_min: float = 0.50,
    tracker_update_score: float = -1.0,
    pose_side_reassign_ratio: float = 0.85,
    pose_ema_alpha: float = 0.0,
    ndjson_path: Optional[Path] = None,
    second_pass: bool = False,
    sp_trigger_below: float = 0.50,
    sp_roi_frac: float = 0.25,
    sp_margin: float = 0.35,
    sp_escalate_step: float = 0.25,
    sp_escalate_max: float = 2.0,
    sp_hands_up_only: bool = False,
    sp_jitter_px: int = 0,
    sp_jitter_rings: int = 1,
    sp_center_penalty: float = 0.30,
    sp_label_relax: float = 0.15,
    sp_overlap_iou: float = 0.15,
    sp_overlap_shrink: float = 0.70,
    sp_overlap_penalty_mult: float = 2.0,
    sp_overlap_require_label: bool = False,
    sp_debug_roi: bool = False,
    interp_hold: int = 0,
    track_max_gap: int = 15,
    track_score_decay: float = 0.90,
    track_reset_ms: int = 250,
    write_hand_mask: bool = False,
    mp_backend: str = "solutions",
    hand_task: Optional[str] = None,
    pose_task: Optional[str] = None,
    mp_tasks_delegate: Optional[str] = None,
    debug_video_path: Optional[Path] = None,
    eval_mode: bool = False,
    seed: int = 0,
    occ_hyst_frames: int = 15,
    occ_return_k: float = 1.2,
    sanity_enable: bool = True,
    sanity_scale_range: Tuple[float, float] = (0.70, 1.35),
    sanity_wrist_k: float = 2.0,
    sanity_bone_tol: float = 0.30,
    sanity_pass2: bool = False,
    sanity_anchor_max_gap: int = 0,
    sanitize_rejects: bool = True,
    postprocess: bool = False,
    pp_max_gap: int = 15,
    pp_smoother: str = "ema",
    pp_only_anchors: bool = True,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    segment_meta: Optional[Dict[str, Any]] = None,
    detector_factory: Optional[DetectorFactoryProtocol] = None,
) -> SamplePayload:
    detector_factory = detector_factory or MediaPipeDetectorFactory()
    uses_mediapipe = isinstance(detector_factory, MediaPipeDetectorFactory)
    mp = mp_solutions = None
    if uses_mediapipe:
        mp, mp_solutions = try_import_mediapipe()
    backend = normalize_backend_name(mp_backend)
    delegate_raw = mp_tasks_delegate or "auto"
    tasks_delegate = normalize_tasks_delegate(delegate_raw)
    hand_lo = min(1.0, max(0.0, max(min_hand_score, hand_score_lo)))
    hand_hi = min(1.0, max(hand_lo, hand_score_hi))
    score_source = (hand_score_source or "handedness").strip().lower()
    if score_source not in {"handedness", "presence"}:
        score_source = "handedness"
    tracker_init_score_eff = min(1.0, max(hand_lo, hand_hi if tracker_init_score < 0 else tracker_init_score))
    anchor_score_eff = min(1.0, max(0.0, hand_hi if anchor_score < 0 else anchor_score))
    pose_dist_qual_min_eff = min(1.0, max(0.0, pose_dist_qual_min))
    tracker_update_score_eff = None
    if tracker_update_score >= 0:
        tracker_update_score_eff = min(1.0, max(0.0, tracker_update_score))
    pose_side_reassign_ratio_eff = min(1.0, max(0.5, pose_side_reassign_ratio if pose_side_reassign_ratio > 0 else 0.85))

    log_metrics(LOGGER, "process_video.start", {
        "video": path.name,
        "sample_id": str(sample_id),
        "world_coords": bool(world_coords),
        "stride": stride,
        "short_side": short_side,
        "pose_every": pose_every,
        "pose_complexity": pose_complexity,
        "second_pass": bool(second_pass),
        "sp_debug_roi": bool(sp_debug_roi),
        "mp_backend": backend,
        "mp_tasks_delegate": tasks_delegate,
        "ndjson": str(ndjson_path) if ndjson_path else None,
        "debug_video": str(debug_video_path) if debug_video_path else None,
        "eval_mode": bool(eval_mode),
        "seed": int(seed),
        "frame_start": frame_start,
        "frame_end": frame_end,
    })

    orig_frame_start = frame_start
    orig_frame_end = frame_end
    frame_start_i = int(frame_start) if frame_start is not None else 0
    frame_end_i = int(frame_end) if frame_end is not None else None
    if frame_start_i < 0:
        frame_start_i = 0
    if frame_end_i is not None and frame_end_i <= frame_start_i:
        raise RuntimeError(f"Invalid frame range: start={frame_start_i}, end={frame_end_i}")

    hand_model_path, pose_model_path = _resolve_model_paths(backend, hand_task, pose_task)
    cap, width_src, height_src, fps = open_video_capture(path, frame_start=frame_start_i)

    hands_detector, pose_detector, hands_sp = detector_factory.create(
        backend=backend,
        mp=mp,
        mp_solutions=mp_solutions,
        hand_model_path=hand_model_path,
        pose_model_path=pose_model_path,
        min_det=min_det,
        min_track=min_track,
        pose_complexity=pose_complexity,
        tasks_delegate=tasks_delegate,
        second_pass=second_pass,
        world_coords=world_coords,
    )

    frame_records = []
    proc_w = width_src
    proc_h = height_src
    ndjson_f = None
    if ndjson_path is not None:
        ndjson_path.parent.mkdir(parents=True, exist_ok=True)
        ndjson_f = open(ndjson_path, 'wt', encoding='utf-8')
        log_metrics(LOGGER, 'process_video.ndjson', {'video': path.name, 'ndjson_path': str(ndjson_path)})

    processing_started = perf_counter()
    pose_runtime = 0.0
    hand_runtime = 0.0
    second_pass_runtime = 0.0
    progress_interval = 500
    sample_state = SampleRuntime(sample_id=str(sample_id))
    pose_state = PoseRuntimeState()
    hands_frames_running = 0
    sp_rec_left = 0
    sp_rec_right = 0
    sp_missing_left_pre = 0
    sp_missing_right_pre = 0
    tracker_left = HandTracker()
    tracker_right = HandTracker()

    step_context = FrameStepContext(
        hands_detector=hands_detector,
        pose_detector=pose_detector,
        hands_sp=hands_sp,
        pose_state=pose_state,
        sample_state=sample_state,
        tracker_left=tracker_left,
        tracker_right=tracker_right,
        world_coords=world_coords,
        keep_pose_indices=keep_pose_indices,
        pose_every=pose_every,
        pose_ema_alpha=pose_ema_alpha,
        min_hand_score=min_hand_score,
        hand_lo=hand_lo,
        hand_hi=hand_hi,
        score_source=score_source,
        anchor_score_eff=anchor_score_eff,
        tracker_init_score_eff=tracker_init_score_eff,
        tracker_update_score_eff=tracker_update_score_eff,
        pose_dist_qual_min_eff=pose_dist_qual_min_eff,
        pose_side_reassign_ratio_eff=pose_side_reassign_ratio_eff,
        second_pass=second_pass,
        sp_trigger_below=sp_trigger_below,
        sp_roi_frac=sp_roi_frac,
        sp_margin=sp_margin,
        sp_escalate_step=sp_escalate_step,
        sp_escalate_max=sp_escalate_max,
        sp_hands_up_only=sp_hands_up_only,
        sp_jitter_px=sp_jitter_px,
        sp_jitter_rings=sp_jitter_rings,
        sp_center_penalty=sp_center_penalty,
        sp_label_relax=sp_label_relax,
        sp_overlap_iou=sp_overlap_iou,
        sp_overlap_shrink=sp_overlap_shrink,
        sp_overlap_penalty_mult=sp_overlap_penalty_mult,
        sp_overlap_require_label=sp_overlap_require_label,
        sp_debug_roi=sp_debug_roi,
        occ_hyst_frames=occ_hyst_frames,
        occ_return_k=occ_return_k,
        sanity_enable=sanity_enable,
        sanity_scale_range=sanity_scale_range,
        sanity_wrist_k=sanity_wrist_k,
        sanity_bone_tol=sanity_bone_tol,
        sanity_pass2=sanity_pass2,
        sanity_anchor_max_gap=sanity_anchor_max_gap,
        sanitize_rejects=sanitize_rejects,
        track_max_gap=track_max_gap,
        track_score_decay=track_score_decay,
        track_reset_ms=track_reset_ms,
        write_hand_mask=write_hand_mask,
    )

    try:
        if hands_detector is None or pose_detector is None:
            raise RuntimeError('Mediapipe detectors failed to initialize')
        for decoded in iter_decoded_frames(
            cap,
            frame_start=frame_start_i,
            frame_end=frame_end_i,
            stride=max(1, stride),
            short_side=short_side,
            fps=fps,
            ts_source=ts_source,
        ):
            proc_w = decoded.proc_w
            proc_h = decoded.proc_h
            step = process_frame_step(decoded, context=step_context)
            hand_runtime += step.hand_runtime
            pose_runtime += step.pose_runtime
            second_pass_runtime += step.second_pass_runtime
            sp_rec_left += int(step.left.sp_recovered)
            sp_rec_right += int(step.right.sp_recovered)
            sp_missing_left_pre += int(step.sp_missing_pre_left)
            sp_missing_right_pre += int(step.sp_missing_pre_right)
            record = build_frame_record(step)
            frame_records.append(record)
            if record.hand_1.landmarks is not None or record.hand_2.landmarks is not None:
                hands_frames_running += 1
            if progress_interval and (len(frame_records) % progress_interval) == 0:
                log_metrics(LOGGER, 'process_video.progress', {
                    'video': path.name,
                    'frames_processed': len(frame_records),
                    'elapsed_sec': round(perf_counter() - processing_started, 3),
                    'hands_detected_frames': hands_frames_running,
                })
            emit_ndjson_line(ndjson_f, {'video': path.name, **legacy_frame_from_record(record)})
    finally:
        if hands_sp is not None:
            hands_sp.close()
        if pose_detector is not None:
            pose_detector.close()
        if hands_detector is not None:
            hands_detector.close()
        cap.release()
        if ndjson_f is not None:
            ndjson_f.close()

    meta_header = {
        'video': path.name,
        'fps': fps,
        'size_src': [int(width_src), int(height_src)],
        'size_proc': [int(proc_w), int(proc_h)],
        'version': 5,
        'coords': 'world' if world_coords else 'image',
        'mp_backend': backend,
        'mp_models': {'hand': str(hand_model_path) if hand_model_path is not None else None, 'pose': str(pose_model_path) if pose_model_path is not None else None} if backend == 'tasks' else None,
        'mp_tasks_delegate': delegate_raw if backend == 'tasks' else None,
        'pose_indices': keep_pose_indices if keep_pose_indices is not None else 'all',
        'hand_mapping': {'hand 1': 'left', 'hand 2': 'right'},
        'second_pass': bool(second_pass),
        'second_pass_params': {
            'trigger_below': float(sp_trigger_below),
            'roi_frac': float(sp_roi_frac),
            'margin': float(sp_margin),
            'escalate_step': float(sp_escalate_step),
            'escalate_max': float(sp_escalate_max),
            'hands_up_only': bool(sp_hands_up_only),
        } if second_pass else None,
        'sp_debug_roi': bool(sp_debug_roi),
        'interp_hold': int(interp_hold),
        'hand_score_gate': {
            'lo': float(hand_lo),
            'hi': float(hand_hi),
            'min_hand_score_legacy': float(min_hand_score),
            'score_source': score_source,
            'tracker_init_score': float(tracker_init_score_eff),
            'anchor_score': float(anchor_score_eff),
            'pose_dist_qual_min': float(pose_dist_qual_min_eff),
            'pose_side_reassign_ratio': float(pose_side_reassign_ratio_eff),
            'sanitize_rejects': bool(sanitize_rejects),
            'tracker_update_score': float(tracker_update_score_eff) if tracker_update_score_eff is not None else None,
        },
        'tracking': {
            'enabled': bool((not world_coords) and track_max_gap > 0),
            'track_max_gap': int(track_max_gap),
            'track_score_decay': float(track_score_decay),
            'track_reset_ms': int(track_reset_ms),
        },
    }
    if orig_frame_start is not None or orig_frame_end is not None:
        meta_header['frame_range'] = {'start': int(frame_start_i), 'end': int(frame_end_i) if frame_end_i is not None else None}
    if segment_meta:
        meta_header['segment'] = segment_meta

    reporting = finalize_records(
        frame_records=frame_records,
        context=ReportingContext(
            sample_id=str(sample_id),
            video_name=path.name,
            source_video=str(path).replace('\\', '/'),
            fps=float(fps),
            backend=backend,
            tasks_delegate=tasks_delegate,
            processing_elapsed=(perf_counter() - processing_started),
            hand_runtime=hand_runtime,
            pose_runtime=pose_runtime,
            second_pass_runtime=second_pass_runtime,
            second_pass_enabled=bool(second_pass),
            hands_present=hands_frames_running,
            sp_rec_left=sp_rec_left,
            sp_rec_right=sp_rec_right,
            sp_missing_left_pre=sp_missing_left_pre,
            sp_missing_right_pre=sp_missing_right_pre,
            ndjson_path=ndjson_path,
            eval_mode=bool(eval_mode),
            postprocess=bool(postprocess),
            pp_max_gap=pp_max_gap,
            pp_smoother=pp_smoother,
            pp_only_anchors=bool(pp_only_anchors),
            hand_hi=hand_hi,
            world_coords=bool(world_coords),
        ),
    )
    log_metrics(LOGGER, 'process_video.summary', reporting.summary_metrics)
    return build_sample_payload(
        sample_id=str(sample_id),
        slug=str(sample_id),
        source_video=str(path).replace('\\', '/'),
        meta_header=meta_header,
        frame_records=frame_records,
        frame_records_pp=reporting.frame_records_pp,
        manifest_dict=reporting.manifest_dict,
        segment_meta=segment_meta,
        fps=float(fps),
    )


def process_task(
    task: TaskSpec | Dict[str, Any],
    *,
    detector_factory: Optional[DetectorFactoryProtocol] = None,
) -> SamplePayload:
    task_spec = task if isinstance(task, TaskSpec) else TaskSpec.from_payload(task)
    config = ExtractorConfig.from_dict(task_spec.config_dict)
    kwargs = config.to_process_video_kwargs()
    ndjson_path = Path(task_spec.ndjson_path) if task_spec.ndjson_path else None
    debug_video_path = Path(task_spec.debug_video_path) if task_spec.debug_video_path else None
    return process_video(
        path=task_spec.source_path,
        sample_id=str(task_spec.sample_id),
        ndjson_path=ndjson_path,
        debug_video_path=debug_video_path,
        frame_start=task_spec.frame_start,
        frame_end=task_spec.frame_end,
        segment_meta=(task_spec.segment_meta or None),
        detector_factory=detector_factory,
        **kwargs,
    )
