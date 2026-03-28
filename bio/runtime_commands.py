from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import cv2

from bio.runtime import BioDecoderConfig, BioSegmenter, export_bio_runtime_bundle, save_bio_inference_result
from runtime.mediapipe_hands import (
    MediaPipeHandsConfig,
    MediaPipeHandsTracker,
    MediaPipeHolisticConfig,
    MediaPipeHolisticTracker,
    extract_video_sequence,
)
from runtime.skeleton import load_skeleton_sequence


def _device_arg(value: str) -> str:
    text = str(value or "").strip()
    return text or ("cuda" if __import__("torch").cuda.is_available() else "cpu")


def _tracker_cfg_from_args(args: argparse.Namespace) -> MediaPipeHandsConfig:
    return MediaPipeHandsConfig(
        static_image_mode=bool(getattr(args, "static_image_mode", False)),
        max_num_hands=int(getattr(args, "max_num_hands", 2)),
        model_complexity=int(getattr(args, "model_complexity", 1)),
        min_detection_confidence=float(getattr(args, "min_detection_confidence", 0.5)),
        min_tracking_confidence=float(getattr(args, "min_tracking_confidence", 0.5)),
    )


def _holistic_cfg_from_args(args: argparse.Namespace) -> MediaPipeHolisticConfig:
    return MediaPipeHolisticConfig(
        static_image_mode=bool(getattr(args, "static_image_mode", False)),
        model_complexity=int(getattr(args, "model_complexity", 1)),
        min_detection_confidence=float(getattr(args, "min_detection_confidence", 0.5)),
        min_tracking_confidence=float(getattr(args, "min_tracking_confidence", 0.5)),
    )


def _decoder_cfg_from_args(args: argparse.Namespace, *, threshold: float | None = None) -> BioDecoderConfig:
    payload: Dict[str, Any] = {}
    raw_json = str(getattr(args, "decoder_config_json", "") or "").strip()
    if raw_json:
        source = Path(raw_json).expanduser().resolve()
        loaded = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"BIO decoder config must be a JSON object: {source}")
        payload.update({k: v for k, v in loaded.items() if k in BioDecoderConfig.__dataclass_fields__})
    kwargs: Dict[str, Any] = {
        "continue_threshold": (None if getattr(args, "continue_threshold", None) is None else float(args.continue_threshold)),
        "continue_threshold_policy": str(getattr(args, "continue_threshold_policy", "fixed_ratio")),
        "continue_threshold_ratio": float(getattr(args, "continue_threshold_ratio", 0.60)),
        "min_segment_frames": int(getattr(args, "min_segment_frames", 3)),
        "min_gap_frames": int(getattr(args, "min_gap_frames", 2)),
        "max_idle_inside_segment": int(getattr(args, "max_idle_inside_segment", 4)),
        "cooldown_frames": int(getattr(args, "cooldown_frames", 2)),
        "emit_partial_segments": bool(getattr(args, "emit_partial_segments", False)),
        "eos_policy": str(getattr(args, "eos_policy", "") or "drop_partial_on_eos"),
        "stream_window": int(getattr(args, "stream_window", 16)),
        "require_hand_presence_to_start": bool(getattr(args, "require_hand_presence_to_start", False)),
        "min_visible_hand_frames_to_start": int(getattr(args, "min_visible_hand_frames_to_start", 2)),
        "min_valid_hand_joints_to_start": int(getattr(args, "min_valid_hand_joints_to_start", 8)),
        "allow_one_hand_to_start": bool(getattr(args, "allow_one_hand_to_start", True)),
        "use_signness_gate": bool(getattr(args, "use_signness_gate", True)),
        "signness_start_threshold": float(getattr(args, "signness_start_threshold", 0.55)),
        "signness_continue_threshold": float(getattr(args, "signness_continue_threshold", 0.50)),
        "use_onset_gate": bool(getattr(args, "use_onset_gate", True)),
        "onset_start_threshold": float(getattr(args, "onset_start_threshold", 0.45)),
        "active_start_threshold": float(getattr(args, "active_start_threshold", 0.25)),
        "active_continue_threshold": float(getattr(args, "active_continue_threshold", 0.20)),
    }
    payload.update(kwargs)
    if threshold is not None:
        payload["start_threshold"] = float(threshold)
    return BioDecoderConfig(**payload)


def _add_decoder_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--decoder_config_json", default="", help="Optional BIO decoder config JSON")
    parser.add_argument("--threshold", type=float, default=None, help="Optional explicit BIO start threshold override")
    parser.add_argument("--continue_threshold", type=float, default=None, help="Optional explicit BIO continue threshold override")
    parser.add_argument("--continue_threshold_policy", default="fixed_ratio", help="BIO continuation policy for raw checkpoint mode")
    parser.add_argument("--continue_threshold_ratio", type=float, default=0.60, help="BIO continue-threshold ratio when using fixed_ratio policy")
    parser.add_argument("--min_segment_frames", type=int, default=3)
    parser.add_argument("--min_gap_frames", type=int, default=2)
    parser.add_argument("--max_idle_inside_segment", type=int, default=4)
    parser.add_argument("--cooldown_frames", type=int, default=2)
    parser.add_argument("--emit_partial_segments", action="store_true")
    parser.add_argument("--eos_policy", default="", choices=["", "drop_partial_on_eos", "close_open_segment_on_eos"], help="Optional EOS policy override")
    parser.add_argument("--stream_window", type=int, default=16)
    parser.add_argument("--require_hand_presence_to_start", action="store_true", help="Block new segment starts until hand presence is stable")
    parser.add_argument("--min_visible_hand_frames_to_start", type=int, default=2, help="How many consecutive hand-visible frames are required before a start is allowed")
    parser.add_argument("--min_valid_hand_joints_to_start", type=int, default=8, help="Minimum valid joints per visible hand frame for the startup guard")
    parser.add_argument("--allow_one_hand_to_start", dest="allow_one_hand_to_start", action="store_true", default=True, help="Allow one visible hand to satisfy the startup guard")
    parser.add_argument("--require_both_hands_to_start", dest="allow_one_hand_to_start", action="store_false", help="Require both hands to satisfy the startup guard")
    parser.add_argument("--use_signness_gate", dest="use_signness_gate", action="store_true", default=True, help="Use auxiliary signness probability as an additional start/continue gate when available")
    parser.add_argument("--no_use_signness_gate", dest="use_signness_gate", action="store_false")
    parser.add_argument("--signness_start_threshold", type=float, default=0.55, help="Minimum p_active required to open a segment when signness head is available")
    parser.add_argument("--signness_continue_threshold", type=float, default=0.50, help="Minimum p_active used to keep a segment active when signness head is available")
    parser.add_argument("--use_onset_gate", dest="use_onset_gate", action="store_true", default=True, help="Allow onset probability to open a segment even when pB is still below the raw BIO threshold")
    parser.add_argument("--no_use_onset_gate", dest="use_onset_gate", action="store_false")
    parser.add_argument("--onset_start_threshold", type=float, default=0.45, help="Minimum p_onset required to open a segment when the onset head is available")
    parser.add_argument("--active_start_threshold", type=float, default=0.25, help="Minimum p_active required before a new segment start is allowed")
    parser.add_argument("--active_continue_threshold", type=float, default=0.20, help="Minimum p_active used to keep a segment active")
    parser.add_argument("--force_final_flush", action="store_true", help="Emit final partial segment on EOS even when decoder config disables it")


def _add_tracker_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--extractor_mode", default="hands_only", choices=["hands_only", "holistic_hands_pose"], help="MediaPipe extraction mode")
    parser.add_argument("--static_image_mode", action="store_true")
    parser.add_argument("--max_num_hands", type=int, default=2)
    parser.add_argument("--model_complexity", type=int, default=1)
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)


def _add_console_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--console_format", default="text", choices=["text", "json"], help="Console output format")


def _make_segmenter(args: argparse.Namespace) -> BioSegmenter:
    decoder_cfg = _decoder_cfg_from_args(args, threshold=(None if getattr(args, "threshold", None) is None else float(args.threshold)))
    if getattr(args, "bundle", ""):
        return BioSegmenter.from_bundle(args.bundle, device=_device_arg(args.device))
    checkpoint = getattr(args, "checkpoint", "") or getattr(args, "run_dir", "")
    if not checkpoint:
        raise ValueError("Provide either --bundle or --checkpoint/--run_dir")
    selection = str(getattr(args, "selection", "best_balanced") or "best_balanced")
    return BioSegmenter.from_checkpoint(
        checkpoint,
        selection=selection,
        device=_device_arg(args.device),
        decoder_cfg=decoder_cfg,
        threshold=(None if getattr(args, "threshold", None) is None else float(args.threshold)),
    )


def _safe_print(text: str) -> None:
    stream = sys.stdout
    try:
        stream.write(text + "\n")
    except UnicodeEncodeError:
        data = (text + "\n").encode(stream.encoding or "utf-8", errors="backslashreplace")
        buffer = getattr(stream, "buffer", None)
        if buffer is not None:
            buffer.write(data)
        else:
            stream.write(data.decode(stream.encoding or "utf-8", errors="ignore"))


def _summary_payload(payload: Dict[str, Any]) -> str:
    runtime = dict(payload.get("runtime_summary", {}) or {})
    extractor_mode = str(runtime.get("extractor_mode", "") or payload.get("sequence_meta", {}).get("extractor_mode", "") or "unknown")
    extractor = str(runtime.get("extractor", "") or payload.get("sequence_meta", {}).get("extractor", "") or "unknown")
    return (
        f"segments={len(payload.get('segments', []))} "
        f"threshold={float(payload.get('threshold', 0.0)):.2f} "
        f"extractor_mode={extractor_mode} extractor={extractor}"
    )


def _emit_payload(payload: Dict[str, Any], *, out_json: str = "", console_format: str = "text") -> None:
    if out_json:
        save_bio_inference_result(out_json, payload)
        _safe_print(f"Wrote {out_json}")
        return
    if str(console_format or "text") == "json":
        _safe_print(json.dumps(payload, ensure_ascii=True, indent=2))
        return
    _safe_print(_summary_payload(payload))


def _print_stream_segment(seg: Dict[str, Any], *, console_format: str = "text") -> None:
    if str(console_format or "text") == "json":
        _safe_print(json.dumps(seg, ensure_ascii=True))
        return
    _safe_print(
        "segment "
        f"id={int(seg.get('segment_id', 0))} "
        f"frames={int(seg.get('start_frame', 0))}:{int(seg.get('end_frame_exclusive', 0))} "
        f"reason={str(seg.get('end_reason', 'unknown'))} "
        f"score={float(seg.get('boundary_score', 0.0)):.3f}"
    )


def _runtime_summary(segmenter: BioSegmenter, *, extractor_mode: str, extractor: str) -> Dict[str, Any]:
    return {
        "config_source": str(segmenter.metadata.get("config_resolution_source", "checkpoint")),
        "selected_threshold": float(segmenter.threshold),
        "decoder_version": str(segmenter.metadata.get("decoder_version", "bio_segment_decoder_v2")),
        "decoder_config": asdict(segmenter.decoder.cfg),
        "preprocessing_version": str(segmenter.metadata.get("preprocessing_version", "")),
        "extractor_mode": str(extractor_mode),
        "extractor": str(extractor),
    }


def main_infer_skeletons(argv: List[str]) -> None:
    p = argparse.ArgumentParser("python -m bio infer-skeletons")
    p.add_argument("--input", required=True, help="Canonical skeleton sequence (.npz/.json/.jsonl)")
    p.add_argument("--bundle", default="", help="BIO runtime bundle dir")
    p.add_argument("--checkpoint", default="", help="BIO checkpoint file or run dir")
    p.add_argument("--run_dir", default="", help="Alias for checkpoint run dir")
    p.add_argument("--selection", default="best_recall_safe", choices=["best_balanced", "best_boundary", "best_recall_safe", "last"])
    p.add_argument("--device", default="")
    p.add_argument("--out_json", default="")
    p.add_argument("--include_frame_outputs", action="store_true")
    _add_console_args(p)
    _add_decoder_args(p)
    args = p.parse_args(argv)
    seq = load_skeleton_sequence(args.input)
    segmenter = _make_segmenter(args)
    result = segmenter.infer_sequence(
        seq,
        return_frame_outputs=bool(args.include_frame_outputs),
        flush_final=True,
        force_flush=bool(args.force_final_flush),
        eos_policy=(str(args.eos_policy or "close_open_segment_on_eos")),
    )
    result["runtime_summary"] = _runtime_summary(
        segmenter,
        extractor_mode=str(seq.meta.get("extractor_mode", "canonical_sequence")),
        extractor=str(seq.meta.get("extractor", "canonical_sequence")),
    )
    _emit_payload(result, out_json=args.out_json, console_format=str(args.console_format))


def main_infer_video(argv: List[str]) -> None:
    p = argparse.ArgumentParser("python -m bio infer-video")
    p.add_argument("--input", required=True, help="Input video path")
    p.add_argument("--bundle", default="", help="BIO runtime bundle dir")
    p.add_argument("--checkpoint", default="", help="BIO checkpoint file or run dir")
    p.add_argument("--run_dir", default="", help="Alias for checkpoint run dir")
    p.add_argument("--selection", default="best_recall_safe", choices=["best_balanced", "best_boundary", "best_recall_safe", "last"])
    p.add_argument("--device", default="")
    p.add_argument("--out_json", default="")
    p.add_argument("--include_frame_outputs", action="store_true")
    p.add_argument("--max_frames", type=int, default=0)
    _add_tracker_args(p)
    _add_console_args(p)
    _add_decoder_args(p)
    args = p.parse_args(argv)
    require_pose = str(args.extractor_mode) == "holistic_hands_pose"
    seq, _frames = extract_video_sequence(
        args.input,
        tracker_cfg=_tracker_cfg_from_args(args),
        holistic_cfg=_holistic_cfg_from_args(args),
        require_pose=require_pose,
        max_frames=int(args.max_frames),
        return_frames=False,
    )
    segmenter = _make_segmenter(args)
    result = segmenter.infer_sequence(
        seq,
        return_frame_outputs=bool(args.include_frame_outputs),
        flush_final=True,
        force_flush=bool(args.force_final_flush),
        eos_policy=(str(args.eos_policy or "close_open_segment_on_eos")),
    )
    result["video_path"] = str(Path(args.input).resolve())
    result["runtime_summary"] = _runtime_summary(
        segmenter,
        extractor_mode=str(seq.meta.get("extractor_mode", args.extractor_mode)),
        extractor=str(seq.meta.get("extractor", "unknown")),
    )
    _emit_payload(result, out_json=args.out_json, console_format=str(args.console_format))


def main_infer_stream(argv: List[str]) -> None:
    p = argparse.ArgumentParser("python -m bio infer-stream")
    p.add_argument("--source", default="0", help="Camera index or video path")
    p.add_argument("--bundle", default="", help="BIO runtime bundle dir")
    p.add_argument("--checkpoint", default="", help="BIO checkpoint file or run dir")
    p.add_argument("--run_dir", default="", help="Alias for checkpoint run dir")
    p.add_argument("--selection", default="best_recall_safe", choices=["best_balanced", "best_boundary", "best_recall_safe", "last"])
    p.add_argument("--device", default="")
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--display", action="store_true")
    _add_tracker_args(p)
    _add_console_args(p)
    _add_decoder_args(p)
    args = p.parse_args(argv)

    source_text = str(args.source)
    source: int | str
    source = int(source_text) if source_text.isdigit() else source_text
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open stream source: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
    dt_ms = 1000.0 / fps
    segmenter = _make_segmenter(args)
    tracker: MediaPipeHandsTracker | MediaPipeHolisticTracker
    extractor_mode = str(args.extractor_mode)
    if extractor_mode == "holistic_hands_pose":
        tracker = MediaPipeHolisticTracker(cfg=_holistic_cfg_from_args(args))
        extractor = "mediapipe.solutions.holistic"
    else:
        tracker = MediaPipeHandsTracker(cfg=_tracker_cfg_from_args(args))
        extractor = "mediapipe.solutions.hands"
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts_ms = float(frame_idx) * dt_ms
            item = tracker.process_bgr(frame, ts_ms=ts_ms)
            out = segmenter.step(item["pts"], item["mask"], ts_ms=ts_ms)
            for seg in out["segments"]:
                _print_stream_segment(seg, console_format=str(args.console_format))
            if args.display:
                cv2.putText(frame, f"frame={frame_idx} label={out['label']}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("bio infer-stream", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            frame_idx += 1
            if int(args.max_frames) > 0 and frame_idx >= int(args.max_frames):
                break
        for seg in segmenter.decoder.flush(
            force=bool(args.force_final_flush),
            eos_policy=str(args.eos_policy or "drop_partial_on_eos"),
        ):
            _print_stream_segment(seg.to_dict(), console_format=str(args.console_format))
        _safe_print(
            f"stream_done threshold={segmenter.threshold:.2f} extractor_mode={extractor_mode} extractor={extractor}"
        )
    finally:
        tracker.close()
        cap.release()
        if args.display:
            cv2.destroyAllWindows()


def main_export_runtime_bundle(argv: List[str]) -> None:
    p = argparse.ArgumentParser("python -m bio export-runtime-bundle")
    p.add_argument("--checkpoint", required=True, help="BIO checkpoint file or run dir")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--selection", default="best_recall_safe", choices=["best_balanced", "best_boundary", "best_recall_safe", "last"])
    _add_decoder_args(p)
    args = p.parse_args(argv)
    manifest = export_bio_runtime_bundle(
        args.checkpoint,
        args.out_dir,
        selection=args.selection,
        decoder_cfg=_decoder_cfg_from_args(args),
    )
    print(f"Wrote {manifest}")
