from __future__ import annotations

import argparse
import json
import sys
from typing import Callable, Dict, List

from pipeline.app import InferencePipelineConfig, build_review_session, run_camera_pipeline, run_mediapipe_debug, run_video_pipeline
from runtime.mediapipe_hands import MediaPipeHandsConfig


def _print_help() -> None:
    print("Inference pipeline CLI")
    print("")
    print("Usage:")
    print("  python -m pipeline <command> [args]")
    print("")
    print("Commands:")
    print("  infer-video   Offline reference pipeline: video -> MediaPipe -> BIO -> MSAGCN -> sentence")
    print("  infer-camera  Live reference pipeline from camera/video source")
    print("  build-review-session  Build a desktop-review session bundle from a video")
    print("  debug-mediapipe  Extract and visualize canonical MediaPipe runtime skeletons")
    print("")
    print("Run 'python -m pipeline <command> -h' for command-specific help.")


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


def _emit_result(payload: Dict[str, object], *, console_format: str = "text") -> None:
    if str(console_format or "text") == "json":
        _safe_print(json.dumps(payload, ensure_ascii=True, indent=2))
        return
    _safe_print(
        f"sentence={payload.get('sentence', '')!r} "
        f"segments={int(payload.get('segments', 0))} "
        f"predictions={int(payload.get('predictions', 0))}"
    )


def main(argv: List[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        _print_help()
        return
    cmd = argv[0]
    rest = argv[1:]
    dispatch: Dict[str, Callable[[List[str]], None]] = {
        "infer-video": _run_infer_video,
        "infer-camera": _run_infer_camera,
        "build-review-session": _run_build_review_session,
        "debug-mediapipe": _run_debug_mediapipe,
    }
    handler = dispatch.get(cmd)
    if handler is None:
        print(f"Unknown command: {cmd}")
        _print_help()
        sys.exit(2)
    handler(rest)


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bio_bundle", default="", help="BIO runtime bundle directory")
    p.add_argument("--bio_checkpoint", default="", help="BIO checkpoint file or run dir")
    p.add_argument("--bio_selection", default="best_balanced", choices=["best_balanced", "best_boundary", "last"])
    p.add_argument("--bio_threshold", type=float, default=None)
    p.add_argument("--bio_decoder_config_json", default="", help="Optional BIO decoder config JSON for raw checkpoint mode")
    p.add_argument("--msagcn_bundle", default="", help="MSAGCN runtime bundle directory")
    p.add_argument("--msagcn_checkpoint", default="", help="MSAGCN checkpoint (.ckpt)")
    p.add_argument("--msagcn_label_map", default="", help="Optional explicit label2idx.json for raw checkpoint mode")
    p.add_argument("--msagcn_ds_config", default="", help="Optional explicit ds_config.json for raw checkpoint mode")
    p.add_argument("--device", default="")
    p.add_argument("--pre_context_frames", type=int, default=4)
    p.add_argument("--post_context_frames", type=int, default=4)
    p.add_argument("--max_buffer_frames", type=int, default=4096)
    p.add_argument("--extractor_mode", default="auto", choices=["auto", "hands_only", "holistic_hands_pose"])
    p.add_argument("--max_pending_segments", type=int, default=8)
    p.add_argument("--classifier_worker_count", type=int, default=1)
    p.add_argument("--drop_or_block_policy", default="block_on_classifier_queue", choices=["block_on_classifier_queue", "drop_segment"])
    p.add_argument("--sentence_min_confidence", type=float, default=0.5)
    p.add_argument("--sentence_duplicate_gap_ms", type=float, default=1200.0)
    p.add_argument("--max_num_hands", type=int, default=2)
    p.add_argument("--model_complexity", type=int, default=1)
    p.add_argument("--min_detection_confidence", type=float, default=0.5)
    p.add_argument("--min_tracking_confidence", type=float, default=0.5)
    p.add_argument("--console_format", default="text", choices=["text", "json"])


def _cfg_from_args(args: argparse.Namespace) -> InferencePipelineConfig:
    if not (args.bio_bundle or args.bio_checkpoint):
        raise ValueError("Provide either --bio_bundle or --bio_checkpoint")
    if not (args.msagcn_bundle or args.msagcn_checkpoint):
        raise ValueError("Provide either --msagcn_bundle or --msagcn_checkpoint")
    return InferencePipelineConfig(
        bio_bundle=str(args.bio_bundle or ""),
        bio_checkpoint=str(args.bio_checkpoint or ""),
        bio_selection=str(args.bio_selection or "best_balanced"),
        bio_decoder_config_json=str(args.bio_decoder_config_json or ""),
        bio_threshold=(None if args.bio_threshold is None else float(args.bio_threshold)),
        msagcn_bundle=str(args.msagcn_bundle or ""),
        msagcn_checkpoint=str(args.msagcn_checkpoint or ""),
        msagcn_label_map=str(args.msagcn_label_map or ""),
        msagcn_ds_config=str(args.msagcn_ds_config or ""),
        device=str(args.device or ""),
        pre_context_frames=int(args.pre_context_frames),
        post_context_frames=int(args.post_context_frames),
        max_buffer_frames=int(args.max_buffer_frames),
        extractor_mode=str(args.extractor_mode or "auto"),
        max_pending_segments=int(args.max_pending_segments),
        classifier_worker_count=int(args.classifier_worker_count),
        drop_or_block_policy=str(args.drop_or_block_policy or "block_on_classifier_queue"),
        sentence_min_confidence=float(args.sentence_min_confidence),
        sentence_duplicate_gap_ms=float(args.sentence_duplicate_gap_ms),
    )


def _tracker_cfg_from_args(args: argparse.Namespace) -> MediaPipeHandsConfig:
    return MediaPipeHandsConfig(
        max_num_hands=int(args.max_num_hands),
        model_complexity=int(args.model_complexity),
        min_detection_confidence=float(args.min_detection_confidence),
        min_tracking_confidence=float(args.min_tracking_confidence),
    )


def _run_infer_video(argv: List[str]) -> None:
    p = argparse.ArgumentParser("python -m pipeline infer-video")
    p.add_argument("--input", required=True, help="Input video path")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--write_preview", action="store_true")
    _add_common_args(p)
    args = p.parse_args(argv)
    result = run_video_pipeline(
        args.input,
        cfg=_cfg_from_args(args),
        out_dir=args.out_dir,
        tracker_cfg=_tracker_cfg_from_args(args),
        max_frames=int(args.max_frames),
        write_preview=bool(args.write_preview),
    )
    _emit_result(
        {"sentence": result["sentence_builder"]["sentence"], "segments": len(result["segments"]), "predictions": len(result["predictions"])},
        console_format=str(args.console_format),
    )


def _run_infer_camera(argv: List[str]) -> None:
    p = argparse.ArgumentParser("python -m pipeline infer-camera")
    p.add_argument("--source", default="0", help="Camera index or video path")
    p.add_argument("--out_dir", default="")
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--display", action="store_true")
    _add_common_args(p)
    args = p.parse_args(argv)
    source_text = str(args.source)
    source: int | str = int(source_text) if source_text.isdigit() else source_text
    result = run_camera_pipeline(
        source,
        cfg=_cfg_from_args(args),
        tracker_cfg=_tracker_cfg_from_args(args),
        out_dir=(args.out_dir or None),
        max_frames=int(args.max_frames),
        display=bool(args.display),
    )
    _emit_result(
        {"sentence": result["sentence_builder"]["sentence"], "segments": len(result["segments"]), "predictions": len(result["predictions"])},
        console_format=str(args.console_format),
    )


def _run_build_review_session(argv: List[str]) -> None:
    p = argparse.ArgumentParser("python -m pipeline build-review-session")
    p.add_argument("--input", required=True, help="Input video path")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_frames", type=int, default=0)
    _add_common_args(p)
    args = p.parse_args(argv)
    result = build_review_session(
        args.input,
        cfg=_cfg_from_args(args),
        out_dir=args.out_dir,
        tracker_cfg=_tracker_cfg_from_args(args),
        max_frames=int(args.max_frames),
    )
    payload = {
        "sentence": result.get("sentence", ""),
        "segments": int(result.get("segments", 0)),
        "predictions": int(result.get("predictions", 0)),
        "session_path": str(result.get("session_path", "")),
        "warnings": int(len(list(result.get("warnings", []) or []))),
    }
    if str(args.console_format or "text") == "json":
        _safe_print(json.dumps(payload, ensure_ascii=True, indent=2))
        return
    _safe_print(
        f"sentence={payload['sentence']!r} "
        f"segments={payload['segments']} "
        f"predictions={payload['predictions']} "
        f"warnings={payload['warnings']} "
        f"session={payload['session_path']}"
    )


def _run_debug_mediapipe(argv: List[str]) -> None:
    p = argparse.ArgumentParser("python -m pipeline debug-mediapipe")
    p.add_argument("--input", required=True, help="Input video path")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--extractor_mode", default="hands_only", choices=["hands_only", "holistic_hands_pose"])
    p.add_argument("--skip_overlay", action="store_true")
    p.add_argument("--max_num_hands", type=int, default=2)
    p.add_argument("--model_complexity", type=int, default=1)
    p.add_argument("--min_detection_confidence", type=float, default=0.5)
    p.add_argument("--min_tracking_confidence", type=float, default=0.5)
    p.add_argument("--console_format", default="text", choices=["text", "json"])
    args = p.parse_args(argv)
    result = run_mediapipe_debug(
        args.input,
        out_dir=args.out_dir,
        tracker_cfg=_tracker_cfg_from_args(args),
        max_frames=int(args.max_frames),
        extractor_mode=str(args.extractor_mode),
        write_overlay=not bool(args.skip_overlay),
    )
    if str(args.console_format) == "json":
        _safe_print(json.dumps(result, ensure_ascii=True, indent=2))
        return
    summary = dict(result.get("summary", {}) or {})
    _safe_print(
        f"frames={int(summary.get('frames', 0))} "
        f"fps={float(summary.get('fps', 0.0)):.2f} "
        f"left_frames={int(summary.get('left_hand_present_frames', 0))} "
        f"right_frames={int(summary.get('right_hand_present_frames', 0))} "
        f"pose_frames={int(summary.get('pose_present_frames', 0))} "
        f"out={result.get('output_dir', '')}"
    )
