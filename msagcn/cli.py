from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

from msagcn.runtime import MSAGCNClassifier, export_msagcn_runtime_bundle
from runtime.skeleton import load_skeleton_sequence


def _print_help() -> None:
    print("MSAGCN CLI")
    print("")
    print("Usage:")
    print("  python -m msagcn <command> [args]")
    print("")
    print("Commands:")
    print("  train               Train Multi-Stream AGCN")
    print("  infer-clip          Classify a canonical skeleton clip")
    print("  infer-segment       Alias of infer-clip")
    print("  infer-skeletons     Alias of infer-clip")
    print("  export-runtime-bundle  Export a deployable MSAGCN runtime bundle")
    print("")
    print("Run 'python -m msagcn <command> -h' for command-specific help.")


def main(argv: List[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        _print_help()
        return

    cmd = argv[0]
    rest = argv[1:]
    dispatch: Dict[str, Callable[[List[str]], None]] = {
        "train": _run_train,
        "infer-clip": _run_infer_clip,
        "infer-segment": _run_infer_clip,
        "infer-skeletons": _run_infer_clip,
        "export-runtime-bundle": _run_export_runtime_bundle,
    }
    handler = dispatch.get(cmd)
    if handler is None:
        print(f"Unknown command: {cmd}")
        _print_help()
        sys.exit(2)
    handler(rest)


def _run_train(args: List[str]) -> None:
    from msagcn import train

    old_argv = sys.argv
    try:
        sys.argv = ["python -m msagcn train", *args]
        train.main()
    finally:
        sys.argv = old_argv


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


def _emit_prediction(payload: Dict[str, object], *, out_json: str = "", console_format: str = "text") -> None:
    if out_json:
        out = Path(out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _safe_print(f"Wrote {out}")
        return
    if str(console_format or "text") == "json":
        _safe_print(json.dumps(payload, ensure_ascii=True, indent=2))
        return
    family = ""
    if payload.get("family_label") is not None:
        family = f" family={payload.get('family_label')} ({float(payload.get('family_confidence', 0.0)):.3f})"
    _safe_print(
        f"label={payload.get('label')} conf={float(payload.get('confidence', 0.0)):.3f}"
        f"{family} input={payload.get('input')}"
    )


def _predict_payload(args: argparse.Namespace) -> Dict[str, object]:
    seq = load_skeleton_sequence(args.input)
    if args.bundle:
        clf = MSAGCNClassifier.from_bundle(args.bundle, device=args.device or None)
    else:
        clf = MSAGCNClassifier.from_checkpoint(
            args.checkpoint,
            device=args.device or None,
            label_map_path=(args.label_map or None),
            ds_config_path=(args.ds_config or None),
        )
    pred = clf.predict_sequence(seq, topk=int(args.topk))
    payload = pred.to_dict()
    payload["input"] = str(Path(args.input).resolve())
    return payload


def _run_infer_clip(argv: List[str]) -> None:
    p = argparse.ArgumentParser("python -m msagcn infer-clip")
    p.add_argument("--input", required=True, help="Canonical skeleton sequence (.npz/.json/.jsonl)")
    p.add_argument("--checkpoint", default="", help="MSAGCN checkpoint (.ckpt)")
    p.add_argument("--bundle", default="", help="MSAGCN runtime bundle dir")
    p.add_argument("--label_map", default="", help="Optional explicit label2idx.json for raw checkpoint mode")
    p.add_argument("--ds_config", default="", help="Optional explicit ds_config.json for raw checkpoint mode")
    p.add_argument("--device", default="")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--out_json", default="")
    p.add_argument("--console_format", default="text", choices=["text", "json"])
    args = p.parse_args(argv)
    if not args.checkpoint and not args.bundle:
        raise ValueError("Provide either --checkpoint or --bundle")
    payload = _predict_payload(args)
    _emit_prediction(payload, out_json=args.out_json, console_format=str(args.console_format))


def _run_export_runtime_bundle(argv: List[str]) -> None:
    p = argparse.ArgumentParser("python -m msagcn export-runtime-bundle")
    p.add_argument("--checkpoint", required=True, help="MSAGCN checkpoint (.ckpt)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label_map", default="", help="Optional explicit label2idx.json")
    p.add_argument("--ds_config", default="", help="Optional explicit ds_config.json")
    args = p.parse_args(argv)
    manifest = export_msagcn_runtime_bundle(
        args.checkpoint,
        args.out_dir,
        label_map_path=(args.label_map or None),
        ds_config_path=(args.ds_config or None),
    )
    _safe_print(f"Wrote {manifest}")
