from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .checkpoint import load_state, infer_num_classes
from .convert import to_coreml_from_torchscript, to_coreml_via_onnx
from .labels import load_labels
from .logging_utils import log
from .model import build_model, make_example
from .torchscript import try_torchscript


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export ST-GCN PyTorch checkpoint to CoreML .mlmodel")
    p.add_argument("--ckpt", type=str, default="best_model.pth", help="Path to PyTorch checkpoint (.pth)")
    p.add_argument("--out", type=str, default="STGCN.mlmodel", help="Output .mlmodel path")
    p.add_argument("--labels", type=str, default=None, help="Optional labels file (.txt or .json)")
    p.add_argument("--vertices", type=int, default=42, help="Number of graph vertices V (default: 42 for two hands)")
    p.add_argument("--channels", type=int, default=None, help="Override channels C (default: infer from model.add_vel)")
    p.add_argument("--fixed-t", type=int, default=None, help="Fix time length T (use this OR --min-t/--max-t)")
    p.add_argument("--min-t", type=int, default=32, help="Min time for flexible T (iOS 16+)")
    p.add_argument("--max-t", type=int, default=128, help="Max time for flexible T (iOS 16+)")
    p.add_argument("--backend", type=str, default="mlprogram", choices=["mlprogram"], help="CoreML backend (mlprogram recommended)")
    p.add_argument("--ios-target", type=str, default="iOS16", help="Minimum deployment target, e.g., iOS16, iOS17")
    p.add_argument("--no-fp16", action="store_true", help="Disable FP16 compression of weights")
    p.add_argument("--prefer-script", action="store_true", help="Prefer torch.jit.script over trace")
    p.add_argument("--via-onnx", action="store_true", help="If set, export via ONNX instead of TorchScript")
    p.add_argument("--opset", type=int, default=13, help="ONNX opset (default 13)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    ckpt = Path(args.ckpt)
    out_path = Path(args.out).with_suffix(".mlmodel")

    if args.fixed_t is None and (args.min_t is None or args.max_t is None):
        # default flexible window if nothing provided explicitly
        args.min_t, args.max_t = 32, 128

    use_fp16 = (not args.no_fp16)

    log(f"Loading checkpoint: {ckpt}")
    state = load_state(ckpt)

    num_classes = infer_num_classes(state)
    log("Building model...")
    model = build_model(state, num_classes=num_classes)

    log("Preparing dummy input...")
    example, C = make_example(model, V=args.vertices, T=(args.fixed_t or args.min_t), channels=args.channels)

    labels = load_labels(Path(args.labels)) if args.labels else None
    if labels and len(labels) != num_classes:
        log(f"WARNING: labels count ({len(labels)}) != num_classes inferred ({num_classes})")

    if not args.via_onnx:
        ts = try_torchscript(model, example, prefer_script=args.prefer_script)
        to_coreml_from_torchscript(
            ts_mod=ts,
            C=C,
            V=args.vertices,
            min_t=args.min_t,
            max_t=args.max_t,
            fixed_t=args.fixed_t,
            class_labels=labels,
            backend=args.backend,
            ios_target=args.ios_target,
            out_path=out_path,
            use_fp16=use_fp16,
        )
    else:
        onnx_path = out_path.with_suffix(".onnx")
        to_coreml_via_onnx(
            model=model,
            example=example,
            onnx_path=onnx_path,
            C=C,
            V=args.vertices,
            min_t=args.min_t,
            max_t=args.max_t,
            fixed_t=args.fixed_t,
            class_labels=labels,
            backend=args.backend,
            ios_target=args.ios_target,
            out_path=out_path,
            use_fp16=use_fp16,
            opset=args.opset,
        )

    log("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)
