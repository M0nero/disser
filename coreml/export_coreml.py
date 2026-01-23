import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import coremltools as ct

# Your project module
from msagcn.model import STGCN  # ensure model.py is on PYTHONPATH / in same folder


# -------------------------------
# Utilities
# -------------------------------
def log(msg: str) -> None:
    print(f"[export] {msg}")


def load_state(ckpt_path: Path) -> dict:
    """Load a checkpoint and return a clean state_dict."""
    # Torch 2.6+ defaults weights_only=True which breaks older checkpoints.
    # Try safe load first, then allowlist numpy scalar if needed.
    try:
        obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except Exception:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])  # trusted local checkpoints
        obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    # common nesting patterns
    for key in ("state_dict", "model_state", "model", "net", "ema_state_dict"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            return obj[key]
    # some training loops save torch.nn.Module directly
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if not isinstance(obj, dict):
        raise ValueError("Unsupported checkpoint format: expected dict or object with state_dict()")
    return obj


def infer_num_classes(state: dict) -> int:
    """Infer num_classes from the last linear head weight (2D tensor)."""
    candidates = []
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim == 2 and ("head" in k or "classifier" in k or k.endswith(".fc.weight")):
            # v shape: (out_features, in_features)
            candidates.append((k, v.shape[0]))
    if not candidates:
        # fallback: pick the largest 2D weight (likely final FC)
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                candidates.append((k, v.shape[0]))
    if not candidates:
        raise ValueError("Could not infer num_classes: no 2D linear weights found in state_dict.")
    # prefer keys with 'head' / 'classifier' by ordering
    candidates.sort(key=lambda kv: (0 if ("head" in kv[0] or "classifier" in kv[0]) else 1, -kv[1]))
    chosen_key, num = candidates[0]
    log(f"Inferred num_classes={num} from '{chosen_key}'")
    return int(num)


def build_model(state: dict, num_classes: int) -> STGCN:
    model = STGCN(num_classes=num_classes)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log(f"Loaded state with {len(missing)} missing keys (ok for buffers / non-critical): e.g. {missing[:5]}")
    if unexpected:
        log(f"State had {len(unexpected)} unexpected keys (ignored): e.g. {unexpected[:5]}")
    model.eval()
    return model


def make_example(model: STGCN, V: int, T: int, channels: Optional[int]) -> Tuple[torch.Tensor, int]:
    # Infer channels C from model config (data_bn) to avoid mismatches.
    if channels is None:
        C = None
        if hasattr(model, "data_bn"):
            try:
                feats = int(model.data_bn.num_features)
                base = feats // V
                C = base // 2 if getattr(model, "add_vel", False) else base
            except Exception:
                C = None
        if C is None:
            C = 3  # sensible default (x,y,z) per joint
    else:
        C = int(channels)
    example = torch.randn(1, C, V, T)
    return example, C


def try_torchscript(model: torch.nn.Module, example: torch.Tensor, prefer_script: bool=True) -> torch.jit.ScriptModule:
    with torch.no_grad():
        if prefer_script:
            try:
                log("Trying torch.jit.script()...")
                scripted = torch.jit.script(model)
                torch.jit.freeze(scripted)
                return scripted
            except Exception as e:
                log(f"script() failed: {e}")
        log("Falling back to torch.jit.trace()...")
        traced = torch.jit.trace(model, example, strict=False)
        torch.jit.freeze(traced)
        return traced


def to_coreml_from_torchscript(
    ts_mod: torch.jit.ScriptModule,
    C: int,
    V: int,
    min_t: Optional[int],
    max_t: Optional[int],
    fixed_t: Optional[int],
    class_labels: Optional[list],
    backend: str,
    ios_target: str,
    out_path: Path,
    use_fp16: bool,
) -> ct.models.MLModel:
    # Input tensor spec
    if fixed_t is not None:
        shape = (1, C, V, fixed_t)
    else:
        if min_t is None or max_t is None:
            raise ValueError("For flexible T, --min-t and --max-t must be provided.")
        shape = (1, C, V, ct.RangeDim(min_t, max_t))

    inputs = [ct.TensorType(name="input", shape=shape, dtype=np.float32)]

    classifier_cfg = ct.ClassifierConfig(class_labels) if class_labels else None

    log("Converting TorchScript → CoreML (MIL/mlprogram)...")
    mlmodel = ct.convert(
        ts_mod,
        inputs=inputs,
        classifier_config=classifier_cfg,
        convert_to=backend,                      # 'mlprogram' recommended
        minimum_deployment_target=getattr(ct.target, ios_target),
    )

    # Optional FP16 compression (safe for most models)
    if use_fp16:
        try:
            log("Applying FP16 weight compression...")
            mlmodel = ct.utils.convert_neural_network_weights_to_fp16(mlmodel)
        except Exception as e:
            log(f"FP16 conversion not applied: {e}")

    # Metadata
    mlmodel.short_description = "ST-GCN gesture recognizer (42 hand keypoints × T)"
    add_vel = "unknown"
    try:
        add_vel = str(getattr(ts_mod, "add_vel", getattr(ts_mod, "_c", object()).__getattribute__("add_vel")))  # best effort
    except Exception:
        pass
    mlmodel.user_defined_metadata.update({
        "input_format": f"NCHW (B=1, C={C}, V={V}, T={'flexible' if fixed_t is None else fixed_t})",
        "add_vel": add_vel,
        "exported_by": "export_coreml.py",
    })

    mlmodel.save(str(out_path))
    log(f"Saved → {out_path}")
    return mlmodel


def to_coreml_via_onnx(
    model: torch.nn.Module,
    example: torch.Tensor,
    onnx_path: Path,
    C: int,
    V: int,
    min_t: Optional[int],
    max_t: Optional[int],
    fixed_t: Optional[int],
    class_labels: Optional[list],
    backend: str,
    ios_target: str,
    out_path: Path,
    use_fp16: bool,
    opset: int,
) -> ct.models.MLModel:
    log("Exporting ONNX...")
    dynamic_axes = None
    if fixed_t is None:
        if min_t is None or max_t is None:
            raise ValueError("For flexible T, --min-t and --max-t must be provided.")
        dynamic_axes = {"input": {3: "time"}}
    torch.onnx.export(
        model,
        example,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
    )
    log(f"Saved ONNX → {onnx_path}")

    if fixed_t is not None:
        shape = (1, C, V, fixed_t)
        inputs = [ct.TensorType(name="input", shape=shape, dtype=np.float32)]
    else:
        shape = (1, C, V, ct.RangeDim(min_t, max_t))
        inputs = [ct.TensorType(name="input", shape=shape, dtype=np.float32)]

    classifier_cfg = ct.ClassifierConfig(class_labels) if class_labels else None

    log("Converting ONNX → CoreML...")
    mlmodel = ct.convert(
        str(onnx_path),
        inputs=inputs,
        classifier_config=classifier_cfg,
        convert_to=backend,
        minimum_deployment_target=getattr(ct.target, ios_target),
    )

    if use_fp16:
        try:
            log("Applying FP16 weight compression...")
            mlmodel = ct.utils.convert_neural_network_weights_to_fp16(mlmodel)
        except Exception as e:
            log(f"FP16 conversion not applied: {e}")

    mlmodel.short_description = "ST-GCN gesture recognizer (42 hand keypoints × T) [ONNX path]"
    mlmodel.user_defined_metadata.update({
        "input_format": f"NCHW (B=1, C={C}, V={V}, T={'flexible' if fixed_t is None else fixed_t})",
        "exported_by": "export_coreml.py",
    })

    mlmodel.save(str(out_path))
    log(f"Saved → {out_path}")
    return mlmodel


def load_labels(path: Optional[Path]) -> Optional[list]:
    if not path:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    if path.suffix.lower() in {".txt"}:
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if path.suffix.lower() in {".json"}:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x) for x in data]
        raise ValueError("JSON labels must be an array of strings.")
    raise ValueError("Unsupported labels format. Use .txt (one per line) or .json (array).")


def main():
    p = argparse.ArgumentParser(description="Export ST-GCN PyTorch checkpoint to CoreML .mlmodel")
    p.add_argument("--ckpt", type=str, default="best_model.pth", help="Path to PyTorch checkpoint (.pth)")
    p.add_argument("--out", type=str, default="STGCN.mlmodel", help="Output .mlmodel path")
    p.add_argument("--labels", type=str, default=None, help="Optional labels file (.txt or .json)")
    p.add_argument("--vertices", type=int, default=42, help="Number of graph vertices V (default: 42 for two hands)")
    p.add_argument("--channels", type=int, default=None, help="Override channels C (default: infer from model.add_vel)"),
    p.add_argument("--fixed-t", type=int, default=None, help="Fix time length T (use this OR --min-t/--max-t)")
    p.add_argument("--min-t", type=int, default=32, help="Min time for flexible T (iOS 16+)")
    p.add_argument("--max-t", type=int, default=128, help="Max time for flexible T (iOS 16+)")
    p.add_argument("--backend", type=str, default="mlprogram", choices=["mlprogram"], help="CoreML backend (mlprogram recommended)")
    p.add_argument("--ios-target", type=str, default="iOS16", help="Minimum deployment target, e.g., iOS16, iOS17")
    p.add_argument("--no-fp16", action="store_true", help="Disable FP16 compression of weights")
    p.add_argument("--prefer-script", action="store_true", help="Prefer torch.jit.script over trace")
    p.add_argument("--via-onnx", action="store_true", help="If set, export via ONNX instead of TorchScript")
    p.add_argument("--opset", type=int, default=13, help="ONNX opset (default 13)")
    args = p.parse_args()

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
            C=C, V=args.vertices,
            min_t=args.min_t, max_t=args.max_t, fixed_t=args.fixed_t,
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
            C=C, V=args.vertices,
            min_t=args.min_t, max_t=args.max_t, fixed_t=args.fixed_t,
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
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)
