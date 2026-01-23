from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import coremltools as ct

from .logging_utils import log


def _build_input_type(
    C: int,
    V: int,
    min_t: Optional[int],
    max_t: Optional[int],
    fixed_t: Optional[int],
) -> ct.TensorType:
    if fixed_t is not None:
        shape = (1, C, V, fixed_t)
    else:
        if min_t is None or max_t is None:
            raise ValueError("For flexible T, --min-t and --max-t must be provided.")
        shape = (1, C, V, ct.RangeDim(min_t, max_t))
    return ct.TensorType(name="input", shape=shape, dtype=np.float32)


def to_coreml_from_torchscript(
    ts_mod: torch.jit.ScriptModule,
    C: int,
    V: int,
    min_t: Optional[int],
    max_t: Optional[int],
    fixed_t: Optional[int],
    class_labels: Optional[List[str]],
    backend: str,
    ios_target: str,
    out_path: Path,
    use_fp16: bool,
) -> ct.models.MLModel:
    inputs = [_build_input_type(C, V, min_t, max_t, fixed_t)]
    classifier_cfg = ct.ClassifierConfig(class_labels) if class_labels else None

    log("Converting TorchScript → CoreML (MIL/mlprogram)...")
    mlmodel = ct.convert(
        ts_mod,
        inputs=inputs,
        classifier_config=classifier_cfg,
        convert_to=backend,                      # 'mlprogram' recommended
        minimum_deployment_target=getattr(ct.target, ios_target),
    )

    if use_fp16:
        try:
            log("Applying FP16 weight compression...")
            mlmodel = ct.utils.convert_neural_network_weights_to_fp16(mlmodel)
        except Exception as exc:
            log(f"FP16 conversion not applied: {exc}")

    mlmodel.short_description = "ST-GCN gesture recognizer (42 hand keypoints × T)"
    add_vel = "unknown"
    try:
        add_vel = str(getattr(ts_mod, "add_vel", getattr(ts_mod, "_c", object()).__getattribute__("add_vel")))
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
    class_labels: Optional[List[str]],
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

    inputs = [_build_input_type(C, V, min_t, max_t, fixed_t)]
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
        except Exception as exc:
            log(f"FP16 conversion not applied: {exc}")

    mlmodel.short_description = "ST-GCN gesture recognizer (42 hand keypoints × T) [ONNX path]"
    mlmodel.user_defined_metadata.update({
        "input_format": f"NCHW (B=1, C={C}, V={V}, T={'flexible' if fixed_t is None else fixed_t})",
        "exported_by": "export_coreml.py",
    })

    mlmodel.save(str(out_path))
    log(f"Saved → {out_path}")
    return mlmodel
