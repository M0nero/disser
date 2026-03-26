from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from msagcn.models import MultiStreamAGCN


def checkpoint_arg(saved_args: dict[str, Any], key: str, default: Any) -> Any:
    return saved_args[key] if isinstance(saved_args, dict) and key in saved_args else default


def load_checkpoint_training_state(ckpt_path: str | Path) -> dict[str, Any]:
    path = Path(ckpt_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved_args = payload.get("args")
    ds_cfg = payload.get("ds_cfg")
    label2idx = payload.get("label2idx")
    if not isinstance(saved_args, dict):
        raise ValueError("Checkpoint is missing saved args metadata")
    if not isinstance(ds_cfg, dict):
        raise ValueError("Checkpoint is missing ds_cfg metadata")
    if not isinstance(label2idx, dict):
        raise ValueError("Checkpoint is missing label2idx metadata")
    return {
        "path": str(path.resolve()),
        "payload": payload,
        "args": saved_args,
        "ds_cfg": ds_cfg,
        "label2idx": label2idx,
    }


def build_model_from_saved_args(
    *,
    saved_args: dict[str, Any],
    train_ds,
    num_classes: int,
    use_family_head: bool = False,
    num_families: int = 0,
) -> MultiStreamAGCN:
    depths = tuple(int(x) for x in str(checkpoint_arg(saved_args, "depths", "64,128,256,256")).split(","))
    temp_ks = tuple(int(x) for x in str(checkpoint_arg(saved_args, "temp_ks", "9,7,5,5")).split(","))
    return MultiStreamAGCN(
        num_classes=int(num_classes),
        V=train_ds.V,
        A=train_ds.build_adjacency(normalize=False),
        in_ch=3,
        streams=tuple(getattr(train_ds.cfg, "use_streams", ("joints", "bones", "velocity"))),
        drop=float(checkpoint_arg(saved_args, "drop", 0.10)),
        droppath=float(checkpoint_arg(saved_args, "droppath", 0.05)),
        depths=depths,
        temp_ks=temp_ks,
        use_groupnorm_stem=bool(checkpoint_arg(saved_args, "use_groupnorm_stem", False)),
        stream_drop_p=float(checkpoint_arg(saved_args, "stream_drop_p", 0.10)),
        use_cosine_head=bool(checkpoint_arg(saved_args, "use_cosine_head", False)),
        cosine_margin=float(checkpoint_arg(saved_args, "cosine_margin", 0.2)),
        cosine_scale=float(checkpoint_arg(saved_args, "cosine_scale", 30.0)),
        cosine_subcenters=int(checkpoint_arg(saved_args, "cosine_subcenters", 1)),
        use_family_head=bool(use_family_head),
        num_families=int(num_families),
        use_ctr_hand_refine=bool(checkpoint_arg(saved_args, "use_ctr_hand_refine", False)),
        ctr_in_stream_encoder=bool(checkpoint_arg(saved_args, "ctr_in_stream_encoder", False)),
        ctr_groups=int(checkpoint_arg(saved_args, "ctr_groups", 4)),
        ctr_hand_nodes=int(checkpoint_arg(saved_args, "ctr_hand_nodes", 42)),
        ctr_rel_channels=checkpoint_arg(saved_args, "ctr_rel_channels", None),
        ctr_alpha_init=float(checkpoint_arg(saved_args, "ctr_alpha_init", 0.0)),
    )


def resolve_amp_settings(saved_args: dict[str, Any], device: torch.device) -> tuple[bool, torch.dtype]:
    no_amp = bool(checkpoint_arg(saved_args, "no_amp", False))
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.float32
    return bool(device.type == "cuda" and not no_amp), amp_dtype


def load_checkpoint_weights(model: torch.nn.Module, payload: dict[str, Any], *, prefer_ema: bool = False) -> None:
    state = payload.get("ema_state") if prefer_ema and payload.get("ema_state") is not None else payload["model_state"]
    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError as exc:
        original_exc = exc
    if isinstance(state, dict) and state and all(str(key).startswith("_orig_mod.") for key in state.keys()):
        remapped = {str(key)[10:]: value for key, value in state.items()}
        model.load_state_dict(remapped, strict=True)
        return
    raise original_exc


def make_sample_id(video: str, label: str, begin: int, end_excl: int) -> str:
    base = f"{video}|{label}|{int(begin)}|{int(end_excl)}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def compute_kinematic_signature(sample: dict[str, torch.Tensor]) -> np.ndarray:
    joints = sample.get("joints")
    velocity = sample.get("velocity")
    bones = sample.get("bones")
    mask = sample.get("mask")
    if joints is None or velocity is None or bones is None or mask is None:
        raise ValueError("Expected sample to contain joints, bones, velocity, and mask tensors")

    vel = velocity.detach().float().cpu()
    bone = bones.detach().float().cpu()
    valid = mask.detach().float().cpu()
    if valid.ndim == 3:
        valid = valid.squeeze(0)
    if valid.ndim != 2:
        raise ValueError(f"Expected mask with shape (1,V,T) or (V,T), got {tuple(mask.shape)}")

    vel_energy = torch.linalg.vector_norm(vel, dim=0)  # (V,T)
    bone_mag = torch.linalg.vector_norm(bone, dim=0)  # (V,T)
    valid_sum = valid.sum().clamp(min=1.0)

    mean_abs_vel = float((vel_energy * valid).sum().item() / valid_sum.item())
    centered = (vel_energy - mean_abs_vel) * valid
    std_abs_vel = float(torch.sqrt((centered.pow(2).sum() / valid_sum).clamp(min=0.0)).item())
    peak_vel = float(vel_energy.max().item()) if vel_energy.numel() > 0 else 0.0

    hand_nodes = min(42, int(vel_energy.shape[0]))
    left_nodes = min(21, hand_nodes)
    right_nodes = max(0, hand_nodes - left_nodes)
    left_valid = valid[:left_nodes]
    right_valid = valid[left_nodes:hand_nodes]
    left_energy = float(((vel_energy[:left_nodes] * left_valid).sum() / left_valid.sum().clamp(min=1.0)).item()) if left_nodes > 0 else 0.0
    right_energy = (
        float(((vel_energy[left_nodes:hand_nodes] * right_valid).sum() / right_valid.sum().clamp(min=1.0)).item())
        if right_nodes > 0
        else 0.0
    )
    pose_energy = 0.0
    if int(vel_energy.shape[0]) > hand_nodes:
        pose_valid = valid[hand_nodes:]
        pose_energy = float(((vel_energy[hand_nodes:] * pose_valid).sum() / pose_valid.sum().clamp(min=1.0)).item())

    mean_bone_mag = float((bone_mag * valid).sum().item() / valid_sum.item())
    coverage = float(valid.mean().item())

    time_energy = (vel_energy * valid).sum(dim=0) / valid.sum(dim=0).clamp(min=1.0)
    bins = 8
    if int(time_energy.numel()) == 0:
        temporal_hist = np.zeros(bins, dtype=np.float32)
    else:
        splits = np.array_split(time_energy.numpy(), bins)
        temporal_hist = np.asarray([float(chunk.mean()) if len(chunk) > 0 else 0.0 for chunk in splits], dtype=np.float32)

    signature = np.asarray(
        [
            mean_abs_vel,
            std_abs_vel,
            peak_vel,
            left_energy,
            right_energy,
            pose_energy,
            mean_bone_mag,
            *temporal_hist.tolist(),
            coverage,
        ],
        dtype=np.float32,
    )
    return signature
