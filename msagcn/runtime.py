from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from msagcn.data.config import DSConfig
from msagcn.data.topology import CROSS_EDGE_PAIRS_ABS, NUM_HAND_NODES, POSE_EDGE_PAIRS_ABS, hand_edges_42
from msagcn.models import MultiStreamAGCN
from runtime.manifest import copy_into_bundle, load_runtime_manifest, resolve_bundle_path, write_runtime_manifest
from runtime.skeleton import CANONICAL_SKELETON_SPEC, CanonicalSkeletonSequence


MSAGCN_RUNTIME_PREPROCESSING_VERSION = "canonical_hands42_optional_pose_v2"


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


def _parse_int_tuple(value: Any, default: Sequence[int]) -> tuple[int, ...]:
    if value is None:
        return tuple(int(x) for x in default)
    if isinstance(value, (list, tuple)):
        return tuple(int(x) for x in value)
    text = str(value).strip()
    if not text:
        return tuple(int(x) for x in default)
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def _resample_indices(length: int, target: int) -> list[int]:
    if target <= 0:
        return []
    if length <= 1:
        return [0] * target
    if target == 1:
        return [0]
    out = [int(round(i * (length - 1) / float(target - 1))) for i in range(target)]
    out[0] = 0
    out[-1] = length - 1
    return [min(max(v, 0), length - 1) for v in out]


def _build_parent_map(edges: Sequence[tuple[int, int]], v: int) -> list[int]:
    parent = [-1] * int(v)
    for p, c in edges:
        parent[int(c)] = int(p)
    return parent


def _load_json_object(path: str | Path) -> Dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {source}")
    return payload


def _normalize_label2idx(raw: Dict[str, Any]) -> Dict[str, int]:
    return {str(k): int(v) for k, v in dict(raw or {}).items()}


def _coerce_ds_cfg(raw: Any) -> DSConfig:
    if isinstance(raw, DSConfig):
        return raw
    if not isinstance(raw, dict):
        raise ValueError("Expected ds_config as DSConfig or dict")
    payload = dict(raw)
    if "use_streams" in payload and not isinstance(payload["use_streams"], tuple):
        payload["use_streams"] = tuple(str(x) for x in payload["use_streams"])
    if "pose_keep" in payload and not isinstance(payload["pose_keep"], tuple):
        payload["pose_keep"] = tuple(int(x) for x in payload["pose_keep"])
    return DSConfig(**payload)


def _resolve_label2idx(
    payload: Dict[str, Any],
    ckpt_path: Path,
    *,
    label_map_path: str | Path | None = None,
) -> tuple[Dict[str, int], str]:
    source = ""
    label2idx: Dict[str, int] = {}
    if label_map_path:
        label2idx = _normalize_label2idx(_load_json_object(label_map_path))
        source = "external_label_map"
    elif payload.get("label2idx"):
        label2idx = _normalize_label2idx(payload.get("label2idx"))
        source = "checkpoint"
    else:
        sibling = ckpt_path.parent / "label2idx.json"
        if sibling.exists():
            label2idx = _normalize_label2idx(_load_json_object(sibling))
            source = "sibling_label_map"
    if not label2idx:
        raise RuntimeError(f"MSAGCN metadata missing label2idx for {ckpt_path}")
    payload_label2idx = _normalize_label2idx(payload.get("label2idx", {}) or {})
    if source != "checkpoint" and payload_label2idx and payload_label2idx != label2idx:
        raise RuntimeError("MSAGCN label map metadata does not match checkpoint payload")
    return label2idx, source


def _resolve_ds_cfg(
    payload: Dict[str, Any],
    ckpt_path: Path,
    *,
    ds_config_path: str | Path | None = None,
) -> tuple[DSConfig, str]:
    source = ""
    if ds_config_path:
        ds_cfg = _coerce_ds_cfg(_load_json_object(ds_config_path))
        source = "external_ds_config"
    elif payload.get("ds_cfg"):
        ds_cfg = _coerce_ds_cfg(payload.get("ds_cfg"))
        source = "checkpoint"
    else:
        sibling = ckpt_path.parent / "ds_config.json"
        if sibling.exists():
            ds_cfg = _coerce_ds_cfg(_load_json_object(sibling))
            source = "sibling_ds_config"
        else:
            raise RuntimeError(f"MSAGCN metadata missing ds_cfg for {ckpt_path}")
    payload_ds_cfg = payload.get("ds_cfg")
    if source != "checkpoint" and isinstance(payload_ds_cfg, dict):
        if asdict(_coerce_ds_cfg(payload_ds_cfg)) != asdict(ds_cfg):
            raise RuntimeError("MSAGCN ds_config metadata does not match checkpoint payload")
    return ds_cfg, source


def _resolve_family_map(
    ckpt_path: Path,
    *,
    manifest: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Any], str]:
    candidates: list[tuple[Path, str]] = []
    if manifest is not None and manifest.get("family_map"):
        candidates.append((resolve_bundle_path(manifest, str(manifest.get("family_map"))), "bundle_family_map"))
    candidates.extend(
        [
            (ckpt_path.parent / "family_map.json", "sibling_family_map"),
            (ckpt_path.parent.parent / "family_map.json", "parent_family_map"),
        ]
    )
    for path, source in candidates:
        if path.exists():
            return _load_json_object(path), source
    return {}, ""


def _infer_num_families(state_dict: Dict[str, Any], ckpt_args: Dict[str, Any]) -> int:
    for key in ("family_head.3.weight", "family_head.weight"):
        tensor = state_dict.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2:
            return int(tensor.shape[0])
    return int(ckpt_args.get("num_families", 0) or 0)


@dataclass
class MSAGCNPrediction:
    label: str
    class_id: int
    confidence: float
    topk: List[Dict[str, Any]]
    logits: List[float]
    probs: List[float]
    meta: Dict[str, Any]
    family_class_id: int | None = None
    family_label: str | None = None
    family_confidence: float | None = None
    family_topk: List[Dict[str, Any]] | None = None
    family_logits: List[float] | None = None
    family_probs: List[float] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MSAGCNFeatureBuilder:
    def __init__(self, ds_cfg: DSConfig) -> None:
        self.ds_cfg = ds_cfg
        self.pose_keep = list(self.ds_cfg.pose_keep) if bool(self.ds_cfg.include_pose) else []
        self.P = len(self.pose_keep) if bool(self.ds_cfg.include_pose) else 0
        self.V = int(NUM_HAND_NODES + self.P)
        self._edges = self._build_edges()
        parent_idx = torch.tensor(_build_parent_map(self._edges, self.V), dtype=torch.long)
        child_mask = parent_idx >= 0
        self._child_idx = torch.where(child_mask)[0]
        self._par_idx = parent_idx[self._child_idx]
        self._mirror_idx = self._build_mirror_idx()
        self._pose_wrist_out = (
            self.pose_keep.index(15) if 15 in self.pose_keep else -1,
            self.pose_keep.index(16) if 16 in self.pose_keep else -1,
        )

    def _build_edges(self) -> list[tuple[int, int]]:
        edges = hand_edges_42()
        if not self.ds_cfg.include_pose or self.P <= 0:
            return edges
        pos_map = {abs_idx: i for i, abs_idx in enumerate(self.pose_keep)}
        for a, b in POSE_EDGE_PAIRS_ABS:
            if a in pos_map and b in pos_map:
                edges.append((NUM_HAND_NODES + pos_map[a], NUM_HAND_NODES + pos_map[b]))
        if self.ds_cfg.connect_cross_edges:
            for hand_tag, pose_abs in CROSS_EDGE_PAIRS_ABS:
                if pose_abs not in pos_map:
                    continue
                pose_idx = NUM_HAND_NODES + pos_map[pose_abs]
                hand_idx = 0 if hand_tag == "LWRIST" else 21
                edges.append((hand_idx, pose_idx))
        return edges

    def _build_mirror_idx(self) -> torch.Tensor:
        idx = list(range(21, 42)) + list(range(0, 21))
        if self.ds_cfg.include_pose and self.P > 0:
            pairs = {(9, 10), (11, 12), (13, 14), (15, 16), (23, 24)}
            pos_map = {abs_i: i for i, abs_i in enumerate(self.pose_keep)}
            for abs_i in self.pose_keep:
                if abs_i == 0:
                    idx.append(NUM_HAND_NODES + pos_map[0])
                    continue
                pair_abs = None
                for a, b in pairs:
                    if abs_i == a:
                        pair_abs = b
                    elif abs_i == b:
                        pair_abs = a
                if pair_abs is None or pair_abs not in pos_map:
                    idx.append(NUM_HAND_NODES + pos_map[abs_i])
                else:
                    idx.append(NUM_HAND_NODES + pos_map[pair_abs])
        return torch.tensor(idx, dtype=torch.long)

    def build_adjacency(self, normalize: bool | str = False) -> torch.Tensor:
        A = torch.zeros((self.V, self.V), dtype=torch.float32)
        for p, c in self._edges:
            A[p, c] = 1.0
            A[c, p] = 1.0
        if normalize is True or normalize == "row":
            d = A.sum(dim=1, keepdim=True).clamp_min(1e-6)
            A = A / d
        elif normalize in ("sym", "symmetric", "symm"):
            d = A.sum(dim=1).clamp_min(1e-6)
            D_inv_sqrt = torch.diag(torch.pow(d, -0.5))
            A = D_inv_sqrt @ A @ D_inv_sqrt
        return A

    def _center_norm(self, pts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.ds_cfg.center:
            if str(self.ds_cfg.center_mode) == "wrists":
                wrists = []
                if float(mask[:, 0, :].sum()) > 0:
                    wrists.append(pts[:, 0:1, :])
                if mask.shape[1] > 21 and float(mask[:, 21, :].sum()) > 0:
                    wrists.append(pts[:, 21:22, :])
                if wrists:
                    center = torch.cat(wrists, dim=1).mean(dim=1, keepdim=True)
                    pts = pts - center
                else:
                    denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
                    mean = (pts * mask).sum(dim=1, keepdim=True) / denom
                    pts = pts - mean
            else:
                denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
                mean = (pts * mask).sum(dim=1, keepdim=True) / denom
                pts = pts - mean
        if self.ds_cfg.normalize:
            if str(self.ds_cfg.norm_method) == "p95":
                flat = pts.abs().reshape(-1)
                k = max(1, int(flat.numel() * 0.95))
                span = flat.kthvalue(k)[0]
            elif str(self.ds_cfg.norm_method) == "mad":
                med = pts.median()
                if hasattr(med, "values"):
                    med = med.values
                mad = (pts - med).abs().median()
                if hasattr(mad, "values"):
                    mad = mad.values
                span = torch.clamp(mad * 1.4826, min=1e-6)
            else:
                span = pts.abs().amax()
            if float(span) > 1e-6:
                scale = span / max(1e-6, float(self.ds_cfg.norm_scale))
                pts = pts / (scale + 1e-6)
        return pts

    def _select_pose_subset(self, seq: CanonicalSkeletonSequence, idx: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if seq.pose_xyz is None or seq.pose_vis is None:
            raise RuntimeError(
                "MSAGCN runtime checkpoint requires pose, but the runtime skeleton sequence does not contain pose sidecar data"
            )
        pose_src = np.asarray(seq.pose_xyz[idx], dtype=np.float32)
        pose_vis_src = np.asarray(seq.pose_vis[idx], dtype=np.float32)
        seq_pose_indices = list(seq.pose_indices or range(int(pose_src.shape[1])))
        pos_map = {abs_idx: i for i, abs_idx in enumerate(seq_pose_indices)}
        T = pose_src.shape[0]
        pose_xyz = torch.zeros((T, self.P, 3), dtype=torch.float32)
        pose_valid = torch.zeros((T, self.P), dtype=torch.float32)
        for out_idx, abs_idx in enumerate(self.pose_keep):
            src_idx = pos_map.get(abs_idx, -1)
            if src_idx < 0:
                continue
            pose_xyz[:, out_idx, :] = torch.from_numpy(pose_src[:, src_idx, :])
            pose_valid[:, out_idx] = torch.from_numpy((pose_vis_src[:, src_idx] >= float(self.ds_cfg.pose_vis_thr)).astype(np.float32))
        return pose_xyz, pose_valid

    def prepare_sequence(self, seq: CanonicalSkeletonSequence) -> Dict[str, torch.Tensor]:
        if seq.length <= 0:
            raise ValueError("MSAGCN runtime requires a non-empty sequence")
        idx = _resample_indices(seq.length, int(self.ds_cfg.max_frames))
        hands_pts = torch.as_tensor(seq.pts[idx], dtype=torch.float32)
        hands_mask = torch.as_tensor(seq.mask[idx], dtype=torch.float32)
        ts = torch.as_tensor(seq.ts_ms[idx], dtype=torch.float32)
        T = int(hands_pts.shape[0])

        pts = torch.zeros((T, self.V, 3), dtype=torch.float32)
        mask = torch.zeros((T, self.V, 1), dtype=torch.float32)
        pts[:, :NUM_HAND_NODES, :] = hands_pts * hands_mask
        mask[:, :NUM_HAND_NODES, :] = hands_mask

        if self.ds_cfg.include_pose and self.P > 0:
            pose_xyz, pose_valid = self._select_pose_subset(seq, idx)
            # `CanonicalSkeletonSequence` already stores pose coordinates after one
            # visibility/mask application. Runtime parity with the train-time dataset
            # therefore requires keeping the pose coordinates as-is here and using
            # `mask` alone to mark invalid pose nodes.
            pts[:, NUM_HAND_NODES : NUM_HAND_NODES + self.P, :] = pose_xyz
            mask[:, NUM_HAND_NODES : NUM_HAND_NODES + self.P, 0] = pose_valid
            coords_tag = str(seq.meta.get("coords", "image")).lower()
            allow_pose_wrist = coords_tag != "world"
            wrist_left_out, wrist_right_out = self._pose_wrist_out
            if allow_pose_wrist and wrist_left_out >= 0:
                left_pose_valid = pose_valid[:, wrist_left_out] > 0.0
                missing_left = mask[:, 0, 0] == 0.0
                use_left = torch.logical_and(left_pose_valid, missing_left)
                if bool(use_left.any()):
                    pts[use_left, 0, :] = pose_xyz[use_left, wrist_left_out, :]
                    mask[use_left, 0, 0] = 1.0
            if allow_pose_wrist and wrist_right_out >= 0:
                right_pose_valid = pose_valid[:, wrist_right_out] > 0.0
                missing_right = mask[:, 21, 0] == 0.0
                use_right = torch.logical_and(right_pose_valid, missing_right)
                if bool(use_right.any()):
                    pts[use_right, 21, :] = pose_xyz[use_right, wrist_right_out, :]
                    mask[use_right, 21, 0] = 1.0

        m = mask.permute(1, 2, 0)
        m = F.avg_pool1d(m, kernel_size=3, stride=1, padding=1)
        mask = (m >= 0.5).float().permute(2, 0, 1).contiguous()

        left_cnt = float(mask[:, 0:21, :].sum())
        right_cnt = float(mask[:, 21:42, :].sum())
        T_frames = mask.shape[0]
        if left_cnt < 0.1 * T_frames * 21 and right_cnt >= 0.6 * T_frames * 21:
            pts = pts[:, self._mirror_idx, :]
            mask = mask[:, self._mirror_idx, :]

        pts = torch.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)
        pts = self._center_norm(pts, mask)

        out: Dict[str, torch.Tensor] = {}
        if "joints" in self.ds_cfg.use_streams:
            out["joints"] = pts.permute(2, 1, 0).contiguous().unsqueeze(0)

        if "velocity" in self.ds_cfg.use_streams:
            vel = torch.zeros_like(pts)
            use_ts = ts.numel() > 1 and torch.isfinite(ts).all() and float((ts[1:] - ts[:-1]).abs().sum()) > 0.0
            if use_ts:
                dt_ms = torch.zeros_like(ts)
                dt_ms[1:] = torch.clamp(ts[1:] - ts[:-1], min=1.0)
                dt_s = dt_ms / 1000.0
            else:
                fps = float(seq.meta.get("fps", 0.0) or 0.0)
                if fps <= 0.0:
                    fps = 30.0
                dt_s = torch.full_like(ts, 1.0 / max(fps, 1e-6))
            vel[1:] = (pts[1:] - pts[:-1]) / dt_s[1:].view(-1, 1, 1).clamp_min(1e-6)
            out["velocity"] = vel.permute(2, 1, 0).contiguous().unsqueeze(0)

        if "bones" in self.ds_cfg.use_streams:
            bones = torch.zeros_like(pts)
            bones[:, self._child_idx, :] = pts[:, self._child_idx, :] - pts[:, self._par_idx, :]
            out["bones"] = bones.permute(2, 1, 0).contiguous().unsqueeze(0)

        out["mask"] = mask.permute(2, 1, 0).contiguous().unsqueeze(0)
        return out


def _build_msagcn_model(
    label2idx: Dict[str, int],
    ds_cfg: DSConfig,
    ckpt_args: Dict[str, Any],
    builder: MSAGCNFeatureBuilder,
    *,
    state_dict: Dict[str, Any],
) -> MultiStreamAGCN:
    depths = _parse_int_tuple(ckpt_args.get("depths"), (64, 128, 256, 256))
    temp_ks = _parse_int_tuple(ckpt_args.get("temp_ks"), (9, 7, 5, 5))
    num_families = _infer_num_families(state_dict, ckpt_args)
    use_family_head = _as_bool(ckpt_args.get("use_family_head"), False) or num_families > 0
    return MultiStreamAGCN(
        num_classes=len(label2idx),
        V=builder.V,
        A=builder.build_adjacency(normalize=False),
        in_ch=3,
        streams=tuple(str(x) for x in ds_cfg.use_streams),
        drop=float(ckpt_args.get("drop", 0.25) or 0.25),
        droppath=float(ckpt_args.get("droppath", 0.1) or 0.1),
        depths=depths,
        temp_ks=temp_ks,
        use_groupnorm_stem=_as_bool(ckpt_args.get("use_groupnorm_stem"), True),
        stream_drop_p=float(ckpt_args.get("stream_drop_p", 0.15) or 0.15),
        use_cosine_head=_as_bool(ckpt_args.get("use_cosine_head"), False),
        cosine_margin=float(ckpt_args.get("cosine_margin", 0.20) or 0.20),
        cosine_scale=float(ckpt_args.get("cosine_scale", 30.0) or 30.0),
        cosine_subcenters=int(ckpt_args.get("cosine_subcenters", 1) or 1),
        use_family_head=use_family_head,
        num_families=num_families,
        use_ctr_hand_refine=_as_bool(ckpt_args.get("use_ctr_hand_refine"), False),
        ctr_in_stream_encoder=_as_bool(ckpt_args.get("ctr_in_stream_encoder"), False),
        ctr_groups=int(ckpt_args.get("ctr_groups", 4) or 4),
        ctr_hand_nodes=int(ckpt_args.get("ctr_hand_nodes", 42) or 42),
        ctr_rel_channels=(None if ckpt_args.get("ctr_rel_channels") in (None, "", 0, "0") else int(ckpt_args.get("ctr_rel_channels"))),
        ctr_alpha_init=float(ckpt_args.get("ctr_alpha_init", 0.0) or 0.0),
    )


def _validate_manifest_against_runtime(
    manifest: Dict[str, Any],
    *,
    label2idx: Dict[str, int],
    ds_cfg: DSConfig,
) -> None:
    if str(manifest.get("model_type", "")) != "msagcn":
        raise ValueError(f"MSAGCN runtime bundle expected model_type='msagcn', got {manifest.get('model_type')!r}")
    use_streams = tuple(str(x) for x in manifest.get("use_streams", []) or [])
    if use_streams and use_streams != tuple(str(x) for x in ds_cfg.use_streams):
        raise RuntimeError("MSAGCN runtime bundle use_streams does not match resolved ds_config")
    if "pose_enabled" in manifest and bool(manifest.get("pose_enabled")) != bool(ds_cfg.include_pose):
        raise RuntimeError("MSAGCN runtime bundle pose_enabled does not match resolved ds_config")
    if "max_frames" in manifest and int(manifest.get("max_frames")) != int(ds_cfg.max_frames):
        raise RuntimeError("MSAGCN runtime bundle max_frames does not match resolved ds_config")
    if "label_count" in manifest and int(manifest.get("label_count")) != int(len(label2idx)):
        raise RuntimeError("MSAGCN runtime bundle label_count does not match resolved label map")
    if "num_families" in manifest:
        family_count = 0
        metadata = dict(manifest.get("metadata", {}) or {})
        if "num_families" in metadata:
            family_count = int(metadata.get("num_families", 0))
        elif int(manifest.get("num_families", 0)) < 0:
            family_count = 0
        if family_count < 0:
            raise RuntimeError("MSAGCN runtime bundle num_families is invalid")
    spec = dict(manifest.get("skeleton_spec", {}) or {})
    if spec:
        if int(spec.get("num_joints", NUM_HAND_NODES)) != int(NUM_HAND_NODES):
            raise RuntimeError("MSAGCN runtime bundle skeleton_spec does not match canonical 42-hand runtime contract")


class MSAGCNClassifier:
    def __init__(
        self,
        model: MultiStreamAGCN,
        *,
        label2idx: Dict[str, int],
        ds_cfg: DSConfig,
        device: str | torch.device | None = None,
        metadata: Optional[Dict[str, Any]] = None,
        family_map: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.model = model.to(self.device).eval()
        self.ds_cfg = ds_cfg
        self.label2idx = dict(label2idx)
        self.idx2label = {int(v): str(k) for k, v in self.label2idx.items()}
        self.builder = MSAGCNFeatureBuilder(ds_cfg)
        self.A = self.builder.build_adjacency(normalize=False).to(self.device)
        self.family_map = dict(family_map or {})
        self.family_class_to_family = {int(k): int(v) for k, v in dict(self.family_map.get("class_to_family", {}) or {}).items()}
        family_names_raw = dict(self.family_map.get("family_names", {}) or {})
        self.family_idx2label = {int(k): str(v) for k, v in family_names_raw.items()}
        if not self.family_idx2label:
            num_families = int(getattr(self.model, "num_families", 0) or 0)
            self.family_idx2label = {idx: str(idx) for idx in range(num_families)}
        self.metadata = {
            "pose_enabled": bool(self.ds_cfg.include_pose),
            "preprocessing_version": MSAGCN_RUNTIME_PREPROCESSING_VERSION,
            "use_family_head": bool(getattr(self.model, "use_family_head", False)),
            "num_families": int(getattr(self.model, "num_families", 0) or 0),
            **dict(metadata or {}),
        }

    @classmethod
    def from_model(
        cls,
        model: MultiStreamAGCN,
        *,
        label2idx: Dict[str, int],
        ds_cfg: DSConfig,
        device: str | torch.device | None = None,
        metadata: Optional[Dict[str, Any]] = None,
        family_map: Optional[Dict[str, Any]] = None,
    ) -> "MSAGCNClassifier":
        return cls(model, label2idx=label2idx, ds_cfg=ds_cfg, device=device, metadata=metadata, family_map=family_map)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device | None = None,
        prefer_ema: bool = True,
        label_map_path: str | Path | None = None,
        ds_config_path: str | Path | None = None,
        manifest: Dict[str, Any] | None = None,
    ) -> "MSAGCNClassifier":
        ckpt_path = Path(checkpoint_path).expanduser().resolve()
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        label2idx, label_source = _resolve_label2idx(payload, ckpt_path, label_map_path=label_map_path)
        ds_cfg, ds_source = _resolve_ds_cfg(payload, ckpt_path, ds_config_path=ds_config_path)
        family_map, family_source = _resolve_family_map(ckpt_path, manifest=manifest)
        ckpt_args = dict(payload.get("args", {}) or {})
        if manifest is not None:
            _validate_manifest_against_runtime(manifest, label2idx=label2idx, ds_cfg=ds_cfg)
        state_dict = payload.get("ema_state") if prefer_ema and payload.get("ema_state") is not None else payload.get("model_state")
        if state_dict is None:
            raise RuntimeError(f"MSAGCN checkpoint has no model state: {ckpt_path}")
        builder = MSAGCNFeatureBuilder(ds_cfg)
        model = _build_msagcn_model(label2idx, ds_cfg, ckpt_args, builder, state_dict=state_dict)
        model.load_state_dict(state_dict, strict=True)
        return cls(
            model,
            label2idx=label2idx,
            ds_cfg=ds_cfg,
            device=device,
            family_map=family_map,
            metadata={
                "checkpoint_path": str(ckpt_path),
                "checkpoint_args": ckpt_args,
                "best_f1": float(payload.get("best_f1", 0.0)),
                "runtime_temporal_policy": "resample",
                "config_resolution_source": ("bundle" if manifest is not None else "checkpoint"),
                "label_map_source": label_source,
                "ds_config_source": ds_source,
                "family_metadata_source": family_source,
                "requires_pose": bool(ds_cfg.include_pose),
                "runtime_manifest": ({k: v for k, v in manifest.items() if not str(k).startswith("_")} if manifest is not None else {}),
            },
        )

    @classmethod
    def from_bundle(
        cls,
        bundle_dir_or_manifest: str | Path,
        *,
        device: str | torch.device | None = None,
        prefer_ema: bool = True,
    ) -> "MSAGCNClassifier":
        manifest = load_runtime_manifest(bundle_dir_or_manifest)
        ckpt = resolve_bundle_path(manifest, str(manifest.get("checkpoint", "")))
        label_map_path = resolve_bundle_path(manifest, str(manifest.get("label_map", ""))) if manifest.get("label_map") else None
        ds_config_path = resolve_bundle_path(manifest, str(manifest.get("ds_config", ""))) if manifest.get("ds_config") else None
        return cls.from_checkpoint(
            ckpt,
            device=device,
            prefer_ema=prefer_ema,
            label_map_path=label_map_path,
            ds_config_path=ds_config_path,
            manifest=manifest,
        )

    @torch.inference_mode()
    def predict_sequence(self, seq: CanonicalSkeletonSequence, *, topk: int = 5) -> MSAGCNPrediction:
        batch = self.builder.prepare_sequence(seq)
        x = {k: v.to(self.device) for k, v in batch.items() if k in ("joints", "bones", "velocity")}
        mask = batch["mask"].to(self.device)
        features = self.model.forward_features(x, mask=mask, A=self.A)
        logits = self.model.forward_logits(features)
        probs = torch.softmax(logits, dim=-1)[0].detach().float().cpu().numpy()
        logits_np = logits[0].detach().float().cpu().numpy()
        k = max(1, min(int(topk), probs.shape[0]))
        top_idx = np.argsort(-probs)[:k]
        top_rows = [
            {
                "rank": int(rank + 1),
                "class_id": int(class_id),
                "label": self.idx2label.get(int(class_id), str(class_id)),
                "prob": float(probs[class_id]),
            }
            for rank, class_id in enumerate(top_idx.tolist())
        ]
        best_id = int(top_idx[0])
        family_class_id: int | None = None
        family_label: str | None = None
        family_confidence: float | None = None
        family_topk: List[Dict[str, Any]] | None = None
        family_logits_np: np.ndarray | None = None
        family_probs_np: np.ndarray | None = None
        family_logits = self.model.forward_family_logits(features)
        if family_logits is not None:
            family_probs_np = torch.softmax(family_logits, dim=-1)[0].detach().float().cpu().numpy()
            family_logits_np = family_logits[0].detach().float().cpu().numpy()
            k_family = max(1, min(int(topk), family_probs_np.shape[0]))
            family_top_idx = np.argsort(-family_probs_np)[:k_family]
            family_topk = [
                {
                    "rank": int(rank + 1),
                    "class_id": int(class_id),
                    "label": self.family_idx2label.get(int(class_id), str(class_id)),
                    "prob": float(family_probs_np[class_id]),
                }
                for rank, class_id in enumerate(family_top_idx.tolist())
            ]
            family_class_id = int(family_top_idx[0])
            family_label = self.family_idx2label.get(family_class_id, str(family_class_id))
            family_confidence = float(family_probs_np[family_class_id])
        return MSAGCNPrediction(
            label=self.idx2label.get(best_id, str(best_id)),
            class_id=best_id,
            confidence=float(probs[best_id]),
            topk=top_rows,
            logits=logits_np.astype(np.float32, copy=False).tolist(),
            probs=probs.astype(np.float32, copy=False).tolist(),
            family_class_id=family_class_id,
            family_label=family_label,
            family_confidence=family_confidence,
            family_topk=family_topk,
            family_logits=(None if family_logits_np is None else family_logits_np.astype(np.float32, copy=False).tolist()),
            family_probs=(None if family_probs_np is None else family_probs_np.astype(np.float32, copy=False).tolist()),
            meta={
                "sequence_length": int(seq.length),
                "max_frames": int(self.ds_cfg.max_frames),
                "runtime_temporal_policy": "resample",
                "pose_enabled": bool(self.ds_cfg.include_pose),
                "preprocessing_version": MSAGCN_RUNTIME_PREPROCESSING_VERSION,
                "use_family_head": bool(getattr(self.model, "use_family_head", False)),
                "num_families": int(getattr(self.model, "num_families", 0) or 0),
                "family_metadata_source": str(self.metadata.get("family_metadata_source", "")),
            },
        )

    def export_runtime_bundle(
        self,
        out_dir: str | Path,
        *,
        checkpoint_source: str | Path,
        label_map_source: str | Path | None = None,
        ds_config_source: str | Path | None = None,
        family_map_source: str | Path | None = None,
    ) -> Path:
        root = Path(out_dir)
        ckpt_copy = copy_into_bundle(root, checkpoint_source, rel_path="checkpoints/msagcn.ckpt")
        rel_label = "metadata/label2idx.json"
        if label_map_source and Path(label_map_source).exists():
            copy_into_bundle(root, label_map_source, rel_path=rel_label)
        else:
            (root / rel_label).parent.mkdir(parents=True, exist_ok=True)
            (root / rel_label).write_text(json.dumps(self.label2idx, ensure_ascii=False, indent=2), encoding="utf-8")
        rel_ds = "metadata/ds_config.json"
        if ds_config_source and Path(ds_config_source).exists():
            copy_into_bundle(root, ds_config_source, rel_path=rel_ds)
        else:
            (root / rel_ds).parent.mkdir(parents=True, exist_ok=True)
            (root / rel_ds).write_text(json.dumps(asdict(self.ds_cfg), ensure_ascii=False, indent=2), encoding="utf-8")
        rel_family = ""
        if family_map_source and Path(family_map_source).exists():
            rel_family = "metadata/family_map.json"
            copy_into_bundle(root, family_map_source, rel_path=rel_family)
        manifest = {
            "model_type": "msagcn",
            "checkpoint": str(ckpt_copy.relative_to(root)),
            "label_map": rel_label,
            "ds_config": rel_ds,
            "family_map": rel_family,
            "skeleton_spec": asdict(CANONICAL_SKELETON_SPEC),
            "runtime_temporal_policy": "resample",
            "preprocessing_version": MSAGCN_RUNTIME_PREPROCESSING_VERSION,
            "use_streams": list(self.ds_cfg.use_streams),
            "pose_enabled": bool(self.ds_cfg.include_pose),
            "max_frames": int(self.ds_cfg.max_frames),
            "label_count": int(len(self.label2idx)),
            "num_families": int(getattr(self.model, "num_families", 0) or 0),
            "metadata": dict(self.metadata),
        }
        return write_runtime_manifest(root, manifest)


def export_msagcn_runtime_bundle(
    checkpoint_path: str | Path,
    out_dir: str | Path,
    *,
    label_map_path: str | Path | None = None,
    ds_config_path: str | Path | None = None,
    prefer_ema: bool = True,
) -> Path:
    ckpt = Path(checkpoint_path).expanduser().resolve()
    classifier = MSAGCNClassifier.from_checkpoint(
        ckpt,
        device="cpu",
        prefer_ema=prefer_ema,
        label_map_path=label_map_path,
        ds_config_path=ds_config_path,
    )
    if label_map_path is None:
        sibling = ckpt.parent / "label2idx.json"
        if sibling.exists():
            label_map_path = sibling
    if ds_config_path is None:
        sibling = ckpt.parent / "ds_config.json"
        if sibling.exists():
            ds_config_path = sibling
    family_map_path: str | Path | None = None
    for candidate in (ckpt.parent / "family_map.json", ckpt.parent.parent / "family_map.json"):
        if candidate.exists():
            family_map_path = candidate
            break
    return classifier.export_runtime_bundle(
        out_dir,
        checkpoint_source=ckpt,
        label_map_source=label_map_path,
        ds_config_source=ds_config_path,
        family_map_source=family_map_path,
    )
