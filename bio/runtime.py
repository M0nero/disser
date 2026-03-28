from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from bio.core.model import BioModelConfig, BioTagger
from bio.core.preprocessing import (
    BIO_PREPROCESSING_VERSION_V2,
    BIO_PREPROCESSING_VERSION_V3,
    BioPreprocessConfig,
    BioPreprocessState,
    init_bio_preprocess_state,
    preprocess_frame_v2,
    preprocess_frame_v3,
    preprocess_sequence,
)
from runtime.manifest import copy_into_bundle, load_runtime_manifest, resolve_bundle_path, write_runtime_manifest
from runtime.skeleton import CANONICAL_SKELETON_SPEC, CanonicalSkeletonSequence


DEFAULT_DECODER_VERSION = "bio_segment_decoder_v2"
BIO_RUNTIME_PREPROCESSING_VERSION = BIO_PREPROCESSING_VERSION_V3


@dataclass
class BioDecoderConfig:
    start_threshold: float = 0.80
    continue_threshold: float | None = None
    continue_threshold_policy: str = "fixed_ratio"
    continue_threshold_ratio: float = 0.60
    min_segment_frames: int = 3
    min_gap_frames: int = 2
    max_idle_inside_segment: int = 4
    cooldown_frames: int = 2
    emit_partial_segments: bool = False
    eos_policy: str = "drop_partial_on_eos"
    stream_window: int = 16
    require_hand_presence_to_start: bool = False
    min_visible_hand_frames_to_start: int = 2
    min_valid_hand_joints_to_start: int = 8
    allow_one_hand_to_start: bool = True
    use_signness_gate: bool = True
    signness_start_threshold: float = 0.55
    signness_continue_threshold: float = 0.50
    use_onset_gate: bool = True
    onset_start_threshold: float = 0.45
    active_start_threshold: float = 0.25
    active_continue_threshold: float = 0.20


@dataclass
class BioSegmentEvent:
    segment_id: int
    start_frame: int
    end_frame_exclusive: int
    start_time_ms: float
    end_time_ms: float
    boundary_score: float
    mean_inside_score: float
    threshold_used: float
    end_reason: str = "gap_closed"
    decoder_version: str = DEFAULT_DECODER_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BioSegmentDecoder:
    def __init__(self, cfg: Optional[BioDecoderConfig] = None) -> None:
        self.cfg = cfg or BioDecoderConfig()
        self.reset()

    def reset(self) -> None:
        self.frame_index = 0
        self._segment_id = 0
        self._cooldown_remaining = 0
        self._active: Optional[Dict[str, Any]] = None
        self._visible_hand_run = 0
        self._last_step_debug: Dict[str, Any] = {
            "hand_presence_ok": (not bool(self.cfg.require_hand_presence_to_start)),
            "start_blocked_by_hand_guard": False,
            "visible_hand_run": 0,
            "p_active": None,
            "p_onset": None,
            "signness_gate_ok": True,
            "clip_hit_candidate": False,
            "startup_false_start_candidate": False,
        }

    def _continue_threshold(self) -> float:
        if self.cfg.continue_threshold is not None:
            return float(self.cfg.continue_threshold)
        policy = str(self.cfg.continue_threshold_policy or "fixed_ratio").strip().lower()
        if policy == "fixed_ratio":
            return max(0.25, min(0.65, float(self.cfg.start_threshold) * float(self.cfg.continue_threshold_ratio)))
        raise ValueError(f"Unsupported BIO continue_threshold_policy={self.cfg.continue_threshold_policy!r}")

    def _open_segment(self, frame_idx: int, ts_ms: float, pb: float, inside_score: float) -> None:
        self._active = {
            "segment_id": self._segment_id,
            "start_frame": int(frame_idx),
            "start_time_ms": float(ts_ms),
            "boundary_score": float(pb),
            "inside_scores": [float(inside_score)],
            "last_active_frame": int(frame_idx),
            "last_active_time_ms": float(ts_ms),
            "idle_frames": 0,
        }
        self._segment_id += 1

    def _hand_presence_frame_ok(self, *, left_valid_joints: int, right_valid_joints: int, total_valid_hand_joints: int) -> bool:
        min_valid = max(1, int(self.cfg.min_valid_hand_joints_to_start))
        left_ok = int(left_valid_joints) >= min_valid
        right_ok = int(right_valid_joints) >= min_valid
        total_ok = int(total_valid_hand_joints) >= min_valid
        if bool(self.cfg.allow_one_hand_to_start):
            return bool(left_ok or right_ok or total_ok)
        return bool(left_ok and right_ok)

    def _hand_presence_ok(
        self,
        *,
        left_valid_joints: int,
        right_valid_joints: int,
        total_valid_hand_joints: int,
    ) -> bool:
        if not bool(self.cfg.require_hand_presence_to_start):
            self._visible_hand_run = 0
            return True
        if self._hand_presence_frame_ok(
            left_valid_joints=int(left_valid_joints),
            right_valid_joints=int(right_valid_joints),
            total_valid_hand_joints=int(total_valid_hand_joints),
        ):
            self._visible_hand_run += 1
        else:
            self._visible_hand_run = 0
        return bool(self._visible_hand_run >= max(1, int(self.cfg.min_visible_hand_frames_to_start)))

    def _finalize_active(self, *, end_reason: str, allow_short: bool = False) -> Optional[BioSegmentEvent]:
        if self._active is None:
            return None
        start = int(self._active["start_frame"])
        end_excl = int(self._active["last_active_frame"]) + 1
        length = max(0, end_excl - start)
        seg = self._active
        self._active = None
        self._cooldown_remaining = int(self.cfg.cooldown_frames)
        if (not allow_short) and length < int(self.cfg.min_segment_frames):
            return None
        scores = list(seg.get("inside_scores", []) or [0.0])
        return BioSegmentEvent(
            segment_id=int(seg["segment_id"]),
            start_frame=start,
            end_frame_exclusive=end_excl,
            start_time_ms=float(seg["start_time_ms"]),
            end_time_ms=float(seg["last_active_time_ms"]),
            boundary_score=float(seg["boundary_score"]),
            mean_inside_score=float(np.mean(scores) if scores else 0.0),
            threshold_used=float(self.cfg.start_threshold),
            end_reason=str(end_reason),
        )

    def step(
        self,
        probs: Sequence[float],
        *,
        ts_ms: float,
        active_prob: float | None = None,
        onset_prob: float | None = None,
        left_valid_joints: int = 0,
        right_valid_joints: int = 0,
        total_valid_hand_joints: int | None = None,
    ) -> List[BioSegmentEvent]:
        arr = np.asarray(probs, dtype=np.float32).reshape(-1)
        if arr.shape[0] != 3:
            raise ValueError(f"BIO decoder expects 3 probabilities, got shape {tuple(arr.shape)}")
        po, pb, pi = float(arr[0]), float(arr[1]), float(arr[2])
        inside_score = max(pb, pi)
        p_active = None if active_prob is None else float(active_prob)
        p_onset = None if onset_prob is None else float(onset_prob)
        signness_gate_enabled = bool(self.cfg.use_signness_gate) and p_active is not None
        signness_start_ok = (not signness_gate_enabled) or bool(p_active >= float(self.cfg.active_start_threshold))
        signness_continue_ok = bool(signness_gate_enabled and p_active >= float(self.cfg.active_continue_threshold))
        onset_gate_enabled = bool(self.cfg.use_onset_gate) and p_onset is not None
        onset_start_ok = bool(onset_gate_enabled and p_onset >= float(self.cfg.onset_start_threshold))
        active_like = bool((inside_score >= self._continue_threshold()) or signness_continue_ok)
        start_like = bool(((pb >= float(self.cfg.start_threshold)) or onset_start_ok) and signness_start_ok)
        events: List[BioSegmentEvent] = []
        frame_idx = int(self.frame_index)
        total_valid = int(
            total_valid_hand_joints
            if total_valid_hand_joints is not None
            else max(0, int(left_valid_joints)) + max(0, int(right_valid_joints))
        )
        hand_presence_ok = self._hand_presence_ok(
            left_valid_joints=int(left_valid_joints),
            right_valid_joints=int(right_valid_joints),
            total_valid_hand_joints=int(total_valid),
        )
        start_blocked = False
        startup_false_start_candidate = False

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        if self._active is None:
            raw_start_candidate = bool(self._cooldown_remaining <= 0 and pb >= float(self.cfg.start_threshold))
            if raw_start_candidate and not hand_presence_ok:
                startup_false_start_candidate = True
            if self._cooldown_remaining <= 0 and start_like:
                if hand_presence_ok:
                    self._open_segment(frame_idx, float(ts_ms), pb, inside_score)
                else:
                    start_blocked = True
        else:
            self._active["boundary_score"] = max(float(self._active["boundary_score"]), pb)
            self._active["inside_scores"].append(float(inside_score))
            if active_like:
                self._active["last_active_frame"] = frame_idx
                self._active["last_active_time_ms"] = float(ts_ms)
                self._active["idle_frames"] = 0
            else:
                self._active["idle_frames"] = int(self._active["idle_frames"]) + 1
            if int(self._active["idle_frames"]) >= int(min(self.cfg.min_gap_frames, self.cfg.max_idle_inside_segment)) or int(self._active["idle_frames"]) > int(self.cfg.max_idle_inside_segment):
                event = self._finalize_active(end_reason="gap_closed")
                if event is not None:
                    events.append(event)
                raw_start_candidate = bool(self._cooldown_remaining <= 0 and pb >= float(self.cfg.start_threshold))
                if raw_start_candidate and not hand_presence_ok:
                    startup_false_start_candidate = True
                if self._cooldown_remaining <= 0 and start_like:
                    if hand_presence_ok:
                        self._open_segment(frame_idx, float(ts_ms), pb, inside_score)
                    else:
                        start_blocked = True

        self._last_step_debug = {
            "hand_presence_ok": bool(hand_presence_ok),
            "start_blocked_by_hand_guard": bool(start_blocked),
            "visible_hand_run": int(self._visible_hand_run),
            "left_valid_joints": int(left_valid_joints),
            "right_valid_joints": int(right_valid_joints),
            "total_valid_hand_joints": int(total_valid),
            "p_active": (None if p_active is None else float(p_active)),
            "p_onset": (None if p_onset is None else float(p_onset)),
            "signness_gate_ok": bool(signness_start_ok),
            "clip_hit_candidate": bool(hand_presence_ok and signness_start_ok and ((pb >= float(self.cfg.start_threshold)) or onset_start_ok)),
            "startup_false_start_candidate": bool(startup_false_start_candidate),
        }
        self.frame_index += 1
        return events

    @property
    def last_step_debug(self) -> Dict[str, Any]:
        return dict(self._last_step_debug)

    def flush(self, *, force: bool = False, eos_policy: str | None = None) -> List[BioSegmentEvent]:
        if self._active is None:
            return []
        if force:
            event = self._finalize_active(end_reason="forced_partial", allow_short=True)
            return [event] if event is not None else []
        policy = str(eos_policy or self.cfg.eos_policy or "drop_partial_on_eos").strip().lower()
        if policy == "drop_partial_on_eos":
            self._active = None
            return []
        if policy == "close_open_segment_on_eos":
            event = self._finalize_active(end_reason="eos_closed", allow_short=bool(self.cfg.emit_partial_segments))
            return [event] if event is not None else []
        raise ValueError(f"Unsupported BIO eos_policy={policy!r}")
        return []


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _filter_cfg_kwargs(raw: Dict[str, Any]) -> Dict[str, Any]:
    keys = set(BioModelConfig.__dataclass_fields__.keys())
    return {k: raw[k] for k in raw if k in keys}


def _default_stream_window(cfg: BioModelConfig) -> int:
    return max(8, int(cfg.conv_layers) * max(1, int(cfg.conv_kernel)) + 4)


def _build_preprocess_config(
    raw: Dict[str, Any] | None = None,
    *,
    fallback_version: str = BIO_PREPROCESSING_VERSION_V2,
) -> BioPreprocessConfig:
    payload = dict(raw or {})
    nested = payload.get("preprocessing_config")
    if isinstance(nested, dict):
        payload = {**payload, **nested}
    version = str(payload.get("preprocessing_version") or payload.get("version") or fallback_version)
    flat = {
        "version": version,
        "center_alpha": float(payload.get("preprocessing_center_alpha", payload.get("center_alpha", 0.2))),
        "scale_alpha": float(payload.get("preprocessing_scale_alpha", payload.get("scale_alpha", 0.1))),
        "min_scale": float(payload.get("preprocessing_min_scale", payload.get("min_scale", 1e-3))),
        "min_visible_joints_for_scale": int(
            payload.get("preprocessing_min_visible_joints_for_scale", payload.get("min_visible_joints_for_scale", 4))
        ),
    }
    return BioPreprocessConfig(**flat)


def _resolve_preprocess_config(
    *sources: Dict[str, Any] | None,
    fallback_version: str = BIO_PREPROCESSING_VERSION_V2,
) -> BioPreprocessConfig:
    for source in sources:
        if not isinstance(source, dict):
            continue
        if source.get("preprocessing_version") or source.get("preprocessing_config") or source.get("version"):
            return _build_preprocess_config(source, fallback_version=fallback_version)
    return BioPreprocessConfig(version=str(fallback_version))


def _select_bio_checkpoint(dir_or_file: str | Path, selection: str) -> Path:
    path = Path(dir_or_file).expanduser()
    if path.is_file():
        return path.resolve()
    if (path / "runtime_manifest.json").exists():
        manifest = load_runtime_manifest(path)
        return resolve_bundle_path(manifest, str(manifest.get("checkpoint", "")))
    role = str(selection or "best_recall_safe").strip().lower()
    candidates: Dict[str, list[str]] = {
        "best_balanced": ["best_balanced_model.pt", "best_balanced.pt", "best_model.pt", "best.pt", "last_model.pt", "last.pt"],
        "best_boundary": ["best_boundary_model.pt", "best_boundary.pt", "best_model.pt", "best.pt", "last_model.pt", "last.pt"],
        "best_recall_safe": ["best_recall_safe_model.pt", "best_recall_safe.pt", "best_balanced_model.pt", "best_balanced.pt", "best_model.pt", "best.pt", "last_model.pt", "last.pt"],
        "last": ["last_model.pt", "last.pt", "best_balanced_model.pt", "best_balanced.pt"],
    }
    for name in candidates.get(role, candidates["best_recall_safe"]):
        candidate = path / name
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"No BIO checkpoint found in {path} for selection={role}")


def _selection_metrics_from_payload(payload: Dict[str, Any], selection: str) -> Dict[str, Any]:
    role = str(selection or "best_recall_safe").strip().lower()
    if role == "best_boundary":
        metrics = dict(payload.get("best_boundary_metrics", {}) or {})
    elif role == "best_recall_safe":
        metrics = (
            dict(payload.get("best_recall_safe_metrics", {}) or {})
            or dict(payload.get("best_balanced_metrics", {}) or {})
            or dict(payload.get("last_metrics", {}) or {})
        )
    elif role == "last":
        metrics = dict(payload.get("last_metrics", {}) or {})
    else:
        metrics = (
            dict(payload.get("best_recall_safe_metrics", {}) or {})
            or dict(payload.get("best_balanced_metrics", {}) or {})
            or dict(payload.get("last_metrics", {}) or {})
        )
    return metrics


class BioSegmenter:
    def __init__(
        self,
        model: BioTagger,
        *,
        device: str | torch.device | None = None,
        decoder_cfg: Optional[BioDecoderConfig] = None,
        threshold: float = 0.80,
        checkpoint_path: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        preprocess_cfg: Optional[BioPreprocessConfig] = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.model = model.to(self.device).eval()
        self.dtype = next(self.model.parameters()).dtype
        self.preprocess_cfg = preprocess_cfg or BioPreprocessConfig(version=BIO_RUNTIME_PREPROCESSING_VERSION)
        cfg = decoder_cfg or BioDecoderConfig()
        self.decoder = BioSegmentDecoder(
            BioDecoderConfig(
                start_threshold=float(threshold),
                continue_threshold=(None if cfg.continue_threshold is None else float(cfg.continue_threshold)),
                continue_threshold_policy=str(cfg.continue_threshold_policy),
                continue_threshold_ratio=float(cfg.continue_threshold_ratio),
                min_segment_frames=int(cfg.min_segment_frames),
                min_gap_frames=int(cfg.min_gap_frames),
                max_idle_inside_segment=int(cfg.max_idle_inside_segment),
                cooldown_frames=int(cfg.cooldown_frames),
                emit_partial_segments=bool(cfg.emit_partial_segments),
                eos_policy=str(cfg.eos_policy),
                stream_window=int(cfg.stream_window),
                require_hand_presence_to_start=bool(cfg.require_hand_presence_to_start),
                min_visible_hand_frames_to_start=int(cfg.min_visible_hand_frames_to_start),
                min_valid_hand_joints_to_start=int(cfg.min_valid_hand_joints_to_start),
                allow_one_hand_to_start=bool(cfg.allow_one_hand_to_start),
                use_signness_gate=bool(cfg.use_signness_gate),
                signness_start_threshold=float(cfg.signness_start_threshold),
                signness_continue_threshold=float(cfg.signness_continue_threshold),
                use_onset_gate=bool(cfg.use_onset_gate),
                onset_start_threshold=float(cfg.onset_start_threshold),
                active_start_threshold=float(cfg.active_start_threshold),
                active_continue_threshold=float(cfg.active_continue_threshold),
            )
        )
        self.threshold = float(threshold)
        self.checkpoint_path = str(checkpoint_path)
        self.metadata = {
            "threshold": float(self.threshold),
            "decoder_version": DEFAULT_DECODER_VERSION,
            "preprocessing_version": str(self.preprocess_cfg.version),
            "preprocessing_config": asdict(self.preprocess_cfg),
            **dict(metadata or {}),
        }
        self._stream_state = self.model.init_stream_state(
            batch_size=1,
            window=max(1, int(self.decoder.cfg.stream_window)),
            device=self.device,
            dtype=self.dtype,
        )
        self._preprocess_state = init_bio_preprocess_state(device=self.device, dtype=self.dtype)

    @classmethod
    def from_model(
        cls,
        model: BioTagger,
        *,
        threshold: float = 0.80,
        decoder_cfg: Optional[BioDecoderConfig] = None,
        device: str | torch.device | None = None,
        metadata: Optional[Dict[str, Any]] = None,
        preprocess_cfg: Optional[BioPreprocessConfig] = None,
    ) -> "BioSegmenter":
        cfg = decoder_cfg or BioDecoderConfig(stream_window=_default_stream_window(model.cfg))
        return cls(
            model,
            device=device,
            decoder_cfg=cfg,
            threshold=threshold,
            metadata=metadata,
            preprocess_cfg=preprocess_cfg,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_or_dir: str | Path,
        *,
        selection: str = "best_recall_safe",
        device: str | torch.device | None = None,
        prefer_ema: bool = True,
        decoder_cfg: Optional[BioDecoderConfig] = None,
        threshold: float | None = None,
        metadata_override: Optional[Dict[str, Any]] = None,
    ) -> "BioSegmenter":
        ckpt_path = _select_bio_checkpoint(checkpoint_or_dir, selection)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw_cfg = dict(payload.get("cfg", {}) or {})
        model_cfg = BioModelConfig(**_filter_cfg_kwargs(raw_cfg))
        model = BioTagger(model_cfg)
        state_dict = payload.get("ema_state") if prefer_ema and payload.get("ema_state") is not None else payload.get("model_state")
        if state_dict is None:
            raise RuntimeError(f"BIO checkpoint has no model state: {ckpt_path}")
        model.load_state_dict(state_dict, strict=True)
        metrics = _selection_metrics_from_payload(payload, selection)
        resolved_threshold = float(threshold if threshold is not None else metrics.get("selection_threshold", metrics.get("threshold", 0.5)))
        cfg = decoder_cfg or BioDecoderConfig(start_threshold=resolved_threshold, stream_window=_default_stream_window(model_cfg))
        preprocess_cfg = _resolve_preprocess_config(
            dict(metadata_override or {}),
            dict(payload.get("runtime_summary", {}) or {}),
            dict(payload.get("args", {}) or {}),
            fallback_version=BIO_PREPROCESSING_VERSION_V2,
        )
        return cls(
            model,
            device=device,
            decoder_cfg=cfg,
            threshold=resolved_threshold,
            checkpoint_path=str(ckpt_path),
            preprocess_cfg=preprocess_cfg,
            metadata={
                "config_resolution_source": "checkpoint",
                "selection": str(selection),
                "selected_metrics": metrics,
                "checkpoint_args": dict(payload.get("args", {}) or {}),
                "runtime_summary": dict(payload.get("runtime_summary", {}) or {}),
                **dict(metadata_override or {}),
            },
        )

    @classmethod
    def from_bundle(
        cls,
        bundle_dir_or_manifest: str | Path,
        *,
        device: str | torch.device | None = None,
        prefer_ema: bool = True,
    ) -> "BioSegmenter":
        manifest = load_runtime_manifest(bundle_dir_or_manifest)
        if str(manifest.get("model_type", "")) != "bio":
            raise ValueError(f"BIO runtime bundle expected model_type='bio', got {manifest.get('model_type')!r}")
        ckpt_path = resolve_bundle_path(manifest, str(manifest.get("checkpoint", "")))
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw_cfg = dict(payload.get("cfg", {}) or {})
        filtered_raw_cfg = _filter_cfg_kwargs(raw_cfg)
        manifest_cfg = _filter_cfg_kwargs(dict(manifest.get("model_config", {}) or {}))
        if manifest_cfg and manifest_cfg != filtered_raw_cfg:
            raise RuntimeError("BIO runtime bundle manifest model_config does not match checkpoint cfg")
        model_cfg = BioModelConfig(**filtered_raw_cfg)
        model = BioTagger(model_cfg)
        state_dict = payload.get("ema_state") if prefer_ema and payload.get("ema_state") is not None else payload.get("model_state")
        if state_dict is None:
            raise RuntimeError(f"BIO checkpoint has no model state: {ckpt_path}")
        model.load_state_dict(state_dict, strict=True)
        decoder_payload = dict(manifest.get("decoder_config", {}) or manifest.get("decoder_defaults", {}) or {})
        decoder_cfg = BioDecoderConfig(**{k: v for k, v in decoder_payload.items() if k in BioDecoderConfig.__dataclass_fields__})
        resolved_threshold = float(manifest.get("threshold", decoder_cfg.start_threshold))
        preprocess_cfg = _resolve_preprocess_config(
            dict(manifest.get("metadata", {}) or {}),
            dict(manifest.get("preprocessing_config", {}) or {}),
            dict(payload.get("runtime_summary", {}) or {}),
            dict(payload.get("args", {}) or {}),
            {"preprocessing_version": str(manifest.get("preprocessing_version", ""))},
            fallback_version=BIO_PREPROCESSING_VERSION_V2,
        )
        return cls(
            model,
            device=device,
            decoder_cfg=decoder_cfg,
            threshold=resolved_threshold,
            checkpoint_path=str(ckpt_path),
            preprocess_cfg=preprocess_cfg,
            metadata={
                "config_resolution_source": "bundle",
                "selection": str(manifest.get("checkpoint_role", "best_recall_safe") or "best_recall_safe"),
                "runtime_manifest": {k: v for k, v in manifest.items() if not str(k).startswith("_")},
                "checkpoint_args": dict(payload.get("args", {}) or {}),
                "selected_metrics": _selection_metrics_from_payload(payload, str(manifest.get("checkpoint_role", "best_recall_safe") or "best_recall_safe")),
                "runtime_summary": dict(payload.get("runtime_summary", {}) or {}),
            },
        )

    def reset(self) -> None:
        self.decoder.reset()
        self._stream_state = self.model.init_stream_state(
            batch_size=1,
            window=max(1, int(self.decoder.cfg.stream_window)),
            device=self.device,
            dtype=self.dtype,
        )
        self._preprocess_state = init_bio_preprocess_state(device=self.device, dtype=self.dtype)

    def _to_tensor_frame(self, arr: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            tensor = arr.detach().to(device=self.device, dtype=self.dtype)
        else:
            tensor = torch.as_tensor(arr, dtype=self.dtype, device=self.device)
        return tensor

    def _preprocess_frame(self, pts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if str(self.preprocess_cfg.version) == BIO_PREPROCESSING_VERSION_V2:
            return preprocess_frame_v2(pts, mask, dtype=self.dtype, device=self.device)
        frame_out, self._preprocess_state, _debug = preprocess_frame_v3(
            pts,
            mask,
            self._preprocess_state,
            cfg=self.preprocess_cfg,
            dtype=self.dtype,
            device=self.device,
        )
        return frame_out

    def _preprocess_sequence(self, seq: CanonicalSkeletonSequence) -> torch.Tensor:
        pts, _debug = preprocess_sequence(
            seq.pts,
            seq.mask,
            cfg=self.preprocess_cfg,
            device=self.device,
            dtype=self.dtype,
        )
        if isinstance(pts, torch.Tensor):
            return pts
        return torch.as_tensor(pts, device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def step(
        self,
        pts: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor,
        *,
        ts_ms: float,
    ) -> Dict[str, Any]:
        pt_t = self._to_tensor_frame(pts)
        mask_t = self._to_tensor_frame(mask)
        mask_frame = mask_t[0] if mask_t.dim() == 3 else mask_t
        mask_flat = mask_frame[..., 0] if mask_frame.dim() == 2 else mask_frame.reshape(-1)
        left_valid_joints = int((mask_flat[:21] > 0.5).sum().item()) if mask_flat.numel() >= 21 else 0
        right_valid_joints = int((mask_flat[21:] > 0.5).sum().item()) if mask_flat.numel() >= 42 else 0
        total_valid_hand_joints = int(left_valid_joints + right_valid_joints)
        pt_proc = self._preprocess_frame(pt_t, mask_t)
        if bool(getattr(self.model.cfg, "use_onset_head", False)):
            logits_t, signness_logits_t, onset_logits_t, self._stream_state = self.model.stream_step_with_heads(pt_proc, mask_t, self._stream_state)
        elif bool(getattr(self.model.cfg, "use_signness_head", False)):
            logits_t, signness_logits_t, self._stream_state = self.model.stream_step_with_aux(pt_proc, mask_t, self._stream_state)
            onset_logits_t = None
        else:
            logits_t, self._stream_state = self.model.stream_step(pt_proc, mask_t, self._stream_state)
            signness_logits_t = None
            onset_logits_t = None
        logits = logits_t[0].detach().float().cpu().numpy()
        probs = torch.softmax(logits_t[0].detach().float(), dim=-1).cpu().numpy()
        p_active = (
            float(torch.sigmoid(signness_logits_t[0].detach().float()).cpu().item())
            if signness_logits_t is not None
            else None
        )
        p_onset = (
            float(torch.sigmoid(onset_logits_t[0].detach().float()).cpu().item())
            if onset_logits_t is not None
            else None
        )
        events = [
            ev.to_dict()
            for ev in self.decoder.step(
                probs,
                ts_ms=float(ts_ms),
                active_prob=p_active,
                onset_prob=p_onset,
                left_valid_joints=int(left_valid_joints),
                right_valid_joints=int(right_valid_joints),
                total_valid_hand_joints=int(total_valid_hand_joints),
            )
        ]
        debug = self.decoder.last_step_debug
        return {
            "frame_index": int(self.decoder.frame_index - 1),
            "ts_ms": float(ts_ms),
            "logits": logits.astype(np.float32, copy=False).tolist(),
            "probs": probs.astype(np.float32, copy=False).tolist(),
            "label": int(int(np.argmax(probs))),
            "threshold": float(self.threshold),
            "segments": events,
            "p_active": p_active,
            "p_onset": p_onset,
            "left_valid_joints": int(left_valid_joints),
            "right_valid_joints": int(right_valid_joints),
            "total_valid_hand_joints": int(total_valid_hand_joints),
            "hand_presence_ok": bool(debug.get("hand_presence_ok", True)),
            "start_blocked_by_hand_guard": bool(debug.get("start_blocked_by_hand_guard", False)),
            "signness_gate_ok": bool(debug.get("signness_gate_ok", True)),
            "clip_hit_candidate": bool(debug.get("clip_hit_candidate", False)),
            "startup_false_start_candidate": bool(debug.get("startup_false_start_candidate", False)),
        }

    @torch.inference_mode()
    def infer_sequence(
        self,
        seq: CanonicalSkeletonSequence,
        *,
        return_frame_outputs: bool = False,
        flush_final: bool = True,
        force_flush: bool = False,
        eos_policy: str | None = None,
    ) -> Dict[str, Any]:
        self.reset()
        frame_outputs: List[Dict[str, Any]] = []
        segments: List[Dict[str, Any]] = []
        for idx in range(seq.length):
            out = self.step(seq.pts[idx], seq.mask[idx], ts_ms=float(seq.ts_ms[idx]))
            if return_frame_outputs:
                frame_outputs.append(out)
            if out["segments"]:
                segments.extend(out["segments"])
        if flush_final:
            segments.extend(ev.to_dict() for ev in self.decoder.flush(force=bool(force_flush), eos_policy=eos_policy))
        result = {
            "threshold": float(self.threshold),
            "decoder_config": asdict(self.decoder.cfg),
            "decoder_version": DEFAULT_DECODER_VERSION,
            "preprocessing_version": str(self.preprocess_cfg.version),
            "preprocessing_config": asdict(self.preprocess_cfg),
            "sequence_meta": dict(seq.meta),
            "segments": segments,
        }
        if return_frame_outputs:
            result["frame_outputs"] = frame_outputs
        return result

    @torch.inference_mode()
    def forward_logits(self, seq: CanonicalSkeletonSequence) -> np.ndarray:
        pts = self._preprocess_sequence(seq).unsqueeze(0)
        mask = torch.as_tensor(seq.mask[None], device=self.device, dtype=self.dtype)
        logits, _ = self.model(pts, mask)
        return logits[0].detach().float().cpu().numpy()

    @torch.inference_mode()
    def stream_logits(self, seq: CanonicalSkeletonSequence) -> np.ndarray:
        self.reset()
        rows: List[np.ndarray] = []
        for idx in range(seq.length):
            out = self.step(seq.pts[idx], seq.mask[idx], ts_ms=float(seq.ts_ms[idx]))
            rows.append(np.asarray(out["logits"], dtype=np.float32))
        if not rows:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack(rows, axis=0)

    def export_runtime_bundle(
        self,
        out_dir: str | Path,
        *,
        checkpoint_source: Optional[str | Path] = None,
        checkpoint_role: str = "best_recall_safe",
    ) -> Path:
        src = Path(checkpoint_source or self.checkpoint_path or "").expanduser()
        if not src.exists():
            raise FileNotFoundError(f"BIO checkpoint source not found for runtime bundle export: {src}")
        root = Path(out_dir)
        copied = copy_into_bundle(root, src, rel_path="checkpoints/bio_model.pt")
        manifest = {
            "model_type": "bio",
            "checkpoint": str(copied.relative_to(root)),
            "checkpoint_role": str(checkpoint_role),
            "skeleton_spec": asdict(CANONICAL_SKELETON_SPEC),
            "model_config": asdict(self.model.cfg),
            "threshold": float(self.threshold),
            "decoder_config": asdict(self.decoder.cfg),
            "decoder_version": DEFAULT_DECODER_VERSION,
            "preprocessing_version": str(self.preprocess_cfg.version),
            "preprocessing_config": asdict(self.preprocess_cfg),
            "metadata": dict(self.metadata),
        }
        return write_runtime_manifest(root, manifest)


def export_bio_runtime_bundle(
    checkpoint_or_dir: str | Path,
    out_dir: str | Path,
    *,
    selection: str = "best_recall_safe",
    prefer_ema: bool = True,
    decoder_cfg: Optional[BioDecoderConfig] = None,
) -> Path:
    segmenter = BioSegmenter.from_checkpoint(
        checkpoint_or_dir,
        selection=selection,
        device="cpu",
        prefer_ema=prefer_ema,
        decoder_cfg=decoder_cfg,
    )
    ckpt_path = _select_bio_checkpoint(checkpoint_or_dir, selection)
    return segmenter.export_runtime_bundle(out_dir, checkpoint_source=ckpt_path, checkpoint_role=selection)


def save_bio_inference_result(path: str | Path, payload: Dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
