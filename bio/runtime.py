from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from bio.core.model import BioModelConfig, BioTagger
from runtime.manifest import copy_into_bundle, load_runtime_manifest, resolve_bundle_path, write_runtime_manifest
from runtime.skeleton import CANONICAL_SKELETON_SPEC, CanonicalSkeletonSequence


DEFAULT_DECODER_VERSION = "bio_segment_decoder_v1"
BIO_RUNTIME_PREPROCESSING_VERSION = "canonical_hands42_v2"


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

    def step(self, probs: Sequence[float], *, ts_ms: float) -> List[BioSegmentEvent]:
        arr = np.asarray(probs, dtype=np.float32).reshape(-1)
        if arr.shape[0] != 3:
            raise ValueError(f"BIO decoder expects 3 probabilities, got shape {tuple(arr.shape)}")
        po, pb, pi = float(arr[0]), float(arr[1]), float(arr[2])
        inside_score = max(pb, pi)
        active_like = inside_score >= self._continue_threshold()
        start_like = pb >= float(self.cfg.start_threshold)
        events: List[BioSegmentEvent] = []
        frame_idx = int(self.frame_index)

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        if self._active is None:
            if self._cooldown_remaining <= 0 and start_like:
                self._open_segment(frame_idx, float(ts_ms), pb, inside_score)
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
                if self._cooldown_remaining <= 0 and start_like:
                    self._open_segment(frame_idx, float(ts_ms), pb, inside_score)

        self.frame_index += 1
        return events

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


def _select_bio_checkpoint(dir_or_file: str | Path, selection: str) -> Path:
    path = Path(dir_or_file).expanduser()
    if path.is_file():
        return path.resolve()
    if (path / "runtime_manifest.json").exists():
        manifest = load_runtime_manifest(path)
        return resolve_bundle_path(manifest, str(manifest.get("checkpoint", "")))
    role = str(selection or "best_balanced").strip().lower()
    candidates: Dict[str, list[str]] = {
        "best_balanced": ["best_balanced_model.pt", "best_balanced.pt", "best_model.pt", "best.pt", "last_model.pt", "last.pt"],
        "best_boundary": ["best_boundary_model.pt", "best_boundary.pt", "best_model.pt", "best.pt", "last_model.pt", "last.pt"],
        "last": ["last_model.pt", "last.pt", "best_balanced_model.pt", "best_balanced.pt"],
    }
    for name in candidates.get(role, candidates["best_balanced"]):
        candidate = path / name
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"No BIO checkpoint found in {path} for selection={role}")


def _selection_metrics_from_payload(payload: Dict[str, Any], selection: str) -> Dict[str, Any]:
    role = str(selection or "best_balanced").strip().lower()
    if role == "best_boundary":
        metrics = dict(payload.get("best_boundary_metrics", {}) or {})
    elif role == "last":
        metrics = dict(payload.get("last_metrics", {}) or {})
    else:
        metrics = dict(payload.get("best_balanced_metrics", {}) or {}) or dict(payload.get("last_metrics", {}) or {})
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
    ) -> None:
        self.device = _resolve_device(device)
        self.model = model.to(self.device).eval()
        self.dtype = next(self.model.parameters()).dtype
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
            )
        )
        self.threshold = float(threshold)
        self.checkpoint_path = str(checkpoint_path)
        self.metadata = {
            "threshold": float(self.threshold),
            "decoder_version": DEFAULT_DECODER_VERSION,
            "preprocessing_version": BIO_RUNTIME_PREPROCESSING_VERSION,
            **dict(metadata or {}),
        }
        self._stream_state = self.model.init_stream_state(
            batch_size=1,
            window=max(1, int(self.decoder.cfg.stream_window)),
            device=self.device,
            dtype=self.dtype,
        )

    @classmethod
    def from_model(
        cls,
        model: BioTagger,
        *,
        threshold: float = 0.80,
        decoder_cfg: Optional[BioDecoderConfig] = None,
        device: str | torch.device | None = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BioSegmenter":
        cfg = decoder_cfg or BioDecoderConfig(stream_window=_default_stream_window(model.cfg))
        return cls(model, device=device, decoder_cfg=cfg, threshold=threshold, metadata=metadata)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_or_dir: str | Path,
        *,
        selection: str = "best_balanced",
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
        return cls(
            model,
            device=device,
            decoder_cfg=cfg,
            threshold=resolved_threshold,
            checkpoint_path=str(ckpt_path),
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
        return cls(
            model,
            device=device,
            decoder_cfg=decoder_cfg,
            threshold=resolved_threshold,
            checkpoint_path=str(ckpt_path),
            metadata={
                "config_resolution_source": "bundle",
                "selection": str(manifest.get("checkpoint_role", "best_balanced") or "best_balanced"),
                "runtime_manifest": {k: v for k, v in manifest.items() if not str(k).startswith("_")},
                "checkpoint_args": dict(payload.get("args", {}) or {}),
                "selected_metrics": _selection_metrics_from_payload(payload, str(manifest.get("checkpoint_role", "best_balanced") or "best_balanced")),
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

    def _to_tensor_frame(self, arr: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            tensor = arr.detach().to(device=self.device, dtype=self.dtype)
        else:
            tensor = torch.as_tensor(arr, dtype=self.dtype, device=self.device)
        return tensor

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
        logits_t, self._stream_state = self.model.stream_step(pt_t, mask_t, self._stream_state)
        logits = logits_t[0].detach().float().cpu().numpy()
        probs = torch.softmax(logits_t[0].detach().float(), dim=-1).cpu().numpy()
        events = [ev.to_dict() for ev in self.decoder.step(probs, ts_ms=float(ts_ms))]
        return {
            "frame_index": int(self.decoder.frame_index - 1),
            "ts_ms": float(ts_ms),
            "logits": logits.astype(np.float32, copy=False).tolist(),
            "probs": probs.astype(np.float32, copy=False).tolist(),
            "label": int(int(np.argmax(probs))),
            "threshold": float(self.threshold),
            "segments": events,
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
            "preprocessing_version": BIO_RUNTIME_PREPROCESSING_VERSION,
            "sequence_meta": dict(seq.meta),
            "segments": segments,
        }
        if return_frame_outputs:
            result["frame_outputs"] = frame_outputs
        return result

    @torch.inference_mode()
    def forward_logits(self, seq: CanonicalSkeletonSequence) -> np.ndarray:
        pts = torch.as_tensor(seq.pts[None], device=self.device, dtype=self.dtype)
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
        checkpoint_role: str = "best_balanced",
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
            "preprocessing_version": BIO_RUNTIME_PREPROCESSING_VERSION,
            "metadata": dict(self.metadata),
        }
        return write_runtime_manifest(root, manifest)


def export_bio_runtime_bundle(
    checkpoint_or_dir: str | Path,
    out_dir: str | Path,
    *,
    selection: str = "best_balanced",
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
