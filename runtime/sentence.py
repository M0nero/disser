from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Set


@dataclass
class SentencePrediction:
    segment_id: int
    label: str
    confidence: float
    start_time_ms: float
    end_time_ms: float
    accepted: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SentenceBuilderConfig:
    min_confidence: float = 0.5
    duplicate_gap_ms: float = 1200.0
    blank_labels: tuple[str, ...] = ("", "__background__", "__blank__", "no_event")


class SentenceBuilder:
    def __init__(self, cfg: SentenceBuilderConfig | None = None) -> None:
        self.cfg = cfg or SentenceBuilderConfig()
        self.words: List[Dict[str, Any]] = []
        self.rejected_predictions: List[Dict[str, Any]] = []
        self._blank_labels: Set[str] = {str(x).strip().lower() for x in self.cfg.blank_labels}

    def add_prediction(
        self,
        *,
        segment_id: int,
        label: str,
        confidence: float,
        start_time_ms: float,
        end_time_ms: float,
        meta: Dict[str, Any] | None = None,
    ) -> SentencePrediction:
        label_norm = str(label).strip()
        conf = float(confidence)
        meta = dict(meta or {})
        if not label_norm or label_norm.lower() in self._blank_labels:
            item = SentencePrediction(segment_id, label_norm, conf, float(start_time_ms), float(end_time_ms), False, "blank_label")
            self.rejected_predictions.append(asdict(item) | {"meta": meta})
            return item
        if conf < float(self.cfg.min_confidence):
            item = SentencePrediction(segment_id, label_norm, conf, float(start_time_ms), float(end_time_ms), False, "low_confidence")
            self.rejected_predictions.append(asdict(item) | {"meta": meta})
            return item
        if self.words:
            prev = self.words[-1]
            gap_ms = float(start_time_ms) - float(prev.get("end_time_ms", 0.0))
            if str(prev.get("label", "")) == label_norm and gap_ms <= float(self.cfg.duplicate_gap_ms):
                item = SentencePrediction(segment_id, label_norm, conf, float(start_time_ms), float(end_time_ms), False, "duplicate_suppressed")
                self.rejected_predictions.append(asdict(item) | {"meta": meta})
                return item
        item = SentencePrediction(segment_id, label_norm, conf, float(start_time_ms), float(end_time_ms), True, "accepted")
        self.words.append(asdict(item) | {"meta": meta})
        return item

    @property
    def sentence(self) -> str:
        return " ".join(str(x.get("label", "")) for x in self.words if str(x.get("label", "")).strip())

    @property
    def committed_until_ms(self) -> float:
        if not self.words:
            return 0.0
        return float(self.words[-1].get("end_time_ms", 0.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "words": list(self.words),
            "sentence": self.sentence,
            "committed_until_ms": float(self.committed_until_ms),
            "rejected_predictions": list(self.rejected_predictions),
        }
