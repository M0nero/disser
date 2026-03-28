from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from PySide6.QtCore import QProcess, QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from desktop_review.session import ReviewSession, load_review_session
from pipeline.app import _draw_hand21_overlay, _draw_pose_overlay


REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_default(path_str: str) -> str:
    path = (REPO_ROOT / path_str).resolve()
    return str(path)


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value)


def _bgr_to_pixmap(frame: np.ndarray | None) -> QPixmap:
    if frame is None or frame.size == 0:
        return QPixmap()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    image = QImage(rgb.data, w, h, int(rgb.strides[0]), QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.deleteLater()
        elif child_layout is not None:
            _clear_layout(child_layout)


def _format_ms(ts_ms: float) -> str:
    value = float(ts_ms or 0.0)
    if abs(value) >= 1000.0:
        return f"{value / 1000.0:.2f}s"
    return f"{value:.0f}ms"


def _format_ratio(value: float) -> str:
    return f"{100.0 * float(value or 0.0):.1f}%"


def _format_xyz(vec: np.ndarray | List[float]) -> str:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return "—"
    return f"({arr[0]:.3f}, {arr[1]:.3f}, {arr[2]:.3f})"


def _warning_group(item: Dict[str, Any]) -> str:
    warning_id = str(item.get("id", "")).strip()
    if warning_id.startswith(("left_hand", "right_hand", "pose_")):
        return "tracking"
    if "segment" in warning_id:
        return "segmentation"
    if "prediction" in warning_id or "classifier" in warning_id:
        return "classification"
    if warning_id.startswith("all_predictions"):
        return "sentence builder"
    return "runtime"


def _chip_style(*, accent: str, fill: str = "#1f2127", text: str = "#f2f3f5") -> str:
    return (
        "QFrame {"
        f"border: 1px solid {accent};"
        f"background: {fill};"
        "border-radius: 10px;"
        "}"
        "QLabel.caption {"
        "font-size: 10px;"
        "letter-spacing: 0.5px;"
        "color: #9fa5b5;"
        "text-transform: uppercase;"
        "}"
        "QLabel.value {"
        "font-size: 16px;"
        "font-weight: 700;"
        f"color: {text};"
        "}"
    )


def _warning_chip_style(severity: str) -> str:
    accent = "#f5c451" if severity == "warning" else "#6bb7ff"
    fill = "#2b2212" if severity == "warning" else "#182433"
    return (
        "QLabel {"
        f"background: {fill};"
        f"border: 1px solid {accent};"
        "border-radius: 10px;"
        "padding: 4px 10px;"
        "font-size: 11px;"
        "font-weight: 600;"
        "}"
    )


def _toggle_button_style() -> str:
    return (
        "QPushButton {"
        "padding: 4px 10px;"
        "border: 1px solid #505463;"
        "border-radius: 10px;"
        "background: #23252c;"
        "color: #d8dbe4;"
        "}"
        "QPushButton:checked {"
        "background: #314664;"
        "border-color: #6ea8ff;"
        "color: #f5f7fb;"
        "}"
    )


def _draw_badge_block(
    frame: np.ndarray,
    lines: List[str],
    *,
    anchor: str,
    fill_bgr: tuple[int, int, int] = (24, 24, 28),
    text_bgr: tuple[int, int, int] = (245, 245, 245),
) -> np.ndarray:
    if frame is None or frame.size == 0 or not lines:
        return frame
    canvas = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1
    pad_x = 10
    pad_y = 8
    line_h = 20
    sizes = [cv2.getTextSize(str(line), font, font_scale, thickness)[0] for line in lines]
    width = max((size[0] for size in sizes), default=0) + 2 * pad_x
    height = line_h * len(lines) + 2 * pad_y - 4
    frame_h, frame_w = canvas.shape[:2]
    margin = 12
    if anchor == "top_left":
        x0, y0 = margin, margin
    elif anchor == "top_right":
        x0, y0 = max(margin, frame_w - width - margin), margin
    elif anchor == "bottom_left":
        x0, y0 = margin, max(margin, frame_h - height - margin)
    else:
        x0, y0 = max(margin, frame_w - width - margin), max(margin, frame_h - height - margin)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), fill_bgr, -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0.0, canvas)
    y = y0 + pad_y + 12
    for line in lines:
        cv2.putText(canvas, str(line), (x0 + pad_x, y), font, font_scale, text_bgr, thickness, cv2.LINE_AA)
        y += line_h
    return canvas


class VideoFrameSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        self.frame_count = max(0, int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        self._cache_idx = -1
        self._cache_frame: np.ndarray | None = None

    @property
    def is_open(self) -> bool:
        return bool(self._cap is not None and self._cap.isOpened())

    def read_frame(self, frame_index: int) -> np.ndarray | None:
        if not self.is_open:
            return None
        idx = max(0, min(int(frame_index), max(0, self.frame_count - 1)))
        if idx == self._cache_idx and self._cache_frame is not None:
            return self._cache_frame.copy()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        if not ok:
            return None
        self._cache_idx = idx
        self._cache_frame = frame.copy()
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ImagePanel(QLabel):
    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._pixmap = QPixmap()
        self._empty_text = title

    def set_frame(self, frame: np.ndarray | None) -> None:
        self._pixmap = _bgr_to_pixmap(frame)
        self._apply_pixmap()

    def _apply_pixmap(self) -> None:
        if self._pixmap.isNull():
            self.setText(self._empty_text)
            super().setPixmap(QPixmap())
            return
        scaled = self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        super().setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # pragma: no cover - GUI
        self._apply_pixmap()
        super().resizeEvent(event)


class InfoChip(QFrame):
    def __init__(self, caption: str, value: str = "—", *, accent: str = "#6ea8ff") -> None:
        super().__init__()
        self.setStyleSheet(_chip_style(accent=accent))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        self.caption_label = QLabel(caption)
        self.caption_label.setObjectName("caption")
        self.caption_label.setProperty("class", "caption")
        self.caption_label.setStyleSheet("font-size: 10px; color: #9fa5b5;")
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #f2f3f5;")
        layout.addWidget(self.caption_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(_safe_text(value))


class CollapsibleSection(QWidget):
    def __init__(self, title: str, *, expanded: bool = False) -> None:
        super().__init__()
        self.toggle_btn = QToolButton(text=title, checkable=True, checked=expanded)
        self.toggle_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_btn.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.toggle_btn.setStyleSheet("QToolButton { font-weight: 600; padding: 2px 0; border: 0; }")
        self.body = QWidget()
        self.body.setVisible(expanded)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.toggle_btn)
        layout.addWidget(self.body)
        self.toggle_btn.toggled.connect(self._on_toggled)

    def set_content_layout(self, layout) -> None:
        self.body.setLayout(layout)

    def _on_toggled(self, expanded: bool) -> None:
        self.toggle_btn.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.body.setVisible(expanded)


class TimelineWidget(QWidget):
    frameSelected = Signal(int)
    viewChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._session: ReviewSession | None = None
        self._current_frame = 0
        self._zoom = 1.0
        self._start_frame = 0
        self._panning = False
        self._pan_anchor_x = 0.0
        self._pan_anchor_frame = 0
        self.setMouseTracking(True)
        self.setMinimumHeight(320)

    def set_session(self, session: ReviewSession | None) -> None:
        self._session = session
        self._current_frame = 0
        self._zoom = 1.0
        self._start_frame = 0
        self.viewChanged.emit()
        self.update()

    def set_current_frame(self, frame_index: int) -> None:
        self._current_frame = max(0, int(frame_index))
        self._ensure_current_frame_visible()
        self.update()

    @property
    def zoom_factor(self) -> float:
        return float(self._zoom)

    def zoom_in(self) -> None:
        anchor = self._current_frame if self._session is not None else None
        self._set_zoom(self._zoom * 1.35, anchor_frame=anchor)

    def zoom_out(self) -> None:
        anchor = self._current_frame if self._session is not None else None
        self._set_zoom(self._zoom / 1.35, anchor_frame=anchor)

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._start_frame = 0
        self.viewChanged.emit()
        self.update()

    def pan_frames(self, delta_frames: int) -> None:
        if self._session is None:
            return
        visible = self._visible_frame_count()
        frame_count = max(1, int(self._session.frame_count))
        max_start = max(0, frame_count - visible)
        self._start_frame = max(0, min(int(self._start_frame + delta_frames), max_start))
        self.viewChanged.emit()
        self.update()

    def visible_frame_range(self) -> tuple[int, int]:
        if self._session is None:
            return 0, 0
        visible = self._visible_frame_count()
        frame_count = max(1, int(self._session.frame_count))
        start = max(0, min(int(self._start_frame), max(0, frame_count - visible)))
        end_excl = min(frame_count, start + visible)
        return start, end_excl

    def _visible_frame_count(self) -> int:
        if self._session is None:
            return 1
        frame_count = max(1, int(self._session.frame_count))
        visible = int(round(frame_count / max(self._zoom, 1.0)))
        return max(12, min(frame_count, visible))

    def _ensure_current_frame_visible(self) -> None:
        if self._session is None:
            return
        start, end_excl = self.visible_frame_range()
        visible = max(1, end_excl - start)
        frame_count = max(1, int(self._session.frame_count))
        if self._current_frame < start:
            self._start_frame = self._current_frame
        elif self._current_frame >= end_excl:
            self._start_frame = max(0, self._current_frame - visible + 1)
        self._start_frame = max(0, min(self._start_frame, max(0, frame_count - visible)))
        self.viewChanged.emit()

    def _set_zoom(self, zoom: float, *, anchor_frame: int | None = None) -> None:
        if self._session is None:
            return
        frame_count = max(1, int(self._session.frame_count))
        new_zoom = float(np.clip(zoom, 1.0, max(1.0, frame_count / 12.0)))
        if abs(new_zoom - self._zoom) < 1e-4:
            return
        old_start, old_end = self.visible_frame_range()
        old_visible = max(1, old_end - old_start)
        if anchor_frame is None:
            anchor_frame = old_start + old_visible // 2
        ratio = float(anchor_frame - old_start) / float(max(old_visible - 1, 1))
        self._zoom = new_zoom
        new_visible = self._visible_frame_count()
        new_start = int(round(float(anchor_frame) - ratio * float(max(new_visible - 1, 1))))
        self._start_frame = max(0, min(new_start, max(0, frame_count - new_visible)))
        self._ensure_current_frame_visible()
        self.viewChanged.emit()
        self.update()

    def _frame_to_x(self, frame_index: int, left: float, width: float, start: int, end_excl: int) -> float:
        visible = max(1, end_excl - start)
        if visible <= 1:
            return left
        return left + (float(frame_index - start) / float(visible - 1)) * max(width, 1.0)

    def _x_to_frame(self, x: float, left: float, width: float, start: int, end_excl: int) -> int:
        visible = max(1, end_excl - start)
        if visible <= 1:
            return start
        ratio = float(np.clip((x - left) / max(width, 1.0), 0.0, 1.0))
        return int(round(start + ratio * float(visible - 1)))

    def _track_geometry(self) -> tuple[float, float]:
        left = 118.0
        width = max(1.0, float(self.width()) - left - 14.0)
        return left, width

    def _tooltip_text(self, frame_index: int) -> str:
        if self._session is None:
            return ""
        row = dict(self._session.frame_row(frame_index))
        return "\n".join(
            [
                f"frame {frame_index} • {_format_ms(float(row.get('ts_ms', 0.0)))}",
                f"BIO {row.get('bio_label', 'O')} • pB={float(row.get('pB', 0.0)):.2f}",
                (
                    f"hands L={int(row.get('left_valid_joints', 0))}/21 "
                    f"R={int(row.get('right_valid_joints', 0))}/21"
                ),
                f"pose valid={_format_ratio(float(row.get('pose_valid_frac', 0.0)))}",
                f"segment={row.get('active_segment_id')}",
                f"word={row.get('predicted_label', '') or '—'}",
                ("warn=" + ", ".join(str(flag) for flag in list(row.get("warnings", []) or [])))
                if list(row.get("warnings", []) or [])
                else "warn=none",
            ]
        )

    def mousePressEvent(self, event) -> None:  # pragma: no cover - GUI
        if self._session is None:
            return
        start, end_excl = self.visible_frame_range()
        left, width = self._track_geometry()
        if event.button() == Qt.MouseButton.LeftButton:
            self.frameSelected.emit(self._x_to_frame(float(event.position().x()), left, width, start, end_excl))
            return
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._panning = True
            self._pan_anchor_x = float(event.position().x())
            self._pan_anchor_frame = int(self._start_frame)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # pragma: no cover - GUI
        if self._session is None:
            return
        start, end_excl = self.visible_frame_range()
        left, width = self._track_geometry()
        if self._panning:
            visible = max(1, end_excl - start)
            dx = float(event.position().x()) - self._pan_anchor_x
            frame_shift = int(round((-dx / max(width, 1.0)) * float(max(visible - 1, 1))))
            self._start_frame = self._pan_anchor_frame + frame_shift
            self.pan_frames(0)
            return
        frame_idx = self._x_to_frame(float(event.position().x()), left, width, start, end_excl)
        QToolTip.showText(event.globalPosition().toPoint(), self._tooltip_text(frame_idx), self)

    def mouseReleaseEvent(self, event) -> None:  # pragma: no cover - GUI
        del event
        self._panning = False
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event) -> None:  # pragma: no cover - GUI
        if self._session is None:
            return
        delta = int(event.angleDelta().y())
        if delta == 0:
            return
        start, end_excl = self.visible_frame_range()
        left, width = self._track_geometry()
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            visible = max(1, end_excl - start)
            self.pan_frames(int(-np.sign(delta) * max(1, visible // 8)))
            return
        anchor = self._x_to_frame(float(event.position().x()), left, width, start, end_excl)
        self._set_zoom(self._zoom * (1.25 if delta > 0 else 0.8), anchor_frame=anchor)

    def leaveEvent(self, event) -> None:  # pragma: no cover - GUI
        del event
        QToolTip.hideText()

    def paintEvent(self, event) -> None:  # pragma: no cover - GUI
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(24, 24, 28))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._session is None:
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No review session loaded")
            return

        frame_count = max(1, int(self._session.frame_count))
        tracks = self._session.timeline_tracks
        left_margin, usable_width = self._track_geometry()
        start, end_excl = self.visible_frame_range()
        track_defs = [
            ("Segments", 24.0),
            ("Words", 24.0),
            ("BIO", 18.0),
            ("pB", 58.0),
            ("Left hand", 18.0),
            ("Right hand", 18.0),
            ("Pose", 18.0),
            ("Warnings", 16.0),
        ]
        y = 14.0
        painter.setPen(QColor(56, 58, 66))
        tick_count = max(4, min(12, end_excl - start))
        for tick in np.linspace(start, max(start, end_excl - 1), num=tick_count):
            x = self._frame_to_x(int(round(tick)), left_margin, usable_width, start, end_excl)
            painter.drawLine(x, y - 4.0, x, self.height() - 18.0)

        def draw_track_label(label: str, top: float, height: float) -> QRectF:
            painter.setPen(QColor(210, 210, 214))
            painter.drawText(
                QRectF(8.0, top, left_margin - 16.0, height),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                label,
            )
            rect = QRectF(left_margin, top, usable_width, height)
            painter.setPen(QColor(72, 72, 80))
            painter.drawRect(rect)
            return rect

        segment_rect = draw_track_label("Segments", y, track_defs[0][1])
        for span in list(tracks.get("segments", []) or []):
            span_start = int(span.get("start_frame", 0))
            span_end_excl = int(span.get("end_frame_exclusive", span_start))
            clip_start = max(span_start, start)
            clip_end_excl = min(span_end_excl, end_excl)
            if clip_end_excl <= clip_start:
                continue
            x0 = self._frame_to_x(clip_start, segment_rect.left(), segment_rect.width(), start, end_excl)
            x1 = self._frame_to_x(max(clip_start, clip_end_excl - 1), segment_rect.left(), segment_rect.width(), start, end_excl)
            rect = QRectF(x0, segment_rect.top() + 2.0, max(4.0, x1 - x0 + 4.0), segment_rect.height() - 4.0)
            accepted = bool(span.get("accepted", False))
            has_label = bool(str(span.get("label", "")).strip())
            color = QColor(72, 178, 110) if accepted else (QColor(232, 145, 62) if has_label else QColor(108, 108, 116))
            painter.fillRect(rect, color)
            if rect.width() > 32.0:
                painter.setPen(QColor(15, 15, 18))
                painter.drawText(rect.adjusted(6, 0, -6, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, str(span.get("segment_id", "")))
        y += track_defs[0][1] + 8.0

        pred_rect = draw_track_label("Words", y, track_defs[1][1])
        for span in list(tracks.get("predictions", []) or []):
            span_start = int(span.get("start_frame", 0))
            span_end_excl = int(span.get("end_frame_exclusive", span_start))
            clip_start = max(span_start, start)
            clip_end_excl = min(span_end_excl, end_excl)
            if clip_end_excl <= clip_start:
                continue
            x0 = self._frame_to_x(clip_start, pred_rect.left(), pred_rect.width(), start, end_excl)
            x1 = self._frame_to_x(max(clip_start, clip_end_excl - 1), pred_rect.left(), pred_rect.width(), start, end_excl)
            rect = QRectF(x0, pred_rect.top() + 2.0, max(4.0, x1 - x0 + 4.0), pred_rect.height() - 4.0)
            accepted = bool(span.get("accepted", False))
            color = QColor(109, 210, 126) if accepted else QColor(215, 113, 83)
            painter.fillRect(rect, color)
            label = str(span.get("label", "")).strip()
            if label and rect.width() > 42.0:
                painter.setPen(QColor(18, 18, 20))
                painter.drawText(rect.adjusted(6, 0, -6, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)
        y += track_defs[1][1] + 8.0

        bio_rect = draw_track_label("BIO", y, track_defs[2][1])
        labels = list(dict(tracks.get("bio", {}) or {}).get("label", []) or [])
        colors = {"O": QColor(96, 96, 108), "B": QColor(255, 193, 7), "I": QColor(54, 145, 255)}
        if labels:
            view_labels = labels[start:end_excl]
            run_start = start
            prev = view_labels[0]
            for idx in range(start + 1, end_excl + 1):
                cur = labels[idx] if idx < end_excl else None
                if cur != prev:
                    x0 = self._frame_to_x(run_start, bio_rect.left(), bio_rect.width(), start, end_excl)
                    x1 = self._frame_to_x(max(run_start, idx - 1), bio_rect.left(), bio_rect.width(), start, end_excl)
                    painter.fillRect(
                        QRectF(x0, bio_rect.top() + 2.0, max(2.0, x1 - x0 + 2.0), bio_rect.height() - 4.0),
                        colors.get(str(prev), QColor(130, 130, 130)),
                    )
                    run_start = idx
                    prev = cur
        y += track_defs[2][1] + 8.0

        pb_rect = draw_track_label("pB", y, track_defs[3][1])
        pb = np.asarray(dict(tracks.get("bio", {}) or {}).get("pB", []) or [], dtype=np.float32)
        threshold = float(dict(tracks.get("bio", {}) or {}).get("threshold", 0.0))
        if pb.size > 0:
            painter.setPen(QPen(QColor(255, 120, 120), 1.0, Qt.PenStyle.DashLine))
            thr_y = pb_rect.bottom() - threshold * pb_rect.height()
            painter.drawLine(pb_rect.left(), thr_y, pb_rect.right(), thr_y)
            painter.setPen(QPen(QColor(240, 240, 120), 1.6))
            view_pb = pb[start:end_excl]
            last_x = self._frame_to_x(start, pb_rect.left(), pb_rect.width(), start, end_excl)
            last_y = pb_rect.bottom() - float(view_pb[0]) * pb_rect.height()
            for idx_offset in range(1, len(view_pb)):
                frame_idx = start + idx_offset
                x = self._frame_to_x(frame_idx, pb_rect.left(), pb_rect.width(), start, end_excl)
                yv = pb_rect.bottom() - float(view_pb[idx_offset]) * pb_rect.height()
                painter.drawLine(last_x, last_y, x, yv)
                last_x, last_y = x, yv
        y += track_defs[3][1] + 8.0

        for label, key, color in (
            ("Left hand", "left_valid_frac", QColor(90, 220, 110)),
            ("Right hand", "right_valid_frac", QColor(90, 180, 255)),
        ):
            rect = draw_track_label(label, y, 18.0)
            values = np.asarray(dict(tracks.get("hands", {}) or {}).get(key, []) or [], dtype=np.float32)
            painter.setPen(QPen(color, 1.0))
            for frame_idx in range(start, min(end_excl, int(values.size))):
                value = float(values[frame_idx])
                if value <= 0.0:
                    continue
                x = self._frame_to_x(frame_idx, rect.left(), rect.width(), start, end_excl)
                painter.drawLine(x, rect.bottom(), x, rect.bottom() - value * rect.height())
            y += 26.0

        pose_rect = draw_track_label("Pose", y, 18.0)
        pose_vals = np.asarray(dict(tracks.get("pose", {}) or {}).get("valid_frac", []) or [], dtype=np.float32)
        painter.setPen(QPen(QColor(200, 180, 255), 1.0))
        for frame_idx in range(start, min(end_excl, int(pose_vals.size))):
            value = float(pose_vals[frame_idx])
            if value <= 0.0:
                continue
            x = self._frame_to_x(frame_idx, pose_rect.left(), pose_rect.width(), start, end_excl)
            painter.drawLine(x, pose_rect.bottom(), x, pose_rect.bottom() - value * pose_rect.height())
        y += 26.0

        warn_rect = draw_track_label("Warnings", y, 16.0)
        warning_rows = list(tracks.get("frame_warnings", []) or [])
        painter.setPen(QPen(QColor(255, 90, 90), 1.4))
        painter.setBrush(QColor(255, 90, 90))
        for frame_idx in range(start, min(end_excl, len(warning_rows))):
            if not warning_rows[frame_idx]:
                continue
            x = self._frame_to_x(frame_idx, warn_rect.left(), warn_rect.width(), start, end_excl)
            painter.drawLine(x, warn_rect.top() + 2.0, x, warn_rect.bottom() - 2.0)
            painter.drawEllipse(QRectF(x - 2.5, warn_rect.top() + 3.0, 5.0, 5.0))

        play_x = self._frame_to_x(self._current_frame, segment_rect.left(), segment_rect.width(), start, end_excl)
        painter.setPen(QPen(QColor(255, 255, 255), 2.5))
        painter.drawLine(play_x, segment_rect.top() - 6.0, play_x, warn_rect.bottom() + 5.0)


class ReviewWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BIO + MSAGCN Desktop Review")
        self.resize(1820, 1060)
        self._session: ReviewSession | None = None
        self._original_video: VideoFrameSource | None = None
        self._overlay_video: VideoFrameSource | None = None
        self._current_frame = 0
        self._process: QProcess | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._play_tick)
        self._segment_rows: List[Dict[str, Any]] = []
        self._segment_lookup: Dict[int, Dict[str, Any]] = {}
        self._build_ui()
        self._apply_defaults()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(12, 10, 12, 12)
        main_layout.setSpacing(10)

        main_layout.addWidget(self._build_summary_bar())

        body = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(body, 1)
        left_panel = self._build_left_panel()
        center_panel = self._build_center_panel()
        right_panel = self._build_right_panel()
        left_panel.setMinimumWidth(320)
        right_panel.setMinimumWidth(480)
        body.addWidget(left_panel)
        body.addWidget(center_panel)
        body.addWidget(right_panel)
        body.setSizes([340, 1120, 560])
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setStretchFactor(2, 0)

    def _build_summary_bar(self) -> QWidget:
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setStyleSheet("QFrame { background: #17181d; border: 1px solid #343844; border-radius: 12px; }")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(14)

        left = QVBoxLayout()
        left.setSpacing(2)
        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("color: #9da5b6; font-size: 12px;")
        self.session_title_label = QLabel("No review session")
        self.session_title_label.setStyleSheet("font-size: 18px; font-weight: 700;")
        left.addWidget(self.status_label)
        left.addWidget(self.session_title_label)
        layout.addLayout(left, 2)

        middle = QVBoxLayout()
        middle.setSpacing(4)
        sentence_caption = QLabel("Sentence")
        sentence_caption.setStyleSheet("font-size: 11px; color: #9199aa; text-transform: uppercase;")
        self.sentence_value_label = QLabel("No accepted sentence")
        self.sentence_value_label.setWordWrap(True)
        self.sentence_value_label.setStyleSheet("font-size: 22px; font-weight: 700;")
        middle.addWidget(sentence_caption)
        middle.addWidget(self.sentence_value_label)
        layout.addLayout(middle, 4)

        stats = QHBoxLayout()
        stats.setSpacing(8)
        self.segment_count_chip = InfoChip("Segments", "0", accent="#f0aa4b")
        self.accepted_count_chip = InfoChip("Accepted", "0", accent="#63d18a")
        self.rejected_count_chip = InfoChip("Rejected", "0", accent="#f16f5b")
        stats.addWidget(self.segment_count_chip)
        stats.addWidget(self.accepted_count_chip)
        stats.addWidget(self.rejected_count_chip)
        layout.addLayout(stats, 0)

        warn_box = QVBoxLayout()
        warn_box.setSpacing(6)
        warn_caption = QLabel("Warnings")
        warn_caption.setStyleSheet("font-size: 11px; color: #9199aa; text-transform: uppercase;")
        self.warning_chip_widget = QWidget()
        self.warning_chip_layout = QHBoxLayout(self.warning_chip_widget)
        self.warning_chip_layout.setContentsMargins(0, 0, 0, 0)
        self.warning_chip_layout.setSpacing(6)
        warn_box.addWidget(warn_caption)
        warn_box.addWidget(self.warning_chip_widget)
        layout.addLayout(warn_box, 3)
        return card

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        inputs_box = QGroupBox("Inputs")
        inputs_form = QFormLayout(inputs_box)
        self.video_edit = QLineEdit()
        self.out_dir_edit = QLineEdit()
        self.extractor_combo = QComboBox()
        self.extractor_combo.addItems(["auto", "hands_only", "holistic_hands_pose"])
        inputs_form.addRow("Video", self.video_edit)
        inputs_form.addRow("Output dir", self.out_dir_edit)
        inputs_form.addRow("Extractor", self.extractor_combo)
        layout.addWidget(inputs_box)

        run_box = QGroupBox("Run")
        run_layout = QGridLayout(run_box)
        self.open_session_btn = QPushButton("Open Session")
        self.run_inference_btn = QPushButton("Run Inference")
        self.browse_video_btn = QPushButton("Browse Video")
        self.browse_output_btn = QPushButton("Browse Output")
        self.open_session_btn.clicked.connect(self._open_session_dialog)
        self.run_inference_btn.clicked.connect(self._start_review_build)
        self.browse_video_btn.clicked.connect(self._browse_video)
        self.browse_output_btn.clicked.connect(self._browse_output_dir)
        run_layout.addWidget(self.browse_video_btn, 0, 0)
        run_layout.addWidget(self.browse_output_btn, 0, 1)
        run_layout.addWidget(self.open_session_btn, 1, 0)
        run_layout.addWidget(self.run_inference_btn, 1, 1)
        layout.addWidget(run_box)

        session_box = QGroupBox("Session Files")
        export_layout = QGridLayout(session_box)
        self.open_artifacts_btn = QPushButton("Open Folder")
        self.open_sentence_btn = QPushButton("Sentence")
        self.open_segments_btn = QPushButton("Segments")
        self.open_predictions_btn = QPushButton("Predictions")
        self.open_timeline_btn = QPushButton("Timeline CSV")
        self.open_overlay_btn = QPushButton("Overlay MP4")
        self.copy_sentence_btn = QPushButton("Copy Sentence")
        for btn, handler in [
            (self.open_artifacts_btn, lambda: self._open_path(self._session.root if self._session else None)),
            (self.open_sentence_btn, lambda: self._open_artifact("sentence")),
            (self.open_segments_btn, lambda: self._open_artifact("segments")),
            (self.open_predictions_btn, lambda: self._open_artifact("predictions")),
            (self.open_timeline_btn, lambda: self._open_artifact("timeline_csv")),
            (self.open_overlay_btn, lambda: self._open_artifact("preview_overlay")),
            (self.copy_sentence_btn, self._copy_sentence),
        ]:
            btn.clicked.connect(handler)
        export_layout.addWidget(self.open_artifacts_btn, 0, 0, 1, 2)
        export_layout.addWidget(self.open_sentence_btn, 1, 0)
        export_layout.addWidget(self.open_segments_btn, 1, 1)
        export_layout.addWidget(self.open_predictions_btn, 2, 0)
        export_layout.addWidget(self.open_timeline_btn, 2, 1)
        export_layout.addWidget(self.open_overlay_btn, 3, 0)
        export_layout.addWidget(self.copy_sentence_btn, 3, 1)
        layout.addWidget(session_box)

        advanced = CollapsibleSection("Advanced", expanded=False)
        advanced_form = QFormLayout()
        self.bio_checkpoint_edit = QLineEdit()
        self.bio_bundle_edit = QLineEdit()
        self.bio_selection_combo = QComboBox()
        self.bio_selection_combo.addItems(["best_balanced", "best_boundary", "last"])
        self.ms_checkpoint_edit = QLineEdit()
        self.ms_bundle_edit = QLineEdit()
        self.ms_label_map_edit = QLineEdit()
        self.ms_ds_config_edit = QLineEdit()
        self.sentence_conf_edit = QLineEdit("0.5")
        advanced_form.addRow("BIO checkpoint", self.bio_checkpoint_edit)
        advanced_form.addRow("BIO bundle", self.bio_bundle_edit)
        advanced_form.addRow("BIO selection", self.bio_selection_combo)
        advanced_form.addRow("MS checkpoint", self.ms_checkpoint_edit)
        advanced_form.addRow("MS bundle", self.ms_bundle_edit)
        advanced_form.addRow("MS label map", self.ms_label_map_edit)
        advanced_form.addRow("MS ds_config", self.ms_ds_config_edit)
        advanced_form.addRow("Sentence min conf", self.sentence_conf_edit)
        advanced.set_content_layout(advanced_form)
        layout.addWidget(advanced)

        layout.addStretch(1)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        toolbar.addWidget(QLabel("View"))
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Split", "Original", "Overlay"])
        self.view_mode_combo.currentTextChanged.connect(self._apply_view_mode)
        toolbar.addWidget(self.view_mode_combo)
        toolbar.addSpacing(12)
        toolbar.addWidget(QLabel("Overlay"))
        self.toggle_hands_btn = self._make_toggle("Hands", True)
        self.toggle_pose_btn = self._make_toggle("Pose", True)
        self.toggle_bio_btn = self._make_toggle("BIO", True)
        self.toggle_probs_btn = self._make_toggle("Probs", True)
        self.toggle_warnings_btn = self._make_toggle("Warnings", True)
        self.toggle_labels_btn = self._make_toggle("Labels", True)
        for btn in (
            self.toggle_hands_btn,
            self.toggle_pose_btn,
            self.toggle_bio_btn,
            self.toggle_probs_btn,
            self.toggle_warnings_btn,
            self.toggle_labels_btn,
        ):
            btn.toggled.connect(self._refresh_current_frame)
            toolbar.addWidget(btn)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        self.video_split = QSplitter(Qt.Orientation.Horizontal)
        self.original_group = self._wrap_media_panel("Original", "Original video")
        self.overlay_group = self._wrap_media_panel("Overlay", "Overlay video")
        self.video_split.addWidget(self.original_group)
        self.video_split.addWidget(self.overlay_group)
        self.video_split.setStretchFactor(0, 1)
        self.video_split.setStretchFactor(1, 1)
        layout.addWidget(self.video_split, 3)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.play_btn = QPushButton("Play")
        self.prev_btn = QPushButton("Prev")
        self.next_btn = QPushButton("Next")
        self.time_label = QLabel("t=0ms")
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_label = QLabel("frame=0/0")
        self.play_btn.clicked.connect(self._toggle_play)
        self.prev_btn.clicked.connect(lambda: self._seek_frame(self._current_frame - 1))
        self.next_btn.clicked.connect(lambda: self._seek_frame(self._current_frame + 1))
        self.frame_slider.valueChanged.connect(self._seek_frame)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.prev_btn)
        controls.addWidget(self.next_btn)
        controls.addWidget(self.time_label)
        controls.addWidget(self.frame_slider, 1)
        controls.addWidget(self.frame_label)
        layout.addLayout(controls)

        timeline_tools = QHBoxLayout()
        timeline_tools.setSpacing(8)
        self.zoom_out_btn = QPushButton("Zoom -")
        self.zoom_in_btn = QPushButton("Zoom +")
        self.zoom_reset_btn = QPushButton("Reset")
        self.timeline_zoom_label = QLabel("Timeline: 1.0x")
        self.timeline_legend_label = QLabel("Legend: L hand • R hand • Pose • O/B/I • accepted/rejected")
        self.timeline_legend_label.setStyleSheet("color: #a0a7b7;")
        self.zoom_out_btn.clicked.connect(lambda: self._timeline_zoom(self.timeline_widget.zoom_out))
        self.zoom_in_btn.clicked.connect(lambda: self._timeline_zoom(self.timeline_widget.zoom_in))
        self.zoom_reset_btn.clicked.connect(lambda: self._timeline_zoom(self.timeline_widget.reset_view))
        timeline_tools.addWidget(self.zoom_out_btn)
        timeline_tools.addWidget(self.zoom_in_btn)
        timeline_tools.addWidget(self.zoom_reset_btn)
        timeline_tools.addWidget(self.timeline_zoom_label)
        timeline_tools.addStretch(1)
        timeline_tools.addWidget(self.timeline_legend_label)
        layout.addLayout(timeline_tools)

        self.timeline_widget = TimelineWidget()
        self.timeline_widget.frameSelected.connect(self._seek_frame)
        self.timeline_widget.viewChanged.connect(self._update_timeline_status)
        layout.addWidget(self.timeline_widget, 2)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        warnings_box = QGroupBox("Warnings")
        warnings_layout = QVBoxLayout(warnings_box)
        self.warning_list = QListWidget()
        warnings_layout.addWidget(self.warning_list)
        layout.addWidget(warnings_box, 1)

        tabs = QTabWidget()
        layout.addWidget(tabs, 5)

        seg_tab = QWidget()
        seg_layout = QVBoxLayout(seg_tab)
        seg_layout.setSpacing(8)
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Show"))
        self.segment_filter_combo = QComboBox()
        self.segment_filter_combo.addItems(["All", "Accepted", "Rejected"])
        self.segment_filter_combo.currentTextChanged.connect(self._populate_segment_table)
        self.segment_count_label = QLabel("0 rows")
        filter_row.addWidget(self.segment_filter_combo)
        filter_row.addStretch(1)
        filter_row.addWidget(self.segment_count_label)
        seg_layout.addLayout(filter_row)

        self.segment_table = QTableWidget(0, 7)
        self.segment_table.setHorizontalHeaderLabels(["ID", "Start", "End", "Label", "Conf", "Family", "Accepted"])
        self.segment_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.segment_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.segment_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.segment_table.setSortingEnabled(True)
        self.segment_table.verticalHeader().setVisible(False)
        self.segment_table.horizontalHeader().setStretchLastSection(False)
        self.segment_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.segment_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.segment_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.segment_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.segment_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.segment_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.segment_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        self.segment_table.cellClicked.connect(self._on_segment_clicked)
        self.segment_table.itemSelectionChanged.connect(self._on_segment_selection_changed)
        seg_layout.addWidget(self.segment_table, 3)

        seg_scroll = QScrollArea()
        seg_scroll.setWidgetResizable(True)
        seg_scroll_widget = QWidget()
        seg_scroll.setWidget(seg_scroll_widget)
        seg_details = QVBoxLayout(seg_scroll_widget)
        self.segment_summary_box, self.segment_summary_labels = self._create_detail_box(
            "Segment",
            [
                ("segment_id", "Segment"),
                ("start", "Start"),
                ("end", "End"),
                ("duration", "Duration"),
                ("label", "Label"),
                ("confidence", "Confidence"),
                ("family", "Family"),
                ("accepted", "Accepted"),
            ],
        )
        self.segment_score_box, self.segment_score_labels = self._create_detail_box(
            "Decision",
            [
                ("decision_reason", "Decision reason"),
                ("end_reason", "End reason"),
                ("boundary_score", "Boundary score"),
                ("mean_inside_score", "Mean inside score"),
                ("topk", "Top-k"),
                ("family_topk", "Family top-k"),
            ],
        )
        self.segment_raw_toggle = QToolButton(text="Show Raw JSON", checkable=True, checked=False)
        self.segment_raw_toggle.toggled.connect(lambda on: self.segment_raw_text.setVisible(on))
        self.segment_raw_text = QPlainTextEdit()
        self.segment_raw_text.setReadOnly(True)
        self.segment_raw_text.setVisible(False)
        seg_details.addWidget(self.segment_summary_box)
        seg_details.addWidget(self.segment_score_box)
        seg_details.addWidget(self.segment_raw_toggle)
        seg_details.addWidget(self.segment_raw_text)
        seg_details.addStretch(1)
        seg_layout.addWidget(seg_scroll, 3)
        tabs.addTab(seg_tab, "Segments")

        frame_tab = QWidget()
        frame_layout = QVBoxLayout(frame_tab)
        frame_scroll = QScrollArea()
        frame_scroll.setWidgetResizable(True)
        frame_scroll_widget = QWidget()
        frame_scroll.setWidget(frame_scroll_widget)
        frame_details = QVBoxLayout(frame_scroll_widget)
        self.frame_summary_box, self.frame_summary_labels = self._create_detail_box(
            "Frame",
            [
                ("frame_index", "Frame"),
                ("timestamp", "Timestamp"),
                ("active_segment_id", "Active segment"),
                ("accepted", "Accepted"),
                ("predicted_label", "Predicted label"),
                ("predicted_confidence", "Prediction conf"),
                ("family_label", "Family"),
                ("family_confidence", "Family conf"),
            ],
        )
        self.frame_bio_box, self.frame_bio_labels = self._create_detail_box(
            "BIO",
            [
                ("bio_label", "State"),
                ("bio_probs", "pO / pB / pI"),
                ("threshold", "Threshold"),
                ("hand_guard", "Hand guard"),
                ("warnings", "Warnings"),
            ],
        )
        self.frame_track_box, self.frame_track_labels = self._create_detail_box(
            "Tracking",
            [
                ("left_valid", "Left valid joints"),
                ("right_valid", "Right valid joints"),
                ("total_valid", "Total valid hand joints"),
                ("pose_valid", "Pose valid"),
                ("slot_layout", "Slot layout"),
                ("left_wrist", "Left wrist"),
                ("right_wrist", "Right wrist"),
            ],
        )
        self.frame_raw_toggle = QToolButton(text="Show Raw JSON", checkable=True, checked=False)
        self.frame_raw_toggle.toggled.connect(lambda on: self.frame_raw_text.setVisible(on))
        self.frame_raw_text = QPlainTextEdit()
        self.frame_raw_text.setReadOnly(True)
        self.frame_raw_text.setVisible(False)
        frame_details.addWidget(self.frame_summary_box)
        frame_details.addWidget(self.frame_bio_box)
        frame_details.addWidget(self.frame_track_box)
        frame_details.addWidget(self.frame_raw_toggle)
        frame_details.addWidget(self.frame_raw_text)
        frame_details.addStretch(1)
        frame_layout.addWidget(frame_scroll, 1)
        tabs.addTab(frame_tab, "Frame")

        logs_tab = QWidget()
        logs_layout = QVBoxLayout(logs_tab)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        logs_layout.addWidget(self.log_box, 1)
        tabs.addTab(logs_tab, "Logs")
        return panel

    def _make_toggle(self, label: str, checked: bool) -> QPushButton:
        btn = QPushButton(label)
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setStyleSheet(_toggle_button_style())
        return btn

    def _wrap_media_panel(self, title: str, empty_text: str) -> QWidget:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        panel = ImagePanel(empty_text)
        if title.lower().startswith("original"):
            self.original_view = panel
        else:
            self.overlay_view = panel
        layout.addWidget(panel)
        return box

    def _create_detail_box(self, title: str, fields: List[tuple[str, str]]) -> tuple[QGroupBox, Dict[str, QLabel]]:
        box = QGroupBox(title)
        form = QFormLayout(box)
        labels: Dict[str, QLabel] = {}
        for key, caption in fields:
            value = QLabel("—")
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(caption, value)
            labels[key] = value
        return box, labels

    def _set_detail_values(self, mapping: Dict[str, QLabel], values: Dict[str, Any]) -> None:
        for key, label in mapping.items():
            label.setText(_safe_text(values.get(key, "—")))

    def _apply_defaults(self) -> None:
        self.video_edit.setText(_repo_default("test_video/first.mp4"))
        self.out_dir_edit.setText(_repo_default("outputs/review_sessions/latest"))
        self.bio_checkpoint_edit.setText(_repo_default("outputs/runs/bio_v2_run/best_balanced_model.pt"))
        self.ms_checkpoint_edit.setText(_repo_default("outputs/families/agcn_supcon_cb_oof/family_finetune/best_rebalance.ckpt"))
        self.ms_label_map_edit.setText(_repo_default("outputs/families/agcn_supcon_cb_oof/family_finetune/label2idx.json"))
        self.ms_ds_config_edit.setText(_repo_default("outputs/families/agcn_supcon_cb_oof/family_finetune/ds_config.json"))
        self._apply_view_mode("Split")

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select video", self.video_edit.text() or str(REPO_ROOT), "Videos (*.mp4 *.avi *.mov *.mkv);;All files (*)")
        if path:
            self.video_edit.setText(path)

    def _browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output directory", self.out_dir_edit.text() or str(REPO_ROOT))
        if path:
            self.out_dir_edit.setText(path)

    def _open_session_dialog(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open review session folder", self.out_dir_edit.text() or str(REPO_ROOT))
        if path:
            self._load_session(path)

    def _append_log(self, text: str) -> None:
        self.log_box.appendPlainText(text.rstrip())

    def _build_review_args(self) -> List[str]:
        args = [
            "-m",
            "pipeline",
            "build-review-session",
            "--input",
            self.video_edit.text().strip(),
            "--out_dir",
            self.out_dir_edit.text().strip(),
            "--bio_selection",
            self.bio_selection_combo.currentText(),
            "--extractor_mode",
            self.extractor_combo.currentText(),
            "--sentence_min_confidence",
            self.sentence_conf_edit.text().strip() or "0.5",
            "--console_format",
            "text",
        ]
        if self.bio_bundle_edit.text().strip():
            args += ["--bio_bundle", self.bio_bundle_edit.text().strip()]
        else:
            args += ["--bio_checkpoint", self.bio_checkpoint_edit.text().strip()]
        if self.ms_bundle_edit.text().strip():
            args += ["--msagcn_bundle", self.ms_bundle_edit.text().strip()]
        else:
            args += ["--msagcn_checkpoint", self.ms_checkpoint_edit.text().strip()]
            if self.ms_label_map_edit.text().strip():
                args += ["--msagcn_label_map", self.ms_label_map_edit.text().strip()]
            if self.ms_ds_config_edit.text().strip():
                args += ["--msagcn_ds_config", self.ms_ds_config_edit.text().strip()]
        return args

    def _start_review_build(self) -> None:
        if self._process is not None:
            QMessageBox.warning(self, "Already running", "Inference is already running.")
            return
        out_dir = self.out_dir_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "Missing output", "Select an output directory first.")
            return
        self._append_log("")
        self._append_log("[desktop] starting build-review-session")
        self.status_label.setText("Running inference…")
        self.run_inference_btn.setEnabled(False)
        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(self._build_review_args())
        process.setWorkingDirectory(str(REPO_ROOT))
        process.readyReadStandardOutput.connect(self._on_process_stdout)
        process.readyReadStandardError.connect(self._on_process_stderr)
        process.finished.connect(self._on_process_finished)
        process.start()
        self._process = process

    def _on_process_stdout(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode(errors="replace")
        if data:
            self._append_log(data)

    def _on_process_stderr(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardError()).decode(errors="replace")
        if data:
            self._append_log(data)

    def _on_process_finished(self, exit_code: int, exit_status) -> None:
        del exit_status
        out_dir = Path(self.out_dir_edit.text().strip())
        self._process = None
        self.run_inference_btn.setEnabled(True)
        if exit_code != 0:
            self.status_label.setText(f"Build failed (exit={exit_code})")
            QMessageBox.critical(self, "Inference failed", f"build-review-session exited with code {exit_code}")
            return
        self.status_label.setText("Review session built")
        self._load_session(out_dir)

    def _load_session(self, path: str | Path) -> None:
        try:
            session = load_review_session(path)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load session", str(exc))
            return
        self._close_video_sources()
        self._session = session
        self._segment_rows = list(session.segment_rows)
        self._segment_lookup = {int(row.get("segment_id", -1)): row for row in self._segment_rows}
        self.video_edit.setText(session.video_path)
        self.out_dir_edit.setText(str(session.root))
        self._original_video = VideoFrameSource(session.video_path) if session.video_path else None
        try:
            overlay_path = session.artifact_path("preview_overlay")
        except Exception:
            overlay_path = None
        self._overlay_video = VideoFrameSource(overlay_path) if overlay_path and overlay_path.exists() else None
        self._populate_from_session()
        self._seek_frame(0)

    def _populate_from_session(self) -> None:
        if self._session is None:
            return
        self.session_title_label.setText(self._session.session_name)
        self.sentence_value_label.setText(self._session.sentence or "No accepted sentence")
        self.segment_count_chip.set_value(str(len(self._segment_rows)))
        self.accepted_count_chip.set_value(str(self._session.accepted_segment_count))
        self.rejected_count_chip.set_value(str(self._session.rejected_segment_count))
        self._populate_warning_summary()
        self.timeline_widget.set_session(self._session)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, self._session.frame_count - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self._populate_segment_table()
        self._update_timeline_status()
        self.status_label.setText(f"Loaded • {self._session.frame_count} frames")

    def _populate_warning_summary(self) -> None:
        _clear_layout(self.warning_chip_layout)
        self.warning_list.clear()
        if self._session is None:
            return
        warnings = list(self._session.warnings or [])
        if not warnings:
            chip = QLabel("No warnings")
            chip.setStyleSheet(_warning_chip_style("info"))
            self.warning_chip_layout.addWidget(chip)
            item = QListWidgetItem("[info] No warnings")
            self.warning_list.addItem(item)
            return
        for item in warnings:
            severity = str(item.get("severity", "warning"))
            message = str(item.get("message", "")).strip()
            group = _warning_group(item)
            chip = QLabel(message)
            chip.setStyleSheet(_warning_chip_style(severity))
            self.warning_chip_layout.addWidget(chip)
            row = QListWidgetItem(f"[{group}] {message}")
            if severity == "warning":
                row.setForeground(QColor(245, 196, 81))
            self.warning_list.addItem(row)
        self.warning_chip_layout.addStretch(1)

    def _filtered_segment_rows(self) -> List[Dict[str, Any]]:
        mode = self.segment_filter_combo.currentText().strip().lower()
        if mode == "accepted":
            return [row for row in self._segment_rows if bool(row.get("accepted", False))]
        if mode == "rejected":
            return [row for row in self._segment_rows if not bool(row.get("accepted", False))]
        return list(self._segment_rows)

    def _populate_segment_table(self) -> None:
        self.segment_table.setSortingEnabled(False)
        self.segment_table.setRowCount(0)
        visible_rows = self._filtered_segment_rows()
        for row_idx, row in enumerate(visible_rows):
            self.segment_table.insertRow(row_idx)
            values = [
                row.get("segment_id", ""),
                row.get("start_frame", ""),
                row.get("end_frame_exclusive", ""),
                row.get("label", ""),
                f"{float(row.get('confidence', 0.0)):.3f}",
                row.get("family_label", ""),
                "yes" if bool(row.get("accepted", False)) else "no",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(_safe_text(value))
                if col == 0:
                    item.setData(Qt.ItemDataRole.UserRole, int(row.get("segment_id", -1)))
                self.segment_table.setItem(row_idx, col, item)
        self.segment_table.setSortingEnabled(True)
        self.segment_table.resizeRowsToContents()
        self.segment_count_label.setText(f"{len(visible_rows)} / {len(self._segment_rows)} shown")
        if visible_rows:
            self.segment_table.selectRow(0)
            self._show_segment_details(int(visible_rows[0].get("segment_id", -1)))
        else:
            self._show_segment_details(-1)

    def _segment_id_from_row(self, row: int) -> int:
        item = self.segment_table.item(row, 0)
        if item is None:
            return -1
        return int(item.data(Qt.ItemDataRole.UserRole) or -1)

    def _on_segment_clicked(self, row: int, column: int) -> None:
        del column
        seg_id = self._segment_id_from_row(row)
        if seg_id < 0:
            return
        self._show_segment_details(seg_id)
        row_payload = self._segment_lookup.get(seg_id, {})
        self._seek_frame(int(row_payload.get("start_frame", 0)))

    def _on_segment_selection_changed(self) -> None:
        items = self.segment_table.selectedItems()
        if not items:
            return
        seg_id = self._segment_id_from_row(items[0].row())
        if seg_id >= 0:
            self._show_segment_details(seg_id)

    def _segment_topk_text(self, row: Dict[str, Any]) -> str:
        pred = dict(row.get("prediction", {}) or {})
        topk = list(pred.get("topk", []) or [])[:3]
        if not topk:
            return "—"
        parts = [f"{int(item.get('rank', 0))}. {item.get('label', '')} ({float(item.get('prob', 0.0)):.2f})" for item in topk]
        return " • ".join(parts)

    def _segment_family_topk_text(self, row: Dict[str, Any]) -> str:
        pred = dict(row.get("prediction", {}) or {})
        topk = list(pred.get("family_topk", []) or [])[:3]
        if not topk:
            return "—"
        parts = [f"{int(item.get('rank', 0))}. {item.get('label', '')} ({float(item.get('prob', 0.0)):.2f})" for item in topk]
        return " • ".join(parts)

    def _show_segment_details(self, segment_id: int) -> None:
        row = dict(self._segment_lookup.get(int(segment_id), {}) or {})
        if not row:
            empty = {
                "segment_id": "—",
                "start": "—",
                "end": "—",
                "duration": "—",
                "label": "—",
                "confidence": "—",
                "family": "—",
                "accepted": "—",
                "decision_reason": "—",
                "end_reason": "—",
                "boundary_score": "—",
                "mean_inside_score": "—",
                "topk": "—",
                "family_topk": "—",
            }
            self._set_detail_values(self.segment_summary_labels, empty)
            self._set_detail_values(self.segment_score_labels, empty)
            self.segment_raw_text.setPlainText("")
            return
        summary = {
            "segment_id": str(row.get("segment_id", "—")),
            "start": f"{int(row.get('start_frame', 0))} • {_format_ms(float(row.get('start_time_ms', 0.0)))}",
            "end": f"{int(row.get('end_frame_exclusive', 0))} • {_format_ms(float(row.get('end_time_ms', 0.0)))}",
            "duration": _format_ms(float(row.get("duration_ms", 0.0))),
            "label": row.get("label", "—") or "—",
            "confidence": f"{float(row.get('confidence', 0.0)):.3f}",
            "family": (
                f"{row.get('family_label', '') or '—'}"
                f" ({float(row.get('family_confidence', 0.0)):.3f})"
            ),
            "accepted": "yes" if bool(row.get("accepted", False)) else "no",
        }
        scores = {
            "decision_reason": row.get("decision_reason", "—") or "—",
            "end_reason": row.get("end_reason", "—") or "—",
            "boundary_score": f"{float(row.get('boundary_score', 0.0)):.3f}",
            "mean_inside_score": f"{float(row.get('mean_inside_score', 0.0)):.3f}",
            "topk": self._segment_topk_text(row),
            "family_topk": self._segment_family_topk_text(row),
        }
        self._set_detail_values(self.segment_summary_labels, summary)
        self._set_detail_values(self.segment_score_labels, scores)
        self.segment_raw_text.setPlainText(json.dumps(row, ensure_ascii=False, indent=2))

    def _seek_frame(self, frame_index: int) -> None:
        if self._session is None:
            return
        max_idx = max(0, self._session.frame_count - 1)
        idx = max(0, min(int(frame_index), max_idx))
        self._current_frame = idx
        self.timeline_widget.set_current_frame(idx)
        if self.frame_slider.value() != idx:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(idx)
            self.frame_slider.blockSignals(False)
        row = dict(self._session.frame_row(idx))
        self.frame_label.setText(f"frame={idx}/{max_idx}")
        self.time_label.setText(f"t={_format_ms(float(row.get('ts_ms', 0.0)))}")
        original_frame = self._original_video.read_frame(idx) if self._original_video is not None else None
        self.original_view.set_frame(original_frame)
        overlay_frame = self._render_overlay_frame(idx, original_frame)
        if overlay_frame is None and self._overlay_video is not None:
            overlay_frame = self._overlay_video.read_frame(idx)
        self.overlay_view.set_frame(overlay_frame)
        self._update_frame_detail(idx)
        self._sync_segment_selection(idx)

    def _render_overlay_frame(self, frame_index: int, original_frame: np.ndarray | None) -> np.ndarray | None:
        if self._session is None or original_frame is None:
            return None
        seq = self._session.load_sequence()
        row = dict(self._session.frame_row(frame_index))
        canvas = original_frame.copy()
        if self.toggle_pose_btn.isChecked():
            canvas = _draw_pose_overlay(
                canvas,
                (seq.pose_xyz[frame_index] if seq.pose_xyz is not None else None),
                (seq.pose_vis[frame_index] if seq.pose_vis is not None else None),
                seq.pose_indices,
            )
        if self.toggle_hands_btn.isChecked():
            canvas = _draw_hand21_overlay(canvas, seq.pts[frame_index, :21, :], seq.mask[frame_index, :21, :], color=(80, 220, 80))
            canvas = _draw_hand21_overlay(canvas, seq.pts[frame_index, 21:, :], seq.mask[frame_index, 21:, :], color=(80, 180, 255))

        if self.toggle_bio_btn.isChecked():
            top_left = [
                f"frame {frame_index}/{max(0, self._session.frame_count - 1)} • {_format_ms(float(row.get('ts_ms', 0.0)))}",
                f"mode {self._session.manifest.get('extractor_mode', 'unknown')}",
                f"BIO {row.get('bio_label', 'O')}",
            ]
            canvas = _draw_badge_block(canvas, top_left, anchor="top_left")
        if self.toggle_probs_btn.isChecked():
            top_right = [
                (
                    f"pO {float(row.get('pO', 0.0)):.2f}  "
                    f"pB {float(row.get('pB', 0.0)):.2f}  "
                    f"pI {float(row.get('pI', 0.0)):.2f}"
                ),
                f"thr {float(row.get('threshold', 0.0)):.2f}",
            ]
            canvas = _draw_badge_block(canvas, top_right, anchor="top_right")
        if self.toggle_labels_btn.isChecked():
            seg_lines: List[str] = []
            seg_id = row.get("active_segment_id")
            if seg_id is not None:
                seg_lines.append(f"segment #{int(seg_id)}")
            pred_label = str(row.get("predicted_label", "")).strip()
            if pred_label:
                seg_lines.append(f"{pred_label} • {float(row.get('predicted_confidence', 0.0)):.2f}")
            family_label = str(row.get("family_label", "")).strip()
            if family_label:
                seg_lines.append(f"family {family_label} • {float(row.get('family_confidence', 0.0)):.2f}")
            if seg_lines:
                canvas = _draw_badge_block(canvas, seg_lines, anchor="bottom_left")
        if self.toggle_warnings_btn.isChecked():
            warnings = list(row.get("warnings", []) or [])
            if warnings:
                canvas = _draw_badge_block(canvas, [f"warn {flag}" for flag in warnings[:3]], anchor="bottom_right", fill_bgr=(54, 34, 24))
        return canvas

    def _sync_segment_selection(self, frame_index: int) -> None:
        if self._session is None:
            return
        active_segment_id = self._session.frame_row(frame_index).get("active_segment_id")
        if active_segment_id is None:
            self.segment_table.clearSelection()
            return
        for row_idx in range(self.segment_table.rowCount()):
            if self._segment_id_from_row(row_idx) == int(active_segment_id):
                self.segment_table.blockSignals(True)
                self.segment_table.selectRow(row_idx)
                self.segment_table.blockSignals(False)
                self._show_segment_details(int(active_segment_id))
                break

    def _update_frame_detail(self, frame_index: int) -> None:
        if self._session is None:
            return
        row = dict(self._session.frame_row(frame_index))
        seq = self._session.load_sequence()
        accepted = bool(row.get("prediction_accepted", False))
        summary = {
            "frame_index": str(frame_index),
            "timestamp": _format_ms(float(row.get("ts_ms", 0.0))),
            "active_segment_id": _safe_text(row.get("active_segment_id", "—")),
            "accepted": "yes" if accepted else "no",
            "predicted_label": row.get("predicted_label", "—") or "—",
            "predicted_confidence": f"{float(row.get('predicted_confidence', 0.0)):.3f}",
            "family_label": row.get("family_label", "—") or "—",
            "family_confidence": f"{float(row.get('family_confidence', 0.0)):.3f}",
        }
        bio = {
            "bio_label": row.get("bio_label", "O"),
            "bio_probs": (
                f"{float(row.get('pO', 0.0)):.2f} / "
                f"{float(row.get('pB', 0.0)):.2f} / "
                f"{float(row.get('pI', 0.0)):.2f}"
            ),
            "threshold": f"{float(row.get('threshold', 0.0)):.2f}",
            "hand_guard": (
                f"ok={bool(row.get('hand_presence_ok', True))} • "
                f"blocked={bool(row.get('start_blocked_by_hand_guard', False))}"
            ),
            "warnings": ", ".join(str(flag) for flag in list(row.get("warnings", []) or [])) or "none",
        }
        tracking = {
            "left_valid": f"{int(row.get('left_valid_joints', 0))}/21 ({_format_ratio(float(row.get('left_valid_frac', 0.0)))})",
            "right_valid": f"{int(row.get('right_valid_joints', 0))}/21 ({_format_ratio(float(row.get('right_valid_frac', 0.0)))})",
            "total_valid": str(int(row.get("total_valid_hand_joints", 0))),
            "pose_valid": _format_ratio(float(row.get("pose_valid_frac", 0.0))),
            "slot_layout": "left 0:21 • right 21:42",
            "left_wrist": _format_xyz(seq.pts[frame_index, 0]),
            "right_wrist": _format_xyz(seq.pts[frame_index, 21]),
        }
        self._set_detail_values(self.frame_summary_labels, summary)
        self._set_detail_values(self.frame_bio_labels, bio)
        self._set_detail_values(self.frame_track_labels, tracking)
        payload = {
            "frame_index": int(frame_index),
            "ts_ms": float(row.get("ts_ms", 0.0)),
            "slot_layout": {"left": "0:21", "right": "21:42"},
            "bio": {
                "label": row.get("bio_label", "O"),
                "pO": float(row.get("pO", 0.0)),
                "pB": float(row.get("pB", 0.0)),
                "pI": float(row.get("pI", 0.0)),
                "threshold": float(row.get("threshold", 0.0)),
            },
            "segment": {
                "active_segment_id": row.get("active_segment_id"),
                "predicted_label": row.get("predicted_label", ""),
                "predicted_confidence": float(row.get("predicted_confidence", 0.0)),
                "family_label": row.get("family_label", ""),
                "family_confidence": float(row.get("family_confidence", 0.0)),
                "accepted": accepted,
                "reason": row.get("sentence_decision_reason", ""),
            },
            "hands": {
                "left_valid_joints": int(row.get("left_valid_joints", 0)),
                "right_valid_joints": int(row.get("right_valid_joints", 0)),
                "left_mask": seq.mask[frame_index, :21, 0].astype(int).tolist(),
                "right_mask": seq.mask[frame_index, 21:, 0].astype(int).tolist(),
                "left_wrist_xyz": seq.pts[frame_index, 0].astype(float).tolist(),
                "right_wrist_xyz": seq.pts[frame_index, 21].astype(float).tolist(),
            },
            "pose": {
                "enabled": bool(seq.pose_xyz is not None),
                "pose_indices": list(seq.pose_indices or []),
                "pose_valid_frac": float(row.get("pose_valid_frac", 0.0)),
                "pose_visibility": (seq.pose_vis[frame_index].astype(float).tolist() if seq.pose_vis is not None else []),
            },
            "warnings": list(row.get("warnings", []) or []),
        }
        self.frame_raw_text.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))

    def _apply_view_mode(self, mode: str) -> None:
        mode_norm = str(mode or "Split").strip().lower()
        show_original = mode_norm in {"split", "original"}
        show_overlay = mode_norm in {"split", "overlay"}
        self.original_group.setVisible(show_original)
        self.overlay_group.setVisible(show_overlay)

    def _refresh_current_frame(self) -> None:
        if self._session is not None:
            self._seek_frame(self._current_frame)

    def _timeline_zoom(self, action) -> None:
        action()
        self._update_timeline_status()

    def _update_timeline_status(self) -> None:
        if self._session is None:
            self.timeline_zoom_label.setText("Timeline: 1.0x")
            return
        start, end_excl = self.timeline_widget.visible_frame_range()
        self.timeline_zoom_label.setText(
            f"Timeline: {self.timeline_widget.zoom_factor:.1f}x • frames {start}:{max(start, end_excl - 1)}"
        )

    def _toggle_play(self) -> None:
        if self._session is None:
            return
        if self._timer.isActive():
            self._timer.stop()
            self.play_btn.setText("Play")
            return
        fps = max(1.0, float(self._session.fps or 30.0))
        self._timer.start(max(1, int(round(1000.0 / fps))))
        self.play_btn.setText("Pause")

    def _play_tick(self) -> None:
        if self._session is None:
            self._timer.stop()
            return
        if self._current_frame >= max(0, self._session.frame_count - 1):
            self._timer.stop()
            self.play_btn.setText("Play")
            return
        self._seek_frame(self._current_frame + 1)

    def _open_path(self, path: Path | None) -> None:
        if path is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_artifact(self, key: str) -> None:
        if self._session is None:
            return
        try:
            self._open_path(self._session.artifact_path(key))
        except Exception as exc:
            QMessageBox.warning(self, "Artifact missing", str(exc))

    def _copy_sentence(self) -> None:
        if self._session is None:
            return
        QApplication.clipboard().setText(self._session.sentence)

    def _close_video_sources(self) -> None:
        if self._original_video is not None:
            self._original_video.close()
        if self._overlay_video is not None:
            self._overlay_video.close()
        self._original_video = None
        self._overlay_video = None

    def closeEvent(self, event) -> None:  # pragma: no cover - GUI
        self._timer.stop()
        self._close_video_sources()
        if self._process is not None:
            self._process.kill()
            self._process = None
        super().closeEvent(event)


def run_app(argv: List[str] | None = None) -> None:  # pragma: no cover - GUI
    parser = argparse.ArgumentParser("python -m desktop_review")
    parser.add_argument("session", nargs="?", default="", help="Optional review session directory or session.json")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    app = QApplication(sys.argv if argv is None else ["desktop_review"])
    window = ReviewWindow()
    window.show()
    if args.session:
        window._load_session(args.session)
    app.exec()
