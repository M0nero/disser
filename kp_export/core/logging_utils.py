from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Dict, Optional, Union
import os

PathLike = Union[str, Path]

__all__ = ["configure_logging", "get_logger", "log_metrics", "track_runtime"]

_LOGGER_NAME = "kp_export"
_DEFAULT_FILE_NAME = "kp_export.log"
_LOGGER_LOCK = Lock()
_LOGGER_CONFIG: Dict[str, Any] = {}
_NOISY_LOGGERS = ("absl", "mediapipe", "tensorflow", "tensorflow_hub", "matplotlib")


def _normalize_level(level: Union[int, str]) -> int:
    if isinstance(level, str):
        return logging._nameToLevel.get(level.upper(), logging.INFO)
    if isinstance(level, int):
        return level
    return logging.INFO


def configure_logging(
    log_dir: PathLike = "outputs/logs",
    level: Union[int, str] = "INFO",
    *,
    file_name: str = _DEFAULT_FILE_NAME,
    console: bool = True,
) -> logging.Logger:
    """
    Configure the shared kp_export logger so that all modules emit structured output
    into a rotating file under ``log_dir`` and, optionally, to stdout.

    ВАЖНО: под Windows при мультипроцессинге нельзя безопасно ротировать один и тот же файл
    из разных процессов. Поэтому по умолчанию лог-файл делается per-process:
    kp_export_<PID>.log, чтобы избежать WinError 32 при os.rename().
    """
    log_dir_path = Path(log_dir)

    # ---------- per-process лог-файл ----------
    base_file_name = file_name or _DEFAULT_FILE_NAME
    # только для дефолтного имени добавляем PID, явный file_name оставляем как есть
    if base_file_name == _DEFAULT_FILE_NAME:
        stem = Path(base_file_name).stem
        suffix = Path(base_file_name).suffix or ".log"
        per_process_name = f"{stem}_{os.getpid()}{suffix}"
    else:
        per_process_name = base_file_name

    file_path = log_dir_path / per_process_name
    # -----------------------------------------

    level_value = _normalize_level(level)
    config_key = {
        "log_dir": str(log_dir_path.resolve()),
        "file": str(file_path.name),
        "level": level_value,
        "console": bool(console),
    }

    with _LOGGER_LOCK:
        logger = logging.getLogger(_LOGGER_NAME)
        logger.setLevel(level_value)
        logger.propagate = False

        # если конфиг не менялся и хендлеры уже есть — просто вернуть существующий
        if _LOGGER_CONFIG == config_key and logger.handlers:
            return logger

        log_dir_path.mkdir(parents=True, exist_ok=True)
        logger.handlers.clear()

        # RotatingFileHandler на per-process файл безопасен
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(level_value)
        file_fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(processName)s[%(process)d] | %(name)s | %(message)s"
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

        if console:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(level_value)
            stream_fmt = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
            stream_handler.setFormatter(stream_fmt)
            logger.addHandler(stream_handler)

        for noisy_name in _NOISY_LOGGERS:
            noisy = logging.getLogger(noisy_name)
            noisy.setLevel(logging.ERROR)
            noisy.propagate = False

        _LOGGER_CONFIG.clear()
        _LOGGER_CONFIG.update(config_key)
        return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Helper to obtain a logger consistent with kp_export configuration.
    """
    return logging.getLogger(name or _LOGGER_NAME)


def _json_ready(value: Any) -> Any:
    from dataclasses import asdict, is_dataclass  # local import to avoid global dependency

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(v) for v in value]
    if is_dataclass(value):
        return _json_ready(asdict(value))
    return str(value)


def log_metrics(logger: Optional[logging.Logger], label: str, metrics: Dict[str, Any]) -> None:
    """
    Emit structured metrics that can be post-processed later.
    """
    target = logger or get_logger()
    payload = _json_ready(metrics)
    target.info("%s | %s", label, json.dumps(payload, ensure_ascii=False, sort_keys=True))


@contextmanager
def track_runtime(logger: Optional[logging.Logger], label: str, **metadata: Any):
    """
    Context manager that records the elapsed wall time for the wrapped block.
    """
    target = logger or get_logger()
    start = perf_counter()
    try:
        yield
    finally:
        duration = perf_counter() - start
        payload = dict(metadata)
        payload["duration_sec"] = round(duration, 6)
        log_metrics(target, f"{label}.duration", payload)
