
from __future__ import annotations

import os

# Suppress noisy third-party native logs before mediapipe/tflite are imported.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

__all__ = ["process", "parallel", "core", "algos", "mp", "annotations"]
__version__ = "0.1.0"
