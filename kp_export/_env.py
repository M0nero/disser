
from __future__ import annotations
import os

# Caps for thread-heavy libs (set before importing cv2/mediapipe).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2  # noqa: E402

try:
    cv2.setNumThreads(1)
except Exception:
    pass
