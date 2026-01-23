from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional


def try_import_mediapipe():
    import importlib
    mp = importlib.import_module("mediapipe")  # type: ignore
    mp_solutions = importlib.import_module("mediapipe.python.solutions")  # type: ignore
    return mp, mp_solutions


def normalize_backend_name(name: Optional[str]) -> str:
    if not name:
        return "solutions"
    name = name.strip().lower()
    if name not in {"solutions", "tasks"}:
        raise ValueError(f"Unsupported MediaPipe backend: {name}")
    return name


def normalize_tasks_delegate(delegate: Optional[str]) -> Optional[str]:
    if delegate is None:
        return None
    delegate = delegate.strip().lower()
    if not delegate or delegate == "auto":
        return None
    if delegate not in {"cpu", "gpu"}:
        raise ValueError(f"Unsupported MediaPipe Tasks delegate: {delegate}")
    return delegate


def resolve_task_model_path(candidate: Optional[str], filename: str) -> Optional[Path]:
    if candidate:
        p = Path(candidate).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"MediaPipe task model not found: {p}")
        return p
    default = Path(__file__).resolve().parent / "tasks" / filename
    if default.exists():
        return default
    return None


_ALIGN_HANDS_TO_POSE = True  # for world coords


class _TaskLandmarkList:
    def __init__(self, pts):
        self.landmark = pts


def _create_base_options(BaseOptions, model_path: Path, tasks_delegate: Optional[str]):
    kwargs = dict(model_asset_path=str(model_path))
    delegate = normalize_tasks_delegate(tasks_delegate)
    if delegate:
        delegate_enum = getattr(BaseOptions, "Delegate", None)
        if delegate_enum is None:
            raise RuntimeError(
                "Requested MediaPipe Tasks delegate is not available in the installed mediapipe version."
            )
        kwargs["delegate"] = delegate_enum.GPU if delegate == "gpu" else delegate_enum.CPU
    return BaseOptions(**kwargs)


def _wrap_landmark_lists(seq):
    if not seq:
        return None
    return [_TaskLandmarkList(lms) for lms in seq]


def _wrap_pose_landmarks(seq):
    if not seq:
        return None
    first = seq[0]
    if not first:
        return None
    return _TaskLandmarkList(first)


def _wrap_handedness(seq):
    if not seq:
        return None
    wrapped = []
    for entry in seq:
        classifications = []
        for cat in entry or []:
            label = str(getattr(cat, "category_name", "")).lower()
            score = float(getattr(cat, "score", 0.0))
            classifications.append(SimpleNamespace(label=label, score=score))
        wrapped.append(SimpleNamespace(classification=classifications))
    return wrapped or None


def _adapt_hand_result_from_tasks(result):
    return SimpleNamespace(
        multi_hand_landmarks=_wrap_landmark_lists(getattr(result, "hand_landmarks", None)),
        multi_hand_world_landmarks=_wrap_landmark_lists(getattr(result, "hand_world_landmarks", None)),
        multi_handedness=_wrap_handedness(getattr(result, "handedness", None)),
    )


def _adapt_pose_result_from_tasks(result):
    pose_landmarks = _wrap_pose_landmarks(getattr(result, "pose_landmarks", None))
    pose_world_landmarks = _wrap_pose_landmarks(getattr(result, "pose_world_landmarks", None))
    return SimpleNamespace(
        pose_landmarks=pose_landmarks,
        pose_world_landmarks=pose_world_landmarks,
    )


class _SolutionsDetector:
    def __init__(self, impl):
        self._impl = impl

    def process(self, rgb, timestamp_ms: Optional[int] = None):
        return self._impl.process(rgb)

    def close(self):
        close_fn = getattr(self._impl, "close", None)
        if callable(close_fn):
            close_fn()


class _TaskHandDetector:
    def __init__(self, mp, model_path: Path, running_mode: str,
                 max_num_hands: int, min_det: float, min_track: float,
                 world_coords: bool, tasks_delegate: Optional[str]):
        try:
            from mediapipe.tasks.python import vision, BaseOptions
        except ImportError as exc:
            raise ImportError("mediapipe.tasks is required for --mp-backend=tasks") from exc

        if model_path is None:
            raise FileNotFoundError("Hand task model path is required for MediaPipe Tasks backend")

        self._mp = mp
        self._vision = vision
        self._running_mode = vision.RunningMode.VIDEO if running_mode == "video" else vision.RunningMode.IMAGE
        base_options = _create_base_options(BaseOptions, model_path, tasks_delegate)
        options_kwargs = dict(
            base_options=base_options,
            running_mode=self._running_mode,
            num_hands=max(1, int(max_num_hands)),
            min_hand_detection_confidence=float(min_det),
            min_hand_presence_confidence=float(min_track),
            min_tracking_confidence=float(min_track),
        )
        annotations = getattr(vision.HandLandmarkerOptions, "__annotations__", {}) or {}
        if bool(world_coords) and "output_hand_world_landmarks" in annotations:
            options_kwargs["output_hand_world_landmarks"] = True
        options = vision.HandLandmarkerOptions(**options_kwargs)
        self._impl = vision.HandLandmarker.create_from_options(options)

    def process(self, rgb, timestamp_ms: Optional[int] = None):
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self._running_mode == self._vision.RunningMode.VIDEO:
            if timestamp_ms is None:
                raise ValueError("timestamp_ms is required for HandLandmarker VIDEO mode")
            result = self._impl.detect_for_video(image, int(timestamp_ms))
        else:
            result = self._impl.detect(image)
        return _adapt_hand_result_from_tasks(result)

    def close(self):
        close_fn = getattr(self._impl, "close", None)
        if callable(close_fn):
            close_fn()


class _TaskPoseDetector:
    def __init__(self, mp, model_path: Path, min_det: float, min_track: float,
                 tasks_delegate: Optional[str]):
        try:
            from mediapipe.tasks.python import vision, BaseOptions
        except ImportError as exc:
            raise ImportError("mediapipe.tasks is required for --mp-backend=tasks") from exc

        if model_path is None:
            raise FileNotFoundError("Pose task model path is required for MediaPipe Tasks backend")

        self._mp = mp
        self._vision = vision
        base_options = _create_base_options(BaseOptions, model_path, tasks_delegate)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=float(min_det),
            min_pose_presence_confidence=float(min_track),
            min_tracking_confidence=float(min_track),
            output_segmentation_masks=False,
        )
        self._impl = vision.PoseLandmarker.create_from_options(options)

    def process(self, rgb, timestamp_ms: Optional[int] = None):
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if timestamp_ms is None:
            raise ValueError("timestamp_ms is required for PoseLandmarker VIDEO mode")
        result = self._impl.detect_for_video(image, int(timestamp_ms))
        return _adapt_pose_result_from_tasks(result)

    def close(self):
        close_fn = getattr(self._impl, "close", None)
        if callable(close_fn):
            close_fn()


def create_hand_detector(backend: str, mp, mp_solutions,
                         *, max_num_hands: int,
                         min_det: float, min_track: float,
                         static_image_mode: bool = False,
                         model_path: Optional[Path] = None,
                         world_coords: bool = False,
                         tasks_delegate: Optional[str] = None):
    backend = normalize_backend_name(backend)
    if backend == "solutions":
        impl = mp_solutions.hands.Hands(
            static_image_mode=bool(static_image_mode),
            max_num_hands=int(max_num_hands),
            min_detection_confidence=float(min_det),
            min_tracking_confidence=float(min_track),
        )
        return _SolutionsDetector(impl)
    running_mode = "image" if static_image_mode else "video"
    return _TaskHandDetector(
        mp, model_path, running_mode,
        max_num_hands=max_num_hands,
        min_det=min_det,
        min_track=min_track,
        world_coords=world_coords,
        tasks_delegate=tasks_delegate,
    )


def create_pose_detector(backend: str, mp, mp_solutions,
                         *, model_complexity: int,
                         min_det: float, min_track: float,
                         enable_segmentation: bool,
                         model_path: Optional[Path] = None,
                         tasks_delegate: Optional[str] = None):
    backend = normalize_backend_name(backend)
    if backend == "solutions":
        impl = mp_solutions.pose.Pose(
            model_complexity=int(model_complexity),
            enable_segmentation=bool(enable_segmentation),
            min_detection_confidence=float(min_det),
            min_tracking_confidence=float(min_track),
        )
        return _SolutionsDetector(impl)
    return _TaskPoseDetector(mp, model_path, min_det=min_det, min_track=min_track,
                             tasks_delegate=tasks_delegate)


def align_hand_xy_to_target(hand_pts, target):
    try:
        if not hand_pts or target is None:
            return hand_pts
        dx = float(target.get('x', 0.0)) - float(hand_pts[0].get('x', 0.0))
        dy = float(target.get('y', 0.0)) - float(hand_pts[0].get('y', 0.0))
        return [dict(x=float(p.get('x', 0.0)) + dx,
                     y=float(p.get('y', 0.0)) + dy,
                     z=float(p.get('z', 0.0))) for p in hand_pts]
    except Exception:
        return hand_pts
