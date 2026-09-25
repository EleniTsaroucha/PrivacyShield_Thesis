import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError("mediapipe is required: pip install mediapipe==0.10.14") from exc

CALIBRATION_PATH = Path("gaze_calibration.json")
EMA_ALPHA        = 0.6
DEFAULT_RADIUS   = 300

# MediaPipe iris landmark indices
_RIGHT_IRIS = [468, 469, 470, 471, 472]
_LEFT_IRIS  = [473, 474, 475, 476, 477]

# MediaPipe eye-socket landmarks (corners + lids), used to normalise iris
# position RELATIVE to the eye socket instead of relative to the whole
# camera frame. This isolates actual eyeball rotation from head translation
# — see _get_iris_normalised() for the rationale.
_RIGHT_EYE_CORNERS = (33, 133)   # (outer, inner)
_RIGHT_EYE_LIDS    = (159, 145)  # (upper, lower)
_LEFT_EYE_CORNERS  = (263, 362)  # (outer, inner)
_LEFT_EYE_LIDS     = (386, 374)  # (upper, lower)


@dataclass
class GazePoint:
    x: int   = 0
    y: int   = 0
    confidence: float = 0.0


    detection_valid: bool = False
    eyes_used: Optional[int] = None
    sample_age_ms: float = 0.0


@dataclass
class CalibrationData:
    iris_pts:   list = field(default_factory=list)
    screen_pts: list = field(default_factory=list)
    A: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None

    def is_fitted(self) -> bool:
        return self.A is not None

    def add_sample(self, iris: Tuple[float, float], screen: Tuple[int, int]) -> None:
        self.iris_pts.append(list(iris))
        self.screen_pts.append(list(screen))

    def fit(self) -> bool:
        if len(self.iris_pts) < 4:
            logger.warning("Need ≥4 calibration points; have %d.", len(self.iris_pts))
            return False
        X = np.array(self.iris_pts,   dtype=np.float64)
        Y = np.array(self.screen_pts, dtype=np.float64)
        X_aug = np.hstack([X, np.ones((len(X), 1))])
        W, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)
        self.A = W[:2].T
        self.b = W[2]
        logger.info("Calibration fitted with %d points.", len(self.iris_pts))
        return True

    def predict(self, iris: Tuple[float, float]) -> Tuple[int, int]:
        v = np.array(iris, dtype=np.float64)
        s = self.A @ v + self.b
        return int(np.clip(s[0], 0, 99999)), int(np.clip(s[1], 0, 99999))

    def to_json(self) -> dict:
        return {
            "iris_pts":   self.iris_pts,
            "screen_pts": self.screen_pts,
            "A": self.A.tolist() if self.A is not None else None,
            "b": self.b.tolist() if self.b is not None else None,
        }

    @classmethod
    def from_json(cls, d: dict) -> "CalibrationData":
        obj = cls(iris_pts=d["iris_pts"], screen_pts=d["screen_pts"])
        if d["A"] is not None:
            obj.A = np.array(d["A"])
            obj.b = np.array(d["b"])
        return obj


class GazeTracker:

    def __init__(
        self,
        screen_width:  int,
        screen_height: int,
        calibration_path: Path = CALIBRATION_PATH,
        fovea_radius: int = DEFAULT_RADIUS,
    ):
        self.screen_w     = screen_width
        self.screen_h     = screen_height
        self.fovea_radius = fovea_radius
        self._calibration_path = calibration_path

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,          # enables iris landmarks 468-477
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self._calib = CalibrationData()
        self._load_calibration()

        self._lock  = threading.Lock()
        self._gaze  = GazePoint()
        self._ema_x = float(screen_width  / 2)
        self._ema_y = float(screen_height / 2)

    # Calibration
    def _load_calibration(self) -> None:
        if not self._calibration_path.exists():
            logger.info("No calibration file. Uncalibrated mode active.")
            return
        with open(self._calibration_path) as fh:
            self._calib = CalibrationData.from_json(json.load(fh))
        logger.info("Calibration loaded.")

    def save_calibration(self) -> None:
        with open(self._calibration_path, "w") as fh:
            json.dump(self._calib.to_json(), fh, indent=2)
        logger.info("Calibration saved.")

    def add_calibration_sample(
        self, frame_bgr: np.ndarray, screen_target: Tuple[int, int]
    ) -> bool:
        iris = self._get_iris_normalised(frame_bgr)
        if iris is None:
            return False
        self._calib.add_sample(iris, screen_target)
        return True

    def fit_calibration(self) -> bool:
        ok = self._calib.fit()
        if ok:
            self.save_calibration()
        return ok

    def reset_calibration(self) -> None:
        self._calib = CalibrationData()
        if self._calibration_path.exists():
            self._calibration_path.unlink()

    def is_calibrated(self) -> bool:

        return self._calib.is_fitted()

    # Core processing
    def process_frame(self, frame_bgr: np.ndarray) -> GazePoint:
        iris = self._get_iris_normalised(frame_bgr)

        if iris is None:
            with self._lock:
                self._gaze.confidence = 0.0
                self._gaze.detection_valid = False
                # eyes_used/sample_age_ms παραμένουν στις προεπιλογές τους
                # (None / 0.0 αντίστοιχα) — δεν εφαρμόζονται σε αυτό το backend.
            return self._get_gaze()

        nx, ny = iris
        if self._calib.is_fitted():
            sx, sy = self._calib.predict((nx, ny))
        else:
            # Uncalibrated: linear mapping (workable for demo)
            sx = int(nx * self.screen_w)
            sy = int(ny * self.screen_h)

        self._ema_x = (1 - EMA_ALPHA) * self._ema_x + EMA_ALPHA * sx
        self._ema_y = (1 - EMA_ALPHA) * self._ema_y + EMA_ALPHA * sy

        with self._lock:
            self._gaze.x = int(self._ema_x)
            self._gaze.y = int(self._ema_y)
            self._gaze.confidence = 1.0
            self._gaze.detection_valid = True
            self._gaze.sample_age_ms = 0.0

        return self._get_gaze()

    def _get_gaze(self) -> GazePoint:
        with self._lock:
            return GazePoint(
                x=self._gaze.x, y=self._gaze.y, confidence=self._gaze.confidence,
                detection_valid=self._gaze.detection_valid,
                eyes_used=self._gaze.eyes_used,
                sample_age_ms=self._gaze.sample_age_ms,
            )

    # MediaPipe iris extraction
    def _get_iris_normalised(
        self, frame_bgr: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        lm   = results.multi_face_landmarks[0].landmark
        h, w = frame_bgr.shape[:2]

        def centroid(indices):
            xs = [lm[i].x * w for i in indices]
            ys = [lm[i].y * h for i in indices]
            return float(np.mean(xs)), float(np.mean(ys))

        def eye_relative_position(iris_indices, corner_indices, lid_indices):
            
            ix, iy = centroid(iris_indices)

            c0x, c0y = lm[corner_indices[0]].x * w, lm[corner_indices[0]].y * h
            c1x, c1y = lm[corner_indices[1]].x * w, lm[corner_indices[1]].y * h
            x_min, x_max = min(c0x, c1x), max(c0x, c1x)

            l0y = lm[lid_indices[0]].y * h
            l1y = lm[lid_indices[1]].y * h
            y_min, y_max = min(l0y, l1y), max(l0y, l1y)

            eye_w = x_max - x_min
            eye_h = y_max - y_min
            if eye_w < 1e-3 or eye_h < 1e-3:
                return None

            rx = (ix - x_min) / eye_w
            ry = (iy - y_min) / eye_h
            return float(np.clip(rx, 0.0, 1.0)), float(np.clip(ry, 0.0, 1.0))

        right = eye_relative_position(_RIGHT_IRIS, _RIGHT_EYE_CORNERS, _RIGHT_EYE_LIDS)
        left  = eye_relative_position(_LEFT_IRIS, _LEFT_EYE_CORNERS, _LEFT_EYE_LIDS)

        candidates = [p for p in (right, left) if p is not None]
        if not candidates:
            return None

        mid_x = float(np.mean([p[0] for p in candidates]))
        mid_y = float(np.mean([p[1] for p in candidates]))

        return mid_x, mid_y
