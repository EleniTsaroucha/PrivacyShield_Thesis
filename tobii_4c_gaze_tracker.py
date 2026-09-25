import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional

from gaze_tracker import GazePoint

logger = logging.getLogger(__name__)

DEFAULT_STREAMER_PATH = Path("tobii_4c_streamer.exe")


class Tobii4CGazeTracker:
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        calibration_path: Optional[Path] = None,  # αγνοείται, συμβατότητα signature
        fovea_radius: int = 120,                   # αγνοείται εδώ
        streamer_path: Path = DEFAULT_STREAMER_PATH,
    ):
        self.screen_w = screen_width
        self.screen_h = screen_height

        streamer_path = Path(streamer_path)
        if not streamer_path.exists():
            raise RuntimeError(
                f"Not found {streamer_path}. You need to build. "
                "tobii_4c_streamer.cpp first (see CMakeLists.txt) and "
                "copy the .exe (along with tobii_stream_engine.dll) "
                "next to main.py, or give an explicit streamer_path."
            )

        self._lock = threading.Lock()
        self._latest_gaze = GazePoint(
            x=screen_width // 2, y=screen_height // 2, confidence=0.0,
            detection_valid=False, eyes_used=None, sample_age_ms=0.0,
        )

        self._proc = subprocess.Popen(
            [str(streamer_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )

        self._stop_event = threading.Event()
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

        
        got_data = threading.Event()
        original_callback_marker = self._latest_gaze

        def _watch():
            import time
            time.sleep(2.0)
            with self._lock:
                if self._latest_gaze is not original_callback_marker or \
                        self._latest_gaze.detection_valid:
                    got_data.set()

        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()
        watcher.join(timeout=2.5)

        stderr_preview = ""
        if self._proc.poll() is not None:
            # Η διεργασία τερμάτισε νωρίς — σίγουρα κάτι πήγε στραβά.
            stderr_preview = self._proc.stderr.read()[:1000]
            raise RuntimeError(
                "Ο tobii_4c_streamer τερμάτισε αμέσως. Έξοδος stderr:\n"
                f"{stderr_preview}"
            )

        logger.info("Tobii4CGazeTracker: streamer subprocess ενεργό (pid=%d).",
                     self._proc.pid)

    def _read_loop(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop_event.is_set():
                break
            line = line.strip()
            if not line:
                continue
            try:
                x_norm, y_norm, valid, _ts = line.split(",")
                valid = bool(int(valid))
                x_norm = float(x_norm)
                y_norm = float(y_norm)
            except ValueError:
                logger.warning("Tobii4CGazeTracker: μη αναμενόμενη γραμμή: %r", line)
                continue

            with self._lock:
                if valid:
                    self._latest_gaze = GazePoint(
                        x=int(round(x_norm * self.screen_w)),
                        y=int(round(y_norm * self.screen_h)),
                        confidence=1.0,
                        detection_valid=True,
                        eyes_used=None,  # ο 4C stream engine δεν διακρίνει ανά μάτι εδώ
                        sample_age_ms=0.0,
                    )
                else:
                    self._latest_gaze.confidence = 0.0
                    self._latest_gaze.detection_valid = False

    # Interface συμβατό με gaze_tracker.GazeTracker
    def process_frame(self, frame_bgr) -> GazePoint:
        with self._lock:
            return GazePoint(
                x=self._latest_gaze.x,
                y=self._latest_gaze.y,
                confidence=self._latest_gaze.confidence,
                detection_valid=self._latest_gaze.detection_valid,
                eyes_used=self._latest_gaze.eyes_used,
                sample_age_ms=self._latest_gaze.sample_age_ms,
            )

    def is_calibrated(self) -> bool:
        return True  # field-of-use interactive δεν έχει δικό του calibration wizard εδώ

    def reset_calibration(self) -> None:
        pass

    def add_calibration_sample(self, frame_bgr, screen_target) -> bool:
        return False

    def fit_calibration(self) -> bool:
        return False

    def save_calibration(self) -> None:
        pass

    def stop(self) -> None:
        self._stop_event.set()
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        logger.info("Tobii4CGazeTracker: streamer subprocess τερματίστηκε.")
