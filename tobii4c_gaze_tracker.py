import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from gaze_tracker import GazePoint

logger = logging.getLogger(__name__)

DEFAULT_EXE_PATH = Path("tobii_bridge") / "build" / "Release" / "tobii_gaze_bridge.exe"


class Tobii4CGazeTracker:

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        calibration_path: Optional[Path] = None,  # αγνοείται· μόνο για συμβατότητα signature
        fovea_radius: int = 120,                   # αγνοείται εδώ
        exe_path: Optional[Path] = None,
        startup_timeout_s: float = 3.0,
    ):
        self.screen_w = screen_width
        self.screen_h = screen_height

        self._exe_path = Path(exe_path) if exe_path else DEFAULT_EXE_PATH
        if not self._exe_path.exists():
            raise RuntimeError(
                f"Δεν βρέθηκε το tobii_gaze_bridge.exe στο "
                f"'{self._exe_path}'.\n"
                "Έλεγξε ότι:\n"
                "  1) Έχεις κάνει build τον C++ wrapper "
                "(cmake --build . --config Release μέσα στο tobii_bridge/build).\n"
                "  2) Το path είναι σωστό — πέρασε ρητά με --tobii4c-exe "
                "PATH αν το build βρίσκεται αλλού."
            )

        self._lock = threading.Lock()
        self._latest_gaze = GazePoint(
            x=screen_width // 2, y=screen_height // 2, confidence=0.0,
            detection_valid=False, eyes_used=None, sample_age_ms=0.0,
        )
        self._last_sample_wall_time: Optional[float] = None
        # Χρονοσφραγίδα ΣΥΣΚΕΥΗΣ (Stream Engine clock) και αύξων αριθμός
        # δείγματος. Χωρίς αυτά είναι αδύνατο να ξεχωρίσει ένα νέο δείγμα από
        # την επανάληψη του τελευταίου γνωστού κατά το polling — με αποτέλεσμα
        # τεχνητά βελτιωμένη μετρούμενη ακρίβεια επανάληψης (precision) και
        # κατασκευασμένο ποσοστό απώλειας δεδομένων.
        self._last_device_ts_us: Optional[int] = None
        self._sample_seq: int = 0

        self._model = "unknown"
        self._serial_number = "unknown"

        self._received_any_data = threading.Event()
        self._device_info_received = threading.Event()
        self._stop_requested = threading.Event()

        logger.info("Tobii4CGazeTracker: εκκίνηση %s", self._exe_path)
        self._proc = subprocess.Popen(
            [str(self._exe_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )

        self._stdout_thread = threading.Thread(
            target=self._read_stdout, daemon=True, name="Tobii4CGazeTracker-stdout"
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, daemon=True, name="Tobii4CGazeTracker-stderr"
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    
        if not self._received_any_data.wait(timeout=startup_timeout_s):
            self.stop()
            raise RuntimeError(
                f"Το {self._exe_path.name} ξεκίνησε αλλά ΔΕΝ έφτασε κανένα "
                f"gaze sample μέσα σε {startup_timeout_s:.1f}s.\n\n"
                "Πιθανές αιτίες:\n"
                "  1) Ο 4C δεν είναι συνδεδεμένος ή δεν αναγνωρίζεται "
                "(έλεγξε στο Tobii Eye Tracking Core / Device Manager).\n"
                "  2) Δεν έχει γίνει calibration μέσω του Tobii Eye "
                "Tracking Core software — το Stream Engine SDK ΔΕΝ διαθέτει "
                "δικό του calibration wizard, σε αντίθεση με τον webcam "
                "backend αυτής της εφαρμογής.\n"
                "  3) Πρόβλημα στο ίδιο το bridge — έλεγξε τα stderr logs "
                "παραπάνω σε αυτό το terminal για μηνύματα ERROR."
            )
        logger.info(
            "Tobii4CGazeTracker: επιβεβαιώθηκε λήψη gaze data (%s, serial=%s).",
            self._model, self._serial_number,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def serial_number(self) -> str:
        return self._serial_number

    @property
    def last_device_timestamp_us(self) -> Optional[int]:
        with self._lock:
            return self._last_device_ts_us

    @property
    def sample_seq(self) -> int:
        """Αύξων αριθμός του τελευταίου δείγματος που έφτασε από το bridge.

        Διαδοχικές κλήσεις process_frame() που επιστρέφουν την ΙΔΙΑ τιμή
        αντιστοιχούν στο ίδιο φυσικό δείγμα. Κατέγραψέ τον και φίλτραρε τα
        διπλότυπα πριν υπολογίσεις ακρίβεια ή απώλεια δεδομένων.
        """
        with self._lock:
            return self._sample_seq

    # Background thread: διαβάζει stdout γραμμή-γραμμή
    def _read_stdout(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop_requested.is_set():
                break
            line = line.rstrip("\r\n")
            if not line:
                continue

            if line.startswith("READY\t"):
                self._parse_ready_line(line)
                continue

            if line.startswith("GAZE\t"):
                self._parse_gaze_line(line)
                continue

            if line.startswith("ERROR\t"):
                logger.error("tobii_gaze_bridge: %s", line[len("ERROR\t"):])
                continue

            logger.warning("Tobii4CGazeTracker: μη αναμενόμενη γραμμή: %r", line)

        logger.info("Tobii4CGazeTracker: stdout thread τερματίστηκε.")

    def _parse_ready_line(self, line: str) -> None:
        # Μορφή (βλ. tobii_gaze_bridge.cpp): READY\t<model>\t<serial>
        parts = line.split("\t")
        if len(parts) == 3:
            self._model = parts[1] or "unknown"
            self._serial_number = parts[2] or "unknown"
        self._device_info_received.set()

    def _parse_gaze_line(self, line: str) -> None:
        # Μορφή (βλ. tobii_gaze_bridge.cpp gaze_point_callback):
        #   GAZE\t<timestamp_us>\t<x>\t<y>\t<valid 0|1>
        parts = line.split("\t")
        if len(parts) != 5:
            logger.warning("Tobii4CGazeTracker: μη αναμενόμενη γραμμή GAZE: %r", line)
            return

        _tag, ts_str, x_str, y_str, valid_str = parts
        self._received_any_data.set()

        valid = valid_str == "1"

        try:
            device_ts_us = int(ts_str)
        except ValueError:
            device_ts_us = None

        with self._lock:
            self._last_sample_wall_time = time.time()
            self._last_device_ts_us = device_ts_us
            self._sample_seq += 1

            if not valid or not x_str or not y_str:
                self._latest_gaze.confidence = 0.0
                self._latest_gaze.detection_valid = False
                self._latest_gaze.eyes_used = 0
                return

            try:
                nx = float(x_str)
                ny = float(y_str)
            except ValueError:
                logger.warning("Tobii4CGazeTracker: μη έγκυρες συντεταγμένες: %r", line)
                return

            self._latest_gaze = GazePoint(
                x=int(round(nx * self.screen_w)),
                y=int(round(ny * self.screen_h)),
                confidence=1.0,
                detection_valid=True,
                eyes_used=2,  
                sample_age_ms=0.0,  # ενημερώνεται σωστά στο process_frame()
            )

    def _read_stderr(self) -> None:
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            line = line.strip()
            if not line:
                continue
            if line.startswith("ERROR:"):
                logger.error("tobii_gaze_bridge: %s", line)
            else:
                logger.info("tobii_gaze_bridge: %s", line)
        logger.info("Tobii4CGazeTracker: stderr thread τερματίστηκε.")

    # Interface συμβατό με gaze_tracker.GazeTracker
    def process_frame(self, frame_bgr) -> GazePoint:
        with self._lock:
            sample_age_ms = 0.0
            if self._last_sample_wall_time is not None:
                sample_age_ms = max(
                    0.0, (time.time() - self._last_sample_wall_time) * 1000.0
                )
            return GazePoint(
                x=self._latest_gaze.x,
                y=self._latest_gaze.y,
                confidence=self._latest_gaze.confidence,
                detection_valid=self._latest_gaze.detection_valid,
                eyes_used=self._latest_gaze.eyes_used,
                sample_age_ms=sample_age_ms,
            )

    def is_calibrated(self) -> bool:
        logger.warning(
            "is_calibrated(): no-op για το backend tobii4c — η βαθμονόμηση "
            "γίνεται αποκλειστικά μέσα από το Tobii Eye Tracking Core."
        )
        return True

    def reset_calibration(self) -> None:
        logger.warning("reset_calibration(): no-op για το backend tobii4c.")

    def add_calibration_sample(self, frame_bgr, screen_target) -> bool:
        logger.warning("add_calibration_sample(): no-op για το backend tobii4c.")
        return False

    def fit_calibration(self) -> bool:
        logger.warning("fit_calibration(): no-op για το backend tobii4c.")
        return False

    def save_calibration(self) -> None:
        pass

    def stop(self) -> None:
        self._stop_requested.set()
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Tobii4CGazeTracker: το exe δεν τερμάτισε καθαρά, "
                    "γίνεται kill()."
                )
                self._proc.kill()
        logger.info("Tobii4CGazeTracker: υποδιεργασία τερματίστηκε.")
