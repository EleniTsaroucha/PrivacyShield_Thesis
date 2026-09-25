import logging
import threading
import time
from pathlib import Path
from typing import Optional

from gaze_tracker import GazePoint

try:
    import tobii_research as tr
except ImportError as exc:
    raise ImportError(
        "Το πακέτο 'tobii-research' απαιτείται για το backend Tobii.\n"
        "Εγκατάστησέ το με: pip install tobii-research\n"
        "(απαιτεί Python 3.10 — δες BUILDING_VALIDATION_ENV.md)"
    ) from exc

logger = logging.getLogger(__name__)


def _select_tracker(
    serial_number: Optional[str] = None,
    model_contains: Optional[str] = None,
):
   
    trackers = tr.find_all_eyetrackers()
    if not trackers:
        raise RuntimeError(
            "Δεν βρέθηκε συνδεδεμένος Tobii eye tracker. Έλεγξε ότι:\n"
            "  1) Είναι συνδεδεμένος μέσω USB.\n"
            "  2) Το Tobii Pro Eye Tracker Manager τον αναγνωρίζει "
            "(χωρίς κίτρινο θαυμαστικό στη Διαχείριση Συσκευών).\n"
            "  3) Έχει ήδη γίνει βαθμονόμηση μέσα από το Eye Tracker "
            "Manager — αυτό το backend ΔΕΝ διαθέτει δικό του calibration "
            "wizard."
        )

    if serial_number:
        matches = [t for t in trackers if t.serial_number == serial_number]
        if not matches:
            found = ", ".join(f"{t.model} (serial={t.serial_number})" for t in trackers)
            raise RuntimeError(
                f"Δεν βρέθηκε tracker με serial_number='{serial_number}'.\n"
                f"Συνδεδεμένοι αυτή τη στιγμή: {found}"
            )
        return matches[0]

    if model_contains:
        needle = model_contains.lower()
        matches = [t for t in trackers if needle in t.model.lower()]
        if not matches:
            found = ", ".join(f"{t.model} (serial={t.serial_number})" for t in trackers)
            raise RuntimeError(
                f"Δεν βρέθηκε tracker με model που να περιέχει '{model_contains}'.\n"
                f"Συνδεδεμένοι αυτή τη στιγμή: {found}"
            )
        if len(matches) > 1:
            found = ", ".join(f"{t.model} (serial={t.serial_number})" for t in matches)
            raise RuntimeError(
                f"Πάνω από μία συσκευή ταιριάζει με model_contains="
                f"'{model_contains}': {found}\n"
                "Χρησιμοποίησε --tobii-serial για μονοσήμαντη επιλογή."
            )
        return matches[0]

    if len(trackers) > 1:
        found = ", ".join(f"{t.model} (serial={t.serial_number})" for t in trackers)
        raise RuntimeError(
            f"Βρέθηκαν {len(trackers)} συνδεδεμένοι Tobii eye trackers "
            f"ταυτόχρονα: {found}\n"
            "Χρειάζεται ρητή επιλογή — χρησιμοποίησε --tobii-serial SERIAL "
            "ή --tobii-model ΜΕΡΟΣ_ΟΝΟΜΑΤΟΣ (π.χ. --tobii-model IS4 για "
            "τον 4C, ή --tobii-model Spectrum)."
        )

    # Ακριβώς μία συσκευή συνδεδεμένη — καμία ασάφεια.
    return trackers[0]


class TobiiGazeTracker:
   
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        calibration_path: Optional[Path] = None,  # αγνοείται μόνο για συμβατότητα signature
        fovea_radius: int = 120,                   # αγνοείται εδώ το χρησιμοποιεί το OverlayWindow
        serial_number: Optional[str] = None,
        model_contains: Optional[str] = None,
    ):
        self.screen_w = screen_width
        self.screen_h = screen_height

        self._tracker = _select_tracker(
            serial_number=serial_number, model_contains=model_contains
        )
        logger.info(
            "TobiiGazeTracker: συνδέθηκε %s (serial=%s)",
            self._tracker.model, self._tracker.serial_number,
        )

        self._lock = threading.Lock()
        self._latest_gaze = GazePoint(
            x=screen_width // 2, y=screen_height // 2, confidence=0.0,
            detection_valid=False, eyes_used=None, sample_age_ms=0.0,
        )
       
        self._last_system_timestamp_us: Optional[int] = None

        self._received_any_data = threading.Event()

        self._tracker.subscribe_to(
            tr.EYETRACKER_GAZE_DATA, self._callback, as_dictionary=True
        )

        if not self._received_any_data.wait(timeout=2.0):
            self._tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self._callback)
            raise RuntimeError(
                f"Ο tracker '{self._tracker.model}' (serial="
                f"{self._tracker.serial_number}) βρέθηκε και έγινε subscribe, "
                "αλλά ΔΕΝ έφτασε κανένα gaze data μέσα σε 2 δευτερόλεπτα.\n\n"
                "Αυτό είναι το τυπικό σύμπτωμα ΑΝΕΠΑΡΚΟΥΣ LICENSE σε Tobii "
                "Eye Tracker 4C: το tobii_research SDK απαιτεί 'Pro Upgrade "
                "Key' για να στέλνει δεδομένα από 4C (η Tobii δεν πουλάει "
                "πλέον αυτό το κλειδί — βλ. Tobii Pro SDK licensing docs).\n"
                "Έλεγξε επίσης:\n"
                "  1) Ότι το Tobii Core Software / Tobii Experience τρέχει "
                "στο background (Windows 11 έχει περιορισμένη υποστήριξη — "
                "βλ. Tobii download page).\n"
                "  2) Ότι ο tracker φαίνεται ενεργός στο Tobii Pro Eye "
                "Tracker Manager, ΚΑΙ ότι το track status δείχνει μάτια, "
                "όχι μόνο 'connected'.\n"
                "  3) Αν διαθέτεις license file (.lic) για τον 4C, ότι "
                "είναι εγκατεστημένο στο σωστό path.\n"
                "Αν το πρόβλημα επιμένει μετά τους παραπάνω ελέγχους, ο 4C "
                "πιθανότατα δεν είναι διαθέσιμος ως backend σε αυτό το "
                "setup — τεκμηρίωσέ το ως περιορισμό (limitation) στη "
                "μεθοδολογία και χρησιμοποίησε το Spectrum backend."
            )
        logger.info(
            "TobiiGazeTracker: επιβεβαιώθηκε λήψη gaze data από %s.",
            self._tracker.model,
        )

    @property
    def model(self) -> str:
        return self._tracker.model

    @property
    def serial_number(self) -> str:
        return self._tracker.serial_number

    # Callback — τρέχει σε background thread διαχειριζόμενο από το SDK
    def _callback(self, gaze_data: dict) -> None:
        lx, ly = gaze_data["left_gaze_point_on_display_area"]
        rx, ry = gaze_data["right_gaze_point_on_display_area"]
        l_valid = gaze_data["left_gaze_point_validity"]
        r_valid = gaze_data["right_gaze_point_validity"]
        system_ts = gaze_data.get("system_time_stamp")
        self._received_any_data.set()

        points = []
        if l_valid:
            points.append((lx, ly))
        if r_valid:
            points.append((rx, ry))

        with self._lock:
            self._last_system_timestamp_us = system_ts

            if not points:
                self._latest_gaze.confidence = 0.0
                self._latest_gaze.detection_valid = False
                self._latest_gaze.eyes_used = 0
                return

            nx = sum(p[0] for p in points) / len(points)
            ny = sum(p[1] for p in points) / len(points)
            self._latest_gaze = GazePoint(
                x=int(round(nx * self.screen_w)),
                y=int(round(ny * self.screen_h)),
                confidence=1.0 if len(points) == 2 else 0.6,
                detection_valid=True,
                eyes_used=len(points),
                sample_age_ms=0.0,  # ενημερώνεται σωστά στο process_frame()
            )

    # Interface συμβατό με gaze_tracker.GazeTracker
    def process_frame(self, frame_bgr) -> GazePoint:
        with self._lock:
            sample_age_ms = 0.0
            if self._last_system_timestamp_us is not None:
                now_us = time.time() * 1_000_000.0
                sample_age_ms = max(
                    0.0, (now_us - self._last_system_timestamp_us) / 1000.0
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
            "is_calibrated(): no-op για το backend Tobii — η βαθμονόμηση "
            "γίνεται αποκλειστικά μέσα από το Tobii Pro Eye Tracker Manager."
        )
        return True

    def reset_calibration(self) -> None:
        logger.warning(
            "reset_calibration(): no-op για το backend Tobii — η βαθμονόμηση "
            "γίνεται αποκλειστικά μέσα από το Tobii Pro Eye Tracker Manager."
        )

    def add_calibration_sample(self, frame_bgr, screen_target) -> bool:
        logger.warning("add_calibration_sample(): no-op για το backend Tobii.")
        return False

    def fit_calibration(self) -> bool:
        logger.warning("fit_calibration(): no-op για το backend Tobii.")
        return False

    def save_calibration(self) -> None:
        pass

    def stop(self) -> None:
        self._tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self._callback)
        logger.info("TobiiGazeTracker: unsubscribe ολοκληρώθηκε.")
