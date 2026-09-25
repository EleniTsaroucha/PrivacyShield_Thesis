import argparse
import atexit
import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from face_auth import FaceAuthenticator, THUMB_DIR
from gaze_tracker import GazeTracker, GazePoint
from screen_lock import ScreenLocker
from experiment_logger import ExperimentLogger
from video_recorder import VideoRecorder

# PyQt6
try:
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint, QRect
    from PyQt6.QtGui import (
        QPainter, QColor, QBrush, QPen, QFont, QRadialGradient, QPixmap,
        QIcon, QAction, QGuiApplication,
    )
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QMessageBox, QInputDialog,
        QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
        QSystemTrayIcon, QMenu, QRadioButton, QButtonGroup, QFrame,
        QSlider, QCheckBox, QLineEdit, QComboBox, QFormLayout,
    )
except ImportError as exc:
    raise ImportError(
        "PyQt6 is required.  Install with:\n"
        "  pip install PyQt6"
    ) from exc


try:
    from tobii_gaze_tracker import TobiiGazeTracker
    _TOBII_AVAILABLE = True
except ImportError:
    _TOBII_AVAILABLE = False

try:
    from tobii4c_gaze_tracker import Tobii4CGazeTracker
    _TOBII4C_AVAILABLE = True
except ImportError:
    _TOBII4C_AVAILABLE = False

# Καταγραφή αναφοράς από το Spectrum ΜΕΣΑ στην ίδια διεργασία. Αντικαθιστά το
# ξεχωριστό main3.py / dual_tobii_recorder: δύο διεργασίες δεν μπορούν να
# ανοίξουν ταυτόχρονα τον 4C μέσω Stream Engine, ενώ εδώ το Spectrum
# καταγράφεται παθητικά και σε κοινό ρολόι με τα δεδομένα του backend.
try:
    from spectrum_reference import SpectrumReferenceRecorder
    _SPECTRUM_REF_AVAILABLE = True
except ImportError:
    _SPECTRUM_REF_AVAILABLE = False

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


APP_BUILD = "logging-v2"

BACKEND_CANONICAL = {
    "webcam": "webcam_mediapipe",
    "tobii4c": "tobii_4c",
    "tobii": "tobii_spectrum",
}


# Εκπομπή δείκτη συγχρονισμού προς εξωτερικό σύστημα καταγραφής
SYNC_KEY_VK = {
    "F1": 0x70, "F2": 0x71, "F3": 0x72, "F4": 0x73, "F5": 0x74, "F6": 0x75,
    "F7": 0x76, "F8": 0x77, "F9": 0x78, "F10": 0x79, "F11": 0x7A, "F12": 0x7B,
    "SCROLLLOCK": 0x91, "PAUSE": 0x13,
}


def make_sync_keystroke_hook(key_name: str):
    """
    Επιστρέφει callable που πατάει ένα πλήκτρο, ώστε το Tobii Pro Lab να
    καταγράψει KeyboardEvent με τη ΔΙΚΗ ΤΟΥ χρονοσφραγίδα.

    Ο ίδιος δείκτης υπάρχει τότε και στα δύο αρχεία και η σταθερή απόκλιση
    των δύο ρολογιών γίνεται μετρήσιμη. Χωρίς αυτό, η ευθυγράμμιση στηρίζεται
    μόνο στο 'Recording start time' του Pro Lab, που έχει ανάλυση χιλιοστού
    και ανεξακρίβωτη απόκλιση από το ρολόι της εφαρμογής.

    Επιστρέφει None αν το κανάλι δεν είναι διαθέσιμο στην πλατφόρμα.
    """
    key = (key_name or "").upper()
    vk = SYNC_KEY_VK.get(key)
    if vk is None:
        logger.error(
            "Άγνωστο πλήκτρο συγχρονισμού '%s'. Διαθέσιμα: %s. "
            "Η αυτόματη εκπομπή απενεργοποιείται.",
            key_name, ", ".join(sorted(SYNC_KEY_VK)),
        )
        return None
    if platform.system() != "Windows":
        logger.warning(
            "Η αυτόματη εκπομπή δείκτη υλοποιείται μόνο σε Windows. "
            "Οι δείκτες θα καταγράφονται μόνο τοπικά."
        )
        return None

    import ctypes
    user32 = ctypes.windll.user32
    KEYEVENTF_KEYUP = 0x0002

    def _emit() -> None:
        user32.keybd_event(vk, 0, 0, 0)
        user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)

    logger.info(
        "Δείκτες συγχρονισμού: αυτόματη εκπομπή πλήκτρου %s προς το Pro Lab.", key,
    )
    return _emit


# Enums / States
class PrivacyState:
    NO_FACE      = "NO_FACE"       # no face detected – full dark mask
    CLEAR        = "CLEAR"         # only known faces – screen fully visible
    PANIC        = "PANIC"         # unknown observer – foveated blur + warning
    HARD_LOCK    = "HARD_LOCK"     # prolonged absence – full input block until
                                    # the trusted user's face is seen again


# Camera + AI Thread
class CameraThread(QThread):


    state_updated = pyqtSignal(str, int, int, float, int, float)
    # args: (PrivacyState, gaze_x, gaze_y, gaze_confidence,
    #        camera_frame_index, t_capture_perf)
    # Τα δύο τελευταία μεταφέρουν τη στιγμή σύλληψης του καρέ στο νήμα GUI,
    # ώστε το overlay να μπορεί να μετρήσει καθυστέρηση έως τη σχεδίαση.

    def __init__(
        self,
        authenticator: FaceAuthenticator,
        tracker: GazeTracker,
        camera_index: int = 0,
        target_fps: int = 20,
        lock_timeout: float = 5.0,
        panic_hold_s: float = 4.0,
        no_face_threshold: int = 5,
        exp_logger: "ExperimentLogger | None" = None,
        video_recorder: "VideoRecorder | None" = None,
        gaze_always: bool = True,
        frame_sync_hook=None,
    ):
        super().__init__()
        self._auth = authenticator
        self._tracker = tracker
        self._camera_index = camera_index
        self._target_fps = target_fps
        self._running = False
        self._exp_logger = exp_logger
        self._video_recorder = video_recorder
        # Callable(camera_frame_index) -> None. Στέλνει στον recorder αναφοράς
        # τη στιγμή του καρέ στο ΚΟΙΝΟ ρολόι, ώστε οι δύο ροές να ενώνονται
        # εκ των υστέρων με join στο camera_frame_index.
        self._frame_sync_hook = frame_sync_hook
        # Η εκτίμηση βλέμματος πρέπει να τρέχει σε ΟΛΕΣ τις καταστάσεις όταν
        # συγκρίνουμε συστήματα βλέμματος: αν τρέχει μόνο σε PANIC, δεν
        # υπάρχουν δεδομένα για τα διαστήματα κανονικής χρήσης.
        self._gaze_always = bool(gaze_always)


        self._AUTH_EVERY_N = 2     # κάθε 2 frames — πολύ πιο responsive

        # Hysteresis για NO_FACE (blink protection)
        self._NO_FACE_THRESHOLD = no_face_threshold
        self._no_face_counter   = 0
        self._last_state        = PrivacyState.NO_FACE
        self._PANIC_HOLD_S      = max(0.0, float(panic_hold_s))
        self._panic_hold_until  = None   # perf_counter() deadline ή None


        self._locker            = ScreenLocker()
        self._lock_timeout      = lock_timeout   # δευτερόλεπτα· 0 ή None = disabled
        self._absent_since      = None           # timestamp έναρξης συνεχούς απουσίας
        self._hard_locked       = False          # True => HARD_LOCK ενεργό

    # PANIC hold helpers
    def _arm_panic_hold(self, now: float) -> None:
        """(Επαν)οπλίζει το παράθυρο παράτασης με αφετηρία το τρέχον frame."""
        self._panic_hold_until = now + self._PANIC_HOLD_S

    def _panic_hold_active(self, now: float) -> bool:
        if self._panic_hold_until is None:
            return False
        if now < self._panic_hold_until:
            return True
        self._panic_hold_until = None
        return False

    def _panic_hold_remaining_ms(self, now: float) -> float:
        if self._panic_hold_until is None:
            return 0.0
        return max(0.0, (self._panic_hold_until - now) * 1000.0)

    @staticmethod
    def _open_camera(camera_index: int) -> "cv2.VideoCapture":
        
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            logger.warning(
                "Άνοιγμα κάμερας %d με backend %s απέτυχε· δοκιμή με "
                "default backend.", camera_index, backend,
            )
            cap = cv2.VideoCapture(camera_index)
        return cap

    def run(self) -> None:
        self._running = True
        cap = self._open_camera(self._camera_index)
        if not cap.isOpened():
            logger.error("Cannot open camera %d.", self._camera_index)
            self.state_updated.emit(PrivacyState.PANIC, 0, 0, 0.0, -1, 0.0)
            return

        # Self-test
        test_ok = False
        for attempt in range(10):
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                test_ok = True
                h, w = test_frame.shape[:2]
                logger.info(
                    "Κάμερα %d: πρώτο frame OK (%dx%d, backend=%s).",
                    self._camera_index, w, h, cap.getBackendName(),
                )
                break
            time.sleep(0.1)
        if not test_ok:
            logger.error(
                "Κάμερα %d: άνοιξε αλλά δεν επέστρεψε έγκυρο frame μετά "
                "από 10 προσπάθειες. Πιθανά αίτια: η συσκευή "
                "χρησιμοποιείται ήδη από άλλη εφαρμογή, λάθος camera "
                "index, ή προβληματικός driver/backend.",
                self._camera_index,
            )
            self.state_updated.emit(PrivacyState.PANIC, 0, 0, 0.0, -1, 0.0)
            cap.release()
            return

        frame_duration = 1.0 / self._target_fps
        frame_count    = 0
        last_auth      = None   # cached AuthResult between auth frames
        _last_heartbeat = time.perf_counter()
        _HEARTBEAT_S    = 3.0

        try:
            while self._running:
                t0 = time.perf_counter()
                ret, frame = cap.read()
                # Στιγμή σύλληψης του καρέ: αφετηρία ΚΑΘΕ μέτρησης καθυστέρησης.
                t_capture = time.perf_counter()
                if not ret:
                    logger.warning("Frame grab failed; defaulting to PANIC.")
                    self.state_updated.emit(PrivacyState.PANIC, 0, 0, 0.0, -1, 0.0)
                    time.sleep(0.1)
                    continue

                frame_count += 1
                t_inference_done = None

                if self._frame_sync_hook is not None:
                    try:
                        self._frame_sync_hook(frame_count)
                    except Exception:
                        logger.exception(
                            "frame_sync_hook απέτυχε στο frame %d — η καταγραφή "
                            "συνεχίζεται.", frame_count,
                        )

                # Face authentication (every N frames)
                if frame_count % self._AUTH_EVERY_N == 1 or last_auth is None:
                    try:
                        last_auth = self._auth.analyse_frame(frame)
                        t_inference_done = time.perf_counter()
                    except Exception:
                        logger.exception("FaceAuth error; defaulting to PANIC.")
                        self.state_updated.emit(
                            PrivacyState.PANIC, 0, 0, 0.0, frame_count, t_capture)
                        continue

                auth_result = last_auth
                prev_state  = self._last_state

                # Hard-lock
                now = time.perf_counter()
                if self._lock_timeout and self._lock_timeout > 0:
                    if auth_result.total_faces == 0:
                        if self._absent_since is None:
                            self._absent_since = now
                        elapsed_absent = now - self._absent_since
                        if elapsed_absent >= self._lock_timeout and not self._hard_locked:
                            self._hard_locked = True
                            logger.warning(
                                "Απουσία χρήστη για %.1f δευτ. (όριο %.1f) — "
                                "HARD_LOCK ενεργό μέχρι επανεμφάνιση του "
                                "εγγεγραμμένου χρήστη.",
                                elapsed_absent, self._lock_timeout,
                            )
                            if self._exp_logger:
                                self._exp_logger.log_event(
                                    "HARD_LOCK_ENTER", from_state=prev_state,
                                    to_state=PrivacyState.HARD_LOCK,
                                    camera_frame_index=frame_count,
                                    note=f"absent_s={elapsed_absent:.2f}",
                                )
                            self._locker.lock()
                    else:
                        # Εντοπίστηκε πρόσωπο. Reset absence timer πάντα.
                        self._absent_since = None
                        if self._hard_locked and auth_result.only_trusted_present:
                            self._hard_locked = False
                            logger.info(
                                "Ο εγγεγραμμένος χρήστης αναγνωρίστηκε ξανά "
                                "— HARD_LOCK απενεργοποιήθηκε."
                            )
                            if self._exp_logger:
                                self._exp_logger.log_event(
                                    "HARD_LOCK_EXIT", from_state=PrivacyState.HARD_LOCK,
                                    to_state=prev_state, camera_frame_index=frame_count,
                                )

                if self._hard_locked:
                    state = PrivacyState.HARD_LOCK
                    self._last_state = state
                    if self._exp_logger:
                        loop_dt_ms = (time.perf_counter() - t0) * 1000.0
                        self._exp_logger.log_frame(
                            camera_frame_index=frame_count,
                            t_capture_perf=t_capture,
                            state=state, prev_state=prev_state,
                            auth_result=auth_result, gaze=GazePoint(),
                            no_face_counter=self._no_face_counter,
                            panic_hold_remaining_ms=self._panic_hold_remaining_ms(now),
                            hard_locked=True,
                            t_inference_done_perf=t_inference_done,
                            t_state_change_perf=(
                                time.perf_counter() if state != prev_state else None),
                            loop_dt_ms=loop_dt_ms,
                        )
                    if self._video_recorder:
                        clock = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        try:
                            self._video_recorder.write_frame(
                                frame, overlay_text=f"{clock}  f{frame_count}  HARD_LOCK",
                            )
                        except Exception:
                            logger.exception(
                                "VideoRecorder.write_frame απέτυχε στο frame %d.",
                                frame_count,
                            )
                    if time.perf_counter() - _last_heartbeat >= _HEARTBEAT_S:
                        logger.info(
                            "Heartbeat: %d frames επεξεργάστηκαν (state=%s).",
                            frame_count, state,
                        )
                        _last_heartbeat = time.perf_counter()
                    self.state_updated.emit(state, 0, 0, 0.0, frame_count, t_capture)
                    elapsed = time.perf_counter() - t0
                    sleep_t = frame_duration - elapsed
                    if sleep_t > 0:
                        time.sleep(sleep_t)
                    continue

                # Determine privacy state με χρονικό PANIC hold
                hold_active = self._panic_hold_active(now)
                if auth_result.total_faces == 0:
                    self._no_face_counter += 1
                    if hold_active:
                        # Είδαμε άγνωστο εντός του παραθύρου — κρατάμε PANIC
                        state = PrivacyState.PANIC
                    elif self._no_face_counter >= self._NO_FACE_THRESHOLD:
                        state = PrivacyState.NO_FACE
                    else:
                        state = self._last_state
                elif not auth_result.only_trusted_present:
                    self._no_face_counter = 0
                    self._arm_panic_hold(now)
                    state = PrivacyState.PANIC
                else:
                    # Μόνο trusted — CLEAR μόνο αφού λήξει το παράθυρο παράτασης
                    self._no_face_counter = 0
                    state = PrivacyState.PANIC if hold_active else PrivacyState.CLEAR

                state_changed = (state != self._last_state)
                t_state_change = time.perf_counter() if state_changed else None
                self._last_state = state

                # Εκτίμηση βλέμματος.
                # ΠΡΟΣΟΧΗ: στην v1 έτρεχε ΜΟΝΟ σε PANIC, οπότε δεν υπήρχαν
                # δεδομένα βλέμματος για τα διαστήματα CLEAR — δηλαδή για το
                # μεγαλύτερο μέρος της συνεδρίας. Για τη σύγκριση των τριών
                # συστημάτων με τον Spectrum απαιτείται συνεχής εκτίμηση.
                gaze = GazePoint()
                if self._gaze_always or state == PrivacyState.PANIC:
                    try:
                        gaze = self._tracker.process_frame(frame)
                    except Exception:
                        logger.exception("GazeTracker error; using centre fallback.")
                        gaze.confidence = 0.0

                if self._exp_logger:
                    loop_dt_ms = (time.perf_counter() - t0) * 1000.0
                    self._exp_logger.log_frame(
                        camera_frame_index=frame_count,
                        t_capture_perf=t_capture,
                        state=state, prev_state=prev_state,
                        auth_result=auth_result, gaze=gaze,
                        no_face_counter=self._no_face_counter,
                        panic_hold_remaining_ms=self._panic_hold_remaining_ms(
                            time.perf_counter()),
                        hard_locked=False,
                        t_inference_done_perf=t_inference_done,
                        t_state_change_perf=t_state_change,
                        loop_dt_ms=loop_dt_ms,
                    )
                    if state != prev_state:
                        self._exp_logger.log_event(
                            "STATE_TRANSITION", from_state=prev_state,
                            to_state=state, camera_frame_index=frame_count,
                            t_state_change_perf=t_state_change,
                        )

                        # Ενεργοποίηση / απενεργοποίηση shield
                        if state == PrivacyState.PANIC and prev_state != PrivacyState.PANIC:
                            self._exp_logger.log_event(
                                "PRIVACY_SHIELD_ACTIVATED", from_state=prev_state,
                                to_state=state, camera_frame_index=frame_count,
                                t_state_change_perf=t_state_change,
                            )
                        elif prev_state == PrivacyState.PANIC and state != PrivacyState.PANIC:
                            self._exp_logger.log_event(
                                "PRIVACY_SHIELD_DEACTIVATED", from_state=prev_state,
                                to_state=state, camera_frame_index=frame_count,
                                t_state_change_perf=t_state_change,
                            )

                        # Παρουσία / απομάκρυνση εγγεγραμμένου χρήστη
                        if state == PrivacyState.NO_FACE and prev_state != PrivacyState.NO_FACE:
                            self._exp_logger.log_event(
                                "USER_LEFT_SCREEN", from_state=prev_state,
                                to_state=state, camera_frame_index=frame_count,
                            )
                        elif prev_state == PrivacyState.NO_FACE and state != PrivacyState.NO_FACE:
                            self._exp_logger.log_event(
                                "USER_RETURNED", from_state=prev_state,
                                to_state=state, camera_frame_index=frame_count,
                            )

                if self._video_recorder:
                    clock = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    try:
                        self._video_recorder.write_frame(
                            frame, overlay_text=f"{clock}  f{frame_count}  {state}",
                        )
                    except Exception:
                        logger.exception(
                            "VideoRecorder.write_frame απέτυχε στο frame %d.",
                            frame_count,
                        )

                if time.perf_counter() - _last_heartbeat >= _HEARTBEAT_S:
                    logger.info(
                        "Heartbeat: %d frames επεξεργάστηκαν (state=%s).",
                        frame_count, state,
                    )
                    _last_heartbeat = time.perf_counter()

                self.state_updated.emit(
                    state, gaze.x, gaze.y, gaze.confidence, frame_count, t_capture)

                # Frame-rate throttle
                elapsed = time.perf_counter() - t0
                sleep_t = frame_duration - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        finally:
            cap.release()
            logger.info("Camera released.")

    def stop(self) -> None:
        self._running = False
        self.wait(3000)


# Overlay Window
class OverlayWindow(QWidget):

    MASK_RGB       = (10, 10, 20)  
    BLUR_OPACITY   = 230  
    BLUR_OPACITY_MIN = 120
    BLUR_OPACITY_MAX = 255
    FOVEA_RADIUS_MIN = 60
    FOVEA_RADIUS_MAX = 500
    SAFE_COLOR     = QColor(10, 10, 20, BLUR_OPACITY)   # legacy· βλ. _mask_color()
    HARD_LOCK_COLOR = QColor(0, 0, 0, 255)   # πλήρως αδιαφανές — καμία διαρροή περιεχομένου
    BORDER_WIDTH   = 8

    THUMB_ICON_SIZE = 40   # px, μέγεθος κάθε εικονιδίου στην οθόνη
    THUMB_MARGIN    = 8    # px, κενό ανάμεσα σε εικονίδια/άκρο οθόνης

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        fovea_radius: int,
        show_trusted_thumbnails: bool = True,
        hide_thumbnails_in_panic: bool = False,
        blur_opacity: int = None,
        preview_mode: bool = False,
    ):
        super().__init__()
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.fovea_radius = fovea_radius
        self.blur_opacity = int(
            self.BLUR_OPACITY if blur_opacity is None else blur_opacity
        )
        self._preview_mode = bool(preview_mode)
        self._show_thumbnails = show_trusted_thumbnails
        self._hide_thumbnails_in_panic = hide_thumbnails_in_panic
        self._trusted_thumbs: list = []   

        self._state = PrivacyState.NO_FACE
        self._gaze_x = screen_w // 2
        self._gaze_y = screen_h // 2
        self._confidence = 0.0

        # Μέτρηση καθυστέρησης έως τη ΣΧΕΔΙΑΣΗ της ασπίδας. Ορίζεται από το
        # main() σε callable(prev_state, state, camera_frame_index,
        # t_capture_perf, t_painted_perf). Ενεργοποιείται μόνο σε μεταβάσεις,
        # ώστε να μην επιβαρύνει κάθε repaint.
        self.paint_probe_callback = None
        self._pending_paint_probe = None

        self._setup_window()

    def _setup_window(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool            # doesn't appear in taskbar
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # Cross-platform click-through
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setGeometry(0, 0, self.screen_w, self.screen_h)
        self.setWindowTitle("Privacy Shield")
        self.setCursor(Qt.CursorShape.ArrowCursor)

    # Ρυθμιζόμενες παράμετροι οπτικής προστασίας
    def _mask_color(self) -> QColor:
        return QColor(*self.MASK_RGB, self.blur_opacity)

    def set_blur_opacity(self, value: int) -> None:
        self.blur_opacity = int(max(0, min(255, value)))
        self.update()

    def set_fovea_radius(self, radius: int) -> None:
        """Ακτίνα καθαρής περιοχής σε px. Ενεργοποιεί άμεση επανασχεδίαση."""
        self.fovea_radius = max(1, int(radius))
        self.update()

    def set_trusted_thumbnails(self, auth: FaceAuthenticator) -> None:

        self._trusted_thumbs = []
        for name in auth.list_trusted():
            path = THUMB_DIR / f"{name}.png"
            if not path.exists():
                continue
            pix = QPixmap(str(path))
            if pix.isNull():
                continue
            pix = pix.scaled(
                self.THUMB_ICON_SIZE, self.THUMB_ICON_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._trusted_thumbs.append((name, pix))
        self.update()

    def _paint_trusted_thumbnails(self, painter: QPainter) -> None:
        x = self.THUMB_MARGIN
        y = self.THUMB_MARGIN
        font = QFont("Sans Serif", 7)
        painter.setFont(font)
        for name, pix in self._trusted_thumbs:
            painter.drawPixmap(x, y, pix)
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(x, y, pix.width(), pix.height())
            label = name if len(name) <= 12 else name[:11] + "…"
            painter.drawText(x, y + pix.height() + 11, label)
            x += pix.width() + self.THUMB_MARGIN

    def showEvent(self, event) -> None:

        super().showEvent(event)
        if platform.system() != "Windows":
            return
        try:
            import ctypes
            hwnd = int(self.winId())
            GWL_EXSTYLE       = -20
            WS_EX_LAYERED     = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT
            )
        except Exception:
            logger.warning("Windows layered-window fallback failed; "
                            "relying on Qt's WA_TransparentForMouseEvents.")

    # State update (called from main thread via signal)
    def update_state(
        self, state: str, gaze_x: int, gaze_y: int, confidence: float,
        camera_frame_index: int = -1, t_capture_perf: float = 0.0,
    ) -> None:
        prev_state = self._state
        entering_hard_lock = (
            state == PrivacyState.HARD_LOCK and self._state != PrivacyState.HARD_LOCK
        )
        leaving_hard_lock = (
            state != PrivacyState.HARD_LOCK and self._state == PrivacyState.HARD_LOCK
        )

        self._state = state
        self._confidence = confidence
        if confidence > 0.1:
            self._gaze_x = gaze_x
            self._gaze_y = gaze_y
        

        if entering_hard_lock:
            self._enter_hard_lock()
        elif leaving_hard_lock:
            self._exit_hard_lock()

        if state != prev_state and t_capture_perf:
            self._pending_paint_probe = (
                prev_state, state, camera_frame_index, t_capture_perf,
            )

        self.update()   # schedule repaint

    def _enter_hard_lock(self) -> None:

        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.grabKeyboard()
        self.raise_()
        self._force_foreground()
        logger.warning("HARD_LOCK: input capture ενεργό (mouse+keyboard grabbed).")

    def _force_foreground(self) -> None:

        if platform.system() != "Windows":
            self.activateWindow()
            return
        try:
            import ctypes
            user32   = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            hwnd        = int(self.winId())
            fg_hwnd     = user32.GetForegroundWindow()
            current_tid = kernel32.GetCurrentThreadId()
            fg_tid      = user32.GetWindowThreadProcessId(fg_hwnd, None)
            if fg_tid and fg_tid != current_tid:
                user32.AttachThreadInput(fg_tid, current_tid, True)
                user32.SetForegroundWindow(hwnd)
                user32.BringWindowToTop(hwnd)
                user32.AttachThreadInput(fg_tid, current_tid, False)
            else:
                user32.SetForegroundWindow(hwnd)
                user32.BringWindowToTop(hwnd)
        except Exception:
            logger.exception(
                "Force-foreground (Windows) απέτυχε· fallback σε activateWindow()."
            )
            self.activateWindow()

    def _exit_hard_lock(self) -> None:
        self.releaseKeyboard()
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        logger.info("HARD_LOCK: input capture απενεργοποιήθηκε.")

    # Painting
    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._preview_mode:
            self._paint_preview(painter)
            painter.end()
            return

        if self._state == PrivacyState.CLEAR:
            # Fully transparent — authorized user only, nothing to draw
            pass
        elif self._state == PrivacyState.HARD_LOCK:
            self._paint_hard_lock(painter)
        elif self._state == PrivacyState.PANIC:
            # Unknown observer: foveated blur centred on user's gaze + red border
            if self._confidence > 0.05:
                self._paint_foveated(painter)
            else:
                self._paint_full_mask(painter)
            self._paint_panic_border(painter)
        else:
            # NO_FACE: no authorized user visible — dark mask, no label
            self._paint_full_mask(painter)

        if self._show_thumbnails and self._trusted_thumbs:
            panic_like = self._state in (PrivacyState.PANIC, PrivacyState.HARD_LOCK)
            if not (panic_like and self._hide_thumbnails_in_panic):
                self._paint_trusted_thumbnails(painter)

        painter.end()
        self._report_paint_completion()

    def _report_paint_completion(self) -> None:
        """
        Καλείται στο τέλος του paintEvent. Καταγράφει τη στιγμή κατά την οποία
        η ασπίδα σχεδιάστηκε πραγματικά, ώστε η καθυστέρηση να μετράται μέχρι
        την ορατή αλλαγή και όχι μέχρι την αλλαγή μεταβλητής κατάστασης.

        Δεν περιλαμβάνει την καθυστέρηση της ίδιας της οθόνης (display
        latency), η οποία μετριέται μόνο με εξωτερική κάμερα υψηλού ρυθμού.
        """
        probe = self._pending_paint_probe
        if probe is None:
            return
        self._pending_paint_probe = None
        t_painted = time.perf_counter()
        cb = self.paint_probe_callback
        if cb is None:
            return
        try:
            cb(probe[0], probe[1], probe[2], probe[3], t_painted)
        except Exception:
            logger.exception("paint_probe_callback απέτυχε — η συνεδρία συνεχίζεται.")

    def _paint_hard_lock(self, painter: QPainter) -> None:
        painter.fillRect(0, 0, self.screen_w, self.screen_h, self.HARD_LOCK_COLOR)
        painter.setPen(QPen(QColor(255, 255, 255, 220)))
        font_title = QFont("Monospace", 30, QFont.Weight.Bold)
        painter.setFont(font_title)
        title_rect = self.rect().adjusted(0, -40, 0, -40)
        painter.drawText(
            title_rect, Qt.AlignmentFlag.AlignCenter,
            "System Locked",
        )
        font_body = QFont("Monospace", 14)
        painter.setFont(font_body)
        body_rect = self.rect().adjusted(0, 40, 0, 40)
        painter.drawText(
            body_rect, Qt.AlignmentFlag.AlignCenter,
            "Ο υπολογιστής παραμένει μη διαθέσιμος μέχρι να αναγνωριστεί\n"
            "ξανά ο εγγεγραμμένος χρήστης από την κάμερα.",
        )

    def _paint_full_mask(self, painter: QPainter) -> None:
        
        painter.fillRect(0, 0, self.screen_w, self.screen_h, self._mask_color())

        if self._state == PrivacyState.NO_FACE:
            painter.setPen(QPen(QColor(180, 180, 180, 160)))
            font = QFont("Monospace", 16)
            painter.setFont(font)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No authorized user detected.\nScreen locked.",
            )

    def _paint_foveated(self, painter: QPainter, centre=None) -> None:
        if centre is None:
            cx, cy = float(self._gaze_x), float(self._gaze_y)
        else:
            cx, cy = float(centre[0]), float(centre[1])
        r = float(self.fovea_radius)

        painter.fillRect(0, 0, self.screen_w, self.screen_h, self._mask_color())

        painter.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_DestinationOut
        )

        gradient = QRadialGradient(cx, cy, r)
        gradient.setColorAt(0.0,  QColor(0, 0, 0, 255))   # πλήρως διαφανές κέντρο (255 = max alpha για DestinationOut)
        gradient.setColorAt(0.50, QColor(0, 0, 0, 255))   # καθαρή ζώνη εώς 50%
        gradient.setColorAt(0.80, QColor(0, 0, 0, 80))    # μαλακή μετάβαση
        gradient.setColorAt(1.0,  QColor(0, 0, 0, 0))     # πλήρες mask στο άκρο

        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(int(cx - r), int(cy - r), int(2 * r), int(2 * r))

        painter.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_SourceOver
        )

    def _paint_preview(self, painter: QPainter) -> None:
        """Στατική προεπισκόπηση PANIC, αγκυρωμένη στο κέντρο της οθόνης."""
        cx, cy = self.screen_w // 2, self.screen_h // 2
        self._paint_foveated(painter, centre=(cx, cy))
        self._paint_panic_border(painter)

        r = int(self.fovea_radius)
        painter.setPen(QPen(QColor(255, 255, 255, 90), 1, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(cx - r, cy - r, 2 * r, 2 * r)
        painter.setPen(QPen(QColor(255, 80, 80, 210), 2))
        painter.drawLine(cx - 12, cy, cx + 12, cy)
        painter.drawLine(cx, cy - 12, cx, cy + 12)

        pct = round(self.blur_opacity / 255 * 100)
        painter.setPen(QPen(QColor(235, 235, 235, 225)))
        painter.setFont(QFont("Sans Serif", 11))
        caption = QRect(0, 40, self.screen_w, 60)
        painter.drawText(
            caption, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            f"Προεπισκόπηση PANIC — σκίαση {pct}%  ·  καθαρή περιοχή "
            f"{r} px (ακτίνα, κέντρο οθόνης)\n"
            "Κατά την πραγματική λειτουργία η περιοχή ακολουθεί το βλέμμα.",
        )

    def _paint_panic_border(self, painter: QPainter) -> None:
        pen = QPen(QColor(255, 60, 60), self.BORDER_WIDTH)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(
            self.BORDER_WIDTH // 2,
            self.BORDER_WIDTH // 2,
            self.screen_w - self.BORDER_WIDTH,
            self.screen_h - self.BORDER_WIDTH,
        )

    # Allow quit via Escape key (disabled while HARD_LOCK is active)
    def keyPressEvent(self, event) -> None:
        if self._state == PrivacyState.HARD_LOCK or self._preview_mode:
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            QApplication.quit()


# Calibration wizard (blocking, runs before main loop)
CALIBRATION_TARGETS_REL = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
]


def run_calibration_wizard(
    tracker: GazeTracker,
    screen_w: int,
    screen_h: int,
    camera_index: int = 0,
) -> None:

    cap = CameraThread._open_camera(camera_index)
    if not cap.isOpened():
        logger.error("Cannot open camera for calibration.")
        return

    tracker.reset_calibration()
    target_idx = 0
    targets = [
        (int(rx * screen_w), int(ry * screen_h))
        for rx, ry in CALIBRATION_TARGETS_REL
    ]

    print("\n=== CALIBRATION WIZARD ===")
    print("Look at the RED DOT on screen, then press SPACE to record.")
    print("Press 'q' to abort.\n")

    win_name = "Calibration – look at the dot, press SPACE"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while target_idx < len(targets):
        ret, frame = cap.read()
        if not ret:
            continue

        tx, ty = targets[target_idx]
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        # Draw dot
        cv2.circle(canvas, (tx, ty), 20, (0, 0, 220), -1)
        cv2.circle(canvas, (tx, ty),  6, (255, 255, 255), -1)
        msg = f"Point {target_idx + 1}/{len(targets)}  – look here, then press SPACE"
        cv2.putText(canvas, msg, (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (200, 200, 200), 2)
        cv2.imshow(win_name, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Calibration aborted.")
            break
        if key == ord(" "):
            ok = tracker.add_calibration_sample(frame, (tx, ty))
            if ok:
                print(f"  ✓ Sample {target_idx + 1} captured.")
                target_idx += 1
            else:
                print("  ✗ No iris detected – try again.")

    cap.release()
    cv2.destroyAllWindows()

    if target_idx == len(targets):
        ok = tracker.fit_calibration()
        if ok:
            print("\nCalibration complete and saved!")
        else:
            print("\nCalibration fitting failed. Please try again.")
    else:
        print("Calibration incomplete.")


# Enrollment helper
def run_enrollment(auth: FaceAuthenticator, name: str, camera_index: int = 0) -> None:
    cap = CameraThread._open_camera(camera_index)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    SAMPLES_NEEDED = 15
    collected_frames = []

    print(f"\nEnrolling '{name}' – look at the camera naturally (slight head movement is OK).")
    print(f"Press SPACE to start collecting {SAMPLES_NEEDED} frames, or 'q' to abort.")
    win = "Enrollment"
    cv2.namedWindow(win)

    capturing = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        display = frame.copy()

        if not capturing:
            cv2.putText(display, "Press SPACE to start enrollment, 'q' to quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        else:
            progress = len(collected_frames)
            cv2.putText(display, f"Collecting... {progress}/{SAMPLES_NEEDED}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.putText(display, "Move your head slightly left/right",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            collected_frames.append(frame.copy())
            if len(collected_frames) >= SAMPLES_NEEDED:
                break

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return
        if key == ord(" "):
            capturing = True

    cap.release()
    cv2.destroyAllWindows()

    ok = auth.enroll_from_frames(collected_frames, name)
    if ok:
        print(f"✓ '{name}' enrolled successfully from {SAMPLES_NEEDED} frames.")
        print("  Tip: re-enroll if recognition is unreliable (different lighting, glasses, etc.)")
    else:
        print("✗ Could not collect enough valid frames. Ensure good lighting and face visibility.")


# Tray icon helper
def _make_tray_icon() -> QIcon:

    pix = QPixmap(32, 32)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(40, 110, 200))
    painter.drawEllipse(1, 1, 30, 30)
    painter.setPen(QPen(QColor(255, 255, 255)))
    font = QFont("Sans Serif", 14, QFont.Weight.Bold)
    painter.setFont(font)
    painter.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "P")
    painter.end()
    return QIcon(pix)


class BackendSelectDialog(QDialog):
    BACKEND_WEBCAM = "webcam"
    BACKEND_TOBII4C = "tobii4c"
    BACKEND_TOBII = "tobii"

    def __init__(
        self,
        tobii_available: bool = False,
        tobii4c_available: bool = False,
        default_backend: str = "webcam",
    ):
        super().__init__()
        self.setWindowTitle(f"Dynamic Privacy Shield — Επιλογή λειτουργίας  [{APP_BUILD}]")
        self.setFixedWidth(380)
        self.chosen_backend = None   # None => ο χρήστης πάτησε Έξοδος

        layout = QVBoxLayout(self)

        title = QLabel("<h2>Dynamic Privacy Shield</h2>")
        layout.addWidget(title)

        subtitle = QLabel("Επίλεξε πηγή εκτίμησης βλέμματος (gaze backend):")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        self._group = QButtonGroup(self)

        def add_option(text: str, value: str, enabled: bool, disabled_hint: str = "") -> QRadioButton:
            rb = QRadioButton(text)
            rb.setEnabled(enabled)
            if not enabled and disabled_hint:
                rb.setToolTip(disabled_hint)
            self._group.addButton(rb)
            layout.addWidget(rb)
            rb.setProperty("backend_value", value)
            return rb

        self._rb_webcam = add_option(
            "Μόνο κάμερα (webcam — MediaPipe iris tracking)",
            self.BACKEND_WEBCAM, enabled=True,
        )
        self._rb_tobii4c = add_option(
            "Tobii 4C (gaming eyetracker — Stream Engine bridge)",
            self.BACKEND_TOBII4C, enabled=tobii4c_available,
            disabled_hint=(
                "Δεν βρέθηκε το module 'tobii4c_gaze_tracker' ή το "
                "tobii_gaze_bridge.exe σε αυτό το μηχάνημα."
            ),
        )
        self._rb_tobii = add_option(
            "Tobii Pro SDK (Spectrum — tobii_research)",
            self.BACKEND_TOBII, enabled=tobii_available,
            disabled_hint=(
                "Δεν είναι εγκατεστημένο το πακέτο 'tobii-research' σε "
                "αυτό το μηχάνημα."
            ),
        )

        by_value = {
            self.BACKEND_WEBCAM: self._rb_webcam,
            self.BACKEND_TOBII4C: self._rb_tobii4c,
            self.BACKEND_TOBII: self._rb_tobii,
        }
        default_rb = by_value.get(default_backend, self._rb_webcam)
        (default_rb if default_rb.isEnabled() else self._rb_webcam).setChecked(True)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(divider)

        btn_row = QHBoxLayout()
        start_btn = QPushButton("▶  Έναρξη")
        start_btn.clicked.connect(self._accept_selection)
        exit_btn = QPushButton("Έξοδος")
        exit_btn.clicked.connect(self._exit_system)
        btn_row.addWidget(start_btn)
        btn_row.addWidget(exit_btn)
        layout.addLayout(btn_row)

    def _accept_selection(self) -> None:
        checked = self._group.checkedButton()
        self.chosen_backend = checked.property("backend_value") if checked else self.BACKEND_WEBCAM
        self.accept()

    def _exit_system(self) -> None:
        self.chosen_backend = None
        self.reject()


# Launcher Dialog (double-click-friendly menu)
class LauncherDialog(QDialog):

    ACTION_START     = "start"
    ACTION_ENROLL    = "enroll"
    ACTION_REMOVE    = "remove"
    ACTION_CALIBRATE = "calibrate"
    ACTION_BACK      = "back"
    ACTION_EXIT = ACTION_BACK

    # Πόσο παραμένει ορατή η προεπισκόπηση μετά την τελευταία μεταβολή.
    PREVIEW_LINGER_MS = 1600

    def __init__(
        self,
        auth: FaceAuthenticator,
        gaze_backend: str = "webcam",
        backend_label: str = "—",
        screen_w: int = 1920,
        screen_h: int = 1080,
        fovea_radius: int = 120,
        blur_opacity: int = OverlayWindow.BLUR_OPACITY,
    ):
        super().__init__()
        self.setWindowTitle(f"Dynamic Privacy Shield — Προετοιμασία συνεδρίας  [{APP_BUILD}]")
        self.setFixedWidth(380)
        self.chosen_action = self.ACTION_BACK

    
        self._screen_w = screen_w
        self._screen_h = screen_h
        self.fovea_radius = int(fovea_radius)
        self.blur_opacity = int(blur_opacity)

        self._preview_overlay = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(self.PREVIEW_LINGER_MS)
        self._preview_timer.timeout.connect(self._hide_preview)

        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)

        layout = QVBoxLayout(self)

        title = QLabel("<h2>Dynamic Privacy Shield</h2>")
        layout.addWidget(title)

        backend_line = QLabel(f"Επιλεγμένο σύστημα: <b>{backend_label}</b>")
        backend_line.setWordWrap(True)
        layout.addWidget(backend_line)

        names = auth.list_trusted()
        if names:
            status_text = f"Εγγεγραμμένοι χρήστες ({len(names)}): {', '.join(names)}"
        else:
            status_text = "⚠ Δεν υπάρχουν εγγεγραμμένοι χρήστες ακόμα."
        status = QLabel(status_text)
        status.setWordWrap(True)
        layout.addWidget(status)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(divider)

        def add_button(text, action, enabled: bool = True, hint: str = ""):
            btn = QPushButton(text)
            btn.setEnabled(enabled)
            if hint:
                btn.setToolTip(hint)
            btn.clicked.connect(lambda: self._choose(action))
            layout.addWidget(btn)
            return btn

        add_button("▶  Έναρξη Προστασίας", self.ACTION_START)
        add_button("➕  Εγγραφή νέου χρήστη", self.ACTION_ENROLL)
        add_button("➖  Αφαίρεση χρήστη", self.ACTION_REMOVE)

        is_webcam = (gaze_backend == "webcam")
        add_button(
            "◎  Βαθμονόμηση βλέμματος (wizard)" if is_webcam
            else "◎  Βαθμονόμηση βλέμματος — εκτός εφαρμογής",
            self.ACTION_CALIBRATE,
            enabled=is_webcam,
            hint=(
                "Βαθμονόμηση 9 σημείων για το backend της κάμερας."
                if is_webcam else
                "Το επιλεγμένο σύστημα Tobii βαθμονομείται εκ των προτέρων, "
                "μέσω του Tobii Eye Tracker Manager / Eye Tracking Core."
            ),
        )

        if not is_webcam:
            note = QLabel(
                "Βεβαιώσου ότι η βαθμονόμηση του Tobii έχει ολοκληρωθεί "
                "<b>πριν</b> την έναρξη της προστασίας."
            )
            note.setWordWrap(True)
            layout.addWidget(note)

        settings_divider = QFrame()
        settings_divider.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(settings_divider)

        layout.addWidget(QLabel("<b>Οπτική προστασία (κατάσταση PANIC)</b>"))

        # Ρυθμιστικό 1 — ένταση σκίασης
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Ένταση σκίασης"))
        opacity_row.addStretch(1)
        self._opacity_value = QLabel()
        opacity_row.addWidget(self._opacity_value)
        layout.addLayout(opacity_row)

        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(
            OverlayWindow.BLUR_OPACITY_MIN, OverlayWindow.BLUR_OPACITY_MAX
        )
        self._opacity_slider.setValue(self.blur_opacity)
        self._opacity_slider.setSingleStep(5)
        self._opacity_slider.setPageStep(15)
        self._opacity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._opacity_slider.setTickInterval(15)
        self._opacity_slider.setToolTip(
            "Αδιαφάνεια της σκούρας μάσκας εκτός της καθαρής περιοχής. "
            "Χαμηλότερες τιμές αφήνουν περισσότερο περιεχόμενο αναγνώσιμο "
            "από τρίτο παρατηρητή."
        )
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        layout.addWidget(self._opacity_slider)

        # Ρυθμιστικό 2 — μέγεθος καθαρής περιοχής
        fovea_row = QHBoxLayout()
        fovea_row.addWidget(QLabel("Μέγεθος καθαρής περιοχής"))
        fovea_row.addStretch(1)
        self._fovea_value = QLabel()
        fovea_row.addWidget(self._fovea_value)
        layout.addLayout(fovea_row)

        self._fovea_slider = QSlider(Qt.Orientation.Horizontal)
        self._fovea_slider.setRange(
            OverlayWindow.FOVEA_RADIUS_MIN, OverlayWindow.FOVEA_RADIUS_MAX
        )
        self._fovea_slider.setValue(self.fovea_radius)
        self._fovea_slider.setSingleStep(10)
        self._fovea_slider.setPageStep(25)
        self._fovea_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._fovea_slider.setTickInterval(50)
        self._fovea_slider.setToolTip(
            "Ακτίνα (σε pixel) της περιοχής που παραμένει καθαρή γύρω από "
            "το σημείο εστίασης του βλέμματος."
        )
        self._fovea_slider.valueChanged.connect(self._on_fovea_changed)
        layout.addWidget(self._fovea_slider)

        self._live_preview_cb = QCheckBox(
            "Ζωντανή προεπισκόπηση στην οθόνη κατά τη ρύθμιση"
        )
        self._live_preview_cb.setChecked(True)
        self._live_preview_cb.setToolTip(
            "Η προεπισκόπηση αγκυρώνεται στο κέντρο της οθόνης και ΔΕΝ "
            "ακολουθεί το βλέμμα, ώστε ο στόχος να παραμένει ακίνητος κατά "
            "τη ρύθμιση."
        )
        self._live_preview_cb.toggled.connect(self._on_live_preview_toggled)
        layout.addWidget(self._live_preview_cb)

        preview_btn = QPushButton("  Δοκιμή προεπισκόπησης (3 δευτ.)")
        preview_btn.clicked.connect(self._flash_preview)
        layout.addWidget(preview_btn)

        # Αρχικοποίηση ετικετών τιμής χωρίς να ανοίξει η προεπισκόπηση
        self._refresh_value_labels()

        final_divider = QFrame()
        final_divider.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(final_divider)

        add_button("◀  Πίσω στην επιλογή συστήματος", self.ACTION_BACK)

       
        self.adjustSize()
        self._place_off_centre()

    # Ρυθμίσεις / προεπισκόπηση
    def _refresh_value_labels(self) -> None:
        pct = round(self.blur_opacity / 255 * 100)
        self._opacity_value.setText(f"<b>{pct}%</b>  ({self.blur_opacity}/255)")
        self._fovea_value.setText(f"<b>{self.fovea_radius} px</b>")

    def _place_off_centre(self) -> None:
        try:
            avail = QGuiApplication.primaryScreen().availableGeometry()
        except Exception:
            return
        x = avail.x() + avail.width() - self.width() - 24
        y = avail.y() + avail.height() - self.height() - 48
        self.move(max(avail.x(), x), max(avail.y(), y))

    def _on_opacity_changed(self, value: int) -> None:
        self.blur_opacity = int(value)
        self._refresh_value_labels()
        if self._preview_overlay is not None:
            self._preview_overlay.set_blur_opacity(self.blur_opacity)
        self._touch_preview()

    def _on_fovea_changed(self, value: int) -> None:
        self.fovea_radius = int(value)
        self._refresh_value_labels()
        if self._preview_overlay is not None:
            self._preview_overlay.set_fovea_radius(self.fovea_radius)
        self._touch_preview()

    def _on_live_preview_toggled(self, checked: bool) -> None:
        if not checked:
            self._preview_timer.stop()
            self._hide_preview()

    def _ensure_preview(self) -> None:
        if self._preview_overlay is None:
            self._preview_overlay = OverlayWindow(
                self._screen_w, self._screen_h,
                fovea_radius=self.fovea_radius,
                show_trusted_thumbnails=False,
                blur_opacity=self.blur_opacity,
                preview_mode=True,
            )
        self._preview_overlay.set_blur_opacity(self.blur_opacity)
        self._preview_overlay.set_fovea_radius(self.fovea_radius)
        if not self._preview_overlay.isVisible():
            self._preview_overlay.show()
        self.raise_()
        self.activateWindow()
        self.setWindowOpacity(0.90)

    def _touch_preview(self) -> None:
        if not self._live_preview_cb.isChecked():
            return
        self._ensure_preview()
        self._preview_timer.start(self.PREVIEW_LINGER_MS)

    def _flash_preview(self) -> None:
        self._ensure_preview()
        self._preview_timer.start(3000)

    def _hide_preview(self) -> None:
        self.setWindowOpacity(1.0)
        if self._preview_overlay is not None:
            self._preview_overlay.hide()

    def _destroy_preview(self) -> None:
        self._preview_timer.stop()
        self.setWindowOpacity(1.0)
        if self._preview_overlay is not None:
            self._preview_overlay.hide()
            self._preview_overlay.deleteLater()
            self._preview_overlay = None

    def done(self, result: int) -> None:
        self._destroy_preview()
        super().done(result)

    def _choose(self, action: str) -> None:
        self.chosen_action = action
        self.accept()


class BlockPlan:
    """
    Προκαθορισμένη ακολουθία πειραματικών κελιών.

    Το πρόγραμμα διαβάζεται από CSV με στήλες:
        block_id, iv_angle, iv_distance, iv_lighting, profile_id, duration_s
    (οι δύο τελευταίες προαιρετικές).

    Η ΣΕΙΡΑ αυτοματοποιείται, η ΜΕΤΑΒΑΣΗ όχι. Κάθε κελί αντιστοιχεί σε
    φυσική διάταξη που πρέπει να στηθεί από τον πειραματιστή· αυτόματη
    προώθηση με χρονόμετρο θα ετικετάριζε δεδομένα με συνθήκη που δεν έχει
    ακόμη εγκατασταθεί. Το duration_s χρησιμοποιείται μόνο ως αντίστροφη
    μέτρηση και ειδοποίηση, ποτέ για αυτόματη μετάβαση.
    """

    FIELDS = ("block_id", "iv_angle", "iv_distance", "iv_lighting",
              "profile_id", "duration_s")

    def __init__(self, cells: list, counterbalance: str = "none",
                 participant_id: str = ""):
        if not cells:
            raise ValueError("Το πρόγραμμα block είναι κενό.")
        self.source_order = [dict(c) for c in cells]
        self.cells = self._order(list(cells), counterbalance, participant_id)
        self.counterbalance = counterbalance
        self.index = 0

    @staticmethod
    def _order(cells: list, mode: str, participant_id: str) -> list:
        if mode == "rotate":
            # Λατινικό-τετράγωνο στυλ: κάθε συμμετέχων ξεκινά από άλλο κελί.
            digits = "".join(ch for ch in participant_id if ch.isdigit())
            k = (int(digits) if digits else 0) % len(cells)
            return cells[k:] + cells[:k]
        if mode == "shuffle":
            # Τυχαία αλλά ΑΝΑΠΑΡΑΓΩΓΙΜΗ σειρά: ίδιος συμμετέχων, ίδια σειρά.
            import random
            rng = random.Random(participant_id or "seed")
            out = list(cells)
            rng.shuffle(out)
            return out
        return cells

    @classmethod
    def from_csv(cls, path, counterbalance: str = "none",
                 participant_id: str = "") -> "BlockPlan":
        import csv as _csv
        rows = []
        with open(path, newline="", encoding="utf-8-sig") as fh:
            for raw in _csv.DictReader(fh):
                row = {k: (raw.get(k) or "").strip() for k in cls.FIELDS}
                if not row["block_id"]:
                    continue
                rows.append(row)
        if not rows:
            raise ValueError(
                f"Το αρχείο '{path}' δεν περιέχει έγκυρες γραμμές. Απαιτείται "
                f"κεφαλίδα με τις στήλες: {', '.join(cls.FIELDS[:4])}."
            )
        return cls(rows, counterbalance, participant_id)

    # -- πρόσβαση ---------------------------------------------------------
    @property
    def total(self) -> int:
        return len(self.cells)

    @property
    def finished(self) -> bool:
        return self.index >= self.total

    def current(self) -> "dict | None":
        return None if self.finished else dict(self.cells[self.index])

    def peek_next(self) -> "dict | None":
        nxt = self.index + 1
        return dict(self.cells[nxt]) if nxt < self.total else None

    def advance(self) -> "dict | None":
        self.index += 1
        return self.current()

    def duration_s(self) -> "float | None":
        cur = self.current()
        if not cur:
            return None
        try:
            v = float(cur.get("duration_s") or 0)
            return v if v > 0 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def describe(cell: "dict | None") -> str:
        if not cell:
            return "—"
        return (f"{cell.get('block_id','?')} — {cell.get('iv_angle','?')}° / "
                f"{cell.get('iv_distance','?')}cm / {cell.get('iv_lighting','?')}")

    def as_metadata(self) -> dict:
        return {
            "counterbalance": self.counterbalance,
            "n_blocks": self.total,
            "order": [c.get("block_id", "") for c in self.cells],
            "cells": self.cells,
        }


class BlockDialog(QDialog):
    """
    Ορισμός του τρέχοντος κελιού του πειραματικού σχεδιασμού.

    Οι τιμές γράφονται σε ΚΑΘΕ γραμμή του log, ώστε η ταξινόμηση των δεδομένων
    ανά κελί να μη γίνεται εκ των υστέρων με βάση χρονοσφραγίδες.
    """

    ANGLES    = ["0", "30", "45", "60", "90"]
    DISTANCES = ["50", "60", "80", "100", "150"]
    LIGHTING  = ["bright", "dim", "backlit", "dark"]

    def __init__(self, block_id="", angle="", distance="", lighting="", profile_id=""):
        super().__init__()
        self.setWindowTitle("Ορισμός block")
        self.setModal(True)
        self.values = None

        form = QFormLayout(self)

        self._block = QLineEdit(str(block_id) or "B01")
        form.addRow("Block ID:", self._block)

        self._angle = QComboBox(); self._angle.setEditable(True)
        self._angle.addItems(self.ANGLES); self._angle.setCurrentText(str(angle))
        form.addRow("Γωνία παρατηρητή (°):", self._angle)

        self._dist = QComboBox(); self._dist.setEditable(True)
        self._dist.addItems(self.DISTANCES); self._dist.setCurrentText(str(distance))
        form.addRow("Απόσταση παρατηρητή (cm):", self._dist)

        self._light = QComboBox(); self._light.setEditable(True)
        self._light.addItems(self.LIGHTING); self._light.setCurrentText(str(lighting))
        form.addRow("Φωτισμός:", self._light)

        self._profile = QLineEdit(str(profile_id))
        form.addRow("Profile ID:", self._profile)

        hint = QLabel(
            "Οι τιμές αυτές αποθηκεύονται σε κάθε γραμμή δεδομένων. "
            "Άλλαξέ τες στο διάλειμμα μεταξύ block, ποτέ στη μέση δοκιμής."
        )
        hint.setWordWrap(True)
        form.addRow(hint)

        buttons = QHBoxLayout()
        ok = QPushButton("Έναρξη block")
        ok.clicked.connect(self._accept)
        cancel = QPushButton("Άκυρο")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel); buttons.addWidget(ok)
        form.addRow(buttons)

    def _accept(self) -> None:
        block_id = self._block.text().strip()
        if not block_id:
            QMessageBox.warning(self, "Block ID", "Το Block ID δεν μπορεί να είναι κενό.")
            return
        self.values = {
            "block_id":    block_id,
            "iv_angle":    self._angle.currentText().strip(),
            "iv_distance": self._dist.currentText().strip(),
            "iv_lighting": self._light.currentText().strip(),
            "profile_id":  self._profile.text().strip(),
        }
        self.accept()


class ControlWindow(QWidget):

    def __init__(self, quit_callback, switch_callback=None,
                 backend_label: str = "—", experimenter_screen: "int | None" = None,
                 block_callback=None, sync_callback=None, block_label: str = "—"):
        super().__init__()
        self.setWindowTitle(f"Dynamic Privacy Shield — Ενεργό  [{APP_BUILD}]")
        self.setFixedSize(380, 360)
        self._quit_callback = quit_callback
        self._switch_callback = switch_callback
        self._block_callback = block_callback
        self._sync_callback = sync_callback
        self._hard_locked = False
        self._plan_finished = False

        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )

        layout = QVBoxLayout(self)
        title = QLabel("<b>Dynamic Privacy Shield</b>")
        layout.addWidget(title)

        self._backend_label = QLabel(f"Σύστημα: <b>{backend_label}</b>")
        self._backend_label.setWordWrap(True)
        layout.addWidget(self._backend_label)

        self._block_label_w = QLabel(f"Block: <b>{block_label}</b>")
        self._block_label_w.setWordWrap(True)
        layout.addWidget(self._block_label_w)

        self._next_label_w = QLabel("")
        self._next_label_w.setWordWrap(True)
        self._next_label_w.setStyleSheet("color: #555;")
        layout.addWidget(self._next_label_w)

        self._status_label = QLabel("Κατάσταση: —")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(divider)

        if block_callback is not None:
            self._block_btn = QPushButton("▶  Επόμενο block")
            self._block_btn.setToolTip(
                "Κλείνει το τρέχον block και ανοίγει το επόμενο. Πάτα το ΑΦΟΥ "
                "έχει στηθεί η νέα φυσική διάταξη, όχι πριν."
            )
            self._block_btn.clicked.connect(self._on_block)
            layout.addWidget(self._block_btn)
        else:
            self._block_btn = None

        if sync_callback is not None:
            self._sync_btn = QPushButton("⏱  Δείκτης συγχρονισμού")
            self._sync_btn.setToolTip(
                "Καταγράφει SYNC_MARKER. Πάτησέ τον ταυτόχρονα με τον "
                "αντίστοιχο δείκτη στο Tobii Pro Lab, στην αρχή και στο τέλος "
                "κάθε block."
            )
            self._sync_btn.clicked.connect(self._on_sync)
            layout.addWidget(self._sync_btn)
        else:
            self._sync_btn = None

        if switch_callback is not None:
            self._switch_btn = QPushButton("⟳  Αλλαγή συστήματος καταγραφής")
            self._switch_btn.setToolTip(
                "Κλείνει τη συνεδρία του τρέχοντος backend και επιστρέφει στο "
                "παράθυρο επιλογής. Χρησιμοποιείται ΜΟΝΟ στο διάλειμμα μεταξύ "
                "block -- ποτέ στη μέση δοκιμής."
            )
            self._switch_btn.clicked.connect(self._on_switch)
            layout.addWidget(self._switch_btn)
        else:
            self._switch_btn = None

        note = QLabel(
            "Η αλλαγή συστήματος κλείνει το τρέχον αρχείο καταγραφής και "
            "ανοίγει νέο. Το νέο backend απαιτεί δική του βαθμονόμηση πριν "
            "συνεχίσει η συνεδρία."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self._quit_btn = QPushButton("Έξοδος")
        self._quit_btn.clicked.connect(self.close)   # περνάει πάντα από closeEvent
        layout.addWidget(self._quit_btn)

        self._place_on_experimenter_screen(experimenter_screen)

    def _place_on_experimenter_screen(self, index: "int | None") -> None:
        screens = QGuiApplication.screens()
        target = None
        if index is not None and 0 <= index < len(screens):
            target = screens[index]
        elif len(screens) > 1:
            primary = QGuiApplication.primaryScreen()
            target = next((s for s in screens if s is not primary), None)
        if target is None:
            if len(screens) <= 1:
                logger.warning(
                    "Μία μόνο οθόνη -- η κονσόλα πειραματιστή θα είναι ορατή "
                    "και στον συμμετέχοντα. Δες τη σημείωση περί demand "
                    "characteristics στην τεκμηρίωση."
                )
            return
        geo = target.availableGeometry()
        self.move(geo.x() + geo.width() - self.width() - 40, geo.y() + 40)

    def set_block_label(self, text: str) -> None:
        self._block_label_w.setText(f"Block: <b>{text}</b>")

    def set_next_block(self, text: str, progress: str = "") -> None:
        """Δείχνει το επόμενο κελί, ώστε να το στήσεις πριν πατήσεις."""
        if text:
            self._next_label_w.setText(f"Επόμενο: {text}")
        else:
            self._next_label_w.setText(
                "<b>Τελευταίο block.</b> Μετά από αυτό, Έξοδος."
            )
        if self._block_btn is not None and progress:
            self._block_btn.setText(f"▶  Επόμενο block  ({progress})")

    def set_block_finished(self, remaining_s=None) -> None:
        """
        Ενημερώνει την αντίστροφη μέτρηση. Ποτέ δεν προχωρά μόνη της:
        η μετάβαση απαιτεί ρητή ενέργεια, γιατί συνοδεύεται από φυσική
        αναδιάταξη που το λογισμικό δεν μπορεί να επιβεβαιώσει.
        """
        if remaining_s is None:
            self._next_label_w.setStyleSheet("color: #555;")
            return
        if remaining_s > 0:
            m, sec = divmod(int(remaining_s), 60)
            self._block_label_w.setText(
                self._block_label_w.text().split("  ⏳")[0] + f"  ⏳ {m}:{sec:02d}"
            )
        else:
            self._block_label_w.setText(
                self._block_label_w.text().split("  ⏳")[0]
                + "  <span style='color:#c00;'><b>ΛΗΞΗ — πάτα Επόμενο</b></span>"
            )

    def disable_block_button(self, reason: str = "") -> None:
        self._plan_finished = True
        if self._block_btn is not None:
            self._block_btn.setEnabled(False)
            self._block_btn.setText("✓  Όλα τα blocks ολοκληρώθηκαν")
        if reason:
            self._next_label_w.setText(reason)

    def _on_block(self) -> None:
        if self._hard_locked or self._block_callback is None:
            return
        try:
            self._block_callback()
        except Exception:
            logger.exception("Ο ορισμός νέου block απέτυχε.")

    def _on_sync(self) -> None:
        if self._sync_callback is None:
            return
        try:
            self._sync_callback("manual")
        except Exception:
            logger.exception("Η καταγραφή δείκτη συγχρονισμού απέτυχε.")

    def _on_switch(self) -> None:
        if self._hard_locked:
            return
        answer = QMessageBox.question(
            self, "Αλλαγή συστήματος καταγραφής",
            "Να κλείσει η τρέχουσα συνεδρία και να ανοίξει το παράθυρο "
            "επιλογής backend;\n\nΓίνεται ΜΟΝΟ στο διάλειμμα μεταξύ block. "
            "Το νέο σύστημα απαιτεί δική του βαθμονόμηση.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        cb = self._switch_callback
        if cb is not None:
            self.detach()
            cb()

    def update_state(self, state: str, *_unused) -> None:
        self._hard_locked = (state == PrivacyState.HARD_LOCK)
        labels = {
            PrivacyState.CLEAR:     "CLEAR — μόνο ο εγγεγραμμένος χρήστης ορατός",
            PrivacyState.PANIC:     "PANIC — εντοπίστηκε άγνωστος παρατηρητής",
            PrivacyState.NO_FACE:   "Κανένα πρόσωπο δεν εντοπίστηκε",
            PrivacyState.HARD_LOCK: "HARD_LOCK — κλειδωμένο, δεν κλείνει η εφαρμογή",
        }
        self._status_label.setText(f"Κατάσταση: {labels.get(state, state)}")
        self._quit_btn.setEnabled(not self._hard_locked)
        if self._switch_btn is not None:
            self._switch_btn.setEnabled(not self._hard_locked)
        if self._block_btn is not None and not self._plan_finished:
            self._block_btn.setEnabled(not self._hard_locked)

    def detach(self) -> None:
        self._quit_callback = lambda: None
        self._switch_callback = None
        self._block_callback = None
        self._sync_callback = None

    def closeEvent(self, event) -> None:
        if self._hard_locked:
            event.ignore()
            QMessageBox.warning(
                self, "HARD_LOCK ενεργό",
                "Δεν μπορείς να κλείσεις την εφαρμογή όσο είναι ενεργό το "
                "HARD_LOCK.\nΠερίμενε να αναγνωριστεί ξανά ο εγγεγραμμένος "
                "χρήστης από την κάμερα.",
            )
            return
        event.accept()
        self._quit_callback()


# Main
def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic Privacy Shield")
    parser.add_argument("--enroll", metavar="NAME",
                        help="Enroll a trusted face and exit.")
    parser.add_argument("--remove-trusted", metavar="NAME",
                        help="Remove a trusted face from the database and exit.")
    parser.add_argument("--list-trusted", action="store_true",
                        help="List enrolled trusted faces and exit.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run the gaze calibration wizard and exit.")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0).")
    parser.add_argument("--fovea-radius", type=int, default=120,
                        help="Ακτίνα της καθαρής (foveated) περιοχής σε pixel "
                             "(default: 120). Ρυθμίζεται και διαδραστικά από το "
                             "παράθυρο προετοιμασίας συνεδρίας.")
    parser.add_argument("--blur-opacity", type=int, default=OverlayWindow.BLUR_OPACITY,
                        help="Αδιαφάνεια της σκούρας μάσκας, 0-255 "
                             f"(default: {OverlayWindow.BLUR_OPACITY}). Ρυθμίζεται "
                             "και διαδραστικά από το παράθυρο προετοιμασίας "
                             "συνεδρίας. Και οι δύο τιμές καταγράφονται στα "
                             "μεταδεδομένα της συνεδρίας.")
    parser.add_argument("--lock-timeout", type=float, default=5.0,
                        help="Δευτερόλεπτα συνεχούς απουσίας χρήστη πριν το αυτόματο "
                             "κλείδωμα του υπολογιστή (default: 5.0). Ορίστε 0 για "
                             "απενεργοποίηση αυτής της λειτουργίας.")
    parser.add_argument("--panic-hold", type=float, default=4.0, dest="panic_hold_s",
                        help="Παράθυρο παράτασης του PANIC σε ΔΕΥΤΕΡΟΛΕΠΤΑ μετά την "
                             "τελευταία ανίχνευση μη εξουσιοδοτημένου προσώπου "
                             "(default: 4.0). Καλύπτει στιγμιαία απώλεια ανίχνευσης "
                             "(στροφή κεφαλής, απόφραξη, motion blur). 0 = καμία "
                             "παράταση. Πειραματική ανεξάρτητη μεταβλητή — π.χ. "
                             "δοκίμασε 2.0/4.0/6.0.")
    parser.add_argument("--panic-persist", type=int, default=None,
                        help="[DEPRECATED] Παλαιό παράθυρο σε frames. Αν δοθεί, "
                             "μετατρέπεται σε δευτερόλεπτα ως panic_persist/target_fps "
                             "και υπερισχύει του --panic-hold. Διατηρείται μόνο για "
                             "συμβατότητα με παλαιότερα scripts.")
    parser.add_argument("--no-face-threshold", type=int, default=5,
                        help="Frames συνεχούς απουσίας προσώπου πριν NO_FACE "
                             "(default: 5).")
    parser.add_argument("--condition", default=None,
                        choices=["no_protection", "webcam_only", "webcam_eyetracker"],
                        help="[DEPRECATED] Παλαιά ετικέτα συνθήκης. Δεν αποτελεί "
                             "πλέον κλειδί ταξινόμησης: το ενεργό backend "
                             "καταγράφεται ανά γραμμή στη στήλη backend_active "
                             "και οι ανεξάρτητες μεταβλητές στις στήλες iv_*. "
                             "Αν δοθεί, αποθηκεύεται μόνο ως ιστορική σημείωση.")
    parser.add_argument("--block-id", default="B01",
                        help="Αναγνωριστικό του πρώτου block της συνεδρίας "
                             "(default: B01). Αλλάζει από την κονσόλα "
                             "πειραματιστή με το κουμπί «Νέο block».")
    parser.add_argument("--iv-angle", default="",
                        help="Γωνία παρατηρητή σε μοίρες — ανεξάρτητη μεταβλητή "
                             "του 2x2x2 σχεδιασμού. Γράφεται σε κάθε γραμμή.")
    parser.add_argument("--iv-distance", default="",
                        help="Απόσταση παρατηρητή σε cm — ανεξάρτητη μεταβλητή.")
    parser.add_argument("--iv-lighting", default="",
                        help="Συνθήκη φωτισμού — ανεξάρτητη μεταβλητή "
                             "(π.χ. bright / dim / backlit).")
    parser.add_argument("--profile-id", default="",
                        help="Αναγνωριστικό προφίλ χρήστη (3 προφίλ ανά "
                             "συμμετέχοντα κατά το Πρωτόκολλο Γ).")
    parser.add_argument("--gaze-mode", choices=["always", "panic_only"],
                        default="always",
                        help="'always' (default): η εκτίμηση βλέμματος τρέχει σε "
                             "κάθε καρέ, ώστε να υπάρχουν δεδομένα και για τα "
                             "διαστήματα CLEAR — ΑΠΑΡΑΙΤΗΤΟ για τη σύγκριση με "
                             "τον Tobii Pro Spectrum. 'panic_only': συμπεριφορά "
                             "v1, βλέμμα μόνο σε PANIC· χαμηλότερο φορτίο CPU "
                             "αλλά ακατάλληλο για τη μελέτη σύγκρισης.")
    parser.add_argument("--block-plan", default=None,
                        help="Αρχείο CSV με την ακολουθία των πειραματικών "
                             "κελιών (στήλες: block_id, iv_angle, iv_distance, "
                             "iv_lighting, προαιρετικά profile_id, duration_s). "
                             "Με αυτό δεν πληκτρολογείς τίποτα στη διάρκεια της "
                             "συνεδρίας: το κουμπί απλώς προχωρά στο επόμενο "
                             "κελί. Η μετάβαση παραμένει χειροκίνητη επειδή "
                             "απαιτεί φυσική αναδιάταξη.")
    parser.add_argument("--counterbalance", choices=["none", "rotate", "shuffle"],
                        default="rotate",
                        help="Αντιστάθμιση σειράς μεταξύ συμμετεχόντων. "
                             "'rotate' (default): κάθε συμμετέχων ξεκινά από "
                             "άλλο κελί, σε στυλ λατινικού τετραγώνου. "
                             "'shuffle': τυχαία αλλά αναπαραγώγιμη σειρά με "
                             "σπόρο το participant-id. 'none': ως έχει στο "
                             "αρχείο. Η σειρά που χρησιμοποιήθηκε γράφεται στα "
                             "μεταδεδομένα κάθε συνεδρίας.")
    parser.add_argument("--sync-key", default="F9",
                        help="Πλήκτρο που πατιέται αυτόματα σε κάθε δείκτη "
                             "συγχρονισμού, ώστε το Tobii Pro Lab να το "
                             "καταγράψει ως KeyboardEvent με τη δική του "
                             "χρονοσφραγίδα (default: F9). Διάλεξε πλήκτρο που "
                             "δεν κάνει τίποτα στην εφαρμογή του συμμετέχοντα.")
    parser.add_argument("--sync-interval", type=float, default=120.0,
                        help="Δευτερόλεπτα μεταξύ ΑΥΤΟΜΑΤΩΝ δεικτών "
                             "συγχρονισμού (default: 120). Χωρίς αυτό, οι μόνοι "
                             "δείκτες είναι στο άνοιγμα και στο κλείσιμο κάθε "
                             "τμήματος, που δεν επαρκούν για αξιόπιστη εκτίμηση "
                             "drift σε μακρές συνεδρίες. Δώσε 0 για "
                             "απενεργοποίηση. ΠΡΟΣΟΧΗ: κάθε δείκτης εκπέμπει το "
                             "πλήκτρο συγχρονισμού — βεβαιώσου ότι δεν κάνει "
                             "τίποτα στην εφαρμογή του συμμετέχοντα.")
    parser.add_argument("--no-sync-keystroke", action="store_true",
                        help="Απενεργοποιεί την αυτόματη εκπομπή πλήκτρου. Οι "
                             "δείκτες καταγράφονται τότε μόνο στο δικό μας "
                             "αρχείο και η ευθυγράμμιση με το Pro Lab "
                             "στηρίζεται αποκλειστικά στο 'Recording start "
                             "time' — λιγότερο ακριβής.")
    parser.add_argument("--no-auto-validate", action="store_true",
                        help="Απενεργοποιεί τον αυτόματο έλεγχο ποιότητας που "
                             "τρέχει στο τέλος κάθε συνεδρίας.")
    parser.add_argument("--participant-id", default="anon",
                        help="Αναγνωριστικό συμμετέχοντα/δοκιμασίας για το logging "
                             "(π.χ. P01_trial2). Default: 'anon'.")
    parser.add_argument("--log-dir", default="experiment_logs",
                        help="Βασικός φάκελος όπου αποθηκεύονται τα αρχεία "
                             "καταγραφής πειράματος (default: ./experiment_logs). "
                             "Μέσα σε αυτόν δημιουργείται αυτόματα ΞΕΧΩΡΙΣΤΟΣ "
                             "υποφάκελος ανά σύστημα/backend — 'webcam/', "
                             "'tobii/', 'tobii4c/' — ώστε τα CSV+MP4 των "
                             "τριών συστημάτων να μην αναμειγνύονται.")
    parser.add_argument("--no-experiment-log", action="store_true",
                        help="Απενεργοποιεί το experiment logging (χρήσιμο για "
                             "κανονική/μη-πειραματική χρήση εκτός δοκιμασιών).")
    parser.add_argument("--target-fps", type=int, default=20,
                        help="Ονομαστικό fps για το camera loop και το εγγραφόμενο "
                             "βίντεο (default: 20).")
    parser.add_argument("--no-video-record", action="store_true",
                        help="Απενεργοποιεί την καταγραφή βίντεο της κάμερας "
                             "(ενεργή by default όποτε είναι ενεργό το experiment "
                             "logging). Χρήσιμο για γρήγορα runs χωρίς να γεμίζει "
                             "ο δίσκος με μεγάλα .mp4 αρχεία.")
    parser.add_argument("--start", action="store_true",
                        help="Παρακάμπτει το γραφικό μενού εκκίνησης και ξεκινά "
                             "κατευθείαν την προστασία (χρήσιμο για scripts/αυτοματισμούς).")
    parser.add_argument("--no-face-thumbnails", action="store_true",
                        help="Απενεργοποιεί εντελώς την ένδειξη εγγεγραμμένων χρηστών "
                             "(thumbnails) πάνω-αριστερά στην οθόνη.")
    parser.add_argument("--hide-thumbnails-in-panic", action="store_true",
                        help="Κρύβει τα thumbnails όσο είναι ενεργό PANIC/HARD_LOCK "
                             "(συντηρητική επιλογή: αποφεύγει διαρροή ταυτότητας "
                             "χρήστη σε πιθανό μη εξουσιοδοτημένο παρατηρητή).")
    parser.add_argument("--gaze-backend", choices=["webcam", "tobii", "tobii4c"], default=None,
                        help="Πηγή εκτίμησης βλέμματος: 'webcam' (MediaPipe iris "
                             "tracking), 'tobii' (Tobii Pro SDK — Spectrum· απαιτεί "
                             "το πακέτο tobii-research και προηγούμενη βαθμονόμηση "
                             "μέσω του Tobii Pro Eye Tracker Manager) ή 'tobii4c' "
                             "(Tobii Eye Tracker 4C μέσω custom C++ Stream Engine "
                             "bridge — παρακάμπτει την ανάγκη Pro Upgrade Key· "
                             "απαιτεί προηγούμενο build του tobii_gaze_bridge.exe). "
                             "Αν παραλειφθεί, εμφανίζεται στην εκκίνηση γραφικό "
                             "παράθυρο επιλογής (BackendSelectDialog) — χρήσιμο για "
                             "το .exe με διπλό κλικ. Δίνοντας ρητά αυτό το flag "
                             "παρακάμπτεται το παράθυρο (χρήσιμο για scripts/πειράματα).")
    parser.add_argument("--tobii4c-exe", metavar="PATH", default=None,
                        help="Διαδρομή προς το compiled tobii_gaze_bridge.exe. "
                             "Αγνοείται αν το backend δεν είναι 'tobii4c'. "
                             "Προεπιλογή: tobii_bridge/build/Release/tobii_gaze_bridge.exe")
    parser.add_argument("--no-spectrum-ref", action="store_true",
                        help="Απενεργοποιεί την παράλληλη καταγραφή αναφοράς από το "
                             "Tobii Pro Spectrum. ΠΡΟΣΟΧΗ: χωρίς αυτήν, η συνεδρία δεν "
                             "παράγει δεδομένα σύγκρισης συστημάτων βλέμματος — "
                             "χρησιμοποίησέ το μόνο σε δοκιμές, όχι σε κανονική "
                             "συλλογή δεδομένων.")
    parser.add_argument("--spectrum-serial", default=None, metavar="SERIAL",
                        help="Ρητό serial number του Spectrum για την καταγραφή "
                             "αναφοράς. Συνιστάται: κάνει την επιλογή συσκευής "
                             "μονοσήμαντη ακόμη και αν γίνει ορατός και δεύτερος "
                             "tracker μέσω tobii_research.")
    parser.add_argument("--spectrum-ref-dir", default=None, metavar="DIR",
                        help="Φάκελος για τα αρχεία αναφοράς του Spectrum. "
                             "Προεπιλογή: <log-dir>/spectrum_reference/")
    parser.add_argument("--experimenter-screen", type=int, default=None,
                        help="Δείκτης οθόνης όπου τοποθετείται η κονσόλα "
                             "πειραματιστή (0 = πρωτεύουσα). Αν παραλειφθεί και "
                             "υπάρχουν πολλαπλές οθόνες, επιλέγεται αυτόματα η "
                             "πρώτη μη πρωτεύουσα -- ώστε η κονσόλα να μη "
                             "φαίνεται στον συμμετέχοντα ούτε στον παρατηρητή.")
    parser.add_argument("--session-tag", default=None,
                        help="Κοινή ετικέτα που παραμένει σταθερή σε ΟΛΕΣ τις "
                             "εναλλαγές backend της ίδιας συνεδρίας. Γράφεται "
                             "στα μεταδεδομένα και είναι το κλειδί με το οποίο "
                             "ενώνονται τα τρία ξεχωριστά αρχεία καταγραφής "
                             "κατά την ανάλυση. Default: το --participant-id.")
    args = parser.parse_args()

    # Περιορισμός των οπτικών παραμέτρων στα επιτρεπτά όρια του overlay.
    args.blur_opacity = max(
        OverlayWindow.BLUR_OPACITY_MIN,
        min(OverlayWindow.BLUR_OPACITY_MAX, args.blur_opacity),
    )
    args.fovea_radius = max(
        OverlayWindow.FOVEA_RADIUS_MIN,
        min(OverlayWindow.FOVEA_RADIUS_MAX, args.fovea_radius),
    )

    # Συμβατότητα: --panic-persist (frames) -> panic_hold_s (δευτερόλεπτα).
    if args.panic_persist is not None:
        fps = max(1, args.target_fps)
        args.panic_hold_s = args.panic_persist / float(fps)
        logger.warning(
            "Το --panic-persist είναι deprecated. %d frames @ %d fps "
            "μεταφράστηκαν σε παράθυρο παράτασης %.2f s. Χρησιμοποίησε "
            "απευθείας --panic-hold ΔΕΥΤΕΡΟΛΕΠΤΑ.",
            args.panic_persist, fps, args.panic_hold_s,
        )

    def _excepthook(exc_type, exc_value, exc_tb):
        logger.critical(
            "ΑΝΕΠΙΛΗΠΤΗ ΕΞΑΙΡΕΣΗ — πιθανός βίαιος τερματισμός:",
            exc_info=(exc_type, exc_value, exc_tb),
        )
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    # Διαγνωστικό banner εκκίνησης.
    print(
        f"\n=== Dynamic Privacy Shield [{APP_BUILD}] ===\n"
        f"  εκτελούμενο αρχείο : {__file__}\n"
        f"  packaged (.exe)    : {getattr(sys, 'frozen', False)}\n"
        f"  ορίσματα           : {sys.argv[1:] or '(κανένα)'}\n"
        f"  --start            : {args.start}\n",
        flush=True,
    )

    # Qt application (needed even for calibration for screen geometry)
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    screen = app.primaryScreen().geometry()
    screen_w, screen_h = screen.width(), screen.height()

    # Shared AI components
    auth = FaceAuthenticator()
    auth.load()

    if args.list_trusted:
        names = auth.list_trusted()
        if names:
            print(f"\nΕγγεγραμμένα έμπιστα πρόσωπα ({len(names)}):")
            for i, name in enumerate(names, 1):
                print(f"  {i}. {name}")
        else:
            print("Δεν υπάρχουν εγγεγραμμένα πρόσωπα.")
        sys.exit(0)

    if args.remove_trusted:
        ok = auth.remove_trusted(args.remove_trusted)
        if ok:
            print(f"✓ Το πρόσωπο '{args.remove_trusted}' αφαιρέθηκε.")
            remaining = auth.list_trusted()
            print(f"  Απομένουν: {remaining if remaining else '(κανένα)'}")
        else:
            print(f"✗ Δεν βρέθηκε πρόσωπο με όνομα '{args.remove_trusted}'.")
            print(f"  Διαθέσιμα: {auth.list_trusted()}")
        sys.exit(0)

    if args.enroll:
        run_enrollment(auth, args.enroll, camera_index=args.camera)
        sys.exit(0)

    # Κανάλι εκπομπής δεικτών συγχρονισμού προς το Pro Lab.
    sync_hook = (
        None if args.no_sync_keystroke
        else make_sync_keystroke_hook(args.sync_key)
    )
    if sync_hook is None and not args.no_sync_keystroke:
        logger.warning(
            "Οι δείκτες συγχρονισμού θα γραφτούν ΜΟΝΟ στο τοπικό αρχείο. "
            "Η ευθυγράμμιση με το Pro Lab θα είναι λιγότερο ακριβής."
        )

    # ── Καταγραφή αναφοράς από το Spectrum ───────────────────────────────
    # Ξεκινά ΜΙΑ φορά και ζει για ολόκληρη τη συνεδρία, ανεξάρτητα από το ποιο
    # backend είναι ενεργό κάθε στιγμή. Έτσι τηρείται η απόφαση πρωτοκόλλου ότι
    # το Spectrum καταγράφει συνεχώς ενώ εναλλάσσονται τα τρία συστήματα, και
    # αποφεύγεται κενό δεδομένων σε κάθε μετάβαση.
    spectrum_ref = None
    if not args.no_spectrum_ref:
        if not _SPECTRUM_REF_AVAILABLE:
            logger.warning(
                "Το module 'spectrum_reference' δεν βρέθηκε — η συνεδρία θα τρέξει "
                "ΧΩΡΙΣ καταγραφή αναφοράς."
            )
        else:
            ref_dir = (
                Path(args.spectrum_ref_dir) if args.spectrum_ref_dir
                else Path(args.log_dir) / "spectrum_reference"
            )
            spectrum_ref = SpectrumReferenceRecorder(
                out_dir=ref_dir,
                session_tag=args.session_tag or args.participant_id,
                participant_id=args.participant_id,
                screen_width=screen_w, screen_height=screen_h,
                serial_number=args.spectrum_serial,
            )
            try:
                spectrum_ref.start()
            except Exception as exc:
                logger.exception("Αποτυχία εκκίνησης καταγραφής αναφοράς.")
                answer = QMessageBox.question(
                    None, "Χωρίς καταγραφή αναφοράς",
                    "Δεν ήταν δυνατή η εκκίνηση της καταγραφής αναφοράς από το "
                    f"Tobii Pro Spectrum:\n\n{exc}\n\n"
                    "Αν συνεχίσεις, η συνεδρία ΔΕΝ θα παράγει δεδομένα σύγκρισης "
                    "συστημάτων βλέμματος.\n\nΣυνέχεια χωρίς αναφορά;",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    sys.exit(1)
                spectrum_ref = None
            else:
                # Ο recorder πρέπει να κλείσει καθαρά σε ΚΑΘΕ έξοδο — το main()
                # τερματίζει με sys.exit() σε τρία διαφορετικά σημεία.
                atexit.register(spectrum_ref.stop)
                print(
                    f"[Spectrum reference] {spectrum_ref.csv_path}\n"
                    f"[Spectrum reference] μεταδεδομένα: {spectrum_ref.meta_path}"
                )

    # Κατάσταση ταξινόμησης που επιβιώνει των εναλλαγών backend.
    segment_index = 0
    block_plan = None
    if args.block_plan:
        try:
            block_plan = BlockPlan.from_csv(
                args.block_plan, args.counterbalance, args.participant_id)
            logger.info(
                "[ΡΟΗ] Πρόγραμμα block: %d κελιά, αντιστάθμιση '%s'. Σειρά: %s",
                block_plan.total, args.counterbalance,
                " -> ".join(block_plan.as_metadata()["order"]),
            )
        except Exception as exc:
            QMessageBox.critical(
                None, "Πρόγραμμα block",
                f"Δεν μπόρεσα να διαβάσω το '{args.block_plan}':\n{exc}\n\n"
                "Η συνεδρία θα ξεκινήσει με χειροκίνητο ορισμό block."
            )
            logger.exception("Αποτυχία φόρτωσης προγράμματος block.")
            block_plan = None

    if block_plan is not None:
        first = block_plan.current()
        block_state = {
            "block_id":    first.get("block_id", ""),
            "iv_angle":    first.get("iv_angle", ""),
            "iv_distance": first.get("iv_distance", ""),
            "iv_lighting": first.get("iv_lighting", ""),
            "profile_id":  first.get("profile_id", "") or args.profile_id,
        }
    else:
        block_state = {
            "block_id":    args.block_id,
            "iv_angle":    args.iv_angle,
            "iv_distance": args.iv_distance,
            "iv_lighting": args.iv_lighting,
            "profile_id":  args.profile_id,
        }

    while True:
        
        if args.gaze_backend is None:
            select_dlg = BackendSelectDialog(
                tobii_available=_TOBII_AVAILABLE,
                tobii4c_available=_TOBII4C_AVAILABLE,
                default_backend="webcam",
            )
            result = select_dlg.exec()
            if result != QDialog.DialogCode.Accepted or select_dlg.chosen_backend is None:
                logger.info("Ο χρήστης επέλεξε Έξοδος στο αρχικό παράθυρο επιλογής backend.")
                sys.exit(0)
            args.gaze_backend = select_dlg.chosen_backend

        try:
            if args.gaze_backend == "tobii":
                if not _TOBII_AVAILABLE:
                    sys.exit(
                        "Ζητήθηκε --gaze-backend tobii αλλά το πακέτο 'tobii-research' "
                        "δεν είναι εγκατεστημένο.\n"
                        "pip install tobii-research"
                    )
                tracker = TobiiGazeTracker(
                    screen_width=screen_w, screen_height=screen_h,
                    fovea_radius=args.fovea_radius,
                )
                backend_label = f"Tobii Pro eye tracker ({tracker._tracker.model})"
                logger.info("Gaze backend: Tobii Pro eye tracker.")
            elif args.gaze_backend == "tobii4c":
                if not _TOBII4C_AVAILABLE:
                    sys.exit(
                        "Ζητήθηκε --gaze-backend tobii4c αλλά δεν βρέθηκε το module "
                        "'tobii4c_gaze_tracker' (έλεγξε ότι το tobii4c_gaze_tracker.py "
                        "βρίσκεται στον ίδιο φάκελο)."
                    )
                tobii4c_kwargs = {}
                if args.tobii4c_exe:
                    tobii4c_kwargs["exe_path"] = args.tobii4c_exe
                tracker = Tobii4CGazeTracker(
                    screen_width=screen_w, screen_height=screen_h,
                    fovea_radius=args.fovea_radius, **tobii4c_kwargs,
                )
                backend_label = "Tobii 4C eye tracker (Stream Engine bridge)"
                logger.info("Gaze backend: Tobii 4C (Stream Engine bridge).")
            else:
                tracker = GazeTracker(
                    screen_width=screen_w,
                    screen_height=screen_h,
                    fovea_radius=args.fovea_radius,  # default now 120px
                )
                backend_label = f"Webcam (κάμερα index {args.camera})"
                logger.info("Gaze backend: webcam (MediaPipe iris tracking).")
        except Exception as exc:
            logger.exception("Αποτυχία αρχικοποίησης backend '%s'.", args.gaze_backend)
            QMessageBox.critical(
                None, "Αποτυχία σύνδεσης",
                f"Δεν ήταν δυνατή η αρχικοποίηση του συστήματος "
                f"'{args.gaze_backend}'.\n\n{exc}\n\n"
                "Έλεγξε τη σύνδεση και επίλεξε ξανά σύστημα καταγραφής.",
            )
            args.gaze_backend = None
            continue

        if args.calibrate:
            if args.gaze_backend != "webcam":
                sys.exit(
                    f"\nΤο backend '{args.gaze_backend}' δεν χρησιμοποιεί αυτόν τον "
                    "wizard.\nΗ βαθμονόμηση γίνεται εκτός εφαρμογής, μέσω του "
                    "αντίστοιχου Tobii λογισμικού (Eye Tracker Manager / Eye "
                    "Tracking Core)."
                )
            run_calibration_wizard(tracker, screen_w, screen_h, camera_index=args.camera)
            sys.exit(0)

        return_to_backend_select = False

        if not args.start:
            while True:
                dlg = LauncherDialog(
                    auth,
                    gaze_backend=args.gaze_backend,
                    backend_label=backend_label,
                    screen_w=screen_w,
                    screen_h=screen_h,
                    fovea_radius=args.fovea_radius,
                    blur_opacity=args.blur_opacity,
                )
                result = dlg.exec()
                action = dlg.chosen_action if result == QDialog.DialogCode.Accepted else LauncherDialog.ACTION_BACK

                args.fovea_radius = dlg.fovea_radius
                args.blur_opacity = dlg.blur_opacity

                if action == LauncherDialog.ACTION_BACK:
                    return_to_backend_select = True
                    break

                elif action == LauncherDialog.ACTION_START:
                    break   # βγαίνουμε από το μενού· η προστασία ξεκινά παρακάτω

                elif action == LauncherDialog.ACTION_ENROLL:
                    name, ok = QInputDialog.getText(
                        None, "Εγγραφή νέου χρήστη", "Όνομα:",
                    )
                    if ok and name.strip():
                        run_enrollment(auth, name.strip(), camera_index=args.camera)
                        auth.load()   # ανανέωση in-memory κατάστασης μετά το save()
                    continue

                elif action == LauncherDialog.ACTION_REMOVE:
                    names = auth.list_trusted()
                    if not names:
                        QMessageBox.information(
                            None, "Αφαίρεση χρήστη",
                            "Δεν υπάρχουν εγγεγραμμένοι χρήστες.",
                        )
                        continue
                    name, ok = QInputDialog.getItem(
                        None, "Αφαίρεση χρήστη", "Επίλεξε χρήστη προς αφαίρεση:",
                        names, 0, False,
                    )
                    if ok and name:
                        auth.remove_trusted(name)
                    continue

                elif action == LauncherDialog.ACTION_CALIBRATE:
                    if args.gaze_backend != "webcam":
                        QMessageBox.information(
                            None, "Βαθμονόμηση",
                            f"Το backend '{args.gaze_backend}' δεν χρησιμοποιεί αυτόν "
                            "τον wizard.\nΗ βαθμονόμηση γίνεται εκτός εφαρμογής, μέσω "
                            "του αντίστοιχου Tobii λογισμικού.",
                        )
                        continue
                    run_calibration_wizard(tracker, screen_w, screen_h, camera_index=args.camera)
                    continue

        if return_to_backend_select:
           
            if hasattr(tracker, "stop"):
                try:
                    tracker.stop()
                except Exception:
                    logger.exception(
                        "tracker.stop() απέτυχε κατά την επιστροφή στο "
                        "παράθυρο επιλογής backend."
                    )
            args.gaze_backend = None
            continue   

       

        # Normal operation
        if not auth.list_trusted():
            print(
                "\nWARNING: Δεν υπάρχουν εγγεγραμμένα πρόσωπα.\n"
                "Εκτέλεσε:  python main.py --enroll Όνομα\n"
                "Μπορείς να εγγράψεις έως 5 (ή περισσότερα) έμπιστα πρόσωπα.\n"
                "Το σύστημα θα τρέχει σε NO_FACE mode (πλήρες θόλωμα).\n"
            )
        else:
            names = auth.list_trusted()
            print(f"\nΈμπιστα πρόσωπα ({len(names)}): {', '.join(names)}")

        overlay = OverlayWindow(
            screen_w, screen_h, fovea_radius=args.fovea_radius,
            show_trusted_thumbnails=not args.no_face_thumbnails,
            hide_thumbnails_in_panic=args.hide_thumbnails_in_panic,
            blur_opacity=args.blur_opacity,
        )
        overlay.set_trusted_thumbnails(auth)
        overlay.show()
        overlay.activateWindow()
        overlay.setFocus()  
                              

        exp_logger = None
        video_recorder = None
        if not args.no_experiment_log:
            # Ξεχωριστός υποφάκελος ανά σύστημα (webcam / tobii / tobii4c)
           
            session_log_dir = Path(args.log_dir) / args.gaze_backend
            segment_index += 1
            backend_canonical = BACKEND_CANONICAL.get(
                args.gaze_backend, args.gaze_backend)
            exp_logger = ExperimentLogger(
                participant_id=args.participant_id,
                log_dir=session_log_dir,
                session_tag=f"{args.session_tag or args.participant_id}"
                            f"-seg{segment_index:02d}-{args.gaze_backend}",
                app_version=APP_BUILD,
                auto_validate=not args.no_auto_validate,
                index_path=Path(args.log_dir) / "sessions_index.csv",
                sync_hook=sync_hook,
                config={
                    "screen_w": screen_w, "screen_h": screen_h,
                    "fovea_radius": args.fovea_radius,
                    "blur_opacity": args.blur_opacity,
                    "lock_timeout": args.lock_timeout,
                    "panic_hold_s": args.panic_hold_s,
                    "no_face_threshold": args.no_face_threshold,
                    "camera_index": args.camera,
                    "target_fps": args.target_fps,
                    "gaze_backend": args.gaze_backend,
                    "gaze_backend_label": backend_label,
                    "backend_canonical": backend_canonical,
                    "gaze_mode": args.gaze_mode,
                    "auth_every_n_frames": 2,
                    # Κλειδί ένωσης των διαδοχικών τμημάτων της ίδιας
                    # συνεδρίας κατά την ανάλυση.
                    "session_tag": args.session_tag or args.participant_id,
                    "segment_index": segment_index,
                    "condition_legacy": args.condition or "",
                    "sync_key": "" if sync_hook is None else args.sync_key.upper(),
                    "sync_interval_s": args.sync_interval,
                    "block_plan": (
                        block_plan.as_metadata() if block_plan is not None else None
                    ),
                },
            )
            # Το block ορίζεται ΠΡΙΝ γραφτεί το πρώτο καρέ, ώστε καμία γραμμή
            # να μη μείνει χωρίς ταξινόμηση.
            exp_logger.sync_marker("segment_open", external_channel="prolab")
            # Κάθε δείγμα αναφοράς φέρει πλέον το backend που δοκιμαζόταν τη
            # στιγμή της καταγραφής· ο διαχωρισμός των τριών συγκρίσεων στην
            # ανάλυση γίνεται με groupby, χωρίς χειροκίνητη αντιστοίχιση.
            if spectrum_ref is not None:
                spectrum_ref.set_segment(
                    segment_index, backend_canonical, block_state["block_id"],
                )
                spectrum_ref.marker(f"segment_open:{exp_logger.session_id}")
            exp_logger.start_block(
                block_state["block_id"], backend=backend_canonical,
                iv_angle=block_state["iv_angle"],
                iv_distance=block_state["iv_distance"],
                iv_lighting=block_state["iv_lighting"],
                profile_id=block_state["profile_id"],
                note=f"backend_label={backend_label}",
            )
            print(f"[Experiment log] session_id = {exp_logger.session_id}\n"
                  f"  φάκελος συστήματος: {session_log_dir.resolve()}\n"
                  f"  δεδομένα (frames+events): {exp_logger._csv_path.resolve()}\n"
                  f"  μεταδεδομένα: {exp_logger._meta_path.resolve()}")

            if not args.no_video_record:
                video_path = session_log_dir / f"session_{exp_logger.session_id}.mp4"
                video_recorder = VideoRecorder(video_path, fps=args.target_fps)
                print(f"  → βίντεο κάμερας: {video_path.resolve()}")

        camera_thread = CameraThread(
            auth, tracker, camera_index=args.camera, target_fps=args.target_fps,
            lock_timeout=args.lock_timeout,
            panic_hold_s=args.panic_hold_s, no_face_threshold=args.no_face_threshold,
            exp_logger=exp_logger, video_recorder=video_recorder,
            gaze_always=(args.gaze_mode == "always"),
            frame_sync_hook=(
                spectrum_ref.frame_tick if spectrum_ref is not None else None
            ),
        )

        # Μέτρηση καθυστέρησης έως τη σχεδίαση της ασπίδας.
        def _on_overlay_painted(prev_state, state, cam_idx, t_capture, t_painted):
            if exp_logger is None:
                return
            exp_logger.log_event(
                "OVERLAY_PAINTED", from_state=prev_state, to_state=state,
                camera_frame_index=cam_idx,
                t_capture_perf=t_capture,
                t_overlay_painted_perf=t_painted,
            )

        overlay.paint_probe_callback = _on_overlay_painted

        camera_thread.state_updated.connect(overlay.update_state)
        camera_thread.start()


        _cleanup_done = {"value": False}

        def _cleanup() -> None:
            if _cleanup_done["value"]:
                return
            _cleanup_done["value"] = True

            try:
                camera_thread.stop()
            except Exception:
                logger.exception("camera_thread.stop() απέτυχε κατά το cleanup.")

            if hasattr(tracker, "stop"):
                try:
                    tracker.stop()
                except Exception:
                    logger.exception("tracker.stop() απέτυχε κατά το cleanup.")

            # ΣΗΜΕΙΩΣΗ: εδώ γράφεται ΜΟΝΟ δείκτης. Ο recorder αναφοράς ΔΕΝ
            # σταματά στο τέλος κάθε segment — αν σταματούσε, κάθε εναλλαγή
            # backend θα άφηνε κενό λίγων δευτερολέπτων (νέα ανακάλυψη συσκευής)
            # ακριβώς στο σημείο μετάβασης. Ο τερματισμός γίνεται μία φορά,
            # μέσω atexit.
            if spectrum_ref is not None:
                try:
                    spectrum_ref.marker("segment_close")
                except Exception:
                    logger.exception("spectrum_ref.marker() απέτυχε κατά το cleanup.")

            video_summary = None
            if video_recorder is not None:
                try:
                    video_summary = video_recorder.close()
                except Exception:
                    logger.exception("video_recorder.close() απέτυχε κατά το cleanup.")

            if exp_logger:
                try:
                    exp_logger.sync_marker("segment_close", external_channel="prolab")
                    exp_logger.end_block(note="segment_cleanup")
                except Exception:
                    logger.exception("Κλείσιμο block απέτυχε κατά το cleanup.")
                try:
                    exp_logger.close(
                        summary={"video": video_summary} if video_summary else None
                    )
                except Exception:
                    logger.exception("exp_logger.close() απέτυχε κατά το cleanup.")

            logger.info("Cleanup ολοκληρώθηκε.")

        session_outcome = {"value": None}

        def _request_outcome(outcome: str) -> None:
            if session_outcome["value"] is not None:
                logger.info(
                    "[ΡΟΗ] Αγνοήθηκε δεύτερο αίτημα '%s' — ισχύει το '%s'.",
                    outcome, session_outcome["value"],
                )
                return
            session_outcome["value"] = outcome
            logger.info("[ΡΟΗ] Έκβαση συνεδρίας: %s", outcome)
            try:
                _cleanup()
            except Exception:
                logger.exception("Σφάλμα στο cleanup — η ροή συνεχίζεται.")
            app.quit()

        def _back_to_backend_select() -> None:
            _request_outcome("restart")

        def _finish_session() -> None:
            """Οριστικός τερματισμός — κουμπί «Έξοδος» της κονσόλας."""
            _request_outcome("exit")

        def _log_sync(label: str) -> None:
            if exp_logger is None:
                logger.warning("Δείκτης συγχρονισμού χωρίς ενεργό logging.")
                return
            exp_logger.sync_marker(label, external_channel="prolab")
            logger.info("[ΣΥΓΧΡΟΝΙΣΜΟΣ] Δείκτης '%s' καταγράφηκε.", label)

        def _refresh_block_ui() -> None:
            control_window.set_block_label(
                f"{block_state['block_id']} — {block_state['iv_angle']}° / "
                f"{block_state['iv_distance']}cm / {block_state['iv_lighting']}"
            )
            if block_plan is not None:
                control_window.set_next_block(
                    BlockPlan.describe(block_plan.peek_next()),
                    progress=f"{block_plan.index + 1}/{block_plan.total}",
                )

        def _new_block() -> None:
            """
            Προχωρά στο επόμενο κελί. Με πρόγραμμα δεν ζητά τίποτα· χωρίς
            πρόγραμμα ανοίγει τον διάλογο χειροκίνητου ορισμού.
            """
            if block_plan is not None:
                nxt = block_plan.advance()
                if nxt is None:
                    if exp_logger is not None:
                        exp_logger.end_block(note="plan_complete")
                        exp_logger.sync_marker("plan_complete",
                                               external_channel="prolab")
                    control_window.disable_block_button(
                        "Το πρόγραμμα ολοκληρώθηκε. Πάτα Έξοδος για να κλείσει "
                        "η συνεδρία και να τρέξει ο έλεγχος ποιότητας."
                    )
                    logger.info("[ΡΟΗ] Όλα τα blocks του προγράμματος εκτελέστηκαν.")
                    return
                block_state.update({
                    "block_id":    nxt.get("block_id", ""),
                    "iv_angle":    nxt.get("iv_angle", ""),
                    "iv_distance": nxt.get("iv_distance", ""),
                    "iv_lighting": nxt.get("iv_lighting", ""),
                    "profile_id":  nxt.get("profile_id", "") or block_state["profile_id"],
                })
            else:
                dlg = BlockDialog(**block_state)
                if dlg.exec() != QDialog.DialogCode.Accepted or dlg.values is None:
                    return
                block_state.update(dlg.values)

            if exp_logger is not None:
                exp_logger.end_block(note="operator_requested")
                exp_logger.sync_marker("block_boundary", external_channel="prolab")
                exp_logger.start_block(
                    block_state["block_id"],
                    backend=BACKEND_CANONICAL.get(args.gaze_backend, args.gaze_backend),
                    iv_angle=block_state["iv_angle"],
                    iv_distance=block_state["iv_distance"],
                    iv_lighting=block_state["iv_lighting"],
                    profile_id=block_state["profile_id"],
                )
            _refresh_block_ui()
            _restart_block_timer()
            logger.info("[ΡΟΗ] Νέο block: %s", block_state)

        def _switch_backend() -> None:
            if exp_logger is not None:
                try:
                    exp_logger.log_event(
                        "BACKEND_SWITCH_REQUESTED",
                        from_state=args.gaze_backend, to_state="pending",
                    )
                except Exception:
                    logger.exception("Καταγραφή BACKEND_SWITCH_REQUESTED απέτυχε.")
            logger.info("Ζητήθηκε αλλαγή backend από '%s'.", args.gaze_backend)
            _back_to_backend_select()

        # Persistent control window 
        block_timer_state = {"deadline": None}

        def _restart_block_timer() -> None:
            if block_plan is None:
                return
            dur = block_plan.duration_s()
            block_timer_state["deadline"] = (
                time.perf_counter() + dur if dur else None
            )

        def _tick_block_timer() -> None:
            dl = block_timer_state["deadline"]
            if dl is None:
                return
            control_window.set_block_finished(max(0.0, dl - time.perf_counter()))

        control_window = ControlWindow(
            _finish_session,
            switch_callback=_switch_backend,
            backend_label=backend_label,
            experimenter_screen=args.experimenter_screen,
            block_callback=_new_block if exp_logger is not None else None,
            sync_callback=_log_sync if exp_logger is not None else None,
            block_label=(
                f"{block_state['block_id']} — {block_state['iv_angle']}° / "
                f"{block_state['iv_distance']}cm / {block_state['iv_lighting']}"
            ),
        )
        _refresh_block_ui()
        _restart_block_timer()
        block_timer = QTimer()
        block_timer.timeout.connect(_tick_block_timer)
        block_timer.start(1000)

        # Αυτόματοι δείκτες συγχρονισμού. Χωρίς αυτούς, μια συνεδρία χωρίς
        # αλλαγές block έχει μόνο δύο δείκτες — αρκετοί για σταθερή απόκλιση,
        # ανεπαρκείς για να φανεί αν τα ρολόγια αποκλίνουν προοδευτικά.
        sync_timer = None
        if args.sync_interval and args.sync_interval > 0 and exp_logger is not None:
            sync_timer = QTimer()
            sync_timer.timeout.connect(lambda: _log_sync("periodic"))
            sync_timer.start(int(args.sync_interval * 1000))
            logger.info("Αυτόματος δείκτης συγχρονισμού κάθε %.0f s.",
                        args.sync_interval)

        control_window.show()
        camera_thread.state_updated.connect(control_window.update_state)

        tray = None
        if QSystemTrayIcon.isSystemTrayAvailable():
            tray = QSystemTrayIcon(_make_tray_icon())
            tray.setToolTip("Dynamic Privacy Shield — ενεργό")
            menu = QMenu()
            quit_action = QAction("Έξοδος", menu)
            quit_action.triggered.connect(_finish_session)
            menu.addAction(quit_action)
            tray.setContextMenu(menu)
            tray.show()


            def _sync_tray_with_state(state: str, *_unused) -> None:
                is_hard_locked = state == PrivacyState.HARD_LOCK
                quit_action.setEnabled(not is_hard_locked)
                tray.setToolTip(
                    "Dynamic Privacy Shield — HARD_LOCK ενεργό (Έξοδος κλειδωμένη)"
                    if is_hard_locked else
                    "Dynamic Privacy Shield — ενεργό"
                )

            camera_thread.state_updated.connect(_sync_tray_with_state)
        else:
            logger.warning(
                "System tray μη διαθέσιμο σε αυτό το περιβάλλον — μόνο το ESC "
                "(εκτός HARD_LOCK) είναι διαθέσιμο για έξοδο."
            )

        lock_msg = (
            f"auto-lock after {args.lock_timeout:.1f}s absence"
            if args.lock_timeout and args.lock_timeout > 0
            else "auto-lock disabled"
        )
        logger.info(
            "Privacy Shield active – %dx%d, fovea radius %dpx, %s. "
            "Screen is CLEAR when only trusted users are present; "
            "foveated blur activates on unknown observer detection.",
            screen_w, screen_h, args.fovea_radius, lock_msg,
        )
        print(
            "Privacy Shield running.\n"
            "Επιστροφή στην επιλογή συστήματος: κλείσε το παράθυρο 'Dynamic "
            "Privacy Shield — Ενεργό' (Χ πάνω δεξιά) ή δεξί κλικ στο tray "
            "icon -> 'Έξοδος'.\n"
            "Σημείωση: καμία από τις δύο μεθόδους δεν λειτουργεί όσο είναι "
            "ενεργό το HARD_LOCK — μόνο η επανεμφάνιση του εγγεγραμμένου "
            "χρήστη στην κάμερα το αίρει."
        )

        exit_code = app.exec()
        logger.info(
            "[ΡΟΗ] app.exec() επέστρεψε %s | έκβαση=%s",
            exit_code, session_outcome["value"],
        )
        _cleanup()

        overlay.close()
        control_window.detach()
        control_window.close()
        if tray is not None:
            tray.hide()

        outcome = session_outcome["value"]

        if outcome == "restart":
            args.gaze_backend = None
            logger.info("[ΡΟΗ] Επιστροφή στο BackendSelectDialog.")
            continue   # πίσω στην αρχή του εξωτερικού βρόχου -> BackendSelectDialog

        if outcome == "exit":
            logger.info("Τερματισμός συνεδρίας από την κονσόλα πειραματιστή.")
            sys.exit(0)

        logger.warning(
            "[ΡΟΗ] Καμία καταγεγραμμένη έκβαση — η έξοδος προήλθε από αλλού "
            "(π.χ. ESC στο overlay). Τερματισμός με κωδικό %s.", exit_code,
        )
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
