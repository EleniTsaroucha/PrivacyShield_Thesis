import logging
import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from main import PrivacyState, OverlayWindow
from tobii_gaze_tracker import TobiiGazeTracker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
logger = logging.getLogger(__name__)

POLL_INTERVAL_MS = 33   # ~30Hz — αρκετό για ομαλή οπτική ανάδραση


def main() -> None:
    app = QApplication(sys.argv)
    screen = app.primaryScreen().geometry()
    screen_w, screen_h = screen.width(), screen.height()
    logger.info("Ανάλυση οθόνης: %dx%d", screen_w, screen_h)

    tracker = TobiiGazeTracker(screen_width=screen_w, screen_height=screen_h)

    overlay = OverlayWindow(
        screen_w, screen_h, fovea_radius=120, show_trusted_thumbnails=False,
    )
    overlay.show()
    overlay.activateWindow()
    overlay.setFocus()   # απαραίτητο ώστε το ESC να δουλέψει αμέσως

    def _tick() -> None:
        gaze = tracker.process_frame(None)   # το backend Tobii αγνοεί το frame
        overlay.update_state(PrivacyState.PANIC, gaze.x, gaze.y, gaze.confidence)

    timer = QTimer()
    timer.timeout.connect(_tick)
    timer.start(POLL_INTERVAL_MS)

    logger.info(
        "Demo ενεργό — το θόλωμα ακολουθεί το βλέμμα από τον Tobii. "
        "Πάτα ESC πάνω στο overlay για έξοδο."
    )

    exit_code = app.exec()
    tracker.stop()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
