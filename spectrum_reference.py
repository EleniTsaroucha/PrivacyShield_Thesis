"""
spectrum_reference.py
─────────────────────
Συνεχής καταγραφή αναφοράς (reference recording) από το Tobii Pro Spectrum,
ΜΕΣΑ στην ίδια διεργασία με το main.py.

Σκεπτικό
────────
Το Pro Lab δεν μπορεί να συνυπάρξει με το Tobii Experience / Gaming stack που
απαιτεί ο 4C. Αντί να τρέχει ξεχωριστό script (main3.py) που ανοίγει ΚΑΙ δεύτερο
tobii_gaze_bridge.exe για τον 4C — πράγμα που δημιουργεί δεύτερο καταναλωτή της
ίδιας συσκευής — εδώ καταγράφεται ΜΟΝΟ το Spectrum, ως παθητικό κανάλι αναφοράς.
Τα δεδομένα του εκάστοτε backend (webcam / 4C / Spectrum) εξακολουθούν να
γράφονται από τον ExperimentLogger του main.py.

Επειδή οι δύο ροές ζουν πλέον στην ίδια διεργασία, ο συγχρονισμός γίνεται με μία
κοινή συνάρτηση χρόνου (now_us) και δεν απαιτεί δείκτες πληκτρολογίου.

Χρήση
─────
    from spectrum_reference import SpectrumReferenceRecorder, now_us

    ref = SpectrumReferenceRecorder(
        out_dir=Path(args.log_dir) / "spectrum_reference",
        session_tag=args.session_tag or args.participant_id,
        participant_id=args.participant_id,
        screen_width=screen_w, screen_height=screen_h,
        serial_number=args.spectrum_serial,   # συνιστάται ρητό serial
    )
    ref.start()                       # μία φορά, πριν τον βρόχο επιλογής backend
    ...
    ref.set_segment(segment_index, backend_canonical, block_id)   # σε κάθε segment
    ref.marker("segment_open")
    ...
    ref.marker("segment_close")
    ...
    ref.stop()                        # μία φορά, στο τέλος της συνεδρίας
"""

from __future__ import annotations

import csv
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import tobii_research as tr

logger = logging.getLogger(__name__)


# ── Κοινός άξονας χρόνου ─────────────────────────────────────────────────────
def now_us() -> int:
    """
    Χρονοσφραγίδα στο ΙΔΙΟ ρολόι με το gaze_data['system_time_stamp'] του Pro SDK
    (μικροδευτερόλεπτα). Χρησιμοποίησέ την ΚΑΙ στον ExperimentLogger, ώστε όλες οι
    ροές της συνεδρίας να βρίσκονται σε έναν άξονα χωρίς post-hoc αντιστοίχιση.
    """
    return tr.get_system_time_stamp()


def clock_pair() -> dict:
    """Ζεύγος (wall clock, Tobii clock) για εκτίμηση διολίσθησης εκ των υστέρων."""
    t0 = time.time()
    tob = tr.get_system_time_stamp()
    t1 = time.time()
    return {"unix_time": (t0 + t1) / 2.0, "tobii_us": tob, "read_span_s": t1 - t0}


CSV_HEADER = [
    "row_type",            # SAMPLE | MARKER | FRAME
    "sync_us",             # κοινός άξονας (Tobii system clock, μs)
    "device_time_stamp",   # ρολόι συσκευής (μs)
    "unix_time",           # μόνο για διασταύρωση/ανάγνωση από άνθρωπο
    "segment_index",
    "backend",             # ποιο backend δοκιμαζόταν όταν καταγράφηκε το δείγμα
    "block_id",
    "marker_label",
    "camera_frame_index",  # γέφυρα προς τις γραμμές του ExperimentLogger
    "left_gaze_x", "left_gaze_y", "left_gaze_validity",
    "right_gaze_x", "right_gaze_y", "right_gaze_validity",
    "left_pupil_diameter", "left_pupil_validity",
    "right_pupil_diameter", "right_pupil_validity",
    # Απαραίτητα για μετατροπή σφάλματος σε μοίρες οπτικής γωνίας:
    "left_origin_x", "left_origin_y", "left_origin_z", "left_origin_validity",
    "right_origin_x", "right_origin_y", "right_origin_z", "right_origin_validity",
]


class SpectrumReferenceRecorder:

    def __init__(
        self,
        out_dir: Path,
        session_tag: str,
        participant_id: str,
        screen_width: int,
        screen_height: int,
        serial_number: Optional[str] = None,
        queue_size: int = 20000,
        discovery_attempts: int = 5,
        discovery_wait_s: float = 2.0,
    ):
        self.out_dir = Path(out_dir)
        self.session_tag = session_tag
        self.participant_id = participant_id
        self.screen_w = screen_width
        self.screen_h = screen_height
        self.serial_number = serial_number
        self._discovery_attempts = discovery_attempts
        self._discovery_wait_s = discovery_wait_s

        self._tracker = None
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._writer_thread: Optional[threading.Thread] = None
        self._fh = None
        self._writer = None

        self._state_lock = threading.Lock()
        self._segment_index = 0
        self._backend = ""
        self._block_id = ""

        self.dropped_samples = 0
        self.written_samples = 0
        self.csv_path: Optional[Path] = None
        self.meta_path: Optional[Path] = None

    # ── Ανακάλυψη συσκευής ───────────────────────────────────────────────
    def _find_spectrum(self):
        for attempt in range(1, self._discovery_attempts + 1):
            found = tr.find_all_eyetrackers()
            if self.serial_number:
                matches = [t for t in found if t.serial_number == self.serial_number]
            else:
                matches = [t for t in found if "spectrum" in t.model.lower()]
            logger.info(
                "[SpectrumRef] Απόπειρα %d/%d — ορατά: %s",
                attempt, self._discovery_attempts,
                [(t.model, t.serial_number) for t in found],
            )
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise RuntimeError(
                    "Περισσότερες από μία συσκευές ταιριάζουν. Δώσε ρητό "
                    "serial_number ώστε η επιλογή να είναι μονοσήμαντη: "
                    f"{[(t.model, t.serial_number) for t in matches]}"
                )
            if attempt < self._discovery_attempts:
                time.sleep(self._discovery_wait_s)

        raise RuntimeError(
            "Δεν βρέθηκε το Tobii Pro Spectrum μέσω tobii_research. Έλεγξε τη "
            "σύνδεση Ethernet, ότι το Eye Tracker Manager το αναγνωρίζει, και ότι "
            "ΔΕΝ είναι ανοιχτό το Tobii Pro Lab (κρατά αποκλειστική πρόσβαση)."
        )

    # ── Κύκλος ζωής ──────────────────────────────────────────────────────
    def start(self) -> None:
        self._tracker = self._find_spectrum()

        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = f"spectrumref_{self.participant_id}_{self.session_tag}_{stamp}"
        self.csv_path = self.out_dir / f"{base}.csv"
        self.meta_path = self.out_dir / f"{base}.meta.json"

        self._fh = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(CSV_HEADER)

        self._write_metadata()

        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="SpectrumRef-writer", daemon=True
        )
        self._writer_thread.start()

        self._tracker.subscribe_to(
            tr.EYETRACKER_GAZE_DATA, self._on_gaze, as_dictionary=True
        )
        logger.info(
            "[SpectrumRef] Ενεργή καταγραφή αναφοράς: %s (serial=%s) → %s",
            self._tracker.model, self._tracker.serial_number, self.csv_path,
        )

    def stop(self) -> None:
        if self._stop_event.is_set():
            return
        if self._tracker is not None:
            try:
                # ΠΡΟΣΟΧΗ: πέρασε ΠΑΝΤΑ το callback. Χωρίς αυτό το Pro SDK
                # αποσυνδέει ΟΛΟΥΣ τους συνδρομητές του stream — δηλαδή και το
                # TobiiGazeTracker του main.py, αν τρέχει το Spectrum backend.
                self._tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self._on_gaze)
            except Exception:
                logger.exception("[SpectrumRef] Αποτυχία unsubscribe.")

        self._append_metadata_close()
        self._stop_event.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5.0)
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
        logger.info(
            "[SpectrumRef] Τέλος: %d δείγματα γραμμένα, %d απορρίφθηκαν (queue full).",
            self.written_samples, self.dropped_samples,
        )

    # ── Πλαισίωση (segment / block) ──────────────────────────────────────
    def set_segment(self, segment_index: int, backend: str, block_id: str = "") -> None:
        with self._state_lock:
            self._segment_index = segment_index
            self._backend = backend
            self._block_id = block_id or ""
        self.marker(f"segment_set:{backend}")

    def frame_tick(self, camera_frame_index: int) -> None:
        """
        Καλείται μία φορά ανά καρέ από το CameraThread.

        Γράφει γραμμή FRAME με τη χρονοσφραγίδα του ΚΟΙΝΟΥ ρολογιού και τον
        δείκτη καρέ. Επειδή ο ExperimentLogger καταγράφει ήδη το
        camera_frame_index σε κάθε γραμμή του, η ένωση των δύο αρχείων στην
        ανάλυση γίνεται με απλό join σε αυτή τη στήλη — χωρίς καμία
        τροποποίηση του experiment_logger.py και χωρίς δείκτες πληκτρολογίου.
        """
        with self._state_lock:
            seg, backend, block = self._segment_index, self._backend, self._block_id
        self._enqueue({
            "row_type": "FRAME",
            "sync_us": now_us(),
            "unix_time": time.time(),
            "segment_index": seg, "backend": backend, "block_id": block,
            "camera_frame_index": camera_frame_index,
        })

    def marker(self, label: str) -> None:
        with self._state_lock:
            seg, backend, block = self._segment_index, self._backend, self._block_id
        row = {
            "row_type": "MARKER",
            "sync_us": now_us(),
            "unix_time": time.time(),
            "segment_index": seg, "backend": backend, "block_id": block,
            "marker_label": label,
        }
        self._enqueue(row)

    # ── Callback: τρέχει σε thread του SDK — καμία I/O εδώ ────────────────
    def _on_gaze(self, g: dict) -> None:
        with self._state_lock:
            seg, backend, block = self._segment_index, self._backend, self._block_id

        lx, ly = g["left_gaze_point_on_display_area"]
        rx, ry = g["right_gaze_point_on_display_area"]
        lo = g["left_gaze_origin_in_user_coordinate_system"]
        ro = g["right_gaze_origin_in_user_coordinate_system"]

        self._enqueue({
            "row_type": "SAMPLE",
            "sync_us": g["system_time_stamp"],
            "device_time_stamp": g["device_time_stamp"],
            "segment_index": seg, "backend": backend, "block_id": block,
            "left_gaze_x": lx, "left_gaze_y": ly,
            "left_gaze_validity": g["left_gaze_point_validity"],
            "right_gaze_x": rx, "right_gaze_y": ry,
            "right_gaze_validity": g["right_gaze_point_validity"],
            "left_pupil_diameter": g["left_pupil_diameter"],
            "left_pupil_validity": g["left_pupil_validity"],
            "right_pupil_diameter": g["right_pupil_diameter"],
            "right_pupil_validity": g["right_pupil_validity"],
            "left_origin_x": lo[0], "left_origin_y": lo[1], "left_origin_z": lo[2],
            "left_origin_validity": g["left_gaze_origin_validity"],
            "right_origin_x": ro[0], "right_origin_y": ro[1], "right_origin_z": ro[2],
            "right_origin_validity": g["right_gaze_origin_validity"],
        })

    def _enqueue(self, row: dict) -> None:
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self.dropped_samples += 1

    def _writer_loop(self) -> None:
        pending = 0
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                row = self._queue.get(timeout=0.2)
            except queue.Empty:
                if pending:
                    self._fh.flush()
                    pending = 0
                continue
            self._writer.writerow([row.get(c, "") for c in CSV_HEADER])
            self.written_samples += 1
            pending += 1
            if pending >= 600:      # ~1 s στα 600 Hz
                self._fh.flush()
                pending = 0
        self._fh.flush()

    # ── Μεταδεδομένα ─────────────────────────────────────────────────────
    def _write_metadata(self) -> None:
        try:
            display_area = self._tracker.get_display_area()
            da = {
                "width_mm": display_area.width,
                "height_mm": display_area.height,
                "top_left": list(display_area.top_left),
                "top_right": list(display_area.top_right),
                "bottom_left": list(display_area.bottom_left),
            }
        except Exception:
            da = None
            logger.exception("[SpectrumRef] Αποτυχία ανάγνωσης display area.")

        try:
            freq = self._tracker.get_gaze_output_frequency()
        except Exception:
            freq = None

        self._meta = {
            "participant_id": self.participant_id,
            "session_tag": self.session_tag,
            "model": self._tracker.model,
            "serial_number": self._tracker.serial_number,
            "firmware_version": getattr(self._tracker, "firmware_version", None),
            "gaze_output_frequency_hz": freq,
            "display_area_mm": da,
            "screen_resolution_px": [self.screen_w, self.screen_h],
            "clock_pair_open": clock_pair(),
            "csv": str(self.csv_path.name),
        }
        self.meta_path.write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _append_metadata_close(self) -> None:
        try:
            self._meta["clock_pair_close"] = clock_pair()
            self._meta["samples_written"] = self.written_samples
            self._meta["samples_dropped"] = self.dropped_samples
            self.meta_path.write_text(
                json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            logger.exception("[SpectrumRef] Αποτυχία ενημέρωσης metadata.")
