from __future__ import annotations

import atexit
import csv
import hashlib
import json
import logging
import math
import platform
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

LOG_SCHEMA_VERSION = "2.2"

CSV_FIELDS: Sequence[str] = [
    # ταυτοποίηση γραμμής 
    "row_type",                    # FRAME | EVENT
    "log_row_index",
    "camera_frame_index",          # μοναδικό ανά καρέ κάμερας (όχι ανά εγγραφή)

    #  χρόνος 
    "t_unix_ns",                   # κοινή βάση με Tobii "Computer timestamp"
    "t_perf_s",                    # μονότονο ρολόι, αφετηρία = έναρξη logger
    "t_capture_perf_s",
    "t_inference_done_perf_s",
    "t_state_change_perf_s",
    "t_overlay_painted_perf_s",
    "inference_ms",                # t_inference_done - t_capture
    "capture_to_log_ms",           # καθυστέρηση αγωγού μέχρι την καταγραφή
    "loop_dt_ms",

    # ταξινόμηση (μεταβάλλονται εντός συνεδρίας) 
    "block_id",
    "trial_id",
    "backend_active",
    "iv_angle",
    "iv_distance",
    "iv_lighting",
    "profile_id",

    #  κατάσταση συστήματος (FRAME)
    "state",
    "prev_state",
    "state_changed",
    "auth_fresh",                  # 1 = η ταυτοποίηση έτρεξε σε αυτό το καρέ
    "total_faces",
    "trusted_count",
    "unknown_count",
    "max_unknown_similarity",
    "min_face_similarity",
    "identities_json",             # ψευδωνυμοποιημένο

    # βλέμμα (FRAME) 
    "gaze_x_raw",
    "gaze_y_raw",
    "gaze_x",                      # μετά από clamping εντός οθόνης
    "gaze_y",
    "gaze_confidence",
    "gaze_valid",
    "gaze_out_of_bounds",

    #  μετρητές (FRAME) 
    "no_face_counter",
    "panic_hold_remaining_ms",
    "hard_locked",

    # συμβάντα (EVENT) 
    "event_index",
    "event_type",
    "from_state",
    "to_state",
    "latency_detect_ms",           # t_capture(πρώτης ανίχνευσης) -> αλλαγή κατάστασης
    "latency_paint_ms",            # t_capture(πρώτης ανίχνευσης) -> σχεδίαση overlay
    "trigger_camera_frame",
    "hold_overshoot_ms",           # πραγματικό hold μείον ονομαστικό
    "note",
]

# Τύποι συμβάντων που αναγνωρίζει το validator.
EVENT_TYPES = (
    "SESSION_START", "SESSION_END",
    "BLOCK_START", "BLOCK_END",
    "BACKEND_SWITCH", "CALIBRATION",
    "SYNC_MARKER",
    "STATE_TRANSITION",
    "PRIVACY_SHIELD_ACTIVATED", "PRIVACY_SHIELD_DEACTIVATED", "OVERLAY_PAINTED",
    "HARD_LOCK_ENTER", "HARD_LOCK_EXIT",
    "USER_LEFT_SCREEN", "USER_RETURNED",
    "NOTE",
)


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _clock_anchor() -> dict:
    p0 = time.perf_counter()
    u = time.time_ns()
    p1 = time.perf_counter()
    return {
        "perf_s": (p0 + p1) / 2.0,
        "unix_ns": u,
        "uncertainty_ns": int((p1 - p0) * 1e9),
        "iso_utc": datetime.now(timezone.utc).isoformat(),
    }


class ExperimentLogger:

    def __init__(
        self,
        participant_id: str,
        log_dir: Path = Path("experiment_logs"),
        config: Optional[dict] = None,
        session_tag: str = "",
        anonymise_identities: bool = True,
        flush_every_n_frames: int = 20,
        app_version: str = "",
        auto_validate: bool = True,
        auto_parquet: bool = False,
        index_path: Optional[Path] = None,
        sync_hook=None,
    ) -> None:
        self.participant_id = participant_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.anonymise_identities = anonymise_identities
        self.flush_every_n_frames = max(1, int(flush_every_n_frames))
        self.auto_validate = auto_validate
        self.auto_parquet = auto_parquet
        self.index_path = Path(index_path) if index_path else self.log_dir / "sessions_index.csv"
        self.last_report = None
        self.sync_hook = sync_hook

        config = dict(config or {})
        self.screen_w = int(config.get("screen_w", 0) or 0)
        self.screen_h = int(config.get("screen_h", 0) or 0)
        self.panic_hold_s = float(config.get("panic_hold_s", 0.0) or 0.0)

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        tag = f"_{session_tag}" if session_tag else ""
        self.session_id = f"{participant_id}{tag}_{ts}"

        self._csv_path = self.log_dir / f"session_{self.session_id}.csv"
        self._meta_path = self.log_dir / f"session_{self.session_id}_meta.json"

        self._lock = threading.RLock()
        self._csv_fh = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_fh, fieldnames=list(CSV_FIELDS))
        self._writer.writeheader()

        self._t_start_perf = time.perf_counter()
        self._row_index = 0
        self._event_index = 0
        self._frames_written = 0
        self._frames_since_flush = 0

        # Καταστολή διπλοεγγραφών: κρατάμε τον τελευταίο δείκτη καρέ κάμερας.
        self._last_camera_frame_index: Optional[int] = None
        self._duplicates_suppressed = 0
        self._out_of_order_frames = 0

        # Κατάσταση ταξινόμησης, ισχύει μέχρι να αλλάξει ρητά.
        self._ctx = {
            "block_id": "",
            "trial_id": "",
            "backend_active": config.get("gaze_backend", "") or "",
            "iv_angle": "",
            "iv_distance": "",
            "iv_lighting": "",
            "profile_id": "",
        }

        # Παρακολούθηση latency: t_capture του πρώτου καρέ με άγνωστο πρόσωπο.
        self._trigger_capture_perf: Optional[float] = None
        self._trigger_camera_frame: Optional[int] = None
        self._last_unknown_capture_perf: Optional[float] = None

        # Χρονικά σταθμισμένη κατανομή καταστάσεων.
        self._state_time_s: dict = defaultdict(float)
        self._prev_capture_perf: Optional[float] = None
        self._prev_state_for_timing: Optional[str] = None

        # Ψευδωνυμοποίηση: σταθερό mapping όνομα -> κωδικός εντός συνεδρίας.
        self._identity_codes: dict = {}
        self._identity_counter = 0

        self._meta = {
            "log_schema_version": LOG_SCHEMA_VERSION,
            "app_version": app_version,
            "git_revision": _git_revision(),
            "session_id": self.session_id,
            "participant_id": participant_id,
            "session_tag": session_tag,
            "start_time_local_iso": datetime.now().astimezone().isoformat(),
            "start_time_utc_iso": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "data_file": self._csv_path.name,
            "anonymise_identities": anonymise_identities,
            "config": config,
            "clock_anchor_start": _clock_anchor(),
        }
        self._write_meta()

        # Η επικύρωση πρέπει να τρέξει ακόμη και σε Ctrl+C ή ανεξέλεγκτη
        # εξαίρεση. Το close() είναι idempotent, οπότε η διπλή κλήση είναι ασφαλής.
        atexit.register(self.close)

        self.log_event("SESSION_START", note=f"schema={LOG_SCHEMA_VERSION}")
        logger.info("ExperimentLogger v%s: συνεδρία '%s' -> %s",
                    LOG_SCHEMA_VERSION, self.session_id, self._csv_path)

    # context manager

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            try:
                self.log_event("NOTE", note=f"abnormal_exit={exc_type.__name__}: {exc}")
            except Exception:
                pass
        self.close()
        return False

    # utils

    def perf_to_unix_ns(self, t_perf: float) -> int:
        a = self._meta["clock_anchor_start"]
        return int(a["unix_ns"] + (t_perf - a["perf_s"]) * 1e9)

    def _write_meta(self) -> None:
        with open(self._meta_path, "w", encoding="utf-8") as fh:
            json.dump(self._meta, fh, ensure_ascii=False, indent=2)

    def _blank_row(self) -> dict:
        row = {f: "" for f in CSV_FIELDS}
        row.update(self._ctx)
        return row

    def _identity_code(self, name: str) -> str:
        if not self.anonymise_identities:
            return name
        if name not in self._identity_codes:
            self._identity_counter += 1
            digest = hashlib.sha256(
                f"{self.session_id}|{name}".encode("utf-8")
            ).hexdigest()[:8]
            self._identity_codes[name] = f"ID{self._identity_counter:02d}-{digest}"
        return self._identity_codes[name]

    def _summarise_identities(self, auth_result) -> str:
        out = []
        for m in getattr(auth_result, "face_matches", []) or []:
            name = m.get("best_name") or ""
            sim = m.get("best_sim")
            out.append({
                "nearest_id": self._identity_code(name) if name else "",
                "sim": round(float(sim), 4) if isinstance(sim, (int, float)) else None,
            })
        return json.dumps(out, ensure_ascii=False)

    # πλαίσιο συνεδρίας

    def start_block(
        self,
        block_id: str,
        backend: str,
        iv_angle=None,
        iv_distance=None,
        iv_lighting=None,
        profile_id=None,
        note: str = "",
    ) -> None:
        with self._lock:
            self._ctx.update({
                "block_id": block_id,
                "backend_active": backend,
                "iv_angle": "" if iv_angle is None else iv_angle,
                "iv_distance": "" if iv_distance is None else iv_distance,
                "iv_lighting": "" if iv_lighting is None else iv_lighting,
                "profile_id": "" if profile_id is None else profile_id,
                "trial_id": "",
            })
            self.log_event("BLOCK_START", note=note)

    def end_block(self, note: str = "") -> None:
        with self._lock:
            self.log_event("BLOCK_END", note=note)
            self._ctx.update({"block_id": "", "trial_id": ""})

    def set_trial(self, trial_id) -> None:
        with self._lock:
            self._ctx["trial_id"] = "" if trial_id is None else trial_id

    def switch_backend(self, backend: str, note: str = "") -> None:
        with self._lock:
            previous = self._ctx["backend_active"]
            self._ctx["backend_active"] = backend
            self.log_event(
                "BACKEND_SWITCH",
                note=(f"from={previous} to={backend} " + note).strip(),
            )

    def log_calibration(self, backend: str, metrics: Optional[dict] = None) -> None:
        self.log_event(
            "CALIBRATION",
            note=json.dumps({"backend": backend, **(metrics or {})}, ensure_ascii=False),
        )

    def sync_marker(self, label: str, external_channel: str = "") -> None:
        emitted = ""
        if self.sync_hook is not None:
            t0 = time.perf_counter()
            try:
                self.sync_hook()
                t1 = time.perf_counter()
                emitted = (f" emitted=1 emit_perf_s={t0 - self._t_start_perf:.6f}"
                           f" emit_uncertainty_ms={(t1 - t0) * 1000:.2f}")
            except Exception:
                logger.exception(
                    "Η εκπομπή δείκτη συγχρονισμού απέτυχε· ο δείκτης "
                    "καταγράφεται μόνο τοπικά."
                )
                emitted = " emitted=0"
        self.log_event(
            "SYNC_MARKER",
            note=f"label={label} channel={external_channel}{emitted}".strip(),
        )

    # καταγραφή

    def log_frame(
        self,
        camera_frame_index: int,
        t_capture_perf: float,
        state: str,
        prev_state: str,
        auth_result,
        gaze,
        no_face_counter: int,
        panic_hold_remaining_ms: float,
        hard_locked: bool,
        t_inference_done_perf: Optional[float] = None,
        t_state_change_perf: Optional[float] = None,
        t_overlay_painted_perf: Optional[float] = None,
        loop_dt_ms: Optional[float] = None,
    ) -> bool:
        with self._lock:
            if self._last_camera_frame_index is not None:
                if camera_frame_index == self._last_camera_frame_index:
                    self._duplicates_suppressed += 1
                    return False
                if camera_frame_index < self._last_camera_frame_index:
                    self._out_of_order_frames += 1
            self._last_camera_frame_index = camera_frame_index

            now_perf = time.perf_counter()
            t_capture_rel = t_capture_perf - self._t_start_perf

            # Χρονικά σταθμισμένη κατανομή καταστάσεων.
            if self._prev_capture_perf is not None and self._prev_state_for_timing:
                dt = t_capture_perf - self._prev_capture_perf
                if 0 <= dt < 5.0:
                    self._state_time_s[self._prev_state_for_timing] += dt
            self._prev_capture_perf = t_capture_perf
            self._prev_state_for_timing = state

            # βλέμμα 
            gx_raw = getattr(gaze, "x", None)
            gy_raw = getattr(gaze, "y", None)
            conf = getattr(gaze, "confidence", None)
            gaze_valid = getattr(gaze, "valid", None)
            if gaze_valid is None:
                gaze_valid = (
                    gx_raw is not None and gy_raw is not None
                    and isinstance(gx_raw, (int, float))
                    and isinstance(gy_raw, (int, float))
                    and math.isfinite(gx_raw) and math.isfinite(gy_raw)
                    and (conf is None or conf > 0.0)
                )

            oob = ""
            gx = gy = ""
            if gaze_valid and self.screen_w and self.screen_h:
                oob = int(not (0 <= gx_raw < self.screen_w and 0 <= gy_raw < self.screen_h))
                gx = min(max(gx_raw, 0), self.screen_w - 1)
                gy = min(max(gy_raw, 0), self.screen_h - 1)
            elif gaze_valid:
                gx, gy = gx_raw, gy_raw

            #  ταυτότητες 
            max_unknown = auth_result.max_unknown_similarity()

            face_sims = [
                m.get("best_sim") for m in (getattr(auth_result, "face_matches", []) or [])
                if isinstance(m.get("best_sim"), (int, float))
            ]

            row = self._blank_row()
            self._row_index += 1
            row.update({
                "row_type": "FRAME",
                "log_row_index": self._row_index,
                "camera_frame_index": camera_frame_index,
                "t_unix_ns": self.perf_to_unix_ns(t_capture_perf),
                "t_perf_s": f"{t_capture_rel:.6f}",
                "t_capture_perf_s": f"{t_capture_rel:.6f}",
                "t_inference_done_perf_s": (
                    f"{t_inference_done_perf - self._t_start_perf:.6f}"
                    if t_inference_done_perf is not None else ""
                ),
                "t_state_change_perf_s": (
                    f"{t_state_change_perf - self._t_start_perf:.6f}"
                    if t_state_change_perf is not None else ""
                ),
                "t_overlay_painted_perf_s": (
                    f"{t_overlay_painted_perf - self._t_start_perf:.6f}"
                    if t_overlay_painted_perf is not None else ""
                ),
                "inference_ms": (
                    f"{(t_inference_done_perf - t_capture_perf) * 1000:.2f}"
                    if t_inference_done_perf is not None else ""
                ),
                "capture_to_log_ms": f"{(now_perf - t_capture_perf) * 1000:.2f}",
                "loop_dt_ms": "" if loop_dt_ms is None else f"{loop_dt_ms:.2f}",
                "state": state,
                "prev_state": prev_state,
                "state_changed": int(state != prev_state),
                "auth_fresh": int(t_inference_done_perf is not None),
                "total_faces": auth_result.total_faces,
                "trusted_count": len(auth_result.trusted_faces),
                "unknown_count": auth_result.unknown_count,
                "max_unknown_similarity": (
                    f"{max_unknown:.4f}" if max_unknown is not None else ""
                ),
                "min_face_similarity": (
                    f"{min(face_sims):.4f}" if face_sims else ""
                ),
                "identities_json": self._summarise_identities(auth_result),
                "gaze_x_raw": "" if gx_raw is None else gx_raw,
                "gaze_y_raw": "" if gy_raw is None else gy_raw,
                "gaze_x": gx,
                "gaze_y": gy,
                "gaze_confidence": "" if conf is None else f"{float(conf):.3f}",
                "gaze_valid": int(bool(gaze_valid)),
                "gaze_out_of_bounds": oob,
                "no_face_counter": no_face_counter,
                "panic_hold_remaining_ms": f"{panic_hold_remaining_ms:.1f}",
                "hard_locked": int(hard_locked),
            })
            self._writer.writerow(row)
            self._frames_written += 1

            # παρακολούθηση latency 
            if auth_result.unknown_count > 0:
                self._last_unknown_capture_perf = t_capture_perf
                if prev_state != "PANIC" and self._trigger_capture_perf is None:
                    self._trigger_capture_perf = t_capture_perf
                    self._trigger_camera_frame = camera_frame_index
            elif state != "PANIC":
                self._trigger_capture_perf = None
                self._trigger_camera_frame = None

            self._frames_since_flush += 1
            if self._frames_since_flush >= self.flush_every_n_frames:
                self._csv_fh.flush()
                self._frames_since_flush = 0
            return True

    def log_event(
        self,
        event_type: str,
        from_state: str = "",
        to_state: str = "",
        camera_frame_index: Optional[int] = None,
        t_state_change_perf: Optional[float] = None,
        t_overlay_painted_perf: Optional[float] = None,
        t_capture_perf: Optional[float] = None,
        note: str = "",
    ) -> None:
        with self._lock:
            self._event_index += 1
            self._row_index += 1
            now_perf = time.perf_counter()

            latency_detect = latency_paint = ""
            trigger_frame = ""
            overshoot = ""

            if event_type == "PRIVACY_SHIELD_ACTIVATED" and self._trigger_capture_perf is not None:
                t_change = t_state_change_perf if t_state_change_perf is not None else now_perf
                latency_detect = f"{(t_change - self._trigger_capture_perf) * 1000:.2f}"
                if t_overlay_painted_perf is not None:
                    latency_paint = (
                        f"{(t_overlay_painted_perf - self._trigger_capture_perf) * 1000:.2f}"
                    )
                trigger_frame = self._trigger_camera_frame
                self._trigger_capture_perf = None
                self._trigger_camera_frame = None

            if event_type == "PRIVACY_SHIELD_DEACTIVATED" and self._last_unknown_capture_perf:
                t_change = t_state_change_perf if t_state_change_perf is not None else now_perf
                observed_hold = t_change - self._last_unknown_capture_perf
                overshoot = f"{(observed_hold - self.panic_hold_s) * 1000:.2f}"
                if t_overlay_painted_perf is not None:
                    latency_paint = (
                        f"{(t_overlay_painted_perf - t_change) * 1000:.2f}"
                    )

        
            if (not latency_paint and t_capture_perf is not None
                    and t_overlay_painted_perf is not None):
                latency_paint = (
                    f"{(t_overlay_painted_perf - t_capture_perf) * 1000:.2f}"
                )
            if not latency_detect and t_capture_perf is not None and t_state_change_perf is not None:
                latency_detect = (
                    f"{(t_state_change_perf - t_capture_perf) * 1000:.2f}"
                )

            row = self._blank_row()
            row.update({
                "row_type": "EVENT",
                "log_row_index": self._row_index,
                "event_index": self._event_index,
                "camera_frame_index": (
                    "" if camera_frame_index is None else camera_frame_index
                ),
                "t_unix_ns": self.perf_to_unix_ns(now_perf),
                "t_perf_s": f"{now_perf - self._t_start_perf:.6f}",
                "event_type": event_type,
                "from_state": from_state,
                "to_state": to_state,
                "latency_detect_ms": latency_detect,
                "latency_paint_ms": latency_paint,
                "trigger_camera_frame": trigger_frame,
                "t_capture_perf_s": (
                    f"{t_capture_perf - self._t_start_perf:.6f}"
                    if t_capture_perf is not None else ""
                ),
                "t_state_change_perf_s": (
                    f"{t_state_change_perf - self._t_start_perf:.6f}"
                    if t_state_change_perf is not None else ""
                ),
                "t_overlay_painted_perf_s": (
                    f"{t_overlay_painted_perf - self._t_start_perf:.6f}"
                    if t_overlay_painted_perf is not None else ""
                ),
                "hold_overshoot_ms": overshoot,
                "note": note,
            })
            self._writer.writerow(row)
            self._csv_fh.flush()

    #  τερματισμός

    def close(self, summary: Optional[dict] = None) -> None:
        with self._lock:
            if self._csv_fh.closed:
                return
            self.log_event("SESSION_END")

            end_perf = time.perf_counter()
            duration = end_perf - self._t_start_perf
            total_state_time = sum(self._state_time_s.values()) or 1.0

            self._meta.update({
                "end_time_local_iso": datetime.now().astimezone().isoformat(),
                "end_time_utc_iso": datetime.now(timezone.utc).isoformat(),
                "duration_s": round(duration, 3),
                "clock_anchor_end": _clock_anchor(),
                "counters": {
                    "frames_written": self._frames_written,
                    "duplicate_frames_suppressed": self._duplicates_suppressed,
                    "out_of_order_frames": self._out_of_order_frames,
                    "events": self._event_index,
                    "capture_span_s": round(total_state_time, 3),
                    "effective_fps": round(self._frames_written / total_state_time, 2)
                    if total_state_time > 0 else None,
                },
                "state_time_s": {k: round(v, 3) for k, v in self._state_time_s.items()},
                "state_time_fraction": {
                    k: round(v / total_state_time, 4)
                    for k, v in self._state_time_s.items()
                },
            })
            if summary:
                self._meta["summary"] = summary
            self._write_meta()
            self._csv_fh.close()

            if self.anonymise_identities and self._identity_codes:
                self._write_identity_key()

            logger.info(
                "Συνεδρία '%s' έκλεισε: %d καρέ, %.2f fps, %d διπλοεγγραφές κατεστάλησαν.",
                self.session_id, self._frames_written,
                self._meta["counters"]["effective_fps"] or 0.0,
                self._duplicates_suppressed,
            )

        # Εκτός lock: η επικύρωση διαβάζει το αρχείο που μόλις έκλεισε.
        self._run_post_session_checks()

    # έλεγχοι μετά τη λήξη

    def _run_post_session_checks(self) -> None:
        if not self.auto_validate:
            return
        try:
            from validate_session import validate, append_to_index
        except Exception as exc:
            logger.warning("Αυτόματη επικύρωση παραλείφθηκε (%s).", exc)
            return

        try:
            report = validate(self._csv_path)
            self.last_report = report

            report_path = self.log_dir / f"session_{self.session_id}_validation.txt"
            report_path.write_text(report.to_text(), encoding="utf-8")
            append_to_index(report, self._csv_path, self.index_path)

            print(report.to_text(), flush=True)
            if report.status == "FAIL":
                print(
                    "\n" + "!" * 78 +
                    f"\nΗ ΣΥΝΕΔΡΙΑ {self.session_id} ΑΠΕΤΥΧΕ ΣΤΟΝ ΕΛΕΓΧΟ ΠΟΙΟΤΗΤΑΣ."
                    "\nΜΗΝ την προσθέσεις στο σύνολο δεδομένων πριν διερευνηθεί."
                    f"\nΑναφορά: {report_path}\n" + "!" * 78 + "\n",
                    flush=True,
                )
            logger.info("Αναφορά επικύρωσης: %s (%s)", report_path, report.status)
        except Exception:
            logger.exception("Η αυτόματη επικύρωση απέτυχε· τα δεδομένα παραμένουν ακέραια.")
            return

        if self.auto_parquet:
            try:
                import pandas as pd
                out = self._csv_path.with_suffix(".parquet")
                pd.read_csv(self._csv_path, low_memory=False).to_parquet(out, index=False)
                logger.info("Parquet: %s", out)
            except Exception as exc:
                logger.warning("Μετατροπή σε Parquet παραλείφθηκε (%s).", exc)

    def _write_identity_key(self) -> None:
        key_dir = self.log_dir / "_identity_keys"
        key_dir.mkdir(parents=True, exist_ok=True)
        path = key_dir / f"{self.session_id}_identity_key.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._identity_codes, fh, ensure_ascii=False, indent=2)
        logger.warning(
            "Το αρχείο αντιστοίχισης ταυτοτήτων γράφτηκε στο %s. "
            "Μετακίνησέ το εκτός του φακέλου δεδομένων πριν από οποιαδήποτε κοινοποίηση.",
            path,
        )


