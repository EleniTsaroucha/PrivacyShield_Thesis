from __future__ import annotations

import argparse
import csv as _csv
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

FAIL, WARN, OK = "FAIL", "WARN", "OK "

REQUIRED_COLUMNS = {
    "row_type", "camera_frame_index", "t_unix_ns", "t_perf_s",
    "t_capture_perf_s", "block_id", "backend_active",
    "iv_angle", "iv_distance", "iv_lighting",
    "state", "unknown_count", "gaze_valid", "gaze_out_of_bounds",
    "event_type", "latency_detect_ms",
}

INDEX_FIELDS = [
    "checked_at", "session_id", "participant_id", "status", "n_fail", "n_warn",
    "duration_s", "frames", "effective_fps", "duplicates_suppressed",
    "backends", "blocks", "gaze_valid_pct", "shield_activations",
    "latency_median_ms", "solo_unknown_pct", "auth_fresh_pct", "failed_checks", "data_file",
]


class Report:
    def __init__(self, name: str, meta: dict | None = None):
        self.name = name
        self.meta = meta or {}
        self.rows: list[tuple[str, str, str]] = []
        self.stats: dict = {}

    def add(self, level: str, check: str, detail: str) -> None:
        self.rows.append((level, check, detail))

    @property
    def n_fail(self) -> int:
        return sum(1 for l, _, _ in self.rows if l == FAIL)

    @property
    def n_warn(self) -> int:
        return sum(1 for l, _, _ in self.rows if l == WARN)

    @property
    def failed(self) -> bool:
        return self.n_fail > 0

    @property
    def status(self) -> str:
        return "FAIL" if self.n_fail else ("WARN" if self.n_warn else "OK")

    def failed_checks(self) -> list:
        return [c for l, c, _ in self.rows if l == FAIL]

    def to_text(self) -> str:
        lines = ["", "=" * 78, f"ΕΠΙΚΥΡΩΣΗ: {self.name}", "=" * 78]
        lines += [f"[{l}] {c:<34} {d}" for l, c, d in self.rows]
        lines += ["-" * 78,
                  f"Κατάσταση: {self.status}  ({self.n_fail} FAIL, {self.n_warn} WARN)"]
        return "\n".join(lines)

    def print(self) -> None:
        print(self.to_text())


def _load_meta(csv_path: Path) -> dict:
    meta_path = csv_path.with_name(csv_path.stem + "_meta.json")
    if not meta_path.exists():
        return {}
    with open(meta_path, encoding="utf-8") as fh:
        return json.load(fh)


def validate(csv_path) -> Report:
    csv_path = Path(csv_path)
    meta = _load_meta(csv_path)
    rep = Report(csv_path.name, meta)
    df = pd.read_csv(csv_path, low_memory=False)
    frames = df[df.row_type == "FRAME"].copy()
    events = df[df.row_type == "EVENT"].copy()
    counters = meta.get("counters") or {}

    st = rep.stats
    st.update({
        "session_id": meta.get("session_id", csv_path.stem),
        "participant_id": meta.get("participant_id", ""),
        "duration_s": meta.get("duration_s", ""),
        "frames": len(frames),
        "effective_fps": counters.get("effective_fps", ""),
        "duplicates_suppressed": counters.get("duplicate_frames_suppressed", ""),
        "data_file": csv_path.name,
    })

    # 1. σχήμα 
    missing = REQUIRED_COLUMNS - set(df.columns)
    rep.add(FAIL if missing else OK, "σχήμα στηλών",
            f"λείπουν: {sorted(missing)}" if missing else "πλήρες")

    version = meta.get("log_schema_version")
    rep.add(OK if version else FAIL, "log_schema_version",
            version or "απουσιάζει από το metadata")

    if not meta.get("git_revision"):
        rep.add(WARN, "git_revision", "κενό: η έκδοση κώδικα δεν είναι ανιχνεύσιμη")

    if frames.empty:
        rep.add(FAIL, "καρέ", "καμία γραμμή FRAME στο αρχείο")
    else:
        # 2. διπλοεγγραφές
        dup = int(frames.camera_frame_index.duplicated().sum())
        rep.add(FAIL if dup else OK, "διπλά camera_frame_index",
                f"{dup} από {len(frames)}")

        supp = counters.get("duplicate_frames_suppressed", 0)
        if supp:
            rep.add(WARN, "διπλοεγγραφές κατεσταλμένες",
                    f"{supp} — το main.py καλεί log_frame πάνω από μία φορά ανά καρέ")

        # 3. ρυθμός καρέ 
        tcap = pd.to_numeric(frames.t_capture_perf_s, errors="coerce").dropna()
        span = (tcap.max() - tcap.min()) if len(tcap) > 1 else meta.get("duration_s")
        if span:
            fps = len(frames) / span
            st["effective_fps"] = round(fps, 2)
            target = (meta.get("config") or {}).get("target_fps")
            detail = f"{fps:.1f} fps πραγματικά"
            if target:
                ratio = fps / float(target)
                detail += f" έναντι {target} ονομαστικά ({ratio:.0%})"
                rep.add(OK if ratio >= 0.9 else WARN, "ρυθμός καρέ", detail)
            else:
                rep.add(WARN, "ρυθμός καρέ", detail + " (χωρίς target_fps στο metadata)")

            gaps = tcap.diff().dropna()
            if len(gaps):
                long_gaps = int((gaps > 1.0).sum())
                if long_gaps:
                    rep.add(WARN, "κενά καταγραφής",
                            f"{long_gaps} διαστήματα >1 s (μέγιστο {gaps.max():.1f} s)")

        # 4. εγκυρότητα βλέμματος
        if "gaze_valid" in frames:
            tracked_all = frames[frames.state.isin(["CLEAR", "PANIC"])]
            valid_rate = float(
                tracked_all.gaze_valid.mean() if len(tracked_all)
                else frames.gaze_valid.mean()
            )
            st["gaze_valid_pct"] = round(valid_rate * 100, 1)
            rep.add(OK if valid_rate > 0.5 else FAIL, "ποσοστό έγκυρου βλέμματος",
                    f"{valid_rate:.1%}")

            tracked = frames[frames.state.isin(["CLEAR", "PANIC"])]
            by_state = tracked.groupby("state").gaze_valid.mean() if len(tracked) else None
            if by_state is not None and len(by_state) > 1:
                spread = by_state.max() - by_state.min()
                detail = ", ".join(f"{k}={v:.0%}" for k, v in by_state.items())
                rep.add(FAIL if spread > 0.5 else OK, "βλέμμα CLEAR έναντι PANIC", detail)

        if "gaze_out_of_bounds" in frames:
            oob = pd.to_numeric(frames.gaze_out_of_bounds, errors="coerce").fillna(0)
            rep.add(OK if oob.mean() < 0.05 else WARN, "βλέμμα εκτός οθόνης",
                    f"{oob.mean():.1%}")

        # 5. ταξινόμηση 
        for col in ("block_id", "backend_active", "iv_angle", "iv_distance", "iv_lighting"):
            if col in frames:
                filled = float((frames[col].notna() & (frames[col].astype(str) != "")).mean())
                rep.add(OK if filled > 0.95 else FAIL, f"συμπλήρωση {col}",
                        f"{filled:.1%} των καρέ")

        backends = sorted(set(frames.backend_active.dropna().astype(str)) - {""})
        blocks = sorted(set(frames.block_id.dropna().astype(str)) - {""})
        st["backends"] = "|".join(backends)
        st["blocks"] = "|".join(blocks)
        rep.add(OK, "backends στη συνεδρία", ", ".join(backends) or "κανένα")

        # 6. μονοτονία χρόνου 
        rep.add(OK if tcap.is_monotonic_increasing else FAIL, "μονοτονία t_capture",
                "αύξουσα" if tcap.is_monotonic_increasing else "παραβιάζεται")

    # 7. συμβάντα 
    if events.empty:
        rep.add(FAIL, "συμβάντα", "καμία γραμμή EVENT")
    else:
        types = events.event_type.value_counts().to_dict()
        for required in ("SESSION_START", "SESSION_END"):
            rep.add(OK if types.get(required) else FAIL, f"συμβάν {required}",
                    "παρόν" if types.get(required) else "απουσιάζει")

        n_sync = types.get("SYNC_MARKER", 0)
        rep.add(OK if n_sync >= 2 else FAIL, "δείκτες συγχρονισμού",
                f"{n_sync} (απαιτούνται ≥2 για διόρθωση drift)")

        if not types.get("CALIBRATION"):
            rep.add(WARN, "συμβάν CALIBRATION",
                    "δεν καταγράφηκαν μετρικές βαθμονόμησης")

        act = types.get("PRIVACY_SHIELD_ACTIVATED", 0)
        deact = types.get("PRIVACY_SHIELD_DEACTIVATED", 0)
        st["shield_activations"] = act
        rep.add(OK if abs(act - deact) <= 1 else WARN, "ζεύγη ασπίδας",
                f"{act} ενεργοποιήσεις / {deact} απενεργοποιήσεις")

        # 8. λογικός έλεγχος latency 
        lat = pd.to_numeric(events.latency_detect_ms, errors="coerce").dropna()
        if len(lat):
            st["latency_median_ms"] = round(float(lat.median()), 2)
            detail = (f"n={len(lat)}, διάμεσος={lat.median():.1f} ms, "
                      f"εύρος={lat.min():.1f}–{lat.max():.1f} ms")
            # Τιμή <5 ms σημαίνει ότι η αφετηρία δεν είναι το t_capture.
            rep.add(FAIL if lat.median() < 5 else OK, "latency ανίχνευσης", detail)
        elif act:
            rep.add(FAIL, "latency ανίχνευσης", "ενεργοποιήσεις χωρίς μετρημένη τιμή")

        # 9. επικύρωση PANIC hold 
        ov = pd.to_numeric(events.hold_overshoot_ms, errors="coerce").dropna()
        if len(ov):
            rep.add(OK if ov.abs().max() < 300 else WARN, "ακρίβεια PANIC hold",
                    f"υπέρβαση {ov.min():.0f} έως {ov.max():.0f} ms έναντι ονομαστικού")

        # 9α. μερίδιο πραγματικών αποφάσεων ταυτοποίησης 
        if "auth_fresh" in frames.columns:
            fresh = pd.to_numeric(frames.auth_fresh, errors="coerce").fillna(0)
            rate = float(fresh.mean())
            st["auth_fresh_pct"] = round(rate * 100, 1)
            rep.add(OK, "καρέ με πραγματική ταυτοποίηση",
                    f"{rate:.0%} — οι ρυθμοί ανίχνευσης υπολογίζονται ΜΟΝΟ σε αυτά")

        # 9β. ψευδώς θετικά: ένα πρόσωπο, χαρακτηρισμένο άγνωστο 
        if {"total_faces", "unknown_count", "trusted_count"} <= set(frames.columns):
            solo = ((frames.total_faces == 1) & (frames.unknown_count == 1)
                    & (frames.trusted_count == 0))
            rate = float(solo.mean())
            st["solo_unknown_pct"] = round(rate * 100, 1)
            rep.add(FAIL if rate > 0.10 else (WARN if rate > 0.02 else OK),
                    "μοναδικό πρόσωπο ως άγνωστο",
                    f"{rate:.1%} των καρέ — πιθανή αποτυχία αναγνώρισης του χρήστη")

    # 10. ιδιωτικότητα 
    if "identities_json" in df.columns:
        sample = df.identities_json.dropna().astype(str).head(5000)
        leaked = bool(sample.str.contains("best_name", na=False).any())
        rep.add(FAIL if leaked else OK, "ψευδωνυμοποίηση ταυτοτήτων",
                "εντοπίστηκε πεδίο ονόματος" if leaked else "καθαρό")

    return rep


def append_to_index(report: Report, csv_path, index_path) -> None:
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: "" for k in INDEX_FIELDS}
    row.update(report.stats)
    row.update({
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": report.status,
        "n_fail": report.n_fail,
        "n_warn": report.n_warn,
        "failed_checks": "; ".join(report.failed_checks()),
        "data_file": Path(csv_path).name,
    })
    write_header = not index_path.exists()
    with open(index_path, "a", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=INDEX_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--strict", action="store_true",
                    help="Αντιμετώπισε τα WARN ως αποτυχία.")
    ap.add_argument("--index", type=Path, default=None,
                    help="Πρόσθεσε τα αποτελέσματα σε ευρετήριο συνεδριών.")
    args = ap.parse_args()

    any_failed = False
    for p in args.paths:
        rep = validate(p)
        rep.print()
        if args.index:
            append_to_index(rep, p, args.index)
        if rep.failed or (args.strict and rep.n_warn):
            any_failed = True

    print("\nΣυνολικό αποτέλεσμα:", "ΑΠΟΤΥΧΙΑ" if any_failed else "ΕΝΤΑΞΕΙ")
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
