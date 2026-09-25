from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _find_col(df: pd.DataFrame, *needles: str):
    for c in df.columns:
        low = str(c).lower()
        if all(n.lower() in low for n in needles):
            return c
    return None


def load_prolab(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        print("  (xlsx — αργό· για τακτική χρήση εξήγαγε TSV από τον Pro Lab)")
        return pd.read_excel(path)
    for sep in ("\t", ";", ","):
        try:
            df = pd.read_csv(path, sep=sep, low_memory=False)
            if df.shape[1] > 3:
                return df
        except Exception:
            continue
    raise SystemExit(f"Δεν μπόρεσα να διαβάσω το {path}")


def prolab_marker_times(df: pd.DataFrame, key: str) -> tuple:
    c_event = _find_col(df, "event")
    c_value = _find_col(df, "event", "value")
    if c_event == c_value:
        c_event = _find_col(df, "event") if c_value is None else None
    c_rts = _find_col(df, "recording", "timestamp")
    c_date = _find_col(df, "recording date utc") or _find_col(df, "recording", "date")
    c_start = _find_col(df, "recording start time utc") or _find_col(df, "recording", "start time")

    missing = [n for n, c in [("Event", c_event), ("Event value", c_value),
                              ("Recording timestamp", c_rts),
                              ("Recording date", c_date),
                              ("Recording start time", c_start)] if c is None]
    if missing:
        raise SystemExit(
            "Λείπουν στήλες από το export του Pro Lab: " + ", ".join(missing) +
            "\nΣτην εξαγωγή επίλεξε να συμπεριληφθούν τα Events και οι "
            "χρονοσφραγίδες εγγραφής."
        )

    date_s = str(df[c_date].dropna().iloc[0]).strip()
    time_s = str(df[c_start].dropna().iloc[0]).strip()
    start = None
    for dfmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"):
        for tfmt in ("%H:%M:%S.%f", "%H:%M:%S"):
            try:
                start = datetime.strptime(f"{date_s} {time_s}", f"{dfmt} {tfmt}")
                start = start.replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
        if start:
            break
    if start is None:
        raise SystemExit(f"Δεν αναγνώρισα την έναρξη εγγραφής: '{date_s} {time_s}'")

    ev = df[df[c_event].astype(str).str.contains("Keyboard", case=False, na=False)]
    hits = ev[ev[c_value].astype(str).str.strip().str.upper() == key.upper()]
    rts = pd.to_numeric(hits[c_rts], errors="coerce").dropna()
    times = [start.timestamp() + v / 1e6 for v in rts]

    all_keys = (ev[c_value].astype(str).str.strip().value_counts().head(10).to_dict()
                if len(ev) else {})
    return times, {"recording_start_utc": start, "keyboard_events": len(ev),
                   "top_keys": all_keys}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_csv", type=Path)
    ap.add_argument("prolab_export", type=Path)
    ap.add_argument("--key", default="F9")
    ap.add_argument("--tolerance-ms", type=float, default=50.0,
                    help="Μέγιστη αποδεκτή διασπορά των διαφορών (default: 50 ms).")
    args = ap.parse_args()

    print(f"\n{'=' * 74}\nΕΛΕΓΧΟΣ ΣΥΓΧΡΟΝΙΣΜΟΥ\n{'=' * 74}")

    d = pd.read_csv(args.session_csv, low_memory=False)
    marks = d[(d.row_type == "EVENT") & (d.event_type == "SYNC_MARKER")].copy()
    ours = [int(v) / 1e9 for v in pd.to_numeric(marks.t_unix_ns, errors="coerce").dropna()]
    print(f"\nΔικό μας log : {len(ours)} SYNC_MARKER")
    for n, t in zip(marks.note.fillna(""), ours):
        print(f"   {datetime.fromtimestamp(t, tz=timezone.utc):%H:%M:%S.%f}  {n}")

    print(f"\nPro Lab      : ανάγνωση {args.prolab_export.name}")
    theirs, diag = prolab_marker_times(load_prolab(args.prolab_export), args.key)
    print(f"   έναρξη εγγραφής (UTC) : {diag['recording_start_utc']:%Y-%m-%d %H:%M:%S.%f}")
    print(f"   KeyboardEvent σύνολο  : {diag['keyboard_events']}")
    print(f"   πατήματα '{args.key}'  : {len(theirs)}")

    if not theirs:
        print(f"\n{'!' * 74}")
        print(f"ΑΠΟΤΥΧΙΑ: κανένα '{args.key}' στο Pro Lab.")
        print("Πλήκτρα που όντως κατέγραψε:", diag["top_keys"] or "(κανένα)")
        print("Το κανάλι συγχρονισμού ΔΕΝ λειτουργεί. Δες τις εναλλακτικές.")
        print("!" * 74)
        return 1

    if len(ours) != len(theirs):
        print(f"\n[ΠΡΟΣΟΧΗ] Άνισο πλήθος: {len(ours)} έναντι {len(theirs)}.")
        print("  Αν το Pro Lab καταγράφει και πάτημα και απελευθέρωση, θα δεις")
        print("  διπλάσιο αριθμό — τότε κράτα κάθε δεύτερο και ξανατρέξε.")

    n = min(len(ours), len(theirs))
    offs = [(ours[i] - theirs[i]) * 1000.0 for i in range(n)]

    print(f"\n{'-' * 74}\nΑΠΟΚΛΙΣΗ ΡΟΛΟΓΙΩΝ (δικό μας μείον Pro Lab)\n{'-' * 74}")
    print(f"{'#':>3}  {'δικό μας (UTC)':<18} {'Pro Lab (UTC)':<18} {'διαφορά ms':>12}")
    for i in range(n):
        print(f"{i+1:>3}  {datetime.fromtimestamp(ours[i], tz=timezone.utc):%H:%M:%S.%f}   "
              f"{datetime.fromtimestamp(theirs[i], tz=timezone.utc):%H:%M:%S.%f}   "
              f"{offs[i]:>12.1f}")

    s = pd.Series(offs)
    print(f"\nμέση απόκλιση : {s.mean():.1f} ms")
    print(f"διασπορά (SD) : {s.std():.1f} ms" if n > 1 else "διασπορά      : —")
    print(f"εύρος         : {s.min():.1f} έως {s.max():.1f} ms")

    if n >= 2:
        span = theirs[-1] - theirs[0]
        drift = offs[-1] - offs[0]
        print(f"\ndrift σε {span:.0f} s : {drift:+.1f} ms"
              f"  ({drift / span * 1000:+.0f} ppm)" if span > 0 else "")
        if abs(drift) > args.tolerance_ms:
            print("  -> Τα ρολόγια αποκλίνουν. Χρειάζεται ΓΡΑΜΜΙΚΗ διόρθωση")
            print("     στην ανάλυση, όχι σταθερή αφαίρεση της μέσης τιμής.")
        else:
            print("  -> Σταθερή απόκλιση· αρκεί αφαίρεση της μέσης τιμής.")

    ok = (len(ours) == len(theirs)) and (n < 2 or s.std() <= args.tolerance_ms)
    print(f"\n{'=' * 74}")
    if ok:
        print("ΕΠΙΤΥΧΙΑ: το κανάλι συγχρονισμού λειτουργεί.")
        print(f"Ακρίβεια ευθυγράμμισης ~{s.std() if n > 1 else 0:.0f} ms. Αφαίρεσε "
              f"{s.mean():.0f} ms από τους δικούς μας χρόνους για να πέσουν στο "
              "ρολόι του Pro Lab.")
    else:
        print("ΠΡΟΣΟΧΗ: το κανάλι λειτουργεί αλλά η ευθυγράμμιση δεν είναι αξιόπιστη.")
        print("Δες τη διασπορά παραπάνω πριν βασίσεις ανάλυση σε αυτήν.")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
