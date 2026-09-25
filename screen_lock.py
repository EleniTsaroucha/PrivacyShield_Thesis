import logging
import platform
import shutil
import subprocess

logger = logging.getLogger(__name__)


class ScreenLocker:
    """Επιχειρεί κλείδωμα του σταθμού εργασίας, ανά λειτουργικό σύστημα."""

    def __init__(self):
        self._system = platform.system()

    def lock(self) -> bool:
        """Επιστρέφει True αν το κλείδωμα εκτελέστηκε επιτυχώς."""
        try:
            if self._system == "Windows":
                return self._lock_windows()
            elif self._system == "Linux":
                return self._lock_linux()
            elif self._system == "Darwin":
                return self._lock_macos()
            logger.warning("Άγνωστο λειτουργικό σύστημα '%s' — δεν έγινε lock.", self._system)
            return False
        except Exception:
            logger.exception("Αποτυχία κατά το κλείδωμα της οθόνης.")
            return False

    # Windows
    def _lock_windows(self) -> bool:
        import ctypes
        ok = bool(ctypes.windll.user32.LockWorkStation())
        logger.info("Windows: LockWorkStation() -> %s", ok)
        return ok

    # Linux (δοκιμάζει διαδοχικά τα πιο συνηθισμένα εργαλεία ανά DE)
    def _lock_linux(self) -> bool:
        candidates = [
            ["loginctl", "lock-session"],
            ["gnome-screensaver-command", "--lock"],
            ["dm-tool", "lock"],
            ["xdg-screensaver", "lock"],
            ["i3lock"],
            ["betterlockscreen", "--lock"],
        ]
        for cmd in candidates:
            if shutil.which(cmd[0]) is None:
                continue
            result = subprocess.run(cmd, capture_output=True, timeout=5)
            if result.returncode == 0:
                logger.info("Linux: κλείδωμα μέσω '%s'.", " ".join(cmd))
                return True
        logger.warning(
            "Linux: δεν βρέθηκε διαθέσιμο εργαλείο κλειδώματος "
            "(δοκιμάστηκαν: loginctl, gnome-screensaver-command, dm-tool, "
            "xdg-screensaver, i3lock, betterlockscreen)."
        )
        return False

    # macOS
    def _lock_macos(self) -> bool:
        cmd = [
            "/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession",
            "-suspend",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=5)
        ok = result.returncode == 0
        logger.info("macOS: CGSession -suspend -> %s", ok)
        return ok
