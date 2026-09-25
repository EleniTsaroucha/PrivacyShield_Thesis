import logging
from pathlib import Path
from typing import Optional

import cv2

logger = logging.getLogger(__name__)


class VideoRecorder:

    def __init__(
        self,
        output_path: Path,
        fps: int = 20,
        fourcc: str = "mp4v",
    ):
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fps = fps
        self._fourcc = fourcc
        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_count = 0
        self._closed = False

        import time
        self._t_start = time.perf_counter()

    def _ensure_writer(self, frame) -> None:
        if self._writer is not None:
            return
        h, w = frame.shape[:2]
        fourcc_code = cv2.VideoWriter_fourcc(*self._fourcc)
        self._writer = cv2.VideoWriter(
            str(self._path), fourcc_code, float(self._fps), (w, h)
        )
        if not self._writer.isOpened():
            logger.error(
                "Δεν άνοιξε το VideoWriter για %s — η εγγραφή βίντεο "
                "απενεργοποιείται γι' αυτή τη συνεδρία.", self._path,
            )
            self._writer = None
        else:
            logger.info(
                "VideoRecorder: εγγραφή σε %s (%dx%d @ %d fps).",
                self._path, w, h, self._fps,
            )

    def write_frame(self, frame_bgr, overlay_text: str = "") -> None:
        if self._closed:
            return
        self._ensure_writer(frame_bgr)
        if self._writer is None:
            return

        out_frame = frame_bgr
        if overlay_text:
            out_frame = frame_bgr.copy()
            cv2.putText(
                out_frame, overlay_text, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
            )

        self._writer.write(out_frame)
        self._frame_count += 1

    def close(self) -> Optional[dict]:
        if self._closed:
            return None
        self._closed = True

        if self._writer is None:
            return None

        self._writer.release()

        import time
        duration_s = round(time.perf_counter() - self._t_start, 3)
        summary = {
            "path": str(self._path.resolve()),
            "frame_count": self._frame_count,
            "duration_s": duration_s,
            "fps": self._fps,
        }
        logger.info(
            "VideoRecorder: έκλεισε (%d frames, %.1fs) -> %s",
            self._frame_count, duration_s, self._path,
        )
        return summary
