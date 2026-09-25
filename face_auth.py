import logging
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Model paths & download URLs 
_BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"

YUNET_PATH  = Path("face_detection_yunet_2023mar.onnx")
SFACE_PATH  = Path("face_recognition_sface_2021dec.onnx")

YUNET_URL = f"{_BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = f"{_BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx"

DB_PATH   = Path("trusted_faces.npz")


THUMB_DIR  = Path("trusted_thumbnails")
THUMB_SIZE = 96   # px, τετράγωνο· η τελική κλιμάκωση γίνεται στο UI layer

# Cosine similarity threshold (SFace embeddings, 128-dim).
# OpenCV SFace documentation: same person > 0.363, different < 0.363
# Χρησιμοποιούμε συντηρητικά 0.38 για λιγότερα false positives.
THRESHOLD = 0.38


def _ensure_model(path: Path, url: str) -> None:
    if path.exists():
        return
    logger.info("Downloading %s …", path.name)
    print(f"[Setup] Κατεβάζω μοντέλο: {path.name}")
    urllib.request.urlretrieve(url, path)
    print(f"[Setup] ✓ {path.name} έτοιμο.")


class FaceAuthenticator:

    def __init__(self, db_path: Path = DB_PATH):
        self._db_path = db_path
        self._lock    = threading.Lock()
        self._names:      list = []
        self._embeddings: list = []   # list of np.ndarray shape (128,)

        _ensure_model(YUNET_PATH, YUNET_URL)
        _ensure_model(SFACE_PATH, SFACE_URL)

        self._detector = cv2.FaceDetectorYN.create(
            str(YUNET_PATH),
            "",
            (320, 320),          # θα αλλάζει δυναμικά ανά frame
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5,             # max 5 πρόσωπα
        )

        # SFace — 128-dim face recognition embeddings
        self._recognizer = cv2.FaceRecognizerSF.create(
            str(SFACE_PATH), ""
        )

    # Persistence 

    def load(self) -> None:
        if not self._db_path.exists():
            logger.info("No face database found. Starting fresh.")
            return
        data = np.load(self._db_path, allow_pickle=True)
        with self._lock:
            self._names      = list(data["names"])
            raw              = data["embeddings"]
            # Υποστήριξη παλιών .npz με λάθος διαστάσεις
            self._embeddings = [
                e for e in raw if hasattr(e, "shape") and e.shape == (128,)
            ]
            if len(self._embeddings) != len(self._names):
                logger.warning(
                    "Database had %d names but %d valid embeddings — clearing.",
                    len(self._names), len(self._embeddings),
                )
                self._names      = []
                self._embeddings = []
        logger.info("Loaded %d trusted face(s).", len(self._names))

    def save(self) -> None:
        with self._lock:
            np.savez(
                self._db_path,
                names=np.array(self._names, dtype=object),
                embeddings=np.array(self._embeddings),
            )
        logger.info("Saved %d trusted face(s).", len(self._names))

    # Enrollment 

    def enroll_from_frames(self, frames: list, name: str) -> bool:
       
        collected     = []
        thumbnail_crop = None

        for frame in frames:
            emb, crop = self._extract_embedding_and_crop(frame)
            if emb is not None:
                collected.append(emb)
                if thumbnail_crop is None:
                    thumbnail_crop = crop

        if len(collected) < 5:
            logger.warning(
                "Enrollment needs >=5 valid frames; got %d.", len(collected)
            )
            return False

        mean_emb = np.mean(collected, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb /= norm

        with self._lock:
            # Αντικατάσταση αν υπάρχει ήδη το όνομα
            if name in self._names:
                idx = self._names.index(name)
                self._embeddings[idx] = mean_emb
                logger.info("Updated enrollment for '%s'.", name)
            else:
                self._names.append(name)
                self._embeddings.append(mean_emb)
                logger.info("Enrolled '%s' from %d frames.", name, len(collected))

        self.save()
        self._save_thumbnail(name, thumbnail_crop)
        return True

    def _save_thumbnail(self, name: str, crop: "np.ndarray | None") -> None:
        if crop is None:
            logger.warning("Δεν βρέθηκε aligned crop για thumbnail του '%s'.", name)
            return
        THUMB_DIR.mkdir(parents=True, exist_ok=True)
        resized = cv2.resize(crop, (THUMB_SIZE, THUMB_SIZE), interpolation=cv2.INTER_AREA)
        path = THUMB_DIR / f"{name}.png"
        cv2.imwrite(str(path), resized)
        logger.info("Thumbnail αποθηκεύτηκε: %s", path)

    def enroll_from_frame(self, frame_bgr: np.ndarray, name: str) -> bool:
        emb = self._extract_embedding(frame_bgr)
        if emb is None:
            return False
        return self.enroll_from_frames([frame_bgr] * 10, name)

    def list_trusted(self) -> list:
        with self._lock:
            return list(self._names)

    def remove_trusted(self, name: str) -> bool:
        with self._lock:
            if name not in self._names:
                return False
            idx = self._names.index(name)
            self._names.pop(idx)
            self._embeddings.pop(idx)
        self.save()
        thumb_path = THUMB_DIR / f"{name}.png"
        if thumb_path.exists():
            thumb_path.unlink()
        return True

    # Real-time analysis 

    def analyse_frame(self, frame_bgr: np.ndarray) -> "AuthResult":
        h, w = frame_bgr.shape[:2]
        self._detector.setInputSize((w, h))

        _, faces = self._detector.detect(frame_bgr)

        if faces is None or len(faces) == 0:
            return AuthResult(total=0, trusted=[], unknown=0)

        total = len(faces)

        with self._lock:
            db_embs  = list(self._embeddings)
            db_names = list(self._names)

        trusted, unknown = [], 0
       
        face_matches = []

        for face in faces:
            # SFace απαιτεί aligned crop
            aligned = self._recognizer.alignCrop(frame_bgr, face)
            emb     = self._recognizer.feature(aligned)
            emb     = emb.flatten()
            norm    = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm

            if not db_embs:
                unknown += 1
                face_matches.append({
                    "best_name": None, "best_sim": None, "is_trusted": False,
                })
                continue

            # Cosine similarity (1 - distance)
            sims     = [float(np.dot(emb, d)) for d in db_embs]
            best_idx = int(np.argmax(sims))
            best_sim = sims[best_idx]
            is_trusted = best_sim >= THRESHOLD

            logger.debug(
                "Face → best=%s  sim=%.4f  threshold=%.4f  → %s",
                db_names[best_idx],
                best_sim,
                THRESHOLD,
                "TRUSTED" if is_trusted else "UNKNOWN",
            )

            face_matches.append({
                "best_name":  db_names[best_idx],
                "best_sim":   best_sim,
                "is_trusted": is_trusted,
            })

            if is_trusted:
                trusted.append(db_names[best_idx])
            else:
                unknown += 1

        return AuthResult(
            total=total, trusted=trusted, unknown=unknown,
            face_matches=face_matches,
        )

    # Helpers 

    def _extract_embedding(self, frame_bgr: np.ndarray):
        """Επιστρέφει 128-dim SFace embedding ή None αν δεν ανιχνευθεί πρόσωπο."""
        emb, _ = self._extract_embedding_and_crop(frame_bgr)
        return emb

    def _extract_embedding_and_crop(self, frame_bgr: np.ndarray):
       
        h, w = frame_bgr.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame_bgr)

        if faces is None or len(faces) != 1:
            return None, None

        aligned = self._recognizer.alignCrop(frame_bgr, faces[0])
        emb     = self._recognizer.feature(aligned).flatten()
        norm    = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return emb, aligned


# Result 

class AuthResult:
    __slots__ = (
        "total_faces", "trusted_faces", "unknown_count", "only_trusted_present",
        "face_matches",
    )

    def __init__(self, total: int, trusted: list, unknown: int, face_matches: list = None):
        self.total_faces          = total
        self.trusted_faces        = trusted
        self.unknown_count        = unknown
        self.only_trusted_present = (total > 0 and unknown == 0)
        # list[dict(best_name, best_sim, is_trusted)] — ένα per ανιχνευμένο
        # πρόσωπο· άδεια λίστα αν total==0. Βλ. analyse_frame().
        self.face_matches         = face_matches if face_matches is not None else []

    def max_unknown_similarity(self):
        
        sims = [m["best_sim"] for m in self.face_matches
                if not m["is_trusted"] and m["best_sim"] is not None]
        return max(sims) if sims else None

    def __repr__(self):
        return (f"AuthResult(total={self.total_faces}, "
                f"trusted={self.trusted_faces}, unknown={self.unknown_count})")
