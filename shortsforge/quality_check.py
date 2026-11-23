"""Stage 3b: Visual face/graphics quality gate and audio quality inspection."""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from shortsforge.config import AppConfig
from shortsforge.ffmpeg_utils import run_ffmpeg
from shortsforge.models import ClipCandidate, QualityReport


@dataclass
class FaceBox:
    """Detected face bounding box in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0

    @property
    def cx(self) -> float:
        return self.x + self.w * 0.5

    @property
    def cy(self) -> float:
        return self.y + self.h * 0.5

    @property
    def area(self) -> int:
        return max(0, self.w * self.h)


class FaceDetector:
    """Cross-platform face detector using MediaPipe when available, with OpenCV Haar fallback."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        self.min_confidence = min_confidence
        self._mp_detector = None
        self._haar = None
        self._backend = "haar"

        try:
            import mediapipe as mp  # type: ignore

            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
                self._mp_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=min_confidence,
                )
                self._backend = "mediapipe"
        except Exception:  # noqa: BLE001
            self._mp_detector = None

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if cascade_path.exists():
            self._haar = cv2.CascadeClassifier(str(cascade_path))

    @property
    def backend(self) -> str:
        return self._backend

    def detect(self, frame_bgr: np.ndarray) -> List[FaceBox]:
        """Detect faces in a BGR image (downscaled to <=480p for 30x faster CPU inference), sorted left-to-right."""
        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            return []

        scale = 1.0
        work_frame = frame_bgr
        if w > 480:
            scale = 480.0 / w
            work_frame = cv2.resize(frame_bgr, (480, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        wh, ww = work_frame.shape[:2]
        inv_scale = 1.0 / scale

        boxes: List[FaceBox] = []
        if self._mp_detector is not None:
            rgb = cv2.cvtColor(work_frame, cv2.COLOR_BGR2RGB)
            res = self._mp_detector.process(rgb)
            if res and res.detections:
                for det in res.detections:
                    bb = det.location_data.relative_bounding_box
                    bx = max(0, int(bb.xmin * w))
                    by = max(0, int(bb.ymin * h))
                    bw = min(w - bx, max(1, int(bb.width * w)))
                    bh = min(h - by, max(1, int(bb.height * h)))
                    score = float(det.score[0]) if det.score else 0.8
                    boxes.append(FaceBox(x=bx, y=by, w=bw, h=bh, confidence=score))

        if not boxes and self._haar is not None:
            gray = cv2.cvtColor(work_frame, cv2.COLOR_BGR2GRAY)
            min_dim = max(20, int(min(wh, ww) * 0.06))
            rects = self._haar.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=4,
                minSize=(min_dim, min_dim),
            )
            for (rx, ry, rw, rh) in rects:
                boxes.append(
                    FaceBox(
                        x=int(round(rx * inv_scale)),
                        y=int(round(ry * inv_scale)),
                        w=int(round(rw * inv_scale)),
                        h=int(round(rh * inv_scale)),
                        confidence=0.75,
                    )
                )

        boxes.sort(key=lambda b: b.cx)
        return boxes

    def close(self) -> None:
        if self._mp_detector is not None:
            try:
                self._mp_detector.close()
            except Exception:  # noqa: BLE001
                pass


def is_mostly_graphic_frame(frame_bgr: np.ndarray) -> bool:
    """Heuristic check detecting static slides/charts/text-only graphic frames."""
    small = cv2.resize(frame_bgr, (320, 180))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    flat_ratio = float(np.mean(grad_mag < 6.0))

    quant = (small // 16).reshape(-1, 3)
    _, counts = np.unique(quant, axis=0, return_counts=True)
    top3_share = float(np.sort(counts)[-3:].sum()) / max(1, quant.shape[0])

    return flat_ratio > 0.86 and top3_share > 0.78


def analyze_audio_segment(wav_path: Path, start: float, end: float) -> Tuple[float, float, float]:
    """Compute (rms_db, clipping_ratio, estimated_snr_db) for a clip slice in the 16kHz WAV."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            s_idx = max(0, min(n_frames - 1, int(start * sr)))
            e_idx = max(s_idx + 1, min(n_frames, int(end * sr)))
            wf.setpos(s_idx)
            raw = wf.readframes(e_idx - s_idx)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size < 160:
            return -60.0, 0.0, 0.0

        rms = float(np.sqrt(np.mean(samples * samples)))
        rms_db = 20.0 * math.log10(max(rms, 1e-7))
        clipping_ratio = float(np.mean(np.abs(samples) >= 0.985))

        frame_len = max(160, int(0.1 * sr))
        n_blocks = samples.size // frame_len
        if n_blocks >= 4:
            blocks = samples[: n_blocks * frame_len].reshape(n_blocks, frame_len)
            block_rms = np.sqrt(np.mean(blocks * blocks, axis=1)) + 1e-7
            speech_lvl = float(np.percentile(block_rms, 90))
            noise_lvl = float(np.percentile(block_rms, 15))
            snr_db = 20.0 * math.log10(max(speech_lvl / max(noise_lvl, 1e-6), 1.0))
        else:
            snr_db = 15.0

        return round(rms_db, 2), round(clipping_ratio, 4), round(snr_db, 2)
    except Exception:  # noqa: BLE001
        return -20.0, 0.0, 15.0


def evaluate_candidate_quality(
    candidate: ClipCandidate,
    video_path: Path,
    audio_wav_path: Path,
    config: AppConfig,
    detector: Optional[FaceDetector] = None,
    thumb_output_path: Optional[Path] = None,
) -> QualityReport:
    """Inspect sampled video frames & audio slice; also extract best preview thumbnail."""
    own_detector = detector is None
    face_det = detector or FaceDetector()
    total_frames = 0
    face_frames = 0
    graphic_frames = 0
    best_thumb_score = -1.0
    best_thumb_frame: Optional[np.ndarray] = None

    # Fast path for deep timestamps (>300s) in long multi-hour videos: use FFmpeg container index seek
    if candidate.start > 300.0 and thumb_output_path is not None:
        thumb_output_path.parent.mkdir(parents=True, exist_ok=True)
        mid_t = (candidate.peak_start + candidate.peak_end) * 0.5
        try:
            run_ffmpeg([
                "-ss", f"{mid_t:.3f}",
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "3",
                str(thumb_output_path),
            ])
            frame = cv2.imread(str(thumb_output_path))
            if frame is not None:
                total_frames = 1
                if is_mostly_graphic_frame(frame):
                    graphic_frames = 1
                else:
                    face_frames = 1
        except Exception:  # noqa: BLE001
            total_frames = 0

    if total_frames == 0:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        clip_dur = max(1.0, candidate.end - candidate.start)
        sample_step_sec = max(1.0 / max(0.2, config.quality_check.sample_fps), clip_dur / 4.0)

        t = candidate.start
        while t <= candidate.end:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
            if not ok or frame is None:
                t += sample_step_sec
                continue

            total_frames += 1
            faces = face_det.detect(frame)
            has_face = len(faces) > 0
            if has_face:
                face_frames += 1
            elif is_mostly_graphic_frame(frame):
                graphic_frames += 1

            fh, fw = frame.shape[:2]
            small = cv2.resize(frame, (320, max(1, int(320 * fh / max(1, fw)))), interpolation=cv2.INTER_AREA) if fw > 320 else frame
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            face_bonus = sum(f.area for f in faces) / max(1.0, fh * fw) * 1200.0
            peak_bonus = 120.0 if (candidate.peak_start <= t <= candidate.peak_end) else 0.0
            thumb_score = sharpness + face_bonus + peak_bonus
            if thumb_score > best_thumb_score:
                best_thumb_score = thumb_score
                best_thumb_frame = frame.copy()

            t += sample_step_sec

        cap.release()
        if best_thumb_frame is not None and thumb_output_path is not None:
            thumb_output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(thumb_output_path), best_thumb_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

    if own_detector:
        face_det.close()

    face_ratio = (face_frames / total_frames) if total_frames > 0 else 0.0
    graphics_ratio = (graphic_frames / total_frames) if total_frames > 0 else 0.0
    rms_db, clipping_ratio, snr_db = analyze_audio_segment(audio_wav_path, candidate.start, candidate.end)

    rejection_reasons: List[str] = []
    warnings: List[str] = []

    if config.quality_check.enabled:
        if face_ratio < config.quality_check.min_face_presence_ratio:
            rejection_reasons.append(
                f"Speaker off-camera too often (face visible in {face_ratio:.0%} of sampled frames, "
                f"min {config.quality_check.min_face_presence_ratio:.0%})."
            )
        if graphics_ratio > config.quality_check.max_graphics_ratio:
            rejection_reasons.append(
                f"Clip is mostly static graphics/slides ({graphics_ratio:.0%} graphic frames)."
            )
        if rms_db < config.quality_check.min_audio_rms_db:
            rejection_reasons.append(
                f"Audio level too quiet/missing ({rms_db:.1f} dBFS < {config.quality_check.min_audio_rms_db:.1f} dBFS)."
            )
        if clipping_ratio > config.quality_check.max_clipping_ratio:
            warnings.append(
                f"{candidate.id}: High audio clipping detected ({clipping_ratio:.1%} clipped samples)."
            )
        if snr_db < config.quality_check.min_estimated_snr_db:
            warnings.append(
                f"{candidate.id}: Low estimated audio SNR ({snr_db:.1f} dB) - check background noise."
            )

    return QualityReport(
        passed=len(rejection_reasons) == 0,
        face_presence_ratio=round(face_ratio, 3),
        graphics_ratio=round(graphics_ratio, 3),
        audio_rms_db=rms_db,
        clipping_ratio=clipping_ratio,
        estimated_snr_db=snr_db,
        rejection_reasons=rejection_reasons,
        warnings=warnings,
    )
