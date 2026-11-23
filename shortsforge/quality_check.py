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


_YUNET_MODEL_PATH = Path(__file__).resolve().parent / "assets" / "face_detection_yunet_2023mar.onnx"
_HAAR_FRONTAL_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
_HAAR_ALT2_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml"
_HAAR_PROFILE_PATH = Path(cv2.data.haarcascades) / "haarcascade_profileface.xml"


class FaceDetector:
    """High-speed cross-platform face detector using OpenCV YuNet DNN with multi-angle Haar frontal+profile fallback."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        self.min_confidence = min_confidence
        self._yunet = None
        self._mp_detector = None
        self._haar = None
        self._haar_alt2 = None
        self._haar_profile = None
        self._backend = "haar"

        # 1. Primary ultra-fast neural detector: OpenCV YuNet ONNX (~5ms init, ~12ms/frame inference, frontal + 3/4 + profile)
        if _YUNET_MODEL_PATH.is_file() and hasattr(cv2, "FaceDetectorYN"):
            try:
                self._yunet = cv2.FaceDetectorYN.create(
                    str(_YUNET_MODEL_PATH),
                    "",
                    (480, 270),
                    max(0.45, float(min_confidence)),
                    0.3,
                    5000,
                )
                self._backend = "yunet"
            except Exception:  # noqa: BLE001
                self._yunet = None

        # 2. Secondary fallback: Frontal + Profile Haar cascades (100% offline & catches synthetic/extreme-profile frames)
        if _HAAR_FRONTAL_PATH.exists():
            self._haar = cv2.CascadeClassifier(str(_HAAR_FRONTAL_PATH))
        if _HAAR_ALT2_PATH.exists():
            self._haar_alt2 = cv2.CascadeClassifier(str(_HAAR_ALT2_PATH))
        if _HAAR_PROFILE_PATH.exists():
            self._haar_profile = cv2.CascadeClassifier(str(_HAAR_PROFILE_PATH))

    @property
    def backend(self) -> str:
        return self._backend

    def detect(self, frame_bgr: np.ndarray) -> List[FaceBox]:
        """Detect faces in a BGR image (downscaled to <=480p for fast CPU inference), sorted left-to-right."""
        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            return []

        scale = 1.0
        work_frame = frame_bgr
        if w > 480:
            scale = 480.0 / w
            work_frame = cv2.resize(frame_bgr, (480, max(2, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        wh, ww = work_frame.shape[:2]
        inv_scale = 1.0 / scale
        min_face_area = (w * h) * 0.0022  # Filter out tiny background poster/artifact faces (<4.7% of frame width)

        boxes: List[FaceBox] = []
        if self._yunet is not None:
            try:
                self._yunet.setInputSize((ww, wh))
                _, dets = self._yunet.detect(work_frame)
                if dets is not None and len(dets) > 0:
                    for det in dets:
                        bx = max(0, int(round(float(det[0]) * inv_scale)))
                        by = max(0, int(round(float(det[1]) * inv_scale)))
                        bw = min(w - bx, max(1, int(round(float(det[2]) * inv_scale))))
                        bh = min(h - by, max(1, int(round(float(det[3]) * inv_scale))))
                        score = float(det[-1])
                        if bw * bh >= min_face_area and score >= self.min_confidence:
                            boxes.append(FaceBox(x=bx, y=by, w=bw, h=bh, confidence=score))
            except Exception:  # noqa: BLE001
                pass

        if not boxes:
            gray = cv2.cvtColor(work_frame, cv2.COLOR_BGR2GRAY)
            min_dim = max(22, int(min(wh, ww) * 0.075))
            for cascade in (self._haar_alt2, self._haar):
                if cascade is None:
                    continue
                rects = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.15,
                    minNeighbors=4,
                    minSize=(min_dim, min_dim),
                )
                if len(rects) > 0:
                    for (rx, ry, rw, rh) in rects:
                        bx = int(round(rx * inv_scale))
                        by = int(round(ry * inv_scale))
                        bw = int(round(rw * inv_scale))
                        bh = int(round(rh * inv_scale))
                        if bw * bh >= min_face_area:
                            boxes.append(FaceBox(x=bx, y=by, w=bw, h=bh, confidence=0.78))
                    break

            # Side-profile fallback (checks both left-facing and right-facing podcast camera angles)
            if not boxes and self._haar_profile is not None:
                for flipped in (False, True):
                    img_g = cv2.flip(gray, 1) if flipped else gray
                    rects = self._haar_profile.detectMultiScale(
                        img_g,
                        scaleFactor=1.16,
                        minNeighbors=4,
                        minSize=(min_dim, min_dim),
                    )
                    if len(rects) > 0:
                        for (rx, ry, rw, rh) in rects:
                            real_rx = (ww - rx - rw) if flipped else rx
                            bx = max(0, int(round(real_rx * inv_scale)))
                            by = max(0, int(round(ry * inv_scale)))
                            bw = int(round(rw * inv_scale))
                            bh = int(round(rh * inv_scale))
                            if bw * bh >= min_face_area:
                                boxes.append(FaceBox(x=bx, y=by, w=bw, h=bh, confidence=0.72))
                        break

        boxes.sort(key=lambda b: b.cx)
        return boxes

    def close(self) -> None:
        self._yunet = None
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
    video_cap: Optional[cv2.VideoCapture] = None,
) -> QualityReport:
    """Inspect sampled video frames & audio slice; also extract best preview thumbnail."""
    own_detector = detector is None
    face_det = detector or FaceDetector()
    total_frames = 0
    face_frames = 0
    graphic_frames = 0
    best_thumb_score = -1.0
    best_thumb_frame: Optional[np.ndarray] = None

    own_cap = video_cap is None
    cap = video_cap if video_cap is not None else cv2.VideoCapture(str(video_path))
    try:
        if cap.isOpened():
            clip_dur = max(1.0, candidate.end - candidate.start)
            sample_step_sec = max(1.0 / max(0.2, config.quality_check.sample_fps), clip_dur / 4.0)

            t = candidate.start + 0.1
            while t <= candidate.end:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
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
                small = (
                    cv2.resize(frame, (320, max(1, int(320 * fh / max(1, fw)))), interpolation=cv2.INTER_AREA)
                    if fw > 320
                    else frame
                )
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                face_bonus = sum(f.area for f in faces) / max(1.0, fh * fw) * 1200.0
                peak_bonus = 120.0 if (candidate.peak_start <= t <= candidate.peak_end) else 0.0
                thumb_score = sharpness + face_bonus + peak_bonus
                if thumb_score > best_thumb_score:
                    best_thumb_score = thumb_score
                    best_thumb_frame = frame.copy()

                t += sample_step_sec
    finally:
        if own_cap:
            cap.release()

    if best_thumb_frame is not None and thumb_output_path is not None:
        thumb_output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(thumb_output_path), best_thumb_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    elif total_frames == 0 and thumb_output_path is not None:
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
