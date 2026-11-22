"""Stage 5: Smart 9:16 reframing with PySceneDetect, smooth face tracking, 2-person split, and 4:3 blurred fit."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from shortsforge.config import AppConfig
from shortsforge.ffmpeg_utils import FFmpegExecutionError, get_ffmpeg_bin, run_ffmpeg
from shortsforge.quality_check import FaceBox, FaceDetector


def detect_scene_cut_frames(video_path: Path, threshold: float = 27.0) -> List[int]:
    """Use PySceneDetect to locate hard scene cut frame indices inside a clip segment."""
    try:
        from scenedetect import ContentDetector, SceneManager, open_video

        video = open_video(str(video_path))
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=threshold))
        manager.detect_scenes(video, show_progress=False)
        scene_list = manager.get_scene_list()
        if not scene_list:
            return [0]
        return sorted({int(start.get_frames()) for start, _ in scene_list} | {0})
    except Exception:  # noqa: BLE001
        return [0]


def _render_blurred_fit(
    frame_bgr: np.ndarray,
    out_w: int,
    out_h: int,
    zoom: float = 1.0,
) -> np.ndarray:
    """Place centered sharp footage over a heavily blurred 9:16 fill background."""
    src_h, src_w = frame_bgr.shape[:2]

    target_ar = out_w / out_h
    src_ar = src_w / max(1, src_h)
    if src_ar > target_ar:
        bg_crop_w = int(round(src_h * target_ar))
        x0 = max(0, (src_w - bg_crop_w) // 2)
        bg_crop = frame_bgr[:, x0 : x0 + bg_crop_w]
    else:
        bg_crop_h = int(round(src_w / target_ar))
        y0 = max(0, (src_h - bg_crop_h) // 2)
        bg_crop = frame_bgr[y0 : y0 + bg_crop_h, :]

    bg = cv2.resize(bg_crop, (out_w // 4, out_h // 4), interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(bg, (31, 31), 0)
    bg = cv2.resize(bg, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    bg = (bg.astype(np.float32) * 0.52).astype(np.uint8)

    fg_w = int(round(out_w * max(1.0, zoom)))
    fg_h = int(round(fg_w / max(1e-6, src_ar)))
    fg = cv2.resize(frame_bgr, (fg_w, fg_h), interpolation=cv2.INTER_LANCZOS4)
    if fg_w > out_w:
        cx0 = (fg_w - out_w) // 2
        fg = fg[:, cx0 : cx0 + out_w]
        fg_w = out_w
    if fg_h > out_h:
        cy0 = (fg_h - out_h) // 2
        fg = fg[cy0 : cy0 + out_h, :]
        fg_h = out_h

    y_off = (out_h - fg_h) // 2
    x_off = (out_w - fg_w) // 2
    bg[y_off : y_off + fg_h, x_off : x_off + fg_w] = fg
    return bg


def _crop_around_center(
    frame_bgr: np.ndarray,
    cx: float,
    cy: float,
    crop_w: int,
    crop_h: int,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    src_h, src_w = frame_bgr.shape[:2]
    cw = min(src_w, max(2, crop_w))
    ch = min(src_h, max(2, crop_h))
    x0 = int(round(cx - cw * 0.5))
    y0 = int(round(cy - ch * 0.42))
    x0 = max(0, min(src_w - cw, x0))
    y0 = max(0, min(src_h - ch, y0))
    patch = frame_bgr[y0 : y0 + ch, x0 : x0 + cw]
    return cv2.resize(patch, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)


def _render_two_person_split(
    frame_bgr: np.ndarray,
    left_center: Tuple[float, float],
    right_center: Tuple[float, float],
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Render stacked top/bottom 1080x960 split view tracking two speakers in a wide shot."""
    src_h, src_w = frame_bgr.shape[:2]
    half_h = out_h // 2
    pane_ar = out_w / half_h

    crop_h = min(src_h, int(round(src_h * 0.85)))
    crop_w = min(src_w // 2 + int(src_w * 0.12), int(round(crop_h * pane_ar)))
    crop_h = min(src_h, int(round(crop_w / pane_ar)))

    top = _crop_around_center(frame_bgr, left_center[0], left_center[1], crop_w, crop_h, out_w, half_h)
    bot = _crop_around_center(frame_bgr, right_center[0], right_center[1], crop_w, crop_h, out_w, out_h - half_h)

    canvas = np.vstack([top, bot])
    cv2.line(canvas, (0, half_h), (out_w, half_h), (18, 18, 22), thickness=4)
    return canvas


def _smooth_step(
    current: float,
    target: float,
    alpha: float,
    deadzone_px: float,
    max_step_px: float,
) -> float:
    delta = target - current
    if abs(delta) <= deadzone_px:
        return current
    step = delta * alpha
    if abs(step) > max_step_px:
        step = max_step_px if step > 0 else -max_step_px
    return current + step


def reframe_clip_to_vertical(
    source_video: Path,
    start_sec: float,
    end_sec: float,
    output_video: Path,
    config: AppConfig,
    detector: FaceDetector | None = None,
) -> Path:
    """Extract [start_sec, end_sec], reframe to 1080x1920 with smooth face tracking, and optionally denoise."""
    output_video.parent.mkdir(parents=True, exist_ok=True)
    work_segment = output_video.with_name(f"{output_video.stem}_raw_slice.mp4")

    out_fps = config.reframe.output_fps
    slice_dur = max(0.5, end_sec - start_sec)
    run_ffmpeg([
        "-ss", f"{start_sec:.3f}",
        "-i", str(source_video),
        "-t", f"{slice_dur:.3f}",
        "-r", str(out_fps),
        "-c:v", "libx264",
        "-crf", "17",
        "-preset", "ultrafast",
        "-c:a", "pcm_s16le",
        "-ar", "48000",
        "-ac", "2",
        str(work_segment),
    ])

    cap = cv2.VideoCapture(str(work_segment))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    src_ar = src_w / max(1, src_h)

    out_w = config.reframe.output_width
    out_h = config.reframe.output_height
    is_four_by_three = src_ar < 1.45

    own_detector = detector is None
    face_det = detector or FaceDetector()

    reframed_tmp = output_video.with_name(f"{output_video.stem}_reframed_raw.mp4")
    vf_chain: List[str] = []
    if config.reframe.enhance_old_footage:
        vf_chain.append("hqdn3d=1.5:1.5:4:4")
        vf_chain.append(f"scale={out_w}:{out_h}:flags=lanczos")
        vf_chain.append("unsharp=5:5:0.45:5:5:0.0")

    ff_cmd = [
        get_ffmpeg_bin(),
        "-hide_banner",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{out_w}x{out_h}",
        "-r", str(out_fps),
        "-i", "pipe:0",
        "-i", str(work_segment),
        "-map", "0:v:0",
        "-map", "1:a:0",
    ]
    if vf_chain:
        ff_cmd.extend(["-vf", ",".join(vf_chain)])
    ff_cmd.extend([
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "17",
        "-preset", "veryfast",
        "-c:a", "pcm_s16le",
        "-shortest",
        str(reframed_tmp),
    ])

    proc = subprocess.Popen(
        ff_cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    target_ar = out_w / out_h
    crop_h = src_h
    crop_w = min(src_w, int(round(crop_h * target_ar)))

    cam_x = src_w * 0.5
    cam_y = src_h * 0.45
    initialized_cam = False

    deadzone_px = src_w * max(0.08, config.reframe.deadzone_ratio)
    max_step_px = (config.reframe.max_pan_speed_px_per_sec * 0.65) / max(1, out_fps)
    alpha = min(0.08, config.reframe.face_smoothing_alpha)

    detect_stride = max(1, out_fps // 5)  # Smooth 5Hz face sampling with zero abrupt mid-clip cuts
    last_faces: List[FaceBox] = []

    frame_idx = 0
    try:
        assert proc.stdin is not None
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if frame_idx % detect_stride == 0:
                detected = face_det.detect(frame)
                if detected:
                    last_faces = detected

            if is_four_by_three:
                prog = frame_idx / max(1, total_frames - 1)
                zoom_now = 1.0 + (config.reframe.four_by_three_zoom - 1.0) * prog
                out_frame = _render_blurred_fit(frame, out_w, out_h, zoom=zoom_now)
            else:
                # Continuous, uninterrupted single-camera 9:16 framing (no mid-clip layout jumps or artificial cuts)
                if last_faces:
                    primary = max(last_faces, key=lambda b: b.area)
                    target_x, target_y = primary.cx, primary.cy
                    if not initialized_cam:
                        cam_x, cam_y = target_x, target_y
                        initialized_cam = True
                    else:
                        cam_x = _smooth_step(cam_x, target_x, alpha, deadzone_px, max_step_px)
                        cam_y = _smooth_step(cam_y, target_y, alpha, deadzone_px, max_step_px)

                out_frame = _crop_around_center(frame, cam_x, cam_y, crop_w, crop_h, out_w, out_h)

            proc.stdin.write(out_frame.tobytes())
            frame_idx += 1

        proc.stdin.close()
        stderr_out = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        ret = proc.wait()
        if ret != 0:
            raise FFmpegExecutionError(f"Reframe ffmpeg pipe failed (exit {ret}):\n{stderr_out[-1200:]}")
    finally:
        cap.release()
        if own_detector:
            face_det.close()
        work_segment.unlink(missing_ok=True)

    reframed_tmp.replace(output_video)
    return output_video
