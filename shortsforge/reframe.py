"""Stage 5: Smart 9:16 reframing with PySceneDetect, smooth face tracking, 2-person split, and 4:3 blurred fit."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from shortsforge.config import AppConfig
from shortsforge.ffmpeg_utils import (
    FFmpegExecutionError,
    extract_stream_info,
    get_ffmpeg_bin,
    probe_media,
    run_ffmpeg,
)
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
    if cw == out_w and ch == out_h:
        return np.ascontiguousarray(patch)
    interp = cv2.INTER_AREA if cw > out_w else cv2.INTER_CUBIC
    return cv2.resize(patch, (out_w, out_h), interpolation=interp)


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
    audio_output_wav: Path | None = None,
    thumb_output_jpg: Path | None = None,
    peak_offset_sec: float = 0.0,
) -> Path:
    """Decode [start_sec, end_sec] losslessly via pipe, smoothly track speaker face to 1080x1920, and write crisp high-bitrate MP4."""
    output_video.parent.mkdir(parents=True, exist_ok=True)
    audio_slice_wav = audio_output_wav or output_video.with_name(f"{output_video.stem}_audio_slice.wav")
    audio_slice_wav.parent.mkdir(parents=True, exist_ok=True)
    slice_dur = max(0.5, end_sec - start_sec)

    # Probe exact source dimensions & native FPS without lossy intermediate MP4 re-encode
    info = extract_stream_info(probe_media(source_video))
    src_w = max(2, int(info["width"]))
    src_h = max(2, int(info["height"]))
    src_fps = float(info.get("fps") or 30.0)
    out_fps_val = (
        round(min(60.0, max(23.976, src_fps)), 3)
        if getattr(config.reframe, "preserve_source_fps", True)
        else float(config.reframe.output_fps)
    )
    out_fps = max(24, int(round(out_fps_val)))
    src_ar = src_w / max(1, src_h)
    total_frames = max(1, int(round(slice_dur * out_fps_val)))

    # Extract raw uncompressed audio slice cleanly (reused by render_single_short if audio_output_wav is set)
    run_ffmpeg([
        "-ss", f"{start_sec:.3f}",
        "-i", str(source_video),
        "-t", f"{slice_dur:.3f}",
        "-vn",
        "-ar", "48000",
        "-ac", "2",
        "-c:a", "pcm_s16le",
        str(audio_slice_wav),
    ])

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
    vf_chain.append("format=yuv420p")

    # Decode raw lossless BGR frames directly from source (zero ultrafast compression artifacts!)
    dec_cmd = [
        get_ffmpeg_bin(),
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{start_sec:.3f}",
        "-i", str(source_video),
        "-t", f"{slice_dur:.3f}",
        "-r", f"{out_fps_val}",
        "-an",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    dec_proc = subprocess.Popen(
        dec_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    is_lossless = getattr(config.export, "lossless_intermediates", True)
    enc_quality_args = (
        ["-c:v", "libx264", "-qp", "0", "-preset", "ultrafast", "-g", "30"]
        if is_lossless
        else ["-c:v", "libx264", "-profile:v", "high", "-crf", "12", "-preset", "ultrafast", "-g", "30"]
    )
    enc_cmd = [
        get_ffmpeg_bin(),
        "-hide_banner",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{out_w}x{out_h}",
        "-r", f"{out_fps_val}",
        "-i", "pipe:0",
        "-i", str(audio_slice_wav),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-vf", ",".join(vf_chain),
        *enc_quality_args,
        "-pix_fmt", "yuv420p",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-c:a", "pcm_s16le",
        "-shortest",
        str(reframed_tmp),
    ]
    proc = subprocess.Popen(
        enc_cmd,
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

    base_stride = max(1, out_fps // 5)
    calm_stride = max(base_stride, out_fps // 3)
    detect_stride = base_stride
    stationary_count = 0
    last_faces: List[FaceBox] = []
    frame_bytes = src_w * src_h * 3

    best_thumb_score = -1.0
    best_thumb_frame: np.ndarray | None = None
    thumb_stride = max(1, out_fps // 2)

    frame_idx = 0
    try:
        assert dec_proc.stdout is not None
        assert proc.stdin is not None
        while True:
            raw_buf = dec_proc.stdout.read(frame_bytes)
            if len(raw_buf) < frame_bytes:
                break
            frame = np.frombuffer(raw_buf, dtype=np.uint8).reshape((src_h, src_w, 3))

            if frame_idx % detect_stride == 0:
                detected = face_det.detect(frame)
                if detected:
                    primary_det = max(detected, key=lambda b: b.area)
                    if initialized_cam and abs(primary_det.cx - cam_x) <= deadzone_px:
                        stationary_count += 1
                        if stationary_count >= 3:
                            detect_stride = calm_stride
                    else:
                        stationary_count = 0
                        detect_stride = base_stride
                    last_faces = detected

            if is_four_by_three:
                prog = frame_idx / max(1, total_frames - 1)
                zoom_now = 1.0 + (config.reframe.four_by_three_zoom - 1.0) * prog
                out_frame = _render_blurred_fit(frame, out_w, out_h, zoom=zoom_now)
            else:
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

            if thumb_output_jpg is not None and (frame_idx % thumb_stride == 0):
                cur_t = frame_idx / max(1.0, out_fps_val)
                small_gray = cv2.cvtColor(
                    cv2.resize(out_frame, (270, 480), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2GRAY,
                )
                sharpness = float(cv2.Laplacian(small_gray, cv2.CV_64F).var())
                face_bonus = sum(f.area for f in last_faces) / max(1.0, src_w * src_h) * 1800.0
                peak_bonus = 200.0 if abs(cur_t - peak_offset_sec) <= 1.8 else 0.0
                t_score = sharpness + face_bonus + peak_bonus
                if t_score > best_thumb_score:
                    best_thumb_score = t_score
                    best_thumb_frame = out_frame.copy()

            proc.stdin.write(out_frame.tobytes())
            frame_idx += 1

        dec_err = dec_proc.stderr.read().decode("utf-8", errors="replace") if dec_proc.stderr else ""
        dec_ret = dec_proc.wait()
        proc.stdin.close()
        stderr_out = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        ret = proc.wait()
        if dec_ret != 0 and frame_idx == 0:
            raise FFmpegExecutionError(f"Reframe decode failed (exit {dec_ret}):\n{dec_err[-1200:]}")
        if ret != 0:
            raise FFmpegExecutionError(f"Reframe ffmpeg pipe failed (exit {ret}):\n{stderr_out[-1200:]}")
    finally:
        if dec_proc.poll() is None:
            dec_proc.kill()
        if own_detector:
            face_det.close()
        if audio_output_wav is None:
            audio_slice_wav.unlink(missing_ok=True)

    if thumb_output_jpg is not None and best_thumb_frame is not None:
        thumb_output_jpg.parent.mkdir(parents=True, exist_ok=True)
        if getattr(config.export, "mirror_video", False):
            best_thumb_frame = cv2.flip(best_thumb_frame, 1)
        cv2.imwrite(str(thumb_output_jpg), best_thumb_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    reframed_tmp.replace(output_video)
    return output_video
