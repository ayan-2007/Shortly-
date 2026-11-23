"""Cross-platform ffmpeg & ffprobe discovery, probing, and safe subprocess execution."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple


class FFmpegNotFoundError(RuntimeError):
    """Raised when ffmpeg or ffprobe cannot be located on the system."""


class FFmpegExecutionError(RuntimeError):
    """Raised when an ffmpeg or ffprobe command returns a non-zero exit code."""


def _find_binary(name: str) -> str:
    """Locate ffmpeg or ffprobe on PATH or common OS package manager directories."""
    found = shutil.which(name)
    if found:
        return found

    exe = f"{name}.exe" if platform.system().lower() == "windows" else name

    candidate_roots: List[Path] = []
    if platform.system().lower() == "windows":
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            candidate_roots.append(Path(local_app) / "Microsoft" / "WinGet" / "Links")
            candidate_roots.append(Path(local_app) / "Microsoft" / "WinGet" / "Packages")
        program_data = os.environ.get("ProgramData", "")
        if program_data:
            candidate_roots.append(Path(program_data) / "chocolatey" / "bin")
        user_profile = os.environ.get("USERPROFILE", "")
        if user_profile:
            candidate_roots.append(Path(user_profile) / "scoop" / "shims")
    else:
        candidate_roots.extend([
            Path("/opt/homebrew/bin"),
            Path("/usr/local/bin"),
            Path("/usr/bin"),
        ])

    for root in candidate_roots:
        direct = root / exe
        if direct.is_file():
            return str(direct)
        if root.exists() and root.name == "Packages":
            for match in root.rglob(exe):
                if match.is_file():
                    return str(match)

    raise FFmpegNotFoundError(
        f"'{name}' was not found on PATH. Install FFmpeg (see README.md for Windows/macOS/Linux instructions)."
    )


def get_ffmpeg_bin() -> str:
    return _find_binary("ffmpeg")


def get_ffprobe_bin() -> str:
    return _find_binary("ffprobe")


def run_ffmpeg(args: List[str], capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    """Run an ffmpeg command with clear error diagnostics."""
    cmd = [get_ffmpeg_bin(), "-hide_banner", "-y", *args]
    proc = subprocess.run(
        cmd,
        capture_output=capture_output,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "")[-1500:]
        raise FFmpegExecutionError(f"ffmpeg failed (exit {proc.returncode}):\n{stderr_tail}")
    return proc


def probe_media(media_path: Path | str) -> Dict[str, Any]:
    """Return parsed ffprobe JSON metadata for a media file."""
    cmd = [
        get_ffprobe_bin(),
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(media_path),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise FFmpegExecutionError(f"ffprobe failed on {media_path}: {proc.stderr}")
    return json.loads(proc.stdout)


def extract_stream_info(probe_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract duration, width, height, fps, aspect_ratio, and audio presence from ffprobe output."""
    streams = probe_data.get("streams", [])
    fmt = probe_data.get("format", {})
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = float(fmt.get("duration") or 0.0)
    width = int(video_stream.get("width", 0)) if video_stream else 0
    height = int(video_stream.get("height", 0)) if video_stream else 0

    fps = 30.0
    if video_stream:
        rate_str = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "30/1"
        try:
            num, den = rate_str.split("/")
            den_f = float(den)
            if den_f > 0:
                fps = float(num) / den_f
        except (ValueError, ZeroDivisionError):
            fps = 30.0
        if duration <= 0.0 and video_stream.get("duration"):
            duration = float(video_stream["duration"])

    aspect_ratio = (width / height) if (width > 0 and height > 0) else 16 / 9
    return {
        "duration": duration,
        "width": width,
        "height": height,
        "fps": fps,
        "aspect_ratio": aspect_ratio,
        "has_video": video_stream is not None,
        "has_audio": audio_stream is not None,
        "video_codec": video_stream.get("codec_name", "") if video_stream else "",
        "audio_codec": audio_stream.get("codec_name", "") if audio_stream else "",
    }


def escape_ffmpeg_filter_path(path: Path | str) -> str:
    """Escape a filesystem path for use inside ffmpeg filter arguments."""
    resolved = str(Path(path).resolve()).replace("\\", "/")
    resolved = resolved.replace(":", "\\:").replace("'", "\\'")
    return resolved


@lru_cache(maxsize=2)
def _probe_working_hw_encoder(allow_hw: bool = True) -> str:
    """Probe the fastest working hardware H.264 encoder once per process (NVENC -> QSV -> AMF -> libx264)."""
    if not allow_hw:
        return "libx264"
    ffmpeg_bin = get_ffmpeg_bin()
    candidates = [
        ("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"]),
        ("h264_qsv", ["-c:v", "h264_qsv", "-preset", "fast", "-global_quality", "20", "-pix_fmt", "nv12"]),
        ("h264_amf", ["-c:v", "h264_amf", "-quality", "balanced", "-rc", "cqp", "-qp_i", "20", "-qp_p", "20"]),
    ]
    for name, enc_args in candidates:
        try:
            proc = subprocess.run(
                [
                    ffmpeg_bin,
                    "-hide_banner",
                    "-loglevel", "error",
                    "-f", "lavfi",
                    "-i", "color=c=black:s=640x360:d=0.15",
                    *enc_args,
                    "-f", "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                timeout=8,
            )
            if proc.returncode == 0:
                return name
        except Exception:  # noqa: BLE001
            continue
    return "libx264"


def get_h264_encode_args(
    crf: int = 18,
    preset: str = "veryfast",
    hw_accel: bool = True,
) -> Tuple[str, List[str], str]:
    """Return (encoder_name, ffmpeg_video_args, filter_pix_fmt) using hardware GPU acceleration when available."""
    enc = _probe_working_hw_encoder(allow_hw=bool(hw_accel))
    q = max(14, min(28, int(crf) + 2))
    if enc == "h264_nvenc":
        return (
            enc,
            [
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-rc", "vbr",
                "-cq", str(q),
                "-b:v", "0",
                "-profile:v", "high",
                "-pix_fmt", "yuv420p",
            ],
            "yuv420p",
        )
    if enc == "h264_qsv":
        return (
            enc,
            [
                "-c:v", "h264_qsv",
                "-preset", "fast",
                "-global_quality", str(q),
                "-g", "60",
                "-pix_fmt", "nv12",
            ],
            "nv12",
        )
    if enc == "h264_amf":
        return (
            enc,
            [
                "-c:v", "h264_amf",
                "-quality", "balanced",
                "-rc", "cqp",
                "-qp_i", str(q),
                "-qp_p", str(q),
                "-pix_fmt", "yuv420p",
            ],
            "yuv420p",
        )
    return (
        "libx264",
        [
            "-c:v", "libx264",
            "-profile:v", "high",
            "-level", "4.2",
            "-tune", "film",
            "-crf", str(crf),
            "-preset", preset,
            "-x264-params", "aq-mode=2",
            "-pix_fmt", "yuv420p",
        ],
        "yuv420p",
    )
