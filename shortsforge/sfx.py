"""Procedural WAV generators for hook transition SFX (whoosh/rewind) and subtle hook-only ambient bed."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def _write_stereo_wav(path: Path, left: np.ndarray, right: np.ndarray, sample_rate: int = 48000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    stereo = np.column_stack([
        np.clip(left * 32767.0, -32767, 32767).astype(np.int16),
        np.clip(right * 32767.0, -32767, 32767).astype(np.int16),
    ]).ravel()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())
    return path


def generate_whoosh_rewind_sfx(output_path: Path, duration: float = 0.35, sample_rate: int = 48000) -> Path:
    """Synthesize a crisp whoosh + tape-rewind sweep SFX for the cold-open transition."""
    if output_path.exists():
        return output_path

    n = int(duration * sample_rate)
    t = np.linspace(0.0, duration, n, endpoint=False)
    norm_t = t / max(duration, 1e-6)

    rng = np.random.default_rng(42)
    noise = rng.standard_normal(n)

    kernel_size = 21
    kernel = np.hanning(kernel_size)
    kernel /= kernel.sum()
    filtered_noise = np.convolve(noise, kernel, mode="same")

    freq = 240.0 + 1800.0 * np.sin(np.pi * norm_t) ** 1.4
    phase = 2.0 * np.pi * np.cumsum(freq) / sample_rate
    rewind_tone = 0.28 * np.sin(phase)

    sub = 0.35 * np.sin(2.0 * np.pi * 85.0 * t) * np.exp(-((norm_t - 0.55) / 0.18) ** 2)

    envelope = np.sin(np.pi * norm_t) ** 1.2
    base = envelope * (0.65 * filtered_noise + rewind_tone) + sub
    pan_l = np.cos(0.5 * np.pi * norm_t) * 0.7 + 0.3
    pan_r = np.sin(0.5 * np.pi * norm_t) * 0.7 + 0.3

    peak = max(float(np.max(np.abs(base))), 1e-6)
    base = 0.75 * base / peak
    return _write_stereo_wav(output_path, base * pan_l, base * pan_r, sample_rate)


def generate_hook_ambient_bed(output_path: Path, duration: float = 3.5, sample_rate: int = 48000) -> Path:
    """Synthesize a warm, restrained low-register tension pad strictly for the hook segment."""
    if output_path.exists():
        return output_path

    n = int(duration * sample_rate)
    t = np.linspace(0.0, duration, n, endpoint=False)

    pad = (
        0.45 * np.sin(2.0 * np.pi * 73.42 * t)
        + 0.30 * np.sin(2.0 * np.pi * 110.00 * t)
        + 0.15 * np.sin(2.0 * np.pi * 174.61 * t)
        + 0.10 * np.sin(2.0 * np.pi * 220.00 * t)
    )
    lfo = 0.85 + 0.15 * np.sin(2.0 * np.pi * 0.6 * t)

    fade_in = np.clip(t / 0.2, 0.0, 1.0)
    fade_out = np.clip((duration - t) / 0.5, 0.0, 1.0)
    env = fade_in * fade_out * lfo

    left = 0.35 * pad * env
    right = 0.35 * (pad * np.cos(0.04 * t) + 0.05 * np.sin(2.0 * np.pi * 146.83 * t)) * env
    return _write_stereo_wav(output_path, left, right, sample_rate)
