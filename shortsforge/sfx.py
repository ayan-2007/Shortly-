"""Procedural studio-quality stereo sound design & mood-adaptive background music beds."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import List, Tuple

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


def _smooth_lowpass(sig: np.ndarray, kernel_len: int = 45) -> np.ndarray:
    """Fast vectorized Hann-window low-pass filter for warm, vocal-friendly pads."""
    k = np.hanning(max(3, kernel_len | 1))
    k /= k.sum()
    return np.convolve(sig, k, mode="same")


def generate_whoosh_rewind_sfx(output_path: Path, duration: float = 0.28, sample_rate: int = 48000) -> Path:
    """Synthesize a crisp cinematic whoosh + punchy sub-bass impact drop for transitions."""
    if output_path.exists():
        return output_path

    n = max(128, int(duration * sample_rate))
    t = np.linspace(0.0, duration, n, endpoint=False)
    norm_t = t / max(duration, 1e-6)

    rng = np.random.default_rng(42)
    noise = rng.standard_normal(n)
    filtered_noise = _smooth_lowpass(noise, kernel_len=25)

    # Pitch sweep + deep sub impact landing near 72% of transition
    freq = 190.0 + 1650.0 * (np.sin(np.pi * norm_t) ** 1.35)
    phase = 2.0 * np.pi * np.cumsum(freq) / sample_rate
    sweep = 0.26 * np.sin(phase)
    sub_hit = 0.65 * np.sin(2.0 * np.pi * (68.0 - 24.0 * norm_t) * t) * np.exp(-((norm_t - 0.68) / 0.16) ** 2)

    envelope = np.sin(np.pi * norm_t) ** 1.15
    base = envelope * (0.62 * filtered_noise + sweep) + sub_hit
    pan_l = np.cos(0.5 * np.pi * norm_t) * 0.65 + 0.35
    pan_r = np.sin(0.5 * np.pi * norm_t) * 0.65 + 0.35

    peak = max(float(np.max(np.abs(base))), 1e-6)
    base = 0.80 * base / peak
    return _write_stereo_wav(output_path, base * pan_l, base * pan_r, sample_rate)


def generate_hook_ambient_bed(output_path: Path, duration: float = 3.5, sample_rate: int = 48000) -> Path:
    """Synthesize a tense, cinematic opening hook drone + pulse + rising tension sweep."""
    if output_path.exists():
        return output_path

    n = max(256, int(duration * sample_rate))
    t = np.linspace(0.0, duration, n, endpoint=False)
    norm_t = t / max(duration, 1e-6)

    # Dark minor suspense chord: D1 (36.71), A1 (55.0), D2 (73.42), F2 (87.31), E2 tension (82.41)
    sub = 0.45 * np.sin(2.0 * np.pi * 55.0 * t) + 0.25 * np.sin(2.0 * np.pi * 110.0 * t)
    pad_l = (
        0.38 * np.tanh(1.6 * np.sin(2.0 * np.pi * 146.83 * t))
        + 0.28 * np.sin(2.0 * np.pi * 220.00 * t + 0.4)
        + 0.20 * np.sin(2.0 * np.pi * 174.61 * t)
        + 0.14 * np.sin(2.0 * np.pi * 164.81 * t)
    )
    pad_r = (
        0.38 * np.tanh(1.6 * np.sin(2.0 * np.pi * 147.65 * t))
        + 0.28 * np.sin(2.0 * np.pi * 219.10 * t + 0.9)
        + 0.20 * np.sin(2.0 * np.pi * 175.35 * t)
        + 0.14 * np.sin(2.0 * np.pi * 164.20 * t)
    )

    # Subtle 120 BPM tension heartbeat pulse
    beat_phase = (t * 2.0) % 1.0
    pulse = 0.72 + 0.28 * np.exp(-beat_phase / 0.14)

    # Riser sweep in final 35% into transition
    rng = np.random.default_rng(7)
    noise = _smooth_lowpass(rng.standard_normal(n), kernel_len=55)
    riser_env = np.clip((norm_t - 0.55) / 0.45, 0.0, 1.0) ** 1.8
    riser = riser_env * (0.28 * noise + 0.20 * np.sin(2.0 * np.pi * (110.0 + 340.0 * riser_env) * t))

    fade_in = np.clip(t / 0.12, 0.0, 1.0)
    fade_out = np.clip((duration - t) / 0.10, 0.0, 1.0)
    env = fade_in * fade_out

    left = _smooth_lowpass((sub + 0.55 * pad_l) * pulse + riser, kernel_len=31) * env
    right = _smooth_lowpass((sub + 0.55 * pad_r) * pulse + riser, kernel_len=31) * env

    peak = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-6)
    return _write_stereo_wav(output_path, 0.72 * left / peak, 0.72 * right / peak, sample_rate)


def _chord_progression_freqs(mood: str) -> Tuple[List[Tuple[float, float, float, float]], float, bool]:
    """Return (4-bar chord root/triad frequencies, bpm, has_pulse) for a given mood."""
    if mood == "emotional":
        # Warm Am - F - C - G cinematic emotional progression
        return [
            (55.00, 110.00, 164.81, 261.63),
            (43.65, 87.31, 174.61, 261.63),
            (65.41, 130.81, 196.00, 329.63),
            (49.00, 98.00, 196.00, 293.66),
        ], 78.0, False
    if mood == "inspiring":
        # Uplifting C - G - Am - F warm motivating progression
        return [
            (65.41, 130.81, 196.00, 329.63),
            (49.00, 98.00, 196.00, 293.66),
            (55.00, 110.00, 164.81, 261.63),
            (43.65, 87.31, 174.61, 261.63),
        ], 96.0, True
    if mood == "mysterious":
        # Deep noir Cm - Ab - Fm - G5
        return [
            (65.41, 130.81, 155.56, 196.00),
            (51.91, 103.83, 155.56, 207.65),
            (43.65, 87.31, 130.81, 207.65),
            (49.00, 98.00, 146.83, 196.00),
        ], 86.0, True
    # Default "tense" / "intense": Dm - Bb - Gm - A7sus
    return [
        (36.71, 73.42, 146.83, 174.61),
        (58.27, 116.54, 174.61, 233.08),
        (49.00, 98.00, 146.83, 196.00),
        (55.00, 110.00, 164.81, 220.00),
    ], 104.0, True


def generate_mood_music_track(
    output_path: Path,
    duration: float,
    mood: str = "tense",
    sample_rate: int = 48000,
) -> Path:
    """Synthesize a rich, low-passed mood-matched stereo music bed sized cleanly to any clip duration."""
    if output_path.exists():
        return output_path

    dur = max(1.0, float(duration))
    n = max(512, int(round(dur * sample_rate)))
    t = np.linspace(0.0, dur, n, endpoint=False)

    chords, bpm, has_pulse = _chord_progression_freqs(mood)
    bar_sec = 4.0 * (60.0 / bpm)

    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)

    # Render crossfaded chord pad progression + soft felt-piano motif
    num_bars = int(np.ceil(dur / bar_sec)) + 1
    for b_i in range(num_bars):
        b_start = b_i * bar_sec
        b_end = min(dur, (b_i + 1) * bar_sec + 0.35)
        if b_start >= dur:
            break
        i0 = int(b_start * sample_rate)
        i1 = min(n, int(b_end * sample_rate))
        if i1 <= i0:
            continue

        lt = t[i0:i1] - b_start
        chord = chords[b_i % len(chords)]
        win_len = (i1 - i0) / sample_rate
        bar_fade_in = np.clip(lt / 0.35, 0.0, 1.0)
        bar_fade_out = np.clip((win_len - lt) / 0.35, 0.0, 1.0)
        bar_env = bar_fade_in * bar_fade_out

        # Sub bass + stereo detuned warm strings
        root, fifth_lo, mid, hi = chord
        sub_voice = 0.50 * np.sin(2.0 * np.pi * root * lt)
        pad_l = (
            0.30 * np.tanh(1.4 * np.sin(2.0 * np.pi * fifth_lo * lt))
            + 0.25 * np.sin(2.0 * np.pi * mid * lt)
            + 0.18 * np.sin(2.0 * np.pi * hi * lt + 0.3)
        )
        pad_r = (
            0.30 * np.tanh(1.4 * np.sin(2.0 * np.pi * (fifth_lo * 1.003) * lt))
            + 0.25 * np.sin(2.0 * np.pi * (mid * 0.997) * lt)
            + 0.18 * np.sin(2.0 * np.pi * (hi * 1.004) * lt + 0.8)
        )

        # Gentle felt-piano harmonic pluck on bar downbeat (especially warm for emotional/inspiring)
        pluck_env = np.exp(-lt / (1.8 if mood in ("emotional", "inspiring") else 0.95))
        pluck = 0.22 * (np.sin(2.0 * np.pi * hi * lt) + 0.35 * np.sin(2.0 * np.pi * (hi * 2.0) * lt) * np.exp(-lt / 0.35)) * pluck_env

        left[i0:i1] += (sub_voice + pad_l + pluck) * bar_env
        right[i0:i1] += (sub_voice + pad_r + pluck * 0.92) * bar_env

    # Optional subtle rhythmic sub pulse for tense / inspiring / mysterious clips
    if has_pulse:
        beat_sec = 60.0 / bpm
        beat_phase = (t % beat_sec) / beat_sec
        kick_env = 0.78 + 0.22 * np.exp(-beat_phase / 0.16)
        left *= kick_env
        right *= kick_env

    # Low-pass filter so background track sits warm underneath vocal presence range
    left = _smooth_lowpass(left, kernel_len=41)
    right = _smooth_lowpass(right, kernel_len=41)

    fade_in = np.clip(t / 0.45, 0.0, 1.0)
    fade_out = np.clip((dur - t) / min(1.3, dur * 0.25), 0.0, 1.0)
    master_env = fade_in * fade_out

    left *= master_env
    right *= master_env

    peak = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-6)
    return _write_stereo_wav(output_path, 0.70 * left / peak, 0.70 * right / peak, sample_rate)
