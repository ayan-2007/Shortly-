"""Generic narrator text-to-speech provider for the optional freeze_narrator hook style."""

from __future__ import annotations

import asyncio
import wave
from pathlib import Path

import numpy as np

from shortsforge.config import AppConfig, get_env_key
from shortsforge.ffmpeg_utils import run_ffmpeg


class TTSError(RuntimeError):
    """Raised when narrator TTS generation fails."""


def synthesize_narrator(
    text: str,
    output_wav: Path,
    config: AppConfig,
) -> Path:
    """Generate generic narrator audio for hook text, saving a 48kHz stereo WAV."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    provider = config.tts.provider

    if provider == "edge":
        try:
            return _synthesize_edge_tts(text, output_wav, config)
        except Exception:  # noqa: BLE001
            return _synthesize_offline_fallback(text, output_wav)

    if provider == "openai":
        try:
            return _synthesize_openai_tts(text, output_wav, config)
        except Exception:  # noqa: BLE001
            return _synthesize_offline_fallback(text, output_wav)

    return _synthesize_offline_fallback(text, output_wav)


def _synthesize_edge_tts(text: str, output_wav: Path, config: AppConfig) -> Path:
    import edge_tts

    tmp_mp3 = output_wav.with_suffix(".mp3")

    async def _run() -> None:
        comm = edge_tts.Communicate(
            text=text,
            voice=config.tts.edge_voice,
            rate=config.tts.rate,
        )
        await comm.save(str(tmp_mp3))

    asyncio.run(_run())
    run_ffmpeg([
        "-i", str(tmp_mp3),
        "-ar", "48000",
        "-ac", "2",
        "-c:a", "pcm_s16le",
        str(output_wav),
    ])
    tmp_mp3.unlink(missing_ok=True)
    return output_wav


def _synthesize_openai_tts(text: str, output_wav: Path, config: AppConfig) -> Path:
    api_key = get_env_key("OPENAI_API_KEY")
    if not api_key:
        raise TTSError("OPENAI_API_KEY is not set in .env for tts.provider='openai'.")

    import json
    import urllib.request

    tmp_mp3 = output_wav.with_suffix(".mp3")
    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/speech",
        data=json.dumps({
            "model": "tts-1",
            "voice": config.tts.openai_voice,
            "input": text,
        }).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        tmp_mp3.write_bytes(resp.read())

    run_ffmpeg([
        "-i", str(tmp_mp3),
        "-ar", "48000",
        "-ac", "2",
        "-c:a", "pcm_s16le",
        str(output_wav),
    ])
    tmp_mp3.unlink(missing_ok=True)
    return output_wav


def _synthesize_offline_fallback(text: str, output_wav: Path, sample_rate: int = 48000) -> Path:
    """Create a subtle rhythmic narrator cadence cue when running fully offline."""
    word_count = max(1, len(text.split()))
    duration = max(1.8, min(3.4, word_count * 0.34))
    n_samples = int(duration * sample_rate)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)

    env = np.sin(np.pi * t / duration) ** 0.7
    syllables = 0.5 + 0.5 * np.sin(2.0 * np.pi * (word_count / duration) * t)
    signal = 0.08 * env * syllables * (
        0.6 * np.sin(2.0 * np.pi * 165.0 * t)
        + 0.3 * np.sin(2.0 * np.pi * 330.0 * t)
        + 0.1 * np.sin(2.0 * np.pi * 495.0 * t)
    )
    pcm = np.clip(signal * 32767.0, -32767, 32767).astype(np.int16)
    stereo = np.column_stack([pcm, pcm]).ravel()

    with wave.open(str(output_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())
    return output_wav
