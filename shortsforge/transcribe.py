"""Stage 2: Fast transcription using YouTube json3 word-level captions or faster-whisper with live progress."""

from __future__ import annotations

import json
import re
import wave
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from shortsforge.config import AppConfig
from shortsforge.models import Sentence, Transcript, WordTimestamp
from shortsforge.state import ProjectWorkspace


def _compute_segment_pitch_proxy(wav_path: Path, start: float, end: float) -> float:
    """Compute a lightweight spectral centroid proxy for speaker-turn change heuristics."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            s_frame = max(0, min(n_frames - 1, int(start * sr)))
            e_frame = max(s_frame + 1, min(n_frames, int(end * sr)))
            wf.setpos(s_frame)
            raw = wf.readframes(min(e_frame - s_frame, sr * 4))
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if samples.size < 256:
            return 0.0
        spectrum = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(samples.size, d=1.0 / sr)
        denom = float(np.sum(spectrum)) + 1e-6
        return float(np.sum(freqs * spectrum) / denom)
    except Exception:  # noqa: BLE001
        return 0.0


def group_words_into_sentences(
    words: List[WordTimestamp],
    wav_path: Path | None = None,
    speaker_pause_sec: float = 1.6,
) -> List[Sentence]:
    """Segment word-level timestamps into sentences and attach heuristic speaker turn labels."""
    if not words:
        return []

    sentences: List[Sentence] = []
    buf: List[WordTimestamp] = []
    speaker_idx = 1
    prev_centroid = 0.0
    # Avoid repeated WAV seeks if there are thousands of sentences (e.g. 3-hour transcript)
    use_pitch_proxy = wav_path is not None and len(words) <= 2500

    def _flush(word_buf: List[WordTimestamp], current_speaker: int) -> Tuple[Sentence, float]:
        text = " ".join(w.word.strip() for w in word_buf if w.word.strip()).strip()
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        s_start = word_buf[0].start
        s_end = word_buf[-1].end
        centroid = _compute_segment_pitch_proxy(wav_path, s_start, s_end) if use_pitch_proxy else 0.0
        sent = Sentence(
            index=len(sentences),
            text=text,
            start=round(s_start, 3),
            end=round(s_end, 3),
            speaker=f"SPEAKER_{current_speaker}",
            words=list(word_buf),
        )
        return sent, centroid

    for i, w in enumerate(words):
        if buf:
            gap = w.start - buf[-1].end
            if gap >= speaker_pause_sec:
                sent, centroid = _flush(buf, speaker_idx)
                sentences.append(sent)
                buf = []
                if (prev_centroid > 0 and centroid > 0 and abs(centroid - prev_centroid) > 280.0) or gap >= speaker_pause_sec * 1.3:
                    speaker_idx = 2 if speaker_idx == 1 else 1
                prev_centroid = centroid

        buf.append(w)
        token = w.word.strip()
        is_terminal = bool(re.search(r"[.!?][\"'\)\]]*$", token))
        next_gap = (words[i + 1].start - w.end) if i + 1 < len(words) else 0.0

        if is_terminal or (next_gap >= 1.45 and len(buf) >= 6) or len(buf) >= 38:
            sent, centroid = _flush(buf, speaker_idx)
            sentences.append(sent)
            if prev_centroid == 0.0 and centroid > 0.0:
                prev_centroid = centroid
            buf = []

    if buf:
        sent, _ = _flush(buf, speaker_idx)
        sentences.append(sent)

    return sentences


def parse_yt_json3_captions(json3_path: Path, speaker_pause_sec: float = 1.6) -> Optional[Transcript]:
    """Parse YouTube's json3 word-level caption file into a full word-timed Transcript in <1 second."""
    if not json3_path.is_file():
        return None
    try:
        data = json.loads(json3_path.read_text(encoding="utf-8", errors="replace"))
        events = data.get("events") or []
        raw_words: List[Tuple[str, float, float]] = []

        for ev in events:
            segs = ev.get("segs")
            if not segs:
                continue
            base_ms = float(ev.get("tStartMs") or 0.0)
            dur_ms = float(ev.get("dDurationMs") or 1500.0)
            for idx, seg in enumerate(segs):
                utf8 = str(seg.get("utf8") or "").replace("\n", " ").strip()
                if not utf8 or utf8.startswith("[") and utf8.endswith("]"):
                    continue
                off_ms = float(seg.get("tOffsetMs") or 0.0)
                w_start = max(0.0, (base_ms + off_ms) / 1000.0)
                if idx + 1 < len(segs) and "tOffsetMs" in segs[idx + 1]:
                    next_off = float(segs[idx + 1]["tOffsetMs"])
                    w_end = max(w_start + 0.08, (base_ms + next_off) / 1000.0)
                else:
                    w_end = max(w_start + 0.16, min(w_start + 0.45, (base_ms + dur_ms) / 1000.0))

                # Split multi-word segment tokens if any
                sub_tokens = [t for t in utf8.split() if t]
                if len(sub_tokens) == 1:
                    raw_words.append((sub_tokens[0], w_start, w_end))
                elif len(sub_tokens) > 1:
                    step = max(0.06, (w_end - w_start) / len(sub_tokens))
                    for s_i, tok in enumerate(sub_tokens):
                        raw_words.append((tok, w_start + s_i * step, w_start + (s_i + 1) * step))

        if len(raw_words) < 15:
            return None

        # Ensure strictly monotonic word end->start boundaries and add punctuation on natural pauses
        all_words: List[WordTimestamp] = []
        for i, (tok, st, en) in enumerate(raw_words):
            if i + 1 < len(raw_words):
                next_st = raw_words[i + 1][1]
                if en > next_st and next_st > st + 0.04:
                    en = next_st
                gap = next_st - en
                if gap >= 1.35 and not re.search(r"[.!?,;:]$", tok):
                    tok = tok + "."
            all_words.append(
                WordTimestamp(
                    word=tok,
                    start=round(st, 3),
                    end=round(max(st + 0.05, en), 3),
                    probability=0.95,
                )
            )

        sentences = group_words_into_sentences(all_words, wav_path=None, speaker_pause_sec=speaker_pause_sec)
        duration = all_words[-1].end if all_words else 0.0
        return Transcript(
            language="en",
            duration=round(duration, 3),
            sentences=sentences,
            words=all_words,
        )
    except Exception:  # noqa: BLE001
        return None


def transcribe_audio(
    audio_path: Path,
    workspace: ProjectWorkspace,
    config: AppConfig,
    force: bool = False,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Transcript:
    """Produce a word-level Transcript via cached YouTube json3 captions (1s) or faster-whisper with live progress."""
    if not force and workspace.transcript_path.exists():
        return Transcript.model_validate_json(workspace.transcript_path.read_text(encoding="utf-8"))

    # 1. Instant path: use YouTube's word-timed json3 caption track if downloaded during Stage 1
    yt_json3 = workspace.source_dir / "yt_captions.json3"
    if yt_json3.exists():
        if on_progress:
            on_progress("Parsing YouTube word-level captions...")
        parsed = parse_yt_json3_captions(yt_json3, speaker_pause_sec=config.transcription.speaker_turn_pause_sec)
        if parsed and len(parsed.sentences) >= 2:
            workspace.transcript_path.write_text(parsed.model_dump_json(indent=2), encoding="utf-8")
            return parsed

    # 2. Acoustic path: faster-whisper on the 16kHz PCM WAV
    from faster_whisper import WhisperModel

    device = config.transcription.device
    compute_type = config.transcription.compute_type
    if device == "auto":
        device = "cpu"
    if compute_type == "auto":
        compute_type = "int8" if device == "cpu" else "float16"

    with wave.open(str(audio_path), "rb") as wf:
        sr = wf.getframerate()
        total_frames = wf.getnframes()
        total_dur = total_frames / max(1, sr)
        raw_pcm = wf.readframes(total_frames)
    audio_np = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0

    # For long videos (>15 min) on CPU without YouTube captions, use fast 'tiny' model on highlight windows
    model_size = "tiny" if (device == "cpu" and total_dur > 900.0) else config.transcription.model_size
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    all_words: List[WordTimestamp] = []

    if device == "cpu" and total_dur > 1200.0:
        # Sample 10 evenly spaced 60-second windows across a multi-hour local video so CPU transcription finishes in ~30s
        n_win = 10
        win_dur = 60.0
        step = max(win_dur, (total_dur - win_dur) / max(1, n_win - 1))
        for w_i in range(n_win):
            offset_sec = round(w_i * step, 2)
            s_sample = int(offset_sec * sr)
            e_sample = min(audio_np.size, int((offset_sec + win_dur) * sr))
            if e_sample - s_sample < sr * 5:
                continue
            if on_progress:
                on_progress(f"[2/4] Fast-scanning audio window {w_i + 1}/{n_win} ({int((w_i + 1) * 100 / n_win)}%)...")
            segs, _ = model.transcribe(
                audio_np[s_sample:e_sample],
                language=config.transcription.language,
                word_timestamps=True,
                vad_filter=config.transcription.vad_filter,
            )
            for seg in segs:
                if seg.words:
                    for w in seg.words:
                        cleaned = (w.word or "").strip()
                        if cleaned:
                            st = round(offset_sec + float(w.start), 3)
                            en = round(max(st + 0.04, offset_sec + float(w.end)), 3)
                            all_words.append(WordTimestamp(word=cleaned, start=st, end=en))
        language = config.transcription.language or "en"
    else:
        segments, info = model.transcribe(
            audio_np,
            language=config.transcription.language,
            word_timestamps=True,
            vad_filter=config.transcription.vad_filter,
        )
        language = str(getattr(info, "language", "en") or "en")
        last_pct = -10
        for seg in segments:
            if total_dur > 0 and on_progress:
                pct = min(99, int(round(float(seg.end) * 100.0 / total_dur)))
                if pct >= last_pct + 10:
                    last_pct = pct
                    on_progress(f"[2/4] Transcribing audio: {pct}% ({float(seg.end):.0f}s / {total_dur:.0f}s)...")
            if seg.words:
                for w in seg.words:
                    cleaned = (w.word or "").strip()
                    if not cleaned:
                        continue
                    all_words.append(
                        WordTimestamp(
                            word=cleaned,
                            start=round(float(w.start), 3),
                            end=round(max(float(w.end), float(w.start) + 0.04), 3),
                            probability=round(float(getattr(w, "probability", 1.0)), 3),
                        )
                    )
            else:
                tokens = seg.text.strip().split()
                if tokens:
                    dur = max(0.1, float(seg.end) - float(seg.start))
                    step_t = dur / len(tokens)
                    for idx, tok in enumerate(tokens):
                        all_words.append(
                            WordTimestamp(
                                word=tok,
                                start=round(float(seg.start) + idx * step_t, 3),
                                end=round(float(seg.start) + (idx + 1) * step_t, 3),
                                probability=0.9,
                            )
                        )

    sentences = group_words_into_sentences(
        all_words,
        wav_path=audio_path,
        speaker_pause_sec=config.transcription.speaker_turn_pause_sec,
    )
    transcript = Transcript(
        language=language,
        duration=round(total_dur, 3),
        sentences=sentences,
        words=all_words,
    )
    workspace.transcript_path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    return transcript
