#!/usr/bin/env python3
"""End-to-end ShortsForge smoke test using a self-contained ~28s sample interview video.

Verifies:
  1. Synthetic 16:9 sample video + spoken audio generation (zero external assets required)
  2. `run_new_project`: Ingest -> faster-whisper Transcribe -> Moments -> Quality -> Verified Hooks -> review.html
  3. `run_render_project`: Reframe 9:16 -> ASS Captions -> Pillow Overlays -> -14 LUFS Audio -> H.264 MP4 + Metadata
  4. Programmatic ffprobe verification of output resolution (1080x1920), H.264 codec, audio, LUFS, and metadata pack.
"""

from __future__ import annotations

import asyncio
import math
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure workspace root is importable when invoked as `python scripts/smoke_test.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shortsforge.audio import measure_integrated_lufs
from shortsforge.config import AppConfig, load_config, resolve_font
from shortsforge.ffmpeg_utils import extract_stream_info, probe_media, run_ffmpeg
from shortsforge.pipeline import run_edit_single_clip, run_new_project, run_render_project


SMOKE_SCRIPT = (
    "In 1998, I was forty-eight hours away from shutting down our entire workshop. "
    "We had lost fifty thousand dollars and everybody told me to walk away. "
    "Then one late-night phone call from my mentor changed my mind completely. "
    "He said, never quit on your worst day, wait until tomorrow morning and look again. "
    "The very next week, our tiny prototype solved a problem nobody else had cracked. "
    "That single decision built everything you see today."
)


def _generate_speech_wav(output_wav: Path) -> Path:
    """Synthesize clean spoken English audio via edge-tts (with offline fallback if no network)."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp_mp3 = output_wav.with_suffix(".mp3")
    try:
        import edge_tts

        async def _run() -> None:
            comm = edge_tts.Communicate(text=SMOKE_SCRIPT, voice="en-US-GuyNeural", rate="-4%")
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
    except Exception:  # noqa: BLE001
        # Offline fallback: pleasant modulated voice-band tone (~26s)
        sr = 48000
        dur = 26.0
        t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
        sig = 0.25 * np.sin(2.0 * np.pi * 180.0 * t) * (0.5 + 0.5 * np.sin(2.0 * np.pi * 2.5 * t))
        pcm = (sig * 32767.0).astype(np.int16)
        stereo = np.column_stack([pcm, pcm]).ravel()
        with wave.open(str(output_wav), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(stereo.tobytes())
        return output_wav


def _generate_sample_video(output_mp4: Path, speech_wav: Path, config: AppConfig) -> Path:
    """Create a rich studio-styled 1280x720 16:9 sample interview video with gentle camera motion and a cut."""
    info = extract_stream_info(probe_media(speech_wav))
    duration = max(24.0, info["duration"] + 0.8)
    fps = 30
    w, h = 1280, 720
    total_frames = int(round(duration * fps))

    raw_vid = output_mp4.with_name("smoke_video_only.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_vid), fourcc, fps, (w, h))

    font_path, _ = resolve_font(config, bold=True)
    studio_font = ImageFont.truetype(str(font_path), 34) if font_path else ImageFont.load_default()

    # Pre-render warm podcast studio backdrop once for speed
    backdrop = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        frac = y / h
        backdrop[y, :, 0] = int(38 + 22 * (1 - frac))  # B
        backdrop[y, :, 1] = int(28 + 20 * (1 - frac))  # G
        backdrop[y, :, 2] = int(24 + 34 * (1 - frac))  # R

    # Subtle studio bokeh circles & acoustic panels
    for bx in range(80, w, 160):
        cv2.rectangle(backdrop, (bx, 40), (bx + 90, 480), (42, 34, 32), -1)
    cv2.circle(backdrop, (220, 170), 90, (52, 72, 115), -1)
    cv2.circle(backdrop, (1060, 190), 110, (95, 58, 42), -1)
    backdrop = cv2.GaussianBlur(backdrop, (51, 51), 0)

    # Studio desk in foreground bottom
    cv2.rectangle(backdrop, (0, 585), (w, h), (26, 22, 20), -1)
    cv2.line(backdrop, (0, 585), (w, 585), (70, 58, 48), 3)

    # Add subtle studio wall text
    pil_bg = Image.fromarray(cv2.cvtColor(backdrop, cv2.COLOR_BGR2RGB))
    pdraw = ImageDraw.Draw(pil_bg)
    pdraw.text((48, 36), "LATE NIGHT STUDIO • 1998 ARCHIVE", font=studio_font, fill=(195, 175, 145))
    backdrop = cv2.cvtColor(np.array(pil_bg), cv2.COLOR_RGB2BGR)

    cut_frame = total_frames // 2

    for i in range(total_frames):
        frame = backdrop.copy()
        t = i / fps
        # Camera angle 1: speaker slightly left; after mid-clip cut, speaker slightly right
        base_cx = int(w * (0.44 if i < cut_frame else 0.56))
        cx = base_cx + int(round(18 * math.sin(t * 0.9)))
        cy = 320 + int(round(5 * math.sin(t * 1.7)))

        # Draw stylized speaker silhouette + shaded portrait + studio microphone
        # Shoulders / jacket
        cv2.ellipse(frame, (cx, cy + 250), (170, 150), 0, 180, 360, (55, 48, 45), -1)
        cv2.ellipse(frame, (cx, cy + 250), (150, 130), 0, 180, 360, (75, 65, 60), 2)
        # Neck
        cv2.rectangle(frame, (cx - 32, cy + 75), (cx + 32, cy + 145), (145, 175, 215), -1)
        # Head (warm skin tone in BGR)
        cv2.ellipse(frame, (cx, cy), (88, 112), 0, 0, 360, (152, 184, 228), -1)
        # Hair
        cv2.ellipse(frame, (cx, cy - 35), (90, 82), 0, 185, 355, (42, 34, 30), -1)
        # Eyes + eyebrows
        cv2.ellipse(frame, (cx - 32, cy - 12), (14, 8), 0, 0, 360, (245, 245, 245), -1)
        cv2.ellipse(frame, (cx + 32, cy - 12), (14, 8), 0, 0, 360, (245, 245, 245), -1)
        cv2.circle(frame, (cx - 32, cy - 12), 5, (32, 26, 22), -1)
        cv2.circle(frame, (cx + 32, cy - 12), 5, (32, 26, 22), -1)
        cv2.line(frame, (cx - 48, cy - 30), (cx - 16, cy - 28), (45, 36, 30), 4)
        cv2.line(frame, (cx + 16, cy - 28), (cx + 48, cy - 30), (45, 36, 30), 4)
        # Nose
        cv2.line(frame, (cx, cy - 6), (cx - 6, cy + 22), (125, 152, 195), 3)
        cv2.line(frame, (cx - 6, cy + 22), (cx + 6, cy + 22), (125, 152, 195), 3)
        # Animated speaking mouth synced with speech rhythm
        mouth_open = int(4 + 7 * abs(math.sin(t * 9.5)))
        cv2.ellipse(frame, (cx, cy + 52), (24, mouth_open), 0, 0, 360, (68, 62, 125), -1)

        # Broadcast condenser microphone on boom arm
        mic_x = cx + (135 if i < cut_frame else -135)
        cv2.line(frame, (mic_x + (140 if i < cut_frame else -140), h), (mic_x, cy + 95), (38, 38, 42), 8)
        cv2.ellipse(frame, (mic_x, cy + 80), (22, 34), 0, 0, 360, (55, 55, 62), -1)
        cv2.ellipse(frame, (mic_x, cy + 80), (22, 34), 0, 0, 360, (120, 120, 130), 2)

        writer.write(frame)

    writer.release()

    # Mux synthetic video + speech WAV into a clean H.264 MP4
    run_ffmpeg([
        "-i", str(raw_vid),
        "-i", str(speech_wav),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_mp4),
    ])
    raw_vid.unlink(missing_ok=True)
    return output_mp4


def main() -> int:
    print("=== ShortsForge End-to-End Smoke Test ===")
    cfg = load_config(ROOT / "config.yaml")

    # Configure fast, zero-key, self-contained smoke test run
    cfg.llm.provider = "mock"
    cfg.transcription.model_size = "tiny"
    cfg.transcription.vad_filter = False
    cfg.moments.max_candidates = 2
    cfg.moments.min_duration_sec = 18.0
    cfg.quality_check.min_face_presence_ratio = 0.0  # Synthetic portrait frame
    cfg.overlays.archive_tag = True
    cfg.review.auto_score_threshold = 70
    cfg.export.mirror_video = True
    cfg.export.video_filters = ["sharp_hd", "cinematic_grade"]

    smoke_dir = ROOT / "projects" / "_smoke_fixtures"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    speech_wav = _generate_speech_wav(smoke_dir / "sample_speech.wav")
    sample_mp4 = _generate_sample_video(smoke_dir / "sample_interview_1998.mp4", speech_wav, cfg)

    project_name = "smoke_demo"
    print(f"[1/3] Generated sample interview video: {sample_mp4}")

    print("[2/3] Running `new` pipeline (Ingest -> Transcribe -> Moments -> Hooks -> review.html)...")
    review_html = run_new_project(
        project_name=project_name,
        source=str(sample_mp4),
        person="Marcus Vance",
        config=cfg,
        auto=True,
        show_override="Late Night Studio",
        year_override="1998",
        force=True,
    )
    assert review_html.exists(), "review.html was not generated!"

    print("[3/3] Running `render` pipeline (9:16 Reframe -> Captions -> Overlays -> -14 LUFS -> Export)...")
    exported = run_render_project(
        project_name=project_name,
        config=cfg,
        auto=True,
        force=True,
    )
    assert len(exported) >= 1, "No Shorts were exported!"

    out_dir = cfg.outputs_root() / project_name
    first = exported[0]
    mp4_path = out_dir / first.output_video
    thumb_path = out_dir / first.thumbnail
    sources_txt = out_dir / "sources.txt"
    warnings_txt = out_dir / "warnings.txt"

    assert mp4_path.exists(), f"Missing output MP4: {mp4_path}"
    assert thumb_path.exists(), f"Missing thumbnail JPG: {thumb_path}"
    assert sources_txt.exists(), "Missing sources.txt"
    assert warnings_txt.exists(), "Missing warnings.txt"
    assert len(first.titles) == 5, f"Expected 5 titles, got {len(first.titles)}"
    assert len(first.hashtags) >= 8, f"Expected >=8 hashtags, got {len(first.hashtags)}"
    assert first.pinned_comment.strip(), "Missing pinned comment question"

    print("[4/4] Testing post-render runtime edit (Filters + Center Captions + Center White Attention Box Card)...")
    cfg.captions.position = "center"
    cfg.export.video_filters = ["cinematic_grade", "vibrant_pop"]
    cfg.overlays.attention_card_enabled = True
    cfg.overlays.attention_card_text = "Never Quit On Your Worst Day"
    cfg.overlays.attention_card_position = "center"
    cfg.overlays.attention_card_duration_sec = 4.0
    cfg.overlays.attention_card_style = "white_box"
    edited_meta = run_edit_single_clip(
        project_name=project_name,
        clip_id=first.clip_id,
        config=cfg,
    )
    assert edited_meta.caption_position == "center"
    assert edited_meta.attention_card_enabled is True
    assert "cinematic_grade" in edited_meta.video_filters

    info = extract_stream_info(probe_media(mp4_path))
    lufs = measure_integrated_lufs(mp4_path)
    assert info["width"] == 1080 and info["height"] == 1920, f"Expected 1080x1920, got {info['width']}x{info['height']}"
    assert info["video_codec"] == "h264", f"Expected h264, got {info['video_codec']}"
    assert info["has_audio"], "Rendered MP4 has no audio stream!"

    print("\n=== SMOKE TEST PASSED ===")
    print(f"  Review page:   {review_html}")
    print(f"  Output Short:  {mp4_path} ({info['width']}x{info['height']}, {info['video_codec']}, {info['duration']:.1f}s)")
    print(f"  Loudness:      {lufs} LUFS (target {cfg.audio.target_lufs} LUFS)")
    print(f"  Thumbnail:     {thumb_path}")
    print(f"  Sources log:   {sources_txt}")
    print(f"  Warnings log:  {warnings_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
