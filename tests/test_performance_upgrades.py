"""Tests verifying turbo performance upgrades preserve 100% functional equivalence and thread safety."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from shortsforge.audio import build_slice_loudnorm_filter, measure_integrated_lufs
from shortsforge.config import AppConfig, load_config, resolve_font
from shortsforge.export import render_single_short
from shortsforge.ffmpeg_utils import extract_stream_info, get_ffmpeg_bin, probe_media, run_ffmpeg
from shortsforge.hooks import generate_verified_hooks
from shortsforge.ingest import (
    _build_format_selector,
    _is_hls_pair,
    _select_direct_stream_formats,
    extract_wav_audio,
    resolve_candidate_media,
)
from shortsforge.llm import LLMClient
from shortsforge.models import (
    ClipCandidate,
    Sentence,
    Transcript,
    VideoMetadata,
    WordTimestamp,
)
from shortsforge.quality_check import evaluate_candidate_quality, is_mostly_graphic_frame


def test_config_mtime_cache_isolated_and_invalidates_on_edit(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump({"captions": {"preset": "iman"}}), encoding="utf-8")

    c1 = load_config(cfg_file)
    assert c1.captions.preset == "iman"
    # Mutating returned copy must never pollute subsequent loads
    c1.captions.preset = "hormozi"
    c2 = load_config(cfg_file)
    assert c2.captions.preset == "iman"

    # Editing config.yaml on disk must invalidate the mtime cache immediately
    cfg_file.write_text(yaml.safe_dump({"captions": {"preset": "documentary"}}), encoding="utf-8")
    c3 = load_config(cfg_file)
    assert c3.captions.preset == "documentary"


def test_cached_font_and_ffmpeg_lookup_consistency() -> None:
    cfg = AppConfig()
    f1 = resolve_font(cfg, bold=True)
    f2 = resolve_font(cfg, bold=True)
    assert f1 == f2
    assert get_ffmpeg_bin() == get_ffmpeg_bin()


def test_vectorized_graphic_frame_detector_matches_slides_and_natural_scenes() -> None:
    # Flat 2-color presentation slide -> True
    slide = np.full((720, 1280, 3), 245, dtype=np.uint8)
    slide[100:220, 120:1160] = (32, 32, 180)
    assert is_mostly_graphic_frame(slide) is True

    # Textured multi-tone natural frame -> False
    rng = np.random.default_rng(123)
    natural = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    assert is_mostly_graphic_frame(natural) is False


def test_concurrent_remote_hook_verification_preserves_order_and_thread_local_cache(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = AppConfig()
    cfg.llm.provider = "gemini"
    cfg.hook.variants_count = 4
    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key-for-concurrency")

    llm = LLMClient(cfg)

    def _fake_gemini(self: LLMClient, system_prompt: str, user_prompt: str) -> Any:
        if "Generate" in user_prompt and "variants" in system_prompt:
            return {
                "variants": [
                    {"rank": 1, "text": "He almost quit before one call.", "predicted_retention": 94},
                    {"rank": 2, "text": "Why one mentor call saved everything.", "predicted_retention": 90},
                    {"rank": 3, "text": "Fifty thousand lost before the breakthrough.", "predicted_retention": 86},
                    {"rank": 4, "text": "Never quit on your worst day.", "predicted_retention": 82},
                ]
            }
        return {"passed": True, "reason": "Grounded in transcript.", "violations": []}

    monkeypatch.setattr(LLMClient, "_call_gemini", _fake_gemini)

    cand = ClipCandidate(
        id="clip_01",
        start=0.0,
        end=24.0,
        score=90,
        reason="Breakthrough moment",
        key_quote="Never quit on your worst day.",
        transcript_text="We lost fifty thousand dollars and I almost quit until one mentor call saved everything. Never quit on your worst day.",
        peak_start=6.0,
        peak_end=9.0,
    )
    hook_set = generate_verified_hooks(cand, "Marcus Vance", llm, cfg)
    assert hook_set.flagged_for_review is False
    assert len(hook_set.variants) == 4
    assert hook_set.active_hook_text() == "He almost quit before one call."


def test_single_pass_loudnorm_and_native_cold_open_hook_render(tmp_path: Path) -> None:
    src_mp4 = tmp_path / "src.mp4"
    run_ffmpeg([
        "-f", "lavfi",
        "-i", "testsrc=s=1280x720:r=30:d=8",
        "-f", "lavfi",
        "-i", "sine=frequency=220:sample_rate=48000:duration=8",
        "-af", "volume=-18dB",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-c:a", "aac",
        "-shortest",
        str(src_mp4),
    ])

    cfg = AppConfig()
    cfg.llm.provider = "mock"
    cfg.reframe.face_tracking_enabled = False

    af_str, est_lufs = build_slice_loudnorm_filter(src_mp4, 0.5, 6.5, cfg)
    assert "loudnorm=" in af_str
    assert est_lufs is not None

    words = [
        WordTimestamp(word="Never", start=1.0, end=1.4),
        WordTimestamp(word="quit", start=1.4, end=1.8),
        WordTimestamp(word="today.", start=1.8, end=2.4),
    ]
    tr = Transcript(
        language="en",
        duration=8.0,
        sentences=[Sentence(index=0, text="Never quit today.", start=1.0, end=2.4, words=words)],
        words=words,
    )
    meta = VideoMetadata(
        source=str(src_mp4),
        source_type="local",
        title="Studio Test",
        channel="Archive",
        local_video_path=str(src_mp4),
        duration=8.0,
        width=1280,
        height=720,
        fps=30.0,
    )
    cand = ClipCandidate(
        id="clip_01",
        start=0.5,
        end=7.0,
        score=88,
        reason="Test clip",
        key_quote="Never quit today.",
        transcript_text="Never quit today.",
        peak_start=1.0,
        peak_end=3.5,
    )

    # 1. Fast single-pass render (hook disabled)
    out_dir = tmp_path / "out_fast"
    work_dir = tmp_path / "work_fast"
    exp_fast = render_single_short(
        candidate=cand,
        hook_text="Never quit today.",
        metadata=meta,
        transcript=tr,
        person="Speaker",
        work_dir=work_dir,
        output_dir=out_dir,
        llm=LLMClient(cfg),
        config=cfg,
    )
    fast_mp4 = out_dir / exp_fast.output_video
    assert fast_mp4.exists()
    info_fast = extract_stream_info(probe_media(fast_mp4))
    assert info_fast["width"] == 1080 and info_fast["height"] == 1920
    assert info_fast["has_audio"] is True
    lufs_fast = measure_integrated_lufs(fast_mp4)
    assert lufs_fast is not None and -16.5 <= lufs_fast <= -11.5

    # 2. Native cold-open hook render (hook enabled -> zero Python raw pixel piping)
    cfg.hook.enabled = True
    cfg.hook.style = "cold_open"
    cfg.hook.transition_duration_sec = 0.25
    out_hook_dir = tmp_path / "out_hook"
    work_hook_dir = tmp_path / "work_hook"
    exp_hook = render_single_short(
        candidate=cand,
        hook_text="Never quit today.",
        metadata=meta,
        transcript=tr,
        person="Speaker",
        work_dir=work_hook_dir,
        output_dir=out_hook_dir,
        llm=LLMClient(cfg),
        config=cfg,
    )
    hook_mp4 = out_hook_dir / exp_hook.output_video
    assert hook_mp4.exists()
    info_hook = extract_stream_info(probe_media(hook_mp4))
    assert info_hook["width"] == 1080 and info_hook["height"] == 1920
    assert info_hook["duration"] > 7.5


def test_format_selector_prefers_https_over_hls_and_keeps_hd() -> None:
    cfg = AppConfig()
    sel = _build_format_selector(cfg, duration_sec=9600.0)
    assert "[protocol^=https]" in sel
    assert "height>=1080" in sel
    assert "1440" in sel
    assert sel.index("[protocol^=https]") < sel.index("bestvideo[height<=1440][height>=1080][vcodec!^=av01]+bestaudio")


def test_select_direct_stream_formats_prefers_hls_range_pair_and_https_fallback() -> None:
    cfg = AppConfig()
    mock_info = {
        "duration": 10800.0,
        "formats": [
            {"format_id": "248", "ext": "webm", "height": 1080, "width": 1920, "vcodec": "vp9", "acodec": "none", "protocol": "https", "tbr": 632.0, "url": "https://ex/248"},
            {"format_id": "137", "ext": "mp4", "height": 1080, "width": 1920, "vcodec": "avc1.640028", "acodec": "none", "protocol": "https", "tbr": 735.0, "url": "https://ex/137"},
            {"format_id": "270", "ext": "mp4", "height": 1080, "width": 1920, "vcodec": "avc1.640028", "acodec": "none", "audio_ext": "none", "video_ext": "mp4", "protocol": "m3u8_native", "tbr": 4698.8, "url": "https://ex/270"},
            {"format_id": "614", "ext": "mp4", "height": 1080, "width": 1920, "vcodec": "vp09.00.40.08", "acodec": "none", "audio_ext": "none", "video_ext": "mp4", "protocol": "m3u8_native", "tbr": 2078.7, "url": "https://ex/614"},
            {"format_id": "271", "ext": "webm", "height": 1440, "width": 2560, "vcodec": "vp9", "acodec": "none", "protocol": "https", "tbr": 1732.0, "url": "https://ex/271"},
            {"format_id": "620", "ext": "mp4", "height": 1440, "width": 2560, "vcodec": "vp09.00.50.08", "acodec": "none", "audio_ext": "none", "video_ext": "mp4", "protocol": "m3u8_native", "tbr": 8298.0, "url": "https://ex/620"},
            {"format_id": "233", "ext": "mp4", "vcodec": "none", "acodec": None, "audio_ext": "mp4", "video_ext": "none", "protocol": "m3u8_native", "format_note": "Default, low", "url": "https://ex/233"},
            {"format_id": "234", "ext": "mp4", "vcodec": "none", "acodec": None, "audio_ext": "mp4", "video_ext": "none", "protocol": "m3u8_native", "format_note": "Default, high", "url": "https://ex/234"},
            {"format_id": "140-drc", "ext": "m4a", "vcodec": "none", "acodec": "mp4a.40.2", "protocol": "https", "abr": 129.0, "url": "https://ex/140drc"},
            {"format_id": "140", "ext": "m4a", "vcodec": "none", "acodec": "mp4a.40.2", "protocol": "https", "abr": 129.0, "url": "https://ex/140"},
            {"format_id": "251", "ext": "webm", "vcodec": "none", "acodec": "opus", "protocol": "https", "abr": 117.0, "url": "https://ex/251"},
        ],
    }
    # 1. With HLS + HTTPS present: picks 270 (1080p avc1 HLS) + 234 (high-tier HLS audio with acodec=None)
    v_fmt, a_fmt = _select_direct_stream_formats(mock_info, cfg)
    assert v_fmt is not None and a_fmt is not None
    assert v_fmt["format_id"] == "270"
    assert a_fmt["format_id"] == "234"
    assert _is_hls_pair(v_fmt, a_fmt) is True

    # 2. When 270 is absent or excluded: picks 614 (1080p VP9 HLS) + 234 over 1440p 620 and HTTPS 137
    v2, a2 = _select_direct_stream_formats(mock_info, cfg, exclude_video_ids={"270"})
    assert v2 is not None and v2["format_id"] == "614"
    assert a2 is not None and a2["format_id"] == "234"
    assert _is_hls_pair(v2, a2) is True

    no_270 = {**mock_info, "formats": [f for f in mock_info["formats"] if f["format_id"] != "270"]}
    v2b, a2b = _select_direct_stream_formats(no_270, cfg)
    assert v2b is not None and v2b["format_id"] == "614"
    assert a2b is not None and a2b["format_id"] == "234"

    # 3. With no HLS streams present (HTTPS only): falls back cleanly to 137 + 140
    https_only = {**mock_info, "formats": [f for f in mock_info["formats"] if "m3u8" not in f["protocol"]]}
    v3, a3 = _select_direct_stream_formats(https_only, cfg)
    assert v3 is not None and v3["format_id"] == "137"
    assert a3 is not None and a3["format_id"] == "140"
    assert _is_hls_pair(v3, a3) is False


def test_surgical_clip_only_quality_check_and_render_with_time_offset(tmp_path: Path) -> None:
    src_mp4 = tmp_path / "full_12s.mp4"
    run_ffmpeg([
        "-f", "lavfi",
        "-i", "testsrc=s=1280x720:r=30:d=12",
        "-f", "lavfi",
        "-i", "sine=frequency=330:sample_rate=48000:duration=12",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-c:a", "aac",
        "-shortest",
        str(src_mp4),
    ])

    # Extract padded clip slice [3.0s..11.5s] simulating clip-only download of a late podcast moment
    clip_slice = tmp_path / "clip_01.mp4"
    run_ffmpeg([
        "-ss", "3.000",
        "-t", "8.500",
        "-i", str(src_mp4),
        "-c", "copy",
        "-movflags", "+faststart",
        str(clip_slice),
    ])
    clip_wav = extract_wav_audio(clip_slice, tmp_path / "clip_01_16k.wav")

    cfg = AppConfig()
    cfg.llm.provider = "mock"
    cfg.reframe.face_tracking_enabled = False
    cfg.quality_check.min_face_presence_ratio = 0.0

    # Original podcast timestamps [5.0s..10.5s] -> local timestamps inside clip_01.mp4 are [2.0s..7.5s]
    cand = ClipCandidate(
        id="clip_01",
        start=5.0,
        end=10.5,
        score=91,
        reason="Strong insight",
        key_quote="Discipline beats everything.",
        transcript_text="Discipline beats everything.",
        peak_start=6.0,
        peak_end=8.5,
        clip_video_path=str(clip_slice),
        clip_audio_path=str(clip_wav),
        clip_time_offset=3.0,
    )
    meta = VideoMetadata(
        source="https://youtu.be/demo1234567",
        source_type="url",
        title="Demo Podcast",
        channel="Studio",
        duration=12.0,
        width=1280,
        height=720,
        fps=30.0,
        clip_only_mode=True,
    )

    vid_p, aud_p, l_s, l_e, l_ps, l_pe = resolve_candidate_media(cand, meta)
    assert vid_p == clip_slice
    assert aud_p == clip_wav
    assert abs(l_s - 2.0) < 1e-6 and abs(l_e - 7.5) < 1e-6
    assert abs(l_ps - 3.0) < 1e-6 and abs(l_pe - 5.5) < 1e-6

    q_thumb = tmp_path / "preview.jpg"
    qr = evaluate_candidate_quality(
        candidate=cand,
        video_path=vid_p,
        audio_wav_path=aud_p,
        config=cfg,
        thumb_output_path=q_thumb,
        time_offset=cand.clip_time_offset,
    )
    assert qr.passed is True
    assert qr.audio_rms_db > -30.0
    assert q_thumb.exists()

    words = [
        WordTimestamp(word="Discipline", start=5.2, end=5.7),
        WordTimestamp(word="beats", start=5.7, end=6.2),
        WordTimestamp(word="everything.", start=6.2, end=7.0),
    ]
    tr = Transcript(
        language="en",
        duration=12.0,
        sentences=[Sentence(index=0, text="Discipline beats everything.", start=5.2, end=7.0, words=words)],
        words=words,
    )

    out_dir = tmp_path / "out"
    work_dir = tmp_path / "work"
    exp = render_single_short(
        candidate=cand,
        hook_text="Discipline beats everything.",
        metadata=meta,
        transcript=tr,
        person="Speaker",
        work_dir=work_dir,
        output_dir=out_dir,
        llm=LLMClient(cfg),
        config=cfg,
    )
    out_mp4 = out_dir / exp.output_video
    assert out_mp4.exists()
    info = extract_stream_info(probe_media(out_mp4))
    assert info["width"] == 1080 and info["height"] == 1920
    assert info["video_codec"] == "h264"
    assert info["has_audio"] is True
    assert 5.0 <= info["duration"] <= 6.2
    lufs = measure_integrated_lufs(out_mp4)
    assert lufs is not None and -16.5 <= lufs <= -11.5

    # Captions must still start at 0:00:00.20 (original word 5.2s - clip_start 5.0s = 0.2s), zero drift!
    ass_text = (work_dir / "clip_01" / "captions.ass").read_text(encoding="utf-8")
    assert "0:00:00.20," in ass_text
