"""Tests for Batch Parts splitting, 16:9 widescreen export, playback speed sync, and Editing Window API."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path

import yaml

from shortsforge.audio import measure_integrated_lufs
from shortsforge.config import AppConfig, apply_aspect_ratio
from shortsforge.export import render_single_short
from shortsforge.ffmpeg_utils import extract_stream_info, probe_media, run_ffmpeg
from shortsforge.gui_server import create_gui_server
from shortsforge.llm import LLMClient
from shortsforge.models import (
    ClipCandidate,
    Sentence,
    Transcript,
    VideoMetadata,
    WordTimestamp,
)
from shortsforge.moments import build_batch_parts


def _make_sample_video(path: Path, duration_sec: float = 12.0, width: int = 1280, height: int = 720) -> Path:
    run_ffmpeg([
        "-f", "lavfi",
        "-i", f"testsrc=s={width}x{height}:r=30:d={duration_sec}",
        "-f", "lavfi",
        "-i", f"sine=frequency=330:sample_rate=48000:duration={duration_sec}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-c:a", "aac",
        "-shortest",
        str(path),
    ])
    return path


def _wait_job_done(base_url: str, job_id: str, timeout_sec: float = 90.0) -> dict:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        with urllib.request.urlopen(f"{base_url}/api/jobs/{job_id}") as resp:
            job = json.loads(resp.read().decode("utf-8"))
        if job["status"] != "running":
            return job
        time.sleep(0.4)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout_sec}s")


def test_build_batch_parts_splits_20min_into_4_parts_with_sentence_snapping() -> None:
    # Build synthetic 20-minute (1200s) transcript with sentence endings every 10s
    sentences = []
    for i in range(120):
        st = float(i * 10)
        en = float((i + 1) * 10)
        if en == 300.0:
            en = 297.5  # Test sentence-boundary snapping near 5-min boundary
        if st == 300.0:
            st = 297.5
        sentences.append(Sentence(index=i, text=f"Segment number {i} completes here.", start=st, end=en))

    tr = Transcript(language="en", duration=1200.0, sentences=sentences, words=[])
    meta = VideoMetadata(
        source="demo.mp4",
        source_type="local",
        title="Twenty Minute Talk",
        duration=1200.0,
    )

    parts = build_batch_parts(tr, meta, part_duration_sec=300.0, max_parts=0)
    assert len(parts) == 4
    assert [p.id for p in parts] == ["clip_01", "clip_02", "clip_03", "clip_04"]
    assert parts[0].start == 0.0
    assert parts[0].end == 297.5
    assert parts[1].start == 297.5
    assert parts[-1].end == 1200.0

    # Max parts limit respected
    capped = build_batch_parts(tr, meta, part_duration_sec=300.0, max_parts=2)
    assert len(capped) == 2
    assert capped[1].end < 700.0


def test_render_16x9_and_playback_speed_sync(tmp_path: Path) -> None:
    src_mp4 = _make_sample_video(tmp_path / "src_16x9.mp4", duration_sec=10.0)

    cfg = AppConfig()
    cfg.llm.provider = "mock"
    cfg.reframe.face_tracking_enabled = False
    apply_aspect_ratio(cfg, "16:9")
    cfg.export.playback_speed = 1.5
    assert cfg.reframe.output_width == 1920
    assert cfg.reframe.output_height == 1080

    words = [
        WordTimestamp(word="Welcome", start=1.5, end=2.4),
        WordTimestamp(word="to", start=2.4, end=2.8),
        WordTimestamp(word="studio", start=2.8, end=4.0),
        WordTimestamp(word="session.", start=4.0, end=5.4),
    ]
    tr = Transcript(
        language="en",
        duration=10.0,
        sentences=[Sentence(index=0, text="Welcome to studio session.", start=1.5, end=5.4, words=words)],
        words=words,
    )
    meta = VideoMetadata(
        source=str(src_mp4),
        source_type="local",
        title="Studio Widescreen Demo",
        channel="Studio",
        local_video_path=str(src_mp4),
        duration=10.0,
        width=1280,
        height=720,
        fps=30.0,
    )
    cand = ClipCandidate(
        id="clip_01",
        start=0.0,
        end=6.0,
        score=95,
        reason="Part 1",
        key_quote="Welcome to studio session.",
        transcript_text="Welcome to studio session.",
        peak_start=0.0,
        peak_end=3.0,
    )

    work_dir = tmp_path / "work"
    out_dir = tmp_path / "out"
    exp = render_single_short(
        candidate=cand,
        hook_text="Part 1",
        metadata=meta,
        transcript=tr,
        person="Host",
        work_dir=work_dir,
        output_dir=out_dir,
        llm=LLMClient(cfg),
        config=cfg,
    )

    out_mp4 = out_dir / exp.output_video
    assert out_mp4.exists()
    assert exp.aspect_ratio == "16:9"
    assert exp.output_width == 1920
    assert exp.output_height == 1080
    assert exp.playback_speed == 1.5

    sinfo = extract_stream_info(probe_media(out_mp4))
    assert sinfo["width"] == 1920 and sinfo["height"] == 1080
    assert sinfo["has_audio"] is True
    # 6.0s clip at 1.5x speed -> ~4.0s output duration
    assert 3.6 <= sinfo["duration"] <= 4.4

    lufs = measure_integrated_lufs(out_mp4)
    assert lufs is not None and -16.5 <= lufs <= -11.5

    # Check ASS resolution and speed-scaled caption timing (word 1 starts at 1.5s / 1.5 = 1.00s)
    ass_text = (work_dir / "clip_01" / "captions.ass").read_text(encoding="utf-8")
    assert "PlayResX: 1920" in ass_text
    assert "PlayResY: 1080" in ass_text
    assert "0:00:01.00," in ass_text


def test_batch_mode_pipeline_and_edit_window_api(tmp_path: Path) -> None:
    src_mp4 = _make_sample_video(tmp_path / "batch_source.mp4", duration_sec=8.0)

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.safe_dump({
            "paths": {
                "projects_dir": str(tmp_path / "projects"),
                "outputs_dir": str(tmp_path / "outputs"),
            },
            "llm": {"provider": "mock"},
            "transcription": {"model_size": "tiny"},
            "reframe": {"face_tracking_enabled": False},
        }),
        encoding="utf-8",
    )

    server = create_gui_server(config_path=cfg_file, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    try:
        # 1. Verify UI HTML has Mode switcher, 16:9 option, Playback Speed slider, and Open in Editing Window modal
        with urllib.request.urlopen(f"{base}/") as resp:
            html_body = resp.read().decode("utf-8")
            assert resp.status == 200
            assert "Batch Videos (Split Parts)" in html_body
            assert "16:9 Widescreen" in html_body
            assert "Open in Editing Window" in html_body
            assert "m-speed-range" in html_body
            assert "Start Text Box" in html_body
            assert "m-attn-font" in html_body
            assert "m-attn-color" in html_body
            assert "gold_luxury" in html_body

        # 2. Verify /api/probe-source works for local video
        req_probe = urllib.request.Request(
            f"{base}/api/probe-source",
            data=json.dumps({"source": str(src_mp4)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_probe) as resp:
            probed = json.loads(resp.read().decode("utf-8"))
            assert 7.5 <= probed["duration"] <= 8.5
            assert probed["width"] == 1280

        # 3. Run autopilot in Batch mode (8s split at 4s/part = 2 parts) in 16:9 @ 1.25x
        req_auto = urllib.request.Request(
            f"{base}/api/jobs/autopilot",
            data=json.dumps({
                "source": str(src_mp4),
                "mode": "batch",
                "part_duration_sec": 4.0,
                "max_clips": 2,
                "aspect_ratio": "16:9",
                "playback_speed": 1.25,
                "face_tracking_enabled": False,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_auto) as resp:
            assert resp.status == 202
            job = json.loads(resp.read().decode("utf-8"))

        done_job = _wait_job_done(base, job["job_id"], timeout_sec=120.0)
        assert done_job["status"] == "completed", f"Autopilot batch job failed: {done_job}"
        proj_name = done_job["project_name"]

        with urllib.request.urlopen(f"{base}/api/projects/{proj_name}") as resp:
            detail = json.loads(resp.read().decode("utf-8"))
            assert detail["state"]["mode"] == "batch"
            assert detail["state"]["aspect_ratio"] == "16:9"
            assert len(detail["outputs"]) == 2
            assert detail["outputs"][0]["output_width"] == 1920
            assert detail["outputs"][0]["playback_speed"] == 1.25

        # 4. Edit clip_01 via Editing Window API changing speed + custom intro box styling (should preserve 16:9)
        req_edit = urllib.request.Request(
            f"{base}/api/jobs/edit-clip",
            data=json.dumps({
                "name": proj_name,
                "clip_id": "clip_01",
                "playback_speed": 1.5,
                "caption_position": "center",
                "video_filters": ["sharp_hd"],
                "attention_card_enabled": True,
                "attention_card_text": "Custom Opening Message",
                "attention_card_duration": 1.0,
                "attention_card_style": "gold_luxury",
                "attention_card_font_style": "impact",
                "attention_card_text_color": "gold",
                "attention_card_font_size": "large",
                "attention_card_text_case": "upper",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_edit) as resp:
            assert resp.status == 202
            ed_job = json.loads(resp.read().decode("utf-8"))

        done_ed = _wait_job_done(base, ed_job["job_id"], timeout_sec=90.0)
        assert done_ed["status"] == "completed", f"Edit clip failed: {done_ed}"

        with urllib.request.urlopen(f"{base}/api/projects/{proj_name}") as resp:
            detail2 = json.loads(resp.read().decode("utf-8"))
            c1 = next(o for o in detail2["outputs"] if o["clip_id"] == "clip_01")
            assert c1["aspect_ratio"] == "16:9"
            assert c1["output_width"] == 1920
            assert c1["playback_speed"] == 1.5
            assert c1["caption_position"] == "center"
            assert "sharp_hd" in c1["video_filters"]
            assert c1["attention_card_enabled"] is True
            assert c1["attention_card_text"] == "Custom Opening Message"
            assert c1["attention_card_duration"] == 1.0
            assert c1["attention_card_style"] == "gold_luxury"
            assert c1["attention_card_font_style"] == "impact"
            assert c1["attention_card_text_color"] == "gold"
            assert c1["attention_card_font_size"] == "large"
            assert c1["attention_card_text_case"] == "upper"
    finally:
        server.shutdown()
        server.server_close()


def test_custom_intro_text_box_styles_and_first_second_render(tmp_path: Path) -> None:
    from PIL import Image

    src_mp4 = _make_sample_video(tmp_path / "src_card.mp4", duration_sec=6.0)
    cfg = AppConfig()
    cfg.llm.provider = "mock"
    cfg.reframe.face_tracking_enabled = False
    cfg.overlays.attention_card_enabled = True
    cfg.overlays.attention_card_text = "Watch This First Second!"
    cfg.overlays.attention_card_duration_sec = 1.0
    cfg.overlays.attention_card_style = "gold_luxury"
    cfg.overlays.attention_card_font_style = "impact"
    cfg.overlays.attention_card_text_color = "gold"
    cfg.overlays.attention_card_font_size = "large"
    cfg.overlays.attention_card_text_case = "upper"

    words = [WordTimestamp(word="Hello", start=0.5, end=1.2)]
    tr = Transcript(
        language="en",
        duration=6.0,
        sentences=[Sentence(index=0, text="Hello", start=0.5, end=1.2, words=words)],
        words=words,
    )
    meta = VideoMetadata(
        source=str(src_mp4),
        source_type="local",
        title="Custom Box Test",
        local_video_path=str(src_mp4),
        duration=6.0,
        width=1280,
        height=720,
        fps=30.0,
    )
    cand = ClipCandidate(
        id="clip_01",
        start=0.0,
        end=5.0,
        score=92,
        reason="Test",
        key_quote="Hello world",
        transcript_text="Hello world",
        peak_start=0.0,
        peak_end=2.5,
    )

    work_dir = tmp_path / "work"
    out_dir = tmp_path / "out"
    exp = render_single_short(
        candidate=cand,
        hook_text="Fallback Hook",
        metadata=meta,
        transcript=tr,
        person="Host",
        work_dir=work_dir,
        output_dir=out_dir,
        llm=LLMClient(cfg),
        config=cfg,
    )

    card_png = work_dir / "clip_01" / "attention_card.png"
    assert card_png.exists()
    with Image.open(card_png) as im:
        assert im.size == (1080, 1920)
        assert im.getextrema()[3][1] > 0  # Non-empty alpha channel rendered

    out_mp4 = out_dir / exp.output_video
    assert out_mp4.exists()
    assert exp.attention_card_enabled is True
    assert exp.attention_card_text == "Watch This First Second!"
    assert exp.attention_card_duration == 1.0
    assert exp.attention_card_style == "gold_luxury"
    assert exp.attention_card_font_style == "impact"
    assert exp.attention_card_text_color == "gold"
    assert exp.attention_card_font_size == "large"
    assert exp.attention_card_text_case == "upper"
