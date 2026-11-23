"""Tests for LLM disk caching, yt-dlp error classification, review.html generation, and --auto filtering."""

from __future__ import annotations

from pathlib import Path

from shortsforge.config import AppConfig
from shortsforge.export import write_sources_and_warnings
from shortsforge.ingest import _classify_yt_dlp_error, extract_show_and_air_date
from shortsforge.llm import LLMClient
from shortsforge.models import (
    ClipCandidate,
    ClipExportMetadata,
    HookSet,
    HookVariant,
    VideoMetadata,
)
from shortsforge.review import generate_review_html, load_decisions, save_auto_decisions
from shortsforge.state import ProjectWorkspace


def test_llm_disk_cache_hits_on_repeat_call(tmp_path: Path) -> None:
    cfg = AppConfig()
    cfg.llm.provider = "mock"
    cfg.llm.cache_enabled = True
    ws = ProjectWorkspace(tmp_path / "proj", tmp_path / "out")

    llm = LLMClient(cfg, ws)
    first = llm.complete_json("test_ns", "sys", "user prompt", mock_payload={"ok": True, "n": 42})
    assert first == {"ok": True, "n": 42}
    assert llm.last_cache_hit is False

    second = llm.complete_json("test_ns", "sys", "user prompt", mock_payload={"ok": False, "n": 99})
    assert second == {"ok": True, "n": 42}
    assert llm.last_cache_hit is True


def test_yt_dlp_error_classification_messages() -> None:
    priv = _classify_yt_dlp_error(RuntimeError("ERROR: Private video. Sign in if you've been granted access"), "https://youtu.be/abc")
    assert "PRIVATE" in str(priv)

    rem = _classify_yt_dlp_error(RuntimeError("ERROR: Video unavailable. This video has been removed"), "https://youtu.be/abc")
    assert "REMOVED" in str(rem)

    age = _classify_yt_dlp_error(RuntimeError("ERROR: Sign in to confirm your age"), "https://youtu.be/abc")
    assert "AGE-RESTRICTED" in str(age)


def test_show_and_air_date_extraction() -> None:
    show, air, year = extract_show_and_air_date(
        "Carl Sagan on The Tonight Show (1980)",
        "Original air date: October 10, 1980. Classic science interview.",
        "2019-05-01",
    )
    assert "Tonight Show" in show
    assert "1980" in air
    assert year == "1980"


def test_review_html_and_auto_decisions_and_sources_warnings(tmp_path: Path) -> None:
    cfg = AppConfig()
    ws = ProjectWorkspace(tmp_path / "proj", tmp_path / "out")

    meta = VideoMetadata(
        source="https://example.com/interview",
        source_type="url",
        title="Full Interview (1998)",
        channel="Archive Channel",
        show_name="Late Night Archive",
        year="1998",
        duration=120.0,
    )
    c1 = ClipCandidate(
        id="clip_01",
        start=5.0,
        end=35.0,
        score=89,
        reason="High emotion and strong takeaway.",
        key_quote="Never stop building.",
        transcript_text="Never stop building even when it gets tough.",
        peak_start=15.0,
        peak_end=17.5,
        hooks=HookSet(
            variants=[HookVariant(rank=1, text="His honest secret to never quitting.", predicted_retention=91)]
        ),
    )
    c2 = ClipCandidate(
        id="clip_02",
        start=50.0,
        end=80.0,
        score=62,
        reason="Lower energy anecdote.",
        key_quote="It rained that Tuesday.",
        transcript_text="It rained that Tuesday when we arrived.",
        peak_start=60.0,
        peak_end=62.5,
        hooks=HookSet(
            variants=[HookVariant(rank=1, text="What happened on that rainy Tuesday.", predicted_retention=65)],
            flagged_for_review=True,
            flag_reason="Failed verification 3x",
        ),
    )

    review_path = generate_review_html("demo", "Alex Rivera", meta, [c1, c2], ws, cfg)
    html_text = review_path.read_text(encoding="utf-8")
    assert "Shortly Review" in html_text
    assert "His honest secret to never quitting." in html_text
    assert "Approve" in html_text and "Reject" in html_text
    assert "hook-input" in html_text

    # --auto threshold = 75 approves c1 (89) and rejects c2 (62)
    auto_doc = save_auto_decisions("demo", [c1, c2], ws, threshold=75)
    loaded = load_decisions(ws, "demo")
    assert loaded.decisions["clip_01"].approved is True
    assert loaded.decisions["clip_02"].approved is False
    assert auto_doc.project_name == "demo"

    # sources.txt & warnings.txt
    src_p, warn_p = write_sources_and_warnings(
        output_dir=ws.output_dir,
        metadata=meta,
        person="Alex Rivera",
        exported_clips=[
            ClipExportMetadata(
                clip_id="clip_01",
                output_video="clip_01_demo.mp4",
                thumbnail="clip_01_demo_thumb.jpg",
                titles=["T1", "T2", "T3", "T4", "T5"],
                description="Source: Archive Channel",
                hashtags=["#shorts", "#a", "#b", "#c", "#d", "#e", "#f", "#g"],
                pinned_comment="What do you think?",
                hook_text="His honest secret to never quitting.",
                start=5.0,
                end=35.0,
                score=89,
                lufs=-14.0,
            )
        ],
        warnings=["clip_02 hook FLAGGED after 3 failed verifications"],
        config=cfg,
    )
    assert "Archive Channel" in src_p.read_text(encoding="utf-8")
    assert "clip_02 hook FLAGGED" in warn_p.read_text(encoding="utf-8")


def test_reframe_4_3_blurred_fit_and_two_person_split_dimensions() -> None:
    import numpy as np
    from shortsforge.reframe import _render_blurred_fit, _render_two_person_split, _smooth_step

    # 4:3 vintage frame -> 1080x1920 blurred fit with gentle zoom
    frame_43 = np.full((480, 640, 3), 120, dtype=np.uint8)
    out_43 = _render_blurred_fit(frame_43, 1080, 1920, zoom=1.06)
    assert out_43.shape == (1920, 1080, 3)

    # 16:9 wide two-person frame -> 1080x1920 stacked split
    frame_169 = np.full((720, 1280, 3), 90, dtype=np.uint8)
    out_split = _render_two_person_split(frame_169, (320.0, 340.0), (960.0, 340.0), 1080, 1920)
    assert out_split.shape == (1920, 1080, 3)

    # Deadzone holds steady; large move clamps to max_step_px
    assert _smooth_step(500.0, 505.0, alpha=0.2, deadzone_px=10.0, max_step_px=15.0) == 500.0
    moved = _smooth_step(500.0, 900.0, alpha=0.5, deadzone_px=10.0, max_step_px=15.0)
    assert moved == 515.0

