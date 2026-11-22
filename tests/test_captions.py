"""Tests for word-by-word caption grouping, timing offsets, emphasis, safe zones, and style presets."""

from __future__ import annotations

import re

from shortsforge.captions import (
    PRESETS,
    build_ass_captions,
    extract_emphasis_keywords,
    format_ass_timestamp,
    group_caption_words,
    is_emphasized_word,
)
from shortsforge.config import AppConfig
from shortsforge.models import WordTimestamp


def _sample_words() -> list[WordTimestamp]:
    tokens = [
        ("In", 10.0, 10.2),
        ("1998,", 10.2, 10.6),
        ("I", 10.6, 10.75),
        ("almost", 10.75, 11.1),
        ("quit.", 11.1, 11.5),
        ("Then", 12.4, 12.7),  # >0.55s pause triggers new group
        ("we", 12.7, 12.9),
        ("made", 12.9, 13.2),
        ("$50,000", 13.2, 13.8),
        ("overnight!", 13.8, 14.4),
    ]
    return [WordTimestamp(word=w, start=s, end=e) for w, s, e in tokens]


def test_group_caption_words_respects_1_to_3_limit_and_breaks() -> None:
    words = _sample_words()
    groups = group_caption_words(words, words_per_group=3)
    assert all(1 <= len(g) <= 3 for g in groups)
    # Punctuation on "quit." should force a group boundary before "Then"
    flat_texts = [" ".join(w.word for w in g) for g in groups]
    assert any(t.endswith("quit.") for t in flat_texts)


def test_emphasis_detects_numbers_and_keywords() -> None:
    kws = extract_emphasis_keywords("I almost quit before the breakthrough.", "Why he refused to quit")
    assert is_emphasized_word("1998,", kws) is True
    assert is_emphasized_word("$50,000", kws) is True
    assert is_emphasized_word("breakthrough", kws) is True
    assert is_emphasized_word("the", kws) is False


def test_ass_caption_timing_monotonic_and_offset_shifted() -> None:
    cfg = AppConfig()
    cfg.captions.preset = "neon"
    cfg.captions.active_color = "yellow"
    words = _sample_words()

    hook_offset = 3.0
    ass = build_ass_captions(
        clip_words=words,
        clip_start=10.0,
        time_offset=hook_offset,
        config=cfg,
        key_quote="I almost quit.",
        hook_text="Why he almost quit",
    )

    # Verify PlayRes and safe-zone margins are present
    assert "PlayResX: 1080" in ass
    assert "PlayResY: 1920" in ass
    assert f",{cfg.captions.safe_margin_left},{cfg.captions.safe_margin_right},{cfg.captions.safe_margin_bottom}," in ass

    # Parse every Dialogue line and verify monotonic non-overlapping timestamps >= hook_offset
    matches = re.findall(r"Dialogue: 0,(\d+:\d{2}:\d{2}\.\d{2}),(\d+:\d{2}:\d{2}\.\d{2}),", ass)
    assert len(matches) == len(words)

    def _to_sec(ts: str) -> float:
        h, m, rest = ts.split(":")
        s, cs = rest.split(".")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100.0

    prev_end = 0.0
    for st_s, en_s in matches:
        st = _to_sec(st_s)
        en = _to_sec(en_s)
        assert st >= hook_offset - 0.01
        assert en > st
        assert st >= prev_end - 0.005
        prev_end = en

    # Active-word pop-in animation transform tag must be present
    assert "\\t(0," in ass


def test_all_three_caption_presets_render_valid_ass() -> None:
    words = _sample_words()
    for preset_name in ("neon", "comic", "minimal"):
        assert preset_name in PRESETS
        cfg = AppConfig()
        cfg.captions.preset = preset_name
        cfg.captions.active_color = "red" if preset_name == "comic" else "yellow"
        ass = build_ass_captions(words, clip_start=10.0, time_offset=2.5, config=cfg)
        assert "[V4+ Styles]" in ass
        assert "[Events]" in ass
        assert "Dialogue:" in ass


def test_format_ass_timestamp() -> None:
    assert format_ass_timestamp(0.0) == "0:00:00.00"
    assert format_ass_timestamp(65.43) == "0:01:05.43"
