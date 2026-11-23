"""Tests for complete-message boundary expansion, zero clip limit, mood music, kinetic hook captions, and video-specific metadata."""

from __future__ import annotations

from pathlib import Path

from shortsforge.captions import PRESETS, build_ass_captions
from shortsforge.config import AppConfig
from shortsforge.export import build_video_enhancement_filter_chain
from shortsforge.ingest import _build_format_selector, auto_detect_person_from_title
from shortsforge.llm import LLMClient, _build_hooks_from_quote
from shortsforge.models import ScoreBreakdown, Sentence, Transcript, VideoMetadata, WordTimestamp
from shortsforge.moments import detect_clip_mood, evaluate_viral_potential, find_best_moments, snap_to_complete_message
from shortsforge.sfx import generate_hook_ambient_bed, generate_mood_music_track


def test_auto_detect_person_preserves_initial_capital_and_apostrophe() -> None:
    title = "Kevin O'Leary: Every Time You Get Paid, Do This!"
    detected = auto_detect_person_from_title(title, "The Diary Of A CEO", "")
    assert detected == "Kevin O'Leary"


def test_format_selector_preserves_high_res_for_long_videos() -> None:
    cfg = AppConfig()
    sel = _build_format_selector(cfg, duration_sec=6800.0)
    assert "720" not in sel or "1440" in sel
    assert "height>=1080" in sel


def test_snap_to_complete_message_never_cuts_off_mid_thought() -> None:
    sents = [
        Sentence(index=0, text="I learned a crucial lesson about money.", start=0.0, end=5.0),
        Sentence(index=1, text="Wait until you hear what happened,", start=5.4, end=12.0),
        Sentence(index=2, text="so I started building an index fund and", start=12.3, end=42.0),
        Sentence(index=3, text="nobody realized how fast it would compound over ten years.", start=42.2, end=66.0),
        Sentence(index=4, text="Now let's switch to another topic.", start=68.0, end=73.0),
    ]
    snapped = snap_to_complete_message(sents, raw_start=5.5, raw_end=41.5, min_duration=12.0, max_duration=180.0)
    assert snapped is not None
    s_start, s_end, selected = snapped
    # Should expand past 55s to finish sentence 3 cleanly instead of stopping on dangling "and" at 42.0s!
    assert s_end >= 66.0
    assert selected[-1].index == 3


def test_unlimited_moments_extracts_at_least_12_varied_clips() -> None:
    cfg = AppConfig()
    cfg.llm.provider = "mock"
    cfg.moments.max_candidates = 0  # Unlimited
    cfg.moments.min_candidates = 12

    # Synthesize a 40-minute (2400s) multi-topic transcript
    sents = []
    words = []
    t = 15.0
    topics = [
        "investing money and dividends can double your wealth faster.",
        "every business founder must protect profit margins.",
        "marriage and divorce come down to honest financial talks.",
        "discipline and morning habits changed my entire career.",
    ]
    for i in range(96):
        txt = f"Truth number {i}: {topics[i % len(topics)]}"
        dur = 22.0
        st = round(t, 2)
        en = round(t + dur, 2)
        w_toks = txt.split()
        step = dur / len(w_toks)
        s_words = []
        for wi, tok in enumerate(w_toks):
            wt = WordTimestamp(word=tok, start=round(st + wi * step, 2), end=round(st + (wi + 1) * step, 2))
            words.append(wt)
            s_words.append(wt)
        sents.append(Sentence(index=i, text=txt, start=st, end=en, speaker=f"SPEAKER_{(i % 2) + 1}", words=s_words))
        t += dur + 2.0

    transcript = Transcript(language="en", duration=t, sentences=sents, words=words)
    meta = VideoMetadata(
        source="demo.mp4",
        source_type="local",
        title="Kevin O'Leary Full Podcast",
        duration=t,
        heatmap=[{"start_time": 120.0 * k, "end_time": 120.0 * k + 35.0, "value": 0.9 - k * 0.02} for k in range(1, 16)],
    )
    llm = LLMClient(cfg)
    cands = find_best_moments(transcript, meta, "Kevin O'Leary", llm, cfg)
    assert 12 <= len(cands) <= 15


def test_quality_over_quantity_rejects_sponsor_and_rambling_clips() -> None:
    # 1. Sponsor read sentence must be rejected by evaluate_viral_potential
    sponsor_sents = [
        Sentence(
            index=0,
            text="Shopify's best-in-class checkout makes sure you get paid from day one.",
            start=10.0,
            end=18.0,
        ),
        Sentence(
            index=1,
            text="Head over to shopify.com/bartlett and start your free trial right now.",
            start=18.2,
            end=26.0,
        ),
    ]
    viable_sp, _ = evaluate_viral_potential(
        sponsor_sents,
        " ".join(s.text for s in sponsor_sents),
    )
    assert viable_sp is False

    # 2. Rambling visual-prop / high-filler fragment must be rejected
    ramble_sents = [
        Sentence(
            index=0,
            text="Yeah, yeah, um, when I was young, uh, you know, I found out something, um, sort of,",
            start=30.0,
            end=40.0,
        ),
        Sentence(
            index=1,
            text="uh, yeah, there she is on the screen right now, Georgia. So,",
            start=40.2,
            end=48.0,
        ),
    ]
    viable_rm, _ = evaluate_viral_potential(
        ramble_sents,
        " ".join(s.text for s in ramble_sents),
    )
    assert viable_rm is False

    # 3. Genuine high-impact viral insight passes with strong score
    viral_sents = [
        Sentence(
            index=0,
            text="90% of the world's germanium comes from China.",
            start=50.0,
            end=56.0,
        ),
        Sentence(
            index=1,
            text="The truth is that every $400 million semiconductor machine depends on this global formula.",
            start=56.2,
            end=66.0,
        ),
        Sentence(
            index=2,
            text="Nobody realizes that technological progress requires specialization and globalization.",
            start=66.2,
            end=74.0,
        ),
    ]
    viable_ok, bd_ok = evaluate_viral_potential(
        viral_sents,
        " ".join(s.text for s in viral_sents),
    )
    assert viable_ok is True
    assert bd_ok.quotable_statement >= 80


def test_hooks_never_produce_dangling_ellipsis_fragments() -> None:
    variants = _build_hooks_from_quote(
        "But, in terms of what you did and your relationship with your money, what are the most important things for someone to understand who's",
        "The spendaholic is someone who always offers to pay for everything to appear successful. It is a huge financial warning sign in relationships.",
    )
    assert len(variants) == 5
    for v in variants:
        assert not v["text"].endswith("...")
        assert "in terms of what" not in v["text"].lower()


def test_iman_hormozi_documentary_kinetic_captions_and_clean_audio_defaults() -> None:
    cfg = AppConfig()
    # Music and whoosh must be off by default
    assert cfg.audio.bg_music_enabled is False
    assert cfg.audio.hook_bed_enabled is False
    assert cfg.audio.transition_sfx_enabled is False
    assert cfg.hook.transition_duration_sec == 0.0
    assert cfg.captions.preset == "iman"

    words = [
        WordTimestamp(word="90%", start=10.0, end=10.3),
        WordTimestamp(word="comes", start=10.3, end=10.6),
        WordTimestamp(word="from", start=10.6, end=10.9),
        WordTimestamp(word="China.", start=10.9, end=11.3),
    ]
    for preset_name in ("iman", "hormozi", "documentary"):
        assert preset_name in PRESETS
        cfg.captions.preset = preset_name
        ass = build_ass_captions(
            clip_words=words,
            clip_start=10.0,
            time_offset=2.8,
            config=cfg,
            key_quote="90% comes from China.",
            hook_text="The 90% Chokehold Nobody Realizes.",
            hook_words=words,
            hook_start=10.0,
            hook_banner_duration=2.8,
        )
        assert "HookTitle" in ass
        assert "Dialogue: 1," in ass  # Ambient glow layer
        assert "Dialogue: 2," in ass  # Sharp kinetic foreground layer
        assert "Dialogue: 0," in ass  # Animated word-by-word captions


def test_mirror_video_and_quality_enhancement_filters() -> None:
    cfg = AppConfig()
    cfg.export.mirror_video = True
    # Put sharp_hd before denoise_clean to verify automatic denoise-first ordering
    cfg.export.video_filters = ["sharp_hd", "denoise_clean", "warm_studio"]
    chain = build_video_enhancement_filter_chain(cfg, was_upscaled=False)

    assert "hflip" in chain
    # Caps at 2 user filters max
    assert "hqdn3d" in chain
    assert "unsharp" in chain
    assert "colorbalance" not in chain
    # Denoise must execute before sharpening
    assert chain.index("hqdn3d") < chain.index("unsharp")


def test_mood_detection_and_procedural_music_tracks(tmp_path: Path) -> None:
    bd = ScoreBreakdown(emotional_peak=90, quotable_statement=88, humor_payoff=60, self_contained=92, clean_boundaries=94)
    assert detect_clip_mood("My mother cried after the family divorce and loss.", bd) == "emotional"
    assert detect_clip_mood("The arrest warning trap put everyone in danger.", bd) == "tense"

    hook_wav = generate_hook_ambient_bed(tmp_path / "hook.wav", duration=2.0)
    assert hook_wav.exists() and hook_wav.stat().st_size > 10000

    for mood in ("tense", "emotional", "inspiring", "mysterious"):
        m_wav = generate_mood_music_track(tmp_path / f"{mood}.wav", duration=3.0, mood=mood)
        assert m_wav.exists() and m_wav.stat().st_size > 10000


def test_kinetic_hook_headline_and_video_specific_metadata() -> None:
    cfg = AppConfig()
    cfg.llm.provider = "mock"
    llm = LLMClient(cfg)

    resp = llm.complete_json(
        namespace="export_metadata_clip_01",
        system_prompt="sys",
        user_prompt=(
            "Subject: Kevin O'Leary\n"
            "Original Show/Channel: The Diary Of A CEO\n"
            "Original Title: How To 10x Your Income\n"
            "Hook: Stop making this costly money mistake.\n"
            "Key quote: Invest in dividend stocks for compounding wealth.\n"
            "Transcript:\nInvest in dividend stocks and index funds every month so your portfolio compounds forever.\n"
        ),
    )
    assert "post_caption" in resp and resp["post_caption"]
    assert any(t in resp["hashtags"] for t in ("#investing", "#money", "#wealthbuilding"))
    assert len(resp["titles"]) == 5

    words = [
        WordTimestamp(word="Invest", start=10.0, end=10.3),
        WordTimestamp(word="every", start=10.3, end=10.6),
        WordTimestamp(word="month.", start=10.6, end=11.0),
    ]
    ass = build_ass_captions(
        clip_words=words,
        clip_start=10.0,
        time_offset=3.2,
        config=cfg,
        key_quote="Invest every month.",
        hook_text="Stop making this money mistake",
        hook_words=words,
        hook_start=10.0,
        hook_banner_duration=2.8,
    )
    assert "HookTitle" in ass
    assert "Dialogue: 1," in ass
    assert "Dialogue: 0," in ass
