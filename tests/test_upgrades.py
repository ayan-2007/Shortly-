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

    title2 = "Gabe Newell Talks Steam Deck's Origin, Goals, and Future"
    assert auto_detect_person_from_title(title2, "IGN", "") == "Gabe Newell"


def test_unpunctuated_youtube_auto_captions_still_discover_viral_clips() -> None:
    """YouTube auto-captions are lowercase without punctuation; moment discovery must still find viral clips."""
    cfg = AppConfig()
    cfg.llm.provider = "mock"
    raw_lines = [
        (10.0, 22.0, "we built this handheld gaming pc because nobody believed portable hardware could run every modern game"),
        (22.0, 34.0, "the hardest decision was hitting the 399 price point while keeping performance high for players"),
        (34.0, 48.0, "it was painful to reach that cost target but critical for the future of open pc gaming platforms"),
        (48.0, 62.0, "if you have game pass or another store you can install anything because customers should own their device"),
        (62.0, 76.0, "that open philosophy changed everything about how developers design software and hardware together"),
    ]
    sents = []
    words = []
    for idx, (st, en, txt) in enumerate(raw_lines):
        toks = txt.split()
        step = (en - st) / len(toks)
        sw = [WordTimestamp(word=w, start=round(st + i * step, 2), end=round(st + (i + 1) * step, 2)) for i, w in enumerate(toks)]
        words.extend(sw)
        sents.append(Sentence(index=idx, text=txt, start=st, end=en, speaker="SPEAKER_1", words=sw))

    tr = Transcript(language="en", duration=80.0, sentences=sents, words=words)
    meta = VideoMetadata(
        source="https://youtu.be/demo1234567",
        source_type="url",
        title="Gabe Newell Talks Steam Deck's Origin, Goals, and Future",
        channel="IGN",
        duration=80.0,
        heatmap=[{"start_time": 18.0, "end_time": 45.0, "value": 0.92}],
    )
    cands = find_best_moments(tr, meta, "Gabe Newell", LLMClient(cfg), cfg)
    assert len(cands) >= 1
    assert cands[0].score >= 78


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
    # Music, whoosh, and opening title text must be off by default; captions default to neon purple
    assert cfg.audio.bg_music_enabled is False
    assert cfg.audio.hook_bed_enabled is False
    assert cfg.audio.transition_sfx_enabled is False
    assert cfg.hook.enabled is False
    assert cfg.hook.show_opening_title is False
    assert cfg.hook.transition_duration_sec == 0.0
    assert cfg.captions.preset == "iman"
    assert cfg.captions.active_color == "purple"

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
            time_offset=0.0,
            config=cfg,
            key_quote="90% comes from China.",
            hook_text="The 90% Chokehold Nobody Realizes.",
            hook_words=words,
            hook_start=10.0,
            hook_banner_duration=2.8,
        )
        # No opening title overlay events when show_opening_title is False (default)
        assert "Dialogue: 1," not in ass
        assert "Dialogue: 2," not in ass
        # Animated word-by-word captions rendered in electric neon purple
        assert "Dialogue: 0," in ass
        assert PRESETS[preset_name].active_purple in ass
        assert PRESETS[preset_name].neon_glow_outline in ass


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
    # By default, no opening title text is burned at the start of the video
    ass_clean = build_ass_captions(
        clip_words=words,
        clip_start=10.0,
        time_offset=0.0,
        config=cfg,
        key_quote="Invest every month.",
        hook_text="Stop making this money mistake",
        hook_words=words,
        hook_start=10.0,
        hook_banner_duration=2.8,
    )
    assert "Dialogue: 1," not in ass_clean
    assert "Dialogue: 0," in ass_clean
    assert PRESETS["iman"].active_purple in ass_clean


def test_post_render_caption_positions_and_white_center_attention_card(tmp_path: Path) -> None:
    from PIL import Image
    from shortsforge.captions import resolve_caption_center_y
    from shortsforge.overlays import render_attention_card_png

    cfg = AppConfig()
    words = [
        WordTimestamp(word="Wait", start=5.0, end=5.3),
        WordTimestamp(word="for", start=5.3, end=5.5),
        WordTimestamp(word="this.", start=5.5, end=5.9),
    ]

    # 1. Verify caption vertical positions (Bottom, Exact Center, Upper-Mid, Lower-Mid, Top)
    cfg.captions.position = "bottom"
    y_bot = resolve_caption_center_y(cfg)
    assert y_bot == 1500
    ass_bot = build_ass_captions(words, clip_start=5.0, time_offset=0.0, config=cfg)
    assert "\\move(540,1526,540,1500," in ass_bot

    cfg.captions.position = "center"
    y_cen = resolve_caption_center_y(cfg)
    assert y_cen == 960  # Exact center of 1080x1920 screen!
    ass_cen = build_ass_captions(words, clip_start=5.0, time_offset=0.0, config=cfg)
    assert "\\move(540,986,540,960," in ass_cen

    cfg.captions.position = "top"
    assert resolve_caption_center_y(cfg) < 500

    # 2. Verify White Box Center Attention Card PNG generator
    cfg.captions.position = "bottom"
    card_png = render_attention_card_png(
        card_text="The $60,000 Rule Nobody Talks About",
        output_path=tmp_path / "center_white_box.png",
        config=cfg,
        position="center",
        style="white_box",
    )
    assert card_png.exists()
    with Image.open(card_png) as im:
        assert im.size == (1080, 1920)
        assert im.mode == "RGBA"
        # Center of screen should have opaque white box / dark text pixels
        center_alpha = im.getpixel((540, int(1920 * 0.46)))[3]
        assert center_alpha > 200


def test_active_speaker_face_tracking_and_balanced_clip_threshold() -> None:
    import numpy as np
    from shortsforge.quality_check import FaceBox, FaceDetector
    from shortsforge.reframe import _build_piecewise_crop_x_expr, _pick_active_speaker_face
    from shortsforge.review import save_auto_decisions
    from shortsforge.state import ProjectWorkspace

    # 1. Fast neural FaceDetector initialization
    det = FaceDetector()
    assert det.backend in ("yunet", "haar")
    det.close()

    # 2. Active speaker selection picks the face with lower-face/mouth motion in a 2-person wide shot
    prev_g = np.full((90, 160), 100, dtype=np.uint8)
    curr_g = prev_g.copy()
    # Add mouth motion on the right speaker (around x=1920 in 2560x1440 -> x=120 in 160x90)
    curr_g[30:55, 105:135] = 220

    left_speaker = FaceBox(x=400, y=300, w=320, h=380, confidence=0.88)
    right_speaker = FaceBox(x=1750, y=300, w=320, h=380, confidence=0.88)
    picked = _pick_active_speaker_face(
        detected=[left_speaker, right_speaker],
        small_gray=curr_g,
        prev_small_gray=prev_g,
        src_w=2560,
        src_h=1440,
        current_cam_x=1280.0,
        initialized_cam=False,
    )
    assert picked.cx == right_speaker.cx

    # 3. Piecewise crop expression handles both scene-cut steps and smooth tracking
    expr = _build_piecewise_crop_x_expr([(0.05, 300.0), (5.00, 300.0), (5.04, 1100.0), (10.0, 1180.0)], max_x=1750)
    assert "clip(" in expr
    assert "1100.0" in expr

    # 4. Balanced clip count sweet spot: high threshold never drops to 3 on a 10-clip podcast, low threshold never exceeds cap
    from shortsforge.models import ClipCandidate

    many_cands = [
        ClipCandidate(
            id=f"clip_{i:02d}",
            start=i * 60.0,
            end=i * 60.0 + 35.0,
            score=96 - i,
            reason="Viral moment",
            key_quote="The truth is discipline wins.",
            transcript_text="The truth is discipline wins every single time.",
            peak_start=i * 60.0 + 10.0,
            peak_end=i * 60.0 + 13.0,
        )
        for i in range(1, 22)
    ]
    ws = ProjectWorkspace(Path("projects/_tmp_balance_test"), Path("outputs/_tmp_balance_test"))
    try:
        # Even if threshold is set too high (95 -> only 2 clips >= 95), adaptive floor approves 6-8 clips
        high_doc = save_auto_decisions("bal", many_cands, ws, threshold=95, max_approved=12)
        high_approved = sum(1 for d in high_doc.decisions.values() if d.approved)
        assert 6 <= high_approved <= 12

        # Even if threshold is set too low (10 -> all 21 clips >= 10), hard ceiling caps at 12
        low_doc = save_auto_decisions("bal", many_cands, ws, threshold=10, max_approved=12)
        low_approved = sum(1 for d in low_doc.decisions.values() if d.approved)
        assert low_approved == 12
    finally:
        import shutil
        shutil.rmtree(ws.project_dir, ignore_errors=True)
        shutil.rmtree(ws.output_dir, ignore_errors=True)
