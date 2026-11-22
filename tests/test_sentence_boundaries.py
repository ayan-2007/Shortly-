"""Tests for sentence-boundary cutting, 20-55s clamping, and rubric scoring."""

from __future__ import annotations

from shortsforge.config import AppConfig
from shortsforge.models import ScoreBreakdown, Sentence, WordTimestamp
from shortsforge.moments import compute_weighted_score, snap_to_sentence_boundaries


def _make_sentences(durations: list[float]) -> list[Sentence]:
    sents: list[Sentence] = []
    t = 0.0
    for idx, dur in enumerate(durations):
        start = round(t, 3)
        end = round(t + dur, 3)
        words = [
            WordTimestamp(word=f"Word{idx}a", start=start, end=round(start + dur * 0.45, 3)),
            WordTimestamp(word=f"word{idx}b.", start=round(start + dur * 0.5, 3), end=end),
        ]
        sents.append(
            Sentence(
                index=idx,
                text=f"Word{idx}a word{idx}b.",
                start=start,
                end=end,
                speaker="SPEAKER_1",
                words=words,
            )
        )
        t += dur + 0.2
    return sents


def test_snaps_mid_sentence_timestamps_to_exact_sentence_boundaries() -> None:
    sentences = _make_sentences([6.0] * 12)  # ~74 seconds total
    # Propose mid-sentence raw times (inside sentence 1 and inside sentence 5)
    snapped = snap_to_sentence_boundaries(sentences, raw_start=8.5, raw_end=35.0, min_duration=20.0, max_duration=55.0)
    assert snapped is not None
    s_start, s_end, selected = snapped

    # Must align exactly with sentence boundaries (never mid-sentence)
    valid_starts = {s.start for s in sentences}
    valid_ends = {s.end for s in sentences}
    assert s_start in valid_starts
    assert s_end in valid_ends
    assert 20.0 <= (s_end - s_start) <= 55.0
    assert selected[0].start == s_start
    assert selected[-1].end == s_end


def test_expands_too_short_proposal_to_at_least_min_duration() -> None:
    sentences = _make_sentences([5.0] * 10)
    # Raw proposal is only ~5 seconds long; must expand cleanly along sentence boundaries to >= 20s
    snapped = snap_to_sentence_boundaries(sentences, raw_start=10.0, raw_end=15.0, min_duration=20.0, max_duration=55.0)
    assert snapped is not None
    s_start, s_end, _ = snapped
    assert 20.0 <= (s_end - s_start) <= 55.0


def test_shrinks_too_long_proposal_within_max_duration() -> None:
    sentences = _make_sentences([8.0] * 12)  # ~98s total
    # Raw proposal spans 85 seconds; must shrink to <= 55s on sentence boundaries
    snapped = snap_to_sentence_boundaries(sentences, raw_start=0.0, raw_end=85.0, min_duration=20.0, max_duration=55.0)
    assert snapped is not None
    s_start, s_end, _ = snapped
    assert 20.0 <= (s_end - s_start) <= 55.0


def test_weighted_score_calculation() -> None:
    cfg = AppConfig()
    bd = ScoreBreakdown(
        emotional_peak=100,
        quotable_statement=80,
        humor_payoff=60,
        self_contained=90,
        clean_boundaries=70,
    )
    score = compute_weighted_score(bd, cfg)
    # 100*.25 + 80*.25 + 60*.15 + 90*.20 + 70*.15 = 25 + 20 + 9 + 18 + 10.5 = 82.5 -> 82/83
    assert score in (82, 83)
