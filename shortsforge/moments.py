"""Stage 3a: Best-moment discovery, 5-axis 0-100 scoring, and deterministic sentence-boundary snapping."""

from __future__ import annotations

from typing import List, Optional, Tuple

from shortsforge.config import AppConfig
from shortsforge.llm import LLMClient
from shortsforge.models import (
    ClipCandidate,
    ScoreBreakdown,
    Sentence,
    Transcript,
    VideoMetadata,
)


MOMENTS_SYSTEM_PROMPT = """You are an elite short-form documentary and podcast editor.
Your job is to find the strongest standalone 20-55 second moments from an interview, show, or podcast transcript.

Score each candidate 0-100 on these five criteria:
1. emotional_peak: genuine emotion, vulnerability, intensity, or dramatic shift
2. quotable_statement: surprising, memorable, or highly quotable line
3. humor_payoff: authentic wit, comedic timing, or unexpected laugh
4. self_contained: makes complete sense on its own with zero prior context
5. clean_boundaries: begins cleanly on a complete thought and ends cleanly on a payoff/sentence boundary

CRITICAL RULES:
- Only use timestamps and quotes that ACTUALLY appear in the provided transcript.
- Each clip must be between 20 and 55 seconds long.
- Start and end strictly on sentence boundaries.
- Identify a 2.0-3.0 second `peak_start`..`peak_end` window inside the clip containing the single most intense/quotable punchline or emotional peak (for a cold open).

Return strict JSON matching:
{
  "candidates": [
    {
      "start": 12.4,
      "end": 46.8,
      "score": 89,
      "score_breakdown": {
        "emotional_peak": 90,
        "quotable_statement": 94,
        "humor_payoff": 70,
        "self_contained": 92,
        "clean_boundaries": 95
      },
      "reason": "One crisp sentence explaining why this moment hooks viewers.",
      "key_quote": "Exact verbatim quote from the transcript.",
      "peak_start": 34.1,
      "peak_end": 36.6
    }
  ]
}"""


def snap_to_sentence_boundaries(
    sentences: List[Sentence],
    raw_start: float,
    raw_end: float,
    min_duration: float = 20.0,
    max_duration: float = 55.0,
) -> Optional[Tuple[float, float, List[Sentence]]]:
    """Snap proposed start/end times to exact sentence boundaries within [min_duration, max_duration]."""
    if not sentences:
        return None

    start_idx = min(range(len(sentences)), key=lambda i: abs(sentences[i].start - raw_start))
    end_idx = min(
        range(start_idx, len(sentences)),
        key=lambda i: abs(sentences[i].end - raw_end),
    )

    def _dur(si: int, ei: int) -> float:
        return sentences[ei].end - sentences[si].start

    while end_idx > start_idx and _dur(start_idx, end_idx) > max_duration:
        trim_end_diff = abs(_dur(start_idx, end_idx - 1) - max_duration)
        trim_start_diff = abs(_dur(start_idx + 1, end_idx) - max_duration)
        if trim_end_diff <= trim_start_diff:
            end_idx -= 1
        else:
            start_idx += 1

    while _dur(start_idx, end_idx) < min_duration:
        can_fwd = end_idx + 1 < len(sentences) and _dur(start_idx, end_idx + 1) <= max_duration
        can_bwd = start_idx - 1 >= 0 and _dur(start_idx - 1, end_idx) <= max_duration
        if can_fwd:
            end_idx += 1
        elif can_bwd:
            start_idx -= 1
        else:
            break

    total_dur = _dur(start_idx, end_idx)
    if total_dur > max_duration:
        return None

    full_transcript_dur = sentences[-1].end - sentences[0].start
    if total_dur < min_duration and full_transcript_dur >= min_duration:
        return None

    selected = sentences[start_idx : end_idx + 1]
    return round(selected[0].start, 3), round(selected[-1].end, 3), selected


def compute_weighted_score(breakdown: ScoreBreakdown, config: AppConfig) -> int:
    """Compute weighted 0-100 score from the 5 rubric dimensions."""
    w = config.moments.scoring_weights
    weight_sum = (
        w.emotional_peak
        + w.quotable_statement
        + w.humor_payoff
        + w.self_contained
        + w.clean_boundaries
    )
    if weight_sum <= 0:
        return 50
    raw = (
        breakdown.emotional_peak * w.emotional_peak
        + breakdown.quotable_statement * w.quotable_statement
        + breakdown.humor_payoff * w.humor_payoff
        + breakdown.self_contained * w.self_contained
        + breakdown.clean_boundaries * w.clean_boundaries
    ) / weight_sum
    return max(0, min(100, int(round(raw))))


def _clamp_peak_window(
    clip_start: float,
    clip_end: float,
    peak_start: float,
    peak_end: float,
    min_sec: float = 2.0,
    max_sec: float = 3.0,
) -> Tuple[float, float]:
    """Ensure the cold-open peak window stays strictly inside [clip_start, clip_end] and 2-3s long."""
    clip_dur = max(0.5, clip_end - clip_start)
    target_len = min(max(peak_end - peak_start, min_sec), max_sec, clip_dur)

    ps = max(clip_start, min(peak_start, clip_end - target_len))
    pe = min(clip_end, ps + target_len)
    if pe - ps < min(min_sec, clip_dur):
        ps = max(clip_start, clip_end - min(min_sec, clip_dur))
        pe = clip_end
    return round(ps, 3), round(pe, 3)


def _chunk_sentences_by_window(
    sentences: List[Sentence],
    window_sec: float,
    overlap_sec: float,
) -> List[List[Sentence]]:
    """Split long transcripts into overlapping time windows for processing."""
    if not sentences:
        return []
    total_end = sentences[-1].end
    if total_end <= window_sec:
        return [sentences]

    windows: List[List[Sentence]] = []
    step = max(60.0, window_sec - overlap_sec)
    cursor = sentences[0].start
    while cursor < total_end:
        win_end = cursor + window_sec
        chunk = [s for s in sentences if s.end >= cursor and s.start <= win_end]
        if chunk:
            windows.append(chunk)
        if win_end >= total_end:
            break
        cursor += step
    return windows


def _overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    denom = max(1e-6, min(a_end - a_start, b_end - b_start))
    return inter / denom


def _heatmap_boost_for_span(
    start: float,
    end: float,
    heatmap: List[ dict ],
) -> float:
    """Return YouTube 'Most Replayed' viewer intensity [0.0..1.0] overlapping [start, end]."""
    if not heatmap:
        return 0.0
    overlapping = [
        float(h.get("value", 0.0))
        for h in heatmap
        if float(h.get("end_time", 0.0)) >= start and float(h.get("start_time", 0.0)) <= end
    ]
    return max(overlapping) if overlapping else 0.0


def find_best_moments(
    transcript: Transcript,
    metadata: VideoMetadata,
    person: str,
    llm: LLMClient,
    config: AppConfig,
) -> List[ClipCandidate]:
    """Identify, snap, score, and deduplicate up to N candidate clips, prioritizing Most Replayed heatmap peaks."""
    if not transcript.sentences:
        return []

    raw_candidates: List[ClipCandidate] = []

    # 1. If YouTube 'Most Replayed' viewer heatmap exists, directly extract the top Most Viewed peaks first!
    if metadata.heatmap:
        sorted_peaks = sorted(
            [h for h in metadata.heatmap if float(h.get("start_time", 0.0)) >= 15.0],
            key=lambda h: float(h.get("value", 0.0)),
            reverse=True,
        )
        for peak in sorted_peaks[: config.moments.max_candidates * 3]:
            h_start = float(peak.get("start_time", 0.0))
            h_end = float(peak.get("end_time", h_start + 35.0))
            h_val = float(peak.get("value", 0.0))
            if h_val < 0.35:
                continue
            center = (h_start + h_end) * 0.5
            target_s = max(0.0, center - 17.5)
            target_e = center + 17.5

            snapped = snap_to_sentence_boundaries(
                transcript.sentences,
                target_s,
                target_e,
                min_duration=config.moments.min_duration_sec,
                max_duration=config.moments.max_duration_sec,
            )
            if not snapped:
                continue
            s_start, s_end, selected_sents = snapped
            clip_text = " ".join(s.text for s in selected_sents).strip()
            if len(clip_text.split()) < 15:
                continue

            # Find strongest quote sentence inside this Most Replayed span
            best_sent = max(selected_sents, key=lambda s: len(s.text))
            heat_pct = int(round(min(99, 75 + h_val * 24)))
            breakdown = ScoreBreakdown(
                emotional_peak=heat_pct,
                quotable_statement=min(98, heat_pct + 1),
                humor_payoff=min(92, max(65, heat_pct - 8)),
                self_contained=92,
                clean_boundaries=95,
            )
            score = max(compute_weighted_score(breakdown, config), heat_pct)
            p_start, p_end = _clamp_peak_window(
                s_start,
                s_end,
                best_sent.start,
                best_sent.start + 2.5,
                min_sec=config.hook.cold_open_min_sec,
                max_sec=config.hook.cold_open_max_sec,
            )
            raw_candidates.append(
                ClipCandidate(
                    id="clip_tmp",
                    start=s_start,
                    end=s_end,
                    score=score,
                    score_breakdown=breakdown,
                    reason=f"YouTube Most Replayed viral peak ({int(round(h_val * 100))}% audience replay intensity).",
                    key_quote=best_sent.text.strip(),
                    transcript_text=clip_text,
                    peak_start=p_start,
                    peak_end=p_end,
                )
            )

    # 2. Also scan transcript windows for high-scoring emotional & quotable statements
    win_sec = config.moments.window_minutes * 60.0
    ov_sec = config.moments.window_overlap_minutes * 60.0
    windows = _chunk_sentences_by_window(transcript.sentences, win_sec, ov_sec)
    per_window_n = max(3, min(config.moments.max_candidates, int(round(config.moments.max_candidates * 1.6 / len(windows)))))

    for w_idx, chunk in enumerate(windows):
        lines = [
            f"[{s.start:.2f}-{s.end:.2f}] ({s.speaker}) {s.text}"
            for s in chunk
        ]
        user_prompt = (
            f"Subject: {person}\n"
            f"Show/Title: {metadata.show_name} - {metadata.title}\n"
            f"Propose up to {per_window_n} best vertical Short clips ({int(config.moments.min_duration_sec)}-"
            f"{int(config.moments.max_duration_sec)} seconds each) from this transcript:\n\n"
            + "\n".join(lines)
        )
        resp = llm.complete_json(
            namespace=f"moments_w{w_idx}",
            system_prompt=MOMENTS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        items = resp.get("candidates", []) if isinstance(resp, dict) else []
        for item in items:
            try:
                raw_start = float(item["start"])
                raw_end = float(item["end"])
            except (KeyError, TypeError, ValueError):
                continue

            snapped = snap_to_sentence_boundaries(
                transcript.sentences,
                raw_start,
                raw_end,
                min_duration=config.moments.min_duration_sec,
                max_duration=config.moments.max_duration_sec,
            )
            if not snapped:
                continue
            s_start, s_end, selected_sents = snapped
            clip_text = " ".join(s.text for s in selected_sents).strip()

            bd_raw = item.get("score_breakdown") or {}
            breakdown = ScoreBreakdown(
                emotional_peak=int(max(0, min(100, bd_raw.get("emotional_peak", item.get("score", 75))))),
                quotable_statement=int(max(0, min(100, bd_raw.get("quotable_statement", item.get("score", 75))))),
                humor_payoff=int(max(0, min(100, bd_raw.get("humor_payoff", 60)))),
                self_contained=int(max(0, min(100, bd_raw.get("self_contained", item.get("score", 80))))),
                clean_boundaries=int(max(0, min(100, bd_raw.get("clean_boundaries", 88)))),
            )
            base_score = compute_weighted_score(breakdown, config)
            # Boost score if this transcript moment aligns with a YouTube Most Replayed spike
            hm_val = _heatmap_boost_for_span(s_start, s_end, metadata.heatmap)
            score = min(99, int(round(base_score + hm_val * 10)))

            raw_p_start = float(item.get("peak_start", s_start + (s_end - s_start) * 0.45))
            raw_p_end = float(item.get("peak_end", raw_p_start + 2.5))
            p_start, p_end = _clamp_peak_window(
                s_start,
                s_end,
                raw_p_start,
                raw_p_end,
                min_sec=config.hook.cold_open_min_sec,
                max_sec=config.hook.cold_open_max_sec,
            )

            key_quote = str(item.get("key_quote") or "").strip()
            if not key_quote:
                key_quote = selected_sents[min(len(selected_sents) - 1, 1)].text

            raw_candidates.append(
                ClipCandidate(
                    id="clip_tmp",
                    start=s_start,
                    end=s_end,
                    score=score,
                    score_breakdown=breakdown,
                    reason=str(item.get("reason") or "Compelling standalone moment.").strip(),
                    key_quote=key_quote,
                    transcript_text=clip_text,
                    peak_start=p_start,
                    peak_end=p_end,
                )
            )

    raw_candidates.sort(key=lambda c: c.score, reverse=True)
    deduped: List[ClipCandidate] = []
    for cand in raw_candidates:
        if any(_overlap_ratio(cand.start, cand.end, kept.start, kept.end) > 0.35 for kept in deduped):
            continue
        cand.id = f"clip_{len(deduped) + 1:02d}"
        deduped.append(cand)
        if len(deduped) >= config.moments.max_candidates:
            break

    return deduped
