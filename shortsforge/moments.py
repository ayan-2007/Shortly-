"""Stage 3a: Best-moment discovery, mood detection, complete-thought snapping, and 0-100 scoring."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from shortsforge.config import AppConfig
from shortsforge.llm import LLMClient
from shortsforge.models import (
    ClipCandidate,
    ScoreBreakdown,
    Sentence,
    Transcript,
    VideoMetadata,
    WordTimestamp,
)


MOMENTS_SYSTEM_PROMPT = """You are an elite short-form documentary and podcast editor (Iman Gadzhi / Alex Hormozi / Diary of a CEO style).
Your job is to find ONLY the strongest standalone, high-retention viral moments from an interview, show, or podcast transcript.
QUALITY OVER QUANTITY: Do NOT include random conversational filler, sponsor reads, rambling tangents, or incomplete thoughts. Only propose clips that have real viral potential (a sharp hook, surprising insight/number/truth, emotional turning point, or complete story with a clear payoff).
Vary clip length naturally (15 to 120 seconds) depending strictly on when the story, explanation, or message genuinely starts and reaches its full payoff. NEVER cut off a speaker mid-thought.

Score each candidate 0-100 on these five criteria:
1. emotional_peak: genuine emotion, vulnerability, intensity, or dramatic shift
2. quotable_statement: surprising, memorable, contrarian, or highly quotable line
3. humor_payoff: authentic wit, surprising twist, or memorable punchline
4. self_contained: makes 100% complete sense on its own with zero prior context or unseen visual props
5. clean_boundaries: begins cleanly on a complete thought and ends cleanly after the final payoff/sentence boundary

CRITICAL RULES:
- Only use timestamps and quotes that ACTUALLY appear in the provided transcript.
- Start and end strictly on complete thought/sentence boundaries (never start mid-sentence and never end on dangling words like "so," or "and").
- Never include sponsor reads, ad plugs, or intro/outro housekeeping.
- Identify a 2.5-3.8 second `peak_start`..`peak_end` window inside the clip containing the single most intense/quotable crux moment or hook line (for the cold open).

Return strict JSON matching:
{
  "candidates": [
    {
      "start": 12.4,
      "end": 68.8,
      "score": 89,
      "score_breakdown": {
        "emotional_peak": 90,
        "quotable_statement": 94,
        "humor_payoff": 70,
        "self_contained": 92,
        "clean_boundaries": 95
      },
      "reason": "One crisp sentence explaining why this moment hooks viewers.",
      "key_quote": "Exact verbatim crux quote from the transcript.",
      "peak_start": 34.1,
      "peak_end": 37.2
    }
  ]
}"""


_CONTINUATION_START_RE = re.compile(
    r"^(?:and|but|so|because|or|then|yet|which|that|who|whom|whose|while|though|although|however|"
    r"meaning|therefore|moreover|plus|also|besides|well,|yeah,|okay,|um,|uh,|you know,)\b",
    re.IGNORECASE,
)

_DANGLING_END_WORDS = {
    "the", "a", "an", "to", "for", "of", "with", "in", "on", "at", "by", "from", "into", "onto",
    "and", "but", "or", "so", "because", "if", "when", "while", "as", "than", "that", "which", "who",
    "who's", "whose", "what", "where", "how", "why", "my", "your", "his", "her", "their", "our", "its",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "can", "could", "will", "would", "should", "might", "may", "i", "you", "we", "they", "he", "she",
    "it", "um", "uh", "like", "mean", "very", "really", "just", "didn't", "don't", "wasn't", "weren't",
}

_FILLER_LEAD_RE = re.compile(
    r"^(?:yeah|yes|no|well|so|but|and|okay|right|you know|i mean|in terms of|uh|um)[,\s]+",
    re.IGNORECASE,
)

_SPONSOR_CLIP_RE = re.compile(
    r"\b(?:shopify|checkout|promo\s+code|use\s+code|discount\s+code|free\s+trial|sponsor(?:ed|ship)?|"
    r"patreon|merch|squarespace|betterhelp|athletic\s+greens|ag1|nordvpn|expressvpn|draftkings|"
    r"manscaped|whoop|audible|hubspot|factor\s+meals|mint\s+mobile|brought\s+to\s+you\s+by|"
    r"link\s+in\s+(?:the\s+)?description|link\s+below|subscribe\s+to\s+(?:the|our|my)\s+channel|"
    r"hit\s+the\s+bell|like\s+and\s+subscribe)\b|[a-z0-9\-]+\.com/[a-z0-9_\-]+",
    re.IGNORECASE,
)

_PLEASANTRY_RE = re.compile(
    r"\b(?:thanks?\s+for\s+having\s+me|thank\s+you\s+so\s+much\s+for\s+coming|welcome\s+back\s+to\s+the\s+(?:show|podcast)|"
    r"pleasure\s+to\s+be\s+here|how\s+are\s+you\s+doing\s+today|great\s+to\s+see\s+you)\b",
    re.IGNORECASE,
)

_VISUAL_PROP_RE = re.compile(
    r"\b(?:there\s+she\s+is|there\s+he\s+is|look\s+at\s+this\s+(?:screen|chart|photo|picture|graph)|"
    r"on\s+the\s+screen\s+right\s+now|as\s+you\s+can\s+see\s+on\s+this\s+chart)\b",
    re.IGNORECASE,
)

_VIRAL_HOOK_PATTERNS = [
    r"\b(?:the\s+truth|the\s+reality|the\s+problem|the\s+secret|the\s+reason|the\s+formula|the\s+rule)\s+is\b",
    r"\b(?:nobody|no\s+one|most\s+people|everybody|everyone)\s+(?:tells|wants|knows|realizes|talks\s+about|ignores|understands)\b",
    r"\b(?:warning\s+sign|biggest\s+mistake|number\s+one|never\s+forget|changed\s+everything|changed\s+my\s+life)\b",
    r"\b(?:i\s+learned|i\s+realized|i\s+discovered|i\s+refused|i\s+almost\s+quit|here\s+is\s+why|here's\s+how)\b",
    r"(\$\d[\d,\.]*|\b\d+(?:\.\d+)?%|\b\d+\s+(?:million|billion|thousand|years|months|days|hours|rules|laws|languages|steps|things)\b)",
]


def _is_ad_or_housekeeping_sentence(sent: Sentence) -> bool:
    txt = sent.text.strip()
    if not txt:
        return False
    return bool(_SPONSOR_CLIP_RE.search(txt))


def _transcript_has_punctuation(sentences: List[Sentence]) -> bool:
    """Return True if the transcript contains normal sentence punctuation/capitalization (vs YouTube auto-captions)."""
    non_empty = [re.sub(r"^>>\s*", "", s.text.strip()).strip() for s in sentences if s.text.strip()]
    if not non_empty:
        return True
    punct_or_cap = sum(
        1
        for t in non_empty
        if re.search(r"[.!?][\"'\)\]]*$", t) or (t and t[0].isupper())
    )
    return (punct_or_cap / len(non_empty)) >= 0.20


def _is_continuation_start(sent: Sentence, has_punctuation: bool = True) -> bool:
    txt = re.sub(r"^>>\s*", "", sent.text.strip()).strip()
    if not txt:
        return False
    if has_punctuation and txt[0].islower():
        return True
    return bool(_CONTINUATION_START_RE.match(txt))


def _is_incomplete_end(sent: Sentence, has_punctuation: bool = True) -> bool:
    txt = sent.text.strip()
    if not txt:
        return False
    if txt.endswith((",", ";", ":", "-", "—", "...")):
        return True
    if has_punctuation and not re.search(r"[.!?][\"'\)\]]*$", txt):
        return True
    tokens = [re.sub(r"[^a-zA-Z']", "", t).lower() for t in txt.split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return True
    if tokens[-1] in _DANGLING_END_WORDS:
        return True
    # Detect trailing filler tail like "... Georgia. So," or "... right. So."
    if len(tokens) >= 2 and tokens[-1] in {"so", "and", "but", "uh", "um", "yeah", "right"}:
        return True
    return False


def snap_to_complete_message(
    sentences: List[Sentence],
    raw_start: float,
    raw_end: float,
    min_duration: float = 12.0,
    max_duration: float = 180.0,
    max_backward_expand_sec: float = 28.0,
    max_forward_expand_sec: float = 55.0,
) -> Optional[Tuple[float, float, List[Sentence]]]:
    """Snap to exact thought/message boundaries so clips NEVER cut off mid-sentence, mid-message, or include sponsor ads."""
    if not sentences:
        return None

    has_punct = _transcript_has_punctuation(sentences)
    start_idx = min(range(len(sentences)), key=lambda i: abs(sentences[i].start - raw_start))
    end_idx = min(range(start_idx, len(sentences)), key=lambda i: abs(sentences[i].end - raw_end))

    def _dur(si: int, ei: int) -> float:
        return sentences[ei].end - sentences[si].start

    # Trim any sponsor/ad sentences at the edges of the initial anchor
    while start_idx < end_idx and _is_ad_or_housekeeping_sentence(sentences[start_idx]):
        start_idx += 1
    while end_idx > start_idx and _is_ad_or_housekeeping_sentence(sentences[end_idx]):
        end_idx -= 1

    anchor_start_t = sentences[start_idx].start
    # 1. Walk backward to a clean thought/question start (never stepping into an ad sentence)
    while start_idx > 0:
        prev_s = sentences[start_idx - 1]
        cur_s = sentences[start_idx]
        if _is_ad_or_housekeeping_sentence(prev_s):
            break
        if _dur(start_idx - 1, end_idx) > max_duration:
            break
        if (anchor_start_t - prev_s.start) > max_backward_expand_sec:
            break
        gap = cur_s.start - prev_s.end
        prev_incomplete = _is_incomplete_end(prev_s, has_punctuation=has_punct)
        cur_continues = _is_continuation_start(cur_s, has_punctuation=has_punct)
        prev_is_short_question = prev_s.text.strip().endswith("?") and (prev_s.end - prev_s.start) <= 14.0
        same_speaker_tight = has_punct and (prev_s.speaker == cur_s.speaker) and gap < 0.75 and not re.search(r"[.!?]$", prev_s.text.strip())

        if prev_incomplete or cur_continues or prev_is_short_question or same_speaker_tight:
            start_idx -= 1
            if prev_is_short_question and not _is_continuation_start(prev_s, has_punctuation=has_punct):
                break
        else:
            break

    # If the first sentence is still an orphan lowercase tail fragment from a prior answer and the next sentence
    # starts cleanly, advance past the orphan fragment so the clip opens on a clean sentence boundary.
    if has_punct:
        while start_idx < end_idx and _dur(start_idx + 1, end_idx) >= min_duration:
            first_txt = re.sub(r"^>>\s*", "", sentences[start_idx].text.strip()).strip()
            if (first_txt and first_txt[0].islower()) or re.match(r"^(?:you know|yeah,\s*yeah|and\s+so\s+that's)\b", first_txt, re.I):
                start_idx += 1
            else:
                break

    anchor_end_t = sentences[end_idx].end
    # 2. Walk forward until the speaker actually completes their thought/payoff (never stepping into an ad)
    while end_idx + 1 < len(sentences):
        cur_s = sentences[end_idx]
        next_s = sentences[end_idx + 1]
        if _is_ad_or_housekeeping_sentence(next_s):
            break
        cand_dur = _dur(start_idx, end_idx + 1)
        if cand_dur > max_duration:
            break

        need_min = _dur(start_idx, end_idx) < min_duration
        cur_incomplete = _is_incomplete_end(cur_s, has_punctuation=has_punct)
        next_continues = _is_continuation_start(next_s, has_punctuation=has_punct)
        tight_pause = has_punct and (next_s.start - cur_s.end) < 0.55 and cur_s.speaker == next_s.speaker
        within_forward_budget = (next_s.end - anchor_end_t) <= max_forward_expand_sec

        if need_min or (within_forward_budget and (cur_incomplete or next_continues or tight_pause)):
            end_idx += 1
            if not need_min and not _is_incomplete_end(sentences[end_idx], has_punctuation=has_punct) and (
                end_idx + 1 >= len(sentences)
                or (
                    not _is_continuation_start(sentences[end_idx + 1], has_punctuation=has_punct)
                    and (not has_punct or (sentences[end_idx + 1].start - sentences[end_idx].end) >= 0.70)
                )
            ):
                break
        else:
            break

    # 3. Ensure minimum duration if transcript is long enough (without crossing into sponsor ads)
    while _dur(start_idx, end_idx) < min_duration:
        if (
            end_idx + 1 < len(sentences)
            and not _is_ad_or_housekeeping_sentence(sentences[end_idx + 1])
            and _dur(start_idx, end_idx + 1) <= max_duration
        ):
            end_idx += 1
        elif (
            start_idx - 1 >= 0
            and not _is_ad_or_housekeeping_sentence(sentences[start_idx - 1])
            and _dur(start_idx - 1, end_idx) <= max_duration
        ):
            start_idx -= 1
        else:
            break

    # 4. Trim any trailing incomplete/dangling sentence at the end if doing so keeps duration >= min_duration
    while end_idx > start_idx and _dur(start_idx, end_idx - 1) >= min_duration:
        last_s = sentences[end_idx]
        if _is_incomplete_end(last_s, has_punctuation=has_punct) or _VISUAL_PROP_RE.search(last_s.text):
            end_idx -= 1
        else:
            break

    selected = sentences[start_idx : end_idx + 1]
    if not selected:
        return None
    return round(selected[0].start, 3), round(selected[-1].end, 3), selected


def _snap_peak_to_words(
    words: List[WordTimestamp],
    clip_start: float,
    clip_end: float,
    raw_peak_start: float,
    raw_peak_end: float,
    min_sec: float = 2.4,
    max_sec: float = 3.8,
) -> Tuple[float, float]:
    """Align the crux hook window strictly to clean spoken word boundaries inside [clip_start, clip_end]."""
    ps, pe = _clamp_peak_window(clip_start, clip_end, raw_peak_start, raw_peak_end, min_sec, max_sec)
    span_words = [w for w in words if w.end >= clip_start and w.start <= clip_end]
    if not span_words:
        return ps, pe

    s_idx = min(range(len(span_words)), key=lambda i: abs(span_words[i].start - ps))
    e_idx = s_idx
    while e_idx + 1 < len(span_words) and (span_words[e_idx + 1].end - span_words[s_idx].start) <= max_sec:
        e_idx += 1
        dur = span_words[e_idx].end - span_words[s_idx].start
        if dur >= min_sec and re.search(r"[.!?,;:]$", span_words[e_idx].word.strip()):
            break
    while (span_words[e_idx].end - span_words[s_idx].start) < min_sec and s_idx > 0:
        if (span_words[e_idx].end - span_words[s_idx - 1].start) <= max_sec + 0.5:
            s_idx -= 1
        else:
            break

    return round(max(clip_start, span_words[s_idx].start - 0.04), 3), round(
        min(clip_end, span_words[e_idx].end + 0.08), 3
    )


def _pick_crux_sentence(selected_sents: List[Sentence], has_punctuation: Optional[bool] = None) -> Sentence:
    """Pick the most punchy, self-contained quote/crux sentence inside a clip (never an incomplete fragment)."""
    has_punct = _transcript_has_punctuation(selected_sents) if has_punctuation is None else has_punctuation

    def _sent_rank(s: Sentence) -> float:
        txt = re.sub(r"^>>\s*", "", s.text.strip()).strip()
        wc = len(txt.split())
        score = 0.0
        if 6 <= wc <= 26:
            score += 24.0
        elif wc < 5:
            score -= 22.0
        elif wc > 32:
            score -= 10.0

        if _is_incomplete_end(s, has_punctuation=has_punct):
            score -= 45.0
        if has_punct and txt and txt[0].islower():
            score -= 20.0
        if _FILLER_LEAD_RE.match(txt):
            score -= 22.0
        if txt.endswith("?"):
            score -= 16.0
        if _VISUAL_PROP_RE.search(txt):
            score -= 30.0

        hesitations = len(re.findall(r"\b(?:uh|um|like,\s|you know,\s)\b", txt, re.I))
        score -= hesitations * 12.0

        for pat in _VIRAL_HOOK_PATTERNS:
            if re.search(pat, txt, re.I):
                score += 18.0
        if re.search(r"(\$\d|\d+%|\b(?:million|billion|thousand|truth|never|always|secret|warning|problem|rule|formula|changed|mistake|future|price|painful|critical|performance|breakthrough|hardest|biggest|first|open)\b)", txt, re.I):
            score += 24.0
        score += min(16.0, wc * 0.75)
        return score

    return max(selected_sents, key=_sent_rank)


def evaluate_viral_potential(
    selected_sents: List[Sentence],
    clip_text: str,
    has_punctuation: Optional[bool] = None,
) -> Tuple[bool, ScoreBreakdown]:
    """Strictly evaluate whether a candidate clip is a genuine standalone viral moment vs random podcast filler."""
    if not selected_sents:
        return False, ScoreBreakdown(emotional_peak=30, quotable_statement=30, humor_payoff=30, self_contained=30, clean_boundaries=30)

    has_punct = _transcript_has_punctuation(selected_sents) if has_punctuation is None else has_punctuation
    words = clip_text.split()
    wc = len(words)
    if wc < 14:
        return False, ScoreBreakdown(emotional_peak=35, quotable_statement=35, humor_payoff=35, self_contained=35, clean_boundaries=35)

    # Hard rejection: sponsor reads, promo plugs, housekeeping
    if _SPONSOR_CLIP_RE.search(clip_text) or any(_is_ad_or_housekeeping_sentence(s) for s in selected_sents):
        return False, ScoreBreakdown(emotional_peak=20, quotable_statement=20, humor_payoff=20, self_contained=20, clean_boundaries=20)

    # Hard rejection: mostly small-talk pleasantries
    if _PLEASANTRY_RE.search(clip_text) and wc < 55:
        return False, ScoreBreakdown(emotional_peak=40, quotable_statement=40, humor_payoff=40, self_contained=45, clean_boundaries=50)

    lower = clip_text.lower()
    # Measure filler / hesitation density ("um", "uh", "you know", "sort of", "kind of", "yeah yeah")
    filler_hits = len(re.findall(r"\b(?:uh|um|er|ah)\b|\b(?:you know|i mean|sort of|kind of|yeah,?\s+yeah)\b", lower))
    filler_ratio = filler_hits / max(1, wc)

    # Detect visual-prop dependency ("there she is", "look at this chart")
    visual_prop_hits = len(_VISUAL_PROP_RE.findall(clip_text))

    # Reject rambling clips with heavy hesitation/stuttering or visual prop dependency
    if filler_ratio > 0.085 or (visual_prop_hits > 0 and filler_ratio > 0.04):
        return False, ScoreBreakdown(emotional_peak=52, quotable_statement=48, humor_payoff=45, self_contained=45, clean_boundaries=50)

    # Check boundary cleanliness (punctuation-aware so unpunctuated YouTube auto-captions are judged fairly)
    first_s = selected_sents[0]
    last_s = selected_sents[-1]
    starts_clean = not _is_continuation_start(first_s, has_punctuation=has_punct)
    ends_clean = not _is_incomplete_end(last_s, has_punctuation=has_punct)
    if has_punct and not ends_clean and len(selected_sents) <= 2:
        return False, ScoreBreakdown(emotional_peak=50, quotable_statement=50, humor_payoff=50, self_contained=45, clean_boundaries=40)

    # Positive viral signals across ALL niches (podcasts, tech/gaming, interviews, business, science, stories)
    hook_pattern_hits = sum(1 for pat in _VIRAL_HOOK_PATTERNS if re.search(pat, clip_text, re.I))
    number_hits = len(re.findall(r"\$\d[\d,\.]*|\b\d+(?:\.\d+)?%|\b\d+\s+(?:million|billion|thousand|years|days|hours|rules|steps|languages|things|dollars|bucks|frames|fps|games|people)\b", clip_text, re.I))
    impact_words = len(re.findall(
        r"\b(?:truth|secret|never|always|mistake|warning|problem|changed|million|billion|wealth|money|invest|dividend|"
        r"freedom|prison|jail|arrest|war|fear|danger|broke|quit|survive|matrix|system|power|rule|formula|discipline|"
        r"marriage|divorce|family|mother|father|death|died|built|company|profit|success|"
        r"future|price|cost|painful|critical|performance|hardware|software|breakthrough|hardest|biggest|crazy|impossible|"
        r"design|decision|challenge|vision|opportunity|industry|gamers|players|customers|users|experience|open|control|"
        r"important|different|better|excited|surprised|honest|strategy|product|category|pc|console|mobile|game|games)\b",
        lower,
    ))
    contrast_turns = len(re.findall(r"\b(?:because|instead|however|realized|discovered|suddenly|actually|basically|truth is|problem is|reason is|formula is|if you look|the thing is)\b", lower))

    best_sent = _pick_crux_sentence(selected_sents, has_punctuation=has_punct)
    crux_clean = (
        (not has_punct or not _is_incomplete_end(best_sent, has_punctuation=True))
        and not best_sent.text.strip().endswith("?")
        and len(best_sent.text.split()) >= 5
    )

    # Require genuine viral substance (at least one strong viral signal + clean crux quote)
    viral_signal_strength = hook_pattern_hits * 12 + number_hits * 10 + min(6, impact_words) * 5 + min(4, contrast_turns) * 5
    if not crux_clean or viral_signal_strength < 12:
        return False, ScoreBreakdown(emotional_peak=62, quotable_statement=60, humor_payoff=55, self_contained=62, clean_boundaries=65)

    penalty = int(round(filler_ratio * 220)) + visual_prop_hits * 12
    emo = max(55, min(98, 70 + min(5, impact_words) * 4 + contrast_turns * 3 + (4 if "!" in clip_text else 0) - penalty))
    quo = max(55, min(99, 72 + hook_pattern_hits * 8 + number_hits * 6 + min(4, impact_words) * 3 - penalty))
    hum = max(55, min(94, 67 + contrast_turns * 5 + hook_pattern_hits * 4 - penalty))
    self_c = max(55, min(98, (92 if starts_clean else 82) - visual_prop_hits * 15 - int(round(filler_ratio * 150))))
    bounds = max(55, min(99, (95 if (starts_clean and ends_clean) else (84 if (ends_clean or not has_punct) else 66))))

    return True, ScoreBreakdown(
        emotional_peak=emo,
        quotable_statement=quo,
        humor_payoff=hum,
        self_contained=self_c,
        clean_boundaries=bounds,
    )


def detect_clip_mood(
    clip_text: str,
    breakdown: ScoreBreakdown,
    video_title: str = "",
) -> str:
    """Classify the clip mood into 'tense' | 'emotional' | 'inspiring' | 'mysterious' for adaptive music."""
    combined = f"{video_title} {clip_text}".lower()
    emotional_hits = sum(
        1 for w in ("mother", "father", "mom", "dad", "family", "love", "heart", "cried", "tears", "died", "loss", "miss", "childhood", "divorce", "pain", "hurt", "grateful")
        if re.search(rf"\b{w}\b", combined)
    )
    tense_hits = sum(
        1 for w in ("arrest", "jail", "prison", "warning", "fail", "bankrupt", "broke", "threat", "enemy", "attack", "danger", "risk", "wrong", "trap", "mistake", "fight", "war", "fear")
        if re.search(rf"\b{w}\b", combined)
    )
    inspiring_hits = sum(
        1 for w in ("wealth", "grow", "invest", "build", "discipline", "habit", "success", "freedom", "dream", "focus", "win", "rich", "10x", "company", "entrepreneur")
        if re.search(rf"\b{w}\b", combined)
    )
    mysterious_hits = sum(
        1 for w in ("secret", "hidden", "nobody", "truth", "matrix", "lie", "illusion", "unexplained", "bizarre", "realized", "shocked")
        if re.search(rf"\b{w}\b", combined)
    )

    scores: Dict[str, float] = {
        "emotional": emotional_hits * 2.2 + (1.0 if breakdown.emotional_peak >= 85 else 0.0),
        "tense": tense_hits * 2.3 + (1.5 if breakdown.emotional_peak >= 82 else 0.2),
        "inspiring": inspiring_hits * 2.0 + (0.8 if breakdown.quotable_statement >= 88 else 0.0),
        "mysterious": mysterious_hits * 2.1,
    }
    best = max(scores.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0.5 else "tense"


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
    """Discover up to 15 max high-impact, complete-thought viral moments (strictly quality over quantity)."""
    if not transcript.sentences:
        return []

    raw_candidates: List[ClipCandidate] = []
    has_punct = _transcript_has_punctuation(transcript.sentences)
    max_cap = 15 if config.moments.max_candidates <= 0 else max(1, min(15, config.moments.max_candidates))
    min_viral_score = int(getattr(config.moments, "min_viral_score", 78))

    # 1. Evaluate YouTube 'Most Replayed' viewer heatmap peaks through strict viral content quality check
    if metadata.heatmap:
        sorted_peaks = sorted(
            [h for h in metadata.heatmap if float(h.get("start_time", 0.0)) >= 12.0],
            key=lambda h: float(h.get("value", 0.0)),
            reverse=True,
        )
        peak_pool = sorted_peaks[: max_cap * 2]
        for peak in peak_pool:
            h_start = float(peak.get("start_time", 0.0))
            h_end = float(peak.get("end_time", h_start + 35.0))
            h_val = float(peak.get("value", 0.0))
            if h_val < 0.28:
                continue

            target_s = max(0.0, h_start - 8.0)
            target_e = h_end + 12.0

            snapped = snap_to_complete_message(
                transcript.sentences,
                target_s,
                target_e,
                min_duration=config.moments.min_duration_sec,
                max_duration=config.moments.max_duration_sec,
            )
            if not snapped:
                continue
            s_start, s_end, selected_sents = snapped
            clip_text = " ".join(re.sub(r"^>>\s*", "", s.text).strip() for s in selected_sents).strip()

            viable, content_bd = evaluate_viral_potential(selected_sents, clip_text, has_punctuation=has_punct)
            if not viable:
                continue

            best_sent = _pick_crux_sentence(selected_sents, has_punctuation=has_punct)
            base_score = compute_weighted_score(content_bd, config)
            score = min(98, base_score + int(round(h_val * 6)))
            p_start, p_end = _snap_peak_to_words(
                transcript.words,
                s_start,
                s_end,
                best_sent.start,
                best_sent.start + 3.0,
                min_sec=config.hook.cold_open_min_sec,
                max_sec=config.hook.cold_open_max_sec,
            )
            mood = detect_clip_mood(clip_text, content_bd, metadata.title)
            clean_quote = re.sub(r"^>>\s*", "", best_sent.text.strip()).strip()
            raw_candidates.append(
                ClipCandidate(
                    id="clip_tmp",
                    start=s_start,
                    end=s_end,
                    score=score,
                    score_breakdown=content_bd,
                    reason=f"High-retention viral story & audience replay peak ({int(round(h_val * 100))}% replay intensity).",
                    key_quote=clean_quote,
                    transcript_text=clip_text,
                    peak_start=p_start,
                    peak_end=p_end,
                    mood=mood,
                )
            )

    # 2. Full-timeline scan across overlapping windows for standalone viral stories/insights
    win_sec = config.moments.window_minutes * 60.0
    ov_sec = config.moments.window_overlap_minutes * 60.0
    windows = _chunk_sentences_by_window(transcript.sentences, win_sec, ov_sec)
    per_window_n = max(3, min(max_cap, int(round(max_cap * 1.5 / max(1, len(windows))))))

    for w_idx, chunk in enumerate(windows):
        lines = [
            f"[{s.start:.2f}-{s.end:.2f}] ({s.speaker}) {s.text}"
            for s in chunk
        ]
        user_prompt = (
            f"Subject: {person}\n"
            f"Show/Title: {metadata.show_name} - {metadata.title}\n"
            f"Propose up to {per_window_n} best varying-length complete vertical Short clips "
            f"({int(config.moments.min_duration_sec)}-{int(config.moments.max_duration_sec)} seconds each, ending strictly when the speaker finishes their message) "
            f"from this transcript:\n\n"
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

            snapped = snap_to_complete_message(
                transcript.sentences,
                raw_start,
                raw_end,
                min_duration=config.moments.min_duration_sec,
                max_duration=config.moments.max_duration_sec,
            )
            if not snapped:
                continue
            s_start, s_end, selected_sents = snapped
            clip_text = " ".join(re.sub(r"^>>\s*", "", s.text).strip() for s in selected_sents).strip()

            viable, content_bd = evaluate_viral_potential(selected_sents, clip_text, has_punctuation=has_punct)
            if not viable:
                continue

            bd_raw = item.get("score_breakdown") or {}
            llm_bd = ScoreBreakdown(
                emotional_peak=int(max(0, min(100, bd_raw.get("emotional_peak", item.get("score", 78))))),
                quotable_statement=int(max(0, min(100, bd_raw.get("quotable_statement", item.get("score", 78))))),
                humor_payoff=int(max(0, min(100, bd_raw.get("humor_payoff", 66)))),
                self_contained=int(max(0, min(100, bd_raw.get("self_contained", item.get("score", 82))))),
                clean_boundaries=int(max(0, min(100, bd_raw.get("clean_boundaries", 90)))),
            )
            # Blend LLM score with deterministic viral content audit so weak/filler clips never get inflated scores
            breakdown = ScoreBreakdown(
                emotional_peak=int(round(0.55 * content_bd.emotional_peak + 0.45 * llm_bd.emotional_peak)),
                quotable_statement=int(round(0.55 * content_bd.quotable_statement + 0.45 * llm_bd.quotable_statement)),
                humor_payoff=int(round(0.55 * content_bd.humor_payoff + 0.45 * llm_bd.humor_payoff)),
                self_contained=int(round(0.60 * content_bd.self_contained + 0.40 * llm_bd.self_contained)),
                clean_boundaries=int(round(0.65 * content_bd.clean_boundaries + 0.35 * llm_bd.clean_boundaries)),
            )
            base_score = compute_weighted_score(breakdown, config)
            hm_val = _heatmap_boost_for_span(s_start, s_end, metadata.heatmap)
            score = min(98, int(round(base_score + hm_val * 6)))

            best_sent = _pick_crux_sentence(selected_sents, has_punctuation=has_punct)
            raw_p_start = float(item.get("peak_start", best_sent.start))
            raw_p_end = float(item.get("peak_end", raw_p_start + 3.0))
            p_start, p_end = _snap_peak_to_words(
                transcript.words,
                s_start,
                s_end,
                raw_p_start,
                raw_p_end,
                min_sec=config.hook.cold_open_min_sec,
                max_sec=config.hook.cold_open_max_sec,
            )

            key_quote = re.sub(r"^>>\s*", "", str(item.get("key_quote") or "").strip()).strip()
            quote_tokens = [re.sub(r"[^a-zA-Z']", "", t).lower() for t in key_quote.split() if t]
            if (
                not key_quote
                or len(quote_tokens) < 5
                or key_quote.endswith(("?", ",", ";", ":", "..."))
                or (quote_tokens and quote_tokens[-1] in _DANGLING_END_WORDS)
                or _FILLER_LEAD_RE.match(key_quote)
            ):
                key_quote = re.sub(r"^>>\s*", "", best_sent.text.strip()).strip()

            mood = detect_clip_mood(clip_text, breakdown, metadata.title)
            raw_candidates.append(
                ClipCandidate(
                    id="clip_tmp",
                    start=s_start,
                    end=s_end,
                    score=score,
                    score_breakdown=breakdown,
                    reason=str(item.get("reason") or "Complete high-impact viral insight with clear payoff.").strip(),
                    key_quote=key_quote,
                    transcript_text=clip_text,
                    peak_start=p_start,
                    peak_end=p_end,
                    mood=mood,
                )
            )

    raw_candidates.sort(key=lambda c: c.score, reverse=True)
    deduped: List[ClipCandidate] = []
    for cand in raw_candidates:
        if any(_overlap_ratio(cand.start, cand.end, kept.start, kept.end) > 0.25 for kept in deduped):
            continue
        # Quality over quantity: only include clips meeting min_viral_score (with a graceful >=72 floor for top picks on calmer technical interviews)
        if cand.score < min_viral_score and (len(deduped) >= 5 or cand.score < 72):
            if len(deduped) >= 1:
                continue
        cand.id = f"clip_{len(deduped) + 1:02d}"
        deduped.append(cand)
        if len(deduped) >= max_cap:
            break

    return deduped
