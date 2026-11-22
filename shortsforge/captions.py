"""Stage 6: Ultra-bold, dynamically animated word-by-word ASS captions with elastic pop-in, glow, emphasis, and presets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

from shortsforge.config import AppConfig, resolve_font
from shortsforge.models import WordTimestamp


@dataclass
class CaptionPreset:
    name: str
    primary_bgr: str      # Inactive word color (&HBBGGRR&)
    active_yellow: str    # Active yellow highlight
    active_red: str       # Active red highlight
    emphasis_bgr: str     # Numbers / power words accent color
    outline_bgr: str
    shadow_bgr: str
    outline_width: float
    shadow_depth: float
    blur_amount: float = 0.0
    tilt_deg: float = 0.0
    border_style: int = 1
    uppercase: bool = True


PRESETS: Dict[str, CaptionPreset] = {
    "neon": CaptionPreset(
        name="neon",
        primary_bgr="&H00FFFFFF",
        active_yellow="&H0000EAFF",
        active_red="&H003224FF",
        emphasis_bgr="&H00FFF000",
        outline_bgr="&H000B0612",
        shadow_bgr="&H70000000",
        outline_width=9.5,
        shadow_depth=6.5,
        blur_amount=1.2,
        tilt_deg=0.0,
        uppercase=True,
    ),
    "comic": CaptionPreset(
        name="comic",
        primary_bgr="&H00FAFCFF",
        active_yellow="&H0000DCFF",
        active_red="&H002020FF",
        emphasis_bgr="&H000099FF",
        outline_bgr="&H00000000",
        shadow_bgr="&H40000000",
        outline_width=11.5,
        shadow_depth=8.0,
        blur_amount=0.4,
        tilt_deg=2.2,
        uppercase=True,
    ),
    "minimal": CaptionPreset(
        name="minimal",
        primary_bgr="&H00F5F7FA",
        active_yellow="&H0032E0FF",
        active_red="&H004848FF",
        emphasis_bgr="&H0098F5B0",
        outline_bgr="&H00101218",
        shadow_bgr="&H78000000",
        outline_width=6.0,
        shadow_depth=4.0,
        blur_amount=0.3,
        tilt_deg=0.0,
        uppercase=False,
    ),
}

_STOPWORDS: Set[str] = {
    "the", "and", "that", "this", "with", "from", "have", "they", "were", "what",
    "when", "where", "your", "about", "there", "their", "would", "could", "should",
    "just", "into", "then", "than", "them", "some", "more", "very", "like", "been",
    "only", "also", "because", "really", "actually",
}

_NUMBER_RE = re.compile(r"^\$?\d[\d,\.]*[%kKmMbB]?$|^#\d+$")


def format_ass_timestamp(seconds: float) -> str:
    """Format seconds into ASS timestamp H:MM:SS.cs."""
    sec = max(0.0, seconds)
    total_cs = int(round(sec * 100))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def extract_emphasis_keywords(key_quote: str, hook_text: str) -> Set[str]:
    """Extract meaningful power/key words from the clip's key quote and hook."""
    combined = f"{key_quote} {hook_text}".lower()
    tokens = re.findall(r"[a-zA-Z']{4,}", combined)
    return {t for t in tokens if t not in _STOPWORDS}


def is_emphasized_word(word: str, keywords: Set[str]) -> bool:
    """Return True if a word is a number/currency/percentage or a key power word."""
    stripped = word.strip(".,!?;:\"'()[]")
    if not stripped:
        return False
    if _NUMBER_RE.match(stripped):
        return True
    if any(ch.isdigit() for ch in stripped):
        return True
    return stripped.lower() in keywords


def group_caption_words(
    words: List[WordTimestamp],
    words_per_group: int = 3,
    max_pause_sec: float = 0.55,
) -> List[List[WordTimestamp]]:
    """Chunk words into rhythmic 1-3 word groups, breaking naturally on pauses, punctuation, and width."""
    max_w = max(1, min(3, words_per_group))
    groups: List[List[WordTimestamp]] = []
    current: List[WordTimestamp] = []

    for w in words:
        if current:
            gap = w.start - current[-1].end
            projected_chars = sum(len(x.word) for x in current) + len(w.word) + len(current)
            if gap >= max_pause_sec or len(current) >= max_w or projected_chars > 22:
                groups.append(current)
                current = []
        current.append(w)
        if re.search(r"[.!?]$", w.word.strip()):
            groups.append(current)
            current = []

    if current:
        groups.append(current)
    return groups


def _escape_ass_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def build_ass_captions(
    clip_words: List[WordTimestamp],
    clip_start: float,
    time_offset: float,
    config: AppConfig,
    key_quote: str = "",
    hook_text: str = "",
) -> str:
    """Generate a high-impact, ultra-bold, elastically animated ASS subtitle document for 1080x1920."""
    preset = PRESETS.get(config.captions.preset, PRESETS["neon"])
    _, auto_family = resolve_font(config, bold=True, preferred_name=config.captions.font_name)
    font_family = config.captions.font_name or auto_family

    shifted_words: List[WordTimestamp] = []
    for w in clip_words:
        rel_start = (w.start - clip_start) + time_offset
        rel_end = (w.end - clip_start) + time_offset
        if rel_end <= time_offset:
            continue
        rel_start = max(time_offset, rel_start)
        rel_end = max(rel_start + 0.05, rel_end)
        shifted_words.append(
            WordTimestamp(word=w.word, start=round(rel_start, 3), end=round(rel_end, 3), probability=w.probability)
        )

    groups = group_caption_words(shifted_words, words_per_group=config.captions.words_per_group)
    keywords = extract_emphasis_keywords(key_quote, hook_text)

    active_color = preset.active_red if config.captions.active_color == "red" else preset.active_yellow
    emph_pct = int(round(config.captions.emphasis_scale * 100))

    play_x = config.reframe.output_width
    play_y = config.reframe.output_height
    center_x = play_x // 2
    center_y = play_y - config.captions.safe_margin_bottom - 60

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_x}
PlayResY: {play_y}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_family},{config.captions.font_size},{preset.primary_bgr},{active_color},{preset.outline_bgr},{preset.shadow_bgr},-1,0,0,0,100,100,2,0,{preset.border_style},{preset.outline_width},{preset.shadow_depth},5,{config.captions.safe_margin_left},{config.captions.safe_margin_right},{config.captions.safe_margin_bottom},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    dialogue_lines: List[str] = []
    prev_event_end = 0.0

    for g_idx, grp in enumerate(groups):
        tilt_tag = ""
        if preset.tilt_deg > 0:
            angle = preset.tilt_deg if (g_idx % 2 == 0) else -preset.tilt_deg
            tilt_tag = f"\\frz{angle:.1f}"

        blur_tag = f"\\blur{preset.blur_amount:.1f}" if preset.blur_amount > 0 else ""

        raw_total_chars = sum(len(w.word) for w in grp) + len(grp) - 1
        split_after_idx = 1 if (len(grp) == 3 and raw_total_chars > 15) else -1

        for active_idx, active_word in enumerate(grp):
            ev_start = max(prev_event_end, active_word.start)
            if active_idx + 1 < len(grp):
                ev_end = max(ev_start + 0.04, grp[active_idx + 1].start)
            else:
                ev_end = max(ev_start + 0.09, active_word.end)
            prev_event_end = ev_end

            if active_idx == 0:
                line_motion = (
                    f"{{\\an5{tilt_tag}\\move({center_x},{center_y + 24},{center_x},{center_y},0,95)"
                    f"\\fscx82\\fscy82\\t(0,65,\\fscx108\\fscy108)\\t(65,125,\\fscx100\\fscy100)}}"
                )
            else:
                line_motion = f"{{\\an5{tilt_tag}\\pos({center_x},{center_y})}}"

            styled_tokens: List[str] = []
            for idx, w in enumerate(grp):
                raw_tok = w.word.upper() if preset.uppercase else w.word
                tok = _escape_ass_text(raw_tok)
                emph = is_emphasized_word(w.word, keywords)

                if idx == active_idx:
                    if emph:
                        peak_s = min(155, emph_pct + 18)
                        settle_s = emph_pct
                        styled_tokens.append(
                            f"{{\\c{preset.emphasis_bgr}{blur_tag}"
                            f"\\fscx{peak_s}\\fscy{peak_s}"
                            f"\\t(0,95,\\fscx{settle_s}\\fscy{settle_s})}}{tok}{{\\r}}"
                        )
                    else:
                        styled_tokens.append(
                            f"{{\\c{active_color}{blur_tag}"
                            f"\\fscx130\\fscy130"
                            f"\\t(0,80,\\fscx108\\fscy108)}}{tok}{{\\r}}"
                        )
                else:
                    if emph:
                        styled_tokens.append(
                            f"{{\\c{preset.emphasis_bgr}\\fscx{emph_pct - 8}\\fscy{emph_pct - 8}}}{tok}{{\\r}}"
                        )
                    else:
                        styled_tokens.append(tok)

                if idx == split_after_idx:
                    styled_tokens.append("\\N")

            rendered = " ".join(styled_tokens).replace(" \\N ", "\\N")
            dialogue_lines.append(
                f"Dialogue: 0,{format_ass_timestamp(ev_start)},{format_ass_timestamp(ev_end)},Default,,0,0,0,,{line_motion}{rendered}"
            )

    return header + "\n".join(dialogue_lines) + "\n"


def write_ass_captions(
    output_ass: Path,
    clip_words: List[WordTimestamp],
    clip_start: float,
    time_offset: float,
    config: AppConfig,
    key_quote: str = "",
    hook_text: str = "",
) -> Path:
    """Write ASS subtitle file to disk."""
    output_ass.parent.mkdir(parents=True, exist_ok=True)
    content = build_ass_captions(
        clip_words=clip_words,
        clip_start=clip_start,
        time_offset=time_offset,
        config=config,
        key_quote=key_quote,
        hook_text=hook_text,
    )
    output_ass.write_text(content, encoding="utf-8")
    return output_ass
