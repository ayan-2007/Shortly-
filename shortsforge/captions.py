"""Stage 6: Iman Gadzhi / Alex Hormozi / Documentary-style kinetic typography & animated word-by-word ASS captions."""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from shortsforge.config import AppConfig, resolve_font
from shortsforge.models import WordTimestamp


@dataclass
class CaptionPreset:
    name: str
    primary_bgr: str      # Inactive word color (&HBBGGRR&)
    active_yellow: str    # Active gold/yellow highlight
    active_red: str       # Active crimson/red highlight
    emphasis_bgr: str     # Numbers / money / power words accent color
    outline_bgr: str
    shadow_bgr: str
    outline_width: float
    shadow_depth: float
    blur_amount: float = 0.0
    tilt_deg: float = 0.0
    border_style: int = 1
    uppercase: bool = True
    letter_spacing: int = 2


PRESETS: Dict[str, CaptionPreset] = {
    "iman": CaptionPreset(
        name="iman",
        primary_bgr="&H00FFFFFF",
        active_yellow="&H0028C8FF",  # Signature warm luxury gold/amber
        active_red="&H003838FF",     # Documentary crimson
        emphasis_bgr="&H0055F0FF",   # Radiant gold-cyan emphasis
        outline_bgr="&H0008080D",
        shadow_bgr="&H64000000",
        outline_width=8.5,
        shadow_depth=7.0,
        blur_amount=1.4,
        tilt_deg=0.0,
        uppercase=True,
        letter_spacing=3,
    ),
    "hormozi": CaptionPreset(
        name="hormozi",
        primary_bgr="&H00FFFFFF",
        active_yellow="&H0000EAFF",  # Electric Hormozi yellow
        active_red="&H002424FF",     # Viral punch red
        emphasis_bgr="&H0038F545",   # Cash green for numbers/power words
        outline_bgr="&H00000000",
        shadow_bgr="&H45000000",
        outline_width=11.0,
        shadow_depth=8.0,
        blur_amount=0.8,
        tilt_deg=2.0,
        uppercase=True,
        letter_spacing=2,
    ),
    "documentary": CaptionPreset(
        name="documentary",
        primary_bgr="&H00F8FAFC",
        active_yellow="&H0038C2FF",  # Editorial documentary gold
        active_red="&H003535FF",
        emphasis_bgr="&H0078E8FF",
        outline_bgr="&H000A0C12",
        shadow_bgr="&H60000000",
        outline_width=7.5,
        shadow_depth=7.5,
        blur_amount=1.2,
        tilt_deg=0.0,
        uppercase=True,
        letter_spacing=4,
    ),
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
        letter_spacing=2,
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
        letter_spacing=2,
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
        letter_spacing=2,
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


def _build_word_dialogue_events(
    words: List[WordTimestamp],
    src_start: float,
    dest_offset: float,
    preset: CaptionPreset,
    keywords: Set[str],
    active_color: str,
    emph_pct: int,
    words_per_group: int,
    center_x: int,
    center_y: int,
    min_floor: float = 0.0,
) -> Tuple[List[str], float]:
    """Emit Layer-0 animated word-by-word ASS Dialogue events aligned cleanly to dest_offset."""
    shifted_words: List[WordTimestamp] = []
    for w in words:
        rel_start = (w.start - src_start) + dest_offset
        rel_end = (w.end - src_start) + dest_offset
        if rel_end <= dest_offset:
            continue
        rel_start = max(dest_offset, rel_start)
        rel_end = max(rel_start + 0.05, rel_end)
        shifted_words.append(
            WordTimestamp(word=w.word, start=round(rel_start, 3), end=round(rel_end, 3), probability=w.probability)
        )

    groups = group_caption_words(shifted_words, words_per_group=words_per_group)
    dialogue_lines: List[str] = []
    prev_event_end = min_floor

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
                    f"{{\\an5{tilt_tag}\\fad(20,0)\\move({center_x},{center_y + 26},{center_x},{center_y},0,100)"
                    f"\\fscx84\\fscy84\\t(0,65,\\fscx107\\fscy107)\\t(65,125,\\fscx100\\fscy100)}}"
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
                        peak_s = min(158, emph_pct + 20)
                        settle_s = emph_pct
                        styled_tokens.append(
                            f"{{\\c{preset.emphasis_bgr}{blur_tag}\\fsp{preset.letter_spacing + 1}"
                            f"\\fscx{peak_s}\\fscy{peak_s}"
                            f"\\t(0,95,\\fscx{settle_s}\\fscy{settle_s})}}{tok}{{\\r}}"
                        )
                    else:
                        styled_tokens.append(
                            f"{{\\c{active_color}{blur_tag}\\fsp{preset.letter_spacing}"
                            f"\\fscx132\\fscy132"
                            f"\\t(0,82,\\fscx109\\fscy109)}}{tok}{{\\r}}"
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

    return dialogue_lines, prev_event_end


def _build_kinetic_hook_events(
    hook_text: str,
    hook_dur: float,
    preset: CaptionPreset,
    keywords: Set[str],
    active_color: str,
    center_x: int,
    top_y: int = 380,
) -> List[str]:
    """Build Iman Gadzhi / Hormozi / Documentary-style staggered kinetic title events (ambient glow + crisp staggered lines)."""
    cleaned = hook_text.strip().strip('"').strip("'")
    if not cleaned or hook_dur <= 0.2:
        return []

    wrapped = textwrap.wrap(cleaned.upper(), width=18)[:3]
    if not wrapped:
        return []

    events: List[str] = []
    line_spacing_px = 92
    total_h = (len(wrapped) - 1) * line_spacing_px
    base_y = top_y - total_h // 2

    for line_idx, line in enumerate(wrapped):
        line_y = base_y + line_idx * line_spacing_px
        stagger_sec = round(min(hook_dur * 0.35, line_idx * 0.11), 2)
        ev_start_ts = format_ass_timestamp(stagger_sec)
        ev_end_ts = format_ass_timestamp(hook_dur)
        is_punchline = (line_idx == len(wrapped) - 1 and len(wrapped) > 1)

        toks: List[str] = []
        plain_toks: List[str] = []
        for raw_w in line.split():
            tok = _escape_ass_text(raw_w)
            plain_toks.append(tok)
            is_emph = is_emphasized_word(raw_w, keywords) or is_punchline
            if is_emph:
                scale_tag = "\\fscx108\\fscy108" if is_punchline else "\\fscx104\\fscy104"
                toks.append(f"{{\\c{active_color}\\blur1.6{scale_tag}}}{tok}{{\\r}}")
            else:
                toks.append(tok)

        styled_line = " ".join(toks)
        plain_line = " ".join(plain_toks)

        # Layer 1: Soft Iman Gadzhi / Documentary ambient glow shadow backdrop
        glow_motion = (
            f"{{\\an5\\move({center_x},{line_y + 30},{center_x},{line_y},0,145)"
            f"\\blur8.5\\1c{active_color if is_punchline else '&H000000&'}\\3c&H000000&\\1a&H65&\\3a&H38&"
            f"\\fscx58\\fscy58\\t(0,105,\\fscx112\\fscy112)\\t(105,180,\\fscx100\\fscy100)"
            f"\\fad(65,130)}}"
        )
        events.append(
            f"Dialogue: 1,{ev_start_ts},{ev_end_ts},HookTitle,,0,0,0,,{glow_motion}{plain_line}"
        )

        # Layer 2: Razor-sharp kinetic foreground typography with overshoot pop & editorial tracking
        fg_motion = (
            f"{{\\an5\\move({center_x},{line_y + 30},{center_x},{line_y},0,145)"
            f"\\fsp{preset.letter_spacing + 1}"
            f"\\fscx58\\fscy58\\t(0,105,\\fscx114\\fscy114)\\t(105,180,\\fscx100\\fscy100)"
            f"\\fad(55,120)}}"
        )
        events.append(
            f"Dialogue: 2,{ev_start_ts},{ev_end_ts},HookTitle,,0,0,0,,{fg_motion}{styled_line}"
        )

    return events


def build_ass_captions(
    clip_words: List[WordTimestamp],
    clip_start: float,
    time_offset: float,
    config: AppConfig,
    key_quote: str = "",
    hook_text: str = "",
    hook_words: Optional[List[WordTimestamp]] = None,
    hook_start: float = 0.0,
    hook_banner_duration: float = 0.0,
) -> str:
    """Generate an Iman Gadzhi / Hormozi / Documentary-style elastically animated ASS subtitle document for 1080x1920."""
    preset = PRESETS.get(config.captions.preset, PRESETS["iman"])
    _, auto_family = resolve_font(config, bold=True, preferred_name=config.captions.font_name)
    font_family = config.captions.font_name or auto_family

    keywords = extract_emphasis_keywords(key_quote, hook_text)
    active_color = preset.active_red if config.captions.active_color == "red" else preset.active_yellow
    emph_pct = int(round(config.captions.emphasis_scale * 100))

    play_x = config.reframe.output_width
    play_y = config.reframe.output_height
    center_x = play_x // 2
    center_y = play_y - config.captions.safe_margin_bottom - 60

    hook_font_size = max(64, int(round(config.captions.font_size * 0.82)))
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_x}
PlayResY: {play_y}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_family},{config.captions.font_size},{preset.primary_bgr},{active_color},{preset.outline_bgr},{preset.shadow_bgr},-1,0,0,0,100,100,{preset.letter_spacing},0,{preset.border_style},{preset.outline_width},{preset.shadow_depth},5,{config.captions.safe_margin_left},{config.captions.safe_margin_right},{config.captions.safe_margin_bottom},1
Style: HookTitle,{font_family},{hook_font_size},&H00FFFFFF,{active_color},&H0006070C,&H78000000,-1,0,0,0,100,100,{preset.letter_spacing + 1},0,1,9.5,7.0,5,72,72,220,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    dialogue_lines: List[str] = []
    floor_t = 0.0

    # Cold-open hook spoken captions + Iman Gadzhi / Documentary staggered kinetic title (0:00 -> hook_banner_duration)
    if hook_banner_duration > 0.2:
        if hook_text:
            hk_evs = _build_kinetic_hook_events(
                hook_text=hook_text,
                hook_dur=hook_banner_duration,
                preset=preset,
                keywords=keywords,
                active_color=active_color,
                center_x=center_x,
                top_y=380,
            )
            dialogue_lines.extend(hk_evs)

        if hook_words:
            hk_lines, floor_t = _build_word_dialogue_events(
                words=hook_words,
                src_start=hook_start,
                dest_offset=0.0,
                preset=preset,
                keywords=keywords,
                active_color=active_color,
                emph_pct=emph_pct,
                words_per_group=config.captions.words_per_group,
                center_x=center_x,
                center_y=center_y,
                min_floor=0.0,
            )
            dialogue_lines.extend(hk_lines)

    main_lines, _ = _build_word_dialogue_events(
        words=clip_words,
        src_start=clip_start,
        dest_offset=time_offset,
        preset=preset,
        keywords=keywords,
        active_color=active_color,
        emph_pct=emph_pct,
        words_per_group=config.captions.words_per_group,
        center_x=center_x,
        center_y=center_y,
        min_floor=max(floor_t, time_offset),
    )
    dialogue_lines.extend(main_lines)

    return header + "\n".join(dialogue_lines) + "\n"


def write_ass_captions(
    output_ass: Path,
    clip_words: List[WordTimestamp],
    clip_start: float,
    time_offset: float,
    config: AppConfig,
    key_quote: str = "",
    hook_text: str = "",
    hook_words: Optional[List[WordTimestamp]] = None,
    hook_start: float = 0.0,
    hook_banner_duration: float = 0.0,
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
        hook_words=hook_words,
        hook_start=hook_start,
        hook_banner_duration=hook_banner_duration,
    )
    output_ass.write_text(content, encoding="utf-8")
    return output_ass
