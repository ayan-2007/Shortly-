"""Stage 7: Pillow RGBA overlay generators for hook banner, lower-third, persistent credit, archive tag, and closing card."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from shortsforge.config import AppConfig, resolve_font


def _load_font(config: AppConfig, size: int, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path, _ = resolve_font(config, bold=bold, preferred_name=config.captions.font_name if bold else "")
    if font_path and font_path.exists():
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _text_metrics(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int, int, int]:
    """Return (width, height, x_offset, y_offset) using exact font ink bounding box."""
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1], box[0], box[1]


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    w, h, _, _ = _text_metrics(draw, text, font)
    return w, h


def render_hook_banner_png(
    hook_text: str,
    output_path: Path,
    config: AppConfig,
) -> Path:
    """Render huge, high-contrast on-screen hook banner (<= 9 words) inside safe zone."""
    w, h = config.reframe.output_width, config.reframe.output_height
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font = _load_font(config, size=82, bold=True)
    text_clean = hook_text.strip().upper()
    wrapped = textwrap.wrap(text_clean, width=18)[:3]

    line_sizes = [_text_bbox(draw, line, font) for line in wrapped]
    max_lw = max((lw for lw, _ in line_sizes), default=200)
    line_h = max((lh for _, lh in line_sizes), default=70) + 18
    total_th = line_h * len(wrapped)

    pad_x, pad_y = 44, 30
    box_w = min(w - 110, max_lw + pad_x * 2)
    box_h = total_th + pad_y * 2
    box_x = (w - box_w) // 2
    box_y = int(h * 0.22)

    spad = 44
    sx0 = max(0, box_x + 6 - spad)
    sy0 = max(0, box_y + 10 - spad)
    sx1 = min(w, box_x + box_w + 6 + spad)
    sy1 = min(h, box_y + box_h + 10 + spad)
    shadow_patch = Image.new("RGBA", (sx1 - sx0, sy1 - sy0), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow_patch)
    sdraw.rounded_rectangle(
        [box_x + 6 - sx0, box_y + 10 - sy0, box_x + box_w + 6 - sx0, box_y + box_h + 10 - sy0],
        radius=28,
        fill=(0, 0, 0, 170),
    )
    shadow_patch = shadow_patch.filter(ImageFilter.GaussianBlur(14))
    img.alpha_composite(shadow_patch, (sx0, sy0))
    draw = ImageDraw.Draw(img)

    accent_border = (255, 224, 36, 255) if config.captions.active_color == "yellow" else (255, 62, 62, 255)
    draw.rounded_rectangle(
        [box_x, box_y, box_x + box_w, box_y + box_h],
        radius=28,
        fill=(14, 15, 22, 228),
        outline=accent_border,
        width=6,
    )

    cur_y = box_y + pad_y
    for idx, line in enumerate(wrapped):
        lw, _ = line_sizes[idx]
        tx = (w - lw) // 2
        draw.text(
            (tx, cur_y),
            line,
            font=font,
            fill=(255, 255, 255, 255) if idx == 0 else accent_border,
            stroke_width=5,
            stroke_fill=(0, 0, 0, 255),
        )
        cur_y += line_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    return output_path


def render_lower_third_png(
    person: str,
    show_name: str,
    year: str,
    output_path: Path,
    config: AppConfig,
) -> Path:
    """Render sleek broadcast lower-third with person's name, show, and year (above caption safe zone)."""
    w, h = config.reframe.output_width, config.reframe.output_height
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    name_font = _load_font(config, size=44, bold=True)
    sub_font = _load_font(config, size=30, bold=False)

    subtitle_parts = [p for p in (show_name.strip(), year.strip()) if p]
    subtitle = " • ".join(subtitle_parts) if subtitle_parts else "Interview Archive"

    nw, nh, nx_off, ny_off = _text_metrics(draw, person, name_font)
    sw, sh, sx_off, sy_off = _text_metrics(draw, subtitle, sub_font)

    pad_top, pad_bot, gap = 20, 20, 14
    card_w = min(w - 220, max(nw, sw) + 78)
    card_h = pad_top + nh + gap + sh + pad_bot
    x0 = 72
    y0 = int(round(h * 0.583)) if w <= h else int(round(h * 0.74))

    draw.rounded_rectangle(
        [x0, y0, x0 + card_w, y0 + card_h],
        radius=16,
        fill=(12, 14, 20, 218),
        outline=(255, 255, 255, 48),
        width=2,
    )
    accent = (255, 215, 40, 255) if config.overlays.archive_tag else (56, 189, 248, 255)
    draw.rounded_rectangle([x0, y0, x0 + 12, y0 + card_h], radius=6, fill=accent)

    draw.text((x0 + 32 - nx_off, y0 + pad_top - ny_off), person, font=name_font, fill=(255, 255, 255, 255))
    draw.text((x0 + 32 - sx_off, y0 + pad_top + nh + gap - sy_off), subtitle, font=sub_font, fill=(215, 224, 238, 240))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    return output_path


def render_persistent_credit_png(
    source_label: str,
    output_path: Path,
    config: AppConfig,
) -> Path:
    """Render persistent subtle top-corner 'Source: <channel/show>' credit and optional 'FROM THE ARCHIVE' tag."""
    w, h = config.reframe.output_width, config.reframe.output_height
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    credit_font = _load_font(config, size=26, bold=False)
    credit_text = f"Source: {source_label.strip() or 'Original Broadcast'}"
    if len(credit_text) > 46:
        credit_text = credit_text[:43].rstrip() + "..."
    cw, ch, cx_off, cy_off = _text_metrics(draw, credit_text, credit_font)

    cx, cy = 64, 130
    pill_h = ch + 26
    draw.rounded_rectangle(
        [cx, cy, cx + cw + 36, cy + pill_h],
        radius=12,
        fill=(10, 12, 16, 185),
    )
    draw.text((cx + 18 - cx_off, cy + (pill_h - ch) // 2 - cy_off), credit_text, font=credit_font, fill=(240, 244, 252, 235))

    if config.overlays.archive_tag:
        tag_font = _load_font(config, size=24, bold=True)
        tag_text = "FROM THE ARCHIVE"
        tw, th, tx_off, ty_off = _text_metrics(draw, tag_text, tag_font)
        tag_w = tw + 38
        tag_h = th + 26
        tx = w - tag_w - 64
        ty = 130
        draw.rounded_rectangle(
            [tx, ty, tx + tag_w, ty + tag_h],
            radius=10,
            fill=(38, 28, 16, 220),
            outline=(245, 200, 90, 245),
            width=3,
        )
        draw.text((tx + 19 - tx_off, ty + (tag_h - th) // 2 - ty_off), tag_text, font=tag_font, fill=(252, 218, 125, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    return output_path


def render_closing_card_png(
    output_path: Path,
    config: AppConfig,
    subtitle: Optional[str] = None,
) -> Path:
    """Render crisp full-frame closing card ('Follow for more') shown for < 1 second."""
    w, h = config.reframe.output_width, config.reframe.output_height
    img = Image.new("RGB", (w, h), (10, 11, 16))
    draw = ImageDraw.Draw(img)

    for step in range(8, 0, -1):
        rad = int(w * 0.14 * step)
        shade = 10 + (9 - step) * 3
        draw.ellipse(
            [w // 2 - rad, h // 2 - rad, w // 2 + rad, h // 2 + rad],
            fill=(shade, shade + 2, shade + 9),
        )

    title_font = _load_font(config, size=78, bold=True)
    sub_font = _load_font(config, size=34, bold=False)

    cta = config.overlays.closing_card_text.strip() or "Follow for more"
    tw, th = _text_bbox(draw, cta, title_font)
    tx = (w - tw) // 2
    ty = (h - th) // 2 - 20

    draw.rounded_rectangle(
        [w // 2 - 140, ty - 42, w // 2 + 140, ty - 30],
        radius=6,
        fill=(255, 215, 40),
    )
    draw.text((tx, ty), cta, font=title_font, fill=(255, 255, 255))

    if subtitle:
        sw, _ = _text_bbox(draw, subtitle, sub_font)
        draw.text(((w - sw) // 2, ty + th + 32), subtitle, font=sub_font, fill=(185, 195, 215))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    return output_path


_CARD_FONT_CANDIDATES = {
    "impact": ("impact", "Anton-Regular", "ariblk", "Montserrat-Black"),
    "clean_sans": ("segoeuib", "Inter-Bold", "arialbd", "LiberationSans-Bold"),
    "editorial_serif": ("georgiab", "timesbd", "Georgia", "DejaVuSerif-Bold"),
    "mono_tech": ("consolab", "courbd", "Consolas", "DejaVuSansMono-Bold"),
}


def _load_card_font(
    config: AppConfig,
    size: int,
    font_style: str = "bold_display",
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    style_key = (font_style or "bold_display").strip().lower()
    for cand_name in _CARD_FONT_CANDIDATES.get(style_key, ()):
        fpath, _ = resolve_font(config, bold=True, preferred_name=cand_name)
        if fpath and fpath.exists():
            try:
                return ImageFont.truetype(str(fpath), size=size)
            except OSError:
                continue
    return _load_font(config, size=size, bold=True)


def render_attention_card_png(
    card_text: str,
    output_path: Path,
    config: AppConfig,
    position: Optional[str] = None,
    style: Optional[str] = None,
    font_style: Optional[str] = None,
    text_color: Optional[str] = None,
    font_size_preset: Optional[str] = None,
    text_case: Optional[str] = None,
) -> Path:
    """Render the custom intro/center attention box with user message, box style, font, color, and size."""
    w, h = config.reframe.output_width, config.reframe.output_height
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    raw_text = (card_text or getattr(config.overlays, "attention_card_text", "") or "").strip()
    if not raw_text:
        raw_text = "WAIT FOR THIS..."

    card_pos = (position or getattr(config.overlays, "attention_card_position", "center") or "center").strip().lower()
    card_style = (style or getattr(config.overlays, "attention_card_style", "white_box") or "white_box").strip().lower()
    c_font_style = (font_style or getattr(config.overlays, "attention_card_font_style", "bold_display") or "bold_display").strip().lower()
    c_text_color = (text_color or getattr(config.overlays, "attention_card_text_color", "auto") or "auto").strip().lower()
    c_size_key = (font_size_preset or getattr(config.overlays, "attention_card_font_size", "medium") or "medium").strip().lower()
    c_case = (text_case or getattr(config.overlays, "attention_card_text_case", "normal") or "normal").strip().lower()

    if c_case == "upper":
        clean_text = raw_text.upper()
    elif c_case == "title":
        clean_text = raw_text.title()
    else:
        clean_text = raw_text

    is_landscape = w > h
    size_mult_map = {"small": 0.80, "medium": 1.0, "large": 1.22, "xl": 1.40}
    size_mult = size_mult_map.get(c_size_key, 1.0)
    base_size = 54 if card_style == "white_pill" else 62
    if is_landscape:
        base_size = int(round(base_size * 0.82))
    font_size = max(28, int(round(base_size * size_mult)))
    font = _load_card_font(config, size=font_size, font_style=c_font_style)

    base_wrap = (34 if card_style == "white_pill" else 30) if is_landscape else (24 if card_style == "white_pill" else 20)
    wrap_width = max(12, int(round(base_wrap / max(0.8, size_mult * 0.92))))
    wrapped = []
    for para in clean_text.splitlines():
        p_clean = para.strip()
        if p_clean:
            wrapped.extend(textwrap.wrap(p_clean, width=wrap_width))
    wrapped = wrapped[:5]
    if not wrapped:
        wrapped = [clean_text[:28]]

    line_metrics = [_text_metrics(draw, line, font) for line in wrapped]
    max_lw = max((m[0] for m in line_metrics), default=240)
    line_h = max((m[1] for m in line_metrics), default=48) + 18
    total_th = line_h * len(wrapped) - 18

    pad_x = 52 if card_style != "white_pill" else 44
    pad_y = 36 if card_style != "white_pill" else 24
    box_w = min(w - (220 if is_landscape else 90), max(420 if is_landscape else 360, max_lw + pad_x * 2))
    box_h = total_th + pad_y * 2
    box_x = (w - box_w) // 2

    cap_pos = str(getattr(config.captions, "position", "bottom") or "bottom").strip().lower()
    if card_pos == "upper":
        target_cy = int(h * 0.25)
    elif card_pos == "lower":
        target_cy = int(h * 0.70)
    else:
        target_cy = int(h * 0.36) if cap_pos == "center" else int(h * 0.46)

    box_y = max(70 if is_landscape else 110, min(h - box_h - (80 if is_landscape else 130), target_cy - box_h // 2))
    radius = min(box_h // 2, 48) if card_style == "white_pill" else 26

    # Soft drop shadow (blurred on tight ROI rather than full canvas for 8x faster rendering)
    spad = 48
    sx0 = max(0, box_x + 4 - spad)
    sy0 = max(0, box_y + 12 - spad)
    sx1 = min(w, box_x + box_w + 4 + spad)
    sy1 = min(h, box_y + box_h + 12 + spad)
    shadow_patch = Image.new("RGBA", (sx1 - sx0, sy1 - sy0), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow_patch)
    sdraw.rounded_rectangle(
        [box_x + 4 - sx0, box_y + 12 - sy0, box_x + box_w + 4 - sx0, box_y + box_h + 12 - sy0],
        radius=radius,
        fill=(0, 0, 0, 165),
    )
    shadow_patch = shadow_patch.filter(ImageFilter.GaussianBlur(16))
    img.alpha_composite(shadow_patch, (sx0, sy0))
    draw = ImageDraw.Draw(img)

    is_dark_box = card_style in ("dark_neon", "glass_dark", "gold_luxury", "red_alert")
    if card_style == "dark_neon":
        box_fill = (14, 16, 26, 244)
        border_col = (192, 75, 255, 255)
        text_primary = (255, 255, 255, 255)
        text_accent = (216, 118, 255, 255)
        border_w = 5
    elif card_style == "glass_dark":
        box_fill = (15, 23, 42, 215)
        border_col = (226, 232, 240, 145)
        text_primary = (248, 250, 252, 255)
        text_accent = (96, 195, 255, 255)
        border_w = 3
    elif card_style == "gold_luxury":
        box_fill = (18, 15, 11, 246)
        border_col = (245, 196, 81, 255)
        text_primary = (255, 252, 242, 255)
        text_accent = (250, 204, 85, 255)
        border_w = 4
    elif card_style == "red_alert":
        box_fill = (30, 10, 15, 246)
        border_col = (244, 63, 94, 255)
        text_primary = (255, 255, 255, 255)
        text_accent = (255, 115, 138, 255)
        border_w = 4
    else:
        box_fill = (255, 255, 255, 250)
        border_col = (230, 234, 242, 255)
        text_primary = (14, 16, 22, 255)
        if config.captions.active_color == "red":
            text_accent = (220, 28, 38, 255)
        elif config.captions.active_color == "yellow":
            text_accent = (205, 115, 0, 255)
        else:
            text_accent = (138, 34, 226, 255)
        border_w = 3

    draw.rounded_rectangle(
        [box_x, box_y, box_x + box_w, box_y + box_h],
        radius=radius,
        fill=box_fill,
        outline=border_col,
        width=border_w,
    )

    if card_style in ("white_box", "gold_luxury", "red_alert"):
        bar_w = min(140, box_w // 3)
        bar_x = (w - bar_w) // 2
        draw.rounded_rectangle(
            [bar_x, box_y + 10, bar_x + bar_w, box_y + 17],
            radius=4,
            fill=text_accent,
        )

    light_palette = {
        "black": (14, 16, 22, 255),
        "white": (255, 255, 255, 255),
        "purple": (138, 34, 226, 255),
        "gold": (194, 108, 0, 255),
        "red": (220, 28, 38, 255),
        "blue": (14, 116, 214, 255),
        "green": (16, 148, 72, 255),
    }
    dark_palette = {
        "black": (14, 16, 22, 255),
        "white": (255, 255, 255, 255),
        "purple": (216, 118, 255, 255),
        "gold": (250, 204, 65, 255),
        "red": (255, 86, 108, 255),
        "blue": (73, 184, 255, 255),
        "green": (74, 222, 128, 255),
    }
    custom_fill = (dark_palette if is_dark_box else light_palette).get(c_text_color)

    stroke_w = 0
    stroke_col = (0, 0, 0, 0)
    if c_text_color == "white" and not is_dark_box:
        stroke_w = 3
        stroke_col = (14, 16, 22, 235)
    elif c_text_color == "black" and is_dark_box:
        stroke_w = 3
        stroke_col = (255, 255, 255, 225)

    cur_y = box_y + pad_y
    for idx, line in enumerate(wrapped):
        lw, lh, lx_off, ly_off = line_metrics[idx]
        tx = (w - lw) // 2 - lx_off
        ty = cur_y - ly_off
        if custom_fill is not None:
            fill_col = custom_fill
        else:
            is_highlight_line = (len(wrapped) > 1 and idx == len(wrapped) - 1) or any(ch.isdigit() or ch in "$%" for ch in line)
            fill_col = text_accent if is_highlight_line else text_primary
        if stroke_w > 0:
            draw.text((tx, ty), line, font=font, fill=fill_col, stroke_width=stroke_w, stroke_fill=stroke_col)
        else:
            draw.text((tx, ty), line, font=font, fill=fill_col)
        cur_y += line_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    return output_path
