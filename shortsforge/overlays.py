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

    shadow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow_layer)
    sdraw.rounded_rectangle(
        [box_x + 6, box_y + 10, box_x + box_w + 6, box_y + box_h + 10],
        radius=28,
        fill=(0, 0, 0, 170),
    )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(14))
    img = Image.alpha_composite(img, shadow_layer)
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
    y0 = 1120

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
