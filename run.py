#!/usr/bin/env python3
"""ShortsForge CLI entrypoint.

Usage:
    python run.py new <name> --source <URL or local file> --person "<Full Name>"
    python run.py render <name>
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from shortsforge.config import load_config
from shortsforge.ffmpeg_utils import FFmpegExecutionError, FFmpegNotFoundError
from shortsforge.gui_server import launch_gui
from shortsforge.ingest import IngestError
from shortsforge.llm import LLMError
from shortsforge.pipeline import run_new_project, run_render_project


console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shortly",
        description="Shortly - Turn long interviews, shows, and podcasts into viral-style vertical Shorts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # `new` subcommand
    p_new = sub.add_parser("new", help="Ingest, transcribe, find moments, verify hooks, and generate review.html")
    p_new.add_argument("name", help="Project name (folder under projects/ and outputs/)")
    p_new.add_argument("--source", required=True, help="YouTube/media URL or local video file path")
    p_new.add_argument("--person", required=True, help='Full name of the speaker/subject (e.g. "Carl Sagan")')
    p_new.add_argument("--show", default="", help="Optional override for show/podcast name")
    p_new.add_argument("--year", default="", help="Optional override for original air/recording year")
    p_new.add_argument("--deceased", action="store_true", help="Enable respectful-tone guardrails for deceased figures")
    p_new.add_argument("--auto", action="store_true", help="Skip manual review and auto-approve clips >= score threshold")
    p_new.add_argument("--serve", action="store_true", help="Serve review.html on 127.0.0.1 for live auto-saving")
    p_new.add_argument("--min-score", type=int, default=None, help="Override auto-approval score threshold (0-100)")
    p_new.add_argument("--max-clips", type=int, default=None, help="Maximum candidate clips / parts to propose")
    p_new.add_argument("--mode", choices=["viral", "batch"], default="viral", help="Generation mode: viral clips or batch sequential parts")
    p_new.add_argument("--part-length", type=float, default=0.0, help="Part duration in seconds for batch mode (e.g. 300 for 5-minute parts)")
    p_new.add_argument("--min-length", type=float, default=None, help="Minimum viral clip length in seconds")
    p_new.add_argument("--max-length", type=float, default=None, help="Maximum viral clip length in seconds")
    p_new.add_argument("--aspect-ratio", choices=["9:16", "16:9"], default="9:16", help="Output video aspect ratio (9:16 or 16:9)")
    p_new.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (0.5 to 2.0)")
    p_new.add_argument("--config", default=None, help="Path to custom config.yaml")
    p_new.add_argument("--force", action="store_true", help="Re-run all stages ignoring cached outputs")

    # `render` subcommand
    p_render = sub.add_parser("render", help="Render approved clips into 1080x1920 or 1920x1080 H.264 MP4s in outputs/<name>/")
    p_render.add_argument("name", help="Project name previously initialized with `new`")
    p_render.add_argument("--auto", action="store_true", help="Ignore review page and render all clips >= threshold")
    p_render.add_argument("--min-score", type=int, default=None, help="Override auto-approval score threshold (0-100)")
    p_render.add_argument("--aspect-ratio", choices=["9:16", "16:9"], default=None, help="Override output aspect ratio (9:16 or 16:9)")
    p_render.add_argument("--speed", type=float, default=None, help="Override playback speed multiplier (0.5 to 2.0)")
    p_render.add_argument("--mirror", action="store_true", help="Horizontally mirror the video footage")
    p_render.add_argument(
        "--filters",
        default="",
        help="1 or 2 comma-separated quality filters (sharp_hd,cinematic_grade,denoise_clean,warm_studio)",
    )
    p_render.add_argument("--config", default=None, help="Path to custom config.yaml")
    p_render.add_argument("--force", action="store_true", help="Re-render clips even if already rendered")
    p_render.add_argument("--intro-box-text", default="", help="Enable custom intro text box with this message")
    p_render.add_argument("--intro-box-duration", type=float, default=None, help="Intro box duration in seconds (e.g. 1.0, 4.0, or 0 for full video)")
    p_render.add_argument("--intro-box-style", choices=["white_box", "white_pill", "dark_neon", "glass_dark", "gold_luxury", "red_alert"], default=None, help="Intro box card style")
    p_render.add_argument("--intro-box-font", choices=["bold_display", "impact", "clean_sans", "editorial_serif", "mono_tech"], default=None, help="Intro box caption font style")
    p_render.add_argument("--intro-box-color", choices=["auto", "black", "white", "purple", "gold", "red", "blue", "green"], default=None, help="Intro box text color")
    p_render.add_argument("--intro-box-size", choices=["small", "medium", "large", "xl"], default=None, help="Intro box text size")

    # `gui` subcommand
    p_gui = sub.add_parser("gui", help="Launch the local Shortly Studio GUI in your browser")
    p_gui.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    p_gui.add_argument("--port", type=int, default=8780, help="Port to bind (default: 8780)")
    p_gui.add_argument("--config", default=None, help="Path to custom config.yaml")
    p_gui.add_argument("--no-browser", action="store_true", help="Start server without auto-opening browser")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if args.command == "new":
            if args.min_length is not None:
                config.moments.min_duration_sec = max(5.0, float(args.min_length))
            if args.max_length is not None:
                config.moments.max_duration_sec = max(float(args.max_length), config.moments.min_duration_sec + 5.0)
            run_new_project(
                project_name=args.name,
                source=args.source,
                person=args.person,
                config=config,
                auto=args.auto,
                serve=args.serve,
                min_score=args.min_score,
                max_clips=args.max_clips,
                show_override=args.show,
                year_override=args.year,
                deceased=args.deceased,
                force=args.force,
                mode=args.mode,
                part_duration_sec=args.part_length,
                aspect_ratio=args.aspect_ratio,
                playback_speed=args.speed,
            )
            return 0

        if args.command == "render":
            if args.mirror:
                config.export.mirror_video = True
            if args.filters:
                config.export.video_filters = [f.strip() for f in args.filters.split(",") if f.strip()][:2]
            if args.intro_box_text:
                config.overlays.attention_card_enabled = True
                config.overlays.attention_card_text = args.intro_box_text.strip()
            if args.intro_box_duration is not None:
                config.overlays.attention_card_duration_sec = max(0.0, float(args.intro_box_duration))
            if args.intro_box_style is not None:
                config.overlays.attention_card_style = args.intro_box_style
            if args.intro_box_font is not None:
                config.overlays.attention_card_font_style = args.intro_box_font
            if args.intro_box_color is not None:
                config.overlays.attention_card_text_color = args.intro_box_color
            if args.intro_box_size is not None:
                config.overlays.attention_card_font_size = args.intro_box_size
            run_render_project(
                project_name=args.name,
                config=config,
                auto=args.auto,
                min_score=args.min_score,
                force=args.force,
                aspect_ratio=args.aspect_ratio,
                playback_speed=args.speed,
            )
            return 0

        if args.command == "gui":
            launch_gui(
                config_path=args.config,
                host=args.host,
                port=args.port,
                open_browser=not args.no_browser,
            )
            return 0
    except (IngestError, FFmpegNotFoundError, FFmpegExecutionError, LLMError, FileNotFoundError) as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
