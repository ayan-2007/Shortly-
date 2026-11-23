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
        prog="shortsforge",
        description="ShortsForge - Turn long interviews, shows, and podcasts into viral-style vertical Shorts.",
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
    p_new.add_argument("--max-clips", type=int, default=None, help="Maximum candidate clips to propose (default 10)")
    p_new.add_argument("--config", default=None, help="Path to custom config.yaml")
    p_new.add_argument("--force", action="store_true", help="Re-run all stages ignoring cached outputs")

    # `render` subcommand
    p_render = sub.add_parser("render", help="Render approved clips into 1080x1920 H.264 Shorts in outputs/<name>/")
    p_render.add_argument("name", help="Project name previously initialized with `new`")
    p_render.add_argument("--auto", action="store_true", help="Ignore review page and render all clips >= threshold")
    p_render.add_argument("--min-score", type=int, default=None, help="Override auto-approval score threshold (0-100)")
    p_render.add_argument("--mirror", action="store_true", help="Horizontally mirror the video footage")
    p_render.add_argument(
        "--filters",
        default="",
        help="1 or 2 comma-separated quality filters (sharp_hd,cinematic_grade,denoise_clean,warm_studio)",
    )
    p_render.add_argument("--config", default=None, help="Path to custom config.yaml")
    p_render.add_argument("--force", action="store_true", help="Re-render clips even if already rendered")

    # `gui` subcommand
    p_gui = sub.add_parser("gui", help="Launch the local ShortsForge Studio GUI in your browser")
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
            )
            return 0

        if args.command == "render":
            if args.mirror:
                config.export.mirror_video = True
            if args.filters:
                config.export.video_filters = [f.strip() for f in args.filters.split(",") if f.strip()][:2]
            run_render_project(
                project_name=args.name,
                config=config,
                auto=args.auto,
                min_score=args.min_score,
                force=args.force,
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
