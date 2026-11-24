#!/usr/bin/env python3
"""Launch the Shortly Local Studio GUI."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import shutil
import sys
from pathlib import Path

from shortsforge.gui_server import launch_gui


def _bootstrap_runtime() -> None:
    """Configure working directory, bundled tools PATH, .env, and windowed logging."""
    if getattr(sys, "frozen", False):
        app_dir = Path(sys.executable).resolve().parent
    else:
        app_dir = Path(__file__).resolve().parent

    os.chdir(app_dir)

    # Prepend bundled bin/ (ffmpeg, ffprobe) to PATH so yt-dlp and subprocesses find them
    bin_dir = app_dir / "bin"
    if bin_dir.is_dir():
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

    # Ensure .env exists on first run
    env_file = app_dir / ".env"
    env_example = app_dir / ".env.example"
    if not env_file.exists() and env_example.exists():
        try:
            shutil.copyfile(env_example, env_file)
        except OSError:
            pass

    # In PyInstaller --windowed mode, sys.stdout/sys.stderr are None; redirect to a log file
    if sys.stdout is None or sys.stderr is None:
        log_path = app_dir / "shortly.log"
        log_fh = open(log_path, "a", encoding="utf-8", buffering=1)
        if sys.stdout is None:
            sys.stdout = log_fh
        if sys.stderr is None:
            sys.stderr = log_fh


def main() -> int:
    multiprocessing.freeze_support()
    _bootstrap_runtime()

    parser = argparse.ArgumentParser(description="Launch the Shortly Local Studio GUI.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8780, help="Port number (default: 8780)")
    parser.add_argument("--config", default=None, help="Path to custom config.yaml")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    launch_gui(
        config_path=args.config,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
