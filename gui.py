#!/usr/bin/env python3
"""Launch the Shortly Local Studio GUI."""

from __future__ import annotations

import argparse

from shortsforge.gui_server import launch_gui


def main() -> int:
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
