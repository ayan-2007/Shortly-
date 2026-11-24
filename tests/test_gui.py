"""Tests for ShortsForge Local Studio GUI HTTP API and Range video streaming."""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import yaml

from shortsforge.gui_server import create_gui_server
from shortsforge.models import (
    ClipCandidate,
    ClipExportMetadata,
    HookSet,
    HookVariant,
    ProjectState,
    VideoMetadata,
)
from shortsforge.state import ProjectWorkspace


def test_gui_server_endpoints_and_range_streaming(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.safe_dump({
            "paths": {
                "projects_dir": str(tmp_path / "projects"),
                "outputs_dir": str(tmp_path / "outputs"),
            },
            "llm": {"provider": "mock"},
        }),
        encoding="utf-8",
    )

    # Seed a sample project + rendered output file
    ws = ProjectWorkspace(tmp_path / "projects" / "demo_ep", tmp_path / "outputs" / "demo_ep")
    ws.save_state(ProjectState(project_name="demo_ep", person="Ada Lovelace", source="sample.mp4"))
    ws.metadata_path.write_text(
        VideoMetadata(
            source="sample.mp4",
            source_type="local",
            title="Ada Interview",
            channel="History Archive",
            show_name="Innovators",
            year="1995",
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    cand = ClipCandidate(
        id="clip_01",
        start=0.0,
        end=25.0,
        score=91,
        reason="Sharp insight.",
        key_quote="Imagination drives discovery.",
        transcript_text="Imagination drives discovery in every machine we build.",
        peak_start=5.0,
        peak_end=7.5,
        hooks=HookSet(variants=[HookVariant(rank=1, text="Why imagination powers every machine.", predicted_retention=93)]),
    )
    ws.candidates_path.write_text(json.dumps([cand.model_dump()]), encoding="utf-8")

    fake_mp4 = ws.output_dir / "clip_01_demo.mp4"
    fake_mp4.write_bytes(b"0123456789ABCDEF" * 16)
    (ws.output_dir / "clip_01_demo_metadata.json").write_text(
        ClipExportMetadata(
            clip_id="clip_01",
            output_video="clip_01_demo.mp4",
            thumbnail="clip_01_demo_thumb.jpg",
            titles=["T1", "T2", "T3", "T4", "T5"],
            description="Source: History Archive",
            hashtags=["#shorts", "#a", "#b", "#c", "#d", "#e", "#f", "#g"],
            pinned_comment="Agree?",
            hook_text="Why imagination powers every machine.",
            start=0.0,
            end=25.0,
            score=91,
            lufs=-14.0,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    server = create_gui_server(config_path=cfg_file, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    try:
        # 1. Root HTML serves Shortly Studio UI + static logo assets + Live Post-Render Video Editor
        with urllib.request.urlopen(f"{base}/") as resp:
            html_body = resp.read().decode("utf-8")
            assert resp.status == 200
            assert "Shortly Studio" in html_body
            assert "pulse-bg-canvas" in html_body
            assert "shiny-cta" in html_body
            assert "/static/shortly-logo.png" in html_body
            assert "LIVE_FILTER_CSS_MAP" in html_body
            assert "/api/jobs/edit-clip" in html_body
            assert "Center White Box Card" in html_body
            assert "Exact Center (Middle of Screen)" in html_body

        for asset in ("/static/shortly-logo.png", "/static/shortly-icon.png", "/static/shortly-pulse.png", "/favicon.ico"):
            with urllib.request.urlopen(f"{base}{asset}") as resp:
                assert resp.status == 200
                assert len(resp.read()) > 100

        # 2. List projects
        with urllib.request.urlopen(f"{base}/api/projects") as resp:
            plist = json.loads(resp.read().decode("utf-8"))
            assert len(plist["projects"]) == 1
            assert plist["projects"][0]["name"] == "demo_ep"
            assert plist["projects"][0]["rendered_count"] == 1

        # 3. Project detail
        with urllib.request.urlopen(f"{base}/api/projects/demo_ep") as resp:
            detail = json.loads(resp.read().decode("utf-8"))
            assert detail["state"]["person"] == "Ada Lovelace"
            assert len(detail["outputs"]) == 1

        # 4. Save decisions via POST
        dec_payload = {
            "project_name": "demo_ep",
            "decisions": {
                "clip_01": {
                    "clip_id": "clip_01",
                    "approved": True,
                    "selected_hook_index": 0,
                    "edited_hook_text": "Her bold vision for computing.",
                }
            },
        }
        req = urllib.request.Request(
            f"{base}/api/projects/demo_ep/decisions",
            data=json.dumps(dec_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
        assert ws.decisions_path.exists()
        assert "Her bold vision for computing." in ws.decisions_path.read_text(encoding="utf-8")

        # 5. Config endpoint reports boolean key status without leaking secret strings
        with urllib.request.urlopen(f"{base}/api/config") as resp:
            cfg_resp = json.loads(resp.read().decode("utf-8"))
            assert isinstance(cfg_resp["keys"]["GEMINI_API_KEY"], bool)

        # 6. Video Range request returns HTTP 206 Partial Content & output URLs are cache-busted
        out0 = detail["outputs"][0]
        assert "?v=" in out0["video_url"]
        assert "?download=1&v=" in out0["download_url"]
        assert out0["download_url"] != out0["video_url"]

        r_req = urllib.request.Request(
            f"{base}/media/output/demo_ep/clip_01_demo.mp4",
            headers={"Range": "bytes=0-15"},
        )
        with urllib.request.urlopen(r_req) as resp:
            assert resp.status == 206
            assert resp.read() == b"0123456789ABCDEF"

        # 6b. Attachment download returns HTTP 200 full file with Content-Disposition & no-store
        dl_req = urllib.request.Request(
            f"{base}{out0['download_url']}",
            headers={"Range": "bytes=0-15"},
        )
        with urllib.request.urlopen(dl_req) as resp:
            assert resp.status == 200
            assert "attachment;" in (resp.headers.get("Content-Disposition") or "")
            assert "clip_01_demo.mp4" in (resp.headers.get("Content-Disposition") or "")
            assert "no-store" in (resp.headers.get("Cache-Control") or "")
            assert resp.read() == b"0123456789ABCDEF" * 16

        # 6c. HEAD request works cleanly for download & streaming URLs
        head_req = urllib.request.Request(f"{base}{out0['download_url']}", method="HEAD")
        with urllib.request.urlopen(head_req) as resp:
            assert resp.status == 200
            assert int(resp.headers.get("Content-Length", "0")) == len(b"0123456789ABCDEF" * 16)
            assert "attachment;" in (resp.headers.get("Content-Disposition") or "")

        # 6d. After post-render re-render updates the MP4, download_url cache tag changes & serves new bytes
        updated_bytes = b"RE_RENDERED_EDIT_BYTES!" * 20
        fake_mp4.write_bytes(updated_bytes)
        with urllib.request.urlopen(f"{base}/api/projects/demo_ep") as resp:
            detail_after = json.loads(resp.read().decode("utf-8"))
        out0_after = detail_after["outputs"][0]
        assert out0_after["download_url"] != out0["download_url"]
        with urllib.request.urlopen(f"{base}{out0_after['download_url']}") as resp:
            assert resp.status == 200
            assert resp.read() == updated_bytes

        # 7. Project delete endpoint removes project & output folders cleanly
        del_req = urllib.request.Request(
            f"{base}/api/projects/demo_ep/delete",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(del_req) as resp:
            assert resp.status == 200
        assert not ws.project_dir.exists()
        assert not ws.output_dir.exists()
    finally:
        server.shutdown()
        server.server_close()
