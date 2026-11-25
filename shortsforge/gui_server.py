"""Local web Studio GUI server for Shortly with 1-Click Link Autopilot."""

from __future__ import annotations

import json
import mimetypes
import os
import platform
import shutil
import subprocess
import threading
import time
import traceback
import uuid
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

import yaml

from shortsforge.config import AppConfig, apply_aspect_ratio, get_env_key, load_config
from shortsforge.ffmpeg_utils import extract_stream_info, probe_media
from shortsforge.ingest import _extract_url_info_cached, is_url
from shortsforge.models import (
    ClipCandidate,
    ClipExportMetadata,
    ProjectState,
    ReviewDecisionsFile,
    VideoMetadata,
)
from shortsforge.pipeline import (
    run_autopilot_from_source,
    run_edit_single_clip,
    run_new_project,
    run_render_project,
)
from shortsforge.review import load_decisions
from shortsforge.state import ProjectWorkspace


STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass
class JobRecord:
    job_id: str
    kind: str  # "autopilot" | "new" | "render"
    project_name: str
    status: str = "running"  # "running" | "completed" | "cancelled" | "error"
    message: str = "Starting..."
    logs: List[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    error: str = ""
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "project_name": self.project_name,
            "status": self.status,
            "message": self.message,
            "logs": self.logs[-120:],
            "elapsed_sec": round((self.finished_at or time.time()) - self.started_at, 1),
            "error": self.error,
            "cancelling": self.cancel_event.is_set(),
        }


class JobManager:
    """Thread-safe manager for background pipeline jobs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, JobRecord] = {}

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is not None and rec.status == "running":
                rec.cancel_event.set()
                rec.message = "Stopping after current clip finishes..."
                rec.logs.append("[CANCEL] Stop requested by user...")
            return rec

    def list_recent(self) -> List[Dict[str, Any]]:
        with self._lock:
            ordered = sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)
            return [j.to_dict() for j in ordered[:10]]

    def start_job(self, kind: str, project_name: str, target_fn) -> JobRecord:
        job_id = uuid.uuid4().hex[:10]
        rec = JobRecord(job_id=job_id, kind=kind, project_name=project_name)
        with self._lock:
            self._jobs[job_id] = rec

        def _runner() -> None:
            try:
                rec.logs.append(f"[{kind.upper()}] Started job...")
                target_fn(rec)
                if rec.cancel_event.is_set():
                    rec.status = "cancelled"
                    rec.message = "Stopped cleanly by user."
                    rec.logs.append(f"[{kind.upper()}] Stopped cleanly.")
                else:
                    rec.status = "completed"
                    rec.message = (
                        "Edits saved! Your updated Short is ready to preview & download."
                        if kind == "edit"
                        else "Completed! Your viral Shorts are ready."
                    )
                    rec.logs.append(f"[{kind.upper()}] Completed successfully.")
            except Exception as exc:  # noqa: BLE001
                rec.status = "error"
                rec.error = str(exc)
                rec.message = f"Failed: {exc}"
                rec.logs.append(f"[ERROR] {exc}")
                rec.logs.extend(traceback.format_exc().splitlines()[-6:])
            finally:
                rec.finished_at = time.time()

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        return rec


def _apply_runtime_overrides(cfg: AppConfig, body: Dict[str, Any]) -> None:
    """Apply runtime video editor overrides (filters, mirror, caption position, white box card) to an AppConfig."""
    if body.get("caption_preset") in ("iman", "hormozi", "documentary", "neon", "comic", "minimal"):
        cfg.captions.preset = body["caption_preset"]
    if body.get("active_color") in ("purple", "yellow", "red"):
        cfg.captions.active_color = body["active_color"]
    if body.get("caption_position") in ("bottom", "lower_mid", "center", "upper_mid", "top"):
        cfg.captions.position = body["caption_position"]
    if body.get("caption_custom_y") is not None:
        try:
            cfg.captions.custom_y = int(body["caption_custom_y"])
        except (TypeError, ValueError):
            cfg.captions.custom_y = None
    if body.get("hook_style") in ("cold_open", "freeze_narrator"):
        cfg.hook.style = body["hook_style"]
    if "hook_enabled" in body:
        cfg.hook.enabled = bool(body["hook_enabled"])
    if "show_opening_title" in body:
        cfg.hook.show_opening_title = bool(body["show_opening_title"])
    if "archive_tag" in body:
        cfg.overlays.archive_tag = bool(body["archive_tag"])
    if "mirror_video" in body:
        cfg.export.mirror_video = bool(body["mirror_video"])
    if isinstance(body.get("video_filters"), list):
        cfg.export.video_filters = [str(f).strip() for f in body["video_filters"] if str(f).strip()][:2]
    if "music_enabled" in body:
        cfg.audio.bg_music_enabled = bool(body["music_enabled"])
        cfg.audio.hook_bed_enabled = bool(body["music_enabled"])
    if "face_tracking_enabled" in body:
        cfg.reframe.face_tracking_enabled = bool(body["face_tracking_enabled"])
    if "attention_card_enabled" in body:
        cfg.overlays.attention_card_enabled = bool(body["attention_card_enabled"])
    if "attention_card_text" in body:
        cfg.overlays.attention_card_text = str(body.get("attention_card_text") or "").strip()
    if body.get("attention_card_position") in ("center", "upper", "lower"):
        cfg.overlays.attention_card_position = body["attention_card_position"]
    if body.get("attention_card_duration") is not None:
        try:
            cfg.overlays.attention_card_duration_sec = max(0.0, float(body["attention_card_duration"]))
        except (TypeError, ValueError):
            pass
    if body.get("attention_card_style") in ("white_box", "white_pill", "dark_neon", "glass_dark", "gold_luxury", "red_alert"):
        cfg.overlays.attention_card_style = body["attention_card_style"]
    if body.get("attention_card_font_style") in ("bold_display", "impact", "clean_sans", "editorial_serif", "mono_tech"):
        cfg.overlays.attention_card_font_style = body["attention_card_font_style"]
    if body.get("attention_card_text_color") in ("auto", "black", "white", "purple", "gold", "red", "blue", "green"):
        cfg.overlays.attention_card_text_color = body["attention_card_text_color"]
    if body.get("attention_card_font_size") in ("small", "medium", "large", "xl"):
        cfg.overlays.attention_card_font_size = body["attention_card_font_size"]
    if body.get("attention_card_text_case") in ("normal", "upper", "title"):
        cfg.overlays.attention_card_text_case = body["attention_card_text_case"]
    if body.get("aspect_ratio") in ("9:16", "16:9"):
        apply_aspect_ratio(cfg, body["aspect_ratio"])
    if body.get("playback_speed") is not None:
        try:
            cfg.export.playback_speed = max(0.5, min(2.0, float(body["playback_speed"])))
        except (TypeError, ValueError):
            pass
    if body.get("min_duration_sec") is not None:
        try:
            cfg.moments.min_duration_sec = max(5.0, min(600.0, float(body["min_duration_sec"])))
        except (TypeError, ValueError):
            pass
    if body.get("max_duration_sec") is not None:
        try:
            mx = max(10.0, min(900.0, float(body["max_duration_sec"])))
            cfg.moments.max_duration_sec = max(mx, cfg.moments.min_duration_sec + 5.0)
        except (TypeError, ValueError):
            pass


def _safe_resolve_under(base_dir: Path, rel_path: str) -> Optional[Path]:
    base = base_dir.resolve()
    target = (base / rel_path).resolve()
    try:
        target.relative_to(base)
        return target
    except ValueError:
        return None


def _list_projects(config: AppConfig) -> List[Dict[str, Any]]:
    root = config.projects_root()
    if not root.exists():
        return []

    items: List[Dict[str, Any]] = []
    for pdir in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not pdir.is_dir() or pdir.name.startswith("_"):
            continue
        state_file = pdir / "state.json"
        if not state_file.exists():
            continue
        try:
            state = ProjectState.model_validate_json(state_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue

        ws = ProjectWorkspace(pdir, config.outputs_root() / pdir.name)
        cand_count = 0
        if ws.candidates_path.exists():
            try:
                cand_count = len(json.loads(ws.candidates_path.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                pass

        decisions = load_decisions(ws, pdir.name)
        approved_count = sum(1 for d in decisions.decisions.values() if d.approved)

        rendered_mp4s = sorted(ws.output_dir.glob("clip_*.mp4")) if ws.output_dir.exists() else []
        title = ""
        show_name = state.show_override
        year = state.year_override
        if ws.metadata_path.exists():
            try:
                meta = VideoMetadata.model_validate_json(ws.metadata_path.read_text(encoding="utf-8"))
                title = meta.title
                show_name = meta.show_name
                year = meta.year
            except Exception:  # noqa: BLE001
                pass

        items.append({
            "name": pdir.name,
            "person": state.person,
            "title": title,
            "show_name": show_name,
            "year": year,
            "deceased": state.deceased,
            "candidate_count": cand_count,
            "approved_count": approved_count,
            "rendered_count": len(rendered_mp4s),
            "mode": state.mode,
            "aspect_ratio": state.aspect_ratio,
            "completed_steps": state.completed_steps,
            "warnings_count": len(state.warnings),
        })
    return items


def _get_project_detail(config: AppConfig, project_name: str) -> Optional[Dict[str, Any]]:
    pdir = config.projects_root() / project_name
    if not (pdir / "state.json").exists():
        return None
    ws = ProjectWorkspace(pdir, config.outputs_root() / project_name)
    state = ws.load_state()
    if state is None:
        return None

    metadata = None
    if ws.metadata_path.exists():
        metadata = VideoMetadata.model_validate_json(ws.metadata_path.read_text(encoding="utf-8")).model_dump()

    candidates: List[Dict[str, Any]] = []
    if ws.candidates_path.exists():
        raw_cands = json.loads(ws.candidates_path.read_text(encoding="utf-8"))
        for item in raw_cands:
            c = ClipCandidate.model_validate(item)
            d = c.model_dump()
            if c.thumbnail_path:
                try:
                    rel = os.path.relpath(Path(c.thumbnail_path), ws.project_dir).replace("\\", "/")
                    d["thumbnail_url"] = f"/media/project/{project_name}/{rel}"
                except ValueError:
                    d["thumbnail_url"] = ""
            else:
                d["thumbnail_url"] = ""
            d["duration"] = round(c.duration, 1)
            candidates.append(d)

    decisions = load_decisions(ws, project_name).model_dump()

    outputs: List[Dict[str, Any]] = []
    if ws.output_dir.exists():
        for mf in sorted(ws.output_dir.glob("*_metadata.json")):
            try:
                em = ClipExportMetadata.model_validate_json(mf.read_text(encoding="utf-8"))
                ed = em.model_dump()
                vid_p = ws.output_dir / em.output_video
                th_p = ws.output_dir / em.thumbnail
                if vid_p.exists():
                    v_st = vid_p.stat()
                    v_tag = f"{v_st.st_mtime_ns}_{v_st.st_size}"
                    v_size = v_st.st_size
                else:
                    v_tag = "0"
                    v_size = 0
                if th_p.exists():
                    t_st = th_p.stat()
                    t_tag = f"{t_st.st_mtime_ns}_{t_st.st_size}"
                else:
                    t_tag = "0"
                q_proj = quote(project_name, safe="")
                q_vid = quote(em.output_video, safe="")
                q_thumb = quote(em.thumbnail, safe="")
                ed["video_url"] = f"/media/output/{q_proj}/{q_vid}?v={v_tag}"
                ed["download_url"] = f"/media/output/{q_proj}/{q_vid}?download=1&v={v_tag}"
                ed["thumbnail_url"] = f"/media/output/{q_proj}/{q_thumb}?v={t_tag}"
                ed["file_size_bytes"] = v_size
                outputs.append(ed)
            except Exception:  # noqa: BLE001
                continue

    sources_txt = (ws.output_dir / "sources.txt").read_text(encoding="utf-8") if (ws.output_dir / "sources.txt").exists() else ""
    warnings_txt = (ws.output_dir / "warnings.txt").read_text(encoding="utf-8") if (ws.output_dir / "warnings.txt").exists() else ""

    return {
        "name": project_name,
        "state": state.model_dump(),
        "metadata": metadata,
        "candidates": candidates,
        "decisions": decisions,
        "outputs": outputs,
        "sources_txt": sources_txt,
        "warnings_txt": warnings_txt,
        "auto_threshold": config.review.auto_score_threshold,
        "max_hook_words": config.hook.max_words,
        "output_dir": str(ws.output_dir),
    }


def _update_env_file(root_dir: Path, updates: Dict[str, str]) -> None:
    env_path = root_dir / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    remaining = {k: v.strip() for k, v in updates.items() if v and v.strip()}
    new_lines: List[str] = []

    for line in lines:
        replaced = False
        for k in list(remaining.keys()):
            if line.strip().startswith(f"{k}="):
                new_lines.append(f"{k}={remaining.pop(k)}")
                replaced = True
                break
        if not replaced:
            new_lines.append(line)

    for k, v in remaining.items():
        new_lines.append(f"{k}={v}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    for k, v in updates.items():
        if v and v.strip():
            os.environ[k] = v.strip()


def _open_native_file_dialog() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="Select Interview / Show / Podcast Video",
            filetypes=[
                ("Video Files", "*.mp4 *.mov *.mkv *.webm *.avi *.m4v"),
                ("All Files", "*.*"),
            ],
        )
        root.destroy()
        return selected or ""
    except Exception:  # noqa: BLE001
        return ""


def _open_folder_in_os(folder: Path) -> bool:
    if not folder.exists():
        return False
    sys_name = platform.system().lower()
    try:
        if sys_name == "windows":
            os.startfile(str(folder))  # type: ignore[attr-defined]
        elif sys_name == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
        return True
    except Exception:  # noqa: BLE001
        return False


def create_gui_server(
    config_path: Optional[Path | str] = None,
    host: str = "127.0.0.1",
    port: int = 8780,
) -> ThreadingHTTPServer:
    """Create and return the configured ThreadingHTTPServer instance."""
    cfg_path = Path(config_path).resolve() if config_path else (Path.cwd() / "config.yaml").resolve()
    jobs = JobManager()

    class StudioHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _cfg(self) -> AppConfig:
            return load_config(cfg_path if cfg_path.exists() else None)

        def _send_json(self, status: int, data: Any) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            return json.loads(raw) if raw.strip() else {}

        def _serve_file_with_range(
            self,
            file_path: Path,
            no_cache: bool = False,
            as_attachment: bool = False,
            head_only: bool = False,
        ) -> None:
            if not file_path.is_file():
                self.send_error(404, "File not found")
                return
            file_size = file_path.stat().st_size
            ctype, _ = mimetypes.guess_type(str(file_path))
            ctype = "application/octet-stream" if as_attachment else (ctype or "application/octet-stream")
            range_hdr = None if as_attachment else self.headers.get("Range")

            def _send_cache_headers() -> None:
                if no_cache or as_attachment:
                    self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Expires", "0")

            def _send_attachment_headers() -> None:
                if as_attachment:
                    safe_ascii = file_path.name.encode("ascii", "ignore").decode("ascii").replace('"', "_") or "short.mp4"
                    quoted_utf8 = quote(file_path.name, safe="")
                    self.send_header(
                        "Content-Disposition",
                        f"attachment; filename=\"{safe_ascii}\"; filename*=UTF-8''{quoted_utf8}",
                    )
                    self.send_header("X-Content-Type-Options", "nosniff")

            try:
                if range_hdr and range_hdr.startswith("bytes="):
                    spec = range_hdr.replace("bytes=", "").split(",")[0].strip()
                    parts = spec.split("-")
                    start = int(parts[0]) if parts[0] else 0
                    end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
                    start = max(0, min(start, file_size - 1))
                    end = max(start, min(end, file_size - 1))
                    length = end - start + 1

                    self.send_response(206)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                    self.send_header("Content-Length", str(length))
                    _send_cache_headers()
                    self.end_headers()
                    if head_only:
                        return
                    with file_path.open("rb") as fh:
                        fh.seek(start)
                        remaining = length
                        while remaining > 0:
                            chunk = fh.read(min(65536, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                    return

                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(file_size))
                _send_attachment_headers()
                _send_cache_headers()
                self.end_headers()
                if head_only:
                    return
                with file_path.open("rb") as fh:
                    while True:
                        chunk = fh.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return

        def _handle_media_request(self, path: str, query: str, head_only: bool = False) -> bool:
            if path.startswith("/media/project/"):
                rest = path.split("/media/project/", 1)[1]
                parts = rest.split("/", 1)
                if len(parts) != 2:
                    self.send_error(400)
                    return True
                proj_name, rel = parts
                target = _safe_resolve_under(self._cfg().projects_root() / proj_name, rel)
                if target is None:
                    self.send_error(403)
                    return True
                self._serve_file_with_range(target, no_cache=True, head_only=head_only)
                return True

            if path.startswith("/media/output/"):
                rest = path.split("/media/output/", 1)[1]
                parts = rest.split("/", 1)
                if len(parts) != 2:
                    self.send_error(400)
                    return True
                proj_name, rel = parts
                target = _safe_resolve_under(self._cfg().outputs_root() / proj_name, rel)
                if target is None:
                    self.send_error(403)
                    return True
                qs = parse_qs(query)
                as_attachment = qs.get("download", ["0"])[0].lower() in ("1", "true", "yes")
                self._serve_file_with_range(
                    target,
                    no_cache=True,
                    as_attachment=as_attachment,
                    head_only=head_only,
                )
                return True

            return False

        def do_HEAD(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if self._handle_media_request(path, parsed.query, head_only=True):
                return
            if path.startswith("/static/"):
                rel = path.split("/static/", 1)[1].lstrip("/")
                target = _safe_resolve_under(STATIC_DIR, rel)
                if target is None or not target.is_file():
                    self.send_error(404, "Static asset not found")
                    return
                self._serve_file_with_range(target, no_cache=True, head_only=True)
                return
            self.send_error(404)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)

            if path in ("/", "/index.html"):
                html_path = STATIC_DIR / "index.html"
                if not html_path.exists():
                    self.send_error(404, "GUI index.html missing")
                    return
                content = html_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(content)
                return

            if path == "/favicon.ico":
                icon_path = STATIC_DIR / "shortly-pulse.png"
                if icon_path.is_file():
                    self._serve_file_with_range(icon_path)
                    return
                self.send_error(404)
                return

            if path.startswith("/static/"):
                rel = path.split("/static/", 1)[1].lstrip("/")
                target = _safe_resolve_under(STATIC_DIR, rel)
                if target is None or not target.is_file():
                    self.send_error(404, "Static asset not found")
                    return
                self._serve_file_with_range(target, no_cache=True)
                return

            if path == "/api/projects":
                self._send_json(200, {"projects": _list_projects(self._cfg())})
                return

            if path.startswith("/api/projects/"):
                proj_name = path.split("/api/projects/", 1)[1].strip("/")
                detail = _get_project_detail(self._cfg(), proj_name)
                if detail is None:
                    self._send_json(404, {"error": f"Project '{proj_name}' not found"})
                else:
                    self._send_json(200, detail)
                return

            if path == "/api/jobs":
                self._send_json(200, {"jobs": jobs.list_recent()})
                return

            if path.startswith("/api/jobs/"):
                job_id = path.split("/api/jobs/", 1)[1].strip("/")
                rec = jobs.get(job_id)
                if rec is None:
                    self._send_json(404, {"error": "Job not found"})
                else:
                    self._send_json(200, rec.to_dict())
                return

            if path == "/api/config":
                cfg = self._cfg()
                self._send_json(200, {
                    "config": cfg.model_dump(),
                    "keys": {
                        "GEMINI_API_KEY": bool(get_env_key("GEMINI_API_KEY")),
                        "ANTHROPIC_API_KEY": bool(get_env_key("ANTHROPIC_API_KEY")),
                        "OPENAI_API_KEY": bool(get_env_key("OPENAI_API_KEY")),
                    },
                })
                return

            if self._handle_media_request(path, parsed.query, head_only=False):
                return

            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)

            try:
                body = self._read_json()
            except Exception as exc:  # noqa: BLE001
                self._send_json(400, {"error": f"Invalid JSON: {exc}"})
                return

            # Probe URL or local video metadata for instant part-count calculator
            if path == "/api/probe-source":
                source = str(body.get("source") or "").strip()
                if not source:
                    self._send_json(400, {"error": "Source URL or file path is required."})
                    return
                cfg = self._cfg()
                try:
                    if is_url(source):
                        info = _extract_url_info_cached(source, cfg)
                        self._send_json(200, {
                            "title": str(info.get("title") or ""),
                            "duration": float(info.get("duration") or 0.0),
                            "width": int(info.get("width") or 1920),
                            "height": int(info.get("height") or 1080),
                        })
                        return
                    src_p = Path(source).expanduser()
                    if not src_p.is_file():
                        self._send_json(400, {"error": f"File not found: {source}"})
                        return
                    sinfo = extract_stream_info(probe_media(src_p))
                    self._send_json(200, {
                        "title": src_p.stem,
                        "duration": float(sinfo.get("duration") or 0.0),
                        "width": int(sinfo.get("width") or 1920),
                        "height": int(sinfo.get("height") or 1080),
                    })
                    return
                except Exception as exc:  # noqa: BLE001
                    self._send_json(400, {"error": str(exc)})
                    return

            # 1-Click Autopilot from just a URL or file path!
            if path == "/api/jobs/autopilot":
                source = str(body.get("source") or "").strip()
                if not source:
                    self._send_json(400, {"error": "Paste a YouTube link or video file path first."})
                    return

                cfg = self._cfg()
                _apply_runtime_overrides(cfg, body)
                mode = "batch" if str(body.get("mode") or "viral").strip().lower() == "batch" else "viral"
                try:
                    part_duration_sec = max(0.0, float(body.get("part_duration_sec") or 0.0))
                except (TypeError, ValueError):
                    part_duration_sec = 0.0
                aspect_ratio = body.get("aspect_ratio") if body.get("aspect_ratio") in ("9:16", "16:9") else None
                raw_max = body.get("max_clips")
                try:
                    raw_max_i = int(raw_max) if raw_max is not None else (0 if mode == "batch" else 12)
                except (TypeError, ValueError):
                    raw_max_i = 0 if mode == "batch" else 12
                if mode == "batch":
                    max_clips = 0 if raw_max_i <= 0 else max(1, min(50, raw_max_i))
                else:
                    max_clips = 12 if raw_max_i <= 0 else max(1, min(15, raw_max_i))
                force = bool(body.get("force", False))

                def _run_auto(rec: JobRecord) -> None:
                    def _prog(msg: str) -> None:
                        rec.message = msg
                        rec.logs.append(msg)

                    def _on_proj(pname: str) -> None:
                        rec.project_name = pname

                    proj_name, exported = run_autopilot_from_source(
                        source=source,
                        config=cfg,
                        max_clips=max_clips,
                        force=force,
                        on_progress=_prog,
                        should_cancel=rec.cancel_event.is_set,
                        on_project_detected=_on_proj,
                        mode=mode,
                        part_duration_sec=part_duration_sec,
                        aspect_ratio=aspect_ratio,
                        playback_speed=cfg.export.playback_speed,
                    )
                    rec.project_name = proj_name
                    lbl = "batch part(s)" if mode == "batch" else "viral Short(s)"
                    rec.logs.append(f"Done! Rendered {len(exported)} {lbl} in outputs/{proj_name}/")

                rec = jobs.start_job("autopilot", "auto", _run_auto)
                self._send_json(202, rec.to_dict())
                return

            if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                job_id = path[len("/api/jobs/") : -len("/cancel")].strip("/")
                rec = jobs.cancel(job_id)
                if rec is None:
                    self._send_json(404, {"error": "Job not found"})
                else:
                    self._send_json(200, rec.to_dict())
                return

            if path.startswith("/api/projects/") and path.endswith("/delete"):
                proj_name = path[len("/api/projects/") : -len("/delete")].strip("/")
                cfg = self._cfg()
                p_dir = _safe_resolve_under(cfg.projects_root(), proj_name)
                o_dir = _safe_resolve_under(cfg.outputs_root(), proj_name)
                if p_dir is None or o_dir is None or not proj_name or "/" in proj_name or "\\" in proj_name:
                    self._send_json(400, {"error": "Invalid project name"})
                    return
                if p_dir.exists() and p_dir != cfg.projects_root().resolve():
                    shutil.rmtree(p_dir, ignore_errors=True)
                if o_dir.exists() and o_dir != cfg.outputs_root().resolve():
                    shutil.rmtree(o_dir, ignore_errors=True)
                self._send_json(200, {"status": "deleted", "project_name": proj_name})
                return

            if path.startswith("/api/projects/") and path.endswith("/decisions"):
                proj_name = path[len("/api/projects/") : -len("/decisions")].strip("/")
                cfg = self._cfg()
                ws = ProjectWorkspace(cfg.projects_root() / proj_name, cfg.outputs_root() / proj_name)
                try:
                    doc = ReviewDecisionsFile.model_validate(body)
                    ws.decisions_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
                    self._send_json(200, {"status": "saved"})
                except Exception as exc:  # noqa: BLE001
                    self._send_json(400, {"error": str(exc)})
                return

            if path == "/api/jobs/new":
                source = str(body.get("source") or "").strip()
                if not source:
                    self._send_json(400, {"error": "source is required."})
                    return

                cfg = self._cfg()
                _apply_runtime_overrides(cfg, body)

                name = str(body.get("name") or "").strip() or "project"
                person = str(body.get("person") or "").strip() or "Speaker"
                mode = "batch" if str(body.get("mode") or "viral").strip().lower() == "batch" else "viral"
                try:
                    part_duration_sec = max(0.0, float(body.get("part_duration_sec") or 0.0))
                except (TypeError, ValueError):
                    part_duration_sec = 0.0
                aspect_ratio = body.get("aspect_ratio") if body.get("aspect_ratio") in ("9:16", "16:9") else None
                raw_max = int(body["max_clips"]) if body.get("max_clips") is not None else (0 if mode == "batch" else cfg.moments.max_candidates)
                if mode == "batch":
                    max_clips = 0 if raw_max <= 0 else max(1, min(50, raw_max))
                else:
                    max_clips = 15 if raw_max <= 0 else max(1, min(15, raw_max))
                min_score = int(body.get("min_score") or cfg.review.auto_score_threshold)
                auto = bool(body.get("auto", True))
                deceased = bool(body.get("deceased", False))
                force = bool(body.get("force", False))
                show_override = str(body.get("show") or "").strip()
                year_override = str(body.get("year") or "").strip()

                def _run_new(rec: JobRecord) -> None:
                    def _prog(msg: str) -> None:
                        rec.message = msg
                        rec.logs.append(msg)

                    run_new_project(
                        project_name=name,
                        source=source,
                        person=person,
                        config=cfg,
                        auto=auto,
                        serve=False,
                        min_score=min_score,
                        max_clips=max_clips,
                        show_override=show_override,
                        year_override=year_override,
                        deceased=deceased,
                        force=force,
                        on_progress=_prog,
                        mode=mode,
                        part_duration_sec=part_duration_sec,
                        aspect_ratio=aspect_ratio,
                        playback_speed=cfg.export.playback_speed,
                    )

                rec = jobs.start_job("new", name, _run_new)
                self._send_json(202, rec.to_dict())
                return

            if path == "/api/jobs/render":
                name = str(body.get("name") or "").strip()
                if not name:
                    self._send_json(400, {"error": "Project name is required."})
                    return

                cfg = self._cfg()
                _apply_runtime_overrides(cfg, body)

                auto = bool(body.get("auto", False))
                force = bool(body.get("force", False))
                min_score = body.get("min_score")
                aspect_ratio = body.get("aspect_ratio") if body.get("aspect_ratio") in ("9:16", "16:9") else None
                playback_speed = cfg.export.playback_speed if body.get("playback_speed") is not None else None

                def _run_render(rec: JobRecord) -> None:
                    def _prog(msg: str) -> None:
                        rec.message = msg
                        rec.logs.append(msg)

                    exported = run_render_project(
                        project_name=name,
                        config=cfg,
                        auto=auto,
                        min_score=int(min_score) if min_score is not None else None,
                        force=force,
                        on_progress=_prog,
                        should_cancel=rec.cancel_event.is_set,
                        aspect_ratio=aspect_ratio,
                        playback_speed=playback_speed,
                    )
                    rec.logs.append(f"Exported {len(exported)} video(s) to outputs/{name}/")

                rec = jobs.start_job("render", name, _run_render)
                self._send_json(202, rec.to_dict())
                return

            if path == "/api/jobs/edit-clip":
                name = str(body.get("name") or "").strip()
                clip_id = str(body.get("clip_id") or "").strip()
                if not name or not clip_id:
                    self._send_json(400, {"error": "Both project name and clip_id are required."})
                    return

                cfg = self._cfg()
                _apply_runtime_overrides(cfg, body)
                aspect_ratio = body.get("aspect_ratio") if body.get("aspect_ratio") in ("9:16", "16:9") else None

                def _run_edit(rec: JobRecord) -> None:
                    def _prog(msg: str) -> None:
                        rec.message = msg
                        rec.logs.append(msg)

                    updated = run_edit_single_clip(
                        project_name=name,
                        clip_id=clip_id,
                        config=cfg,
                        on_progress=_prog,
                        aspect_ratio=aspect_ratio,
                    )
                    rec.logs.append(f"Baked runtime edits into {updated.output_video}")

                rec = jobs.start_job("edit", name, _run_edit)
                self._send_json(202, rec.to_dict())
                return

            if path == "/api/config":
                cfg = self._cfg()
                cfg_updates = body.get("config") or {}
                if cfg_updates:
                    try:
                        merged = cfg.model_dump()
                        for section, vals in cfg_updates.items():
                            if section in merged and isinstance(merged[section], dict) and isinstance(vals, dict):
                                merged[section].update(vals)
                        validated = AppConfig.model_validate(merged)
                        cfg_path.write_text(
                            yaml.safe_dump(validated.model_dump(), sort_keys=False),
                            encoding="utf-8",
                        )
                    except Exception as exc:  # noqa: BLE001
                        self._send_json(400, {"error": f"Invalid config update: {exc}"})
                        return

                env_updates = body.get("keys") or {}
                if isinstance(env_updates, dict) and any(str(v).strip() for v in env_updates.values()):
                    _update_env_file(cfg.root_dir, {k: str(v) for k, v in env_updates.items()})

                self._send_json(200, {"status": "saved"})
                return

            if path == "/api/browse-file":
                picked = _open_native_file_dialog()
                self._send_json(200, {"path": picked})
                return

            if path == "/api/open-folder":
                proj_name = str(body.get("project_name") or "").strip()
                folder = self._cfg().outputs_root() / proj_name if proj_name else self._cfg().outputs_root()
                ok = _open_folder_in_os(folder)
                self._send_json(200, {"opened": ok, "path": str(folder)})
                return

            self.send_error(404)

    return ThreadingHTTPServer((host, port), StudioHandler)


def _open_browser_url(url: str) -> None:
    """Reliably open URL in the user's default browser on Windows/macOS/Linux (including pythonw)."""
    if platform.system().lower() == "windows" and hasattr(os, "startfile"):
        try:
            os.startfile(url)  # type: ignore[attr-defined]
            return
        except Exception:  # noqa: BLE001
            pass
    webbrowser.open(url)


def launch_gui(
    config_path: Optional[Path | str] = None,
    host: str = "127.0.0.1",
    port: int = 8780,
    open_browser: bool = True,
) -> None:
    """Start the local Shortly Studio GUI server (or open the browser if already running)."""
    import urllib.request

    url = f"http://{host}:{port}/"
    # Probe first (Windows SO_REUSEADDR can otherwise bind twice instead of raising OSError)
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/config", timeout=0.8) as resp:
            if resp.status == 200:
                print(f"Shortly Studio is already running at {url}")
                if open_browser:
                    _open_browser_url(url)
                return
    except Exception:  # noqa: BLE001
        pass

    server = create_gui_server(config_path=config_path, host=host, port=port)
    print(f"Shortly Studio GUI running at {url}  (press Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.45, lambda: _open_browser_url(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Shortly Studio GUI.")
    finally:
        server.server_close()
