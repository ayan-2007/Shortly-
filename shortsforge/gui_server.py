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
from urllib.parse import unquote, urlparse

import yaml

from shortsforge.config import AppConfig, get_env_key, load_config
from shortsforge.models import (
    ClipCandidate,
    ClipExportMetadata,
    ProjectState,
    ReviewDecisionsFile,
    VideoMetadata,
)
from shortsforge.pipeline import (
    run_autopilot_from_source,
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
                    rec.message = "Completed! Your viral Shorts are ready."
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
                v_tag = int(vid_p.stat().st_mtime * 1000) if vid_p.exists() else 0
                t_tag = int(th_p.stat().st_mtime * 1000) if th_p.exists() else 0
                ed["video_url"] = f"/media/output/{project_name}/{em.output_video}?v={v_tag}"
                ed["thumbnail_url"] = f"/media/output/{project_name}/{em.thumbnail}?v={t_tag}"
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

        def _serve_file_with_range(self, file_path: Path, no_cache: bool = False) -> None:
            if not file_path.is_file():
                self.send_error(404, "File not found")
                return
            file_size = file_path.stat().st_size
            ctype, _ = mimetypes.guess_type(str(file_path))
            ctype = ctype or "application/octet-stream"
            range_hdr = self.headers.get("Range")

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
                    if no_cache:
                        self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
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
                if no_cache:
                    self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                with file_path.open("rb") as fh:
                    while True:
                        chunk = fh.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return

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

            if path.startswith("/media/project/"):
                rest = path.split("/media/project/", 1)[1]
                parts = rest.split("/", 1)
                if len(parts) != 2:
                    self.send_error(400)
                    return
                proj_name, rel = parts
                target = _safe_resolve_under(self._cfg().projects_root() / proj_name, rel)
                if target is None:
                    self.send_error(403)
                    return
                self._serve_file_with_range(target, no_cache=True)
                return

            if path.startswith("/media/output/"):
                rest = path.split("/media/output/", 1)[1]
                parts = rest.split("/", 1)
                if len(parts) != 2:
                    self.send_error(400)
                    return
                proj_name, rel = parts
                target = _safe_resolve_under(self._cfg().outputs_root() / proj_name, rel)
                if target is None:
                    self.send_error(403)
                    return
                self._serve_file_with_range(target, no_cache=True)
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

            # 1-Click Autopilot from just a URL or file path!
            if path == "/api/jobs/autopilot":
                source = str(body.get("source") or "").strip()
                if not source:
                    self._send_json(400, {"error": "Paste a YouTube link or video file path first."})
                    return

                cfg = self._cfg()
                if body.get("caption_preset") in ("iman", "hormozi", "documentary", "neon", "comic", "minimal"):
                    cfg.captions.preset = body["caption_preset"]
                if body.get("active_color") in ("purple", "yellow", "red"):
                    cfg.captions.active_color = body["active_color"]
                if body.get("hook_style") in ("cold_open", "freeze_narrator"):
                    cfg.hook.style = body["hook_style"]
                if "hook_enabled" in body:
                    cfg.hook.enabled = bool(body["hook_enabled"])
                if "show_opening_title" in body:
                    cfg.hook.show_opening_title = bool(body["show_opening_title"])
                if "mirror_video" in body:
                    cfg.export.mirror_video = bool(body["mirror_video"])
                if isinstance(body.get("video_filters"), list):
                    cfg.export.video_filters = [str(f).strip() for f in body["video_filters"] if str(f).strip()][:2]
                if "music_enabled" in body:
                    cfg.audio.bg_music_enabled = bool(body["music_enabled"])
                    cfg.audio.hook_bed_enabled = bool(body["music_enabled"])
                raw_max = body.get("max_clips")
                max_clips = 15 if (raw_max is None or int(raw_max) <= 0) else max(1, min(15, int(raw_max)))
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
                    )
                    rec.project_name = proj_name
                    rec.logs.append(f"Done! Rendered {len(exported)} viral Short(s) in outputs/{proj_name}/")

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
                if body.get("caption_preset") in ("iman", "hormozi", "documentary", "neon", "comic", "minimal"):
                    cfg.captions.preset = body["caption_preset"]
                if body.get("active_color") in ("purple", "yellow", "red"):
                    cfg.captions.active_color = body["active_color"]
                if body.get("hook_style") in ("cold_open", "freeze_narrator"):
                    cfg.hook.style = body["hook_style"]

                name = str(body.get("name") or "").strip() or "project"
                person = str(body.get("person") or "").strip() or "Speaker"
                raw_max = int(body["max_clips"]) if body.get("max_clips") is not None else cfg.moments.max_candidates
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
                if body.get("hook_style") in ("cold_open", "freeze_narrator"):
                    cfg.hook.style = body["hook_style"]
                if "hook_enabled" in body:
                    cfg.hook.enabled = bool(body["hook_enabled"])
                if "show_opening_title" in body:
                    cfg.hook.show_opening_title = bool(body["show_opening_title"])
                if body.get("caption_preset") in ("iman", "hormozi", "documentary", "neon", "comic", "minimal"):
                    cfg.captions.preset = body["caption_preset"]
                if body.get("active_color") in ("purple", "yellow", "red"):
                    cfg.captions.active_color = body["active_color"]
                if "archive_tag" in body:
                    cfg.overlays.archive_tag = bool(body["archive_tag"])
                if "mirror_video" in body:
                    cfg.export.mirror_video = bool(body["mirror_video"])
                if isinstance(body.get("video_filters"), list):
                    cfg.export.video_filters = [str(f).strip() for f in body["video_filters"] if str(f).strip()][:2]

                auto = bool(body.get("auto", False))
                force = bool(body.get("force", False))
                min_score = body.get("min_score")

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
                    )
                    rec.logs.append(f"Exported {len(exported)} Short(s) to outputs/{name}/")

                rec = jobs.start_job("render", name, _run_render)
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
                    webbrowser.open(url)
                return
    except Exception:  # noqa: BLE001
        pass

    server = create_gui_server(config_path=config_path, host=host, port=port)
    print(f"Shortly Studio GUI running at {url}  (press Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.45, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Shortly Studio GUI.")
    finally:
        server.server_close()
