"""Idempotent project state persistence and SHA-256 disk caching for model calls."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

from shortsforge.models import ProjectState


class ProjectWorkspace:
    """Manages directory structure, resumable stage markers, and model call caches."""

    def __init__(self, project_dir: Path, output_dir: Path) -> None:
        self.project_dir = project_dir
        self.output_dir = output_dir
        self.source_dir = project_dir / "source"
        self.cache_dir = project_dir / "cache"
        self.llm_cache_dir = self.cache_dir / "llm"
        self.work_dir = project_dir / "work"
        self.thumbs_dir = project_dir / "thumbs"

        for d in (
            self.project_dir,
            self.output_dir,
            self.source_dir,
            self.cache_dir,
            self.llm_cache_dir,
            self.work_dir,
            self.thumbs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def state_path(self) -> Path:
        return self.project_dir / "state.json"

    @property
    def metadata_path(self) -> Path:
        return self.project_dir / "metadata.json"

    @property
    def transcript_path(self) -> Path:
        return self.project_dir / "transcript.json"

    @property
    def candidates_path(self) -> Path:
        return self.project_dir / "candidates.json"

    @property
    def review_html_path(self) -> Path:
        return self.project_dir / "review.html"

    @property
    def decisions_path(self) -> Path:
        return self.project_dir / "decisions.json"

    def load_state(self) -> Optional[ProjectState]:
        if not self.state_path.exists():
            return None
        return ProjectState.model_validate_json(self.state_path.read_text(encoding="utf-8"))

    def save_state(self, state: ProjectState) -> None:
        tmp = self.state_path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def mark_step_complete(self, state: ProjectState, step_name: str) -> None:
        if step_name not in state.completed_steps:
            state.completed_steps.append(step_name)
        self.save_state(state)

    def add_warning(self, state: ProjectState, warning: str) -> None:
        if warning not in state.warnings:
            state.warnings.append(warning)
            self.save_state(state)

    def llm_cache_key(self, namespace: str, payload: dict[str, Any]) -> str:
        canonical = json.dumps({"ns": namespace, "payload": payload}, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get_llm_cache(self, key: str) -> Optional[Any]:
        path = self.llm_cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def set_llm_cache(self, key: str, value: Any) -> None:
        path = self.llm_cache_dir / f"{key}.json"
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
