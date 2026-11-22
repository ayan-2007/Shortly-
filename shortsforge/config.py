"""Configuration loader, .env integration, and cross-platform font discovery."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    projects_dir: str = "projects"
    outputs_dir: str = "outputs"
    fonts_dir: str = ""


class IngestConfig(BaseModel):
    max_video_height: int = 1080
    prefer_h264: bool = True
    concurrent_fragments: int = 4
    retries: int = 10
    cookies_from_browser: str = ""


class LLMConfig(BaseModel):
    provider: Literal["gemini", "claude", "mock"] = "gemini"
    gemini_model: str = "gemini-2.5-flash"
    claude_model: str = "claude-3-5-sonnet-latest"
    temperature: float = 0.3
    max_retries: int = 2
    cache_enabled: bool = True


class TTSConfig(BaseModel):
    provider: Literal["edge", "openai", "offline"] = "edge"
    edge_voice: str = "en-US-AndrewNeural"
    openai_voice: str = "onyx"
    rate: str = "+4%"


class TranscriptionConfig(BaseModel):
    model_size: str = "small"
    device: str = "auto"
    compute_type: str = "auto"
    language: Optional[str] = None
    vad_filter: bool = True
    speaker_turn_pause_sec: float = 1.6


class ScoringWeightsConfig(BaseModel):
    emotional_peak: float = 0.25
    quotable_statement: float = 0.25
    humor_payoff: float = 0.15
    self_contained: float = 0.20
    clean_boundaries: float = 0.15


class MomentsConfig(BaseModel):
    max_candidates: int = 10
    min_duration_sec: float = 20.0
    max_duration_sec: float = 55.0
    window_minutes: int = 18
    window_overlap_minutes: int = 2
    scoring_weights: ScoringWeightsConfig = Field(default_factory=ScoringWeightsConfig)


class QualityCheckConfig(BaseModel):
    enabled: bool = True
    sample_fps: float = 0.5
    min_face_presence_ratio: float = 0.30
    max_graphics_ratio: float = 0.70
    min_audio_rms_db: float = -42.0
    max_clipping_ratio: float = 0.05
    min_estimated_snr_db: float = 6.0


class HookConfig(BaseModel):
    style: Literal["cold_open", "freeze_narrator"] = "cold_open"
    variants_count: int = 5
    max_words: int = 9
    cold_open_min_sec: float = 2.0
    cold_open_max_sec: float = 3.0
    freeze_duration_sec: float = 2.8
    freeze_zoom_end: float = 1.12
    transition_duration_sec: float = 0.28
    max_verification_attempts: int = 3
    deceased_subject: bool = False


class ReframeConfig(BaseModel):
    output_width: int = 1080
    output_height: int = 1920
    output_fps: int = 30
    face_smoothing_alpha: float = 0.12
    deadzone_ratio: float = 0.05
    max_pan_speed_px_per_sec: float = 380.0
    two_person_layout: Literal["split", "blurred_fit"] = "split"
    four_by_three_zoom: float = 1.06
    scene_threshold: float = 27.0
    enhance_old_footage: bool = False


class CaptionsConfig(BaseModel):
    preset: Literal["neon", "comic", "minimal"] = "neon"
    words_per_group: int = Field(default=3, ge=1, le=3)
    font_name: str = ""
    font_size: int = 94
    emphasis_scale: float = 1.28
    active_color: Literal["yellow", "red"] = "yellow"
    safe_margin_left: int = 80
    safe_margin_right: int = 160
    safe_margin_bottom: int = 360


class OverlaysConfig(BaseModel):
    lower_third_duration_sec: float = 4.0
    archive_tag: bool = False
    closing_card_text: str = "Follow for more"
    closing_card_duration_sec: float = Field(default=0.85, gt=0.1, lt=1.0)


class AudioConfig(BaseModel):
    target_lufs: float = -14.0
    true_peak_db: float = -1.5
    lra: float = 11.0
    hook_bed_enabled: bool = True
    hook_bed_path: str = ""
    hook_bed_volume_db: float = -22.0
    transition_sfx_path: str = ""
    transition_sfx_volume_db: float = -10.0


class ReviewConfig(BaseModel):
    auto_score_threshold: int = 75
    server_port: int = 8765


class ExportConfig(BaseModel):
    video_crf: int = 20
    video_preset: str = "medium"
    audio_bitrate: str = "192k"
    rights_note: str = (
        "Verify licensing/fair-use permissions for original broadcast footage before publishing."
    )


class AppConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    moments: MomentsConfig = Field(default_factory=MomentsConfig)
    quality_check: QualityCheckConfig = Field(default_factory=QualityCheckConfig)
    hook: HookConfig = Field(default_factory=HookConfig)
    reframe: ReframeConfig = Field(default_factory=ReframeConfig)
    captions: CaptionsConfig = Field(default_factory=CaptionsConfig)
    overlays: OverlaysConfig = Field(default_factory=OverlaysConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    root_dir: Path = Field(default_factory=lambda: Path.cwd(), exclude=True)

    def resolve_path(self, raw: str) -> Path:
        p = Path(raw)
        return p if p.is_absolute() else (self.root_dir / p).resolve()

    def projects_root(self) -> Path:
        return self.resolve_path(self.paths.projects_dir)

    def outputs_root(self) -> Path:
        return self.resolve_path(self.paths.outputs_dir)


def load_config(config_path: Optional[Path | str] = None) -> AppConfig:
    """Load .env and config.yaml into an AppConfig instance."""
    root = Path.cwd().resolve()
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=False)
    else:
        load_dotenv(override=False)

    cfg_file = Path(config_path).resolve() if config_path else (root / "config.yaml")
    data: Dict = {}
    if cfg_file.exists():
        with cfg_file.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
            if isinstance(loaded, dict):
                data = loaded
        root = cfg_file.parent.resolve()

    cfg = AppConfig.model_validate(data)
    cfg.root_dir = root
    return cfg


def get_env_key(env_var: str) -> str:
    """Fetch an API key from environment variables without ever hardcoding secrets."""
    return os.environ.get(env_var, "").strip()


_BOLD_FONT_CANDIDATES: List[Tuple[str, str]] = [
    ("Montserrat-Black.ttf", "Montserrat Black"),
    ("Montserrat-ExtraBold.ttf", "Montserrat ExtraBold"),
    ("Anton-Regular.ttf", "Anton"),
    ("ariblk.ttf", "Arial Black"),
    ("seguibl.ttf", "Segoe UI Black"),
    ("impact.ttf", "Impact"),
    ("Montserrat-Bold.ttf", "Montserrat"),
    ("Inter-ExtraBold.ttf", "Inter"),
    ("Inter-Bold.ttf", "Inter"),
    ("DejaVuSans-Bold.ttf", "DejaVu Sans"),
    ("LiberationSans-Bold.ttf", "Liberation Sans"),
    ("arialbd.ttf", "Arial"),
    ("segoeuib.ttf", "Segoe UI"),
    ("HelveticaNeue.ttc", "Helvetica Neue"),
    ("Arial Bold.ttf", "Arial"),
]

_REGULAR_FONT_CANDIDATES: List[Tuple[str, str]] = [
    ("Montserrat-SemiBold.ttf", "Montserrat"),
    ("Inter-Regular.ttf", "Inter"),
    ("DejaVuSans.ttf", "DejaVu Sans"),
    ("LiberationSans-Regular.ttf", "Liberation Sans"),
    ("arial.ttf", "Arial"),
    ("segoeui.ttf", "Segoe UI"),
    ("calibri.ttf", "Calibri"),
]


def _system_font_dirs(custom_dir: Optional[Path] = None) -> List[Path]:
    dirs: List[Path] = []
    if custom_dir and custom_dir.exists():
        dirs.append(custom_dir)

    sys_name = platform.system().lower()
    if sys_name == "windows":
        win_dir = os.environ.get("WINDIR", "")
        if win_dir:
            dirs.append(Path(win_dir) / "Fonts")
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            dirs.append(Path(local_app) / "Microsoft" / "Windows" / "Fonts")
    elif sys_name == "darwin":
        dirs.extend([
            Path("/Library/Fonts"),
            Path("/System/Library/Fonts"),
            Path.home() / "Library" / "Fonts",
        ])
    else:
        dirs.extend([
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
            Path.home() / ".local" / "share" / "fonts",
            Path.home() / ".fonts",
        ])
    return [d for d in dirs if d.exists()]


def resolve_font(
    config: AppConfig,
    bold: bool = True,
    preferred_name: str = "",
) -> Tuple[Optional[Path], str]:
    """Dynamically locate a cross-platform font file and family name (no hardcoded paths)."""
    custom_dir = config.resolve_path(config.paths.fonts_dir) if config.paths.fonts_dir else None
    search_dirs = _system_font_dirs(custom_dir)

    if preferred_name:
        maybe_path = Path(preferred_name)
        if maybe_path.is_file():
            return maybe_path, maybe_path.stem
        for d in search_dirs:
            for ext in ("", ".ttf", ".otf", ".ttc"):
                matches = list(d.rglob(f"{preferred_name}{ext}"))
                if matches:
                    return matches[0], preferred_name

    candidates = _BOLD_FONT_CANDIDATES if bold else _REGULAR_FONT_CANDIDATES
    lower_map: Dict[str, Path] = {}
    for d in search_dirs:
        try:
            for font_file in d.rglob("*"):
                if font_file.is_file() and font_file.suffix.lower() in {".ttf", ".otf", ".ttc"}:
                    key = font_file.name.lower()
                    if key not in lower_map:
                        lower_map[key] = font_file
        except PermissionError:
            continue

    for filename, family in candidates:
        found = lower_map.get(filename.lower())
        if found:
            return found, family

    for found in lower_map.values():
        return found, found.stem
    return None, "Sans"
