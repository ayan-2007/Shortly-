"""Pydantic data models shared across all ShortsForge pipeline stages."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    """Metadata extracted from a URL (yt-dlp) or local media file (ffprobe)."""

    source: str
    source_type: Literal["url", "local"]
    title: str = ""
    channel: str = ""
    upload_date: str = ""
    description: str = ""
    show_name: str = ""
    original_air_date: str = ""
    year: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 30.0
    aspect_ratio: float = 1.7778
    local_video_path: str = ""
    local_audio_path: str = ""
    heatmap: List[Dict[str, float]] = Field(default_factory=list)
    chapters: List[Dict[str, Any]] = Field(default_factory=list)


class WordTimestamp(BaseModel):
    """Single word with precise start/end timing in seconds."""

    word: str
    start: float
    end: float
    probability: float = 1.0


class Sentence(BaseModel):
    """Sentence grouped from word timestamps with an optional speaker turn label."""

    index: int
    text: str
    start: float
    end: float
    speaker: str = "SPEAKER_1"
    words: List[WordTimestamp] = Field(default_factory=list)


class Transcript(BaseModel):
    """Complete word- and sentence-level transcript."""

    language: str = "en"
    duration: float = 0.0
    sentences: List[Sentence] = Field(default_factory=list)
    words: List[WordTimestamp] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    """0-100 rubric breakdown for a candidate clip."""

    emotional_peak: int = Field(ge=0, le=100, default=50)
    quotable_statement: int = Field(ge=0, le=100, default=50)
    humor_payoff: int = Field(ge=0, le=100, default=50)
    self_contained: int = Field(ge=0, le=100, default=50)
    clean_boundaries: int = Field(ge=0, le=100, default=50)


class QualityReport(BaseModel):
    """Visual and audio quality assessment for a candidate clip."""

    passed: bool = True
    face_presence_ratio: float = 1.0
    graphics_ratio: float = 0.0
    audio_rms_db: float = -20.0
    clipping_ratio: float = 0.0
    estimated_snr_db: float = 20.0
    rejection_reasons: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class HookVerification(BaseModel):
    """Result of the second-pass LLM transcript factuality check."""

    passed: bool
    reason: str = ""
    violations: List[str] = Field(default_factory=list)


class HookVariant(BaseModel):
    """A single ranked hook candidate (<= 9 words)."""

    rank: int = Field(ge=1)
    text: str
    predicted_retention: int = Field(ge=0, le=100, default=75)
    verification: HookVerification = Field(
        default_factory=lambda: HookVerification(passed=True, reason="Verified")
    )


class HookSet(BaseModel):
    """Collection of 5 ranked hook variants and verification status for a clip."""

    selected_index: int = 0
    variants: List[HookVariant] = Field(default_factory=list)
    attempts: int = 1
    flagged_for_review: bool = False
    flag_reason: str = ""
    edited_text: Optional[str] = None

    def active_hook_text(self) -> str:
        """Return the user-edited hook if present, otherwise the selected variant text."""
        if self.edited_text and self.edited_text.strip():
            return self.edited_text.strip()
        if self.variants:
            idx = max(0, min(self.selected_index, len(self.variants) - 1))
            return self.variants[idx].text.strip()
        return ""


class ClipCandidate(BaseModel):
    """A proposed vertical Short candidate."""

    id: str
    start: float
    end: float
    score: int = Field(ge=0, le=100)
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    reason: str
    key_quote: str
    transcript_text: str
    peak_start: float
    peak_end: float
    quality: Optional[QualityReport] = None
    hooks: Optional[HookSet] = None
    thumbnail_path: Optional[str] = None
    rejected_by_quality: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class ReviewDecision(BaseModel):
    """Human (or --auto) review decision for a single clip."""

    clip_id: str
    approved: bool = False
    selected_hook_index: int = 0
    edited_hook_text: Optional[str] = None


class ReviewDecisionsFile(BaseModel):
    """Persisted review decisions loaded before rendering."""

    project_name: str
    decisions: Dict[str, ReviewDecision] = Field(default_factory=dict)


class ClipExportMetadata(BaseModel):
    """Exported publishing metadata pack for a single rendered Short."""

    clip_id: str
    output_video: str
    thumbnail: str
    titles: List[str]
    description: str
    hashtags: List[str]
    pinned_comment: str
    hook_text: str
    start: float
    end: float
    score: int
    lufs: Optional[float] = None


class ProjectState(BaseModel):
    """Idempotent resumable project state manifest."""

    project_name: str
    person: str
    source: str
    show_override: str = ""
    year_override: str = ""
    deceased: bool = False
    completed_steps: List[str] = Field(default_factory=list)
    rendered_clips: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
