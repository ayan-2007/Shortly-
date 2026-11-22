"""Stage 9: Final H.264 vertical Short compositing, publishing metadata pack, thumbnail pick, sources.txt & warnings.txt."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import cv2

from shortsforge.audio import normalize_to_lufs
from shortsforge.captions import write_ass_captions
from shortsforge.config import AppConfig, resolve_font
from shortsforge.ffmpeg_utils import (
    escape_ffmpeg_filter_path,
    extract_stream_info,
    probe_media,
    run_ffmpeg,
)
from shortsforge.llm import LLMClient
from shortsforge.models import (
    ClipCandidate,
    ClipExportMetadata,
    Transcript,
    VideoMetadata,
    WordTimestamp,
)
from shortsforge.overlays import (
    render_closing_card_png,
    render_hook_banner_png,
    render_lower_third_png,
    render_persistent_credit_png,
)
from shortsforge.quality_check import FaceDetector
from shortsforge.reframe import reframe_clip_to_vertical
from shortsforge.tts import synthesize_narrator


EXPORT_META_SYSTEM = """You write YouTube Shorts / Reels / TikTok publishing metadata grounded strictly in the transcript.

RULES:
- Never invent facts, quotes, or claims not present in the transcript.
- Return exactly 5 distinct title options (concise, high CTR, truthful).
- Write a clear 2-3 sentence description that credits the original source/channel/show.
- Return exactly 8 relevant hashtags starting with '#'.
- Return 1 engaging pinned-comment question that sparks respectful discussion.

Return strict JSON:
{
  "titles": ["Title 1", "Title 2", "Title 3", "Title 4", "Title 5"],
  "description": "Description crediting the original source...",
  "hashtags": ["#shorts", "#interview", "#tag3", "#tag4", "#tag5", "#tag6", "#tag7", "#tag8"],
  "pinned_comment": "Thought-provoking question for the comments?"
}"""


def _slugify(text: str, max_len: int = 36) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return (slug[:max_len].strip("_")) or "short"


def _extract_best_vertical_thumbnail(
    reframed_video: Path,
    output_jpg: Path,
    peak_offset_sec: float,
    detector: Optional[FaceDetector] = None,
) -> Path:
    """Pick the sharpest, best-framed face frame from the vertical Short as the thumbnail."""
    own_det = detector is None
    face_det = detector or FaceDetector()

    cap = cv2.VideoCapture(str(reframed_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    duration = total_frames / fps

    best_score = -1.0
    best_frame = None

    sample_times = [
        min(duration - 0.1, max(0.1, peak_offset_sec + delta))
        for delta in (-0.8, 0.0, 0.5, 1.2)
    ] + [duration * frac for frac in (0.15, 0.35, 0.55, 0.75)]

    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        faces = face_det.detect(frame)
        face_bonus = sum(f.area for f in faces) / max(1.0, frame.shape[0] * frame.shape[1]) * 1800.0
        score = sharpness + face_bonus
        if score > best_score:
            best_score = score
            best_frame = frame.copy()

    cap.release()
    if own_det:
        face_det.close()

    output_jpg.parent.mkdir(parents=True, exist_ok=True)
    if best_frame is not None:
        cv2.imwrite(str(output_jpg), best_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return output_jpg


def _build_hook_segment(
    reframed_main: Path,
    candidate: ClipCandidate,
    hook_text: str,
    hook_banner_png: Path,
    work_dir: Path,
    config: AppConfig,
) -> tuple[Path, Path, float]:
    """Build either a cold-open peak clip OR a freeze-frame slow-zoom + narrator TTS hook segment."""
    fps = config.reframe.output_fps
    out_w = config.reframe.output_width
    out_h = config.reframe.output_height

    hook_video = work_dir / "hook_video.mp4"
    hook_audio = work_dir / "hook_audio.wav"

    if config.hook.style == "freeze_narrator":
        synthesize_narrator(hook_text, hook_audio, config)
        audio_info = extract_stream_info(probe_media(hook_audio))
        hook_dur = max(config.hook.freeze_duration_sec, min(4.2, audio_info["duration"] + 0.25))

        freeze_png = work_dir / "freeze_frame.png"
        peak_rel = max(0.0, candidate.peak_start - candidate.start)
        run_ffmpeg([
            "-ss", f"{peak_rel:.3f}",
            "-i", str(reframed_main),
            "-vframes", "1",
            str(freeze_png),
        ])

        total_frames = max(1, int(round(hook_dur * fps)))
        zoom_end = max(1.02, config.hook.freeze_zoom_end)
        zoom_inc = (zoom_end - 1.0) / total_frames

        run_ffmpeg([
            "-loop", "1",
            "-i", str(freeze_png),
            "-loop", "1",
            "-i", str(hook_banner_png),
            "-filter_complex",
            (
                f"[0:v]scale={out_w}:{out_h},zoompan=z='min(zoom+{zoom_inc:.6f},{zoom_end:.3f})':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={total_frames}:s={out_w}x{out_h}:fps={fps},"
                f"format=yuv420p[bg];"
                f"[1:v]format=rgba,fade=t=in:st=0:d=0.22:alpha=1[bn];"
                f"[bg][bn]overlay=0:0:shortest=1,format=yuv420p[vout]"
            ),
            "-map", "[vout]",
            "-t", f"{hook_dur:.3f}",
            "-r", str(fps),
            "-c:v", "libx264",
            "-crf", "17",
            "-preset", "veryfast",
            "-an",
            str(hook_video),
        ])
        return hook_video, hook_audio, hook_dur

    peak_rel_start = max(0.0, candidate.peak_start - candidate.start)
    peak_dur = max(
        config.hook.cold_open_min_sec,
        min(config.hook.cold_open_max_sec, candidate.peak_end - candidate.peak_start),
    )
    main_info = extract_stream_info(probe_media(reframed_main))
    if peak_rel_start + peak_dur > main_info["duration"]:
        peak_rel_start = max(0.0, main_info["duration"] - peak_dur)
        peak_dur = min(peak_dur, main_info["duration"])

    run_ffmpeg([
        "-ss", f"{peak_rel_start:.3f}",
        "-t", f"{peak_dur:.3f}",
        "-i", str(reframed_main),
        "-loop", "1",
        "-i", str(hook_banner_png),
        "-filter_complex",
        (
            "[0:v]setpts=PTS-STARTPTS,format=yuv420p[base];"
            "[1:v]format=rgba,fade=t=in:st=0:d=0.18:alpha=1[bn];"
            "[base][bn]overlay=0:0:shortest=1,format=yuv420p[vout]"
        ),
        "-map", "[vout]",
        "-t", f"{peak_dur:.3f}",
        "-r", str(fps),
        "-c:v", "libx264",
        "-crf", "17",
        "-preset", "veryfast",
        "-an",
        str(hook_video),
    ])

    run_ffmpeg([
        "-ss", f"{peak_rel_start:.3f}",
        "-t", f"{peak_dur:.3f}",
        "-i", str(reframed_main),
        "-vn",
        "-ar", "48000",
        "-ac", "2",
        "-c:a", "pcm_s16le",
        str(hook_audio),
    ])
    return hook_video, hook_audio, peak_dur


def _build_transition_video(
    reframed_main: Path,
    duration: float,
    output_path: Path,
    config: AppConfig,
) -> Path:
    """Create a crisp quick whoosh/rewind flash transition into the clip's real start."""
    fps = config.reframe.output_fps
    out_w = config.reframe.output_width
    out_h = config.reframe.output_height
    half_d = max(0.06, duration * 0.5)

    run_ffmpeg([
        "-ss", "0",
        "-t", f"{duration:.3f}",
        "-i", str(reframed_main),
        "-vf",
        (
            f"scale={out_w}:{out_h},fps={fps},"
            "gblur=sigma=18:steps=2,"
            f"fade=t=out:st=0:d={half_d:.3f}:color=white,"
            f"fade=t=in:st={half_d:.3f}:d={half_d:.3f}:color=white,"
            "format=yuv420p"
        ),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-an",
        str(output_path),
    ])
    return output_path


def _generate_publishing_metadata(
    candidate: ClipCandidate,
    metadata: VideoMetadata,
    person: str,
    hook_text: str,
    llm: LLMClient,
) -> dict:
    """Use LLM (cached) to generate 5 titles, credited description, 8 hashtags, and pinned comment."""
    source_credit = metadata.channel or metadata.show_name or "Original Broadcast"
    user_prompt = (
        f"Subject: {person}\n"
        f"Original Show/Channel: {source_credit}\n"
        f"Original Title: {metadata.title}\n"
        f"Year: {metadata.year or 'Archive'}\n"
        f"Hook: {hook_text}\n"
        f"Key quote: {candidate.key_quote}\n"
        f"Transcript:\n{candidate.transcript_text}\n"
    )
    raw = llm.complete_json(
        namespace=f"export_metadata_{candidate.id}",
        system_prompt=EXPORT_META_SYSTEM,
        user_prompt=user_prompt,
    )
    titles = list(raw.get("titles", []))[:5] if isinstance(raw, dict) else []
    while len(titles) < 5:
        titles.append(f"{person}: {hook_text}"[:80])

    desc = str(raw.get("description", "")).strip() if isinstance(raw, dict) else ""
    if source_credit.lower() not in desc.lower():
        desc = f"{desc}\n\nSource: {source_credit} ({metadata.title})".strip()

    hashtags = [
        h if str(h).startswith("#") else f"#{h}"
        for h in (raw.get("hashtags", []) if isinstance(raw, dict) else [])
    ][:8]
    defaults = ["#shorts", "#interview", "#podcast", "#quotes", "#storytime", "#mindset", "#viral", "#archive"]
    for d in defaults:
        if len(hashtags) >= 8:
            break
        if d not in hashtags:
            hashtags.append(d)

    pinned = (
        str(raw.get("pinned_comment", "")).strip()
        if isinstance(raw, dict)
        else ""
    ) or f"What stood out most to you about what {person} said here?"

    return {
        "titles": titles,
        "description": desc,
        "hashtags": hashtags[:8],
        "pinned_comment": pinned,
    }


def render_single_short(
    candidate: ClipCandidate,
    hook_text: str,
    metadata: VideoMetadata,
    transcript: Transcript,
    person: str,
    work_dir: Path,
    output_dir: Path,
    llm: LLMClient,
    config: AppConfig,
    detector: Optional[FaceDetector] = None,
) -> ClipExportMetadata:
    """Render a raw, uninterrupted 9:16 vertical Short from [candidate.start, candidate.end] with bold animated captions."""
    clip_work = work_dir / candidate.id
    clip_work.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(hook_text or candidate.key_quote or candidate.id)
    base_name = f"{candidate.id}_{slug}"
    final_mp4 = output_dir / f"{base_name}.mp4"
    final_thumb = output_dir / f"{base_name}_thumb.jpg"
    meta_json_path = output_dir / f"{base_name}_metadata.json"
    meta_txt_path = output_dir / f"{base_name}_metadata.txt"

    # 1. Reframe raw continuous clip [candidate.start, candidate.end] to 1080x1920 (no stitched intro, no artificial cuts)
    reframed_main = clip_work / "reframed_main.mp4"
    reframe_clip_to_vertical(
        source_video=Path(metadata.local_video_path),
        start_sec=candidate.start,
        end_sec=candidate.end,
        output_video=reframed_main,
        config=config,
        detector=detector,
    )

    # 2. Extract raw clip audio and normalize cleanly to -14 LUFS (no synthetic intro SFX or drone beds)
    main_audio_wav = clip_work / "main_audio.wav"
    run_ffmpeg([
        "-i", str(reframed_main),
        "-vn",
        "-ar", "48000",
        "-ac", "2",
        "-c:a", "pcm_s16le",
        str(main_audio_wav),
    ])
    master_wav, measured_lufs = normalize_to_lufs(
        input_wav=main_audio_wav,
        output_wav=clip_work / "master_loudnorm.wav",
        config=config,
    )

    # 3. Generate ultra-bold, dynamically animated word-by-word ASS captions synced directly from 0.0s
    clip_words: List[WordTimestamp] = [
        w for w in transcript.words if w.end >= candidate.start and w.start <= candidate.end
    ]
    ass_path = write_ass_captions(
        output_ass=clip_work / "captions.ass",
        clip_words=clip_words,
        clip_start=candidate.start,
        time_offset=0.0,
        config=config,
        key_quote=candidate.key_quote,
        hook_text=hook_text,
    )

    escaped_ass = escape_ffmpeg_filter_path(ass_path)
    font_path, _ = resolve_font(config, bold=True, preferred_name=config.captions.font_name)
    fontsdir_arg = ""
    if font_path and font_path.parent.exists():
        fontsdir_arg = f":fontsdir='{escape_ffmpeg_filter_path(font_path.parent)}'"

    # 4. Burn ONLY the bold animated captions onto the raw reframed 9:16 video + mastered audio
    run_ffmpeg([
        "-i", str(reframed_main),
        "-i", str(master_wav),
        "-filter_complex",
        f"[0:v]ass=filename='{escaped_ass}'{fontsdir_arg},format=yuv420p[vfinal]",
        "-map", "[vfinal]",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-crf", str(config.export.video_crf),
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", config.export.audio_bitrate,
        "-movflags", "+faststart",
        "-shortest",
        str(final_mp4),
    ])

    _extract_best_vertical_thumbnail(
        reframed_video=reframed_main,
        output_jpg=final_thumb,
        peak_offset_sec=max(0.0, candidate.peak_start - candidate.start),
        detector=detector,
    )

    pub = _generate_publishing_metadata(candidate, metadata, person, hook_text, llm)
    export_meta = ClipExportMetadata(
        clip_id=candidate.id,
        output_video=final_mp4.name,
        thumbnail=final_thumb.name,
        titles=pub["titles"],
        description=pub["description"],
        hashtags=pub["hashtags"],
        pinned_comment=pub["pinned_comment"],
        hook_text=hook_text,
        start=candidate.start,
        end=candidate.end,
        score=candidate.score,
        lufs=measured_lufs,
    )
    meta_json_path.write_text(export_meta.model_dump_json(indent=2), encoding="utf-8")

    txt_lines = [
        f"CLIP: {candidate.id} ({candidate.start:.2f}s - {candidate.end:.2f}s) | Score: {candidate.score}/100",
        f"VIDEO: {final_mp4.name}",
        f"THUMBNAIL: {final_thumb.name}",
        f"MEASURED LOUDNESS: {measured_lufs if measured_lufs is not None else 'N/A'} LUFS",
        f"HOOK: {hook_text}",
        "",
        "TITLE OPTIONS:",
        *[f"  {i}. {t}" for i, t in enumerate(export_meta.titles, start=1)],
        "",
        "DESCRIPTION:",
        export_meta.description,
        "",
        "HASHTAGS:",
        " ".join(export_meta.hashtags),
        "",
        "PINNED COMMENT QUESTION:",
        export_meta.pinned_comment,
        "",
    ]
    meta_txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
    return export_meta


def write_sources_and_warnings(
    output_dir: Path,
    metadata: VideoMetadata,
    person: str,
    exported_clips: List[ClipExportMetadata],
    warnings: List[str],
    config: AppConfig,
) -> tuple[Path, Path]:
    """Write outputs/<name>/sources.txt and outputs/<name>/warnings.txt."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sources_path = output_dir / "sources.txt"
    src_lines = [
        "SHORTSFORGE SOURCE PROVENANCE",
        "=" * 40,
        f"Subject:           {person}",
        f"Source:            {metadata.source}",
        f"Title:             {metadata.title}",
        f"Channel / Creator: {metadata.channel}",
        f"Show Name:         {metadata.show_name}",
        f"Upload Date:       {metadata.upload_date or 'Unknown'}",
        f"Original Air Date: {metadata.original_air_date or 'Unknown'}",
        f"Year:              {metadata.year or 'Unknown'}",
        "",
        "EXPORTED CLIPS:",
    ]
    for c in exported_clips:
        src_lines.append(
            f"  - {c.output_video} | {c.start:.2f}s - {c.end:.2f}s | Score {c.score}/100 | Hook: {c.hook_text}"
        )
    sources_path.write_text("\n".join(src_lines) + "\n", encoding="utf-8")

    warnings_path = output_dir / "warnings.txt"
    warn_lines = [
        "SHORTSFORGE WARNINGS & RIGHTS NOTES",
        "=" * 40,
        f"RIGHTS NOTE: {config.export.rights_note}",
        "",
    ]
    if warnings:
        warn_lines.append("FLAGGED ITEMS:")
        for w in warnings:
            warn_lines.append(f"  [!] {w}")
    else:
        warn_lines.append("No flagged hooks or audio/visual warnings recorded for this run.")
    warnings_path.write_text("\n".join(warn_lines) + "\n", encoding="utf-8")
    return sources_path, warnings_path
