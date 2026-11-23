"""Stage 9: Final H.264 vertical Short compositing, publishing metadata pack, thumbnail pick, sources.txt & warnings.txt."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import cv2

from shortsforge.audio import build_master_audio, normalize_to_lufs
from shortsforge.captions import write_ass_captions
from shortsforge.config import AppConfig, resolve_font
from shortsforge.ffmpeg_utils import (
    FFmpegExecutionError,
    escape_ffmpeg_filter_path,
    extract_stream_info,
    get_h264_encode_args,
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
from shortsforge.quality_check import FaceDetector
from shortsforge.reframe import build_native_reframe_filter, reframe_clip_to_vertical
from shortsforge.tts import synthesize_narrator


EXPORT_META_SYSTEM = """You write YouTube Shorts / Reels / TikTok publishing metadata grounded strictly in the clip and source video.

RULES:
- Never invent facts, quotes, or claims not present in the transcript.
- Return exactly 5 distinct, specific, high-CTR titles matching this exact clip's topic/numbers/takeaway.
- Write a ready-to-paste social `post_caption` (hook + key quote + engaging question + relevant hashtags).
- Write a clear 2-3 sentence `description` summarizing the clip and crediting the original source/channel/show.
- Return 8-9 topic-specific `hashtags` starting with '#' tailored to what is actually discussed in this clip.
- Return 1 engaging `pinned_comment` question grounded in the clip.

Return strict JSON:
{
  "titles": ["Title 1", "Title 2", "Title 3", "Title 4", "Title 5"],
  "post_caption": "Punchy hook + takeaway + question + hashtags",
  "description": "Description crediting the original source...",
  "hashtags": ["#topic1", "#topic2", "#speaker", "#shorts", "#viral", "#tag6", "#tag7", "#tag8"],
  "pinned_comment": "Thought-provoking question for the comments?"
}"""


def _slugify(text: str, max_len: int = 36) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return (slug[:max_len].strip("_")) or "short"


VIDEO_QUALITY_FILTERS: dict[str, str] = {
    "cinematic_grade": "eq=contrast=1.08:brightness=0.015:saturation=1.14:gamma=0.97",
    "sharp_hd": "cas=0.38,unsharp=5:5:0.55:3:3:0.0",
    "denoise_clean": "hqdn3d=2.0:1.5:3.0:2.5",
    "warm_studio": "colorbalance=rs=0.03:gs=0.01:bs=-0.03:rh=0.04:gh=0.01:bh=-0.03,vignette=PI/6",
}


def build_video_enhancement_filter_chain(config: AppConfig, was_upscaled: bool = False) -> str:
    """Build FFmpeg video filter chain for horizontal mirroring and 1-2 user-chosen quality enhancement filters."""
    stages: List[str] = []
    if getattr(config.export, "mirror_video", False):
        stages.append("hflip")

    raw_filters = list(getattr(config.export, "video_filters", []) or [])
    chosen: List[str] = []
    for f_name in raw_filters:
        key = str(f_name).strip().lower()
        if key in VIDEO_QUALITY_FILTERS and key not in chosen:
            chosen.append(key)
        if len(chosen) >= 2:
            break

    # Always run denoise before sharpening when both are selected for cleaner detail
    if "denoise_clean" in chosen and "sharp_hd" in chosen:
        if chosen.index("denoise_clean") > chosen.index("sharp_hd"):
            chosen.remove("denoise_clean")
            chosen.insert(0, "denoise_clean")

    for key in chosen:
        stages.append(VIDEO_QUALITY_FILTERS[key])

    if "sharp_hd" not in chosen and was_upscaled:
        sharpen_val = max(0.0, min(1.0, config.export.sharpen_strength))
        if sharpen_val > 0:
            stages.append(f"cas={sharpen_val:.2f}")

    return ("," + ",".join(stages)) if stages else ""


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


def _intermediate_codec_args(config: AppConfig) -> List[str]:
    if getattr(config.export, "lossless_intermediates", True):
        return ["-c:v", "libx264", "-qp", "0", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-profile:v", "high", "-crf", "12", "-preset", "fast", "-pix_fmt", "yuv420p"]


def _build_hook_segment(
    reframed_main: Path,
    candidate: ClipCandidate,
    hook_text: str,
    work_dir: Path,
    config: AppConfig,
) -> tuple[Path, Path, float]:
    """Build the crux cold-open hook clip with kinetic zoom, flash, contrast punch, and dramatic framing."""
    main_info = extract_stream_info(probe_media(reframed_main))
    fps = float(main_info.get("fps") or config.reframe.output_fps)
    out_w = config.reframe.output_width
    out_h = config.reframe.output_height
    enc_args = _intermediate_codec_args(config)

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
        zoom_end = max(1.04, config.hook.freeze_zoom_end)
        zoom_inc = (zoom_end - 1.0) / total_frames

        run_ffmpeg([
            "-loop", "1",
            "-i", str(freeze_png),
            "-filter_complex",
            (
                f"[0:v]scale={out_w}:{out_h},zoompan=z='min(zoom+{zoom_inc:.6f},{zoom_end:.3f})':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={total_frames}:s={out_w}x{out_h}:fps={fps:.3f},"
                "eq=contrast=1.08:saturation=1.12,vignette=PI/5,fade=t=in:st=0:d=0.14:color=white,format=yuv420p[vout]"
            ),
            "-map", "[vout]",
            "-t", f"{hook_dur:.3f}",
            "-r", f"{fps:.3f}",
            *enc_args,
            "-an",
            str(hook_video),
        ])
        return hook_video, hook_audio, hook_dur

    peak_rel_start = max(0.0, candidate.peak_start - candidate.start)
    peak_dur = max(
        config.hook.cold_open_min_sec,
        min(config.hook.cold_open_max_sec, candidate.peak_end - candidate.peak_start),
    )
    if peak_rel_start + peak_dur > main_info["duration"]:
        peak_rel_start = max(0.0, main_info["duration"] - peak_dur)
        peak_dur = min(peak_dur, main_info["duration"])

    # Smooth kinetic zoom-in (1.0 -> 1.11) + dramatic contrast/vibrance pop + vignette + white opener flash
    # Extract both hook_video and hook_audio in a single FFmpeg pass
    zoom_rate = 0.11 / max(0.5, peak_dur)
    run_ffmpeg([
        "-ss", f"{peak_rel_start:.3f}",
        "-t", f"{peak_dur:.3f}",
        "-i", str(reframed_main),
        "-vf",
        (
            "setpts=PTS-STARTPTS,"
            f"scale=w='trunc({out_w}*(1+{zoom_rate:.5f}*t)/2)*2':h='trunc({out_h}*(1+{zoom_rate:.5f}*t)/2)*2':eval=frame,"
            f"crop={out_w}:{out_h}:(iw-{out_w})/2:(ih-{out_h})/2,"
            "eq=contrast=1.08:saturation=1.12,vignette=PI/5,"
            "fade=t=in:st=0:d=0.12:color=white,format=yuv420p"
        ),
        "-t", f"{peak_dur:.3f}",
        "-r", f"{fps:.3f}",
        *enc_args,
        "-an",
        str(hook_video),
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
    """Create a crisp quick whoosh/whip flash transition into the clip's real start."""
    main_info = extract_stream_info(probe_media(reframed_main))
    fps = float(main_info.get("fps") or config.reframe.output_fps)
    out_w = config.reframe.output_width
    out_h = config.reframe.output_height
    half_d = max(0.05, duration * 0.5)
    enc_args = _intermediate_codec_args(config)

    run_ffmpeg([
        "-ss", "0",
        "-t", f"{duration:.3f}",
        "-i", str(reframed_main),
        "-vf",
        (
            f"scale={out_w}:{out_h},fps={fps:.3f},"
            "gblur=sigma=18:steps=2,"
            f"fade=t=out:st=0:d={half_d:.3f}:color=white,"
            f"fade=t=in:st={half_d:.3f}:d={half_d:.3f}:color=white,"
            "format=yuv420p"
        ),
        "-t", f"{duration:.3f}",
        *enc_args,
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
    """Generate 5 video-specific titles, post caption, credited description, topic hashtags, and comment."""
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
        titles.append(f"{person}: {hook_text}"[:85])

    desc = str(raw.get("description", "")).strip() if isinstance(raw, dict) else ""
    if source_credit.lower() not in desc.lower():
        desc = f"{desc}\n\nSource: {source_credit} ({metadata.title})".strip()

    hashtags = [
        h if str(h).startswith("#") else f"#{h}"
        for h in (raw.get("hashtags", []) if isinstance(raw, dict) else [])
    ][:9]
    defaults = ["#shorts", "#viral", "#podcastclips", "#interview", "#clips"]
    for d in defaults:
        if len(hashtags) >= 8:
            break
        if d not in hashtags:
            hashtags.append(d)

    post_caption = str(raw.get("post_caption", "")).strip() if isinstance(raw, dict) else ""
    if not post_caption:
        post_caption = f"{hook_text}\n\n{candidate.key_quote[:130]}\n\n{' '.join(hashtags[:8])}"

    pinned = (
        str(raw.get("pinned_comment", "")).strip()
        if isinstance(raw, dict)
        else ""
    ) or f"What stood out most to you about what {person} said here?"

    return {
        "titles": titles,
        "post_caption": post_caption,
        "description": desc,
        "hashtags": hashtags[:9],
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
    """Render crisp 9:16 Short: crux hook opener (zoom/flash + tense music + kinetic caption) -> continuous clip + mood music + bold captions."""
    clip_work = work_dir / candidate.id
    clip_work.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(hook_text or candidate.key_quote or candidate.id)
    base_name = f"{candidate.id}_{slug}"
    final_mp4 = output_dir / f"{base_name}.mp4"
    final_thumb = output_dir / f"{base_name}_thumb.jpg"
    meta_json_path = output_dir / f"{base_name}_metadata.json"
    meta_txt_path = output_dir / f"{base_name}_metadata.txt"

    # Remove stale files from prior renders of the same clip_id with a different slug so duplicates never appear
    for old_f in output_dir.glob(f"{candidate.id}_*"):
        if not old_f.name.startswith(f"{base_name}.") and not old_f.name.startswith(f"{base_name}_"):
            old_f.unlink(missing_ok=True)

    source_path = Path(metadata.local_video_path)
    main_audio_wav = clip_work / "main_audio.wav"
    peak_offset_sec = max(0.0, candidate.peak_start - candidate.start)
    slice_dur = max(0.5, candidate.end - candidate.start)

    mood = getattr(candidate, "mood", "tense") or "tense"
    was_upscaled = (metadata.height * (9.0 / 16.0)) < config.reframe.output_width
    pre_ass_filters = build_video_enhancement_filter_chain(config, was_upscaled=was_upscaled)

    use_direct_fast_path = not (config.hook.enabled and candidate.duration >= 5.0)
    cleanup_paths: List[Path] = []
    hook_words: Optional[List[WordTimestamp]] = None
    hook_dur = 0.0
    trans_dur = 0.0

    enc_name, v_enc_args, filter_pix_fmt = get_h264_encode_args(
        crf=config.export.video_crf,
        preset=config.export.video_preset,
        hw_accel=getattr(config.export, "hw_accel", True),
    )

    def _run_encode_with_hw_fallback(
        input_args: List[str],
        fc_template: str,
        audio_idx: int,
        extra_out_args: Optional[List[str]] = None,
    ) -> None:
        extra = extra_out_args or []
        cmd = [
            *input_args,
            "-filter_complex", fc_template.format(pix_fmt=filter_pix_fmt),
            "-map", "[vfinal]",
            "-map", f"{audio_idx}:a:0",
            *v_enc_args,
            *extra,
            "-colorspace", "bt709",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-c:a", "aac",
            "-b:a", config.export.audio_bitrate,
            "-movflags", "+faststart",
            "-shortest",
            str(final_mp4),
        ]
        try:
            run_ffmpeg(cmd)
        except FFmpegExecutionError:
            if enc_name == "libx264":
                raise
            # Seamless fallback to software libx264 if hardware encoder rejects a stream
            _, sw_args, sw_pix = get_h264_encode_args(
                crf=config.export.video_crf,
                preset=config.export.video_preset,
                hw_accel=False,
            )
            sw_cmd = [
                *input_args,
                "-filter_complex", fc_template.format(pix_fmt=sw_pix),
                "-map", "[vfinal]",
                "-map", f"{audio_idx}:a:0",
                *sw_args,
                *extra,
                "-colorspace", "bt709",
                "-color_primaries", "bt709",
                "-color_trc", "bt709",
                "-c:a", "aac",
                "-b:a", config.export.audio_bitrate,
                "-movflags", "+faststart",
                "-shortest",
                str(final_mp4),
            ]
            run_ffmpeg(sw_cmd)

    if use_direct_fast_path:
        # FAST SINGLE-PASS PATH: Zero intermediate MP4, zero 4K Python pixel piping, native FFmpeg face-tracked crop + GPU H.264!
        run_ffmpeg([
            "-ss", f"{candidate.start:.3f}",
            "-i", str(source_path),
            "-t", f"{slice_dur:.3f}",
            "-vn",
            "-ar", "48000",
            "-ac", "2",
            "-c:a", "pcm_s16le",
            str(main_audio_wav),
        ])
        if config.audio.bg_music_enabled:
            master_wav, measured_lufs = build_master_audio(
                hook_audio_wav=main_audio_wav,
                main_audio_wav=main_audio_wav,
                hook_duration=0.2,
                transition_duration=0.0,
                closing_duration=0.0,
                work_dir=clip_work,
                config=config,
                mood=mood,
            )
        else:
            master_wav, measured_lufs = normalize_to_lufs(
                input_wav=main_audio_wav,
                output_wav=clip_work / "master_loudnorm.wav",
                config=config,
            )

        reframe_filter, out_fps_val, _ = build_native_reframe_filter(
            source_video=source_path,
            start_sec=candidate.start,
            end_sec=candidate.end,
            config=config,
            detector=detector,
            thumb_output_jpg=final_thumb,
            peak_offset_sec=peak_offset_sec,
        )

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
            hook_words=None,
            hook_start=candidate.peak_start,
            hook_banner_duration=0.0,
        )
        escaped_ass = escape_ffmpeg_filter_path(ass_path)
        font_path, _ = resolve_font(config, bold=True, preferred_name=config.captions.font_name)
        fontsdir_arg = ""
        if font_path and font_path.parent.exists():
            fontsdir_arg = f":fontsdir='{escape_ffmpeg_filter_path(font_path.parent)}'"

        if ";" in reframe_filter:
            fc_template = (
                f"[0:v]setpts=PTS-STARTPTS,{reframe_filter}{pre_ass_filters},"
                f"ass=filename='{escaped_ass}'{fontsdir_arg},format={{pix_fmt}}[vfinal]"
            )
        else:
            fc_template = (
                f"[0:v]setpts=PTS-STARTPTS,{reframe_filter}{pre_ass_filters},"
                f"ass=filename='{escaped_ass}'{fontsdir_arg},format={{pix_fmt}}[vfinal]"
            )

        input_args = [
            "-hwaccel", "auto",
            "-ss", f"{candidate.start:.3f}",
            "-i", str(source_path),
            "-t", f"{slice_dur:.3f}",
            "-i", str(master_wav),
        ]
        _run_encode_with_hw_fallback(
            input_args=input_args,
            fc_template=fc_template,
            audio_idx=1,
            extra_out_args=["-r", f"{out_fps_val}"],
        )
        cleanup_paths.extend([main_audio_wav, master_wav])
    else:
        # Hook-enabled path: reframe clip, build cold-open hook segment, then single-pass composite with GPU H.264
        reframed_main = clip_work / "reframed_main.mp4"
        reframe_clip_to_vertical(
            source_video=source_path,
            start_sec=candidate.start,
            end_sec=candidate.end,
            output_video=reframed_main,
            config=config,
            detector=detector,
            audio_output_wav=main_audio_wav,
            thumb_output_jpg=final_thumb,
            peak_offset_sec=peak_offset_sec,
        )
        if not main_audio_wav.exists():
            run_ffmpeg([
                "-i", str(reframed_main),
                "-vn",
                "-ar", "48000",
                "-ac", "2",
                "-c:a", "pcm_s16le",
                str(main_audio_wav),
            ])

        cleanup_paths.append(reframed_main)
        hook_video, hook_audio, hook_dur = _build_hook_segment(
            reframed_main=reframed_main,
            candidate=candidate,
            hook_text=hook_text,
            work_dir=clip_work,
            config=config,
        )
        trans_dur = max(0.0, float(config.hook.transition_duration_sec))
        if trans_dur > 0.02:
            trans_video = _build_transition_video(
                reframed_main=reframed_main,
                duration=trans_dur,
                output_path=clip_work / "transition_flash.mp4",
                config=config,
            )
            video_inputs = [hook_video, trans_video, reframed_main]
            cleanup_paths.extend([hook_video, trans_video])
        else:
            trans_dur = 0.0
            video_inputs = [hook_video, reframed_main]
            cleanup_paths.append(hook_video)

        master_wav, measured_lufs = build_master_audio(
            hook_audio_wav=hook_audio,
            main_audio_wav=main_audio_wav,
            hook_duration=hook_dur,
            transition_duration=trans_dur,
            closing_duration=0.0,
            work_dir=clip_work,
            config=config,
            mood=mood,
        )
        hook_words = [
            w for w in transcript.words if w.end >= candidate.peak_start and w.start <= candidate.peak_end
        ]

        main_offset = hook_dur + trans_dur
        clip_words = [
            w for w in transcript.words if w.end >= candidate.start and w.start <= candidate.end
        ]
        ass_path = write_ass_captions(
            output_ass=clip_work / "captions.ass",
            clip_words=clip_words,
            clip_start=candidate.start,
            time_offset=main_offset,
            config=config,
            key_quote=candidate.key_quote,
            hook_text=hook_text,
            hook_words=hook_words,
            hook_start=candidate.peak_start,
            hook_banner_duration=hook_dur,
        )

        escaped_ass = escape_ffmpeg_filter_path(ass_path)
        font_path, _ = resolve_font(config, bold=True, preferred_name=config.captions.font_name)
        fontsdir_arg = ""
        if font_path and font_path.parent.exists():
            fontsdir_arg = f":fontsdir='{escape_ffmpeg_filter_path(font_path.parent)}'"

        input_args = []
        for vp in video_inputs:
            input_args.extend(["-i", str(vp)])
        audio_input_idx = len(video_inputs)
        input_args.extend(["-i", str(master_wav)])

        pts_parts = [f"[{i}:v]setpts=PTS-STARTPTS[v{i}]" for i in range(len(video_inputs))]
        v_labels = "".join(f"[v{i}]" for i in range(len(video_inputs)))
        fc_template = (
            ";".join(pts_parts)
            + f";{v_labels}concat=n={len(video_inputs)}:v=1:a=0{pre_ass_filters},"
            + f"ass=filename='{escaped_ass}'{fontsdir_arg},format={{pix_fmt}}[vfinal]"
        )
        _run_encode_with_hw_fallback(
            input_args=input_args,
            fc_template=fc_template,
            audio_idx=audio_input_idx,
        )

        if not final_thumb.exists():
            _extract_best_vertical_thumbnail(
                reframed_video=reframed_main,
                output_jpg=final_thumb,
                peak_offset_sec=peak_offset_sec,
                detector=detector,
            )

    if not final_thumb.exists() and final_mp4.exists():
        try:
            run_ffmpeg([
                "-ss", f"{min(peak_offset_sec, max(0.1, slice_dur - 0.2)):.3f}",
                "-i", str(final_mp4),
                "-vframes", "1",
                "-q:v", "3",
                str(final_thumb),
            ])
        except Exception:  # noqa: BLE001
            pass

    for tmp_p in cleanup_paths:
        tmp_p.unlink(missing_ok=True)

    pub = _generate_publishing_metadata(candidate, metadata, person, hook_text, llm)
    export_meta = ClipExportMetadata(
        clip_id=candidate.id,
        output_video=final_mp4.name,
        thumbnail=final_thumb.name,
        titles=pub["titles"],
        post_caption=pub["post_caption"],
        description=pub["description"],
        hashtags=pub["hashtags"],
        pinned_comment=pub["pinned_comment"],
        hook_text=hook_text,
        mood=mood,
        start=candidate.start,
        end=candidate.end,
        score=candidate.score,
        lufs=measured_lufs,
    )
    meta_json_path.write_text(export_meta.model_dump_json(indent=2), encoding="utf-8")

    txt_lines = [
        f"CLIP: {candidate.id} ({candidate.start:.2f}s - {candidate.end:.2f}s) | Score: {candidate.score}/100 | Mood: {mood.upper()}",
        f"VIDEO: {final_mp4.name}",
        f"THUMBNAIL: {final_thumb.name}",
        f"MEASURED LOUDNESS: {measured_lufs if measured_lufs is not None else 'N/A'} LUFS",
        f"HOOK: {hook_text}",
        "",
        "TITLE OPTIONS:",
        *[f"  {i}. {t}" for i, t in enumerate(export_meta.titles, start=1)],
        "",
        "READY-TO-POST SOCIAL CAPTION:",
        export_meta.post_caption,
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
