"""Stage 8: Audio assembly (hook SFX, hook-only ambient bed, narrator mix) and 2-pass -14 LUFS loudnorm."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Tuple

from shortsforge.config import AppConfig
from shortsforge.ffmpeg_utils import extract_stream_info, probe_media, run_ffmpeg
from shortsforge.sfx import (
    generate_hook_ambient_bed,
    generate_mood_music_track,
    generate_whoosh_rewind_sfx,
)


def _parse_loudnorm_json(stderr_text: str) -> Optional[dict]:
    matches = re.findall(r"\{[\s\S]*?\"input_i\"[\s\S]*?\}", stderr_text)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return None


def measure_integrated_lufs(media_path: Path) -> Optional[float]:
    """Measure integrated EBU R128 loudness (LUFS) of an audio or video file."""
    proc = run_ffmpeg([
        "-i", str(media_path),
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
        "-f", "null",
        "-",
    ])
    stats = _parse_loudnorm_json(proc.stderr)
    if not stats:
        return None
    try:
        val = float(stats["input_i"])
        return None if val <= -90.0 else round(val, 2)
    except (KeyError, ValueError, TypeError):
        return None


def normalize_to_lufs(
    input_wav: Path,
    output_wav: Path,
    config: AppConfig,
) -> Tuple[Path, Optional[float]]:
    """Run two-pass EBU R128 loudness normalization targeting config.audio.target_lufs (-14 LUFS)."""
    target_i = config.audio.target_lufs
    target_tp = config.audio.true_peak_db
    target_lra = config.audio.lra

    p1 = run_ffmpeg([
        "-i", str(input_wav),
        "-af", f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}:print_format=json",
        "-f", "null",
        "-",
    ])
    stats = _parse_loudnorm_json(p1.stderr)

    if stats and all(k in stats for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")):
        try:
            in_i = float(stats["input_i"])
            if in_i > -70.0:
                af = (
                    f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}:linear=true:"
                    f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:"
                    f"measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}:"
                    f"offset={stats['target_offset']}"
                )
            else:
                af = f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"
        except ValueError:
            af = f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"
    else:
        af = f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"

    p2 = run_ffmpeg([
        "-i", str(input_wav),
        "-af", f"{af}:print_format=json",
        "-ar", "48000",
        "-ac", "2",
        "-c:a", "pcm_s16le",
        str(output_wav),
    ])

    measured_after: Optional[float] = None
    stats_after = _parse_loudnorm_json(p2.stderr)
    if stats_after and "output_i" in stats_after:
        try:
            val = float(stats_after["output_i"])
            if val > -90.0:
                measured_after = round(val, 2)
        except (ValueError, TypeError):
            measured_after = None
    if measured_after is None:
        measured_after = measure_integrated_lufs(output_wav)
    return output_wav, measured_after


def _resolve_custom_music_track(mood: str, config: AppConfig) -> Optional[Path]:
    """Optional folder lookup for custom assets/music/<mood>.mp3|wav before falling back to synthesized music."""
    music_dir = config.root_dir / "assets" / "music"
    if not music_dir.exists():
        return None
    for ext in ("wav", "mp3", "m4a", "ogg"):
        p = music_dir / f"{mood}.{ext}"
        if p.is_file():
            return p
    return None


def build_master_audio(
    hook_audio_wav: Path,
    main_audio_wav: Path,
    hook_duration: float,
    transition_duration: float,
    closing_duration: float,
    work_dir: Path,
    config: AppConfig,
    mood: str = "tense",
) -> Tuple[Path, Optional[float]]:
    """Assemble clean hook opener + main dialogue (no whoosh SFX or music by default), then 2-pass -14 LUFS loudnorm."""
    work_dir.mkdir(parents=True, exist_ok=True)
    pre_norm = work_dir / "pre_norm_master.wav"

    use_hook_bed = bool(getattr(config.audio, "hook_bed_enabled", False))
    use_bg_music = bool(getattr(config.audio, "bg_music_enabled", False))
    use_whoosh = bool(getattr(config.audio, "transition_sfx_enabled", False)) and transition_duration > 0.02

    # Fast optimized path when music and whoosh SFX are disabled (clean original dialogue audio)
    if not use_hook_bed and not use_bg_music and not use_whoosh and closing_duration <= 0.05:
        fade_out_st = max(0.0, hook_duration - 0.03)
        if transition_duration > 0.01:
            filter_graph = (
                f"[0:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{hook_duration:.3f},asetpts=PTS-STARTPTS,"
                f"afade=t=out:st={fade_out_st:.3f}:d=0.03,apad=pad_dur={transition_duration:.3f}[a0];"
                "[1:a]aresample=48000,aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.025[a1];"
                "[a0][a1]concat=n=2:v=0:a=1[concat_out]"
            )
        else:
            filter_graph = (
                f"[0:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{hook_duration:.3f},asetpts=PTS-STARTPTS,"
                f"afade=t=out:st={fade_out_st:.3f}:d=0.03[a0];"
                "[1:a]aresample=48000,aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.025[a1];"
                "[a0][a1]concat=n=2:v=0:a=1[concat_out]"
            )
        run_ffmpeg([
            "-i", str(hook_audio_wav),
            "-i", str(main_audio_wav),
            "-filter_complex", filter_graph,
            "-map", "[concat_out]",
            "-c:a", "pcm_s16le",
            str(pre_norm),
        ])
        master_wav = work_dir / "master_loudnorm.wav"
        return normalize_to_lufs(pre_norm, master_wav, config)

    # 1. Hook Segment Audio
    hook_mixed = work_dir / "hook_audio_mixed.wav"
    if use_hook_bed:
        if config.audio.hook_bed_path and Path(config.audio.hook_bed_path).is_file():
            bed_path = Path(config.audio.hook_bed_path)
        else:
            bed_path = generate_hook_ambient_bed(
                work_dir / "hook_tense_bed.wav",
                duration=max(hook_duration + 0.35, 3.0),
            )
        fade_out_start = max(0.05, hook_duration - 0.25)
        bed_vol_db = config.audio.hook_bed_volume_db
        run_ffmpeg([
            "-i", str(hook_audio_wav),
            "-i", str(bed_path),
            "-filter_complex",
            (
                f"[0:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{hook_duration:.3f},asetpts=PTS-STARTPTS[voice];"
                f"[1:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{hook_duration:.3f},asetpts=PTS-STARTPTS,"
                f"volume={bed_vol_db}dB,afade=t=in:st=0:d=0.10,afade=t=out:st={fade_out_start:.3f}:d=0.25[bed];"
                "[voice]asplit=2[vp][sc];"
                "[bed][sc]sidechaincompress=threshold=0.025:ratio=4:attack=30:release=220:makeup=1[duckbed];"
                "[vp][duckbed]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[out]"
            ),
            "-map", "[out]",
            "-c:a", "pcm_s16le",
            str(hook_mixed),
        ])
    else:
        run_ffmpeg([
            "-i", str(hook_audio_wav),
            "-af", f"aresample=48000,aformat=channel_layouts=stereo,atrim=0:{hook_duration:.3f},asetpts=PTS-STARTPTS",
            "-c:a", "pcm_s16le",
            str(hook_mixed),
        ])

    # 2. Main Clip Audio
    main_info = extract_stream_info(probe_media(main_audio_wav))
    main_dur = max(1.0, float(main_info.get("duration") or 10.0))
    main_mixed = work_dir / "main_audio_mixed.wav"
    if use_bg_music:
        custom_track = _resolve_custom_music_track(mood, config)
        if custom_track is not None:
            music_path = custom_track
        else:
            music_path = generate_mood_music_track(
                work_dir / f"mood_music_{mood}.wav",
                duration=main_dur + 0.5,
                mood=mood,
            )
        bg_vol_db = config.audio.bg_music_volume_db
        bg_fade_out = max(0.1, main_dur - 1.1)
        run_ffmpeg([
            "-i", str(main_audio_wav),
            "-stream_loop", "-1",
            "-i", str(music_path),
            "-filter_complex",
            (
                f"[0:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{main_dur:.3f},asetpts=PTS-STARTPTS[sp_raw];"
                f"[1:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{main_dur:.3f},asetpts=PTS-STARTPTS,"
                f"volume={bg_vol_db}dB,afade=t=in:st=0:d=0.45,afade=t=out:st={bg_fade_out:.3f}:d=1.0[mus];"
                "[sp_raw]asplit=2[sp][sc];"
                "[mus][sc]sidechaincompress=threshold=0.018:ratio=5.5:attack=35:release=320:makeup=1[ducked];"
                "[sp][ducked]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
            ),
            "-map", "[aout]",
            "-c:a", "pcm_s16le",
            str(main_mixed),
        ])
    else:
        main_mixed = main_audio_wav

    seg_inputs = [hook_mixed]
    filter_parts = [
        f"[0:a]aresample=48000,aformat=channel_layouts=stereo,afade=t=out:st={max(0.0, hook_duration - 0.04):.3f}:d=0.04[a0]",
    ]
    concat_labels = "[a0]"

    if use_whoosh:
        if config.audio.transition_sfx_path and Path(config.audio.transition_sfx_path).is_file():
            sfx_path = Path(config.audio.transition_sfx_path)
        else:
            sfx_path = generate_whoosh_rewind_sfx(
                work_dir / "whoosh_rewind_sfx.wav",
                duration=max(0.24, transition_duration + 0.04),
            )
        sfx_seg = work_dir / "transition_sfx_seg.wav"
        sfx_vol_db = config.audio.transition_sfx_volume_db
        run_ffmpeg([
            "-i", str(sfx_path),
            "-af",
            (
                f"aresample=48000,aformat=channel_layouts=stereo,volume={sfx_vol_db}dB,"
                f"apad=whole_dur={transition_duration:.3f},atrim=0:{transition_duration:.3f},asetpts=PTS-STARTPTS"
            ),
            "-c:a", "pcm_s16le",
            str(sfx_seg),
        ])
        idx_sfx = len(seg_inputs)
        seg_inputs.append(sfx_seg)
        filter_parts.append(f"[{idx_sfx}:a]aresample=48000,aformat=channel_layouts=stereo[a{idx_sfx}]")
        concat_labels += f"[a{idx_sfx}]"

    idx_main = len(seg_inputs)
    seg_inputs.append(main_mixed)
    filter_parts.append(f"[{idx_main}:a]aresample=48000,aformat=channel_layouts=stereo,afade=t=in:st=0:d=0.03[a{idx_main}]")
    concat_labels += f"[a{idx_main}]"

    if closing_duration > 0.05:
        closing_silence = work_dir / "closing_silence.wav"
        run_ffmpeg([
            "-f", "lavfi",
            "-i", "anullsrc=r=48000:cl=stereo",
            "-t", f"{closing_duration:.3f}",
            "-c:a", "pcm_s16le",
            str(closing_silence),
        ])
        idx_cl = len(seg_inputs)
        seg_inputs.append(closing_silence)
        filter_parts.append(f"[{idx_cl}:a]aresample=48000,aformat=channel_layouts=stereo[a{idx_cl}]")
        concat_labels += f"[a{idx_cl}]"

    cmd = []
    for p in seg_inputs:
        cmd.extend(["-i", str(p)])
    cmd.extend([
        "-filter_complex",
        ";".join(filter_parts) + f";{concat_labels}concat=n={len(seg_inputs)}:v=0:a=1[concat_out]",
        "-map", "[concat_out]",
        "-c:a", "pcm_s16le",
        str(pre_norm),
    ])
    run_ffmpeg(cmd)

    master_wav = work_dir / "master_loudnorm.wav"
    return normalize_to_lufs(pre_norm, master_wav, config)
