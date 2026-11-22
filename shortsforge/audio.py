"""Stage 8: Audio assembly (hook SFX, hook-only ambient bed, narrator mix) and 2-pass -14 LUFS loudnorm."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Tuple

from shortsforge.config import AppConfig
from shortsforge.ffmpeg_utils import run_ffmpeg
from shortsforge.sfx import generate_hook_ambient_bed, generate_whoosh_rewind_sfx


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

    run_ffmpeg([
        "-i", str(input_wav),
        "-af", af,
        "-ar", "48000",
        "-ac", "2",
        "-c:a", "pcm_s16le",
        str(output_wav),
    ])

    measured_after = measure_integrated_lufs(output_wav)
    return output_wav, measured_after


def build_master_audio(
    hook_audio_wav: Path,
    main_audio_wav: Path,
    hook_duration: float,
    transition_duration: float,
    closing_duration: float,
    work_dir: Path,
    config: AppConfig,
) -> Tuple[Path, Optional[float]]:
    """Assemble hook + optional hook-only bed + transition SFX + main speaker audio + closing tail, then loudnorm."""
    work_dir.mkdir(parents=True, exist_ok=True)

    if config.audio.transition_sfx_path and Path(config.audio.transition_sfx_path).is_file():
        sfx_path = Path(config.audio.transition_sfx_path)
    else:
        sfx_path = generate_whoosh_rewind_sfx(
            work_dir / "whoosh_rewind_sfx.wav",
            duration=max(0.28, transition_duration + 0.08),
        )

    hook_mixed = work_dir / "hook_audio_mixed.wav"
    if config.audio.hook_bed_enabled:
        if config.audio.hook_bed_path and Path(config.audio.hook_bed_path).is_file():
            bed_path = Path(config.audio.hook_bed_path)
        else:
            bed_path = generate_hook_ambient_bed(
                work_dir / "hook_ambient_bed.wav",
                duration=max(hook_duration + 0.5, 3.0),
            )
        fade_out_start = max(0.05, hook_duration - 0.35)
        bed_vol_db = config.audio.hook_bed_volume_db
        run_ffmpeg([
            "-i", str(hook_audio_wav),
            "-i", str(bed_path),
            "-filter_complex",
            (
                f"[0:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{hook_duration:.3f},asetpts=PTS-STARTPTS[voice];"
                f"[1:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{hook_duration:.3f},asetpts=PTS-STARTPTS,"
                f"volume={bed_vol_db}dB,afade=t=in:st=0:d=0.15,afade=t=out:st={fade_out_start:.3f}:d=0.35[bed];"
                "[voice][bed]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[out]"
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

    closing_silence = work_dir / "closing_silence.wav"
    run_ffmpeg([
        "-f", "lavfi",
        "-i", "anullsrc=r=48000:cl=stereo",
        "-t", f"{closing_duration:.3f}",
        "-c:a", "pcm_s16le",
        str(closing_silence),
    ])

    pre_norm = work_dir / "pre_norm_master.wav"
    run_ffmpeg([
        "-i", str(hook_mixed),
        "-i", str(sfx_seg),
        "-i", str(main_audio_wav),
        "-i", str(closing_silence),
        "-filter_complex",
        (
            "[0:a]aresample=48000,aformat=channel_layouts=stereo,afade=t=out:st="
            f"{max(0.0, hook_duration - 0.04):.3f}:d=0.04[a0];"
            "[1:a]aresample=48000,aformat=channel_layouts=stereo[a1];"
            "[2:a]aresample=48000,aformat=channel_layouts=stereo,afade=t=in:st=0:d=0.03[a2];"
            "[3:a]aresample=48000,aformat=channel_layouts=stereo[a3];"
            "[a0][a1][a2][a3]concat=n=4:v=0:a=1[concat_out]"
        ),
        "-map", "[concat_out]",
        "-c:a", "pcm_s16le",
        str(pre_norm),
    ])

    master_wav = work_dir / "master_loudnorm.wav"
    return normalize_to_lufs(pre_norm, master_wav, config)
