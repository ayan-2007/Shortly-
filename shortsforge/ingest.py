"""Stage 1: Fast URL (yt-dlp) & local file ingest, auto-person detection, cross-project cache reuse, and YouTube json3 caption fetching."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from shortsforge.config import AppConfig
from shortsforge.ffmpeg_utils import extract_stream_info, get_ffmpeg_bin, probe_media, run_ffmpeg
from shortsforge.models import VideoMetadata
from shortsforge.state import ProjectWorkspace


class IngestError(RuntimeError):
    """Raised when media ingest fails with a human-readable, actionable error."""


def is_url(source: str) -> bool:
    parsed = urlparse(source.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def extract_youtube_id(url: str) -> str:
    """Extract canonical 11-char YouTube video ID from youtu.be or youtube.com URLs."""
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        return parsed.path.strip("/").split("/")[0][:11]
    if "youtube.com" in host:
        qs = parse_qs(parsed.query)
        if "v" in qs and qs["v"]:
            return qs["v"][0][:11]
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
            return parts[1][:11]
    return ""


def _classify_yt_dlp_error(exc: Exception, url: str) -> IngestError:
    msg = str(exc).lower()
    if "private video" in msg or "sign in if you've been granted access" in msg:
        return IngestError(
            f"Video is PRIVATE and cannot be downloaded: {url}\n"
            "Make the video accessible or supply cookies / a local media file."
        )
    if any(k in msg for k in ("video unavailable", "has been removed", "account associated", "deleted", "terminated")):
        return IngestError(
            f"Video is REMOVED or UNAVAILABLE: {url}\n"
            "Check that the URL is still live and public."
        )
    if any(k in msg for k in ("sign in to confirm your age", "age-restricted", "age restricted", "inappropriate for some users")):
        return IngestError(
            f"Video is AGE-RESTRICTED: {url}\n"
            "Set ingest.cookies_from_browser in config.yaml or download the video locally."
        )
    if any(k in msg for k in ("not available in your country", "geo-restricted", "blocked it in your country")):
        return IngestError(f"Video is GEO-RESTRICTED in your region: {url}")
    return IngestError(f"Failed to download video from {url}: {exc}")


def extract_show_and_air_date(title: str, description: str, upload_date: str = "") -> Tuple[str, str, str]:
    """Heuristically extract (show_name, original_air_date, year) from video title & description."""
    combined = f"{title}\n{description}"

    air_date = ""
    air_patterns = [
        r"(?:original(?:ly)?\s+aired|air\s*date|broadcast\s*date|recorded)\s*[:\-]?\s*([A-Z][a-z]{2,8}\s+\d{1,2},?\s+\d{4})",
        r"(?:original(?:ly)?\s+aired|air\s*date|broadcast\s*date|recorded)\s*[:\-]?\s*(\d{4}-\d{2}-\d{2})",
        r"(?:original(?:ly)?\s+aired|air\s*date|broadcast\s*date|recorded)\s*[:\-]?\s*(\d{4})",
    ]
    for pat in air_patterns:
        m = re.search(pat, combined, re.IGNORECASE)
        if m:
            air_date = m.group(1).strip()
            break

    year = ""
    if air_date:
        ym = re.search(r"(19\d{2}|20\d{2})", air_date)
        if ym:
            year = ym.group(1)
    if not year:
        ym = re.search(r"[\(\[\|,\s](19[5-9]\d|20[0-3]\d)[\)\]\|\s]?", title)
        if ym:
            year = ym.group(1)
    if not year and upload_date:
        ym = re.search(r"(19\d{2}|20\d{2})", upload_date)
        if ym:
            year = ym.group(1)

    show_name = ""
    show_patterns = [
        r"(?:show|program|podcast)\s*[:\-]\s*([^\n\|]{3,60})",
        r"\bon\s+((?:The\s+)?[A-Z][A-Za-z0-9&'\- ]{2,45}(?:Show|Podcast|Tonight|Live|Interview|Sessions|Talks|Experience))",
        r"\|\s*#?([A-Za-z0-9&'\- ]{2,45}?)(?:\s+with\s+[A-Z][a-z]+.*|\s*\|\s*Chapter.*)?$",
    ]
    for pat in show_patterns:
        m = re.search(pat, title)
        if m:
            cand = m.group(1).strip(" -|:#")
            if len(cand) >= 3:
                show_name = cand
                break
    if not show_name:
        for pat in show_patterns[:1]:
            m = re.search(pat, description, re.IGNORECASE)
            if m:
                show_name = m.group(1).strip(" -|:")
                break

    return show_name, air_date, year


def auto_detect_person_from_title(title: str, channel: str = "", description: str = "") -> str:
    """Automatically extract the speaker/guest's full name from a video title or channel."""
    # Only strip explicit bracketed/emoji intro badges like "[SHOCKING] " or "🔥 EXCLUSIVE: ", never a name's first capital letter!
    clean_title = re.sub(r"^\s*(?:[\(\[][^\)\]]{1,25}[\)\]]|[🔥🚨🎙️❗]+)\s*", "", title).strip()

    name_tok = r"[A-Z](?:[a-z]+|'[A-Z][a-z]+)(?:-[A-Z][a-z]+)?"
    patterns = [
        rf"\bwith\s+({name_tok}(?:\s+{name_tok}){{1,2}})\b",
        rf"^({name_tok}(?:\s+{name_tok}){{1,2}})(?:'s|[:\-]|\s+(?:Speaks|Talks|Discusses|Shares|Answers|Reacts|Opens|Explains|Reveals|Breaks|Interview|Every|How|Why|On)\b)",
        rf"Interview\s+With\s+({name_tok}(?:\s+{name_tok}){{1,2}})\b",
    ]
    for pat in patterns:
        m = re.search(pat, clean_title)
        if m:
            name = m.group(1).strip()
            if name.lower() not in {"the matrix", "late night", "full interview", "emergency interview", "every time", "the man", "the woman", "this man", "this woman"}:
                return name

    for pat in patterns[:1] + patterns[2:]:
        m = re.search(pat, description[:500])
        if m:
            return m.group(1).strip()

    if channel and len(channel.split()) <= 4:
        return channel.strip()
    words = [w for w in re.findall(r"[A-Za-z]+", clean_title) if len(w) > 2][:2]
    return " ".join(w.capitalize() for w in words) or "Featured Speaker"


def auto_detect_project_and_person(source: str, config: AppConfig) -> Tuple[str, str, str, str]:
    """Given ONLY a URL or file path, return (project_slug, person_name, show_name, year).

    If an existing project in projects/ already matches this exact URL/video ID, returns that project's name
    so cached downloads are reused immediately!
    """
    yt_id = extract_youtube_id(source) if is_url(source) else ""
    proj_root = config.projects_root()

    # Check if any existing project already has this exact source or YouTube ID cached
    if proj_root.exists():
        for pdir in sorted(proj_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not pdir.is_dir() or pdir.name.startswith("_"):
                continue
            meta_file = pdir / "metadata.json"
            state_file = pdir / "state.json"
            if meta_file.exists():
                try:
                    m = VideoMetadata.model_validate_json(meta_file.read_text(encoding="utf-8"))
                    same_id = yt_id and extract_youtube_id(m.source) == yt_id
                    if m.source == source or same_id:
                        person = auto_detect_person_from_title(m.title, m.channel, m.description)
                        if state_file.exists():
                            st_data = json.loads(state_file.read_text(encoding="utf-8"))
                            if st_data.get("person") and st_data["person"] != "Na":
                                person = st_data["person"]
                        return pdir.name, person, m.show_name, m.year
                except Exception:  # noqa: BLE001
                    continue

    if is_url(source):
        import yt_dlp

        ydl_opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
        js_rt = _detect_js_runtimes()
        if js_rt:
            ydl_opts["js_runtimes"] = js_rt
            ydl_opts["remote_components"] = ["ejs:github"]
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source, download=False)
            title = str(info.get("title") or "interview")
            channel = str(info.get("channel") or info.get("uploader") or "")
            desc = str(info.get("description") or "")
            raw_date = str(info.get("upload_date") or "")
            person = auto_detect_person_from_title(title, channel, desc)
            show_name, _, year = extract_show_and_air_date(title, desc, raw_date)
            slug_base = re.sub(r"[^a-z0-9]+", "_", f"{person}_{yt_id or title[:18]}".lower()).strip("_")
            return slug_base[:32] or "short_project", person, show_name or channel, year
        except Exception:  # noqa: BLE001
            slug = f"yt_{yt_id.lower()}" if yt_id else f"video_{int(time.time()) % 100000}"
            return slug, "Featured Guest", "", ""

    src_path = Path(source)
    stem = src_path.stem.replace("_", " ").replace("-", " ")
    person = auto_detect_person_from_title(stem)
    slug = re.sub(r"[^a-z0-9]+", "_", src_path.stem.lower()).strip("_")[:28] or "local_video"
    return slug, person, "Local Archive", ""


def extract_wav_audio(video_path: Path, output_wav: Path) -> Path:
    """Extract 16kHz mono PCM WAV for faster-whisper transcription and audio analysis."""
    if output_wav.exists() and output_wav.stat().st_size > 1024:
        return output_wav
    run_ffmpeg([
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output_wav),
    ])
    return output_wav


def _detect_js_runtimes() -> Dict[str, Dict[str, str]]:
    """Auto-detect available JS runtimes (deno/node/bun) for full YouTube yt-dlp extraction."""
    runtimes: Dict[str, Dict[str, str]] = {}
    for rt in ("deno", "node", "bun"):
        found = shutil.which(rt)
        if found:
            runtimes[rt] = {"path": found}
    return runtimes


def _build_format_selector(config: AppConfig, duration_sec: float = 0.0) -> str:
    """Build an original-quality-preserving format selector (4K/1440p on normal videos, 1440p/1080p minimum on long podcasts)."""
    h = max(1080, int(config.ingest.max_video_height))
    if duration_sec > 2700 and h > 1440:
        h = 1440
    min_h = max(720, int(getattr(config.ingest, "min_preferred_height", 1080)))
    if config.ingest.prefer_h264:
        return (
            f"bestvideo[height<={h}][height>={min_h}][vcodec^=avc1]+bestaudio[ext=m4a]/"
            f"bestvideo[height<={h}][height>={min_h}][vcodec!^=av01]+bestaudio/"
            f"bestvideo[height<={h}][vcodec^=avc1]+bestaudio[ext=m4a]/"
            f"bestvideo[height<={h}][vcodec!^=av01]+bestaudio/"
            f"bestvideo[height<={h}]+bestaudio/"
            f"best[height<={h}]/best"
        )
    # Prioritize maximum crispness (2160p/1440p/1080p VP9 or H.264, avoiding slow AV1 software decode)
    return (
        f"bestvideo[height<={h}][height>={min_h}][vcodec!^=av01]+bestaudio[ext=m4a]/"
        f"bestvideo[height<={h}][height>={min_h}][vcodec!^=av01]+bestaudio/"
        f"bestvideo[height<={h}][vcodec!^=av01]+bestaudio/"
        f"bestvideo[height<={h}]+bestaudio/"
        f"best[height<={h}]/best"
    )


def _fetch_yt_json3_captions(info: dict, dest_path: Path) -> Optional[Path]:
    """Download YouTube's word-timed json3 captions in ~1s if available."""
    if dest_path.exists() and dest_path.stat().st_size > 512:
        return dest_path

    for field in ("subtitles", "automatic_captions"):
        tracks = info.get(field) or {}
        en_keys = [k for k in ("en-orig", "en", "en-US", "en-GB") if k in tracks]
        if not en_keys:
            en_keys = [k for k in tracks if k.startswith("en")]
        for k in en_keys:
            fmts = tracks.get(k) or []
            j3_urls = [f.get("url") for f in fmts if f.get("ext") == "json3" and f.get("url")]
            if j3_urls:
                try:
                    raw = urllib.request.urlopen(j3_urls[0], timeout=15).read()  # noqa: S310
                except Exception:  # noqa: BLE001
                    try:
                        import ssl

                        raw = urllib.request.urlopen(  # noqa: S310
                            j3_urls[0],
                            context=ssl._create_unverified_context(),
                            timeout=15,
                        ).read()
                    except Exception:  # noqa: BLE001
                        continue
                if len(raw) > 256:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    dest_path.write_bytes(raw)
                    return dest_path
    return None


def _reuse_existing_download_if_any(url: str, workspace: ProjectWorkspace, config: AppConfig) -> Optional[Path]:
    """If another project folder already downloaded this exact YouTube video, reuse it immediately."""
    yt_id = extract_youtube_id(url)
    if not yt_id:
        return None
    proj_root = config.projects_root()
    if not proj_root.exists():
        return None
    for pdir in proj_root.iterdir():
        if not pdir.is_dir() or pdir.resolve() == workspace.project_dir.resolve():
            continue
        meta_path = pdir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            m = VideoMetadata.model_validate_json(meta_path.read_text(encoding="utf-8"))
            if extract_youtube_id(m.source) == yt_id and Path(m.local_video_path).is_file():
                src_vid = Path(m.local_video_path)
                dst_vid = workspace.source_dir / src_vid.name
                if not dst_vid.exists():
                    try:
                        os.link(src_vid, dst_vid)
                    except OSError:
                        shutil.copy2(src_vid, dst_vid)
                src_wav = pdir / "source" / "audio_16k.wav"
                dst_wav = workspace.source_dir / "audio_16k.wav"
                if src_wav.is_file() and not dst_wav.exists():
                    try:
                        os.link(src_wav, dst_wav)
                    except OSError:
                        shutil.copy2(src_wav, dst_wav)
                return dst_vid
        except Exception:  # noqa: BLE001
            continue
    return None


def ingest_source(
    source: str,
    workspace: ProjectWorkspace,
    config: AppConfig,
    show_override: str = "",
    year_override: str = "",
    force: bool = False,
    on_progress: Optional[Callable[[str], None]] = None,
) -> VideoMetadata:
    """Ingest a URL or local video file, caching source media, YouTube json3 captions, 16kHz WAV, and metadata.json."""
    if workspace.metadata_path.exists():
        existing = VideoMetadata.model_validate_json(workspace.metadata_path.read_text(encoding="utf-8"))
        same_src = existing.source == source or (
            is_url(source) and extract_youtube_id(existing.source) == extract_youtube_id(source) and extract_youtube_id(source) != ""
        )
        need_hd_upgrade = force and is_url(source) and existing.height > 0 and existing.height < 900 and config.ingest.max_video_height >= 1080
        if same_src and not need_hd_upgrade and Path(existing.local_video_path).exists() and Path(existing.local_audio_path).exists():
            if on_progress:
                on_progress("Loaded cached source video & audio.")
            # Enrich with YouTube word-level captions + Most Replayed heatmap if not yet cached
            if is_url(source) and (force or not (workspace.source_dir / "yt_captions.json3").exists() or not existing.heatmap):
                try:
                    import yt_dlp

                    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
                        info = ydl.extract_info(source, download=False)
                    _fetch_yt_json3_captions(info, workspace.source_dir / "yt_captions.json3")
                    raw_hm = info.get("heatmap") or []
                    if raw_hm and not existing.heatmap:
                        existing.heatmap = [
                            {
                                "start_time": float(h.get("start_time", 0.0)),
                                "end_time": float(h.get("end_time", 0.0)),
                                "value": float(h.get("value", 0.0)),
                            }
                            for h in raw_hm
                        ]
                    raw_ch = info.get("chapters") or []
                    if raw_ch and not existing.chapters:
                        existing.chapters = [
                            {
                                "start_time": float(c.get("start_time", 0.0)),
                                "end_time": float(c.get("end_time", 0.0)),
                                "title": str(c.get("title") or ""),
                            }
                            for c in raw_ch
                        ]
                    workspace.metadata_path.write_text(existing.model_dump_json(indent=2), encoding="utf-8")
                except Exception:  # noqa: BLE001
                    pass
            return existing
        if need_hd_upgrade and Path(existing.local_video_path).exists():
            if on_progress:
                on_progress(f"Upgrading previously {existing.height}p source video to HD ({config.ingest.max_video_height}p max)...")
            Path(existing.local_video_path).unlink(missing_ok=True)

    if is_url(source):
        meta = _ingest_url(source, workspace, config, on_progress=on_progress)
    else:
        if on_progress:
            on_progress("Inspecting local media file...")
        meta = _ingest_local(source, workspace)

    probe = probe_media(meta.local_video_path)
    info = extract_stream_info(probe)
    if not info["has_video"]:
        raise IngestError(f"Input media has no video stream: {meta.local_video_path}")
    if not info["has_audio"]:
        raise IngestError(f"Input media has no audio stream: {meta.local_video_path}")

    meta.duration = info["duration"]
    meta.width = info["width"]
    meta.height = info["height"]
    meta.fps = info["fps"]
    meta.aspect_ratio = info["aspect_ratio"]

    parsed_show, parsed_air, parsed_year = extract_show_and_air_date(
        meta.title, meta.description, meta.upload_date
    )
    meta.show_name = show_override or parsed_show or meta.channel or "Interview"
    meta.original_air_date = parsed_air
    meta.year = year_override or parsed_year

    wav_path = workspace.source_dir / "audio_16k.wav"
    if not wav_path.exists():
        if on_progress:
            on_progress(f"Extracting 16kHz audio track ({meta.duration / 60:.1f} min)...")
        extract_wav_audio(Path(meta.local_video_path), wav_path)
    meta.local_audio_path = str(wav_path.resolve())

    workspace.metadata_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    return meta


def _ingest_url(
    url: str,
    workspace: ProjectWorkspace,
    config: AppConfig,
    on_progress: Optional[Callable[[str], None]] = None,
) -> VideoMetadata:
    import yt_dlp

    ffmpeg_dir = str(Path(get_ffmpeg_bin()).parent)
    js_runtimes = _detect_js_runtimes()

    # 1. Fast metadata + json3 caption probe first (takes ~2s)
    probe_opts: Dict = {
        "ffmpeg_location": ffmpeg_dir,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    if js_runtimes:
        probe_opts["js_runtimes"] = js_runtimes
        probe_opts["remote_components"] = ["ejs:github"]
    if config.ingest.cookies_from_browser.strip():
        probe_opts["cookiesfrombrowser"] = (config.ingest.cookies_from_browser.strip(),)

    if on_progress:
        on_progress("Fetching YouTube video metadata & word-level captions...")

    try:
        with yt_dlp.YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # noqa: BLE001
        raise _classify_yt_dlp_error(exc, url) from exc

    # Save YouTube word-level json3 captions immediately if available
    j3_file = _fetch_yt_json3_captions(info, workspace.source_dir / "yt_captions.json3")
    if j3_file and on_progress:
        on_progress("Fetched YouTube word-level transcript in 1s!")

    # Check if video file is already in this workspace or another project folder
    existing_files = [
        p for p in sorted(workspace.source_dir.glob("source_video.*"))
        if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"} and ".part" not in p.name
    ]
    video_path: Optional[Path] = existing_files[0] if existing_files else None
    if video_path is None:
        video_path = _reuse_existing_download_if_any(url, workspace, config)
        if video_path and on_progress:
            on_progress(f"Reused cached video from existing project ({video_path.name})!")

    if video_path is None:
        duration_sec = float(info.get("duration") or 0.0)
        out_tmpl = str(workspace.source_dir / "source_video.%(ext)s")
        last_emit = {"t": 0.0}

        def _progress_hook(d: dict) -> None:
            if not on_progress:
                return
            now = time.time()
            status = d.get("status")
            if status == "downloading" and (now - last_emit["t"]) >= 1.5:
                last_emit["t"] = now
                pct = str(d.get("_percent_str") or "").strip()
                spd = str(d.get("_speed_str") or "").strip()
                eta = str(d.get("_eta_str") or "").strip()
                on_progress(f"Downloading video: {pct} @ {spd} (ETA {eta})")
            elif status == "finished":
                on_progress("Stream download finished; merging video + audio with FFmpeg...")

        ydl_opts: Dict = {
            **probe_opts,
            "format": _build_format_selector(config, duration_sec=duration_sec),
            "merge_output_format": "mp4",
            "outtmpl": out_tmpl,
            "continuedl": True,
            "retries": max(3, config.ingest.retries),
            "fragment_retries": max(3, config.ingest.retries),
            "concurrent_fragment_downloads": max(4, config.ingest.concurrent_fragments),
            "socket_timeout": 30,
            "progress_hooks": [_progress_hook],
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as exc:  # noqa: BLE001
            raise _classify_yt_dlp_error(exc, url) from exc

        candidates = sorted(workspace.source_dir.glob("source_video.*"))
        video_files = [
            p for p in candidates
            if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"} and ".part" not in p.name
        ]
        if not video_files:
            raise IngestError(f"yt-dlp completed but no video file was found in {workspace.source_dir}")
        video_path = video_files[0]

    raw_date = str(info.get("upload_date") or "")
    formatted_date = (
        f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}" if len(raw_date) == 8 else raw_date
    )

    heatmap = [
        {
            "start_time": float(h.get("start_time", 0.0)),
            "end_time": float(h.get("end_time", 0.0)),
            "value": float(h.get("value", 0.0)),
        }
        for h in (info.get("heatmap") or [])
    ]
    chapters = [
        {
            "start_time": float(c.get("start_time", 0.0)),
            "end_time": float(c.get("end_time", 0.0)),
            "title": str(c.get("title") or ""),
        }
        for c in (info.get("chapters") or [])
    ]

    return VideoMetadata(
        source=url,
        source_type="url",
        title=str(info.get("title") or video_path.stem),
        channel=str(info.get("channel") or info.get("uploader") or ""),
        upload_date=formatted_date,
        description=str(info.get("description") or ""),
        local_video_path=str(video_path.resolve()),
        heatmap=heatmap,
        chapters=chapters,
    )


def _ingest_local(source: str, workspace: ProjectWorkspace) -> VideoMetadata:
    src_path = Path(source).expanduser().resolve()
    if not src_path.is_file():
        raise IngestError(f"Local source file not found: {src_path}")

    cached_video = workspace.source_dir / f"source_video{src_path.suffix.lower() or '.mp4'}"
    if not cached_video.exists() or cached_video.stat().st_size != src_path.stat().st_size:
        shutil.copy2(src_path, cached_video)

    probe = probe_media(cached_video)
    tags = probe.get("format", {}).get("tags", {}) or {}
    title = tags.get("title") or src_path.stem.replace("_", " ").replace("-", " ")
    channel = tags.get("artist") or tags.get("album") or tags.get("author") or "Local Archive"
    upload_date = tags.get("creation_time") or tags.get("date") or ""
    description = tags.get("comment") or tags.get("description") or ""

    return VideoMetadata(
        source=str(src_path),
        source_type="local",
        title=str(title),
        channel=str(channel),
        upload_date=str(upload_date)[:10],
        description=str(description),
        local_video_path=str(cached_video.resolve()),
    )
