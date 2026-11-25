"""End-to-end ShortsForge pipeline orchestrator (`new`, `render`, and 1-click link `autopilot`)."""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from shortsforge.config import AppConfig, apply_aspect_ratio
from shortsforge.export import render_single_short, write_sources_and_warnings
from shortsforge.hooks import generate_verified_hooks
from shortsforge.ingest import (
    auto_detect_person_from_title,
    auto_detect_project_and_person,
    download_clip_slices,
    ingest_source,
    resolve_candidate_media,
)
from shortsforge.llm import LLMClient
from shortsforge.models import (
    ClipCandidate,
    ClipExportMetadata,
    HookSet,
    HookVariant,
    ProjectState,
    Transcript,
    VideoMetadata,
)
from shortsforge.moments import build_batch_parts, find_best_moments
from shortsforge.quality_check import FaceDetector, evaluate_candidate_quality
from shortsforge.review import (
    generate_review_html,
    load_decisions,
    save_auto_decisions,
    serve_review_page,
)
from shortsforge.state import ProjectWorkspace
from shortsforge.transcribe import transcribe_audio


console = Console()


def _save_candidates(workspace: ProjectWorkspace, candidates: List[ClipCandidate]) -> None:
    payload = [c.model_dump() for c in candidates]
    workspace.candidates_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_candidates(workspace: ProjectWorkspace) -> List[ClipCandidate]:
    if not workspace.candidates_path.exists():
        return []
    raw = json.loads(workspace.candidates_path.read_text(encoding="utf-8"))
    return [ClipCandidate.model_validate(item) for item in raw]


def run_new_project(
    project_name: str,
    source: str,
    person: str,
    config: AppConfig,
    auto: bool = False,
    serve: bool = False,
    min_score: Optional[int] = None,
    max_clips: Optional[int] = None,
    show_override: str = "",
    year_override: str = "",
    deceased: bool = False,
    force: bool = False,
    on_progress: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    mode: str = "viral",
    part_duration_sec: float = 0.0,
    aspect_ratio: Optional[str] = None,
    playback_speed: Optional[float] = None,
) -> Path:
    """Execute Stages 1-4 (Ingest -> Transcribe -> Moments/Batch Split + Quality Gate -> Hooks) & build review.html."""
    run_mode = "batch" if str(mode or "viral").strip().lower() == "batch" else "viral"
    if aspect_ratio in ("9:16", "16:9"):
        apply_aspect_ratio(config, aspect_ratio)
    if playback_speed is not None:
        config.export.playback_speed = max(0.5, min(2.0, float(playback_speed)))

    if run_mode == "batch":
        batch_max_parts = 0 if (max_clips is None or max_clips <= 0) else max(1, min(50, int(max_clips)))
        part_dur_val = max(0.0, float(part_duration_sec or 0.0))
    else:
        batch_max_parts = 0
        part_dur_val = 0.0
        if max_clips is not None:
            config.moments.max_candidates = 15 if max_clips <= 0 else max(1, min(15, max_clips))
        else:
            config.moments.max_candidates = max(1, min(15, config.moments.max_candidates or 15))
    if min_score is not None:
        config.review.auto_score_threshold = min_score
    deceased_flag = deceased or config.hook.deceased_subject

    workspace = ProjectWorkspace(
        project_dir=config.projects_root() / project_name,
        output_dir=config.outputs_root() / project_name,
    )

    state = workspace.load_state()
    mode_changed = False
    if state is None or force:
        state = ProjectState(
            project_name=project_name,
            person=person,
            source=source,
            show_override=show_override,
            year_override=year_override,
            deceased=deceased_flag,
            mode=run_mode,
            aspect_ratio=config.reframe.aspect_ratio,
            part_duration_sec=part_dur_val,
            playback_speed=float(config.export.playback_speed),
        )
        workspace.save_state(state)
    else:
        mode_changed = (
            state.mode != run_mode
            or (run_mode == "batch" and abs(float(state.part_duration_sec or 0.0) - part_dur_val) > 0.5)
        )
        if person:
            state.person = person
        state.source = source
        state.deceased = deceased_flag
        state.mode = run_mode
        state.aspect_ratio = config.reframe.aspect_ratio
        state.part_duration_sec = part_dur_val
        state.playback_speed = float(config.export.playback_speed)
        if show_override:
            state.show_override = show_override
        if year_override:
            state.year_override = year_override
        workspace.save_state(state)

    llm = LLMClient(config=config, workspace=workspace)

    with Progress(
        SpinnerColumn("line"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        t_ingest = progress.add_task("1/4 Ingesting source media & metadata...", total=1)
        if on_progress:
            on_progress("[1/4] Ingesting source video & metadata...")
        allow_clip_only = bool(getattr(config.ingest, "clip_only_download", True)) and run_mode != "batch"
        metadata = ingest_source(
            source=source,
            workspace=workspace,
            config=config,
            show_override=state.show_override,
            year_override=state.year_override,
            force=force or (mode_changed and run_mode == "batch"),
            on_progress=on_progress,
            allow_clip_only=allow_clip_only,
        )
        # Auto-refine speaker name from actual video title if placeholder was used
        if not state.person or state.person.lower() in {"na", "speaker", "featured guest", "featured speaker"}:
            state.person = auto_detect_person_from_title(metadata.title, metadata.channel, metadata.description)
            person = state.person
            workspace.save_state(state)

        workspace.mark_step_complete(state, "ingest")
        progress.advance(t_ingest)

        t_trans = progress.add_task("2/4 Transcribing with word timestamps...", total=1)
        if on_progress:
            on_progress(f"[2/4] Preparing word-level transcript ({metadata.duration / 60:.1f} min video)...")
        if metadata.clip_only_mode:
            transcript_opt: Optional[Transcript] = None
            if (workspace.source_dir / "yt_captions.json3").exists():
                try:
                    transcript_opt = transcribe_audio(
                        audio_path=workspace.source_dir / "audio_16k.wav",
                        workspace=workspace,
                        config=config,
                        force=force,
                        on_progress=on_progress,
                    )
                except Exception:  # noqa: BLE001
                    transcript_opt = None
            if transcript_opt is None or not transcript_opt.sentences:
                if on_progress:
                    on_progress("[2/4] YouTube captions unavailable; downloading full media for Whisper transcription...")
                metadata = ingest_source(
                    source=source,
                    workspace=workspace,
                    config=config,
                    show_override=state.show_override,
                    year_override=state.year_override,
                    force=True,
                    on_progress=on_progress,
                    allow_clip_only=False,
                )
                transcript = transcribe_audio(
                    audio_path=Path(metadata.local_audio_path),
                    workspace=workspace,
                    config=config,
                    force=force,
                    on_progress=on_progress,
                )
            else:
                transcript = transcript_opt
        else:
            transcript = transcribe_audio(
                audio_path=Path(metadata.local_audio_path),
                workspace=workspace,
                config=config,
                force=force,
                on_progress=on_progress,
            )
        workspace.mark_step_complete(state, "transcribe")
        if on_progress:
            on_progress(f"[2/4] Transcript ready: {len(transcript.words)} words, {len(transcript.sentences)} sentences.")
        progress.advance(t_trans)

        candidates: List[ClipCandidate] = []
        if not force and not mode_changed and "hooks" in state.completed_steps and workspace.candidates_path.exists():
            cached_cands = _load_candidates(workspace)
            if run_mode == "batch":
                if cached_cands and (batch_max_parts <= 0 or len(cached_cands) == batch_max_parts):
                    candidates = cached_cands
                    t_mom = progress.add_task("3/4 Loaded cached batch parts & quality checks", total=1)
                    progress.advance(t_mom)
                    t_hooks = progress.add_task("4/4 Loaded batch part titles", total=1)
                    progress.advance(t_hooks)
            else:
                # Invalidate stale caches that had too few (<=4 on a long podcast) or too many (>15) clips
                stale_count = (metadata.duration >= 600.0 and len(cached_cands) <= 4) or len(cached_cands) > 15
                if cached_cands and not stale_count:
                    candidates = cached_cands
                    t_mom = progress.add_task("3/4 Loaded cached candidate clips & quality checks", total=1)
                    progress.advance(t_mom)
                    t_hooks = progress.add_task("4/4 Loaded cached verified hooks", total=1)
                    progress.advance(t_hooks)
        if not candidates:
            t_mom = progress.add_task("3/4 Finding best moments & checking video/audio quality...", total=2)
            if run_mode == "batch":
                if on_progress:
                    on_progress("[3/4] Splitting video into sequential batch parts...")
                candidates = build_batch_parts(
                    transcript=transcript,
                    metadata=metadata,
                    part_duration_sec=part_dur_val,
                    max_parts=batch_max_parts,
                )
                if on_progress:
                    on_progress(f"[3/4] Created {len(candidates)} sequential part(s) ({metadata.duration / 60:.1f} min video).")
            else:
                if on_progress:
                    cap_val = max(1, min(15, config.moments.max_candidates or 12))
                    on_progress(f"[3/4] Finding balanced sweet spot (up to {cap_val} viral moments)...")
                candidates = find_best_moments(
                    transcript=transcript,
                    metadata=metadata,
                    person=person,
                    llm=llm,
                    config=config,
                )
            progress.advance(t_mom)

            if metadata.clip_only_mode and candidates:
                metadata = download_clip_slices(
                    candidates=candidates,
                    metadata=metadata,
                    workspace=workspace,
                    config=config,
                    on_progress=on_progress,
                    should_cancel=should_cancel,
                )
                _save_candidates(workspace, candidates)

            if on_progress:
                on_progress(f"[3/4] Inspecting video frames & audio on {len(candidates)} candidate clip(s)...")

            has_shared_source = bool(metadata.local_video_path and Path(metadata.local_video_path).is_file())
            video_path_obj = Path(metadata.local_video_path) if has_shared_source else Path()
            q_workers = min(4, max(1, (os.cpu_count() or 4) // 2), max(1, len(candidates)))

            if q_workers <= 1:
                detector = FaceDetector()
                shared_cap = cv2.VideoCapture(str(video_path_obj)) if has_shared_source else None
                try:
                    q_results = []
                    for cand in candidates:
                        vid_p, aud_p, _, _, _, _ = resolve_candidate_media(cand, metadata)
                        is_per_clip = bool(cand.clip_video_path and Path(cand.clip_video_path).is_file())
                        t_off = float(cand.clip_time_offset or 0.0) if is_per_clip else 0.0
                        thumb_path = workspace.thumbs_dir / f"{cand.id}_preview.jpg"
                        qreport = evaluate_candidate_quality(
                            candidate=cand,
                            video_path=vid_p,
                            audio_wav_path=aud_p,
                            config=config,
                            detector=detector,
                            thumb_output_path=thumb_path,
                            video_cap=(
                                None
                                if is_per_clip
                                else (shared_cap if (shared_cap is not None and shared_cap.isOpened()) else None)
                            ),
                            time_offset=t_off,
                        )
                        q_results.append((cand, thumb_path, qreport))
                finally:
                    if shared_cap is not None:
                        shared_cap.release()
                    detector.close()
            else:
                tls = threading.local()
                created_caps: List[cv2.VideoCapture] = []
                created_dets: List[FaceDetector] = []
                res_lock = threading.Lock()

                def _eval_cand(cand: ClipCandidate) -> Tuple[ClipCandidate, Path, object]:
                    det_w = getattr(tls, "det", None)
                    if det_w is None:
                        det_w = FaceDetector()
                        tls.det = det_w
                        with res_lock:
                            created_dets.append(det_w)

                    vid_p, aud_p, _, _, _, _ = resolve_candidate_media(cand, metadata)
                    is_per_clip = bool(cand.clip_video_path and Path(cand.clip_video_path).is_file())
                    t_off = float(cand.clip_time_offset or 0.0) if is_per_clip else 0.0

                    cap_arg = None
                    if not is_per_clip and has_shared_source:
                        cap_w = getattr(tls, "cap", None)
                        if cap_w is None:
                            cap_w = cv2.VideoCapture(str(video_path_obj))
                            tls.cap = cap_w
                            with res_lock:
                                created_caps.append(cap_w)
                        if cap_w.isOpened():
                            cap_arg = cap_w

                    thumb_path = workspace.thumbs_dir / f"{cand.id}_preview.jpg"
                    qreport = evaluate_candidate_quality(
                        candidate=cand,
                        video_path=vid_p,
                        audio_wav_path=aud_p,
                        config=config,
                        detector=det_w,
                        thumb_output_path=thumb_path,
                        video_cap=cap_arg,
                        time_offset=t_off,
                    )
                    return cand, thumb_path, qreport

                try:
                    with ThreadPoolExecutor(max_workers=q_workers) as pool:
                        q_results = list(pool.map(_eval_cand, candidates))
                finally:
                    for cap_w in created_caps:
                        cap_w.release()
                    for det_w in created_dets:
                        det_w.close()

            for cand, thumb_path, qreport in q_results:
                cand.quality = qreport
                cand.rejected_by_quality = False if run_mode == "batch" else (not qreport.passed)
                if thumb_path.exists():
                    cand.thumbnail_path = str(thumb_path.resolve())
                if run_mode != "batch":
                    for w in qreport.warnings:
                        workspace.add_warning(state, w)
                    if not qreport.passed:
                        workspace.add_warning(
                            state,
                            f"{cand.id} rejected by quality gate: {'; '.join(qreport.rejection_reasons)}",
                        )

            _save_candidates(workspace, candidates)
            workspace.mark_step_complete(state, "moments")
            progress.advance(t_mom)

            t_hooks = progress.add_task(
                "4/4 Preparing part labels..." if run_mode == "batch" else "4/4 Generating & verifying 3-second hooks (2nd-pass truth check)...",
                total=max(1, len(candidates)),
            )
            if run_mode == "batch":
                short_title = " ".join((metadata.title or "Video").split()[:5]).strip(" -|:")
                for idx_c, cand in enumerate(candidates, start=1):
                    part_label = f"Part {idx_c}: {short_title}" if short_title else f"Part {idx_c}"
                    part_label = " ".join(part_label.split()[:9])
                    cand.hooks = HookSet(
                        variants=[HookVariant(rank=1, text=part_label, predicted_retention=95)],
                        attempts=1,
                    )
                    progress.advance(t_hooks)
            elif len(candidates) > 1 and getattr(llm, "is_remote_provider", lambda: False)():
                if on_progress:
                    on_progress(f"[4/4] Generating & verifying hooks concurrently for {len(candidates)} clips...")

                def _gen_hook_item(pair: Tuple[int, ClipCandidate]) -> Tuple[int, HookSet]:
                    idx_c, cand_item = pair
                    hs = generate_verified_hooks(
                        candidate=cand_item,
                        person=person,
                        llm=llm,
                        config=config,
                        deceased_subject=deceased_flag,
                    )
                    return idx_c, hs

                hook_results: dict[int, HookSet] = {}
                with ThreadPoolExecutor(max_workers=min(4, len(candidates))) as pool:
                    fut_map = {
                        pool.submit(_gen_hook_item, (idx_c, cand)): (idx_c, cand)
                        for idx_c, cand in enumerate(candidates)
                    }
                    done_n = 0
                    for fut in as_completed(fut_map):
                        idx_c, hs = fut.result()
                        hook_results[idx_c] = hs
                        done_n += 1
                        if on_progress:
                            on_progress(
                                f"[4/4] Verified hooks for {candidates[idx_c].id} ({done_n}/{len(candidates)})..."
                            )
                        progress.advance(t_hooks)

                for idx_c, cand in enumerate(candidates):
                    hook_set = hook_results[idx_c]
                    cand.hooks = hook_set
                    if hook_set.flagged_for_review:
                        workspace.add_warning(
                            state,
                            f"{cand.id} hook FLAGGED after {hook_set.attempts} failed verifications: {hook_set.flag_reason}",
                        )
            else:
                for idx_c, cand in enumerate(candidates, start=1):
                    if on_progress:
                        on_progress(f"[4/4] Generating & verifying 5 hooks for {cand.id} ({idx_c}/{len(candidates)})...")
                    hook_set = generate_verified_hooks(
                        candidate=cand,
                        person=person,
                        llm=llm,
                        config=config,
                        deceased_subject=deceased_flag,
                    )
                    cand.hooks = hook_set
                    if hook_set.flagged_for_review:
                        workspace.add_warning(
                            state,
                            f"{cand.id} hook FLAGGED after {hook_set.attempts} failed verifications: {hook_set.flag_reason}",
                        )
                    progress.advance(t_hooks)
            if not candidates:
                progress.advance(t_hooks)

            _save_candidates(workspace, candidates)
            workspace.mark_step_complete(state, "hooks")

    review_path = generate_review_html(
        project_name=project_name,
        person=person,
        metadata=metadata,
        candidates=candidates,
        workspace=workspace,
        config=config,
    )

    if auto:
        save_auto_decisions(
            project_name=project_name,
            candidates=candidates,
            workspace=workspace,
            threshold=config.review.auto_score_threshold,
        )

    table = Table(title=f"ShortsForge Candidates - {project_name}")
    table.add_column("ID", style="cyan")
    table.add_column("Window")
    table.add_column("Score", justify="right")
    table.add_column("Quality")
    table.add_column("Top Hook (#1)")
    for c in candidates:
        q_str = "[green]PASS[/green]" if not c.rejected_by_quality else "[red]REJECT[/red]"
        hook_str = c.hooks.active_hook_text() if c.hooks else ""
        if c.hooks and c.hooks.flagged_for_review:
            hook_str = f"[red][FLAGGED][/red] {hook_str}"
        table.add_row(
            c.id,
            f"{c.start:.1f}s-{c.end:.1f}s ({c.duration:.0f}s)",
            f"{c.score}/100",
            q_str,
            hook_str,
        )
    console.print(table)
    console.print(f"\n[bold green]Review page generated:[/bold green] {review_path}")

    if auto:
        console.print(
            f"[yellow]--auto enabled:[/yellow] auto-approved clips >= {config.review.auto_score_threshold}. "
            f"Run [bold]python run.py render {project_name}[/bold] to export."
        )
    elif serve:
        port = config.review.server_port
        console.print(
            f"[cyan]Serving review UI at http://127.0.0.1:{port}/review.html "
            "(auto-saves decisions.json; press Ctrl+C when done)[/cyan]"
        )
        serve_review_page(workspace, port=port, open_browser=True)
    else:
        console.print(
            f"Open [bold]{review_path}[/bold] to approve/reject clips, save [bold]decisions.json[/bold] "
            f"into [bold]{workspace.project_dir}[/bold], then run [bold]python run.py render {project_name}[/bold]."
        )

    return review_path


def run_render_project(
    project_name: str,
    config: AppConfig,
    auto: bool = False,
    min_score: Optional[int] = None,
    force: bool = False,
    on_progress: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    max_render: Optional[int] = None,
    aspect_ratio: Optional[str] = None,
    playback_speed: Optional[float] = None,
) -> List[ClipExportMetadata]:
    """Execute Stages 5-9 for all approved clips in a project and write outputs/<name>/."""
    if min_score is not None:
        config.review.auto_score_threshold = min_score

    workspace = ProjectWorkspace(
        project_dir=config.projects_root() / project_name,
        output_dir=config.outputs_root() / project_name,
    )
    state = workspace.load_state()
    if state is None:
        raise FileNotFoundError(
            f"Project '{project_name}' not found in {workspace.project_dir}. "
            f"Run `python run.py new {project_name} --source ... --person ...` first."
        )

    if aspect_ratio in ("9:16", "16:9"):
        apply_aspect_ratio(config, aspect_ratio)
        state.aspect_ratio = config.reframe.aspect_ratio
        workspace.save_state(state)
    elif state.aspect_ratio == "16:9" and config.reframe.aspect_ratio == "9:16":
        apply_aspect_ratio(config, state.aspect_ratio)

    if playback_speed is not None:
        config.export.playback_speed = max(0.5, min(2.0, float(playback_speed)))
        state.playback_speed = float(config.export.playback_speed)
        workspace.save_state(state)

    metadata = VideoMetadata.model_validate_json(workspace.metadata_path.read_text(encoding="utf-8"))
    transcript = Transcript.model_validate_json(workspace.transcript_path.read_text(encoding="utf-8"))
    candidates = _load_candidates(workspace)

    if auto or not workspace.decisions_path.exists():
        decisions_doc = save_auto_decisions(
            project_name=project_name,
            candidates=candidates,
            workspace=workspace,
            threshold=config.review.auto_score_threshold,
        )
    else:
        decisions_doc = load_decisions(workspace, project_name)

    approved_pairs = []
    for c in candidates:
        dec = decisions_doc.decisions.get(c.id)
        if dec and dec.approved:
            hook_text = (dec.edited_hook_text or "").strip()
            if not hook_text and c.hooks:
                c.hooks.selected_index = dec.selected_hook_index
                hook_text = c.hooks.active_hook_text()
            approved_pairs.append((c, hook_text or c.key_quote[:60]))

    if not approved_pairs and candidates:
        # Ensure at least the top-scoring candidate renders when invoked in autopilot
        top_c = max(candidates, key=lambda x: x.score)
        approved_pairs.append((top_c, top_c.hooks.active_hook_text() if top_c.hooks else top_c.key_quote[:60]))

    max_limit = 50 if state.mode == "batch" else 15
    cap_n = max_limit if (max_render is None or max_render <= 0) else max(1, min(max_limit, max_render))
    approved_pairs = approved_pairs[:cap_n]

    if not approved_pairs:
        console.print("[yellow]No approved clips to render.[/yellow]")
        write_sources_and_warnings(workspace.output_dir, metadata, state.person, [], state.warnings, config)
        return []

    llm = LLMClient(config=config, workspace=workspace)
    exported: List[ClipExportMetadata] = []
    state_lock = threading.Lock()

    cfg_workers = int(getattr(config.export, "render_workers", 0) or 0)
    cpu_n = os.cpu_count() or 4
    auto_workers = min(3, max(2, cpu_n // 3)) if cpu_n >= 4 else 1
    max_workers = max(1, min(4, cfg_workers if cfg_workers > 0 else auto_workers))

    # Separate already-cached clips from clips that need rendering
    pending_items: List[Tuple[int, ClipCandidate, str]] = []
    target_ar = config.reframe.aspect_ratio
    target_spd = float(getattr(config.export, "playback_speed", 1.0) or 1.0)
    for idx_r, (cand, hook_text) in enumerate(approved_pairs, start=1):
        existing_meta = None
        if not force and cand.id in state.rendered_clips:
            matches = sorted(workspace.output_dir.glob(f"{cand.id}_*_metadata.json"))
            if matches:
                try:
                    parsed = ClipExportMetadata.model_validate_json(matches[0].read_text(encoding="utf-8"))
                    if (
                        (workspace.output_dir / parsed.output_video).exists()
                        and parsed.aspect_ratio == target_ar
                        and abs(float(parsed.playback_speed or 1.0) - target_spd) < 0.01
                    ):
                        existing_meta = parsed
                except Exception:  # noqa: BLE001
                    existing_meta = None
        if existing_meta is not None:
            exported.append(existing_meta)
            if on_progress:
                on_progress(f"Reused cached rendered Short for {cand.id} ({idx_r}/{len(approved_pairs)}).")
        else:
            pending_items.append((idx_r, cand, hook_text))

    if metadata.clip_only_mode and pending_items:
        missing_slice_cands = [
            cand
            for _, cand, _ in pending_items
            if not (
                cand.clip_video_path
                and Path(cand.clip_video_path).is_file()
                and Path(cand.clip_video_path).stat().st_size > 4096
            )
        ]
        if missing_slice_cands:
            metadata = download_clip_slices(
                candidates=missing_slice_cands,
                metadata=metadata,
                workspace=workspace,
                config=config,
                on_progress=on_progress,
                should_cancel=should_cancel,
            )
            _save_candidates(workspace, candidates)

    workers = min(max_workers, max(1, len(pending_items)))
    if pending_items and on_progress and workers > 1:
        on_progress(f"Turbo render: encoding {len(pending_items)} Short(s) across {workers} parallel workers (single-pass + GPU acceleration)...")

    def _render_worker(item: Tuple[int, ClipCandidate, str]) -> Optional[ClipExportMetadata]:
        idx_r, cand, hook_text = item
        if should_cancel and should_cancel():
            return None
        if on_progress:
            on_progress(
                f"Rendering Short {idx_r}/{len(approved_pairs)} ({cand.id}, {cand.duration:.0f}s): "
                "single-pass 9:16 crop -> neon purple captions -> -14 LUFS..."
            )
        local_det = FaceDetector()
        try:
            return render_single_short(
                candidate=cand,
                hook_text=hook_text,
                metadata=metadata,
                transcript=transcript,
                person=state.person,
                work_dir=workspace.work_dir,
                output_dir=workspace.output_dir,
                llm=llm,
                config=config,
                detector=local_det,
            )
        finally:
            local_det.close()

    with Progress(
        SpinnerColumn("line"),
        TextColumn("[bold magenta]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Rendering {len(approved_pairs)} vertical Short(s) ({workers}x parallel)...",
            total=len(approved_pairs),
            completed=len(exported),
        )
        if workers <= 1:
            for item in pending_items:
                if should_cancel and should_cancel():
                    if on_progress:
                        on_progress(f"Stopped by user cleanly after rendering {len(exported)} Short(s).")
                    break
                progress.update(task, description=f"Rendering {item[1].id} ({item[1].duration:.0f}s)...")
                meta_out = _render_worker(item)
                if meta_out is not None:
                    exported.append(meta_out)
                    if item[1].id not in state.rendered_clips:
                        state.rendered_clips.append(item[1].id)
                    workspace.save_state(state)
                    write_sources_and_warnings(workspace.output_dir, metadata, state.person, exported, state.warnings, config)
                progress.advance(task)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_map = {pool.submit(_render_worker, item): item for item in pending_items}
                for fut in as_completed(future_map):
                    item = future_map[fut]
                    meta_out = fut.result()
                    if meta_out is not None:
                        with state_lock:
                            exported.append(meta_out)
                            if item[1].id not in state.rendered_clips:
                                state.rendered_clips.append(item[1].id)
                            exported.sort(key=lambda m: m.clip_id)
                            workspace.save_state(state)
                            write_sources_and_warnings(workspace.output_dir, metadata, state.person, exported, state.warnings, config)
                            if on_progress:
                                on_progress(f"Finished {item[1].id} ({len(exported)}/{len(approved_pairs)} Shorts ready)")
                    progress.advance(task)

    exported.sort(key=lambda m: m.clip_id)

    sources_path, warnings_path = write_sources_and_warnings(
        output_dir=workspace.output_dir,
        metadata=metadata,
        person=state.person,
        exported_clips=exported,
        warnings=state.warnings,
        config=config,
    )

    console.print(f"\n[bold green]Rendered {len(exported)} Short(s) into:[/bold green] {workspace.output_dir}")
    console.print(f"  - Sources log:  {sources_path}")
    console.print(f"  - Warnings log: {warnings_path}")
    return exported


def run_edit_single_clip(
    project_name: str,
    clip_id: str,
    config: AppConfig,
    on_progress: Optional[Callable[[str], None]] = None,
    aspect_ratio: Optional[str] = None,
) -> ClipExportMetadata:
    """Fast post-render edit for a single video (filters, mirror, speed, captions, aspect ratio, white center card)."""
    workspace = ProjectWorkspace(
        project_dir=config.projects_root() / project_name,
        output_dir=config.outputs_root() / project_name,
    )
    state = workspace.load_state()
    if state is None:
        raise FileNotFoundError(f"Project '{project_name}' not found.")

    if aspect_ratio in ("9:16", "16:9"):
        apply_aspect_ratio(config, aspect_ratio)
    else:
        target_ar = state.aspect_ratio
        matches = sorted(workspace.output_dir.glob(f"{clip_id}_*_metadata.json"))
        if matches:
            try:
                old_m = ClipExportMetadata.model_validate_json(matches[0].read_text(encoding="utf-8"))
                if old_m.aspect_ratio in ("9:16", "16:9"):
                    target_ar = old_m.aspect_ratio
            except Exception:  # noqa: BLE001
                pass
        if target_ar in ("9:16", "16:9"):
            apply_aspect_ratio(config, target_ar)

    metadata = VideoMetadata.model_validate_json(workspace.metadata_path.read_text(encoding="utf-8"))
    transcript = Transcript.model_validate_json(workspace.transcript_path.read_text(encoding="utf-8"))
    candidates = _load_candidates(workspace)
    cand = next((c for c in candidates if c.id == clip_id), None)
    if cand is None:
        raise ValueError(f"Clip '{clip_id}' not found in project '{project_name}'.")

    decisions_doc = load_decisions(workspace, project_name)
    dec = decisions_doc.decisions.get(cand.id)
    hook_text = (dec.edited_hook_text or "").strip() if dec else ""
    if not hook_text and cand.hooks:
        if dec:
            cand.hooks.selected_index = dec.selected_hook_index
        hook_text = cand.hooks.active_hook_text()
    hook_text = hook_text or cand.key_quote[:60]

    if on_progress:
        on_progress(f"Applying runtime edits (filters, caption position & center attention card) to {clip_id}...")

    if metadata.clip_only_mode and not (
        cand.clip_video_path
        and Path(cand.clip_video_path).is_file()
        and Path(cand.clip_video_path).stat().st_size > 4096
    ):
        metadata = download_clip_slices(
            candidates=[cand],
            metadata=metadata,
            workspace=workspace,
            config=config,
            on_progress=on_progress,
        )
        _save_candidates(workspace, candidates)

    llm = LLMClient(config=config, workspace=workspace)
    detector = FaceDetector()
    try:
        updated_meta = render_single_short(
            candidate=cand,
            hook_text=hook_text,
            metadata=metadata,
            transcript=transcript,
            person=state.person,
            work_dir=workspace.work_dir,
            output_dir=workspace.output_dir,
            llm=llm,
            config=config,
            detector=detector,
        )
    finally:
        detector.close()

    if cand.id not in state.rendered_clips:
        state.rendered_clips.append(cand.id)
        workspace.save_state(state)

    return updated_meta


def run_autopilot_from_source(
    source: str,
    config: AppConfig,
    max_clips: int = 12,
    force: bool = False,
    on_progress: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    on_project_detected: Optional[Callable[[str], None]] = None,
    mode: str = "viral",
    part_duration_sec: float = 0.0,
    aspect_ratio: Optional[str] = None,
    playback_speed: Optional[float] = None,
) -> Tuple[str, List[ClipExportMetadata]]:
    """One-click autopilot: given ONLY a URL or file path, discovers viral clips OR splits sequential batch parts and renders."""
    run_mode = "batch" if str(mode or "viral").strip().lower() == "batch" else "viral"
    if run_mode == "batch":
        capped = 0 if (max_clips is None or max_clips <= 0) else max(1, min(50, int(max_clips)))
    else:
        capped = 12 if (max_clips is None or max_clips <= 0) else max(1, min(15, int(max_clips)))
    if on_progress:
        on_progress("Analyzing link to auto-detect speaker, show & project...")
    project_name, person, show_name, year = auto_detect_project_and_person(source, config)
    if on_project_detected:
        on_project_detected(project_name)
    if on_progress:
        on_progress(f"Detected Subject: {person} | Show: {show_name or 'Interview'} | Project: {project_name}")

    run_new_project(
        project_name=project_name,
        source=source,
        person=person,
        config=config,
        auto=True,
        serve=False,
        max_clips=capped,
        show_override=show_name,
        year_override=year,
        force=force,
        on_progress=on_progress,
        should_cancel=should_cancel,
        mode=run_mode,
        part_duration_sec=part_duration_sec,
        aspect_ratio=aspect_ratio,
        playback_speed=playback_speed,
    )
    exported = run_render_project(
        project_name=project_name,
        config=config,
        auto=True,
        force=force,
        on_progress=on_progress,
        should_cancel=should_cancel,
        max_render=capped,
        aspect_ratio=aspect_ratio,
        playback_speed=playback_speed,
    )
    return project_name, exported
