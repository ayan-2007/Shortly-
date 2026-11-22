"""End-to-end ShortsForge pipeline orchestrator (`new`, `render`, and 1-click link `autopilot`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from shortsforge.config import AppConfig
from shortsforge.export import render_single_short, write_sources_and_warnings
from shortsforge.hooks import generate_verified_hooks
from shortsforge.ingest import auto_detect_person_from_title, auto_detect_project_and_person, ingest_source
from shortsforge.llm import LLMClient
from shortsforge.models import (
    ClipCandidate,
    ClipExportMetadata,
    ProjectState,
    Transcript,
    VideoMetadata,
)
from shortsforge.moments import find_best_moments
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
) -> Path:
    """Execute Stages 1-4 (Ingest -> Transcribe -> Moments + Quality Gate -> Verified Hooks) & build review.html."""
    if max_clips is not None:
        config.moments.max_candidates = max(1, max_clips)
    if min_score is not None:
        config.review.auto_score_threshold = min_score
    deceased_flag = deceased or config.hook.deceased_subject

    workspace = ProjectWorkspace(
        project_dir=config.projects_root() / project_name,
        output_dir=config.outputs_root() / project_name,
    )

    state = workspace.load_state()
    if state is None or force:
        state = ProjectState(
            project_name=project_name,
            person=person,
            source=source,
            show_override=show_override,
            year_override=year_override,
            deceased=deceased_flag,
        )
        workspace.save_state(state)
    else:
        if person:
            state.person = person
        state.source = source
        state.deceased = deceased_flag
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
        metadata = ingest_source(
            source=source,
            workspace=workspace,
            config=config,
            show_override=state.show_override,
            year_override=state.year_override,
            force=force,
            on_progress=on_progress,
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

        candidates: List[ClipCandidate]
        if not force and "hooks" in state.completed_steps and workspace.candidates_path.exists():
            candidates = _load_candidates(workspace)
            t_mom = progress.add_task("3/4 Loaded cached candidate clips & quality checks", total=1)
            progress.advance(t_mom)
            t_hooks = progress.add_task("4/4 Loaded cached verified hooks", total=1)
            progress.advance(t_hooks)
        else:
            t_mom = progress.add_task("3/4 Finding best moments & checking video/audio quality...", total=2)
            if on_progress:
                on_progress(f"[3/4] Finding up to {config.moments.max_candidates} best viral moments...")
            candidates = find_best_moments(
                transcript=transcript,
                metadata=metadata,
                person=person,
                llm=llm,
                config=config,
            )
            progress.advance(t_mom)

            if on_progress:
                on_progress(f"[3/4] Inspecting video frames & audio on {len(candidates)} candidate clip(s)...")
            detector = FaceDetector()
            try:
                for cand in candidates:
                    thumb_path = workspace.thumbs_dir / f"{cand.id}_preview.jpg"
                    qreport = evaluate_candidate_quality(
                        candidate=cand,
                        video_path=Path(metadata.local_video_path),
                        audio_wav_path=Path(metadata.local_audio_path),
                        config=config,
                        detector=detector,
                        thumb_output_path=thumb_path,
                    )
                    cand.quality = qreport
                    cand.rejected_by_quality = not qreport.passed
                    if thumb_path.exists():
                        cand.thumbnail_path = str(thumb_path.resolve())
                    for w in qreport.warnings:
                        workspace.add_warning(state, w)
                    if not qreport.passed:
                        workspace.add_warning(
                            state,
                            f"{cand.id} rejected by quality gate: {'; '.join(qreport.rejection_reasons)}",
                        )
            finally:
                detector.close()

            _save_candidates(workspace, candidates)
            workspace.mark_step_complete(state, "moments")
            progress.advance(t_mom)

            t_hooks = progress.add_task(
                "4/4 Generating & verifying 3-second hooks (2nd-pass truth check)...",
                total=max(1, len(candidates)),
            )
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

    if not approved_pairs:
        console.print("[yellow]No approved clips to render.[/yellow]")
        write_sources_and_warnings(workspace.output_dir, metadata, state.person, [], state.warnings, config)
        return []

    llm = LLMClient(config=config, workspace=workspace)
    exported: List[ClipExportMetadata] = []
    detector = FaceDetector()

    try:
        with Progress(
            SpinnerColumn("line"),
            TextColumn("[bold magenta]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Rendering {len(approved_pairs)} vertical Short(s)...",
                total=len(approved_pairs),
            )
            for idx_r, (cand, hook_text) in enumerate(approved_pairs, start=1):
                existing_meta = None
                if not force and cand.id in state.rendered_clips:
                    matches = sorted(workspace.output_dir.glob(f"{cand.id}_*_metadata.json"))
                    if matches:
                        try:
                            parsed = ClipExportMetadata.model_validate_json(matches[0].read_text(encoding="utf-8"))
                            if (workspace.output_dir / parsed.output_video).exists():
                                existing_meta = parsed
                        except Exception:  # noqa: BLE001
                            existing_meta = None

                if existing_meta is not None:
                    progress.update(task, description=f"Skipping cached {cand.id} (already rendered)...")
                    if on_progress:
                        on_progress(f"Reused cached rendered Short for {cand.id} ({idx_r}/{len(approved_pairs)}).")
                    exported.append(existing_meta)
                    progress.advance(task)
                    continue

                progress.update(task, description=f"Rendering {cand.id} ({cand.duration:.0f}s)...")
                if on_progress:
                    on_progress(
                        f"Rendering Short {idx_r}/{len(approved_pairs)} ({cand.id}, {cand.duration:.0f}s): "
                        "9:16 face-tracking -> bold animated captions -> -14 LUFS..."
                    )
                meta_out = render_single_short(
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
                exported.append(meta_out)
                if cand.id not in state.rendered_clips:
                    state.rendered_clips.append(cand.id)
                workspace.save_state(state)
                progress.advance(task)
    finally:
        detector.close()

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


def run_autopilot_from_source(
    source: str,
    config: AppConfig,
    max_clips: int = 3,
    force: bool = False,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Tuple[str, List[ClipExportMetadata]]:
    """One-click autopilot: given ONLY a URL or file path, auto-detects project/person, finds clips, and renders Shorts."""
    if on_progress:
        on_progress("Analyzing link to auto-detect speaker, show & project...")
    project_name, person, show_name, year = auto_detect_project_and_person(source, config)
    if on_progress:
        on_progress(f"Detected Subject: {person} | Show: {show_name or 'Interview'} | Project: {project_name}")

    run_new_project(
        project_name=project_name,
        source=source,
        person=person,
        config=config,
        auto=True,
        serve=False,
        max_clips=max_clips,
        show_override=show_name,
        year_override=year,
        force=force,
        on_progress=on_progress,
    )
    exported = run_render_project(
        project_name=project_name,
        config=config,
        auto=True,
        force=force,
        on_progress=on_progress,
    )
    return project_name, exported
