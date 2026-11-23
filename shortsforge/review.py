"""Local review.html generator, decisions.json loader, and optional 127.0.0.1 auto-save review server."""

from __future__ import annotations

import html
import json
import os
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List

from shortsforge.config import AppConfig
from shortsforge.models import (
    ClipCandidate,
    ReviewDecision,
    ReviewDecisionsFile,
    VideoMetadata,
)
from shortsforge.state import ProjectWorkspace


def generate_review_html(
    project_name: str,
    person: str,
    metadata: VideoMetadata,
    candidates: List[ClipCandidate],
    workspace: ProjectWorkspace,
    config: AppConfig,
) -> Path:
    """Generate a self-contained offline projects/<name>/review.html page."""
    existing_decisions = load_decisions(workspace, project_name)

    payload_items = []
    for c in candidates:
        dec = existing_decisions.decisions.get(c.id)
        default_approved = (
            dec.approved
            if dec is not None
            else (not c.rejected_by_quality and c.score >= config.review.auto_score_threshold)
        )
        selected_idx = dec.selected_hook_index if dec is not None else (c.hooks.selected_index if c.hooks else 0)
        edited_text = dec.edited_hook_text if dec is not None else (c.hooks.edited_text if c.hooks else None)

        thumb_rel = ""
        if c.thumbnail_path:
            try:
                thumb_rel = os.path.relpath(Path(c.thumbnail_path), workspace.project_dir).replace("\\", "/")
            except ValueError:
                thumb_rel = Path(c.thumbnail_path).name

        variants = []
        if c.hooks:
            for v in c.hooks.variants:
                variants.append({
                    "rank": v.rank,
                    "text": v.text,
                    "predicted_retention": v.predicted_retention,
                    "passed": v.verification.passed,
                    "reason": v.verification.reason,
                })

        payload_items.append({
            "id": c.id,
            "start": c.start,
            "end": c.end,
            "duration": round(c.duration, 1),
            "score": c.score,
            "breakdown": c.score_breakdown.model_dump(),
            "reason": c.reason,
            "key_quote": c.key_quote,
            "transcript_text": c.transcript_text,
            "thumbnail": thumb_rel,
            "rejected_by_quality": c.rejected_by_quality,
            "quality_notes": (c.quality.rejection_reasons + c.quality.warnings) if c.quality else [],
            "flagged_hook": bool(c.hooks and c.hooks.flagged_for_review),
            "flag_reason": c.hooks.flag_reason if c.hooks else "",
            "variants": variants,
            "approved": default_approved,
            "selected_hook_index": selected_idx,
            "edited_hook_text": edited_text or (variants[selected_idx]["text"] if variants else ""),
        })

    data_json = json.dumps(
        {
            "project_name": project_name,
            "person": person,
            "title": metadata.title,
            "show_name": metadata.show_name,
            "year": metadata.year,
            "auto_threshold": config.review.auto_score_threshold,
            "max_hook_words": config.hook.max_words,
            "candidates": payload_items,
        },
        ensure_ascii=False,
    )

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Shortly Review - {html.escape(project_name)}</title>
<style>
  :root {{
    --bg: #121c2b; --panel: rgba(20, 31, 49, 0.88); --panel-2: rgba(27, 41, 64, 0.82); --border: rgba(251, 243, 231, 0.14);
    --text: #f7f8fc; --muted: #93a6c4; --accent: #f26a3e; --green: #22c55e; --red: #f43f5e; --blue: #229bf0;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: radial-gradient(circle at 20% 0%, #123a6b 0%, #121c2b 58%, #0c1420 100%); color: var(--text); min-height: 100vh; }}
  header {{ position: sticky; top: 0; z-index: 20; background: rgba(14, 23, 36, 0.88); backdrop-filter: blur(18px); border-bottom: 1px solid var(--border); padding: 16px 28px; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 14px; }}
  .brand h1 {{ margin: 0; font-size: 20px; color: var(--blue); }}
  .brand .sub {{ font-size: 13px; color: var(--muted); margin-top: 2px; }}
  .toolbar {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
  .pill-stat {{ background: var(--panel-2); border: 1px solid var(--border); border-radius: 999px; padding: 6px 14px; font-size: 13px; font-weight: 600; }}
  button {{ cursor: pointer; border: none; border-radius: 999px; font-weight: 600; font-size: 13px; padding: 9px 16px; }}
  .btn-primary {{ background: linear-gradient(135deg, #229bf0, #f26a3e); color: #ffffff; }}
  .btn-secondary {{ background: var(--panel-2); color: var(--text); border: 1px solid var(--border); }}
  .btn-approve {{ background: rgba(34, 197, 94, 0.14); color: #86efac; border: 1px solid rgba(34, 197, 94, 0.45); }}
  .btn-approve.active {{ background: var(--green); color: #052e16; }}
  .btn-reject {{ background: rgba(244, 63, 94, 0.14); color: #fda4af; border: 1px solid rgba(244, 63, 94, 0.45); }}
  .btn-reject.active {{ background: var(--red); color: #ffffff; }}
  main {{ max-width: 1280px; margin: 24px auto 64px; padding: 0 24px; display: grid; grid-template-columns: repeat(auto-fill, minmax(560px, 1fr)); gap: 20px; }}
  .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 16px; overflow: hidden; display: flex; flex-direction: column; }}
  .card.approved {{ border-color: rgba(34, 197, 94, 0.7); }}
  .card.rejected {{ opacity: 0.72; }}
  .card-top {{ display: grid; grid-template-columns: 200px 1fr; gap: 16px; padding: 16px; }}
  .thumb-wrap {{ position: relative; border-radius: 10px; overflow: hidden; background: #07090e; aspect-ratio: 16 / 10; }}
  .thumb-wrap img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
  .score-badge {{ position: absolute; top: 8px; left: 8px; background: rgba(12, 20, 34, 0.9); color: #ffd3c4; border: 1px solid rgba(242, 106, 62, 0.58); border-radius: 8px; padding: 4px 9px; font-weight: 800; font-size: 14px; }}
  .time-badge {{ position: absolute; bottom: 8px; right: 8px; background: rgba(0, 0, 0, 0.78); color: #fff; border-radius: 6px; padding: 2px 7px; font-size: 11px; font-family: ui-monospace, monospace; }}
  .clip-id {{ font-weight: 700; font-size: 15px; color: var(--blue); }}
  .reason {{ font-size: 13px; color: var(--muted); margin: 6px 0 8px; }}
  .quote {{ font-size: 13px; font-style: italic; background: var(--panel-2); border-left: 3px solid var(--accent); padding: 8px 10px; border-radius: 6px; color: #f8fafc; }}
  .scores-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
  .mini-pill {{ font-size: 11px; color: var(--muted); background: #10141e; border: 1px solid var(--border); padding: 2px 7px; border-radius: 6px; }}
  .warn-box {{ margin: 0 16px 10px; padding: 9px 12px; border-radius: 8px; background: rgba(244, 63, 94, 0.14); border: 1px solid rgba(244, 63, 94, 0.45); color: #fecaca; font-size: 12px; }}
  .hook-section {{ padding: 12px 16px 16px; border-top: 1px solid var(--border); background: rgba(11, 18, 31, 0.75); }}
  .hook-label-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 12px; color: var(--muted); }}
  .word-count.over {{ color: var(--red); font-weight: 700; }}
  .hook-input {{ width: 100%; padding: 10px 12px; border-radius: 8px; border: 1px solid var(--border); background: #0a0d14; color: #fff; font-size: 15px; font-weight: 700; }}
  .variants-select {{ width: 100%; margin-top: 8px; padding: 7px 10px; border-radius: 7px; border: 1px solid var(--border); background: var(--panel-2); color: var(--text); font-size: 12px; }}
  .card-actions {{ display: flex; gap: 10px; padding: 12px 16px; border-top: 1px solid var(--border); background: rgba(10, 16, 28, 0.8); }}
  .card-actions button {{ flex: 1; }}
</style>
</head>
<body>
<header>
  <div class="brand">
    <h1>Shortly Review - <span id="proj-name"></span></h1>
    <div class="sub" id="proj-sub"></div>
  </div>
  <div class="toolbar">
    <span class="pill-stat" id="count-stat">0 / 0 Approved</span>
    <button class="btn-secondary" id="btn-approve-thresh">Approve &ge; Threshold</button>
    <button class="btn-primary" id="btn-save">Save decisions.json</button>
    <span id="save-status"></span>
  </div>
</header>
<main id="cards"></main>
<script>
const DATA = {data_json};
const IS_HTTP = window.location.protocol.startsWith("http");
document.getElementById("proj-name").textContent = DATA.project_name;
document.getElementById("proj-sub").textContent = `${{DATA.person}} • ${{DATA.show_name}}${{DATA.year ? " (" + DATA.year + ")" : ""}} - ${{DATA.title}}`;
const state = {{}};
for (const c of DATA.candidates) {{
  state[c.id] = {{ clip_id: c.id, approved: !!c.approved, selected_hook_index: c.selected_hook_index || 0, edited_hook_text: c.edited_hook_text || "" }};
}}
function countWords(txt) {{ const s = (txt || "").trim(); return s ? s.split(/\\s+/).length : 0; }}
function fmtTime(sec) {{ const m = Math.floor(sec / 60); const s = Math.floor(sec % 60).toString().padStart(2, "0"); return `${{m}}:${{s}}`; }}
function updateCount() {{ const total = DATA.candidates.length; const approved = Object.values(state).filter(d => d.approved).length; document.getElementById("count-stat").textContent = `${{approved}} / ${{total}} Approved`; }}
function buildDecisionsPayload() {{ return {{ project_name: DATA.project_name, decisions: state }}; }}
async function autoSaveIfServer() {{
  if (!IS_HTTP) return;
  try {{ await fetch("/api/decisions", {{ method: "POST", headers: {{ "Content-Type": "application/json" }}, body: JSON.stringify(buildDecisionsPayload(), null, 2) }}); }} catch (_) {{}}
}}
async function saveDecisions() {{
  const jsonStr = JSON.stringify(buildDecisionsPayload(), null, 2);
  if (IS_HTTP) {{ await autoSaveIfServer(); return; }}
  const blob = new Blob([jsonStr], {{ type: "application/json" }});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "decisions.json"; a.click();
}}
function render() {{
  const container = document.getElementById("cards"); container.innerHTML = "";
  for (const c of DATA.candidates) {{
    const st = state[c.id];
    const card = document.createElement("article");
    card.className = "card " + (st.approved ? "approved" : "rejected");
    const wc = countWords(st.edited_hook_text);
    const over = wc > DATA.max_hook_words;
    const variantOpts = (c.variants || []).map((v, idx) => `<option value="${{idx}}" ${{idx === st.selected_hook_index ? "selected" : ""}}>#${{v.rank}} [${{v.passed ? "✓" : "✗"}} ${{v.predicted_retention}}%] ${{v.text}}</option>`).join("");
    card.innerHTML = `
      <div class="card-top">
        <div class="thumb-wrap">${{c.thumbnail ? `<img src="${{c.thumbnail}}" alt="${{c.id}} preview" />` : ""}}<div class="score-badge">${{c.score}}/100</div><div class="time-badge">${{fmtTime(c.start)}}-${{fmtTime(c.end)}} (${{c.duration}}s)</div></div>
        <div><span class="clip-id">${{c.id.toUpperCase()}}</span><div class="reason">${{c.reason}}</div><div class="quote">"${{c.key_quote}}"</div>
          <div class="scores-row"><span class="mini-pill">Emotion ${{c.breakdown.emotional_peak}}</span><span class="mini-pill">Quote ${{c.breakdown.quotable_statement}}</span><span class="mini-pill">Humor ${{c.breakdown.humor_payoff}}</span><span class="mini-pill">Standalone ${{c.breakdown.self_contained}}</span><span class="mini-pill">Bounds ${{c.breakdown.clean_boundaries}}</span></div>
        </div>
      </div>
      <div class="hook-section">
        <div class="hook-label-row"><span>3-Second Hook (editable)</span><span class="word-count ${{over ? "over" : ""}}" id="wc-${{c.id}}">${{wc}} / ${{DATA.max_hook_words}} words</span></div>
        <input type="text" class="hook-input" id="hook-${{c.id}}" value="${{st.edited_hook_text.replace(/"/g, '&quot;')}}" />
        <select class="variants-select" id="var-${{c.id}}">${{variantOpts}}</select>
      </div>
      <div class="card-actions">
        <button class="btn-approve ${{st.approved ? "active" : ""}}" id="app-${{c.id}}">Approve</button>
        <button class="btn-reject ${{!st.approved ? "active" : ""}}" id="rej-${{c.id}}">Reject</button>
      </div>`;
    container.appendChild(card);
    document.getElementById(`app-${{c.id}}`).addEventListener("click", () => {{ st.approved = true; render(); autoSaveIfServer(); }});
    document.getElementById(`rej-${{c.id}}`).addEventListener("click", () => {{ st.approved = false; render(); autoSaveIfServer(); }});
    document.getElementById(`var-${{c.id}}`).addEventListener("change", (ev) => {{
      const idx = parseInt(ev.target.value, 10) || 0; st.selected_hook_index = idx;
      if (c.variants[idx]) {{ st.edited_hook_text = c.variants[idx].text; render(); }}
      autoSaveIfServer();
    }});
    document.getElementById(`hook-${{c.id}}`).addEventListener("input", (ev) => {{
      st.edited_hook_text = ev.target.value;
      const nwc = countWords(st.edited_hook_text);
      const wEl = document.getElementById(`wc-${{c.id}}`);
      wEl.textContent = `${{nwc}} / ${{DATA.max_hook_words}} words`;
      wEl.className = "word-count" + (nwc > DATA.max_hook_words ? " over" : "");
      autoSaveIfServer();
    }});
  }}
  updateCount();
}}
document.getElementById("btn-save").addEventListener("click", saveDecisions);
document.getElementById("btn-approve-thresh").addEventListener("click", () => {{
  for (const c of DATA.candidates) state[c.id].approved = !c.rejected_by_quality && c.score >= DATA.auto_threshold;
  render(); autoSaveIfServer();
}});
render();
</script>
</body>
</html>"""
    workspace.review_html_path.write_text(page_html, encoding="utf-8")
    return workspace.review_html_path


def load_decisions(workspace: ProjectWorkspace, project_name: str) -> ReviewDecisionsFile:
    """Load decisions.json from the project folder if it exists."""
    if workspace.decisions_path.exists():
        try:
            return ReviewDecisionsFile.model_validate_json(
                workspace.decisions_path.read_text(encoding="utf-8")
            )
        except Exception:  # noqa: BLE001
            pass
    return ReviewDecisionsFile(project_name=project_name, decisions={})


def save_auto_decisions(
    project_name: str,
    candidates: List[ClipCandidate],
    workspace: ProjectWorkspace,
    threshold: int,
    max_approved: int = 12,
) -> ReviewDecisionsFile:
    """Auto-approve non-rejected candidates meeting threshold, keeping clip counts in a balanced 6-12 sweet spot."""
    decisions: Dict[str, ReviewDecision] = {}
    cap = 12 if max_approved <= 0 else max(1, min(15, max_approved))

    # Rank quality-passed candidates by score descending to enforce the upper ceiling
    passed_sorted = sorted(
        [c for c in candidates if not c.rejected_by_quality],
        key=lambda c: c.score,
        reverse=True,
    )
    approved_ids = {c.id for c in passed_sorted if c.score >= threshold}
    if len(approved_ids) > cap:
        approved_ids = {c.id for c in passed_sorted[:cap]}

    # Adaptive sweet-spot floor: if a high threshold on a multi-clip podcast approved <=3 clips,
    # include the top quality-passed viral picks (score >= 72) up to a balanced 6-8 clip batch.
    min_floor = min(8, max(5, cap // 2))
    if len(approved_ids) < min_floor and len(passed_sorted) >= 4:
        for c in passed_sorted:
            if len(approved_ids) >= min_floor:
                break
            if c.score >= 72:
                approved_ids.add(c.id)

    for c in candidates:
        decisions[c.id] = ReviewDecision(
            clip_id=c.id,
            approved=(c.id in approved_ids),
            selected_hook_index=c.hooks.selected_index if c.hooks else 0,
            edited_hook_text=c.hooks.active_hook_text() if c.hooks else None,
        )
    if not approved_ids and candidates:
        top = max(candidates, key=lambda x: x.score)
        decisions[top.id].approved = True

    doc = ReviewDecisionsFile(project_name=project_name, decisions=decisions)
    workspace.decisions_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    return doc


def serve_review_page(workspace: ProjectWorkspace, port: int = 8765, open_browser: bool = True) -> None:
    """Serve projects/<name>/review.html locally on 127.0.0.1 with auto-saving POST /api/decisions."""
    proj_dir = workspace.project_dir

    class ReviewHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(proj_dir), **kwargs)

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/decisions":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            try:
                parsed = ReviewDecisionsFile.model_validate_json(raw)
                workspace.decisions_path.write_text(parsed.model_dump_json(indent=2), encoding="utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            except Exception as exc:  # noqa: BLE001
                self.send_response(400)
                self.end_headers()
                self.wfile.write(str(exc).encode("utf-8"))

    server = ThreadingHTTPServer(("127.0.0.1", port), ReviewHandler)
    url = f"http://127.0.0.1:{port}/review.html"
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
