# ShortsForge

Turn long interviews, shows, and podcasts into viral-style `1080×1920` (9:16) vertical Shorts locally—with truthful, transcript-verified 3-second hooks, smooth face-tracked reframing, animated word-by-word captions, broadcast lower-thirds, `-14 LUFS` mastered audio, and a local HTML review page.

---

## Quick Workflow

### Option A — Local Studio GUI (Recommended)

```bash
python gui.py
# or:
python run.py gui
```

Opens **ShortsForge Studio** in your browser (`http://127.0.0.1:8780`) with:
- **New Project** wizard (URL or native **Browse File...** picker, hook style, caption preset, deceased-subject guardrail, archive tag)
- **Live Review & Hooks** editor with instant auto-save and one-click **Render Approved Shorts**
- **Rendered Shorts Player** with built-in 9:16 video preview, measured LUFS, one-click Copy buttons for all 5 titles / credited description / 8 hashtags / pinned comment, and `sources.txt` / `warnings.txt`
- **Settings & API Keys** panel for `config.yaml` and `.env`

### Option B — Command Line (CLI)

```bash
# 1. Ingest, transcribe, discover best clips, generate & verify hooks, and build review.html
python run.py new my_episode --source "https://www.youtube.com/watch?v=..." --person "Full Name"

#    (Or open review.html with live auto-saving on http://127.0.0.1:8765/review.html)
python run.py new my_episode --source "path/to/interview.mp4" --person "Full Name" --serve

# 2. Review suggested clips on projects/my_episode/review.html
#    Approve/reject clips, switch between 5 hook variants, or edit the hook text.
#    (Or skip manual review with --auto --min-score 75)

# 3. Render approved clips into finished 9:16 Shorts in outputs/my_episode/
python run.py render my_episode
```

---

## Setup & Installation

### 1. Prerequisites
- **Python 3.11+** (Python 3.12 recommended for smooth prebuilt wheels across `faster-whisper`, `mediapipe`, and `opencv-python`)
- **FFmpeg** (with `ffprobe` and `libass` subtitle filter support)
- **yt-dlp** (installed automatically with `requirements.txt`, or available globally)

#### Installing FFmpeg
- **Windows (WinGet)**:
  ```powershell
  winget install -e --id Gyan.FFmpeg
  ```
- **macOS (Homebrew)**:
  ```bash
  brew install ffmpeg yt-dlp
  ```
- **Ubuntu / Debian Linux**:
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```

### 2. Python Virtual Environment & Dependencies

```bash
# Using standard venv (or: uv venv --python 3.12 .venv)
python -m venv .venv

# Activate:
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure API Keys (`.env`) & Settings (`config.yaml`)

```bash
cp .env.example .env
```

Edit `.env` and add the API key matching `llm.provider` in `config.yaml`:
- `GEMINI_API_KEY=...` (default: `llm.provider: "gemini"`)
- `ANTHROPIC_API_KEY=...` (if `llm.provider: "claude"`)

Optional settings in `config.yaml`:
- `hook.style`: `"cold_open"` (intense 2–3s peak + whoosh/rewind transition) or `"freeze_narrator"` (freeze-frame slow zoom + generic TTS narrator via `edge-tts` / `openai`)
- `hook.deceased_subject`: `true` enables additional respectful-tone guardrails (also available via `--deceased`)
- `captions.preset`: `"neon"` | `"comic"` | `"minimal"`
- `reframe.two_person_layout`: `"split"` (stacked top/bottom) | `"blurred_fit"`
- `reframe.enhance_old_footage`: `true` enables light ffmpeg-only denoise/upscale (`hqdn3d` + `lanczos` + subtle `unsharp`, no AI face alteration)
- `overlays.archive_tag`: `true` adds a `"FROM THE ARCHIVE"` badge alongside the persistent source credit

---

## Verify Your Install (Smoke Test & Unit Tests)

Run the automated unit test suite (sentence-boundary snapping, caption timing/presets, 9-word hook limits, 2nd-pass transcript verification, and caching):

```bash
pytest
```

Run the self-contained end-to-end smoke test (generates a ~28s sample interview video locally and runs the full `new` → `review.html` → `render` pipeline using offline mock LLM mode):

```bash
python scripts/smoke_test.py
```

Expected outputs after the smoke test:
- `projects/smoke_demo/review.html`
- `outputs/smoke_demo/clip_01_*.mp4` (`1080×1920`, H.264, `-14 LUFS`)
- `outputs/smoke_demo/clip_01_*_thumb.jpg`
- `outputs/smoke_demo/clip_01_*_metadata.json` & `.txt` (5 titles, credited description, 8 hashtags, pinned comment)
- `outputs/smoke_demo/sources.txt` & `outputs/smoke_demo/warnings.txt`

---

## Worked Example (Batch Mode: 1 Episode → up to 10 Shorts)

```bash
# 1. Analyze a 1-hour interview and open the local review page with live auto-save
python run.py new sagan_1980 \
  --source "https://www.youtube.com/watch?v=EXAMPLE" \
  --person "Carl Sagan" \
  --show "The Tonight Show" \
  --year "1980" \
  --deceased \
  --max-clips 10 \
  --serve

# 2. In your browser, approve your favorite clips and tweak any hook text (<= 9 words).
#    Press Ctrl+C in the terminal once saved.

# 3. Batch-render all approved clips into outputs/sagan_1980/
python run.py render sagan_1980
```

To run completely hands-free without opening the review page:

```bash
python run.py new sagan_1980 --source "interview.mp4" --person "Carl Sagan" --auto --min-score 78
python run.py render sagan_1980 --auto --min-score 78
```

---

## Troubleshooting

- **`'ffmpeg' was not found on PATH`**: Install FFmpeg (see above) and restart your terminal so the updated `PATH` loads. On Windows WinGet installs, ShortsForge also checks `%LOCALAPPDATA%\Microsoft\WinGet\Links` automatically.
- **Private / Removed / Age-restricted YouTube videos**: ShortsForge detects and reports these cleanly during Stage 1. Download the file locally or verify the URL is public before re-running.
- **Slow transcription on CPU**: Lower `transcription.model_size` in `config.yaml` from `"small"` to `"base"` or `"tiny"`, or set `transcription.device: "cuda"` and `compute_type: "float16"` if you have an NVIDIA GPU with CUDA.
- **Hook flagged after 3 verification attempts**: Check `outputs/<name>/warnings.txt` or the red warning box in `projects/<name>/review.html`, edit the hook manually so it strictly reflects the transcript, save `decisions.json`, and re-run `python run.py render <name>`.
- **Resuming interrupted runs**: Every stage (`ingest`, `transcribe`, `moments`, `hooks`, LLM calls, and rendered clips) is cached under `projects/<name>/`. Simply re-run the same command to resume where it left off, or pass `--force` to recompute from scratch.
