# DemoDay Backend Service (FastAPI)

Autonomous backend engine that powers the DemoDay Launch Studio:
- **Repo ingest**: GitHub metadata, README, dependencies (package.json / requirements.txt / pyproject.toml) and page routes (Next.js, SvelteKit).
- **Live app probe**: renders the deployed app in headless Chromium and inventories its visible links, buttons and inputs, so the planner only uses selectors that exist.
- **Planner**: Gemini (`gemini-3.6-flash` by default) writes the scene-by-scene storyboard and the launch kit (README, Devpost pitch, X thread, LinkedIn post). Template fallbacks keep the pipeline working without a key.
- **Narration**: fal.ai Kokoro TTS (`fal-ai/kokoro/american-english`), loudness-normalized, with exact durations measured by ffprobe. Falls back to local TTS (macOS `say` / `espeak-ng`).
- **Browser automation**: one continuous 1920x1080 Playwright session; each scene's footage is timed to its narration, with a visible cursor and graceful skips for missing selectors.
- **Video assembly**: FFmpeg cuts each scene, muxes H.264/AAC clips and concatenates them into a web-ready `demo.mp4` (fast start).
- **API**: FastAPI with Server-Sent Events for live progress.

---

## Quickstart (Local)

### 1. Requirements
- Python 3.11+ (macOS: `brew install python@3.11`; the system `python3` is often older)
- FFmpeg with ffprobe (`brew install ffmpeg` or `apt-get install ffmpeg`)

### 2. Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
```

### 3. Environment Variables
```bash
cp .env.example .env
```
| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Google AI Studio key. Without it, storyboards and launch kits use templates. |
| `FAL_KEY` | fal.ai key for Kokoro narration. Without it, narration uses local TTS. |
| `GEMINI_MODEL` | Optional, default `gemini-3.6-flash` (`gemini-2.0-flash` was shut down on 2026-06-01). |
| `GITHUB_TOKEN` | Optional; raises the GitHub API limit above 60 requests/hour. |
| `STORAGE_DIR` | Where job media is written. Relative paths resolve against this folder (default `storage`). |
| `CORS_ORIGINS` | Comma-separated dashboard origins, default `http://localhost:3000,http://127.0.0.1:3000`. |

### 4. Run Server
```bash
uvicorn app.main:app --port 8000
```
API documentation is served at `http://localhost:8000/docs`.

| Endpoint | Description |
| --- | --- |
| `GET /api/health` | Status plus capabilities (`ffmpeg`, `gemini`, `fal`, `gemini_model`) |
| `POST /api/pipeline/start` | Start a job: `{repo_url, app_url, voice_style, target_seconds}` |
| `GET /api/pipeline/events/{job_id}` | SSE stream of full job snapshots (`/stream/{job_id}` is an alias) |
| `GET /api/pipeline/status/{job_id}` | Current job snapshot |
| `GET /api/pipeline/video/{job_id}` | The MP4 (supports range requests; `?download=1` for an attachment) |

Each job writes its media and a `job.json` (storyboard, launch kit, logs) to `storage/<job_id>/`.

### 5. End-to-end test
```bash
python test_e2e.py                     # FastAPI repo + fastapi.tiangolo.com, 30 s
python test_e2e.py --repo <github-url> --url <app-url> --seconds 45 --voice calm_expert
```
Runs a full job in-process and checks the output: H.264/yuv420p 1920x1080, AAC 48 kHz, audio/video durations within 100 ms, fast start, and a populated launch kit. Exits non-zero on any failure.

---

## Docker Deployment

Build and run in a self-contained container with all OS dependencies and Chromium pre-installed. Secrets are passed at runtime and are excluded from the image by `.dockerignore`:

```bash
docker build -t demoday-backend .
docker run -p 8000:8000 --env-file .env demoday-backend
```
