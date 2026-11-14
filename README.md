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
| `DEMODAY_SECRET_KEY` | Encrypts users' saved GitHub tokens and app logins (32 random bytes, base64; `run.sh` generates one). Keep it stable: changing it makes saved credentials unreadable. |
| `SIGNUP_CODE` | Optional; when set, creating an account requires this code. |
| `COOKIE_SECURE` | Set `true` when the dashboard is served over HTTPS. |
| `ALLOW_PRIVATE_URLS` | Local development only: allow recording `localhost` or private-network apps. Never on a hosted server. |
| `GITHUB_TOKEN` | Optional server-wide fallback when a user hasn't saved their own token. |
| `STORAGE_DIR` | Job media, `job.json` files and the `demoday.db` database. Relative paths resolve against this folder (default `storage`). |
| `CORS_ORIGINS` | Comma-separated dashboard origins, default `http://localhost:3000,http://127.0.0.1:3000`. |

### 4. Run Server
```bash
uvicorn app.main:app --port 8000
```
API documentation is served at `http://localhost:8000/docs`.

Everything except `/api/health` and signup/login requires a signed-in user (HttpOnly session cookie), and jobs,
videos and saved credentials are only visible to their owner.

| Endpoint | Description |
| --- | --- |
| `GET /api/health` | Status plus capabilities (`ffmpeg`, `gemini`, `fal`, `gemini_model`, `vault`, `signup_code`) |
| `POST /api/auth/signup`, `/login`, `/logout`; `GET /api/auth/me` | Accounts and sessions |
| `POST /api/pipeline/start` | Start a job: `{repo_urls[], app_url, voice_style, target_seconds, use_app_login}` (`repo_url` still accepted) |
| `GET /api/pipeline/events/{job_id}` | SSE stream of full job snapshots (`/stream/{job_id}` is an alias) |
| `GET /api/pipeline/status/{job_id}` | Current job snapshot (finished jobs reload from `job.json` after a restart) |
| `GET /api/pipeline/video/{job_id}` | The MP4 (supports range requests; `?download=1` for an attachment) |
| `GET /api/settings`; `PUT`/`DELETE /api/settings/github-token` | GitHub token for private repositories (validated with GitHub, stored encrypted) |
| `GET /api/app-logins`; `DELETE /api/app-logins/{id}` | Saved sign-in sessions, metadata only |
| `POST /api/app-logins/capture`; `WS .../capture/{id}/ws`; `POST .../capture/{id}/finish` | Live sign-in: a server browser streamed to the dashboard; finishing saves only its session, encrypted |

Each job writes its media and a `job.json` (storyboard, launch kit, logs) to `storage/<job_id>/`.

### Signed-in recording and private repositories

- **Private repositories:** each user can save a GitHub token (a fine-grained token with read-only Contents and
  Metadata is enough). Repositories the token can't see are reported in the job log rather than silently skipped.
- **Pages behind a login:** in Settings, the user signs in to their app once through a live view of a server-side
  browser (SSO and two-factor flows work). Only the resulting cookies and local storage are saved, encrypted with
  AES-256-GCM and bound to that user; passwords are never stored. Jobs for the same app origin then record signed in.
- **Safety:** recordings never click sign-out or destructive controls (delete, remove, cancel, purchase, ...), and a
  sign-in form that appears despite a saved login is reported as a likely expired session. Footage and AI planning
  see whatever the account shows, so a demo account with sample data is recommended.

### Hosted deployment checklist

- Set `DEMODAY_SECRET_KEY` (keep it out of the image and stable across deploys), `COOKIE_SECURE=true`, and usually
  `SIGNUP_CODE`. Mount `STORAGE_DIR` on a persistent volume: it holds the user database and job media.
- Serve the dashboard and the API from the same site (for example `app.example.com` and `api.example.com`) and list
  the dashboard origin in `CORS_ORIGINS`; the session cookie is `SameSite=Lax`.
- Keep `ALLOW_PRIVATE_URLS` off. Every server-side browser request to private, loopback or link-local addresses
  (including cloud metadata at `169.254.169.254`) is refused. DNS rebinding is not fully covered, so don't run the
  service next to sensitive internal endpoints without network-level egress rules.

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
