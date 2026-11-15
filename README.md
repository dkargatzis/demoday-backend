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
| `FAL_KEY` | fal.ai key for Kokoro narration and the optional AI intro/outro. Without it, narration uses local TTS. |
| `FAL_VIDEO_MODEL` | Optional text-to-video model for the AI intro/outro, default `fal-ai/bytedance/seedance/v1/pro/fast/text-to-video` (1080p, about $0.25 per clip). Kling (`fal-ai/kling-video/v3/pro/text-to-video`) also works. |
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
API documentation (Swagger UI) is served at `http://localhost:8000/docs`, ReDoc at `/redoc`, and the OpenAPI spec at
`/openapi.json`; every operation lists its auth requirement (session cookie), response models and error codes.

Everything except `/api/health` and signup/login requires a signed-in user (HttpOnly session cookie), and jobs,
videos and saved credentials are only visible to their owner.

| Endpoint | Description |
| --- | --- |
| `GET /api/health` | Status plus capabilities (`ffmpeg`, `gemini`, `fal`, `gemini_model`, `vault`, `signup_code`) |
| `POST /api/auth/signup`, `/login`, `/logout`; `GET /api/auth/me` | Accounts and sessions |
| `POST /api/pipeline/start` | Start a job: `{repo_urls[], app_url, voice_style, target_seconds, use_app_login, ai_clips, use_staged_data}` (`repo_url` still accepted) |
| `GET /api/pipeline/events/{job_id}` | SSE stream of full job snapshots (`/stream/{job_id}` is an alias) |
| `GET /api/pipeline/status/{job_id}` | Current job snapshot (finished jobs reload from `job.json` after a restart) |
| `GET /api/pipeline/video/{job_id}` | The MP4 (supports range requests; `?download=1` for an attachment) |
| `GET /api/settings`; `PUT`/`DELETE /api/settings/github-token` | GitHub token for private repositories (validated with GitHub, stored encrypted) |
| `GET /api/app-logins`; `DELETE /api/app-logins/{id}` | Saved sign-in sessions, metadata only |
| `POST /api/app-logins/capture`; `WS .../capture/{id}/ws`; `POST .../capture/{id}/finish` | Live sign-in: a server browser streamed to the dashboard; finishing saves only its session, encrypted |

Each job writes its media and a `job.json` (storyboard, launch kit, logs) to `storage/<job_id>/`.

### AI intro and outro (optional)

With `ai_clips`, Gemini also describes two short cinematic shots (no text, screens or people), and fal.ai generates
them while the browser records, so they add little time. The project name, tagline and URL are typeset on top by the
browser (never drawn by the model), and the demo's voice reads one line over each. If fal.ai is unavailable or slow,
each bookend falls back to the typeset title on a slowly zooming gradient. The storyboard's `intro_sec` tells the
dashboard where scene 1 starts. Two clips cost about $0.50 with the default model.

### Polished walkthroughs

- **Clean frames:** before each scene the recorder accepts (or hides) cookie-consent banners, hides chat,
  scheduling and feedback widgets and promo pop-ups, and waits for skeletons and spinners to finish, so no
  frame shows a half-loaded page. Everything it does is listed in the job log.
- **Focus zoom and highlights:** every click or typed field gets a highlight ring, and the finished video glides
  toward it (about 1.5x) and back, so dashboard details stay readable at 1080p. Scrolls and navigations release
  the zoom.
- **Staged sample data:** in Settings, DemoDay loads the app (signed in, when a login is saved), follows its
  navigation links and records the JSON its pages fetch. Gemini rewrites those responses as well-populated
  sample data with exactly the same structure (reviewed and editable in Settings), and demos replay them instead
  of calling the real API. The video carries a "Sample data" label, and the form has a toggle per run. This works
  for pages that load their data from JSON APIs in the browser (not server-rendered HTML).

| Endpoint | Description |
| --- | --- |
| `GET /api/fixtures`, `GET /api/fixtures/{id}` | The user's staged data sets (one per app origin) |
| `POST /api/fixtures/capture` | Load `app_url` and record its JSON API responses into a set |
| `POST /api/fixtures/{id}/polish` | Rewrite the set with Gemini (`brief` steers it); schema-checked |
| `PUT /api/fixtures/{id}`, `DELETE /api/fixtures/{id}` | Edit (JSON-validated) or remove a set |

### Signed-in recording and private repositories

- **Private repositories:** each user can save a GitHub token (a fine-grained token with read-only Contents and
  Metadata is enough). Repositories the token can't see are reported in the job log rather than silently skipped.
- **Pages behind a login:** in Settings, the user signs in to their app once through a live view of a server-side
  browser (SSO and two-factor flows work). Only the resulting cookies, local storage and IndexedDB (where Firebase-style sign-ins keep the user) are saved, encrypted with
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
python test_e2e.py --ai-clips             # with the AI intro/outro (uses fal.ai video credits)
```
Runs a full job in-process and checks the output: H.264/yuv420p 1920x1080, AAC 48 kHz, audio/video durations within 100 ms, fast start, and a populated launch kit. Exits non-zero on any failure.

---

## Google Cloud Run (live deployment)

`../deploy.sh` deploys DemoDay as **one Cloud Run service**: this API plus the dashboard's static build, served from
the same origin (so the session cookie, the event stream and the sign-in WebSocket never cross sites; `*.run.app`
is a public suffix, so two separate services would). It creates, idempotently:

- a service account with access only to the secrets and the bucket;
- Secret Manager secrets `demoday-gemini-api-key`, `demoday-fal-key`, `demoday-secret-key` (generated once),
  `demoday-signup-code` (generated; required to create accounts) and optionally `demoday-github-token`,
  seeded from `demoday-backend/.env`;
- a Cloud Storage bucket mounted at `/mnt/storage` for videos and job files, with the SQLite database kept on
  local disk and snapshotted to the bucket after every write (restored on start);
- the Cloud Run service (gen2, 4 vCPU / 8 GiB, one always-on instance, no CPU throttling, 60-minute requests),
  built by Cloud Build from this Dockerfile, with Google Cloud Text-to-Speech (Chirp 3 HD) enabled as the
  narration fallback.

```bash
./deploy.sh                      # setup + build + deploy; prints the URL and how to read the signup code
./deploy.sh --enable-share-links # optional: a public bucket so "Copy Link" gives links anyone can open
```

One instance keeps jobs, the recording slot and sign-in captures in memory; a deploy restarts it, so finish
running demos first. Finished demos reload from the bucket.

## Docker Deployment

Build and run in a self-contained container with all OS dependencies and Chromium pre-installed. Secrets are passed at runtime and are excluded from the image by `.dockerignore`:

```bash
docker build -t demoday-backend .
docker run -p 8000:8000 --env-file .env demoday-backend
```
