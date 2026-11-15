import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, List

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.config import BACKEND_ROOT, GEMINI_MODEL, GITHUB_API_URL, capabilities, storage_root  # loads .env first
from app import auth, db, vault
from app.models.schemas import (AppLogin, CaptureStart, ErrorDetail, ErrorMessage, FixtureSet, FixtureSetSummary, GitHubConnection,
                                JobStarted, JobSummary,
                                HealthStatus, JobStatus, Ok, PipelineRequest, SettingsOut, ShareLink, _http_url)
from app.services import cloud, fixtures
from app.services.capture import CAPTURES, indexed_db_records, origin_of, unpack_login
from app.services.capture import VIEWPORT as CAPTURE_VIEWPORT
from app.services.netguard import BlockedURL, ensure_public_url
from app.services.orchestrator import JOBS, create_pipeline_job, stream_job_events

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per outbound request is noise
logger = logging.getLogger("demoday")

JOB_ID_RE = re.compile(r"^[a-f0-9]{8,32}$")
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
           if o.strip()]


@asynccontextmanager
async def lifespan(_: FastAPI):
    caps = capabilities()
    if not caps["ffmpeg"]:
        logger.warning("ffmpeg/ffprobe not found on PATH: rendering will fail (brew install ffmpeg)")
    if not caps["gemini"]:
        logger.warning("GEMINI_API_KEY not set: storyboard and launch kit will use templates")
    else:
        logger.info("Gemini model: %s", GEMINI_MODEL)
    if not caps["fal"]:
        logger.warning("FAL_KEY not set: narration will use local TTS (macOS say / espeak-ng)")
    if not caps["vault"]:
        logger.warning("DEMODAY_SECRET_KEY not set: users can't save GitHub tokens or app logins")
    yield
    await CAPTURES.shutdown()
    db.flush_backup()  # the last writes reach the bucket before the instance goes away


API_DESCRIPTION = """
Autonomous demo videos and launch kits: give it a GitHub repository (or several) and a live app URL, and it plans a
storyboard with Gemini, narrates it (fal.ai Kokoro, Google Cloud Text-to-Speech or a local voice), records the real
app in headless Chromium, assembles a 1080p MP4 with FFmpeg, and writes the README, Devpost pitch and launch posts.

**Authentication.** Every endpoint except `/api/health`, `/api/auth/signup` and `/api/auth/login` needs the
`demoday_session` cookie (HttpOnly, `SameSite=Lax`) that signup and login set. Creating an account requires the
server's signup code when one is configured. Jobs, videos, saved logins and sample data are only visible to their
owner.

**Live updates.** `GET /api/pipeline/events/{job_id}` is a Server-Sent Events stream of typed messages, each with an
`id`, an `event` name and a JSON `data` body: `job` (`{id, mode}`, first), `step` (`{id, status, label, detail?,
message?, tool, t, extra?}`; status `running`, `done`, `skipped` or `error`; step ids `repos`, `scrape`, `audience`,
`understand`, `storyboard`, `plancheck`, `launchpack`, `ai_clips`, `cover`, `music`, `narration`, `record`, `review`,
`retake`, `video`, each in its own order only; `tool` is gemini, fal, chromium, ffmpeg or github; `extra` carries
one-off payloads such as page thumbnails and review verdicts), `log` (`{message}`), `status` (a full
`JobStatus` snapshot, what the dashboard renders), then `done` (the full job) or `error` (`{message}`) last. The
server sends `: keep-alive` comments. Reconnecting with `Last-Event-ID` replays what was missed; a job no longer in
memory (server restart) gets a single `done` or `error` with id 1. `GET /api/jobs/{job_id}` returns the same object as
`done`, `GET /api/jobs` lists recent finished jobs, and `GET /api/pipeline/status/{job_id}` is the older alias.

**Personalized demos.** `PipelineRequest.audience` (company, persona, website, first name, goal, link, notes) makes
the run for one customer: their public site is read for at most 15 s, a source-tagged brief steers the storyboard,
the intro/outro cards name them, and the launch kit gains `outreach_email` and `linkedin_dm` carrying the literal
`{{DEMO_LINK}}` placeholder. The job carries `audience` from the 202 on (source `pending` until research finishes).

**Errors.** `4xx` responses carry `detail` (a sentence). Validation errors (422) carry `message` (one plain sentence)
plus Pydantic's per-field `detail`.

**Sign-in capture.** `POST /api/app-logins/capture` opens a server-side browser; the dashboard drives it over the
WebSocket `/api/app-logins/capture/{capture_id}/ws` (binary frames are JPEG screencast frames; JSON text messages
`{"type": "status"|"notice"|"closed"}` flow out, and `{"type": "click"|"text"|"key"|"wheel"|"back"|"navigate"}`
flow in). WebSockets are not part of OpenAPI, so that endpoint is only described here.
"""

OPENAPI_TAGS = [
    {"name": "health", "description": "Service status and which providers are configured."},
    {"name": "auth", "description": "Accounts and the session cookie."},
    {"name": "pipeline", "description": "Start a demo job, follow it live, fetch the video."},
    {"name": "jobs", "description": "Jobs by id and the recent-jobs list (Launch Kit v2 contract, backend issue #3)."},
    {"name": "settings", "description": "Per-user GitHub token for private repositories."},
    {"name": "app-logins", "description": "Saved sign-in sessions so demos can record pages behind a login."},
    {"name": "fixtures", "description": "Staged sample data: captured API responses, polished by Gemini, replayed while recording."},
]

app = FastAPI(
    title="DemoDay Launch Kit Engine",
    description=API_DESCRIPTION,
    version="1.4.0",
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,  # the dashboard authenticates with an HttpOnly session cookie
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth.router)


def _validation_sentence(errors: List[Dict]) -> str:
    """One plain sentence for a 422: a validator's own message, or `field: what is wrong`."""
    if not errors:
        return "Invalid request."
    first = errors[0]
    msg = re.sub(r"^(Value|Assertion) error, ", "", str(first.get("msg") or "invalid value")).strip()
    if str(first.get("type", "")).startswith("value_error"):
        return msg if msg.endswith(".") else msg + "."
    field = ".".join(str(p) for p in first.get("loc", []) if p not in ("body", "query", "path")) or "request"
    return f"{field}: {msg[:1].lower() + msg[1:]}."


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = jsonable_encoder(exc.errors())
    return JSONResponse(status_code=422, content={"message": _validation_sentence(errors), "detail": errors})


@app.get("/api/health", response_model=HealthStatus, tags=["health"], summary="Service status and capabilities")
async def health():
    return {"status": "ok", "service": "demoday-backend", **capabilities()}


@app.post("/api/pipeline/share/{job_id}", response_model=ShareLink, tags=["pipeline"],
          summary="A link to the finished video that works without signing in",
          responses={401: {"model": ErrorDetail}, 404: {"model": ErrorDetail},
                     503: {"model": ErrorDetail, "description": "Share links are not enabled on this server"}})
async def share_video(job_id: str, user: Dict = Depends(auth.current_user)):
    """A link to the finished video that works without signing in (a copy in the public share bucket)."""
    job = _job_for(job_id, user)
    video_path = storage_root() / job_id / "demo.mp4"
    if not job.video_url or not video_path.is_file():
        raise HTTPException(404, "Video file not found")
    try:
        url = await cloud.share_video_url(job_id, video_path)
    except cloud.CloudUnavailable as exc:
        raise HTTPException(503, str(exc))
    return {"url": url, "expires_at": None}


# --------------------------------------------------------------------------- pipeline


def _job_for(job_id: str, user: Dict) -> JobStatus:
    """The caller's job (live, or reloaded from job.json after a restart). Other users' jobs are a 404."""
    row = db.one("SELECT user_id FROM jobs WHERE id = ?", (job_id,)) if JOB_ID_RE.match(job_id) else None
    if row is None or row["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS.get(job_id)
    if job is None:
        saved = storage_root() / job_id / "job.json"
        if not saved.is_file():
            raise HTTPException(status_code=404, detail="Job not found (the server restarted while it was running)")
        job = JobStatus.model_validate_json(saved.read_text())
        JOBS[job_id] = job
    return job


@app.post("/api/pipeline/start", response_model=JobStarted, status_code=202, tags=["pipeline"], summary="Start a demo job",
          responses={401: {"model": ErrorDetail},
                     422: {"model": ErrorMessage, "description": "Invalid request (a non-http URL, an unknown goal, ...)"}})
async def start_pipeline(req: PipelineRequest, user: Dict = Depends(auth.current_user)):
    job = create_pipeline_job(req, user_id=user["id"])
    return JobStarted(**job.model_dump(), jobId=job.job_id)


@app.get("/api/jobs", response_model=List[JobSummary], tags=["jobs"], summary="Your recent finished jobs, newest first",
         responses={401: {"model": ErrorDetail}})
async def list_jobs(user: Dict = Depends(auth.current_user), limit: int = Query(20, ge=1, le=100)):
    rows = db.query("SELECT id FROM jobs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user["id"], limit * 3))
    summaries: List[JobSummary] = []
    for row in rows:
        try:
            job = _job_for(row["id"], user)
        except HTTPException:
            continue  # lost in a server restart
        if job.status not in ("completed", "failed"):
            continue
        created = datetime.fromtimestamp(job.created_at, timezone.utc).isoformat() if job.created_at else None
        summaries.append(JobSummary(id=job.job_id, url=job.app_url, createdAt=created, status=job.status,
                                    productName=job.storyboard.project_name if job.storyboard else None,
                                    customer=job.audience.company if job.audience else None,
                                    persona=job.audience.persona if job.audience else None,
                                    customerGoal=job.audience.goal if job.audience else None,
                                    durationSec=job.storyboard.total_duration_sec if job.storyboard else None))
        if len(summaries) >= limit:
            break
    return summaries


@app.get("/api/jobs/{job_id}", response_model=JobStatus, tags=["jobs"],
         summary="The full job, running or finished (the same object as the `done` event)",
         responses={401: {"model": ErrorDetail}, 404: {"model": ErrorDetail, "description": "Not your job, or lost in a server restart"}})
async def get_job(job_id: str, user: Dict = Depends(auth.current_user)):
    return _job_for(job_id, user)


@app.get("/api/pipeline/status/{job_id}", response_model=JobStatus, tags=["pipeline"], summary="Current job snapshot",
         responses={401: {"model": ErrorDetail}, 404: {"model": ErrorDetail, "description": "Not your job, or lost in a server restart"}})
async def get_status(job_id: str, user: Dict = Depends(auth.current_user)):
    return _job_for(job_id, user)


@app.get("/api/pipeline/events/{job_id}", tags=["pipeline"], summary="Live job updates (Server-Sent Events)",
         response_class=StreamingResponse,
         responses={200: {"description": "text/event-stream of typed events (`job`, `step`, `log`, `status`, then `done` or "
                                         "`error`), each with an `id`; send `Last-Event-ID` to replay what was missed",
                          "content": {"text/event-stream": {"schema": {"type": "string",
                                      "example": "retry: 3000\n\nid: 1\nevent: job\ndata: {\"id\": \"...\", \"mode\": {...}}\n\n"
                                                 "id: 2\nevent: step\ndata: {\"id\": \"scrape\", \"status\": \"running\", ...}\n\n"}}}},
                    401: {"model": ErrorDetail}, 404: {"model": ErrorDetail}})
@app.get("/api/pipeline/stream/{job_id}", include_in_schema=False)  # backwards-compatible alias
async def stream_events(job_id: str, request: Request, user: Dict = Depends(auth.current_user)):
    _job_for(job_id, user)
    last = (request.headers.get("last-event-id") or "").strip()
    return StreamingResponse(stream_job_events(job_id, int(last) if last.isdigit() else 0),
                             media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/api/pipeline/video/{job_id}", tags=["pipeline"], summary="The finished MP4 (supports Range requests)",
         response_class=FileResponse,
         responses={200: {"description": "video/mp4; inline, or an attachment with ?download=1",
                          "content": {"video/mp4": {"schema": {"type": "string", "format": "binary"}}}},
                    401: {"model": ErrorDetail}, 404: {"model": ErrorDetail}})
async def get_video(job_id: str, download: bool = False, user: Dict = Depends(auth.current_user)):
    _job_for(job_id, user)
    video_path = storage_root() / job_id / "demo.mp4"
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"demo-{job_id}.mp4",
        content_disposition_type="attachment" if download else "inline",
    )


# --------------------------------------------------------------------------- settings: GitHub token


def _require_vault() -> None:
    if not vault.available():
        raise HTTPException(503, "Saving credentials is turned off on this server: DEMODAY_SECRET_KEY is not set")


class GitHubTokenIn(BaseModel):
    token: str = Field(..., min_length=20, max_length=255)


@app.get("/api/settings", response_model=SettingsOut, tags=["settings"], summary="Saved credentials (metadata only)",
         responses={401: {"model": ErrorDetail}})
async def get_settings(user: Dict = Depends(auth.current_user)):
    return {
        "vault": vault.available(),
        "github": vault.meta(user["id"], "github_token", "default"),
        "app_logins": vault.list_kind(user["id"], "app_login"),
    }


@app.put("/api/settings/github-token", response_model=GitHubConnection, tags=["settings"],
         summary="Save a GitHub token (checked with GitHub, stored encrypted)",
         responses={400: {"model": ErrorDetail, "description": "GitHub rejected the token"},
                    401: {"model": ErrorDetail}, 502: {"model": ErrorDetail}, 503: {"model": ErrorDetail}})
async def save_github_token(body: GitHubTokenIn, user: Dict = Depends(auth.current_user)):
    _require_vault()
    token = body.token.strip()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
               "User-Agent": "DemoDay-Launch-Studio"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{GITHUB_API_URL}/user", headers=headers)
    except httpx.HTTPError:
        raise HTTPException(502, "Couldn't reach GitHub to check the token. Try again.")
    if resp.status_code == 401:
        raise HTTPException(400, "GitHub rejected this token. Check that you copied all of it and that it hasn't expired.")
    if resp.status_code != 200:
        raise HTTPException(502, f"GitHub answered {resp.status_code} while checking the token")
    scopes = resp.headers.get("x-oauth-scopes")  # only classic tokens report scopes
    return vault.put(user["id"], "github_token", "default", token, {
        "login": resp.json().get("login", ""),
        "token_type": "classic" if scopes is not None else "fine-grained",
        "scopes": scopes or "",
    })


@app.delete("/api/settings/github-token", response_model=Ok, tags=["settings"], summary="Remove the GitHub token",
            responses={401: {"model": ErrorDetail}})
async def delete_github_token(user: Dict = Depends(auth.current_user)):
    vault.delete(user["id"], "github_token", name="default")
    return {"ok": True}


# --------------------------------------------------------------------------- app logins (signed-in recording)


class CaptureStartIn(BaseModel):
    app_url: str

    @field_validator("app_url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        return _http_url(value)


@app.get("/api/app-logins", response_model=list[AppLogin], tags=["app-logins"], summary="Saved app logins",
         responses={401: {"model": ErrorDetail}})
async def list_app_logins(user: Dict = Depends(auth.current_user)):
    return vault.list_kind(user["id"], "app_login")


@app.delete("/api/app-logins/{login_id}", response_model=Ok, tags=["app-logins"], summary="Remove a saved login",
            responses={401: {"model": ErrorDetail}, 404: {"model": ErrorDetail}})
async def delete_app_login(login_id: str, user: Dict = Depends(auth.current_user)):
    if not vault.delete(user["id"], "app_login", secret_id=login_id):
        raise HTTPException(404, "Saved login not found")
    return {"ok": True}


@app.post("/api/app-logins/capture", response_model=CaptureStart, tags=["app-logins"],
          summary="Open a server-side browser to sign in to an app",
          description="Returns a capture id; drive the browser over the WebSocket "
                      "`/api/app-logins/capture/{capture_id}/ws` (see the API overview), then call `/finish` to save "
                      "the session. The browser closes after 5 idle minutes or 15 minutes in total.",
          responses={400: {"model": ErrorDetail, "description": "Private-network URL refused"},
                     401: {"model": ErrorDetail}, 503: {"model": ErrorDetail}})
async def start_capture(body: CaptureStartIn, user: Dict = Depends(auth.current_user)):
    _require_vault()
    try:
        session = await CAPTURES.start(user["id"], body.app_url)
    except BlockedURL as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        logger.exception("could not start sign-in capture")
        raise HTTPException(500, "Couldn't start the sign-in browser")
    return {"capture_id": session.id, "origin": session.origin, "viewport": CAPTURE_VIEWPORT}


@app.post("/api/app-logins/capture/{capture_id}/finish", response_model=AppLogin, tags=["app-logins"],
          summary="Save the signed-in session (cookies, local storage and IndexedDB, encrypted)",
          responses={400: {"model": ErrorDetail, "description": "No session yet: sign in first"},
                     401: {"model": ErrorDetail}, 404: {"model": ErrorDetail}})
async def finish_capture(capture_id: str, user: Dict = Depends(auth.current_user)):
    session = CAPTURES.get(capture_id, user["id"])
    if session is None:
        raise HTTPException(404, "This sign-in window has closed. Start again.")
    _require_vault()
    # IndexedDB included: Firebase Auth (GitHub/Google sign-in) keeps the signed-in user there, not in cookies.
    state = await session.context.storage_state(indexed_db=True)
    cookies, origins = state.get("cookies", []), state.get("origins", [])
    if not cookies and not any(o.get("localStorage") for o in origins) and not indexed_db_records(state):
        raise HTTPException(400, "No session was found yet. Sign in to the app first, then save.")
    last_url = session.page.url if session.page else session.app_url
    # Some apps keep their access token in sessionStorage, which Playwright's storage state leaves out.
    session_storage = {}
    try:
        if session.page and not session.page.is_closed():
            items = await session.page.evaluate("() => Object.assign({}, window.sessionStorage)")
            if isinstance(items, dict) and items:
                session_storage[origin_of(session.page.url)] = items
    except Exception:
        pass
    await CAPTURES.close(session)
    saved = {"storage_state": state, "session_storage": session_storage}
    return vault.put(user["id"], "app_login", session.origin, json.dumps(saved),
                     {"origin": session.origin, "cookie_count": len(cookies), "last_url": last_url,
                      "session_storage_keys": sum(len(v) for v in session_storage.values()),
                      "indexed_db_records": indexed_db_records(state)})


@app.delete("/api/app-logins/capture/{capture_id}", response_model=Ok, tags=["app-logins"], summary="Close the sign-in browser without saving",
            responses={401: {"model": ErrorDetail}})
async def cancel_capture(capture_id: str, user: Dict = Depends(auth.current_user)):
    session = CAPTURES.get(capture_id, user["id"])
    if session is not None:
        await CAPTURES.close(session)
    return {"ok": True}


@app.websocket("/api/app-logins/capture/{capture_id}/ws")
async def capture_socket(websocket: WebSocket, capture_id: str):
    """Live view of the sign-in browser: JPEG frames and status out, clicks/typing/scrolling in."""
    user = auth.user_from_token(websocket.cookies.get(auth.COOKIE))
    session = CAPTURES.get(capture_id, user["id"]) if user else None
    origin = websocket.headers.get("origin", "")
    # Same-origin (the dashboard served by this service) or a listed dashboard origin; anything else is refused.
    same_origin = origin and origin.split("://", 1)[-1] == websocket.headers.get("host", "")
    if not (same_origin or origin in ORIGINS) or session is None:
        await websocket.close(code=4403)
        return
    await websocket.accept()

    async def receive_inputs() -> None:
        try:
            while True:
                try:
                    message = json.loads(await websocket.receive_text())
                except (ValueError, TypeError, KeyError):  # malformed or binary frame: ignore it
                    continue
                if isinstance(message, dict):
                    await CAPTURES.handle_input(session, message)
        except (WebSocketDisconnect, RuntimeError):
            return

    receiver = asyncio.create_task(receive_inputs())
    receiver.add_done_callback(lambda _: session.changed.set())
    sent_seq, sent_url = -1, None
    try:
        while CAPTURES.is_open(session) and not receiver.done():
            try:
                await asyncio.wait_for(session.changed.wait(), timeout=20)
            except asyncio.TimeoutError:
                continue
            session.changed.clear()
            if session.url != sent_url:
                sent_url = session.url
                await websocket.send_json({"type": "status", "url": session.url})
            if session.notice:
                await websocket.send_json({"type": "notice", "message": session.notice})
                session.notice = None
            if session.frame is not None and session.frame_seq != sent_seq:
                sent_seq = session.frame_seq
                await websocket.send_bytes(session.frame)
        if not CAPTURES.is_open(session):
            await websocket.send_json({"type": "closed"})
            await websocket.close()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        receiver.cancel()


# --------------------------------------------------------------------------- staged sample data


class FixtureCaptureIn(BaseModel):
    app_url: str

    @field_validator("app_url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        return _http_url(value)


class FixturePolishIn(BaseModel):
    brief: str = Field(default="", max_length=500)


class FixtureUpdateIn(BaseModel):
    entries: list = Field(..., description="Full list of FixtureEntry objects; `body` may be an object or a JSON string")


@app.get("/api/fixtures", response_model=list[FixtureSetSummary], tags=["fixtures"], summary="Staged data sets (one per app origin)",
         responses={401: {"model": ErrorDetail}})
async def list_fixtures(user: Dict = Depends(auth.current_user)):
    return fixtures.list_sets(user["id"])


@app.get("/api/fixtures/{set_id}", response_model=FixtureSet, tags=["fixtures"], summary="A staged data set with its entries",
         responses={401: {"model": ErrorDetail}, 404: {"model": ErrorDetail}})
async def get_fixture(set_id: str, user: Dict = Depends(auth.current_user)):
    found = fixtures.get_set(user["id"], set_id)
    if found is None:
        raise HTTPException(404, "Sample data set not found")
    return found


@app.post("/api/fixtures/capture", response_model=FixtureSet, tags=["fixtures"],
          summary="Capture the JSON API responses of an app (signed in when a login is saved)",
          responses={400: {"model": ErrorDetail}, 401: {"model": ErrorDetail},
                     422: {"model": ErrorDetail, "description": "The pages made no JSON API calls"},
                     502: {"model": ErrorDetail, "description": "The app could not be loaded"}})
async def capture_fixture(body: FixtureCaptureIn, user: Dict = Depends(auth.current_user)):
    """Loads the app (signed in when a login is saved) and records its JSON API responses as a new set."""
    from app.services.recorder import declutter_page, settle_page

    try:
        await ensure_public_url(body.app_url)
    except BlockedURL as exc:
        raise HTTPException(400, str(exc))
    origin = origin_of(body.app_url)
    storage_state = None
    if vault.available():
        try:
            saved = vault.get(user["id"], "app_login", origin)
            if saved:
                playwright_state, session_storage = unpack_login(saved)
                storage_state = dict(playwright_state, _session_storage=session_storage) if session_storage else playwright_state
        except vault.VaultUnavailable:
            storage_state = None
    try:
        entries = await fixtures.capture_fixtures(body.app_url, storage_state, settle_page, declutter_page)
    except Exception as exc:
        logger.exception("fixture capture failed")
        raise HTTPException(502, f"Couldn't load {body.app_url}: {str(exc).strip().splitlines()[0][:160]}")
    if not entries:
        raise HTTPException(422, "The page made no JSON API calls, so there is nothing to stage. "
                                 "Sample data needs pages that load their data from JSON APIs in the browser.")
    return fixtures.save_set(user["id"], origin, entries, polished=False)


@app.post("/api/fixtures/{set_id}/polish", response_model=FixtureSet, tags=["fixtures"],
          summary="Rewrite the set as rich sample data with Gemini (same schema)",
          responses={401: {"model": ErrorDetail}, 404: {"model": ErrorDetail}, 502: {"model": ErrorDetail}})
async def polish_fixture(set_id: str, body: FixturePolishIn, user: Dict = Depends(auth.current_user)):
    found = fixtures.get_set(user["id"], set_id)
    if found is None:
        raise HTTPException(404, "Sample data set not found")
    try:
        entries, note = await fixtures.polish_with_gemini(found["entries"], found["origin"], body.brief)
    except Exception as exc:
        raise HTTPException(502, f"Gemini couldn't improve the data: {str(exc).strip().splitlines()[0][:200]}")
    saved = fixtures.save_set(user["id"], found["origin"], entries, polished=True, brief=body.brief)
    return {**saved, "note": note}


@app.put("/api/fixtures/{set_id}", response_model=FixtureSet, tags=["fixtures"], summary="Replace the entries (JSON-validated)",
         responses={401: {"model": ErrorDetail}, 404: {"model": ErrorDetail}, 422: {"model": ErrorDetail}})
async def update_fixture(set_id: str, body: FixtureUpdateIn, user: Dict = Depends(auth.current_user)):
    found = fixtures.get_set(user["id"], set_id)
    if found is None:
        raise HTTPException(404, "Sample data set not found")
    try:
        entries = fixtures.validate_entries(body.entries)
        return fixtures.save_set(user["id"], found["origin"], entries, polished=found["polished"], brief=found["brief"])
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.delete("/api/fixtures/{set_id}", response_model=Ok, tags=["fixtures"], summary="Remove a staged data set",
            responses={401: {"model": ErrorDetail}, 404: {"model": ErrorDetail}})
async def delete_fixture(set_id: str, user: Dict = Depends(auth.current_user)):
    if not fixtures.delete_set(user["id"], set_id):
        raise HTTPException(404, "Sample data set not found")
    return {"ok": True}


# --------------------------------------------------------------------------- static frontend (production)

# In production the Next.js static export (next build with NEXT_OUTPUT=export) is served from the same origin as
# the API, so cookies, the event stream and the sign-in WebSocket never cross sites. Registered last: API routes win.
_static_dir = os.getenv("STATIC_DIR", str(BACKEND_ROOT / "static"))
if os.path.isdir(_static_dir) and os.path.isfile(os.path.join(_static_dir, "index.html")):
    @app.api_route("/api/{_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"], include_in_schema=False)
    async def api_not_found(_path: str):  # unknown API paths stay JSON 404s instead of the site's 404 page
        raise HTTPException(status_code=404, detail="Not found")

    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="frontend")
    logger.info("serving the dashboard from %s", _static_dir)
