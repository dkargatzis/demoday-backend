import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Dict

import httpx
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.config import GEMINI_MODEL, GITHUB_API_URL, capabilities, storage_root  # loads .env before anything else
from app import auth, db, vault
from app.models.schemas import JobStatus, PipelineRequest, _http_url
from app.services import fixtures
from app.services.capture import CAPTURES, origin_of
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


app = FastAPI(
    title="DemoDay Launch Kit Engine",
    description="Autonomous browser recording, fal.ai voiceover, and pitch generation",
    version="1.2.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,  # the dashboard authenticates with an HttpOnly session cookie
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth.router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "demoday-backend", **capabilities()}


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


@app.post("/api/pipeline/start", response_model=JobStatus)
async def start_pipeline(req: PipelineRequest, user: Dict = Depends(auth.current_user)):
    return create_pipeline_job(req, user_id=user["id"])


@app.get("/api/pipeline/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str, user: Dict = Depends(auth.current_user)):
    return _job_for(job_id, user)


@app.get("/api/pipeline/events/{job_id}")
@app.get("/api/pipeline/stream/{job_id}", include_in_schema=False)  # backwards-compatible alias
async def stream_events(job_id: str, user: Dict = Depends(auth.current_user)):
    _job_for(job_id, user)
    return StreamingResponse(stream_job_events(job_id), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/api/pipeline/video/{job_id}")
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


@app.get("/api/settings")
async def get_settings(user: Dict = Depends(auth.current_user)):
    return {
        "vault": vault.available(),
        "github": vault.meta(user["id"], "github_token", "default"),
        "app_logins": vault.list_kind(user["id"], "app_login"),
    }


@app.put("/api/settings/github-token")
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


@app.delete("/api/settings/github-token")
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


@app.get("/api/app-logins")
async def list_app_logins(user: Dict = Depends(auth.current_user)):
    return vault.list_kind(user["id"], "app_login")


@app.delete("/api/app-logins/{login_id}")
async def delete_app_login(login_id: str, user: Dict = Depends(auth.current_user)):
    if not vault.delete(user["id"], "app_login", secret_id=login_id):
        raise HTTPException(404, "Saved login not found")
    return {"ok": True}


@app.post("/api/app-logins/capture")
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


@app.post("/api/app-logins/capture/{capture_id}/finish")
async def finish_capture(capture_id: str, user: Dict = Depends(auth.current_user)):
    session = CAPTURES.get(capture_id, user["id"])
    if session is None:
        raise HTTPException(404, "This sign-in window has closed. Start again.")
    _require_vault()
    state = await session.context.storage_state()
    cookies, origins = state.get("cookies", []), state.get("origins", [])
    if not cookies and not any(o.get("localStorage") for o in origins):
        raise HTTPException(400, "No session was found yet. Sign in to the app first, then save.")
    last_url = session.page.url if session.page else session.app_url
    await CAPTURES.close(session)
    return vault.put(user["id"], "app_login", session.origin, json.dumps(state),
                     {"origin": session.origin, "cookie_count": len(cookies), "last_url": last_url})


@app.delete("/api/app-logins/capture/{capture_id}")
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
    if websocket.headers.get("origin") not in ORIGINS or session is None:
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
    entries: list


@app.get("/api/fixtures")
async def list_fixtures(user: Dict = Depends(auth.current_user)):
    return fixtures.list_sets(user["id"])


@app.get("/api/fixtures/{set_id}")
async def get_fixture(set_id: str, user: Dict = Depends(auth.current_user)):
    found = fixtures.get_set(user["id"], set_id)
    if found is None:
        raise HTTPException(404, "Sample data set not found")
    return found


@app.post("/api/fixtures/capture")
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
            storage_state = json.loads(saved) if saved else None
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


@app.post("/api/fixtures/{set_id}/polish")
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


@app.put("/api/fixtures/{set_id}")
async def update_fixture(set_id: str, body: FixtureUpdateIn, user: Dict = Depends(auth.current_user)):
    found = fixtures.get_set(user["id"], set_id)
    if found is None:
        raise HTTPException(404, "Sample data set not found")
    try:
        entries = fixtures.validate_entries(body.entries)
        return fixtures.save_set(user["id"], found["origin"], entries, polished=found["polished"], brief=found["brief"])
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.delete("/api/fixtures/{set_id}")
async def delete_fixture(set_id: str, user: Dict = Depends(auth.current_user)):
    if not fixtures.delete_set(user["id"], set_id):
        raise HTTPException(404, "Sample data set not found")
    return {"ok": True}
