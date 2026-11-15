"""Pipeline orchestration and live job state (in memory) streamed to the dashboard over SSE."""
import asyncio
import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from app import db, vault
from app.config import capabilities, feature_on, get_secret, storage_root, work_root
from app.services import analysis as analysis_mod
from app.services import capabilities as capabilities_mod
from app.services import funnels as funnels_mod
from app.services import ledger, music, plancheck, review, vision
from app.models.schemas import JobStatus, PipelineRequest, StepState
from app.services import audience
from app.services.audio import synthesize_scene_audio, warm_up_fal
from app.services.bookends import (FAL_IMAGE_MODEL, FAL_VIDEO_MODEL, POSTER_HEIGHT, POSTER_WIDTH, prepare_bookends,
                                   prepare_poster)
from app.services.capture import describe_login, origin_of, unpack_login
from app.services.fixtures import FixtureReplayer, get_set_for_origin
from app.services.ingest import extract_repos_context, probe_app_url
from app.services.netguard import BlockedURL, ensure_public_url
from app.services.planner import generate_launch_kit, generate_storyboard, project_name
from app.services.recorder import fit_dwells, record_session
from app.services.stitcher import concat_scenes, describe_media, frame_aligned, make_bookend, mux_scene_clip

logger = logging.getLogger("demoday.pipeline")

JOBS: Dict[str, JobStatus] = {}
_TASKS: Set[asyncio.Task] = set()  # strong refs: asyncio only keeps weak refs to running tasks
# Screen recording is timing-sensitive and encoding is CPU-heavy: render one job at a time.
_RENDER_SLOT = asyncio.Semaphore(1)
SCENE_TAIL_SEC = 0.4  # breathing room after each narration line
BOOKEND_VOICE_DELAY_SEC = 0.4  # the intro/outro line starts after the title fades in
COVER_GRACE_SEC = 20  # a poster still painting when the cut is done gets this long, then the job ships without it
TERMINAL = ("completed", "failed")
STEP_LABELS = {
    "repos": "Reading the repositories", "scrape": "Exploring the live app", "audience": "Researching the customer",
    "understand": "Understanding the pages", "storyboard": "Planning the storyboard", "plancheck": "Checking the plan",
    "launchpack": "Writing the launch kit", "ai_clips": "AI intro and outro", "cover": "Making the cover and title cards",
    "funnels": "Finding the strongest funnels", "code": "Reading the code", "analysis": "Analyzing for the prospect",
    "capabilities": "Mapping the backend against the app",
    "music": "Composing the soundtrack", "narration": "Synthesizing narration", "record": "Recording the browser session",
    "review": "Reviewing the footage", "retake": "Retaking a scene", "video": "Assembling demo.mp4",
}
STEP_TOOLS = {
    "repos": "github", "scrape": "chromium", "audience": "gemini", "understand": "gemini", "storyboard": "gemini",
    "plancheck": "gemini", "launchpack": "gemini", "ai_clips": "fal", "cover": "fal", "music": "fal", "narration": "fal",
    "funnels": "gemini", "code": "github", "analysis": "gemini", "capabilities": "github",
    "record": "chromium", "review": "gemini", "retake": "chromium", "video": "ffmpeg",
}


class JobEvents:
    """One job's typed event log: `id`-numbered SSE messages, replayable from `Last-Event-ID`."""

    def __init__(self) -> None:
        self.items: List[Dict[str, Any]] = []
        self.closed = False
        self._changed = asyncio.Event()

    def emit(self, event: str, data: Any) -> None:
        self.items.append({"id": len(self.items) + 1, "event": event, "data": data})
        self._wake()

    def close(self) -> None:
        self.closed = True
        self._wake()

    def _wake(self) -> None:
        waiting, self._changed = self._changed, asyncio.Event()
        waiting.set()

    async def wait(self, timeout: float) -> None:
        try:
            await asyncio.wait_for(self._changed.wait(), timeout)
        except asyncio.TimeoutError:
            pass


class _Steps:
    """Keeps the agent trace (`job.steps`, one row per step) and emits `step` events with the tool and elapsed time.

    `extra` (thumbnails, verdicts) travels only in the one-off event, never in the JobStatus snapshot.
    """

    def __init__(self, events: JobEvents, job: Optional[JobStatus] = None, t0: Optional[float] = None) -> None:
        self.events, self.job, self.t0 = events, job, t0 if t0 is not None else time.monotonic()
        self.running: List[str] = []
        self.tools: Dict[str, str] = {}  # a step's tool when this run uses another one (the cover framed by chromium)

    def __call__(self, step_id: str, status: str, detail: Optional[str] = None, message: Optional[str] = None,
                 extra: Optional[Dict[str, Any]] = None, tool: Optional[str] = None) -> None:
        if tool:
            self.tools[step_id] = tool
        if status == "running":
            self.running.append(step_id)
        else:
            self.running = [s for s in self.running if s != step_id]
        label, tool = STEP_LABELS.get(step_id, step_id), self.tools.get(step_id) or STEP_TOOLS.get(step_id)
        self._upsert(step_id, status, label, tool, detail, message)
        data: Dict[str, Any] = {"id": step_id, "status": status, "label": label, "tool": tool,
                                "t": round(time.monotonic() - self.t0, 1)}
        if detail:
            data["detail"] = detail
        if message:
            data["message"] = message
        if extra:
            data["extra"] = extra
        self.events.emit("step", data)

    def note(self, step_id: str, detail: str) -> None:
        """Updates a running row's detail (live counters) without emitting an event."""
        for row in (self.job.steps if self.job else []):
            if row.id == step_id:
                row.detail = detail
                return

    def fail(self, message: str) -> None:
        for step_id in list(self.running):
            self(step_id, "error", message=message)

    def _upsert(self, step_id: str, status: str, label: str, tool: Optional[str], detail: Optional[str],
                message: Optional[str]) -> None:
        if self.job is None:
            return
        now = time.time()
        row = next((r for r in self.job.steps if r.id == step_id), None)
        if row is None:
            row = StepState(id=step_id, status=status, label=label, tool=tool)
            self.job.steps.append(row)
        row.status, row.label, row.tool = status, label, tool
        if detail:
            row.detail = detail
        if message:
            row.message = message
        if status == "running":
            row.started_at = row.started_at or now
            row.ended_at = None
        else:
            row.started_at = row.started_at or now
            row.ended_at = now


EVENTS: Dict[str, JobEvents] = {}


def run_mode() -> Dict[str, Any]:
    """What produces the copy and the video on this server (the `job` event's `mode`)."""
    caps = capabilities()
    return {"demo": False, "copySource": "gemini" if caps.get("gemini") else "template",
            "copyModel": caps.get("gemini_model") if caps.get("gemini") else None, "videoSource": "playwright"}


async def _watch_job(job: JobStatus, events: JobEvents) -> None:
    """Emits a `status` snapshot whenever the job changes, then one `done` or `error`, and closes the log."""
    last = None
    while True:
        payload = job.model_dump(mode="json")
        if payload != last:
            last = payload
            events.emit("status", payload)
        if job.status in TERMINAL:
            if job.status == "completed":
                events.emit("done", payload)
            else:
                events.emit("error", {"message": job.error or "The job failed"})
            events.close()
            return
        await asyncio.sleep(0.4)


def _path_of(url: str) -> str:
    from urllib.parse import urlparse

    return (urlparse(url).path or "/").rstrip("/") or "/"


def _job_logger(job: JobStatus, events: Optional[JobEvents] = None) -> Callable[[str], None]:
    def log(message: str) -> None:
        job.logs.append(message)
        logger.info("[%s] %s", job.job_id, message)
        if events is not None:
            events.emit("log", {"message": message})
    return log


def _user_secrets(user_id: Optional[str], req: PipelineRequest, log: Callable[[str], None]
                  ) -> Tuple[Optional[str], Optional[Dict]]:
    """The job owner's GitHub token and the saved sign-in session for the app's origin. Never logged."""
    if not user_id:
        return None, None
    origin = origin_of(req.app_url)
    try:
        token = vault.get(user_id, "github_token", "default")
        state = vault.get(user_id, "app_login", origin) if req.use_app_login else None
    except vault.VaultUnavailable as exc:
        log(f"Warning: saved credentials are unavailable ({exc}); continuing without them")
        return None, None
    if state is None:
        if not req.use_app_login:
            log("Recording signed out: saved login turned off for this run")
        elif vault.list_kind(user_id, "app_login"):
            log(f"No saved login for {origin}; recording as a signed-out visitor")
        return (token.decode() if token else None), None
    storage_state, session_storage = unpack_login(state)
    log(f"Recording signed in with your saved login for {origin} ({describe_login(storage_state, session_storage)})")
    if session_storage:
        storage_state = dict(storage_state, _session_storage=session_storage)  # carried to the browser contexts
    return (token.decode() if token else None), storage_state


async def run_pipeline_task(job_id: str, req: PipelineRequest, user_id: Optional[str] = None) -> None:
    job = JOBS[job_id]
    ledger.bind(job)  # first: every task created from here on inherits the ledger target
    events = EVENTS.get(job_id) or EVENTS.setdefault(job_id, JobEvents())
    log = _job_logger(job, events)
    started = time.monotonic()
    step = _Steps(events, job, started)
    job.status = "running"
    kit_task: Optional[asyncio.Task] = None
    bookend_task: Optional[asyncio.Task] = None
    audience_task: Optional[asyncio.Task] = None
    music_task: Optional[asyncio.Task] = None
    music_skipped: Optional[Callable[[str], None]] = None
    review_task: Optional[asyncio.Task] = None
    poster_task: Optional[asyncio.Task] = None
    bookend_mode: Optional[str] = None
    background: List[asyncio.Task] = []  # optional work (warm-up, cover, music, review): cancelled before completion
    job_dir: Optional[Path] = None
    _optional = _optional_runner(job_id, log, step)

    try:  # everything, including setup, must land in a terminal state or the job hangs in "running"
        job_dir = work_root() / job_id  # intermediates; the results are copied to storage_root()/job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        try:
            await ensure_public_url(req.app_url)
        except BlockedURL as exc:
            raise RuntimeError(str(exc)) from None
        github_token, storage_state = _user_secrets(user_id, req, log)
        warm_task = asyncio.create_task(warm_up_fal())  # fal.ai's queue warms up while we ingest and plan
        _TASKS.add(warm_task)
        warm_task.add_done_callback(_TASKS.discard)
        background.append(warm_task)
        replayer = None
        if user_id and req.use_staged_data:
            fixture_set = get_set_for_origin(user_id, origin_of(req.app_url))
            if fixture_set and fixture_set["enabled_count"]:
                replayer = FixtureReplayer(fixture_set["entries"])
                log(f"Staged sample data: {fixture_set['enabled_count']} endpoint(s) for {fixture_set['origin']}"
                    + (" (Gemini-polished)" if fixture_set["polished"] else "") + "; the video is labelled 'Sample data'")

        if req.audience:  # researched beside the ingest; build_audience never raises
            step("audience", "running", detail=audience.running_detail(req.audience))
            audience_task = asyncio.create_task(
                audience.build_audience(req.audience, req.target_seconds, log, req.app_url))
        else:
            step("audience", "skipped", detail="No customer in this run")

        # 1. Ingest repositories + live app
        job.stage, job.progress = "ingesting", 5
        if req.repo_urls:
            step("repos", "running", detail=f"{len(req.repo_urls)} repositor{'y' if len(req.repo_urls) == 1 else 'ies'}")
        else:
            step("repos", "skipped", detail="No repositories in this run")
        step("scrape", "running", detail=req.app_url)
        for url in req.repo_urls:
            log(f"Analyzing GitHub repository: {url}")
        log(f"Rendering live app in headless Chromium: {req.app_url}")
        shots_dir = job_dir / "shots"  # local scratch for the vision brief; the trace gets 320 px thumbnails
        repo_info, app_info = await asyncio.gather(
            extract_repos_context(req.repo_urls, github_token),
            probe_app_url(req.app_url, storage_state, replayer, shots_dir=shots_dir if feature_on("VISION") else None,
                          focus=req.instructions,
                          on_page=lambda p: log(f"Explored {p['path']}: '{p['title']}' "
                                                f"({len(p['headings'])} headings, {len(p['buttons'])} buttons)")),
        )
        if app_info.get("error"):
            raise RuntimeError(f"Navigation to {req.app_url} failed: {app_info['error']}")
        for note in repo_info.get("notes", []):
            log(note)
        for repo in repo_info.get("repos", []):
            log(f"Repo {repo['full_name']}{' (private)' if repo['private'] else ''} added to the context")
        log(f"Repository context: {len(repo_info.get('readme', ''))} chars of README, "
            f"{len(repo_info.get('dependencies', []))} dependencies, {len(repo_info.get('routes', []))} page routes")
        if req.repo_urls:
            step("repos", "done", detail=f"{len(repo_info.get('repos', []))} of {len(req.repo_urls)} read, "
                                        f"{len(repo_info.get('routes', []))} routes")
        status = app_info.get("status")
        log(f"Live app \"{app_info.get('title') or 'untitled'}\" (HTTP {status}): {len(app_info['links'])} links, "
            f"{len(app_info['buttons'])} buttons, {len(app_info['inputs'])} inputs")
        pages = app_info.get("pages") or []
        if pages:
            log(f"Brief: explored {len(pages)} more page(s) for the storyboard: " + ", ".join(p["path"] for p in pages))
        step("scrape", "done", detail=f"HTTP {status}: {app_info.get('title') or 'untitled'}; {len(pages) + 1} page(s) explored")
        if status and status >= 400:
            log(f"Warning: the app answered HTTP {status}; the recording may show an error page")
        landed = app_info.get("url") or req.app_url
        if _path_of(landed) != _path_of(req.app_url):
            log(f"Warning: {req.app_url} redirected to {landed}. "
                + ("Your saved login may have expired or not apply to this page: open Settings and choose "
                   "\"Sign in again\", then re-run." if storage_state else
                   "This page may need a login: save an app login in Settings to record it."))
        if app_info.get("login_form"):
            log("Warning: the app still shows a sign-in form with your saved login; it may have expired. "
                "Capture the login again in Settings." if storage_state else
                "Warning: the app shows a sign-in form. Save an app login in Settings to record signed-in pages.")
        brief = None
        if audience_task is not None:
            if not audience_task.done():
                log("Waiting for the customer brief...")
            brief = await audience_task
            job.audience = brief
            step("audience", "done", detail=audience.step_detail(brief))
        job.progress = 20

        # 1b. Funnels: Gemini reads the codebase and the live app for the product's strongest paths, beside Understand
        funnels: List[Dict[str, Any]] = []
        funnel_task: Optional[asyncio.Task] = None
        if feature_on("FUNNELS") and get_secret("GEMINI_API_KEY"):
            evidence = len(repo_info.get("routes") or []) + len(repo_info.get("flow_files") or [])
            step("funnels", "running", detail=f"{evidence} routes and flow files in the code, {len(pages) + 1} live pages")
            funnel_task = asyncio.create_task(funnels_mod.find_funnels(repo_info, app_info, project_name(repo_info, app_info),
                                                                       log, instructions=req.instructions))
            _TASKS.add(funnel_task)
            funnel_task.add_done_callback(_TASKS.discard)
            background.append(funnel_task)

        # 1c. Understand: Gemini looks at the page screenshots once; a failure costs one log line, never the demo
        shots = [p for p in [app_info.get("screenshot"), *(pg.get("screenshot") for pg in pages)] if p]
        if feature_on("VISION") and get_secret("GEMINI_API_KEY") and shots:
            step("understand", "running", detail=f"{len(shots)} screenshots → Gemini")
            log(f"Gemini is looking at {len(shots)} page screenshots...")
            insights = await _optional(vision.describe_pages(app_info, log), 20, "understand", "Understand")
            if insights:
                await vision.thumbnails(insights)
                log(vision.summary_line(insights))
                step("understand", "done", detail=" · ".join(f"{i['path']}: {i['purpose']}" for i in insights)[:200],
                     extra={"pages": insights})
        if funnel_task is not None:
            found = await _optional(funnel_task, funnels_mod.FUNNEL_TIMEOUT_SEC + 15, "funnels", "Funnels")
            if found is not None:
                funnels = found
                step("funnels", "done", detail=funnels_mod.step_detail(found),
                     extra=funnels_mod.as_extra(found) if found else None)

        # 2. Plan the storyboard; write the launch kit in the background meanwhile
        job.stage = "planning"
        step("storyboard", "running", detail="following your instructions" if req.instructions else None)
        if req.instructions:
            log("Your instructions for the tour: \"" + " / ".join(req.instructions.splitlines())[:240] + "\"")
        log("Planning the storyboard...")
        want_music = feature_on("MUSIC") and bool(get_secret("FAL_KEY"))
        # Cover: the poster and the title cards' backdrop, the product's own screen in a browser frame (a FLUX still
        # when the probe saved no screenshot). Title cards: always for a personalized run ("Made for Acme"); for a
        # generic run only with FEATURE_TITLE_CARDS=1.
        product_shots = _product_shots(app_info, funnels) if feature_on("COVER") else []
        cover_off = ("FEATURE_COVER is off" if not feature_on("COVER")
                     else "FAL_KEY not set" if not get_secret("FAL_KEY") and not product_shots else None)
        cover_tool = "chromium" if product_shots else None
        host = _display_host(app_info)
        title_cards = not req.ai_clips and (brief is not None or feature_on("TITLE_CARDS", default=False))
        storyboard, note = await generate_storyboard(
            repo_info, app_info, req.target_seconds, req.voice_style, ", ".join(req.repo_urls),
            ai_clips=req.ai_clips, audience=brief, title_cards=title_cards, want_music=want_music,
            want_cover=cover_off is None and not product_shots, funnels=funnels, instructions=req.instructions,
        )
        log(note)
        job.storyboard = storyboard
        scenes = storyboard.scenes
        n_pages = len({s.route for s in scenes})
        why = f" — why: {storyboard.plan_rationale}" if storyboard.plan_rationale else ""
        log(f"Storyboard: {len(scenes)} scenes across {n_pages} page(s) for \"{storyboard.project_name}\"{why}")
        if brief is not None:
            for line in audience.check_storyboard(storyboard, brief, app_info, repo_info):
                log(line)
        step("storyboard", "done", detail=f"{len(scenes)} scenes across {n_pages} page(s)"
                                          + (f" · {storyboard.plan_rationale}" if storyboard.plan_rationale else f"; {note}"))

        # 2b. Plan check: verified against the inventory before any voice or camera time is spent
        repo_routes = repo_info.get("routes", [])
        verified = sum(len(s.actions) for s in scenes)
        step("plancheck", "running", detail=f"{verified} actions across {len(scenes)} scenes")
        found = plancheck.verify(storyboard, app_info, repo_routes)
        for line in plancheck.notes(storyboard, req.target_seconds, req.voice_style, req.ai_clips):
            log(f"Plan check: {line}")
        left, revised, repair_sec, fixed, reason = [], [], None, 0, None
        if not found:
            log(f"Plan check: {len(scenes)} scenes, {verified} actions verified against the inventory, 0 issues")
        else:
            can_repair = bool(get_secret("GEMINI_API_KEY")) and feature_on("PLANCHECK")
            log(f"Plan check: {plancheck.summarize(found)}" + (" → asking Gemini to revise" if can_repair else ""))
            left = found
            if can_repair:
                t_repair = time.monotonic()
                repaired = await _optional(plancheck.repair(storyboard, found, app_info, log, repo_routes),
                                           15, "plancheck", "Plan check")
                if repaired is not None:
                    repair_sec = time.monotonic() - t_repair
                    revised = plancheck.changed_scenes(storyboard, repaired)
                    storyboard = repaired  # only the revised scenes' routes/actions differ; the narration is the original
                    job.storyboard = storyboard
                    scenes = storyboard.scenes
                    left = plancheck.verify(storyboard, app_info, repo_routes)
                else:
                    reason = "Gemini repair skipped"
            else:
                reason = "no Gemini key" if not get_secret("GEMINI_API_KEY") else "FEATURE_PLANCHECK off"
            fixed = plancheck.drop_invalid(storyboard, left, log)  # the deterministic last word
        step("plancheck", "done", detail=plancheck.describe(verified, found, revised, repair_sec, left, fixed, reason))
        step("launchpack", "running")
        kit_task = asyncio.create_task(
            generate_launch_kit(storyboard, repo_info, app_info, ", ".join(req.repo_urls), req.app_url, audience=brief,
                                log=log)
        )
        # Bookends: "video" = the AI clips (ai_clips); "cards" = typeset title cards on a fal.ai FLUX still (step "cover").
        bookend_mode = "video" if req.ai_clips else "cards" if title_cards else None
        if bookend_mode != "video":
            step("ai_clips", "skipped", detail="Not requested in this run")
        overrides = (audience.bookend_overrides(brief, storyboard.project_name, req.app_url, req.target_seconds)
                     if brief is not None else None)
        if cover_off:
            storyboard.cover = f"skipped: {cover_off}"
            step("cover", "skipped", detail=cover_off + ("; the title cards sit on a gradient" if bookend_mode == "cards" else ""))
        if bookend_mode:
            if overrides:
                log(f"Intro card: \"{overrides['intro']['kicker']}\"; outro card: \"{overrides['outro']['title']}\" -> "
                    f"{overrides['outro']['subtitle']}")
            if bookend_mode == "cards":
                if not cover_off:
                    source = "the product's own screens" if product_shots else FAL_IMAGE_MODEL
                    step("cover", "running", detail=(f"Title cards and poster for {brief.company} · {source}"
                                                     if brief is not None else f"Title cards and poster · {source}"),
                         tool=cover_tool)
            else:
                step("ai_clips", "running", detail=f"Personalized for {brief.company}" if brief is not None else None)
                log(f"Generating the AI intro and outro with fal.ai ({FAL_VIDEO_MODEL})..." if get_secret("FAL_KEY")
                    else "FAL_KEY not set: the intro and outro will be typeset title cards")
            bookend_task = asyncio.create_task(prepare_bookends(
                job_dir, storyboard.project_name, storyboard.tagline, req.app_url, req.repo_urls, req.voice_style,
                {"cover": storyboard.cover_visual, "intro": storyboard.intro_visual, "outro": storyboard.outro_visual},
                log, mode=bookend_mode, overrides=overrides, video=req.ai_clips, paint=cover_off is None,
                product_shots=product_shots, window_host=host, theme_color=app_info.get("theme_color") or ""))
            if bookend_mode == "cards" and not cover_off:
                def _cover_ready(task: asyncio.Task) -> None:  # the thumbnail reaches the trace while the camera rolls
                    if task.cancelled() or task.exception() is not None or job.status in TERMINAL:
                        return  # a failure is reported by _optional when the cut collects the cards
                    ready = task.result()
                    intro = ready["intro"]
                    storyboard.cover = intro.note
                    if not intro.painted:  # FLUX failed: the cards fall back to the gradient and there is no poster
                        step("cover", "error", message=intro.note[:140])
                        return
                    d1, d2 = (frame_aligned(max(3.0, ready[k].narration_sec + BOOKEND_VOICE_DELAY_SEC + 0.6))
                              for k in ("intro", "outro"))
                    step("cover", "done", detail=f"{intro.note} · intro {d1:.1f}s · outro {d2:.1f}s"
                                                 + (f" · poster {POSTER_WIDTH}x{POSTER_HEIGHT}" if intro.poster else ""),
                         extra={"thumb": intro.thumb} if intro.thumb else None)
                bookend_task.add_done_callback(_cover_ready)
        if bookend_mode != "cards" and not cover_off:  # no title cards: the poster is its own small task
            step("cover", "running", tool=cover_tool, detail="The video's poster · "
                 + ("the product's own screen" if product_shots else FAL_IMAGE_MODEL))
            poster_task = asyncio.create_task(prepare_poster(
                job_dir, storyboard.project_name, storyboard.tagline, storyboard.cover_visual or storyboard.intro_visual,
                log, card=overrides["intro"] if overrides else None,
                product_shot=product_shots[0] if product_shots else None, window_host=host,
                theme_color=app_info.get("theme_color") or ""))
            _TASKS.add(poster_task)
            poster_task.add_done_callback(_TASKS.discard)
            background.append(poster_task)

            def _poster_ready(task: asyncio.Task) -> None:  # the thumbnail reaches the trace while the camera rolls
                if task.cancelled() or job.status in TERMINAL:
                    return
                exc = task.exception()
                if exc is not None:
                    reason = (str(exc).strip().splitlines() or [type(exc).__name__])[0][:140]
                    logger.warning("[%s] cover poster failed: %s", job_id, reason)
                    log(f"Cover art skipped: {reason}")
                    storyboard.cover = f"skipped: {reason}"
                    step("cover", "error", message=reason)
                    return
                poster = task.result()
                storyboard.cover = poster.note
                step("cover", "done", detail=f"{poster.note} · {POSTER_WIDTH}x{POSTER_HEIGHT}",
                     extra={"thumb": poster.thumb} if poster.thumb else None)
            poster_task.add_done_callback(_poster_ready)

        if want_music:  # fal.ai composes while we narrate and record; ducked under the voice at the final cut
            music_brief, _ = music.brief_for(storyboard.music_prompt, req.voice_style)

            def _music_skipped(reason: str) -> None:  # amber row + the storyboard note, from compose or finish
                storyboard.soundtrack = f"skipped: {reason}"
                step("music", "error", message=reason)

            music_skipped = _music_skipped
            step("music", "running", detail=f"{music_brief[:90]} → {music.model_label()}")
            music_task = asyncio.create_task(music.compose(
                storyboard.music_prompt, music.bed_seconds_for(req.target_seconds), job_dir, log,
                voice_style=req.voice_style, on_skip=music_skipped))
            _TASKS.add(music_task)
            music_task.add_done_callback(_TASKS.discard)
            background.append(music_task)
        job.progress = 35

        # 3. Narration (all scenes in parallel); durations measured with ffprobe
        job.stage = "recording"
        step("narration", "running", detail=f"{len(scenes)} scenes")
        log(f"Synthesizing narration for {len(scenes)} scenes...")
        audio_paths = [job_dir / f"scene_{i + 1:02d}.wav" for i in range(len(scenes))]
        narrations = await asyncio.gather(*(
            synthesize_scene_audio(scene.narration, path, req.voice_style)
            for scene, path in zip(scenes, audio_paths)
        ))
        for i, (scene, (duration, engine)) in enumerate(zip(scenes, narrations)):
            scene.duration_sec = round(duration, 3)
            log(f"Scene {i + 1}/{len(scenes)}: {duration:.2f}s of narration via {engine}")
        clip_durations = [frame_aligned(duration + SCENE_TAIL_SEC) for duration, _ in narrations]
        retimed = fit_dwells(scenes, clip_durations, log)  # dwells re-timed to the measured voice; never raises
        step("narration", "done", detail=f"{sum(d for d, _ in narrations):.1f}s across {len(scenes)} scenes"
             + (f" · {retimed} scene{'' if retimed == 1 else 's'} re-timed to the voice" if retimed else ""))
        job.progress = 45

        # 4. Record + encode
        if _RENDER_SLOT.locked():
            log("Waiting for another job to finish rendering...")
        async with _RENDER_SLOT:
            step("record", "running", detail=f"{sum(clip_durations):.1f}s of footage")
            log(f"Recording one continuous 1920x1080 browser session ({sum(clip_durations):.1f}s of footage)...")

            def on_scene(idx: int) -> None:
                job.progress = 45 + int(20 * idx / len(scenes))
                log(f"Scene {idx + 1}/{len(scenes)}: recording \"{scenes[idx].title}\" ({clip_durations[idx]:.1f}s)")

            session_path = job_dir / "session.webm"
            segments = await record_session(scenes, req.app_url, clip_durations, session_path, log, on_scene,
                                            storage_state=storage_state, replayer=replayer)
            job.progress = 65

            clips = []
            for i, segment in enumerate(segments):
                log(f"Scene {i + 1}/{len(scenes)}: muxing H.264/AAC clip ({segment.duration:.2f}s)")
                clip = job_dir / f"scene_{i + 1:02d}.mp4"
                await mux_scene_clip(session_path, audio_paths[i], clip, segment.start, segment.duration,
                                     focus=segment.focus)
                if segment.focus:
                    log(f"Scene {i + 1}/{len(scenes)}: {len(segment.focus)} focus zoom(s) toward clicked elements")
                clips.append(clip)
                # Clip length (narration + tail), so cumulative durations are exact scene offsets in demo.mp4.
                scenes[i].duration_sec = round(segment.duration, 3)
                job.progress = 65 + int(17 * (i + 1) / len(segments))
            step("record", "done", detail=f"{len(clips)} clips, {sum(len(s.focus or []) for s in segments)} focus zooms")

        # 4b. Review the footage in the background (Gemini on contact sheets of the muxed clips); it overlaps the
        # bookend wait and encodes, and is collected before the cut while clips[i] still pairs with scenes[i].
        if feature_on("REVIEW") and get_secret("GEMINI_API_KEY"):
            step("review", "running", detail=f"{len(clips)} scenes, {review.FRAMES_PER_SCENE} frames each")
            review_task = asyncio.create_task(review.review_scenes(list(clips), scenes, job_dir, log,
                                                                   project_name=storyboard.project_name))
            _TASKS.add(review_task)
            review_task.add_done_callback(_TASKS.discard)
            background.append(review_task)

        # 5. Intro and outro (outside the render slot while fal.ai may still be generating)
        job.stage = "assembling"
        step("video", "running")
        bookends = None
        if bookend_task is not None and bookend_mode == "cards":
            if not bookend_task.done():
                log("Waiting for the title cards...")
            bookends = await _optional(bookend_task, 60, "cover", "Cover art")  # done row comes from _cover_ready
            if bookends is None:  # the cards failed outright: the reason is on the cover row
                row = next((r for r in job.steps if r.id == "cover"), None)
                storyboard.cover = f"skipped: {row.message if row and row.message else 'the title cards failed'}"
        elif bookend_task is not None:
            if not bookend_task.done():
                log("Waiting for the AI intro and outro from fal.ai...")
            try:
                bookends = await bookend_task
                step("ai_clips", "done")
            except Exception as exc:  # the demo never depends on its bookends
                logger.exception("[%s] bookends failed", job_id)
                log(f"Warning: skipping the intro and outro ({str(exc).strip()[:140] or type(exc).__name__})")
                step("ai_clips", "error", message=str(exc).strip()[:140] or type(exc).__name__)
        intro_path: Optional[Path] = None
        outro_path: Optional[Path] = None
        if bookends:
            async with _RENDER_SLOT:
                for kind, fade in (("intro", "in"), ("outro", "out")):
                    bookend = bookends[kind]
                    duration = frame_aligned(max(3.0, bookend.narration_sec + BOOKEND_VOICE_DELAY_SEC + 0.6))
                    path = job_dir / f"{kind}.mp4"
                    await make_bookend(path, bookend.narration, bookend.overlay, duration, fade,
                                       video_path=bookend.video, background_png=bookend.background,
                                       voice_delay_sec=BOOKEND_VOICE_DELAY_SEC)
                    if kind == "intro":
                        intro_path, storyboard.intro_sec = path, duration
                    else:
                        outro_path, storyboard.outro_sec = path, duration
                    log(f"{kind.capitalize()} clip ready ({duration:.1f}s)")

        # 5b. Review verdicts and the gated retake, while clips[i] still pairs with scenes[i]
        if review_task is not None:
            verdicts = await _optional(review_task, 20, "review", "Review")
            if verdicts is not None:
                step("review", "done", detail=review.summarize(verdicts), extra=review.as_extra(verdicts))
                idx = review.should_retake(verdicts)
                if idx is None:
                    step("retake", "skipped", detail="Not needed")
                elif not feature_on("RETAKE", default=False):
                    step("retake", "skipped", detail=f"scene {idx + 1} qualifies; retakes are off (FEATURE_RETAKE=0)")
                else:
                    flagged = verdicts[idx]
                    step("retake", "running", detail=f"scene {idx + 1}: {flagged.issue or flagged.seen or flagged.verdict}")
                    log(f"Retake: re-recording scene {idx + 1} '{scenes[idx].title}' once in a fresh session...")
                    async with _RENDER_SLOT:  # the recording is timing-sensitive
                        new_clip = await _optional(
                            review.retake_scene(scenes[idx], idx, req.app_url, clip_durations[idx], audio_paths[idx],
                                                job_dir, log, storage_state=storage_state, replayer=replayer,
                                                project_name=storyboard.project_name),
                            clip_durations[idx] + 45, "retake", "Retake")
                    if new_clip:
                        clips[idx] = new_clip  # same length and codec parameters: drops straight into the cut
                        step("retake", "done", detail=f"scene {idx + 1} re-recorded, verified {scenes[idx].verdict}, spliced in")
                    elif any(r.id == "retake" and r.status == "running" for r in job.steps):
                        step("retake", "skipped", detail=f"scene {idx + 1}: the retake was not better, kept the original")

        # 5c. Assemble: intro and outro go in only now
        async with _RENDER_SLOT:
            clips = ([intro_path] if intro_path else []) + clips + ([outro_path] if outro_path else [])
            job.progress = 86
            log("Stitching scenes into demo.mp4...")
            raw_path = job_dir / "demo_raw.mp4"
            await concat_scenes(clips, raw_path)

        # 5d. Soundtrack: a short grace for a late track, then the ducked mix; the raw cut ships on any failure
        final_path = job_dir / "demo.mp4"
        if music_task is not None and not music_task.done():
            log(f"Waiting up to {music.GRACE_SEC}s for the soundtrack from fal.ai...")
        soundtrack = await music.finish(music_task, raw_path, final_path, log, on_skip=music_skipped)
        if soundtrack:
            storyboard.soundtrack = soundtrack
            step("music", "done", detail=soundtrack)

        # 5e. The cover poster: usually painted long ago; a late one gets a short grace, a missing one costs nothing
        poster_path = bookends["intro"].poster if bookends and bookend_mode == "cards" else None
        if poster_task is not None:
            if not poster_task.done():
                log(f"Waiting up to {COVER_GRACE_SEC}s for the cover art from fal.ai...")
                await asyncio.wait({poster_task}, timeout=COVER_GRACE_SEC)
            if not poster_task.done():
                poster_task.cancel()
                storyboard.cover = "skipped: took too long"
                log("Cover art skipped: took too long")
                step("cover", "error", message="took too long")
            elif not poster_task.cancelled() and poster_task.exception() is None:
                poster_path = poster_task.result().path
        media = await describe_media(final_path)
        final_path = _publish(job_dir, job_id, final_path, poster_path)  # demo.mp4 (+ cover.jpg) into storage
        storyboard.total_duration_sec = round(media["duration"], 2)
        log(f"demo.mp4 ready: {media['duration']:.1f}s, {media['width']}x{media['height']} "
            f"{media['video_codec']}/{media['audio_codec']}, {media['size_bytes'] / 1e6:.1f} MB")
        job.video_url = f"/api/pipeline/video/{job_id}"
        if (final_path.parent / "cover.jpg").is_file():
            job.cover_url = f"/api/pipeline/cover/{job_id}"
        step("video", "done", detail=f"{media['duration']:.1f}s, {media['width']}x{media['height']}")
        job.progress = 90

        # 6. Launch kit (usually finished while we were recording)
        job.stage = "launch_kit"
        log("Collecting the Devpost pitch, README, and launch posts...")
        launch_kit, kit_note = await kit_task
        log(kit_note)
        job.launch_kit = launch_kit
        step("launchpack", "done", detail=kit_note)

        job.stage, job.progress = "completed", 100
        for pending in background:  # stragglers must not touch the job after it is done
            if not pending.done():
                pending.cancel()
        ledger.seal(job)
        log("Tools used: " + ledger.summary(job))
        log(f"Pipeline completed in {time.monotonic() - started:.0f}s: demo.mp4 and launch kit ready.")
        job.status = "completed"
    except Exception as exc:
        logger.exception("[%s] pipeline failed", job_id)
        for pending in (kit_task, bookend_task, audience_task, *background):
            if pending and not pending.done():
                pending.cancel()
        job.error = str(exc).strip() or type(exc).__name__
        log(f"Pipeline failed: {job.error}")
        step.fail(job.error)
        ledger.seal(job)
        job.status = "failed"

    if job_dir is not None:  # storyboard, launch kit and logs next to the media, for debugging and re-download
        try:
            result_dir = storage_root() / job_id
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "job.json").write_text(job.model_dump_json(indent=2))
        except OSError as exc:
            logger.warning("[%s] could not write job.json: %s", job_id, exc)
        if job.status == "failed":
            shutil.rmtree(job_dir, ignore_errors=True)


def _display_host(app_info: Dict[str, Any]) -> str:
    """'https://www.warestack.com/dashboard' -> 'warestack.com' (the framed window's address bar)."""
    host = (urlparse(app_info.get("url") or "").hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _product_shots(app_info: Dict[str, Any], funnels: List[Dict[str, Any]]) -> List[Tuple[Path, str]]:
    """The product's own screens for the cards and the poster, best first: the landing page, then the strongest
    funnel's last page (its payoff), then the other explored pages. Pages Gemini flagged (a sign-in wall, an error,
    an empty state) and a landing page that shows a sign-in form are left out."""
    found: List[Tuple[Path, str]] = []
    landing = app_info.get("screenshot")
    if (landing and not app_info.get("login_form")
            and (app_info.get("landing_insight") or {}).get("quality") not in vision.FLAGGED):
        found.append((Path(landing), urlparse(app_info.get("url") or "").path or "/"))
    pages = [page for page in app_info.get("pages") or []
             if page.get("screenshot") and (page.get("insight") or {}).get("quality") not in vision.FLAGGED]
    payoff = funnels[0]["steps"][-1] if funnels else None
    pages.sort(key=lambda page: page.get("path") != payoff)  # stable: the payoff page first, the rest in order
    found += [(Path(page["screenshot"]), page.get("path") or "/") for page in pages]
    return [(path, route) for path, route in found if path.is_file()][:3]


def _publish(job_dir: Path, job_id: str, final_path: Path, poster: Optional[Path] = None) -> Path:
    """Moves demo.mp4 (and the poster, as cover.jpg) from the scratch dir into storage; deletes every intermediate."""
    result_dir = storage_root() / job_id
    result_dir.mkdir(parents=True, exist_ok=True)
    target = result_dir / "demo.mp4"
    if final_path.resolve() != target.resolve():
        shutil.copyfile(final_path, target)
    cover = result_dir / "cover.jpg"
    if poster is not None and poster.is_file():
        if poster.resolve() != cover.resolve():
            shutil.copyfile(poster, cover)
    else:
        cover.unlink(missing_ok=True)  # a half-written poster from a cancelled task is not a cover
    for path in job_dir.iterdir():
        if job_dir == result_dir and path.name in ("demo.mp4", "cover.jpg", "job.json"):
            continue
        try:
            shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink()
        except OSError as exc:
            logger.warning("could not remove %s: %s", path, exc)
    if job_dir != result_dir:
        shutil.rmtree(job_dir, ignore_errors=True)
    return target


def _optional_runner(job_id: str, log: Callable[[str], None], step: "_Steps"):
    """`await _optional(awaitable, timeout, step_id, what)`: an optional agent step with a hard timeout; a failure
    costs one log line and an error row, never the job."""
    async def _optional(awaitable, timeout: float, step_id: str, what: str):
        try:
            return await asyncio.wait_for(awaitable, timeout)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - optional steps must never take the job down
            reason = "took too long" if isinstance(exc, asyncio.TimeoutError) else (str(exc).strip().splitlines() or [type(exc).__name__])[0]
            logger.warning("[%s] %s skipped: %s", job_id, what, reason, exc_info=not isinstance(exc, asyncio.TimeoutError))
            log(f"{what} skipped: {reason[:140]}")
            step(step_id, "error", message=reason[:140])
            return None
    return _optional


async def run_analysis_task(job_id: str, req: PipelineRequest, user_id: Optional[str] = None) -> None:
    """Step 1 of 2 for a prospect demo: the repositories (and a deep read of their code), the live app, the prospect's
    site and the page screenshots, then one Gemini analysis: features, the strongest funnels for this prospect with
    demo steps, and the gaps they would notice. No narration, recording or video; the founder confirms a funnel and
    starts the video with its steps as `instructions`."""
    job = JOBS[job_id]
    ledger.bind(job)
    events = EVENTS.get(job_id) or EVENTS.setdefault(job_id, JobEvents())
    log = _job_logger(job, events)
    started = time.monotonic()
    step = _Steps(events, job, started)
    _optional = _optional_runner(job_id, log, step)
    job.status = "running"
    audience_task: Optional[asyncio.Task] = None
    job_dir: Optional[Path] = None
    try:
        job_dir = work_root() / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        try:
            await ensure_public_url(req.app_url)
        except BlockedURL as exc:
            raise RuntimeError(str(exc)) from None
        github_token, storage_state = _user_secrets(user_id, req, log)
        replayer = None
        if user_id and req.use_staged_data:
            fixture_set = get_set_for_origin(user_id, origin_of(req.app_url))
            if fixture_set and fixture_set["enabled_count"]:
                replayer = FixtureReplayer(fixture_set["entries"])
        if req.audience:
            step("audience", "running", detail=audience.running_detail(req.audience))
            audience_task = asyncio.create_task(audience.build_audience(req.audience, req.target_seconds, log, req.app_url))
        else:
            step("audience", "skipped", detail="No prospect in this analysis")
        if req.instructions:
            log("Your focus for the analysis: \"" + " / ".join(req.instructions.splitlines())[:240] + "\"")

        job.stage, job.progress = "ingesting", 5
        step("repos", "running" if req.repo_urls else "skipped",
             detail=f"{len(req.repo_urls)} repositor{'y' if len(req.repo_urls) == 1 else 'ies'}" if req.repo_urls
             else "No repositories in this analysis")
        step("scrape", "running", detail=req.app_url)
        shots_dir = job_dir / "shots"
        repo_info, app_info = await asyncio.gather(
            extract_repos_context(req.repo_urls, github_token),
            probe_app_url(req.app_url, storage_state, replayer, shots_dir=shots_dir if feature_on("VISION") else None,
                          focus=req.instructions,
                          on_page=lambda p: log(f"Explored {p['path']}: '{p['title']}' ({len(p['headings'])} headings)")),
        )
        if app_info.get("error"):
            raise RuntimeError(f"Navigation to {req.app_url} failed: {app_info['error']}")
        for note in repo_info.get("notes", []):
            log(note)
        if req.repo_urls:
            step("repos", "done", detail=f"{len(repo_info.get('repos', []))} of {len(req.repo_urls)} read, "
                                        f"{len(repo_info.get('routes', []))} routes")
        pages = app_info.get("pages") or []
        step("scrape", "done", detail=f"HTTP {app_info.get('status')}: {app_info.get('title') or 'untitled'}; "
                                      f"{len(pages) + 1} page(s) explored")
        landed = app_info.get("url") or req.app_url
        if _path_of(landed) != _path_of(req.app_url):
            log(f"Warning: {req.app_url} redirected to {landed}. "
                + ("Your saved login may have expired: open Settings, choose \"Sign in again\", then re-run."
                   if storage_state else "This page may need a login: save an app login in Settings."))
        job.progress = 25

        sources = repo_info.get("sources") or []
        code: List[Dict[str, str]] = []
        capability_map: Optional[Dict[str, Any]] = None
        if sources:
            step("code", "running", detail=f"{len(sources)} repositor{'y' if len(sources) == 1 else 'ies'}: navigation, "
                                           "UI copy, pages, APIs, models")
            step("capabilities", "running", detail="route declarations against the frontend's API calls")
            code, capability_map = await asyncio.gather(
                _optional(analysis_mod.read_code(sources, github_token, log), analysis_mod.READ_TIMEOUT_SEC,
                          "code", "Code reading"),
                _optional(capabilities_mod.scan(sources, github_token, log), analysis_mod.READ_TIMEOUT_SEC,
                          "capabilities", "Capability map"))
            code = code or []
            if code:
                step("code", "done", detail=f"{len(code)} files, {sum(len(f['text']) for f in code) / 1000:.0f} kB")
            if capability_map is not None:
                step("capabilities", "done", detail=capabilities_mod.summary_line(capability_map).replace("Capability map: ", ""))
        else:
            step("code", "skipped", detail="No readable repository")
            step("capabilities", "skipped", detail="No readable repository")
        brief = None
        if audience_task is not None:
            brief = await audience_task
            job.audience = brief
            step("audience", "done", detail=audience.step_detail(brief))
        job.progress = 45

        shots = [p for p in [app_info.get("screenshot"), *(pg.get("screenshot") for pg in pages)] if p]
        if feature_on("VISION") and get_secret("GEMINI_API_KEY") and shots:
            step("understand", "running", detail=f"{len(shots)} screenshots → Gemini")
            insights = await _optional(vision.describe_pages(app_info, log), 20, "understand", "Understand")
            if insights:
                await vision.thumbnails(insights)
                step("understand", "done", detail=" · ".join(f"{i['path']}: {i['purpose']}" for i in insights)[:200],
                     extra={"pages": insights})
        job.progress = 60

        job.stage = "analysis"
        project = project_name(repo_info, app_info)
        if not get_secret("GEMINI_API_KEY"):
            raise RuntimeError("The analysis needs Gemini: GEMINI_API_KEY is not set")
        step("analysis", "running", detail=f"{len(code)} files and {len(pages) + 1} pages"
                                           + (f" for {brief.company}" if brief is not None else ""))
        log(f"Gemini is analyzing {project}" + (f" for {brief.company}" if brief is not None else "") + "...")
        result = await analysis_mod.analyze(project, repo_info, app_info, code, brief, req.instructions, capability_map)
        job.analysis = result
        step("analysis", "done", detail=analysis_mod.step_detail(result))
        log(analysis_mod.summary_line(result))

        job.stage, job.progress = "completed", 100
        ledger.seal(job)
        log("Tools used: " + ledger.summary(job))
        log(f"Analysis ready in {time.monotonic() - started:.0f}s: pick a funnel to make its video.")
        job.status = "completed"
    except Exception as exc:
        logger.exception("[%s] analysis failed", job_id)
        if audience_task is not None and not audience_task.done():
            audience_task.cancel()
        job.error = str(exc).strip() or type(exc).__name__
        log(f"Analysis failed: {job.error}")
        step.fail(job.error)
        ledger.seal(job)
        job.status = "failed"
    if job_dir is not None:
        try:
            result_dir = storage_root() / job_id
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "job.json").write_text(job.model_dump_json(indent=2))
        except OSError as exc:
            logger.warning("[%s] could not write job.json: %s", job_id, exc)
        if job_dir != storage_root() / job_id:
            shutil.rmtree(job_dir, ignore_errors=True)  # Cloud Run: scratch on local disk, job.json on the bucket
        else:
            shutil.rmtree(job_dir / "shots", ignore_errors=True)  # local: the scratch dir is the result dir


def create_analysis_job(req: PipelineRequest, user_id: Optional[str] = None) -> JobStatus:
    """A prospect analysis job (kind "analysis"): the same events, steps and job endpoints as a video job."""
    return create_pipeline_job(req, user_id=user_id, kind="analysis")


def create_pipeline_job(req: PipelineRequest, user_id: Optional[str] = None, kind: str = "video") -> JobStatus:
    job_id = uuid.uuid4().hex[:12]
    options: Dict[str, Any] = {"goals": req.goals, "requestedLanguage": req.language}
    if req.language != "auto":
        options["language"] = req.language
    if req.instructions:
        options["instructions"] = req.instructions
    job = JobStatus(job_id=job_id, status="queued", progress=0, stage="queued", logs=["Job queued"],
                    app_url=req.app_url, created_at=time.time(), options=options, kind=kind,
                    audience=audience.pending_brief(req.audience) if req.audience else None)
    JOBS[job_id] = job
    events = EVENTS[job_id] = JobEvents()
    events.emit("job", {"id": job_id, "mode": run_mode()})
    if user_id:
        db.execute("INSERT INTO jobs (id, user_id, created_at) VALUES (?, ?, ?)", (job_id, user_id, job.created_at))
    runner = run_analysis_task if kind == "analysis" else run_pipeline_task
    for coro in (runner(job_id, req, user_id), _watch_job(job, events)):
        task = asyncio.create_task(coro)
        _TASKS.add(task)
        task.add_done_callback(_TASKS.discard)
    return job


def _sse(item: Dict[str, Any]) -> str:
    return f"id: {item['id']}\nevent: {item['event']}\ndata: {json.dumps(item['data'], separators=(',', ':'))}\n\n"


async def stream_job_events(job_id: str, last_event_id: int = 0) -> AsyncGenerator[str, None]:
    """Server-Sent Events: the job's typed event log from `last_event_id` on, keep-alive comments every 15 s.

    Events: `job` first; `step`, `log` and `status` (a full JobStatus snapshot) while it runs; `done` (the full job)
    or `error` (`{message}`) last. A job that is no longer in memory gets a single `done` or `error` with id 1.
    """
    yield "retry: 3000\n\n"
    events = EVENTS.get(job_id)
    if events is None:
        job = JOBS.get(job_id)
        if job is None:
            yield _sse({"id": 1, "event": "error", "data": {"message": "Job not found"}})
        elif job.status == "completed":
            yield _sse({"id": 1, "event": "done", "data": job.model_dump(mode="json")})
        else:
            yield _sse({"id": 1, "event": "error",
                        "data": {"message": job.error or "The server restarted while this job was running"}})
        return
    index = max(0, min(last_event_id, len(events.items)))  # ids are 1..n, so the id is also the count to skip
    while True:
        while index < len(events.items):
            yield _sse(events.items[index])
            index += 1
        if events.closed:
            return
        before = len(events.items)
        await events.wait(15)
        if len(events.items) == before and not events.closed:
            yield ": keep-alive\n\n"
