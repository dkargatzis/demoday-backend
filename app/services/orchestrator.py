"""Pipeline orchestration and live job state (in memory) streamed to the dashboard over SSE."""
import asyncio
import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Tuple

from app import db, vault
from app.config import capabilities, get_secret, storage_root, work_root
from app.models.schemas import JobStatus, PipelineRequest
from app.services.audio import synthesize_scene_audio, warm_up_fal
from app.services.bookends import FAL_VIDEO_MODEL, prepare_bookends
from app.services.capture import describe_login, origin_of, unpack_login
from app.services.fixtures import FixtureReplayer, get_set_for_origin
from app.services.ingest import extract_repos_context, probe_app_url
from app.services.netguard import BlockedURL, ensure_public_url
from app.services.planner import generate_launch_kit, generate_storyboard
from app.services.recorder import record_session
from app.services.stitcher import concat_scenes, describe_media, frame_aligned, make_bookend, mux_scene_clip

logger = logging.getLogger("demoday.pipeline")

JOBS: Dict[str, JobStatus] = {}
_TASKS: Set[asyncio.Task] = set()  # strong refs: asyncio only keeps weak refs to running tasks
# Screen recording is timing-sensitive and encoding is CPU-heavy: render one job at a time.
_RENDER_SLOT = asyncio.Semaphore(1)
SCENE_TAIL_SEC = 0.4  # breathing room after each narration line
BOOKEND_VOICE_DELAY_SEC = 0.4  # the intro/outro line starts after the title fades in
TERMINAL = ("completed", "failed")
STEP_LABELS = {
    "repos": "Reading the repositories", "scrape": "Reading the live app", "storyboard": "Planning the storyboard",
    "launchpack": "Writing the launch kit", "ai_clips": "AI intro and outro", "narration": "Synthesizing narration",
    "record": "Recording the browser session", "video": "Assembling demo.mp4",
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
    """Emits `step` events and remembers which steps are still running, so a failure can mark them."""

    def __init__(self, events: JobEvents) -> None:
        self.events, self.running = events, []  # type: JobEvents, List[str]

    def __call__(self, step_id: str, status: str, detail: Optional[str] = None, message: Optional[str] = None) -> None:
        if status == "running":
            self.running.append(step_id)
        else:
            self.running = [s for s in self.running if s != step_id]
        data: Dict[str, Any] = {"id": step_id, "status": status, "label": STEP_LABELS.get(step_id, step_id)}
        if detail:
            data["detail"] = detail
        if message:
            data["message"] = message
        self.events.emit("step", data)

    def fail(self, message: str) -> None:
        for step_id in list(self.running):
            self(step_id, "error", message=message)


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
    events = EVENTS.get(job_id) or EVENTS.setdefault(job_id, JobEvents())
    log = _job_logger(job, events)
    step = _Steps(events)
    job.status = "running"
    kit_task: Optional[asyncio.Task] = None
    bookend_task: Optional[asyncio.Task] = None
    job_dir: Optional[Path] = None
    started = time.monotonic()

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
        replayer = None
        if user_id and req.use_staged_data:
            fixture_set = get_set_for_origin(user_id, origin_of(req.app_url))
            if fixture_set and fixture_set["enabled_count"]:
                replayer = FixtureReplayer(fixture_set["entries"])
                log(f"Staged sample data: {fixture_set['enabled_count']} endpoint(s) for {fixture_set['origin']}"
                    + (" (Gemini-polished)" if fixture_set["polished"] else "") + "; the video is labelled 'Sample data'")

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
        repo_info, app_info = await asyncio.gather(
            extract_repos_context(req.repo_urls, github_token), probe_app_url(req.app_url, storage_state, replayer)
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
        step("scrape", "done", detail=f"HTTP {status}: {app_info.get('title') or 'untitled'}")
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
        job.progress = 20

        # 2. Plan the storyboard; write the launch kit in the background meanwhile
        job.stage = "planning"
        step("storyboard", "running")
        log("Planning the storyboard...")
        storyboard, note = await generate_storyboard(
            repo_info, app_info, req.target_seconds, req.voice_style, ", ".join(req.repo_urls), ai_clips=req.ai_clips
        )
        log(note)
        job.storyboard = storyboard
        scenes = storyboard.scenes
        log(f"Storyboard: {len(scenes)} scenes for \"{storyboard.project_name}\"")
        step("storyboard", "done", detail=f"{len(scenes)} scenes; {note}")
        step("launchpack", "running")
        kit_task = asyncio.create_task(
            generate_launch_kit(storyboard, repo_info, app_info, ", ".join(req.repo_urls), req.app_url)
        )
        if not req.ai_clips:
            step("ai_clips", "skipped", detail="Not requested in this run")
        else:  # generated by fal.ai while the browser records
            step("ai_clips", "running")
            log(f"Generating the AI intro and outro with fal.ai ({FAL_VIDEO_MODEL})..." if get_secret("FAL_KEY")
                else "FAL_KEY not set: the intro and outro will be typeset title cards")
            bookend_task = asyncio.create_task(prepare_bookends(
                job_dir, storyboard.project_name, storyboard.tagline, req.app_url, req.repo_urls, req.voice_style,
                {"intro": storyboard.intro_visual, "outro": storyboard.outro_visual}, log))
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
        step("narration", "done", detail=f"{sum(d for d, _ in narrations):.1f}s across {len(scenes)} scenes")
        job.progress = 45

        # 4. Record + encode
        if _RENDER_SLOT.locked():
            log("Waiting for another job to finish rendering...")
        async with _RENDER_SLOT:
            clip_durations = [frame_aligned(duration + SCENE_TAIL_SEC) for duration, _ in narrations]
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

        # 5. Intro and outro (outside the render slot while fal.ai may still be generating)
        job.stage = "assembling"
        step("video", "running")
        bookends = None
        if bookend_task is not None:
            if not bookend_task.done():
                log("Waiting for the AI intro and outro from fal.ai...")
            try:
                bookends = await bookend_task
                step("ai_clips", "done")
            except Exception as exc:  # the demo never depends on its bookends
                logger.exception("[%s] bookends failed", job_id)
                log(f"Warning: skipping the intro and outro ({str(exc).strip()[:140] or type(exc).__name__})")
                step("ai_clips", "error", message=str(exc).strip()[:140] or type(exc).__name__)
        async with _RENDER_SLOT:
            if bookends:
                for kind, fade in (("intro", "in"), ("outro", "out")):
                    bookend = bookends[kind]
                    duration = frame_aligned(max(3.0, bookend.narration_sec + BOOKEND_VOICE_DELAY_SEC + 0.6))
                    path = job_dir / f"{kind}.mp4"
                    await make_bookend(path, bookend.narration, bookend.overlay, duration, fade,
                                       video_path=bookend.video, background_png=bookend.background,
                                       voice_delay_sec=BOOKEND_VOICE_DELAY_SEC)
                    if kind == "intro":
                        clips.insert(0, path)
                        storyboard.intro_sec = duration
                    else:
                        clips.append(path)
                        storyboard.outro_sec = duration
                    log(f"{kind.capitalize()} clip ready ({duration:.1f}s)")
            job.progress = 86
            log("Stitching scenes into demo.mp4...")
            final_path = job_dir / "demo.mp4"
            await concat_scenes(clips, final_path)

        media = await describe_media(final_path)
        final_path = _publish(job_dir, job_id, final_path)  # demo.mp4 into storage; scratch removed
        storyboard.total_duration_sec = round(media["duration"], 2)
        log(f"demo.mp4 ready: {media['duration']:.1f}s, {media['width']}x{media['height']} "
            f"{media['video_codec']}/{media['audio_codec']}, {media['size_bytes'] / 1e6:.1f} MB")
        job.video_url = f"/api/pipeline/video/{job_id}"
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
        log(f"Pipeline completed in {time.monotonic() - started:.0f}s: demo.mp4 and launch kit ready.")
        job.status = "completed"
    except Exception as exc:
        logger.exception("[%s] pipeline failed", job_id)
        for pending in (kit_task, bookend_task):
            if pending and not pending.done():
                pending.cancel()
        job.error = str(exc).strip() or type(exc).__name__
        log(f"Pipeline failed: {job.error}")
        step.fail(job.error)
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


def _publish(job_dir: Path, job_id: str, final_path: Path) -> Path:
    """Moves demo.mp4 from the scratch dir into storage and deletes every intermediate."""
    result_dir = storage_root() / job_id
    result_dir.mkdir(parents=True, exist_ok=True)
    target = result_dir / "demo.mp4"
    if final_path.resolve() != target.resolve():
        shutil.copyfile(final_path, target)
    for path in job_dir.iterdir():
        if job_dir == result_dir and path.name in ("demo.mp4", "job.json"):
            continue
        try:
            shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink()
        except OSError as exc:
            logger.warning("could not remove %s: %s", path, exc)
    if job_dir != result_dir:
        shutil.rmtree(job_dir, ignore_errors=True)
    return target


def create_pipeline_job(req: PipelineRequest, user_id: Optional[str] = None) -> JobStatus:
    job_id = uuid.uuid4().hex[:12]
    options: Dict[str, Any] = {"goals": req.goals, "requestedLanguage": req.language}
    if req.language != "auto":
        options["language"] = req.language
    job = JobStatus(job_id=job_id, status="queued", progress=0, stage="queued", logs=["Job queued"],
                    app_url=req.app_url, created_at=time.time(), options=options)
    JOBS[job_id] = job
    events = EVENTS[job_id] = JobEvents()
    events.emit("job", {"id": job_id, "mode": run_mode()})
    if user_id:
        db.execute("INSERT INTO jobs (id, user_id, created_at) VALUES (?, ?, ?)", (job_id, user_id, job.created_at))
    for coro in (run_pipeline_task(job_id, req, user_id), _watch_job(job, events)):
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
