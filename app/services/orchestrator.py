"""Pipeline orchestration and live job state (in memory) streamed to the dashboard over SSE."""
import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Callable, Dict, Optional, Set, Tuple

from app import db, vault
from app.config import storage_root
from app.models.schemas import JobStatus, PipelineRequest
from app.services.audio import synthesize_scene_audio
from app.services.capture import origin_of
from app.services.ingest import extract_repos_context, probe_app_url
from app.services.netguard import BlockedURL, ensure_public_url
from app.services.planner import generate_launch_kit, generate_storyboard
from app.services.recorder import record_session
from app.services.stitcher import concat_scenes, describe_media, frame_aligned, mux_scene_clip

logger = logging.getLogger("demoday.pipeline")

JOBS: Dict[str, JobStatus] = {}
_TASKS: Set[asyncio.Task] = set()  # strong refs: asyncio only keeps weak refs to running tasks
# Screen recording is timing-sensitive and encoding is CPU-heavy: render one job at a time.
_RENDER_SLOT = asyncio.Semaphore(1)
SCENE_TAIL_SEC = 0.4  # breathing room after each narration line
TERMINAL = ("completed", "failed")


def _job_logger(job: JobStatus) -> Callable[[str], None]:
    def log(message: str) -> None:
        job.logs.append(message)
        logger.info("[%s] %s", job.job_id, message)
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
    if state is not None:
        log(f"Recording signed in with your saved login for {origin}")
    elif not req.use_app_login:
        log("Recording signed out: saved login turned off for this run")
    elif vault.list_kind(user_id, "app_login"):
        log(f"No saved login for {origin}; recording as a signed-out visitor")
    return (token.decode() if token else None), (json.loads(state) if state else None)


async def run_pipeline_task(job_id: str, req: PipelineRequest, user_id: Optional[str] = None) -> None:
    job = JOBS[job_id]
    log = _job_logger(job)
    job.status = "running"
    kit_task: Optional[asyncio.Task] = None
    job_dir: Optional[Path] = None
    started = time.monotonic()

    try:  # everything, including setup, must land in a terminal state or the job hangs in "running"
        job_dir = storage_root() / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        try:
            await ensure_public_url(req.app_url)
        except BlockedURL as exc:
            raise RuntimeError(str(exc)) from None
        github_token, storage_state = _user_secrets(user_id, req, log)

        # 1. Ingest repositories + live app
        job.stage, job.progress = "ingesting", 5
        for url in req.repo_urls:
            log(f"Analyzing GitHub repository: {url}")
        log(f"Rendering live app in headless Chromium: {req.app_url}")
        repo_info, app_info = await asyncio.gather(
            extract_repos_context(req.repo_urls, github_token), probe_app_url(req.app_url, storage_state)
        )
        if app_info.get("error"):
            raise RuntimeError(f"Navigation to {req.app_url} failed: {app_info['error']}")
        for note in repo_info.get("notes", []):
            log(note)
        for repo in repo_info.get("repos", []):
            log(f"Repo {repo['full_name']}{' (private)' if repo['private'] else ''} added to the context")
        log(f"Repository context: {len(repo_info.get('readme', ''))} chars of README, "
            f"{len(repo_info.get('dependencies', []))} dependencies, {len(repo_info.get('routes', []))} page routes")
        status = app_info.get("status")
        log(f"Live app \"{app_info.get('title') or 'untitled'}\" (HTTP {status}): {len(app_info['links'])} links, "
            f"{len(app_info['buttons'])} buttons, {len(app_info['inputs'])} inputs")
        if status and status >= 400:
            log(f"Warning: the app answered HTTP {status}; the recording may show an error page")
        if app_info.get("login_form"):
            log("Warning: the app still shows a sign-in form with your saved login; it may have expired. "
                "Capture the login again in Settings." if storage_state else
                "Warning: the app shows a sign-in form. Save an app login in Settings to record signed-in pages.")
        job.progress = 20

        # 2. Plan the storyboard; write the launch kit in the background meanwhile
        job.stage = "planning"
        log("Planning the storyboard...")
        storyboard, note = await generate_storyboard(
            repo_info, app_info, req.target_seconds, req.voice_style, ", ".join(req.repo_urls)
        )
        log(note)
        job.storyboard = storyboard
        scenes = storyboard.scenes
        log(f"Storyboard: {len(scenes)} scenes for \"{storyboard.project_name}\"")
        kit_task = asyncio.create_task(
            generate_launch_kit(storyboard, repo_info, app_info, req.repo_url, req.app_url)
        )
        job.progress = 35

        # 3. Narration (all scenes in parallel); durations measured with ffprobe
        job.stage = "recording"
        log(f"Synthesizing narration for {len(scenes)} scenes...")
        audio_paths = [job_dir / f"scene_{i + 1:02d}.wav" for i in range(len(scenes))]
        narrations = await asyncio.gather(*(
            synthesize_scene_audio(scene.narration, path, req.voice_style)
            for scene, path in zip(scenes, audio_paths)
        ))
        for i, (scene, (duration, engine)) in enumerate(zip(scenes, narrations)):
            scene.duration_sec = round(duration, 3)
            log(f"Scene {i + 1}/{len(scenes)}: {duration:.2f}s of narration via {engine}")
        job.progress = 45

        # 4. Record + encode
        if _RENDER_SLOT.locked():
            log("Waiting for another job to finish rendering...")
        async with _RENDER_SLOT:
            clip_durations = [frame_aligned(duration + SCENE_TAIL_SEC) for duration, _ in narrations]
            log(f"Recording one continuous 1920x1080 browser session ({sum(clip_durations):.1f}s of footage)...")

            def on_scene(idx: int) -> None:
                job.progress = 45 + int(20 * idx / len(scenes))
                log(f"Scene {idx + 1}/{len(scenes)}: recording \"{scenes[idx].title}\" ({clip_durations[idx]:.1f}s)")

            session_path = job_dir / "session.webm"
            segments = await record_session(scenes, req.app_url, clip_durations, session_path, log, on_scene,
                                            storage_state=storage_state)
            job.progress = 65

            clips = []
            for i, segment in enumerate(segments):
                log(f"Scene {i + 1}/{len(scenes)}: muxing H.264/AAC clip ({segment.duration:.2f}s)")
                clip = job_dir / f"scene_{i + 1:02d}.mp4"
                await mux_scene_clip(session_path, audio_paths[i], clip, segment.start, segment.duration)
                clips.append(clip)
                # Clip length (narration + tail), so cumulative durations are exact scene offsets in demo.mp4.
                scenes[i].duration_sec = round(segment.duration, 3)
                job.progress = 65 + int(17 * (i + 1) / len(segments))

            job.stage = "assembling"
            log("Stitching scenes into demo.mp4...")
            final_path = job_dir / "demo.mp4"
            await concat_scenes(clips, final_path)

        media = await describe_media(final_path)
        storyboard.total_duration_sec = round(media["duration"], 2)
        log(f"demo.mp4 ready: {media['duration']:.1f}s, {media['width']}x{media['height']} "
            f"{media['video_codec']}/{media['audio_codec']}, {media['size_bytes'] / 1e6:.1f} MB")
        job.video_url = f"/api/pipeline/video/{job_id}"
        job.progress = 90

        # 5. Launch kit (usually finished while we were recording)
        job.stage = "launch_kit"
        log("Collecting the Devpost pitch, README, and launch posts...")
        launch_kit, kit_note = await kit_task
        log(kit_note)
        job.launch_kit = launch_kit

        job.stage, job.progress = "completed", 100
        log(f"Pipeline completed in {time.monotonic() - started:.0f}s: demo.mp4 and launch kit ready.")
        job.status = "completed"
    except Exception as exc:
        logger.exception("[%s] pipeline failed", job_id)
        if kit_task and not kit_task.done():
            kit_task.cancel()
        job.error = str(exc).strip() or type(exc).__name__
        log(f"Pipeline failed: {job.error}")
        job.status = "failed"

    if job_dir is not None:  # storyboard, launch kit and logs next to the media, for debugging and re-download
        try:
            (job_dir / "job.json").write_text(job.model_dump_json(indent=2))
        except OSError as exc:
            logger.warning("[%s] could not write job.json: %s", job_id, exc)


def create_pipeline_job(req: PipelineRequest, user_id: Optional[str] = None) -> JobStatus:
    job_id = uuid.uuid4().hex[:12]
    job = JobStatus(job_id=job_id, status="queued", progress=0, stage="queued", logs=["Job queued"])
    JOBS[job_id] = job
    if user_id:
        db.execute("INSERT INTO jobs (id, user_id, created_at) VALUES (?, ?, ?)", (job_id, user_id, time.time()))
    task = asyncio.create_task(run_pipeline_task(job_id, req, user_id))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return job


async def stream_job_events(job_id: str) -> AsyncGenerator[str, None]:
    """Server-Sent Events: a full JobStatus snapshot whenever anything changes, plus keep-alive comments."""
    yield "retry: 3000\n\n"
    last_payload, last_sent = None, time.monotonic()
    while True:
        job = JOBS.get(job_id)
        if job is None:
            yield 'event: error\ndata: {"error": "Job not found"}\n\n'
            return
        payload = job.model_dump_json()
        now = time.monotonic()
        if payload != last_payload:
            last_payload, last_sent = payload, now
            yield f"data: {payload}\n\n"
        elif now - last_sent >= 15:
            last_sent = now
            yield ": keep-alive\n\n"
        if job.status in TERMINAL:
            return
        await asyncio.sleep(0.4)
