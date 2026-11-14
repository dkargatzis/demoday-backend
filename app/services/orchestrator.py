import os
import uuid
import asyncio
from typing import Dict, AsyncGenerator
from app.models.schemas import PipelineRequest, JobStatus
from app.services.ingest import extract_github_context, probe_app_url
from app.services.planner import generate_storyboard, generate_launch_kit
from app.services.audio import synthesize_scene_audio
from app.services.recorder import record_scene
from app.services.stitcher import mux_scene_clip, concat_scenes

JOBS: Dict[str, JobStatus] = {}

async def run_pipeline_task(job_id: str, req: PipelineRequest):
    job = JOBS[job_id]
    job.status = "running"
    storage_dir = os.path.abspath(f"storage/{job_id}")
    os.makedirs(storage_dir, exist_ok=True)

    try:
        # Step 1: Ingestion
        job.stage = "ingesting"
        job.progress = 10
        job.logs.append(f"Analyzing GitHub repository: {req.repo_url}")
        job.logs.append(f"Probing live application: {req.app_url}")
        
        repo_info, app_info = await asyncio.gather(
            extract_github_context(req.repo_url),
            probe_app_url(req.app_url)
        )
        job.progress = 25

        # Step 2: Planning with Gemini
        job.stage = "planning"
        job.logs.append("Synthesizing product context with Gemini 2.0...")
        storyboard = await generate_storyboard(repo_info, app_info, req.target_seconds)
        job.storyboard = storyboard
        job.progress = 40
        job.logs.append(f"Storyboard generated with {len(storyboard.scenes)} scenes.")

        # Step 3: Parallel Voice & Browser Capture
        job.stage = "recording"
        scene_clips = []
        for idx, scene in enumerate(storyboard.scenes):
            scene_id = scene.id
            audio_path = os.path.join(storage_dir, f"{scene_id}.wav")
            raw_video_path = os.path.join(storage_dir, f"{scene_id}_raw.webm")
            muxed_clip_path = os.path.join(storage_dir, f"{scene_id}_muxed.mp4")

            job.logs.append(f"Scene {idx+1}/{len(storyboard.scenes)}: Synthesizing narration with fal.ai...")
            duration = await synthesize_scene_audio(scene.narration, audio_path, req.voice_style)
            scene.duration_sec = duration

            job.logs.append(f"Scene {idx+1}/{len(storyboard.scenes)}: Recording headless browser interactions ({duration:.1f}s)...")
            await record_scene(scene, req.app_url, raw_video_path, duration)

            job.logs.append(f"Scene {idx+1}/{len(storyboard.scenes)}: Muxing video and audio tracks...")
            await mux_scene_clip(raw_video_path, audio_path, muxed_clip_path)
            scene_clips.append(muxed_clip_path)
            
            job.progress = 40 + int(35 * ((idx + 1) / len(storyboard.scenes)))

        # Step 4: Video Stitching & Launch Kit Generation
        job.stage = "assembling"
        job.logs.append("Stitching final MP4 demo video with FFmpeg...")
        final_video_path = os.path.join(storage_dir, "demo.mp4")
        await concat_scenes(scene_clips, final_video_path)
        job.video_url = f"/api/pipeline/video/{job_id}"
        job.progress = 85

        job.stage = "launch_kit"
        job.logs.append("Generating Devpost pitch, README, and social launch copy...")
        launch_kit = await generate_launch_kit(storyboard, repo_info)
        job.launch_kit = launch_kit

        # Completed
        job.stage = "completed"
        job.status = "completed"
        job.progress = 100
        job.logs.append("DemoDay launch kit pipeline completed successfully!")

    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.logs.append(f"Pipeline error: {str(e)}")

def create_pipeline_job(req: PipelineRequest) -> JobStatus:
    job_id = str(uuid.uuid4())[:8]
    job = JobStatus(job_id=job_id, status="queued", progress=0, stage="queued", logs=["Job queued"])
    JOBS[job_id] = job
    asyncio.create_task(run_pipeline_task(job_id, req))
    return job

async def stream_job_events(job_id: str) -> AsyncGenerator[str, None]:
    last_progress = -1
    last_logs_len = 0
    while True:
        if job_id not in JOBS:
            yield 'data: {"error": "Job not found"}\n\n'
            break
        job = JOBS[job_id]
        if job.progress != last_progress or len(job.logs) != last_logs_len:
            last_progress = job.progress
            last_logs_len = len(job.logs)
            data = job.model_dump_json()
            yield f"data: {data}\n\n"
        
        if job.status in ["completed", "failed"]:
            break
        await asyncio.sleep(0.5)
