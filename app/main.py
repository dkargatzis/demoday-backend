import logging
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from app.config import GEMINI_MODEL, capabilities, storage_root  # loads .env before anything reads it
from app.models.schemas import JobStatus, PipelineRequest
from app.services.orchestrator import JOBS, create_pipeline_job, stream_job_events

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per outbound request is noise
logger = logging.getLogger("demoday")

JOB_ID_RE = re.compile(r"^[a-f0-9]{8,32}$")
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


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
    yield


app = FastAPI(
    title="DemoDay Launch Kit Engine",
    description="Autonomous browser recording, fal.ai voiceover, and pitch generation",
    version="1.1.0",
    lifespan=lifespan,
)

origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
           if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "demoday-backend", **capabilities()}


@app.post("/api/pipeline/start", response_model=JobStatus)
async def start_pipeline(req: PipelineRequest):
    return create_pipeline_job(req)


@app.get("/api/pipeline/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]


@app.get("/api/pipeline/events/{job_id}")
@app.get("/api/pipeline/stream/{job_id}", include_in_schema=False)  # backwards-compatible alias
async def stream_events(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return StreamingResponse(stream_job_events(job_id), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/api/pipeline/video/{job_id}")
async def get_video(job_id: str, download: bool = False):
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="Video file not found")
    video_path = storage_root() / job_id / "demo.mp4"
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"demo-{job_id}.mp4",
        content_disposition_type="attachment" if download else "inline",
    )
