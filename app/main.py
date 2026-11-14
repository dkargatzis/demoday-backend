import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from dotenv import load_dotenv

load_dotenv()

from app.models.schemas import PipelineRequest, JobStatus
from app.services.orchestrator import create_pipeline_job, JOBS, stream_job_events

app = FastAPI(
    title="DemoDay Launch Kit Engine",
    description="Autonomous browser recording, fal.ai voiceover, and pitch generation",
    version="1.0.0"
)

# CORS support for Next.js frontend
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "demoday-backend"}

@app.post("/api/pipeline/start", response_model=JobStatus)
async def start_pipeline(req: PipelineRequest):
    return create_pipeline_job(req)

@app.get("/api/pipeline/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]

@app.get("/api/pipeline/stream/{job_id}")
async def stream_events(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return StreamingResponse(stream_job_events(job_id), media_type="text/event-stream")

@app.get("/api/pipeline/video/{job_id}")
async def get_video(job_id: str):
    video_path = os.path.abspath(f"storage/{job_id}/demo.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(video_path, media_type="video/mp4", filename=f"demo-{job_id}.mp4")
