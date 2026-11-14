from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class PipelineRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub repository URL")
    app_url: str = Field(..., description="Deployed live web application URL")
    voice_style: str = Field(default="energetic_tech", description="Voice tone: energetic_tech, calm_expert, fast_founder")
    target_seconds: int = Field(default=60, description="Target total runtime in seconds")

class Action(BaseModel):
    type: Literal["navigate", "click", "type", "scroll", "wait"]
    selector: Optional[str] = None
    value: Optional[str] = None
    dwell_ms: int = 1500

class Scene(BaseModel):
    id: str
    title: str
    route: str
    narration: str
    actions: List[Action] = []
    duration_sec: float = 0.0
    video_path: Optional[str] = None
    audio_path: Optional[str] = None

class Storyboard(BaseModel):
    project_name: str
    tagline: str
    scenes: List[Scene]
    total_duration_sec: float = 0.0

class LaunchKit(BaseModel):
    readme_markdown: str
    devpost_pitch: str
    x_thread: List[str]
    linkedin_post: str

class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: int = 0
    stage: str = "initialized"
    logs: List[str] = []
    storyboard: Optional[Storyboard] = None
    launch_kit: Optional[LaunchKit] = None
    video_url: Optional[str] = None
    error: Optional[str] = None
