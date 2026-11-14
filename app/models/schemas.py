from typing import List, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

VoiceStyle = Literal["energetic_tech", "calm_expert", "fast_founder"]


class PipelineRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub repository URL")
    app_url: str = Field(..., description="Deployed live web application URL")
    voice_style: VoiceStyle = Field(default="energetic_tech", description="Narration voice and tone")
    target_seconds: int = Field(default=60, ge=15, le=120, description="Target total runtime in seconds")

    @field_validator("repo_url", "app_url")
    @classmethod
    def _require_http_url(cls, value: str) -> str:
        # The app URL is opened in a real browser, so only http(s) is allowed (no file://, chrome://, ...).
        value = value.strip()
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("must be an http(s) URL")
        return value


class Action(BaseModel):
    type: Literal["navigate", "click", "type", "scroll", "wait"]
    selector: Optional[str] = None
    value: Optional[str] = None
    dwell_ms: int = Field(default=1500, ge=0, le=15000)


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
