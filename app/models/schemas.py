from typing import Any, List, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

VoiceStyle = Literal["energetic_tech", "calm_expert", "fast_founder"]
MAX_REPOS = 5


def _http_url(value: str) -> str:
    # URLs are opened by a real browser, so only http(s) is allowed (no file://, chrome://, ...).
    value = value.strip()
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"{value!r} must be an http(s) URL")
    return value


class PipelineRequest(BaseModel):
    repo_urls: List[str] = Field(default_factory=list, description=f"GitHub repositories, up to {MAX_REPOS}")
    repo_url: Optional[str] = Field(default=None, description="Single repository (older clients); merged into repo_urls")
    app_url: str = Field(..., description="Deployed live web application URL")
    voice_style: VoiceStyle = Field(default="energetic_tech", description="Narration voice and tone")
    target_seconds: int = Field(default=60, ge=15, le=120, description="Target total runtime in seconds")
    use_app_login: bool = Field(default=True, description="Record signed in when a saved app login matches the app")
    ai_clips: bool = Field(default=False, description="Add an AI-generated intro and outro (fal.ai video credits)")
    use_staged_data: bool = Field(default=True, description="Replay staged sample data when a set exists for the app")

    @field_validator("app_url")
    @classmethod
    def _require_http_url(cls, value: str) -> str:
        return _http_url(value)

    @model_validator(mode="after")
    def _collect_repos(self) -> "PipelineRequest":
        urls: List[str] = []
        for raw in [*self.repo_urls, self.repo_url]:
            if raw and raw.strip():
                url = _http_url(raw)
                if url not in urls:
                    urls.append(url)
        if len(urls) > MAX_REPOS:
            raise ValueError(f"at most {MAX_REPOS} repositories per demo")
        self.repo_urls = urls
        self.repo_url = urls[0] if urls else None
        return self


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
    intro_sec: float = 0.0  # length of the intro before scene 1 in demo.mp4 (0 without AI clips)
    outro_sec: float = 0.0
    intro_visual: Optional[str] = None  # Gemini's shot descriptions for the AI clips
    outro_visual: Optional[str] = None


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


# --------------------------------------------------------------------------- API response models


class Ok(BaseModel):
    ok: bool = True


class ErrorDetail(BaseModel):
    detail: str = Field(..., examples=["Sign in to continue"])


class HealthStatus(BaseModel):
    status: Literal["ok"]
    service: str
    ffmpeg: bool = Field(..., description="ffmpeg and ffprobe found on PATH")
    gemini: bool = Field(..., description="GEMINI_API_KEY configured (otherwise template storyboard/launch kit)")
    fal: bool = Field(..., description="FAL_KEY configured (otherwise Google Cloud TTS, then local TTS)")
    gemini_model: str
    vault: bool = Field(..., description="DEMODAY_SECRET_KEY configured, so credentials can be saved")
    signup_code: bool = Field(..., description="Creating an account requires the signup code")


class User(BaseModel):
    id: str
    email: str


class SavedSecret(BaseModel):
    id: str
    kind: str
    name: str
    created_at: float = Field(..., description="Unix timestamp")


class GitHubConnection(SavedSecret):
    login: str = Field(..., description="GitHub login the token belongs to")
    token_type: Literal["classic", "fine-grained"]
    scopes: str = Field("", description="Classic token scopes; empty for fine-grained tokens")


class AppLogin(SavedSecret):
    origin: str = Field(..., description="scheme://host[:port] the saved session belongs to")
    cookie_count: int
    last_url: Optional[str] = None


class SettingsOut(BaseModel):
    vault: bool
    github: Optional[GitHubConnection] = None
    app_logins: List[AppLogin] = []


class CaptureStart(BaseModel):
    capture_id: str
    origin: str
    viewport: dict = Field(..., examples=[{"width": 1280, "height": 800}])


class ShareLink(BaseModel):
    url: str
    expires_at: Optional[float] = Field(None, description="Unix timestamp, or null when the link does not expire")


class FixtureEntry(BaseModel):
    url: str
    method: str = "GET"
    status: int = 200
    body: Any = Field(..., description="The JSON response to replay")
    post_data: Optional[str] = None
    enabled: bool = True
    source: str = Field("captured", description="captured, edited, or gemini (<model>)")


class FixtureSetSummary(BaseModel):
    id: str
    origin: str
    polished: bool
    brief: str = ""
    updated_at: float
    entry_count: int
    enabled_count: int


class FixtureSet(FixtureSetSummary):
    entries: List[FixtureEntry] = []
    note: Optional[str] = Field(None, description="Set by /polish: what Gemini changed")
