import re
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.audience import Audience, AudienceBrief, AudienceGoal, OutreachEmail  # noqa: F401 (re-exported)

GOALS = ("social", "community", "investors", "customers")  # Launch Kit v2 contract (backend issue #3)
LANGUAGE_RE = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z]{2,4})?$")

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
    goals: List[str] = Field(default_factory=lambda: list(GOALS),
                             description="What the launch is for: social, community, investors, customers (default all four). "
                                         "Echoed as `options.goals` on the job; the demo recording runs regardless")
    language: str = Field(default="auto", description='Run language: "auto" or a code such as "en", "el", "de". '
                                                       "Echoed as `options.requestedLanguage`")
    audience: Optional[Audience] = Field(default=None,
                                         description="Personalize this demo for one customer; omitted = the generic demo")
    instructions: Optional[str] = Field(default=None, max_length=1000,
                                        description="What the demo must show, in your own words: pages to open and actions "
                                                    "to take, in order (\"Open Rules, enable 'Require a linked ticket', then "
                                                    "open the pull request it blocked\"). The tour follows it, and the funnel "
                                                    "that matches it ranks first. Echoed as `options.instructions`")

    @field_validator("instructions")
    @classmethod
    def _plain_instructions(cls, value: Optional[str]) -> Optional[str]:
        text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]+", " ", value or "")  # control characters out, line breaks kept
        text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())
        return re.sub(r"\n{3,}", "\n\n", text).strip() or None

    @field_validator("goals")
    @classmethod
    def _known_goals(cls, value: List[str]) -> List[str]:
        goals: List[str] = []
        for raw in value:
            goal = str(raw).strip().lower()
            if goal not in GOALS:
                raise ValueError(f"Unknown goal '{raw}'. Valid goals: {', '.join(GOALS)}.")
            if goal not in goals:
                goals.append(goal)
        return goals or list(GOALS)

    @field_validator("language")
    @classmethod
    def _language_code(cls, value: str) -> str:
        code = (value or "auto").strip()
        if code.lower() == "auto":
            return "auto"
        if not LANGUAGE_RE.match(code):
            raise ValueError(f"Unknown language '{value}'. Use \"auto\" or a code such as en, el or de.")
        return code.lower()

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
    why: Optional[str] = Field(default=None, description="The planner's one line on why this scene is in the tour")
    verdict: Optional[str] = Field(default=None, description="The footage review's verdict for this scene")
    seen: Optional[str] = Field(default=None, description="What the reviewer saw on screen in this scene")


class Storyboard(BaseModel):
    project_name: str
    tagline: str
    scenes: List[Scene]
    total_duration_sec: float = 0.0
    intro_sec: float = 0.0  # length of the intro before scene 1 in demo.mp4 (0 without AI clips)
    outro_sec: float = 0.0
    intro_visual: Optional[str] = None  # Gemini's shot descriptions for the AI clips
    outro_visual: Optional[str] = None
    audience_note: Optional[str] = Field(default=None, description="Why these pages, in this order, for this viewer")
    plan_rationale: Optional[str] = Field(default=None, description="Why these pages in this order (one sentence)")
    music_prompt: Optional[str] = Field(default=None, description="Gemini's brief for the fal.ai soundtrack")
    cover_visual: Optional[str] = Field(default=None, description="Gemini's brief for the fal.ai cover art")
    soundtrack: Optional[str] = Field(default=None, description="How the soundtrack was made, or why it was skipped")
    cover: Optional[str] = Field(default=None, description="How the cover art was made, or why it was skipped")


class LaunchKit(BaseModel):
    readme_markdown: str
    devpost_pitch: str
    x_thread: List[str]
    linkedin_post: str
    outreach_email: Optional[OutreachEmail] = Field(default=None, description="Personalized runs only; body carries {{DEMO_LINK}} once")
    linkedin_dm: Optional[str] = Field(default=None, description="Personalized runs only; carries {{DEMO_LINK}} once")


class StepState(BaseModel):
    """One row of the agent trace: a pipeline step and the tool behind it (timestamps, never ticking counters)."""
    id: str
    status: Literal["running", "done", "skipped", "error"]
    label: str
    tool: Optional[str] = Field(default=None, description="gemini, fal, chromium, ffmpeg or github")
    detail: Optional[str] = None
    message: Optional[str] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None


class ToolCall(BaseModel):
    """One Gemini or fal.ai call the job made (the ledger)."""
    tool: Literal["gemini", "fal"]
    purpose: str
    model: str
    started_at: float
    ended_at: Optional[float] = None
    ms: Optional[int] = None
    ok: Optional[bool] = None
    note: str = ""


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: int = 0
    stage: str = "initialized"
    logs: List[str] = []
    storyboard: Optional[Storyboard] = None
    launch_kit: Optional[LaunchKit] = None
    video_url: Optional[str] = None
    cover_url: Optional[str] = Field(default=None, description="The cover poster (1280x720 JPEG, for <video poster>), "
                                                               "when one was painted")
    error: Optional[str] = None
    app_url: Optional[str] = None
    created_at: Optional[float] = Field(default=None, description="Unix time the job was accepted")
    options: Dict[str, Any] = Field(default_factory=dict,
                                    description="Request options echoed for the kit contract: goals, requestedLanguage")
    audience: Optional[AudienceBrief] = Field(default=None, description="The customer brief of a personalized run "
                                                                        "(source 'pending' until research finishes)")
    steps: List[StepState] = Field(default_factory=list, description="The agent trace, one row per pipeline step")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Every Gemini and fal.ai call, with latency")


class JobStarted(JobStatus):
    """`POST /api/pipeline/start` (202): the queued job, plus `jobId` as the Launch Kit contract names it."""
    jobId: str = Field(..., description="The same id as job_id")


class JobSummary(BaseModel):
    """One row of `GET /api/jobs`, camelCase as the Launch Kit contract asks."""
    id: str
    url: Optional[str] = None
    createdAt: Optional[str] = Field(default=None, description="ISO 8601, UTC")
    productName: Optional[str] = None
    status: Literal["completed", "failed"]
    customer: Optional[str] = Field(default=None, description="Company of a personalized run")
    persona: Optional[str] = None
    customerGoal: Optional[str] = None
    durationSec: Optional[float] = None
    coverUrl: Optional[str] = Field(default=None, description="The cover poster, when one was painted")


class ErrorMessage(BaseModel):
    """422: one plain sentence in `message`, Pydantic's per-field errors in `detail`."""
    message: str = Field(..., examples=["Unknown goal 'press'. Valid goals: social, community, investors, customers."])
    detail: List[Dict[str, Any]] = Field(default_factory=list)


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
    session_storage_keys: int = Field(0, description="sessionStorage items saved from the signed-in tab")
    indexed_db_records: int = Field(0, description="IndexedDB records saved (Firebase Auth keeps the user there)")
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
