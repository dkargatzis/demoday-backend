"""Footage review: Gemini judges contact sheets of every recorded scene, and one broken scene may be retaken once.

No video is uploaded. ffmpeg samples each MUXED scene clip (so Gemini sees the zoomed output) into one 2x2 contact
sheet of four frames at 0, 25, 50 and 75 % of the scene, and ONE multimodal call reviews all scenes at once. The
verdicts land on `Scene.verdict` / `Scene.seen` and in the `review` step event. `should_retake` is the gate for the
single allowed retake; `retake_scene` re-records one scene in a fresh browser session, re-checks it with one more
call and hands back the replacement clip only when it is verified. Everything is fail-soft: a review that fails or
times out leaves the cut unchanged, and a retake that fails or is not better keeps the original clip.
"""
import asyncio
import inspect
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from app.models.schemas import Scene
from app.services.gemini import generate_json
from app.services.recorder import record_session
from app.services.stitcher import mux_scene_clip, probe_duration, run_ffmpeg

logger = logging.getLogger("demoday.review")

PathLike = Union[str, Path]
LogFn = Callable[[str], None]

FRAMES_PER_SCENE = 4      # 2x2 sheet: frames at 0, 25, 50 and 75 % of the scene
FRAME_WIDTH = 640         # per frame; a 2x2 sheet is 1280x720 (about 100 KB JPEG, two Gemini image tiles)
JPEG_QUALITY = 5          # ffmpeg -q:v (2 best ... 31 worst)
MAX_SHEET_JOBS = 6        # concurrent ffmpeg extractions
GEMINI_TIMEOUT_SEC = 20   # HTTP backstop only; the orchestrator's _optional wall is the budget (see REVIEW_THINKING)
# Measured on six sheets: the model's default thinking level answers in 10-15 s (1.6-2.6k thinking tokens), "minimal"
# gives the same verdicts in 3.6 s. Passed through as soon as the shared generate_json accepts `thinking=`.
REVIEW_THINKING = "minimal"
VERDICTS = ("ok", "blank", "error_page", "login_wall", "static", "mismatch", "unknown")
RETAKE_VERDICTS = ("blank", "error_page")  # what a fresh recording can fix (never a sign-in wall)
RETAKE_MIN_CONFIDENCE = 0.7
LOOKS = {"ok": "ok", "blank": "blank", "error_page": "like an error page", "login_wall": "like a sign-in wall",
         "static": "static", "mismatch": "off-script", "unknown": "unreviewed"}
_SYNONYMS = {
    "ok": "ok", "okay": "ok", "pass": "ok", "passed": "ok", "good": "ok", "fine": "ok",
    "blank": "blank", "empty": "blank", "white": "blank", "loading": "blank", "skeleton": "blank", "spinner": "blank",
    "error_page": "error_page", "error": "error_page", "http_error": "error_page", "not_found": "error_page",
    "404": "error_page", "500": "error_page", "crash": "error_page",
    "login_wall": "login_wall", "login": "login_wall", "sign_in": "login_wall", "signin": "login_wall",
    "sign_in_form": "login_wall", "login_form": "login_wall", "auth": "login_wall", "sign_up": "login_wall",
    "static": "static", "frozen": "static", "identical": "static", "no_change": "static",
    "mismatch": "mismatch", "mismatched": "mismatch", "wrong_page": "mismatch", "off_script": "mismatch",
}


@dataclass
class Verdict:
    """Gemini's verdict for one scene (`scene` is 1-based; `title` is for the dashboard chip)."""
    scene: int
    verdict: str
    confidence: float
    seen: str = ""
    issue: str = ""
    title: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict == "ok"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- frames


async def contact_sheet(clip: PathLike, duration: float, dest: PathLike, frames: int = FRAMES_PER_SCENE) -> Path:
    """One JPEG of `frames` frames of `clip` sampled evenly from 0 % on: 2x2 for four frames, one row otherwise."""
    period = max(float(duration) / max(frames, 1), 0.1)
    cols, rows = (2, 2) if frames == 4 else (max(frames, 1), 1)
    vf = f"fps={1 / period:.4f},scale={FRAME_WIDTH}:-2,tile={cols}x{rows}"
    await run_ffmpeg(["-i", str(clip), "-vf", vf, "-frames:v", "1", "-q:v", str(JPEG_QUALITY), "-update", "1",
                      str(dest)], timeout=60)
    return Path(dest)


async def _sheets(clips: Sequence[PathLike], durations: Sequence[float], dest_dir: Path, names: Sequence[str],
                  frames: int) -> List[Optional[Path]]:
    """Contact sheets for all clips, at most MAX_SHEET_JOBS ffmpeg processes at a time; None where ffmpeg failed."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(MAX_SHEET_JOBS)

    async def one(clip: PathLike, duration: float, name: str) -> Path:
        async with gate:
            if not duration or duration <= 0:
                duration = await probe_duration(clip)
            return await contact_sheet(clip, duration, dest_dir / f"{name}.jpg", frames)

    results = await asyncio.gather(*(one(c, d, n) for c, d, n in zip(clips, durations, names)), return_exceptions=True)
    sheets: List[Optional[Path]] = []
    for name, result in zip(names, results):
        if isinstance(result, BaseException):
            logger.warning("no contact sheet for %s: %s", name, result)
            sheets.append(None)
        else:
            sheets.append(result)
    return sheets


# --------------------------------------------------------------------------- prompt + sanitizing


def _actions_text(scene: Scene) -> str:
    parts = []
    for action in scene.actions or []:
        selector = (action.selector or "").strip()[:60]
        if action.type == "click":
            parts.append(f"click {selector or 'an element'}")
        elif action.type == "type":
            parts.append(f"type \"{(action.value or '').strip()[:40]}\" into {selector or 'a field'}")
        elif action.type == "scroll":
            parts.append(f"scroll {(action.value or 'down').strip()[:40]}")
        elif action.type == "navigate":
            parts.append(f"open {(action.value or '').strip()[:60]}")
    return "; ".join(parts)[:240] or "none (the page is shown as it is)"


def build_prompt(name: str, scenes: Sequence[Scene], reviewed: Sequence[int], frames: int = FRAMES_PER_SCENE) -> str:
    """The review prompt. `reviewed` lists the 1-based scene numbers whose sheets are attached, in image order."""
    percents = ", ".join(f"{int(100 * k / frames)} %" for k in range(frames))
    numbers = ", ".join(str(n) for n in reviewed)
    lines = [
        f'You are reviewing a recorded product demo of "{name}" before it is published.',
        f"Each attached image is a contact sheet of ONE scene: {frames} frames taken at {percents} of the scene, "
        f"left to right, then top to bottom. The images are, in order, scenes {numbers}.",
        "Scenes:",
    ]
    for n in reviewed:
        scene = scenes[n - 1]
        lines.append(f'{n}. "{scene.title}" on {scene.route or "/"}: narration "{scene.narration.strip()}"; '
                     f"planned actions: {_actions_text(scene)}")
    most = max(2, (frames + 1) // 2 + (1 if frames >= 4 else 0))
    lines += [
        "Judge ONLY from the frames. Return one entry for every scene listed, as JSON:",
        '{"scenes": [{"scene": 1, "verdict": "ok|blank|error_page|login_wall|static|mismatch", "confidence": 0.0-1.0, '
        '"seen": "<= 15 words: what is visible", "issue": "<= 15 words, or empty when ok"}]}',
        f"Definitions: blank = mostly empty, white, or a loading state (spinner, skeleton) in {most} or more frames; "
        "error_page = an HTTP error, crash or not-found page; login_wall = a sign-in or sign-up form fills the page; "
        "static = every frame is identical although a click or typing was planned; mismatch = the narration describes "
        "something clearly not on screen.",
        'A green cursor, a highlight ring, zoom-ins and a small "Sample data" pill are expected, not issues. '
        "Use ok unless the scene would embarrass the presenter; confidence is how sure you are of the verdict.",
    ]
    return "\n".join(lines)


def normalize_verdict(raw: Any) -> str:
    key = re.sub(r"[\s\-]+", "_", str(raw or "").strip().lower().strip(".'\""))
    return _SYNONYMS.get(key, "unknown")


def _confidence(raw: Any) -> float:
    if isinstance(raw, str):
        word = raw.strip().lower().rstrip("%")
        words = {"high": 0.9, "very high": 0.95, "medium": 0.6, "moderate": 0.6, "low": 0.3, "very low": 0.15}
        if word in words:
            return words[word]
        raw = word
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if value != value:  # NaN
        return 0.0
    if 1.0 < value <= 100.0:
        value /= 100.0
    return round(min(max(value, 0.0), 1.0), 2)


def _scene_number(raw: Any, fallback: int) -> int:
    match = re.search(r"\d+", str(raw if raw is not None else ""))
    return int(match.group()) if match else fallback


def _text(raw: Any, limit: int = 120) -> str:
    return re.sub(r"\s+", " ", str(raw if raw is not None else "")).strip()[:limit]


def sanitize(payload: Any, scenes: Sequence[Scene]) -> List[Verdict]:
    """Gemini's payload as one Verdict per scene: enums validated, numbers clamped, missing scenes `unknown`."""
    items: Any = payload
    if isinstance(payload, dict):
        items = next((payload[k] for k in ("scenes", "results", "verdicts", "review") if isinstance(payload.get(k), list)),
                     None)
        if items is None:  # {"1": {...}, "2": {...}}
            items = [dict(v, scene=k) for k, v in payload.items() if isinstance(v, dict)]
    if not isinstance(items, list):
        items = []
    by_scene: Dict[int, Verdict] = {}
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        number = _scene_number(item.get("scene", item.get("id")), position + 1)
        if not 1 <= number <= len(scenes) or number in by_scene:
            continue
        verdict = normalize_verdict(item.get("verdict", item.get("status")))
        confidence = _confidence(item.get("confidence")) if verdict != "unknown" else 0.0
        by_scene[number] = Verdict(scene=number, verdict=verdict, confidence=confidence, seen=_text(item.get("seen")),
                                   issue=_text(item.get("issue")), title=scenes[number - 1].title)
    return [by_scene.get(i) or Verdict(scene=i, verdict="unknown", confidence=0.0, issue="no verdict returned",
                                       title=scene.title)
            for i, scene in enumerate(scenes, 1)]


def summarize(verdicts: Sequence[Verdict]) -> str:
    """'5/5 scenes pass' or '4/5 pass · scene 4 looks blank (0.85)' for the step detail."""
    total = len(verdicts)
    flagged = [v for v in verdicts if not v.passed]
    if not flagged:
        return f"{total}/{total} scenes pass"
    notes = " · ".join(f"scene {v.scene} looks {LOOKS.get(v.verdict, v.verdict)} ({v.confidence:.2f})" for v in flagged[:3])
    if len(flagged) > 3:
        notes += f" · +{len(flagged) - 3} more"
    return f"{total - len(flagged)}/{total} pass · {notes}"


def as_extra(verdicts: Sequence[Verdict]) -> Dict[str, Any]:
    """The `review` step event's `extra` (chips on the dashboard; never part of JobStatus)."""
    return {"scenes": [v.as_dict() for v in verdicts]}


def _short(exc: BaseException) -> str:
    text = str(exc).strip()
    return (text.splitlines()[0] if text else type(exc).__name__)[:140]


def _thinking_kwargs() -> Dict[str, str]:
    """{'thinking': REVIEW_THINKING} when the shared generate_json accepts it (forward-compatible), else {}."""
    try:
        params = inspect.signature(generate_json).parameters
    except (TypeError, ValueError):
        return {}
    for name in ("thinking", "thinking_level"):
        if name in params:
            return {name: REVIEW_THINKING}
    return {}


# --------------------------------------------------------------------------- review


async def _review(clips: Sequence[PathLike], scenes: Sequence[Scene], durations: Sequence[float], sheet_dir: Path,
                  names: Sequence[str], project_name: str, timeout_sec: float, frames: int, purpose: str
                  ) -> List[Verdict]:
    """Sheets + ONE Gemini call for `scenes` (no logging, no writes to the scenes)."""
    sheets = await _sheets(clips, durations, sheet_dir, names, frames)
    reviewed = [i + 1 for i, sheet in enumerate(sheets) if sheet is not None]
    if not reviewed:
        raise RuntimeError("no frames could be extracted from the scene clips")
    images = [sheets[n - 1].read_bytes() for n in reviewed]  # type: ignore[union-attr]
    payload = await generate_json(build_prompt(project_name, scenes, reviewed, frames), images=images,
                                  timeout_sec=int(timeout_sec), purpose=purpose, **_thinking_kwargs())
    verdicts = sanitize(payload, scenes)
    for verdict in verdicts:
        if verdict.scene not in reviewed:  # Gemini never saw this scene, whatever it answered
            verdict.verdict, verdict.confidence, verdict.issue = "unknown", 0.0, "no frames extracted"
    return verdicts


async def review_scenes(clips: Sequence[PathLike], scenes: Sequence[Scene], job_dir: PathLike, log: LogFn, *,
                        durations: Optional[Sequence[float]] = None, project_name: Optional[str] = None,
                        timeout_sec: float = GEMINI_TIMEOUT_SEC, frames: int = FRAMES_PER_SCENE) -> List[Verdict]:
    """Reviews the muxed scene `clips` (scene order) in one Gemini call; writes `Scene.verdict` / `Scene.seen`.

    Raises when no frames could be extracted or Gemini failed, so the orchestrator's `_optional` wrapper reports the
    review as skipped and the cut stays unchanged. Contact sheets go to `job_dir/review/` and vanish with the job dir.
    """
    scenes = list(scenes)[:len(clips)]
    clips = list(clips)[:len(scenes)]
    if not scenes:
        return []
    name = project_name or "the app"
    durations = list(durations)[:len(scenes)] if durations else [float(s.duration_sec or 0) for s in scenes]
    log(f"Review: Gemini is checking {len(scenes)} scene{'' if len(scenes) == 1 else 's'} against their narration "
        f"({frames} frames each)...")
    t0 = time.monotonic()
    verdicts = await _review(clips, scenes, durations, Path(job_dir) / "review",
                             [f"scene_{i + 1:02d}" for i in range(len(scenes))], name, timeout_sec, frames,
                             purpose="review footage")
    for scene, verdict in zip(scenes, verdicts):
        scene.verdict = verdict.verdict
        scene.seen = verdict.seen or None
    took = time.monotonic() - t0
    flagged = [v for v in verdicts if not v.passed]
    passed = len(verdicts) - len(flagged)
    if not flagged:
        example = next((v for v in verdicts if v.seen), None)
        tail = f" — scene {example.scene} '{example.title}': {example.seen}" if example else ""
        log(f"Review: {passed}/{len(verdicts)} scenes pass ({took:.1f}s){tail}")
    else:
        log(f"Review: {passed}/{len(verdicts)} scenes pass ({took:.1f}s)")
        for v in flagged:
            note = v.issue or v.seen
            log(f"Review: scene {v.scene} '{v.title}' looks {LOOKS.get(v.verdict, v.verdict)} "
                f"(confidence {v.confidence:.2f}){': ' + note if note else ''}")
    return verdicts


# --------------------------------------------------------------------------- the retake gate


def should_retake(verdicts: Sequence[Verdict]) -> Optional[int]:
    """0-based index of the ONE scene worth re-recording, or None.

    A retake records in a fresh session, so only a scene that came out blank or as an error page can be fixed by it.
    A sign-in wall anywhere means the app is gated for this browser (a second attempt hits it too), two broken scenes
    point at the app rather than the recording, and anything under RETAKE_MIN_CONFIDENCE could replace a good clip.
    """
    if any(v.verdict == "login_wall" for v in verdicts):
        return None
    broken = [v for v in verdicts if v.verdict in RETAKE_VERDICTS]
    if len(broken) != 1 or broken[0].confidence < RETAKE_MIN_CONFIDENCE:
        return None
    return broken[0].scene - 1


pick_retake = should_retake  # the name the lane brief uses


# --------------------------------------------------------------------------- the retake


async def retake_scene(scene: Scene, idx: int, app_url: str, clip_duration: float, audio_path: PathLike,
                       job_dir: PathLike, log: LogFn, storage_state: Optional[Dict] = None, replayer=None, *,
                       project_name: Optional[str] = None, timeout_sec: float = GEMINI_TIMEOUT_SEC,
                       verify: bool = True) -> Optional[Path]:
    """Re-records scene `idx` (0-based) alone in a fresh browser session, muxes it with the scene's narration and
    re-checks the frames with one Gemini call.

    Returns the verified take-2 clip (same length and codec parameters as the original, so it drops into the concat
    list in place of `scene_{idx+1:02d}.mp4`), or None when the recording failed or Gemini still sees a blank or error
    page; the caller keeps the original clip. Run it inside the render slot: the recording is timing-sensitive.
    """
    job_dir = Path(job_dir)
    number = idx + 1
    t0 = time.monotonic()

    def relog(message: str) -> None:  # the recorder numbers its one scene "Scene 1"
        log("Retake: " + re.sub(r"^Scene 1\b", f"Scene {number}", message))

    try:
        session = job_dir / f"retake_{number:02d}.webm"
        segments = await record_session([scene], app_url, [clip_duration], session, relog,
                                        storage_state=storage_state, replayer=replayer)
        segment = segments[0]
        clip = job_dir / f"scene_{number:02d}_take2.mp4"
        await mux_scene_clip(session, audio_path, clip, segment.start, segment.duration, focus=segment.focus)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - a retake never takes the job down
        logger.warning("retake of scene %d failed: %s", number, exc, exc_info=True)
        log(f"Retake: scene {number} could not be re-recorded ({_short(exc)}); keeping the original clip")
        return None
    took = time.monotonic() - t0
    if not verify:
        log(f"Retake: scene {number} re-recorded ({took:.1f}s), spliced in unverified")
        return clip
    try:
        verdict = (await _review([clip], [scene], [segment.duration], job_dir / "review", [f"scene_{number:02d}_take2"],
                                 project_name or "the app", timeout_sec, FRAMES_PER_SCENE, purpose="re-check retake"))[0]
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("re-check of scene %d failed: %s", number, exc, exc_info=True)
        log(f"Retake: scene {number} re-recorded ({took:.1f}s) but the re-check failed ({_short(exc)}); "
            "keeping the original clip")
        return None
    if verdict.verdict in RETAKE_VERDICTS or verdict.verdict == "login_wall":
        log(f"Retake: scene {number} re-recorded ({took:.1f}s) but Gemini still sees it "
            f"{LOOKS.get(verdict.verdict, verdict.verdict)} ({verdict.confidence:.2f}); keeping the original clip")
        return None
    scene.verdict, scene.seen = verdict.verdict, verdict.seen or None
    log(f"Retake: scene {number} re-recorded ({took:.1f}s), Gemini re-checked it: {verdict.verdict}"
        f"{' (' + verdict.seen + ')' if verdict.seen else ''}, spliced in before the final cut")
    return clip
