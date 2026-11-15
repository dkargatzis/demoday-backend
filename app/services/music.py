"""fal.ai soundtrack: an instrumental bed composed while the browser records, ducked under the narration at the cut.

`compose` asks fal's text-to-music model for a bed briefed by Gemini (`Storyboard.music_prompt`, or a template per
voice style) and downloads the raw file untouched. `mix_under` loops that bed under demo.mp4 through a sidechain
compressor keyed by the voice, so the music dips while the narrator speaks and comes back between sentences: the video
stream is copied, the audio re-encoded once, and the integrated loudness stays where the narration put it. `finish` is
the one call the orchestrator makes at concat time: a short grace wait for a late track, the mix, and the narration-only
fallback. Nothing here may take the job down: every failure costs one log line and an amber step row.

Log lines: "Soundtrack: Gemini's brief '...' → fal.ai {model} ({n}s requested)", "Soundtrack ready: 40.0s of music in
12.3s", "Soundtrack mixed under the narration (40s bed at -14 dB, ducked while the voice speaks)" or
"Soundtrack skipped: {reason}; the video has narration only".
"""
import asyncio
import base64
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

import httpx

from app.config import feature_on, get_secret
from app.services import ledger
from app.services.stitcher import AUDIO_RATE, MediaToolError, PathLike, describe_media, probe_duration, run_ffmpeg

logger = logging.getLogger("demoday.music")

# Stable Audio 3 small returned a 30 s stereo 44.1 kHz mp3 in 10 s; ACE-Step 1.5 took 168 s for the same request.
FAL_MUSIC_MODEL = os.getenv("FAL_MUSIC_MODEL", "fal-ai/stable-audio-3/small/music/base/text-to-audio")
MUSIC_TIMEOUT_SEC = int(os.getenv("FAL_MUSIC_TIMEOUT_SEC", "150"))
GRACE_SEC = 12                 # how long the final cut waits for a track that is still composing
MIN_TRACK_SEC = 5.0            # anything shorter is a broken generation
MAX_TRACK_SEC = 120            # the endpoint's ceiling
EXTRA_BED_SEC = 10             # requested on top of the target runtime so the loop seam rarely plays
MAX_MUSIC_BYTES = 30 * 1024 * 1024
# Generations differ by 10+ LU (probe bed -11.9 LUFS, a later one +1.0), so the bed is measured and normalized to the
# probe bed's level before the gain; MUSIC_LEVEL_DB (default -14) is then exactly the validated graph's volume filter,
# about 10 LU under the -16 LUFS narration before ducking.
BED_REF_LUFS = -12.0
DEFAULT_LEVEL_DB = -14.0
PROMPT_SUFFIX = "Instrumental, no vocals, no lyrics, steady energy, background bed for a product demo."
NEGATIVE_PROMPT = "vocals, singing, lyrics, speech, voice"
TEMPLATE_MUSIC = {
    "energetic_tech": "upbeat minimal electronic, 118 bpm, soft synth plucks and warm pads, instrumental, no vocals",
    "calm_expert": "calm ambient piano and warm pads, 84 bpm, instrumental, no vocals",
    "fast_founder": "driving indie electronic, 126 bpm, bass and bright synths, instrumental, no vocals",
}
# Measured on a -16.5 LUFS narration: the mix lands at -16.7 LUFS with the bed 14 dB under the voice and this ducking.
DUCK = "threshold=0.02:ratio=6:attack=40:release=500:makeup=1"
FADE_IN_SEC, FADE_OUT_SEC = 1.0, 2.5
MIX_TIMEOUT_SEC = 180

SkipCallback = Optional[Callable[[str], None]]
Log = Callable[[str], None]


# --------------------------------------------------------------------------- small helpers (sync, pure)


def level_db() -> float:
    """The gain on the normalized bed before ducking (env MUSIC_LEVEL_DB, default -14; clamped to [-40, 0])."""
    try:
        return max(-40.0, min(0.0, float(os.getenv("MUSIC_LEVEL_DB", "") or DEFAULT_LEVEL_DB)))
    except ValueError:
        return DEFAULT_LEVEL_DB


def bed_seconds_for(target_seconds: float) -> int:
    """How long a bed to request for a demo of `target_seconds`: the runtime plus a margin, within the endpoint's range."""
    return int(max(MIN_TRACK_SEC, min(MAX_TRACK_SEC, round(target_seconds) + EXTRA_BED_SEC)))


def brief_for(prompt: Optional[str], voice_style: str = "energetic_tech") -> Tuple[str, str]:
    """(brief, source): Gemini's music_prompt when it wrote one, else the template for the voice style."""
    text = " ".join((prompt or "").split()).strip().strip("\"'")
    if text:
        return text[:300], "Gemini's brief"
    return TEMPLATE_MUSIC.get(voice_style, TEMPLATE_MUSIC["energetic_tech"]), "template brief"


def model_label(model: Optional[str] = None) -> str:
    """'stable-audio-3/small' for the UI, from the full endpoint id."""
    model = model or FAL_MUSIC_MODEL
    label = model[len("fal-ai/"):] if model.startswith("fal-ai/") else model
    for marker in ("/music/", "/text-to-"):
        if marker in label:
            label = label.split(marker, 1)[0]
    return label or model


def skip_line(reason: str) -> str:
    """The log line for every way the soundtrack can be missing."""
    return f"Soundtrack skipped: {reason}; the video has narration only"


def late_reason(grace: float = GRACE_SEC) -> str:
    return f"fal.ai did not deliver within {grace:.0f}s of the final cut"


def describe(track_seconds: float, *, model: Optional[str] = None, level: Optional[float] = None) -> str:
    """The Storyboard.soundtrack note: 'stable-audio-3/small · 40 s track · mixed at -14 dB, ducked under the voice'."""
    level = level_db() if level is None else level
    return f"{model_label(model)} · {track_seconds:.0f} s track · mixed at {level:.0f} dB, ducked under the voice"


def _music_arguments(model: str, prompt: str, seconds: int) -> Dict[str, Any]:
    """Request body per model family (keys from the endpoints' OpenAPI schemas; stable-audio's are the probe's)."""
    if "stable-audio" in model:
        return {"prompt": prompt, "duration": seconds, "output_format": "mp3", "num_inference_steps": 50,
                "negative_prompt": NEGATIVE_PROMPT}
    if "ace-step" in model:  # empty or "[inst]" lyrics make ACE-Step render an instrumental
        return {"prompt": prompt, "duration": seconds, "lyrics": "[inst]"}
    return {"prompt": prompt, "duration": seconds}


def _audio_url(result: Any) -> Optional[str]:
    """The track URL from any of the shapes fal's audio endpoints use: audio_file.url, audio.url, audio_url, audios[0].url."""
    if not isinstance(result, dict):
        return None
    for key in ("audio_file", "audio"):
        value = result.get(key)
        if isinstance(value, dict) and isinstance(value.get("url"), str) and value["url"]:
            return value["url"]
        if isinstance(value, str) and value:
            return value
    if isinstance(result.get("audio_url"), str) and result["audio_url"]:
        return result["audio_url"]
    audios = result.get("audios")
    if isinstance(audios, list) and audios and isinstance(audios[0], dict) and audios[0].get("url"):
        return str(audios[0]["url"])
    return None


def _reason(exc: BaseException) -> str:
    if isinstance(exc, asyncio.TimeoutError) or "timed out" in str(exc).lower():
        return f"no answer from fal.ai within {MUSIC_TIMEOUT_SEC}s"
    text = str(exc).strip()
    return (text.splitlines()[0] if text else type(exc).__name__)[:140]


def _report(log: Log, on_skip: SkipCallback, reason: str) -> None:
    """One log line and the amber row (through the caller's callback); a callback bug never breaks the pipeline."""
    reason = reason.strip()[:140] or "unknown error"
    log(skip_line(reason))
    if on_skip is not None:
        try:
            on_skip(reason)
        except Exception:  # noqa: BLE001
            logger.exception("music on_skip callback failed")


# --------------------------------------------------------------------------- compose (fal.ai)


async def compose(prompt: Optional[str], seconds: float, job_dir: PathLike, log: Log, *,
                  voice_style: str = "energetic_tech", on_skip: SkipCallback = None) -> Optional[Path]:
    """Composes an instrumental bed of about `seconds` seconds into `job_dir/music.bin` and returns its path.

    Fail-soft: without FAL_KEY, with FEATURE_MUSIC off, or when fal.ai fails, times out or returns less than
    MIN_TRACK_SEC of audio, it logs the skip line, tells `on_skip` why, and returns None. Only cancellation
    propagates. The file is the model's raw output (mp3, flac, ...: ffmpeg probes the container), so nothing is
    re-encoded while the browser records; `mix_under` reads it as is.
    """
    brief, source = brief_for(prompt, voice_style)
    seconds = int(max(MIN_TRACK_SEC, min(MAX_TRACK_SEC, round(seconds))))
    model = FAL_MUSIC_MODEL
    if not feature_on("MUSIC"):
        _report(log, on_skip, "FEATURE_MUSIC is off")
        return None
    key = get_secret("FAL_KEY")
    if not key:
        _report(log, on_skip, "FAL_KEY not set")
        return None
    log(f"Soundtrack: {source} '{brief}' → fal.ai {model} ({seconds}s requested)")
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / "music.bin"
    t0 = time.monotonic()
    try:
        duration = await _generate(model, f"{brief.rstrip('. ')}. {PROMPT_SUFFIX}", seconds, dest, key)
    except asyncio.CancelledError:
        dest.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001 - the soundtrack is best effort by design
        logger.warning("soundtrack composition failed: %s", exc, exc_info=not isinstance(exc, asyncio.TimeoutError))
        dest.unlink(missing_ok=True)
        _report(log, on_skip, _reason(exc))
        return None
    log(f"Soundtrack ready: {duration:.1f}s of music in {time.monotonic() - t0:.1f}s")
    return dest


async def _generate(model: str, prompt: str, seconds: int, dest: Path, key: str) -> float:
    """One ledger row: the fal.ai call, the download and the sanity check of what came back; returns the seconds."""
    import fal_client

    client = fal_client.AsyncClient(key=key)
    async with ledger.call("fal", "soundtrack", model):
        result = await asyncio.wait_for(
            client.subscribe(model, arguments=_music_arguments(model, prompt, seconds), client_timeout=MUSIC_TIMEOUT_SEC),
            timeout=MUSIC_TIMEOUT_SEC + 10,
        )
        url = _audio_url(result)
        if not url:
            raise RuntimeError(f"unexpected fal.ai response: {str(result)[:160]}")
        if await _download(url, dest) == 0:
            raise RuntimeError("fal.ai returned an empty audio file")
        duration = await probe_duration(dest)
        if duration < MIN_TRACK_SEC:
            raise RuntimeError(f"fal.ai returned only {duration:.1f}s of audio")
    return duration


async def _download(url: str, dest: Path) -> int:
    """Streams the track to `dest` (any container, never renamed by URL) under a size cap; returns the byte count."""
    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        data = base64.b64decode(payload) if header.endswith(";base64") else payload.encode()
        if len(data) > MAX_MUSIC_BYTES:
            raise RuntimeError("the generated track is unexpectedly large")
        dest.write_bytes(data)
        return len(data)
    size = 0
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http:
        async with http.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as out:
                async for chunk in resp.aiter_bytes():
                    size += len(chunk)
                    if size > MAX_MUSIC_BYTES:
                        raise RuntimeError("the generated track is unexpectedly large")
                    out.write(chunk)
    return size


# --------------------------------------------------------------------------- mix (FFmpeg)


async def mix_under(video_in: PathLike, music_path: PathLike, video_out: PathLike, log: Log) -> Path:
    """Writes `video_out`: `video_in` with `music_path` looped underneath and ducked while the narration speaks.

    The video stream is copied; the audio is re-encoded once (AAC 160k, 48 kHz stereo): the bed is measured
    (EBU R128), normalized to BED_REF_LUFS and set `level_db()` lower, so every generation sits at the same level,
    faded in and out, compressed with the voice as the sidechain key, then summed with the untouched voice. The mix
    goes to a temp name and is validated (same video codec, same length within 0.2 s, A/V within 0.1 s, AAC) before
    it takes `video_out`'s name, so a failure raises MediaToolError and leaves nothing behind: the caller keeps
    `video_in`.
    """
    video_in, music_path, video_out = Path(video_in), Path(music_path), Path(video_out)
    if video_out.resolve() == video_in.resolve():
        raise MediaToolError("mix_under writes a new file: video_out must differ from video_in")
    if not music_path.is_file():
        raise MediaToolError(f"soundtrack file missing: {music_path}")
    source = await describe_media(video_in)
    if not source.get("video_codec") or not source.get("audio_codec"):
        raise MediaToolError(f"{video_in.name} needs both a video and an audio stream to mix under")
    total = source["duration"]
    bed_sec = await probe_duration(music_path)
    bed_lufs = await _integrated_lufs(music_path)
    level = level_db()
    # An unmeasurable bed (silence, odd file) gets the plain gain, as if it were at the reference level already.
    gain = level if bed_lufs is None else max(-40.0, min(10.0, BED_REF_LUFS + level - bed_lufs))
    logger.info("mixing %s: bed %.1fs at %s LUFS, gain %.1f dB", music_path.name, bed_sec,
                "?" if bed_lufs is None else f"{bed_lufs:.1f}", gain)
    fade_out = min(FADE_OUT_SEC, total / 2)
    graph = ";".join([
        f"[1:a]aresample={AUDIO_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo,"
        f"atrim=0:{total:.4f},asetpts=PTS-STARTPTS,volume={gain:.1f}dB,"
        f"afade=t=in:st=0:d={FADE_IN_SEC},afade=t=out:st={max(0.0, total - fade_out):.4f}:d={fade_out:.4f}[bed]",
        "[0:a]asplit=2[voice][key]",
        f"[bed][key]sidechaincompress={DUCK}[ducked]",
        "[voice][ducked]amix=inputs=2:duration=first:normalize=0[mix]",
    ])
    tmp = video_out.with_name(f"{video_out.stem}.mixing{video_out.suffix or '.mp4'}")
    try:
        await run_ffmpeg([
            "-i", str(video_in), "-stream_loop", "-1", "-i", str(music_path),
            "-filter_complex", graph, "-map", "0:v", "-map", "[mix]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-ar", str(AUDIO_RATE), "-ac", "2",
            "-movflags", "+faststart", str(tmp),
        ], timeout=MIX_TIMEOUT_SEC)
        mixed = await describe_media(tmp)
        problems = []
        if mixed["video_codec"] != source["video_codec"]:
            problems.append(f"video codec {mixed['video_codec']} (expected {source['video_codec']})")
        if mixed["audio_codec"] != "aac":
            problems.append(f"audio codec {mixed['audio_codec']}")
        if abs(mixed["duration"] - total) > 0.2:
            problems.append(f"length {mixed['duration']:.2f}s (expected {total:.2f}s)")
        if abs(mixed["video_duration"] - mixed["audio_duration"]) > 0.1:
            problems.append(f"A/V drift {abs(mixed['video_duration'] - mixed['audio_duration']):.3f}s")
        if problems:
            raise MediaToolError("the mixed file failed validation: " + "; ".join(problems))
        tmp.replace(video_out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    log(f"Soundtrack mixed under the narration ({bed_sec:.0f}s bed at {level:.0f} dB, ducked while the voice speaks)")
    return video_out


async def _integrated_lufs(path: Path) -> Optional[float]:
    """The bed's integrated loudness (ffmpeg ebur128), or None when it cannot be measured."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-i", str(path), "-vn",
            "-af", "ebur128=framelog=quiet", "-f", "null", "-",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        logger.warning("could not measure the bed's loudness: %s", exc)
        return None
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), 60)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return None
    match = re.search(r"\bI:\s+(-?[\d.]+) LUFS", stderr.decode(errors="replace"))
    return float(match.group(1)) if match else None


# --------------------------------------------------------------------------- the final cut


async def finish(task: Optional[Awaitable[Optional[Path]]], raw_video: PathLike, final_video: PathLike, log: Log, *,
                 grace: float = GRACE_SEC, on_skip: SkipCallback = None) -> Optional[str]:
    """Turns `raw_video` (the concatenated cut) into `final_video`, with the soundtrack when there is one.

    Waits up to `grace` seconds for `task` (the `compose` task; a late one is cancelled), mixes its bed under
    `raw_video` and returns the Storyboard.soundtrack note. Without a task, or when the track or the mix fails,
    `raw_video` is renamed to `final_video` and None is returned; every skip is logged and sent to `on_skip`
    (except the ones `compose` already reported itself). `final_video` always exists afterwards.
    """
    raw_video, final_video = Path(raw_video), Path(final_video)
    track: Optional[Path] = None
    if task is not None:
        future = asyncio.ensure_future(task)
        try:
            track = await asyncio.wait_for(future, None if future.done() else grace)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            _report(log, on_skip, late_reason(grace))
        except Exception as exc:  # noqa: BLE001 - compose is fail-soft; this guards a bug in it
            logger.warning("soundtrack task failed: %s", exc, exc_info=True)
            _report(log, on_skip, _reason(exc))
    if track is not None:
        try:
            await mix_under(raw_video, track, final_video, log)
            return describe(await probe_duration(track))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the un-mixed cut ships instead
            logger.warning("soundtrack mix failed: %s", exc, exc_info=True)
            _report(log, on_skip, f"the mix failed ({_reason(exc)})")
    raw_video.replace(final_video)
    return None
