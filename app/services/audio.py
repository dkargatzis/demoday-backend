"""Narration synthesis: fal.ai Kokoro first, Gemini TTS when fal's queue is cold, then local TTS and silence.

A batch of lines (a video's scenes, or its intro and outro) is voiced by one engine, so the voice never changes
between scenes. fal.ai gets a head start of TTS_HEDGE_SEC; when it has not finished every line by then, Gemini TTS
voices the same lines side by side and fal keeps FAL_TTS_GRACE_SEC more to land first. A cold fal queue therefore
costs seconds, not the minutes its worker can take to start. `bind_job_voice()` carries the engine that won the
first batch to the job's later batches.

Every engine's output is normalized to 48 kHz stereo PCM WAV, and the returned duration is measured with ffprobe so
the browser recording can be timed to the real narration length.
"""
import asyncio
import base64
import contextvars
import logging
import os
import re
import shutil
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, List, Optional, Sequence, Tuple

import httpx

from app.config import feature_on, get_secret
from app.services import cloud, ledger
from app.services.stitcher import AUDIO_RATE, probe_duration, run_ffmpeg

logger = logging.getLogger("demoday.audio")

# fal.ai's PlayHT endpoints were retired (404); Kokoro is fal-hosted. Schema: prompt/voice/speed -> audio.url
FAL_TTS_ENDPOINT = "fal-ai/kokoro/american-english"
# A warm Kokoro worker answers in 2-4 s; a cold one has taken from 35 s to over 4 minutes to start.
FAL_TTS_TIMEOUT_SEC = int(os.getenv("FAL_TTS_TIMEOUT_SEC", "90"))
TTS_HEDGE_SEC = float(os.getenv("TTS_HEDGE_SEC", "20"))  # fal's head start before the next engine joins
FAL_TTS_GRACE_SEC = float(os.getenv("FAL_TTS_GRACE_SEC", "10"))  # once a fallback is ready, how much longer fal may take

# Gemini TTS: the same GEMINI_API_KEY. Plain text only: a style direction in the prompt is spoken aloud or adds a
# long preamble, so the tone comes from the voice.
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts").strip()
GEMINI_TTS_TIMEOUT_SEC = int(os.getenv("GEMINI_TTS_TIMEOUT_SEC", "45"))
GOOGLE_TTS_TIMEOUT_SEC = 60
_ENGINE_MARGIN_SEC = 5  # past its own timeout, a line still running counts as failed

# voice_style -> (Kokoro voice, Kokoro speed, local TTS words-per-minute)
VOICE_PRESETS = {
    "energetic_tech": ("af_heart", 1.1, 195),
    "calm_expert": ("am_michael", 0.95, 170),
    "fast_founder": ("af_bella", 1.2, 220),
}
# voice_style -> Gemini prebuilt voice (the Chirp 3 HD names in cloud.GOOGLE_VOICES are the same voices)
GEMINI_VOICES = {"energetic_tech": "Aoede", "calm_expert": "Charon", "fast_founder": "Leda"}


@dataclass
class _Engine:
    key: str  # "fal" | "gemini" | "google"
    label: str  # how the log names it: "fal.ai Kokoro (af_heart, 1.1x)"
    short: str  # how a miss is named: "fal.ai"
    timeout: float
    run: Callable[[str, Path], Awaitable[Path]]  # (text, destination without extension) -> the audio file written


class _JobVoice:
    """The engine that voiced a job's first batch; later batches of the same job try it first."""

    def __init__(self) -> None:
        self.engine: Optional[str] = None
        self.lock = asyncio.Lock()


_VOICE: contextvars.ContextVar[Optional[_JobVoice]] = contextvars.ContextVar("demoday_voice", default=None)


def bind_job_voice() -> None:
    """One voice per job: call at the start of a job, before its tasks are created, so they share the choice."""
    _VOICE.set(_JobVoice())


async def synthesize_scene_audio(text: str, output_path, voice_style: str = "energetic_tech") -> Tuple[float, str]:
    """Synthesizes `text` into a normalized WAV at `output_path`.

    Returns (exact duration in seconds via ffprobe, description of the engine that produced it).
    """
    return (await synthesize_narration([text], [output_path], voice_style))[0]


async def synthesize_narration(texts: Sequence[str], output_paths: Sequence, voice_style: str = "energetic_tech"
                               ) -> List[Tuple[float, str]]:
    """Voices every line with one engine and writes normalized WAVs to `output_paths` (same order).

    Returns one (duration via ffprobe, engine description) per line. Lines no engine could voice fall back to local
    TTS, then silence, one by one.
    """
    engines = _engines(voice_style)
    job_voice = _VOICE.get()
    if job_voice is None:
        results, _ = await _synthesize(texts, output_paths, voice_style, engines)
        return results
    async with job_voice.lock:  # the job's batches take turns, so the first one's engine is known to the next
        if job_voice.engine:
            engines.sort(key=lambda engine: engine.key != job_voice.engine)
        results, chosen = await _synthesize(texts, output_paths, voice_style, engines)
        if job_voice.engine is None and chosen:
            job_voice.engine = chosen
    return results


def voice_tool(engines: Sequence[str]) -> Optional[str]:
    """The agent-trace tool for a batch's engine descriptions, when every line ended on the same engine."""
    tools = set()
    for engine in engines:
        last = engine.rsplit("→", 1)[-1].strip()
        tools.add("fal" if last.startswith("fal.ai") else "gemini" if last.startswith("Gemini") else
                  "google" if last.startswith("Google") else "local")
    return tools.pop() if len(tools) == 1 else None


async def warm_up_fal() -> None:
    """Fire-and-forget: asks fal.ai for a one-word clip so its worker is warm by the time narration is needed."""
    key = get_secret("FAL_KEY")
    if not key:
        return
    try:
        import fal_client

        client = fal_client.AsyncClient(key=key)
        async with ledger.call("fal", "warm-up", FAL_TTS_ENDPOINT):
            await client.subscribe(FAL_TTS_ENDPOINT, arguments={"prompt": "Ready.", "voice": "af_heart", "speed": 1.0},
                                   client_timeout=300)
    except Exception as exc:
        logger.info("fal.ai warm-up did not complete: %s", _short(exc))


def _engines(voice_style: str) -> List[_Engine]:
    """The engines with a key, in order of preference."""
    voice, speed, _ = VOICE_PRESETS.get(voice_style, VOICE_PRESETS["energetic_tech"])
    engines = []
    fal_key = get_secret("FAL_KEY")
    if fal_key:
        engines.append(_Engine("fal", f"fal.ai Kokoro ({voice}, {speed}x)", "fal.ai", FAL_TTS_TIMEOUT_SEC,
                               lambda text, dest: _fal_tts(text, dest, voice, speed, fal_key)))
    gemini_key = get_secret("GEMINI_API_KEY")
    if gemini_key and GEMINI_TTS_MODEL and feature_on("GEMINI_TTS"):
        name = GEMINI_VOICES.get(voice_style, GEMINI_VOICES["energetic_tech"])
        engines.append(_Engine("gemini", f"Gemini TTS ({GEMINI_TTS_MODEL}, {name})", "Gemini TTS",
                               GEMINI_TTS_TIMEOUT_SEC, lambda text, dest: _gemini_tts(text, dest, name, gemini_key)))
    if cloud.google_tts_enabled():
        google_name = cloud.GOOGLE_VOICES.get(voice_style, cloud.GOOGLE_VOICES["energetic_tech"])[0]
        engines.append(_Engine("google", f"Google Cloud TTS ({google_name})", "Google TTS", GOOGLE_TTS_TIMEOUT_SEC,
                               lambda text, dest: cloud.google_tts(text, dest.with_suffix(".wav"), voice_style)))
    return engines


async def _synthesize(texts: Sequence[str], output_paths: Sequence, voice_style: str, engines: List[_Engine]
                      ) -> Tuple[List[Tuple[float, str]], Optional[str]]:
    _, _, wpm = VOICE_PRESETS.get(voice_style, VOICE_PRESETS["energetic_tech"])
    outputs = [Path(p) for p in output_paths]
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=outputs[0].parent) as tmp:
        tmp_dir = Path(tmp)
        if engines:
            raws, labels, chosen = await _voice_batch(texts, tmp_dir, engines)
        else:
            raws, labels, chosen = [None] * len(texts), [""] * len(texts), None

        async def finish(i: int) -> None:
            raw = raws[i]
            if raw is None:
                scene_dir = tmp_dir / f"local_{i:02d}"
                scene_dir.mkdir()
                raw = await _local_tts(texts[i], scene_dir, wpm)
                if raw is not None:
                    labels[i] += ("local TTS fallback (macOS say)" if raw.suffix == ".aiff"
                                  else "local TTS fallback (espeak-ng)")
            if raw is None:
                raw = tmp_dir / f"silence_{i:02d}.wav"
                _write_silence(raw, max(2.5, len(texts[i].split()) / 2.5))
                labels[i] += "silent placeholder fallback (no TTS engine available)"
            # EBU R128 loudness normalization: every engine and every scene lands at the same voiceover level.
            await run_ffmpeg(["-i", str(raw), "-af", "loudnorm=I=-16:LRA=11:TP=-1.5",
                              "-ar", str(AUDIO_RATE), "-ac", "2", "-c:a", "pcm_s16le", str(outputs[i])])

        await asyncio.gather(*(finish(i) for i in range(len(texts))))
    durations = await asyncio.gather(*(probe_duration(path) for path in outputs))
    return list(zip(durations, labels)), chosen


async def _voice_batch(texts: Sequence[str], tmp_dir: Path, engines: List[_Engine]
                       ) -> Tuple[List[Optional[Path]], List[str], Optional[str]]:
    """Runs the engines as a hedged race over the whole batch.

    The first engine starts at once; the next one joins when the newest has failed a line or has had TTS_HEDGE_SEC
    without finishing them all. The most preferred engine that voices every line wins; a less preferred one that
    finishes first waits FAL_TTS_GRACE_SEC for the others. Returns (audio per line or None, description per line,
    the winning engine's key). With no winner, each line keeps any audio an engine made for it (voices then differ).
    """
    runs: List[Tuple[_Engine, List[asyncio.Task], float]] = []
    queue = list(engines)

    def launch() -> None:
        engine = queue.pop(0)
        tasks = [asyncio.create_task(asyncio.wait_for(engine.run(text, tmp_dir / f"{engine.key}_{i:02d}"),
                                                      engine.timeout + _ENGINE_MARGIN_SEC))
                 for i, text in enumerate(texts)]
        for task in tasks:
            task.add_done_callback(_retrieve)
        runs.append((engine, tasks, time.monotonic()))

    def state(tasks: List[asyncio.Task]) -> str:
        if any(task.done() and (task.cancelled() or task.exception() is not None) for task in tasks):
            return "failed"
        return "ok" if all(task.done() for task in tasks) else "pending"

    winner: Optional[int] = None
    grace_until: Optional[float] = None
    states: List[str] = []
    decided_at = time.monotonic()
    launch()
    try:
        while True:
            states = [state(tasks) for _, tasks, _ in runs]
            decided_at = now = time.monotonic()
            live = [i for i, s in enumerate(states) if s != "failed"]
            if live and states[live[0]] == "ok":
                winner = live[0]
                break
            ready = [i for i in live if states[i] == "ok"]
            if ready:
                grace_until = grace_until or now + FAL_TTS_GRACE_SEC
                if now >= grace_until:
                    winner = ready[0]
                    break
            hedge = bool(queue) and not ready  # a fallback that voiced every line already needs no third engine
            if hedge and (states[-1] == "failed" or now - runs[-1][2] >= TTS_HEDGE_SEC):
                launch()
                continue
            pending = [task for _, tasks, _ in runs for task in tasks if not task.done()]
            if not pending:
                break
            if not live:  # every engine has failed a line: let the rest finish (each has a timeout) to keep them
                await asyncio.wait(pending)
                continue
            wakeups = [runs[-1][2] + TTS_HEDGE_SEC - now] if hedge else []
            if grace_until:
                wakeups.append(grace_until - now)
            await asyncio.wait(pending, timeout=max(0.05, min(wakeups)) if wakeups else None,
                               return_when=asyncio.FIRST_COMPLETED)
    finally:
        for i, (_, tasks, _) in enumerate(runs):
            if i != winner:
                for task in tasks:
                    task.cancel()

    def missed(index: int) -> str:
        engine, tasks, launched = runs[index]
        if states[index] == "failed":
            exc = next((task.exception() for task in tasks if task.done() and not task.cancelled()
                        and task.exception() is not None), None)
            reason = (f"no answer within {engine.timeout:.0f}s" if isinstance(exc, asyncio.TimeoutError)
                      else _short(exc) if exc is not None else "cancelled")
            logger.warning("%s TTS failed: %s", engine.short, reason)
            return f"{engine.short} {reason} → "
        return f"{engine.short} no answer within {decided_at - launched:.0f}s → "

    if winner is not None:
        engine, tasks, _ = runs[winner]
        note = "".join(missed(i) for i in range(winner))
        return [task.result() for task in tasks], [note + engine.label] * len(texts), engine.key

    raws: List[Optional[Path]] = []
    labels: List[str] = []
    for line in range(len(texts)):
        made = next(((engine, tasks[line]) for engine, tasks, _ in runs if tasks[line].done()
                     and not tasks[line].cancelled() and tasks[line].exception() is None), None)
        raws.append(made[1].result() if made else None)
        labels.append(made[0].label + " (other lines in another voice)" if made
                      else "".join(missed(i) for i in range(len(runs))))
    return raws, labels, None


async def _fal_tts(text: str, dest: Path, voice: str, speed: float, key: str) -> Path:
    import fal_client

    client = fal_client.AsyncClient(key=key)
    async with ledger.call("fal", "narration", FAL_TTS_ENDPOINT):
        result = await client.subscribe(
            FAL_TTS_ENDPOINT,
            arguments={"prompt": text, "voice": voice, "speed": speed},
            client_timeout=FAL_TTS_TIMEOUT_SEC,
        )
        audio = (result or {}).get("audio") or {}
        url = audio.get("url") if isinstance(audio, dict) else None
        if not url:
            raise RuntimeError(f"unexpected fal.ai response: {str(result)[:200]}")

        if url.startswith("data:"):
            dest.write_bytes(base64.b64decode(url.split(",", 1)[1]))
        else:
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http:
                resp = await http.get(url)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
        if dest.stat().st_size == 0:
            raise RuntimeError("fal.ai returned an empty audio file")
    return dest


async def _gemini_tts(text: str, dest: Path, voice: str, key: str) -> Path:
    from google import genai
    from google.genai import types

    # speech_config exists on v1beta only (v1 rejects the field); 429/5xx are retried with backoff.
    client = genai.Client(api_key=key, http_options=types.HttpOptions(
        api_version="v1beta", timeout=GEMINI_TTS_TIMEOUT_SEC * 1000,
        retry_options=types.HttpRetryOptions(attempts=3, http_status_codes=[429, 500, 503])))
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))))
    try:
        async with ledger.call("gemini", "narration", GEMINI_TTS_MODEL):
            response = await client.aio.models.generate_content(model=GEMINI_TTS_MODEL, contents=text, config=config)
            blob = next((part.inline_data for candidate in (response.candidates or [])
                         for part in ((candidate.content.parts if candidate.content else None) or [])
                         if part.inline_data and part.inline_data.data), None)
            if blob is None:
                raise RuntimeError("Gemini TTS returned no audio")
            return write_tts_audio(blob.data, blob.mime_type or "", dest.with_suffix(".wav"))
    finally:
        await client.aio.aclose()


def write_tts_audio(data: bytes, mime: str, dest: Path) -> Path:
    """Gemini TTS answers with a WAV file or with raw 16-bit mono PCM (audio/L16;rate=24000); both become a WAV."""
    if not data:
        raise RuntimeError("Gemini TTS returned an empty audio file")
    if data[:4] == b"RIFF":
        dest.write_bytes(data)
        return dest
    match = re.search(r"rate=(\d+)", mime)
    with wave.open(str(dest), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(match.group(1)) if match else 24000)
        wav_file.writeframes(data)
    return dest


async def _local_tts(text: str, tmp_dir: Path, wpm: int) -> Optional[Path]:
    """macOS `say` or Linux `espeak-ng`. Text goes through a file, never argv, so it can't be parsed as flags."""
    text_file = tmp_dir / "narration.txt"
    text_file.write_text(text, encoding="utf-8")
    if shutil.which("say"):
        dest = tmp_dir / "local.aiff"
        cmd = ["say", "-r", str(wpm), "-f", str(text_file), "-o", str(dest)]
    elif shutil.which("espeak-ng"):
        dest = tmp_dir / "local.wav"
        cmd = ["espeak-ng", "-s", str(wpm), "-f", str(text_file), "-w", str(dest)]
    else:
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
    except (OSError, asyncio.TimeoutError) as exc:
        logger.warning("local TTS failed: %s", exc)
        return None
    if proc.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        logger.warning("local TTS failed: %s", stderr.decode(errors="replace")[-200:])
        return None
    return dest


def _write_silence(path: Path, duration_sec: float) -> None:
    sample_rate = 24000
    with wave.open(str(path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * int(sample_rate * duration_sec))


def _retrieve(task: asyncio.Task) -> None:
    """Marks a losing engine's error as seen, so asyncio does not log it as never retrieved."""
    if not task.cancelled():
        task.exception()


def _short(exc: BaseException) -> str:
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return message[:120]
