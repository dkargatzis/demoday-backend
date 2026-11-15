"""Narration synthesis: fal.ai Kokoro TTS, with local TTS and silence fallbacks.

Every engine's output is normalized to 48 kHz stereo PCM WAV, and the returned duration is measured
with ffprobe so the browser recording can be timed to the real narration length.
"""
import asyncio
import os
import base64
import logging
import shutil
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional, Tuple

import httpx

from app.services import ledger
from app.config import get_secret
from app.services import cloud
from app.services.stitcher import AUDIO_RATE, probe_duration, run_ffmpeg

logger = logging.getLogger("demoday.audio")

# fal.ai's PlayHT endpoints were retired (404); Kokoro is fal-hosted. Schema: prompt/voice/speed -> audio.url
FAL_TTS_ENDPOINT = "fal-ai/kokoro/american-english"
# How long to wait for fal.ai per clip before falling back (Google Cloud TTS / local). fal's Kokoro queue can take
# 1-2 minutes to spin a worker up when it has gone cold; warm_up_fal() starts that early in the job.
FAL_TTS_TIMEOUT_SEC = int(os.getenv("FAL_TTS_TIMEOUT_SEC", "90"))
FAL_TTS_GRACE_SEC = int(os.getenv("FAL_TTS_GRACE_SEC", "20"))  # once Google TTS is ready, how much longer fal may take

# voice_style -> (Kokoro voice, Kokoro speed, local TTS words-per-minute)
VOICE_PRESETS = {
    "energetic_tech": ("af_heart", 1.1, 195),
    "calm_expert": ("am_michael", 0.95, 170),
    "fast_founder": ("af_bella", 1.2, 220),
}


async def synthesize_scene_audio(text: str, output_path, voice_style: str = "energetic_tech") -> Tuple[float, str]:
    """Synthesizes `text` into a normalized WAV at `output_path`.

    Returns (exact duration in seconds via ffprobe, description of the engine that produced it).
    """
    voice, speed, wpm = VOICE_PRESETS.get(voice_style, VOICE_PRESETS["energetic_tech"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=output_path.parent) as tmp:
        tmp_dir = Path(tmp)
        raw, engine = await _race_tts(text, tmp_dir, voice, speed, voice_style)
        if raw is None:
            raw = await _local_tts(text, tmp_dir, wpm)
            if raw is not None:
                engine += "local TTS fallback (macOS say)" if raw.suffix == ".aiff" else "local TTS fallback (espeak-ng)"
        if raw is None:
            raw = tmp_dir / "silence.wav"
            _write_silence(raw, max(2.5, len(text.split()) / 2.5))
            engine += "silent placeholder fallback (no TTS engine available)"

        # EBU R128 loudness normalization: every engine and every scene lands at the same voiceover level.
        await run_ffmpeg(["-i", str(raw), "-af", "loudnorm=I=-16:LRA=11:TP=-1.5",
                          "-ar", str(AUDIO_RATE), "-ac", "2", "-c:a", "pcm_s16le", str(output_path)])

    return await probe_duration(output_path), engine


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


async def _race_tts(text: str, tmp_dir: Path, voice: str, speed: float, voice_style: str) -> Tuple[Optional[Path], str]:
    """fal.ai Kokoro and Google Cloud TTS (Chirp 3 HD, runtime credentials) run side by side.

    fal wins whenever it answers first, or within FAL_TTS_GRACE_SEC of Google; a slow fal queue therefore costs
    seconds, not the full FAL_TTS_TIMEOUT_SEC. Returns (audio path or None, a description of what happened).
    """
    fal_key = get_secret("FAL_KEY")
    fal_task = asyncio.create_task(_fal_tts(text, tmp_dir, voice, speed, fal_key)) if fal_key else None
    google_task = (asyncio.create_task(cloud.google_tts(text, tmp_dir / "google.wav", voice_style))
                   if cloud.google_tts_enabled() else None)
    google_name = cloud.GOOGLE_VOICES.get(voice_style, cloud.GOOGLE_VOICES["energetic_tech"])[0]
    started = time.monotonic()
    note = ""
    if fal_task is not None:
        deadline = started + FAL_TTS_TIMEOUT_SEC
        while not fal_task.done() and time.monotonic() < deadline:
            pending = {task for task in (fal_task, google_task) if task is not None and not task.done()}
            await asyncio.wait(pending, timeout=deadline - time.monotonic(), return_when=asyncio.FIRST_COMPLETED)
            if google_task is not None and google_task.done() and not fal_task.done() and google_task.exception() is None:
                grace = max(0.0, min(FAL_TTS_GRACE_SEC, deadline - time.monotonic()))
                try:
                    await asyncio.wait_for(asyncio.shield(fal_task), grace)
                except (asyncio.TimeoutError, Exception):
                    pass
                break
        if fal_task.done() and not fal_task.cancelled() and fal_task.exception() is None:
            if google_task is not None:
                google_task.cancel()
            return fal_task.result(), f"fal.ai Kokoro ({voice}, {speed}x)"
        if fal_task.done() and not fal_task.cancelled():
            exc = fal_task.exception()
            logger.warning("fal.ai TTS failed: %s", exc)
            note = f"fal.ai {_short(exc)} → "
        else:
            fal_task.cancel()
            note = f"fal.ai no answer within {time.monotonic() - started:.0f}s, its queue is slow → "
    if google_task is not None:
        try:
            return await google_task, note + f"Google Cloud TTS ({google_name})"
        except Exception as exc:
            logger.warning("Google TTS failed: %s", exc)
            note += f"Google TTS unavailable ({_short(exc)}) → "
    return None, note


async def _fal_tts(text: str, tmp_dir: Path, voice: str, speed: float, key: str) -> Path:
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

        dest = tmp_dir / "fal_audio"
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


def _short(exc: Exception) -> str:
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return message[:120]
