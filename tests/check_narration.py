"""Offline checks for app/services/audio.py: one voice per video when fal.ai's Kokoro queue is cold.

Fake engines with set delays and failures stand in for fal.ai, Gemini TTS and Google Cloud TTS; each writes a tone of
its own length, so the measured WAV says which engine voiced a line. Covers the hedge (Gemini joins after
TTS_HEDGE_SEC, at once when fal fails a line), the grace that lets fal land first, one engine for every line, the
salvage when no engine voices them all, the job voice shared by later batches, timeouts, the engine list, the
Gemini TTS request (plain text, v1beta, the preset's voice) and its audio formats.

    .venv/bin/python tests/check_narration.py          # offline only (ffmpeg needed)
    LIVE=1 .venv/bin/python tests/check_narration.py   # plus one real Gemini TTS line (GEMINI_API_KEY from .env)
"""
import asyncio
import array
import math
import os
import sys
import tempfile
import time
import types
import wave
from pathlib import Path

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the backend root (tests/..)
WORK = Path(tempfile.mkdtemp(prefix="demoday-narration-"))
LIVE_KEY = ""
if os.getenv("LIVE") == "1":
    from dotenv import dotenv_values

    LIVE_KEY = dotenv_values(os.path.join(BACKEND, ".env")).get("GEMINI_API_KEY") or ""
for name in ("FAL_KEY", "GEMINI_API_KEY", "GOOGLE_TTS", "FEATURE_GEMINI_TTS", "TTS_HEDGE_SEC", "FAL_TTS_GRACE_SEC"):
    os.environ.pop(name, None)
os.environ["STORAGE_DIR"] = str(WORK / "storage")
sys.path.insert(0, BACKEND)

from app.models.schemas import JobStatus  # noqa: E402
from app.services import audio, ledger  # noqa: E402
from app.services.stitcher import probe_duration  # noqa: E402

import logging  # noqa: E402

logging.getLogger("demoday.audio").setLevel(logging.ERROR)  # the failure cases below log expected warnings
results = []
SECONDS = {"fal": 1.0, "gemini": 1.6, "google": 2.2}  # each engine's tone length: the WAV names its engine
LABELS = {"fal": "fal.ai Kokoro (af_heart, 1.1x)", "gemini": "Gemini TTS (gemini-3.8-flash-tts, Aoede)",
          "google": "Google Cloud TTS (en-US-Chirp3-HD-Aoede)"}
SHORT = {"fal": "fal.ai", "gemini": "Gemini TTS", "google": "Google TTS"}
CALLS, CANCELLED = [], []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail else ''}")


def tone(path: Path, seconds: float, hz: float = 440.0, rate: int = 24000) -> None:
    samples = array.array("h", (int(6000 * math.sin(2 * math.pi * hz * n / rate)) for n in range(int(rate * seconds))))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(samples.tobytes())


def fake(key, delay=0.05, delays=None, fail=None, timeout=30.0):
    """An engine that waits `delay` (or delays[line]) and writes its tone; fail maps a line to its error message."""
    fail = fail or {}

    async def run(text, dest):
        line = int(dest.name.rsplit("_", 1)[-1])
        CALLS.append((key, line, time.monotonic()))
        try:
            await asyncio.sleep(delays[line] if delays else delay)
        except asyncio.CancelledError:
            CANCELLED.append((key, line))
            raise
        if line in fail:
            raise RuntimeError(fail[line])
        path = dest.with_suffix(".wav")
        tone(path, SECONDS[key])
        return path

    return audio._Engine(key, LABELS[key], SHORT[key], timeout, run)


def engine_of(duration):
    return next((key for key, seconds in SECONDS.items() if abs(duration - seconds) < 0.06), f"? ({duration:.2f}s)")


async def batch(name, engines, lines=3):
    """Voices `lines` lines with the given engines; returns (engine per line, label per line, seconds, calls)."""
    CALLS.clear()
    CANCELLED.clear()
    audio._engines = lambda voice_style: list(engines)
    out = WORK / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    voiced = await audio.synthesize_narration([f"Line {i} of {name}." for i in range(lines)],
                                              [out / f"line_{i:02d}.wav" for i in range(lines)])
    took = time.monotonic() - t0
    return [engine_of(d) for d, _ in voiced], [label for _, label in voiced], took, [(k, l, t - t0) for k, l, t in CALLS]


async def test_race():
    print("hedged race, one engine per batch")
    audio.TTS_HEDGE_SEC, audio.FAL_TTS_GRACE_SEC = 0.3, 0.3

    used, labels, took, calls = await batch("warm", [fake("fal"), fake("gemini")])
    check("a warm fal.ai voices every line", used == ["fal"] * 3, used)
    check("Gemini TTS never starts when fal.ai answers within the head start",
          not any(k == "gemini" for k, _, _ in calls), calls)
    check("the label names Kokoro and nothing else", labels == [LABELS["fal"]] * 3, labels[0])

    used, labels, took, calls = await batch("cold", [fake("fal", delay=3.0), fake("gemini")])
    gemini_start = min((t for k, _, t in calls if k == "gemini"), default=None)
    check("a cold fal.ai: Gemini TTS joins after the head start, not before",
          gemini_start is not None and 0.25 <= gemini_start < 0.6, f"{gemini_start:.2f}s" if gemini_start else calls)
    check("Gemini TTS then voices every line, so the voice never changes", used == ["gemini"] * 3, used)
    check("the batch takes the head start plus the grace, not fal's timeout", took < 1.5, f"{took:.2f}s")
    check("the label says fal.ai did not answer", all(l.startswith("fal.ai no answer within ") and
                                                     l.endswith("→ " + LABELS["gemini"]) for l in labels), labels[0])
    check("fal.ai's unfinished lines are cancelled", sorted(l for k, l in CANCELLED if k == "fal") == [0, 1, 2],
          CANCELLED)

    used, labels, _, _ = await batch("grace", [fake("fal", delay=0.5), fake("gemini")])
    check("fal.ai landing within the grace still wins (the preferred voice)", used == ["fal"] * 3, used)
    check("…and its label has no miss in it", labels == [LABELS["fal"]] * 3, labels[0])

    used, labels, _, calls = await batch("fal-fails",
                                         [fake("fal", delays=[0.05, 0.05, 0.1], fail={2: "HTTP 503 from fal"}),
                                          fake("gemini")])
    gemini_start = min((t for k, _, t in calls if k == "gemini"), default=None)
    check("a line fal.ai fails starts Gemini TTS at once, without waiting for the head start",
          gemini_start is not None and gemini_start < 0.22, f"{gemini_start:.2f}s" if gemini_start else calls)
    check("Gemini TTS voices all three lines, not only the failed one", used == ["gemini"] * 3, used)
    check("the label carries fal.ai's error", labels[0] == "fal.ai HTTP 503 from fal → " + LABELS["gemini"], labels[0])

    used, labels, _, calls = await batch("three", [fake("fal", delay=3.0), fake("gemini"), fake("google")])
    check("a fallback that voiced every line needs no third engine",
          used == ["gemini"] * 3 and not any(k == "google" for k, _, _ in calls), (used, calls))

    used, labels, _, _ = await batch("gemini-fails", [fake("fal", delay=3.0), fake("gemini", fail={0: "429"}),
                                                      fake("google")])
    check("when Gemini TTS fails a line too, Google Cloud TTS voices them all", used == ["google"] * 3, used)

    used, labels, _, _ = await batch("salvage", [fake("fal", fail={0: "HTTP 500"}), fake("gemini", fail={1: "429"})])
    check("no engine voices every line: each line keeps what one engine made, fal.ai first",
          used == ["gemini", "fal", "fal"], used)
    check("salvaged lines say the voices differ", all("other lines in another voice" in l for l in labels), labels[0])

    used, labels, _, _ = await batch("all-fail", [fake("fal", fail={0: "a", 1: "a", 2: "a"}),
                                                  fake("gemini", fail={0: "b", 1: "b", 2: "b"})], lines=2)
    check("every engine failing every line falls back to local TTS or silence, per line",
          all("fal.ai a → Gemini TTS b → " in l and ("local TTS fallback" in l or "silent placeholder" in l)
              for l in labels), labels[0])

    used, labels, _, _ = await batch("no-engines", [], lines=1)
    check("no engine configured: local TTS or silence, with no miss in the label",
          labels[0].startswith("local TTS fallback") or labels[0].startswith("silent placeholder"), labels[0])

    audio._ENGINE_MARGIN_SEC, audio.TTS_HEDGE_SEC = 0, 30
    used, labels, took, _ = await batch("timeout", [fake("fal", delay=5.0, timeout=0.4), fake("gemini")], lines=2)
    check("a fal.ai timeout counts as a failure and Gemini TTS takes over before the head start",
          used == ["gemini"] * 2 and took < 1.5, (used, f"{took:.2f}s"))
    check("the timeout reads as no answer within the engine's timeout",
          labels[0] == "fal.ai no answer within 0s → " + LABELS["gemini"], labels[0])
    audio._ENGINE_MARGIN_SEC, audio.TTS_HEDGE_SEC = 5, 0.3


async def test_job_voice():
    print("one voice per job")
    audio.TTS_HEDGE_SEC, audio.FAL_TTS_GRACE_SEC = 0.3, 0.3

    async def job():
        audio.bind_job_voice()
        first, _, _, _ = await batch("job-1", [fake("fal", delay=3.0), fake("gemini")], lines=2)
        second, labels, _, calls = await batch("job-2", [fake("fal"), fake("gemini")], lines=2)
        return first, second, labels, calls

    first, second, labels, calls = await asyncio.create_task(job())
    check("the first batch (a cold fal.ai) is voiced by Gemini TTS", first == ["gemini"] * 2, first)
    check("a later batch of the same job keeps Gemini TTS though fal.ai is warm now",
          second == ["gemini"] * 2 and not any(k == "fal" for k, _, _ in calls), (second, calls))
    check("…without a miss in its label", labels[0] == LABELS["gemini"], labels[0])

    used, _, _, _ = await batch("no-job", [fake("fal"), fake("gemini")], lines=2)
    check("outside a job each batch starts from fal.ai again", used == ["fal"] * 2, used)

    async def concurrent():
        audio.bind_job_voice()
        audio._engines = lambda voice_style: [fake("fal", delay=3.0), fake("gemini")]
        out = WORK / "concurrent"
        out.mkdir(exist_ok=True)
        intro, scenes = await asyncio.gather(
            audio.synthesize_narration(["Intro."], [out / "intro.wav"]),
            audio.synthesize_narration(["One.", "Two."], [out / "s1.wav", out / "s2.wav"]))
        return [engine_of(d) for d, _ in intro + scenes]

    CALLS.clear()
    used = await asyncio.create_task(concurrent())
    check("an intro and the scenes voiced at the same time end on one engine", len(set(used)) == 1, used)

    single = WORK / "single"
    single.mkdir(exist_ok=True)
    audio._engines = lambda voice_style: [fake("fal")]
    duration, label = await audio.synthesize_scene_audio("One line.", single / "one.wav")
    check("synthesize_scene_audio still returns (duration, engine) for one line",
          engine_of(duration) == "fal" and label == LABELS["fal"], (duration, label))


def test_voice_tool():
    print("the trace's tool")
    check("Kokoro lines are fal", audio.voice_tool([LABELS["fal"]] * 2) == "fal")
    check("a miss then Gemini TTS is gemini",
          audio.voice_tool(["fal.ai no answer within 20s → " + LABELS["gemini"]] * 2) == "gemini")
    check("mixed engines have no single tool", audio.voice_tool([LABELS["fal"], LABELS["gemini"]]) is None)
    check("local fallbacks are local", audio.voice_tool(["fal.ai x → local TTS fallback (macOS say)"]) == "local")


def test_engines():
    print("the engine list")
    real = ENGINES
    os.environ.update({"FAL_KEY": "fal-test", "GEMINI_API_KEY": "gemini-test"})
    check("fal.ai first, Gemini TTS second", [e.key for e in real("energetic_tech")] == ["fal", "gemini"])
    check("Gemini TTS uses the preset's voice",
          [e.label for e in real("calm_expert")][1] == f"Gemini TTS ({audio.GEMINI_TTS_MODEL}, Charon)")
    os.environ["FEATURE_GEMINI_TTS"] = "0"
    check("FEATURE_GEMINI_TTS=0 leaves fal.ai alone", [e.key for e in real("energetic_tech")] == ["fal"])
    os.environ.pop("FEATURE_GEMINI_TTS")
    os.environ["GOOGLE_TTS"] = "true"
    check("GOOGLE_TTS=true adds Google Cloud TTS last",
          [e.key for e in real("energetic_tech")] == ["fal", "gemini", "google"])
    for name in ("FAL_KEY", "GEMINI_API_KEY", "GOOGLE_TTS"):
        os.environ.pop(name)
    check("no keys: no engines", real("energetic_tech") == [])


async def test_gemini_request():
    print("the Gemini TTS request")
    import google.genai as genai_module

    seen = {}
    pcm = array.array("h", [0, 1000, -1000] * 8000).tobytes()

    class FakeClient:
        def __init__(self, api_key=None, http_options=None):
            seen["key"], seen["options"] = api_key, http_options

            async def generate_content(model, contents, config):
                seen.update(model=model, contents=contents, config=config)
                part = types.SimpleNamespace(inline_data=types.SimpleNamespace(data=pcm,
                                                                                mime_type="audio/L16;codec=pcm;rate=24000"))
                return types.SimpleNamespace(candidates=[types.SimpleNamespace(content=types.SimpleNamespace(parts=[part]))])

            async def aclose():
                seen["closed"] = True

            self.aio = types.SimpleNamespace(models=types.SimpleNamespace(generate_content=generate_content),
                                             aclose=aclose)

    original = genai_module.Client
    genai_module.Client = FakeClient
    job = JobStatus(job_id="narration01", status="running", stage="recording", progress=0, message="")
    ledger.bind(job)
    try:
        dest = WORK / "gemini_00"
        path = await audio._gemini_tts("Warestack checks every pull request.", dest, "Charon", "gemini-test")
    finally:
        genai_module.Client = original
        ledger.bind(None)
    voice = seen["config"].speech_config.voice_config.prebuilt_voice_config.voice_name
    check("the text is sent as is, with no style direction to be read aloud",
          seen["contents"] == "Warestack checks every pull request.", seen["contents"])
    check("v1beta (v1 has no speech_config) with retries on 429 and 5xx",
          seen["options"].api_version == "v1beta" and 429 in seen["options"].retry_options.http_status_codes)
    check("audio only, in the preset's voice", seen["config"].response_modalities == ["AUDIO"] and voice == "Charon")
    check("the model is GEMINI_TTS_MODEL", seen["model"] == audio.GEMINI_TTS_MODEL, seen["model"])
    with wave.open(str(path)) as wav_file:
        frames, rate = wav_file.getnframes(), wav_file.getframerate()
    check("raw L16 PCM becomes a 24 kHz WAV", path.suffix == ".wav" and rate == 24000 and frames == len(pcm) // 2,
          (path.name, rate, frames))
    check("the client is closed", seen.get("closed") is True)
    check("the ledger records one Gemini narration call",
          [(c.tool, c.purpose, c.ok) for c in job.tool_calls] == [("gemini", "narration", True)],
          [(c.tool, c.purpose, c.ok) for c in job.tool_calls])

    riff = WORK / "riff.wav"
    tone(riff, 0.2)
    copy = audio.write_tts_audio(riff.read_bytes(), "audio/wav", WORK / "copy.wav")
    check("a WAV answer is kept byte for byte", copy.read_bytes() == riff.read_bytes())
    try:
        audio.write_tts_audio(b"", "audio/wav", WORK / "empty.wav")
        check("an empty answer is an error", False)
    except RuntimeError:
        check("an empty answer is an error", True)


async def test_live():
    print("LIVE: one real Gemini TTS line")
    path = await audio._gemini_tts("Warestack checks every pull request against your rules before it merges.",
                                   WORK / "live_00", "Charon", LIVE_KEY)
    duration = await probe_duration(path)
    check("Gemini TTS returns a few seconds of speech", 2.0 < duration < 12.0, f"{duration:.2f}s")


ENGINES = audio._engines


async def main():
    test_engines()
    test_voice_tool()
    await test_gemini_request()
    await test_race()
    await test_job_voice()
    if LIVE_KEY:
        await test_live()
    print(f"\n{sum(results)}/{len(results)} checks passed (work dir {WORK})")
    return 0 if all(results) else 1


sys.exit(asyncio.run(main()))
