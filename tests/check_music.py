"""Offline checks for app/services/music.py (lane 5): the ducked mix on the sample files, the fail-soft compose with
fake fal responses, the final-cut wrapper, the argument dispatch and the note/log formats. The samples are made
with ffmpeg at start (two beds at different raw levels, one narrated demo).

    .venv/bin/python tests/check_music.py          # offline only (ffmpeg needed)
    LIVE=1 .venv/bin/python tests/check_music.py   # plus one real 40 s composition on fal.ai (FAL_KEY from .env)
"""
import asyncio
import base64
import os
import re
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the backend root (tests/..)
WORK = Path(tempfile.mkdtemp(prefix="demoday-music-"))
BED_MP3 = WORK / "bed.mp3"        # like fal stable-audio-3 small: 30 s stereo 44.1 kHz MP3
BED_FLAC = WORK / "bed.bin"       # like fal ace-step: FLAC under a .bin name (the container is probed, never inferred)
BED_LIVE = WORK / "bed-live.mp3"  # never made offline; test_mix runs on it when present
DEMO = WORK / "demo.mp4"          # a narrated 47.52 s demo (h264/aac): tone bursts with pauses stand in for the voice
os.environ.update({"STORAGE_DIR": str(WORK / "storage"), "MUSIC_LEVEL_DB": "", "FEATURE_MUSIC": ""})
sys.path.insert(0, BACKEND)

from app.models.schemas import JobStatus  # noqa: E402
from app.services import ledger, music  # noqa: E402
from app.services.stitcher import MediaToolError, describe_media, probe_duration  # noqa: E402

import logging  # noqa: E402

logging.getLogger("demoday.music").setLevel(logging.ERROR)  # the failure cases below log expected warnings
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail else ''}")


def ffprobe_video(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,nb_frames,bit_rate", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return out


def loudness(path):
    """Integrated loudness (LUFS) via ffmpeg's ebur128."""
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af", "ebur128=framelog=quiet",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    match = re.search(r"I:\s+(-?[\d.]+) LUFS", err)
    return float(match.group(1)) if match else None


def residual_db(mixed, original):
    """Mean level of (mixed - original): what the mix added. About -90 dB would mean no music at all."""
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(mixed), "-i", str(original), "-filter_complex",
                          "[0:a][1:a]amerge=inputs=2,pan=stereo|c0=c0-c2|c1=c1-c3,volumedetect", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    match = re.search(r"mean_volume:\s+(-?[\d.]+) dB", err)
    return float(match.group(1)) if match else None


class Log:
    def __init__(self):
        self.lines = []

    def __call__(self, message):
        self.lines.append(message)

    def has(self, prefix):
        return any(line.startswith(prefix) for line in self.lines)

    def find(self, prefix):
        return next((line for line in self.lines if line.startswith(prefix)), None)


def fake_fal(result=None, exc=None, delay=0.0):
    """A stand-in for the fal_client module: records the subscribe call and returns `result` (or raises `exc`)."""
    module = types.ModuleType("fal_client")
    module.calls = []

    class AsyncClient:
        def __init__(self, key=None, **kwargs):
            module.calls.append(("key", key))

        async def subscribe(self, model, arguments=None, **kwargs):
            module.calls.append((model, arguments, kwargs))
            if delay:
                await asyncio.sleep(delay)
            if exc is not None:
                raise exc
            return result

    module.AsyncClient = AsyncClient
    return module


def data_url(path, mime):
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


async def with_fake_fal(module, coro):
    real = sys.modules.get("fal_client")
    sys.modules["fal_client"] = module
    try:
        return await coro
    finally:
        if real is not None:
            sys.modules["fal_client"] = real
        else:
            sys.modules.pop("fal_client", None)


async def test_mix(bed, label):
    print(f"mix_under with the {label}")
    source = await describe_media(DEMO)
    out = WORK / f"demo_{label.replace(' ', '_')}.mp4"
    log = Log()
    t0 = time.monotonic()
    returned = await music.mix_under(DEMO, bed, out, log)
    elapsed = time.monotonic() - t0
    mixed = await describe_media(out)
    check("returns video_out and the file exists", returned == out and out.is_file(), f"{elapsed:.1f}s")
    check("no temp file left behind", not list(WORK.glob("*.mixing*")))
    check("video stream copied (same codec, frame count and bitrate)", ffprobe_video(out) == ffprobe_video(DEMO),
          f"{ffprobe_video(out)} vs {ffprobe_video(DEMO)}")
    check("same duration (<=0.2 s)", abs(mixed["duration"] - source["duration"]) <= 0.2,
          f"{mixed['duration']:.3f}s vs {source['duration']:.3f}s")
    check("audio and video durations match (<=100 ms, test_e2e's rule)",
          abs(mixed["video_duration"] - mixed["audio_duration"]) <= 0.1,
          f"video {mixed['video_duration']:.3f}s, audio {mixed['audio_duration']:.3f}s")
    check("audio is AAC 48 kHz stereo", mixed["audio_codec"] == "aac" and mixed["audio_rate"] == 48000
          and mixed["audio_channels"] == 2, f"{mixed['audio_codec']} {mixed['audio_rate']} {mixed['audio_channels']}ch")
    before, after = loudness(DEMO), loudness(out)
    check("integrated loudness within 1 LU of the input", before is not None and after is not None
          and abs(after - before) <= 1.0, f"{before} -> {after} LUFS")
    added = residual_db(out, DEMO)
    check("the music is really in there (mix minus original is not silence)", added is not None and added > -50,
          f"residual mean {added} dB")
    check("logs the mixed line", log.has("Soundtrack mixed under the narration ("), log.find("Soundtrack mixed"))
    return after


async def test_mix_failures():
    print("mix_under failures leave nothing behind")
    log = Log()
    try:
        await music.mix_under(DEMO, BED_MP3, DEMO, log)
        check("refuses to overwrite its own input", False)
    except MediaToolError as exc:
        check("refuses to overwrite its own input", True, exc)
    out = WORK / "never.mp4"
    try:
        await music.mix_under(DEMO, WORK / "missing.bin", out, log)
        check("a missing bed raises", False)
    except MediaToolError as exc:
        check("a missing bed raises", True, exc)
    check("no output or temp file on failure", not out.exists() and not list(WORK.glob("never.mixing*")))
    check("nothing logged on failure (the caller logs the skip)", not log.lines)


async def test_compose_offline():
    print("compose without a key / feature off")
    job = JobStatus(job_id="music-test", status="running")
    ledger.bind(job)
    saved = os.environ.pop("FAL_KEY", None)
    try:
        os.environ["FAL_KEY"] = ""
        log, skips = Log(), []
        track = await music.compose("warm pads", 40, WORK / "nokey", log, on_skip=skips.append)
        check("returns None without FAL_KEY", track is None)
        check("logs the brief's skip line", log.lines == ["Soundtrack skipped: FAL_KEY not set; the video has narration only"],
              log.lines)
        check("tells on_skip why", skips == ["FAL_KEY not set"], skips)
        check("no ledger row without a call", not job.tool_calls)
        os.environ["FEATURE_MUSIC"] = "0"
        os.environ["FAL_KEY"] = "test-key"
        log, skips = Log(), []
        track = await music.compose("warm pads", 40, WORK / "off", log, on_skip=skips.append)
        check("FEATURE_MUSIC=0 skips before any call", track is None and skips == ["FEATURE_MUSIC is off"]
              and log.lines == [music.skip_line("FEATURE_MUSIC is off")], log.lines)
        os.environ["FEATURE_MUSIC"] = ""

        print("compose with a fake fal.ai")
        flac_url = data_url(BED_FLAC, "audio/flac")
        shapes = {
            "audio_file.url": {"audio_file": {"url": flac_url}},
            "audio.url": {"audio": {"url": flac_url, "content_type": "audio/flac"}},
            "audio_url": {"audio_url": flac_url},
            "audios[0].url": {"audios": [{"url": flac_url}]},
        }
        for name, result in shapes.items():
            fake = fake_fal(result=result)
            log, skips = Log(), []
            job_dir = WORK / f"shape_{name}"
            track = await with_fake_fal(fake, music.compose("warm minimal electronic, soft pads, 95 bpm", 40, job_dir,
                                                            log, on_skip=skips.append))
            ok = track is not None and track == job_dir / "music.bin" and track.is_file()
            duration = await probe_duration(track) if ok else 0
            check(f"{name}: FLAC body lands untouched in music.bin ({duration:.1f}s)", ok and abs(duration - 30) < 0.5
                  and not skips)
        model, arguments, kwargs = fake.calls[-1]
        check("start line names the brief, the model and the requested length",
              log.find("Soundtrack: ") == f"Soundtrack: Gemini's brief 'warm minimal electronic, soft pads, 95 bpm' → "
                                          f"fal.ai {music.FAL_MUSIC_MODEL} (40s requested)", log.find("Soundtrack: "))
        check("ready line reports the measured length and latency",
              re.fullmatch(r"Soundtrack ready: 30\.0s of music in \d+\.\ds", log.find("Soundtrack ready") or ""),
              log.find("Soundtrack ready"))
        check("fal receives the probe's keys", model == music.FAL_MUSIC_MODEL and arguments["duration"] == 40
              and arguments["output_format"] == "mp3" and arguments["num_inference_steps"] == 50
              and kwargs.get("client_timeout") == music.MUSIC_TIMEOUT_SEC, arguments)
        check("the prompt carries the instrumental suffix",
              arguments["prompt"] == "warm minimal electronic, soft pads, 95 bpm. " + music.PROMPT_SUFFIX, arguments["prompt"])
        check("the key reaches the client", ("key", "test-key") in fake.calls)
        rows = [r for r in job.tool_calls if r.purpose == "soundtrack"]
        check("ledger rows for every composition", len(rows) == 4 and all(r.ok and r.ms is not None for r in rows)
              and rows[0].tool == "fal" and rows[0].model == music.FAL_MUSIC_MODEL, [(r.ok, r.ms) for r in rows])

        log, skips = Log(), []
        fake = fake_fal(result={"audio": {"url": data_url(BED_MP3, "audio/mpeg")}})
        track = await with_fake_fal(fake, music.compose(None, 200, WORK / "template", log, voice_style="calm_expert",
                                                        on_skip=skips.append))
        check("template brief when Gemini wrote none, length capped at 120",
              track is not None and log.find("Soundtrack: ") == f"Soundtrack: template brief "
              f"'{music.TEMPLATE_MUSIC['calm_expert']}' → fal.ai {music.FAL_MUSIC_MODEL} (120s requested)"
              and fake.calls[-1][1]["duration"] == 120, log.find("Soundtrack: "))

        print("compose failures are soft")
        short = WORK / "short.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                        str(short)], check=True)
        cases = [
            ("a 1 s clip", fake_fal(result={"audio": {"url": data_url(short, "audio/wav")}}), "fal.ai returned only 1.0s of audio"),
            ("a fal error", fake_fal(exc=RuntimeError("Unprocessable Entity: duration must be <= 120")),
             "Unprocessable Entity: duration must be <= 120"),
            ("an unexpected response", fake_fal(result={"seed": 1}), "unexpected fal.ai response: {'seed': 1}"),
        ]
        for name, fake, reason in cases:
            log, skips = Log(), []
            job_dir = WORK / f"fail_{name.replace(' ', '_')}"
            track = await with_fake_fal(fake, music.compose("x", 40, job_dir, log, on_skip=skips.append))
            check(f"{name}: None, skip line, on_skip, no file", track is None and skips == [reason]
                  and log.lines[-1] == music.skip_line(reason) and not (job_dir / "music.bin").exists(),
                  (skips, log.lines[-1:]))
        bad = [r for r in job.tool_calls if r.purpose == "soundtrack" and r.ok is False]
        check("failed calls are ledger fallbacks with the reason", len(bad) == 3 and all(r.note for r in bad),
              [r.note for r in bad])
    finally:
        if saved is None:
            os.environ.pop("FAL_KEY", None)
        else:
            os.environ["FAL_KEY"] = saved
        os.environ["FEATURE_MUSIC"] = ""
        ledger.bind(None)


async def test_finish():
    print("finish (the final cut)")

    def fresh_raw(name):
        raw = WORK / f"{name}_raw.mp4"
        raw.write_bytes(DEMO.read_bytes())
        return raw, WORK / f"{name}.mp4"

    raw, final = fresh_raw("no_task")
    log, skips = Log(), []
    note = await music.finish(None, raw, final, log, on_skip=skips.append)
    check("no task: raw renamed to demo.mp4, no note, nothing logged", note is None and final.is_file()
          and not raw.exists() and not log.lines and not skips)

    raw, final = fresh_raw("ready")
    log, skips = Log(), []
    task = asyncio.create_task(asyncio.sleep(0.1, result=BED_MP3))
    t0 = time.monotonic()
    note = await music.finish(task, raw, final, log, grace=5, on_skip=skips.append)
    check("ready track: mixed into demo.mp4 and the note returned",
          note == "stable-audio-3/small · 30 s track · mixed at -14 dB, ducked under the voice" and final.is_file()
          and raw.is_file() and not skips, f"{note!r} in {time.monotonic() - t0:.1f}s")
    mixed = await describe_media(final)
    check("the mixed demo.mp4 is valid", mixed["video_codec"] == "h264" and mixed["audio_codec"] == "aac"
          and abs(mixed["duration"] - 47.52) < 0.2)

    raw, final = fresh_raw("late")
    log, skips = Log(), []
    task = asyncio.create_task(asyncio.sleep(5, result=BED_MP3))
    t0 = time.monotonic()
    note = await music.finish(task, raw, final, log, grace=0.5, on_skip=skips.append)
    elapsed = time.monotonic() - t0
    check("late track: gives up after the grace, cancels the task, ships the raw cut",
          note is None and elapsed < 2 and task.cancelled() and final.is_file() and not raw.exists(), f"{elapsed:.1f}s")
    check("late track: the brief's skip line and the amber reason",
          log.lines == [music.skip_line("fal.ai did not deliver within 0s of the final cut")]
          and skips == ["fal.ai did not deliver within 0s of the final cut"], (log.lines, skips))

    raw, final = fresh_raw("skipped")
    log, skips = Log(), []
    task = asyncio.create_task(asyncio.sleep(0, result=None))
    note = await music.finish(task, raw, final, log, on_skip=skips.append)
    check("compose returned None: raw shipped, nothing reported twice", note is None and final.is_file()
          and not log.lines and not skips)

    raw, final = fresh_raw("badbed")
    log, skips = Log(), []
    task = asyncio.create_task(asyncio.sleep(0, result=WORK / "not-audio.bin"))
    (WORK / "not-audio.bin").write_bytes(b"\x00" * 1000)
    note = await music.finish(task, raw, final, log, on_skip=skips.append)
    check("a broken bed: mix failure is soft, raw shipped, amber reason", note is None and final.is_file()
          and not raw.exists() and len(skips) == 1 and skips[0].startswith("the mix failed (")
          and log.lines[-1] == music.skip_line(skips[0]), skips)

    raw, final = fresh_raw("crashed")
    log, skips = Log(), []

    async def boom():
        raise ValueError("compose bug")

    note = await music.finish(asyncio.create_task(boom()), raw, final, log, on_skip=skips.append)
    check("a crashing task is soft too", note is None and final.is_file() and skips == ["compose bug"], skips)


def test_helpers():
    print("helpers")
    check("bed_seconds_for: target + 10, capped at 120, floor 5",
          [music.bed_seconds_for(t) for t in (30, 60, 110, 115, 120, 1)] == [40, 70, 120, 120, 120, 11])
    check("level_db default", music.level_db() == -14.0)
    os.environ["MUSIC_LEVEL_DB"] = "-20"
    check("MUSIC_LEVEL_DB knob", music.level_db() == -20.0)
    os.environ["MUSIC_LEVEL_DB"] = "loud"
    check("garbage MUSIC_LEVEL_DB falls back", music.level_db() == -14.0)
    os.environ["MUSIC_LEVEL_DB"] = "5"
    check("MUSIC_LEVEL_DB is clamped to <= 0 dB", music.level_db() == 0.0)
    os.environ["MUSIC_LEVEL_DB"] = ""
    check("model_label", music.model_label("fal-ai/stable-audio-3/small/music/base/text-to-audio") == "stable-audio-3/small"
          and music.model_label("fal-ai/ace-step-1.5") == "ace-step-1.5")
    check("describe note", music.describe(40.2) == "stable-audio-3/small · 40 s track · mixed at -14 dB, ducked under the voice",
          music.describe(40.2))
    check("skip and late lines", music.skip_line("x") == "Soundtrack skipped: x; the video has narration only"
          and music.late_reason(12) == "fal.ai did not deliver within 12s of the final cut")
    check("brief_for: Gemini's prompt wins, template otherwise",
          music.brief_for("  'warm  pads'  ", "fast_founder") == ("warm pads", "Gemini's brief")
          and music.brief_for(None, "fast_founder") == (music.TEMPLATE_MUSIC["fast_founder"], "template brief")
          and music.brief_for("", "unknown") == (music.TEMPLATE_MUSIC["energetic_tech"], "template brief"))
    stable = music._music_arguments("fal-ai/stable-audio-3/small/music/base/text-to-audio", "p", 40)
    ace = music._music_arguments("fal-ai/ace-step-1.5", "p", 40)
    other = music._music_arguments("fal-ai/some-music-model", "p", 40)
    check("argument dispatch per model family",
          stable == {"prompt": "p", "duration": 40, "output_format": "mp3", "num_inference_steps": 50,
                     "negative_prompt": music.NEGATIVE_PROMPT}
          and ace == {"prompt": "p", "duration": 40, "lyrics": "[inst]"} and other == {"prompt": "p", "duration": 40})
    check("_audio_url shapes", [music._audio_url(x) for x in (
        {"audio_file": {"url": "a"}}, {"audio": {"url": "b"}}, {"audio_url": "c"}, {"audios": [{"url": "d"}]},
        {"audio": "e"}, {"seed": 1}, None)] == ["a", "b", "c", "d", "e", None, None])


async def test_live():
    print("LIVE: one real 40 s composition on fal.ai")
    from app.config import get_secret
    if not get_secret("FAL_KEY"):
        check("FAL_KEY configured for the live check", False)
        return
    job = JobStatus(job_id="music-live", status="running")
    ledger.bind(job)
    log = Log()
    job_dir = WORK / "live"
    t0 = time.monotonic()
    track = await music.compose("warm minimal electronic, soft synth pads, light percussion, 95 bpm, optimistic", 40,
                                job_dir, log)
    latency = time.monotonic() - t0
    for line in log.lines:
        print("    log:", line)
    row = next((r for r in job.tool_calls if r.purpose == "soundtrack"), None)
    check(f"fal.ai composed a track ({latency:.1f}s wall, ledger {row.ms if row else '?'} ms)", track is not None)
    if track is None:
        return
    info = await describe_media(track)
    duration = await probe_duration(track)
    check("about 40 s of audio", abs(duration - 40) <= 2, f"{duration:.1f}s {info['audio_codec']} "
          f"{info['audio_rate']} Hz {info['audio_channels']}ch, {track.stat().st_size / 1e6:.2f} MB")
    out = WORK / "live_mix.mp4"
    t0 = time.monotonic()
    await music.mix_under(DEMO, track, out, log)
    check("mixed under the sample demo", out.is_file(), f"{time.monotonic() - t0:.1f}s; "
          f"loudness {loudness(DEMO)} -> {loudness(out)} LUFS; listen: {out}")
    ledger.bind(None)


def make_samples():
    """The beds differ in timbre and raw level (the mix normalizes them); the demo has pauses for the ducking."""
    def run(*args):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)

    chord = "0.25*sin(2*PI*220*t)+0.2*sin(2*PI*277.2*t)+0.15*sin(2*PI*329.6*t)"
    run("-f", "lavfi", "-i", f"aevalsrc='{chord}|{chord}':s=44100:d=30", "-c:a", "libmp3lame", "-b:a", "192k", str(BED_MP3))
    pulse = f"0.45*({chord})*(0.6+0.4*sin(2*PI*0.5*t))"
    run("-f", "lavfi", "-i", f"aevalsrc='{pulse}|{pulse}':s=48000:d=30", "-c:a", "flac", "-f", "flac", str(BED_FLAC))
    voice = "0.5*sin(2*PI*180*t)*(0.6+0.4*sin(2*PI*4*t))*lt(mod(t,3),2.4)"
    run("-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=47.52", "-f", "lavfi", "-i", f"aevalsrc='{voice}':s=48000:d=47.52",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-ac", "2", "-shortest", str(DEMO))


async def main():
    make_samples()
    levels = [await test_mix(BED_MP3, "mp3 bed"), await test_mix(BED_FLAC, "flac bed in a .bin")]
    if BED_LIVE.is_file():
        levels.append(await test_mix(BED_LIVE, "louder live 40 s bed"))
    print("bed level is deterministic across generations")
    check("every bed lands the mix within 0.5 LU of the others", all(levels) and max(levels) - min(levels) <= 0.5,
          f"{levels} LUFS")
    await test_mix_failures()
    await test_compose_offline()
    await test_finish()
    test_helpers()
    if os.getenv("LIVE") == "1":
        await test_live()
    else:
        print("LIVE=1 not set: skipping the real fal.ai composition")
    print(f"\n{sum(results)}/{len(results)} checks passed (work dir {WORK})")
    return 0 if all(results) else 1


sys.exit(asyncio.run(main()))
