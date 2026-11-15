"""Offline checks for the joins between scenes: click-through on camera, page loads held off screen, dissolves.

A fake app on 127.0.0.1:8781 has a full-screen magenta loading screen on every page change (a server page that boots
behind it, and a client-side tab that shows it like a Next.js loading.tsx), so a loading frame in a clip is easy to
spot. Three scenes are recorded in real Chromium: one opens on the home page, one clicks through to /rules on camera
(and opens a tab there), one has to open /teams off camera. Then the clips are muxed: no magenta frame survives the
holds (and plenty do without them), the dissolve starts on the previous clip's last frame, every clip keeps its
exact length, and the trace line says how the scenes joined.

    ALLOW_PRIVATE_URLS=true .venv/bin/python tests/check_transitions.py
"""
import asyncio
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the backend root (tests/..)
WORK = Path(tempfile.mkdtemp(prefix="demoday-transitions-"))
sys.path.insert(0, BACKEND)
os.environ.update({"STORAGE_DIR": str(WORK / "storage"), "ALLOW_PRIVATE_URLS": "true"})

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402

from app.models.schemas import Action, Scene  # noqa: E402
from app.services import orchestrator  # noqa: E402
from app.services.recorder import record_session  # noqa: E402
from app.services.stitcher import (DISSOLVE_SEC, FPS, concat_scenes, freeze_frames, last_frame,  # noqa: E402
                                   mux_scene_clip, probe_duration)

results = []
SITE = "http://127.0.0.1:8781"
LOADING = ('<div id="boot" class="loading-screen" style="position:fixed;inset:0;background:#ff00ff;z-index:50">'
           '<div class="animate-spin" style="width:40px;height:40px;margin:500px auto;border:4px solid #fff;'
           'border-radius:50%"></div></div>')


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + str(detail)[:400] if detail != '' else ''}")


def page(body: str) -> HTMLResponse:
    style = ("body{margin:0;font:28px sans-serif;background:#fff}nav a{display:inline-block;margin:24px;"
             "padding:16px 28px;background:#e5f6ef;color:#065f46}main{padding:24px}")
    return HTMLResponse(f"<!doctype html><html><head><title>Fake</title><style>{style}</style></head>"
                        f"<body>{body}</body></html>")


site = FastAPI()


@site.get("/")
async def home():
    return page('<nav><a href="/">Overview</a><a href="/rules">Rules</a></nav>'
                "<main><h1>Overview</h1>" + "<p>Pull requests checked today.</p>" * 12 + "</main>")


@site.get("/rules")
async def rules():
    # Boots behind a loading screen for 900 ms; its Scope tab is a client-side route with its own 700 ms loading screen.
    return page(LOADING + '<nav><a href="/">Overview</a><a href="/rules">Rules</a>'
                '<a id="scope" href="/rules?tab=scope">Scope</a></nav><main><h1 id="title">Rules</h1>'
                + "<p>Require a linked ticket.</p>" * 10 + "</main>"
                "<script>"
                "setTimeout(() => document.getElementById('boot').remove(), 900);"
                "document.getElementById('scope').addEventListener('click', (event) => {"
                "  event.preventDefault();"
                "  history.pushState({}, '', '/rules?tab=scope');"
                "  const screen = document.createElement('div'); screen.className = 'loading';"
                "  screen.style.cssText = 'position:fixed;inset:0;background:#ff00ff;z-index:50';"
                "  document.body.appendChild(screen);"
                "  setTimeout(() => { screen.remove(); document.getElementById('title').textContent = 'Scope rules'; },"
                "             700);"
                "});"
                "</script>")


@site.get("/teams")
async def teams():
    return page('<style>body{background:#bfdbfe}</style><nav><a href="/">Overview</a><a href="/teams">Teams</a></nav>'
                '<main><h1>Teams</h1>' + "<p>Backend, Frontend.</p>" * 10 + "</main>")  # blue: the dissolve shows


def serve():
    threading.Thread(target=uvicorn.run, args=(site,),
                     kwargs={"host": "127.0.0.1", "port": 8781, "log_level": "error"}, daemon=True).start()
    import urllib.request

    for _ in range(50):
        try:
            urllib.request.urlopen(SITE + "/", timeout=1)
            return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("the fake app did not start")


def silence(path: Path, seconds: float) -> Path:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(b"\x00\x00" * int(24000 * seconds))
    return path


def frame_colors(clip: Path):
    """The mean RGB of every frame."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-vf", "scale=1:1:flags=area", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return [tuple(raw[i:i + 3]) for i in range(0, len(raw) - 2, 3)]


def magenta(rgb) -> bool:
    r, g, b = rgb
    return r > 170 and b > 170 and g < 130


def frame_count(clip: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(clip)], capture_output=True, text=True).stdout
    return int(out.strip() or 0)


def frozen_spans(clip: Path):
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(clip), "-vf", "freezedetect=n=0.002:d=0.3",
                          "-map", "0:v", "-f", "null", "-"], capture_output=True, text=True).stderr
    starts = [float(x) for x in re.findall(r"freeze_start: ([\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"freeze_end: ([\d.]+)", err)]
    return list(zip(starts, ends))


def test_freeze_frames():
    print("freeze_frames")
    check("a hold becomes frames on the clip's clock, replaced by the frame before it",
          freeze_frames([(1.0, 2.0)], 5.0) == [(30, 59, 29)], freeze_frames([(1.0, 2.0)], 5.0))
    check("overlapping and touching holds merge and keep the first replacement",
          freeze_frames([(1.0, 1.5), (1.4, 2.0), (2.0, 2.2)], 5.0) == [(30, 65, 29)],
          freeze_frames([(1.0, 1.5), (1.4, 2.0), (2.0, 2.2)], 5.0))
    check("a hold never reaches past the last frame", freeze_frames([(4.5, 9.0)], 5.0) == [(135, 149, 134)],
          freeze_frames([(4.5, 9.0)], 5.0))
    check("a hold at the very start keeps frame 0 to freeze on", freeze_frames([(0.0, 0.5)], 5.0) == [(1, 14, 0)],
          freeze_frames([(0.0, 0.5)], 5.0))
    check("an empty hold is ignored", freeze_frames([(2.0, 2.01)], 5.0) == [], freeze_frames([(2.0, 2.01)], 5.0))


async def test_session():
    print("a recorded session: click-through, held loads, a cut")
    scenes = [
        Scene(id="s1", title="Overview", route="/", narration="Overview.",
              actions=[Action(type="wait", dwell_ms=1200)]),
        Scene(id="s2", title="Rules", route="/rules", narration="Rules.",
              actions=[Action(type="click", selector='a[href="/rules"]', dwell_ms=1500),
                       Action(type="click", selector='a[href="/rules?tab=scope"]', dwell_ms=1200)]),
        Scene(id="s3", title="Teams", route="/teams", narration="Teams.",
              actions=[Action(type="click", selector='a[href="/teams"]', dwell_ms=1000)]),
    ]
    durations = [2.5, 7.0, 3.0]
    lines = []
    session = WORK / "session.webm"
    segments = await record_session(scenes, SITE, durations, session, lines.append)
    openings = [segment.opening for segment in segments]
    check("scene 1 opens first, scene 2 clicks through on camera, scene 3 opens off camera",
          openings == ["first", "clicks", "cut"], openings)
    check("the log says scene 2 clicks through",
          any("Scene 2: starts where scene 1 ended and clicks through to /rules on camera" in l for l in lines), lines)
    holds = segments[1].holds
    check("scene 2 holds both of its page loads (the server page and the client-side tab)", len(holds) == 2, holds)
    check("each hold covers its loading screen (at least 0.6 s)", all(end - begin >= 0.6 for begin, end in holds),
          holds)
    check("no hold in a scene without a page change", not segments[0].holds and not segments[2].holds,
          (segments[0].holds, segments[2].holds))
    check("scene 3's click on the page it already shows is skipped",
          any("Scene 3: already on /teams, so that click is skipped" in l for l in lines), lines)
    check("nothing had to be opened directly", not any("navigating to it directly" in l for l in lines), lines)
    check("the log counts the held loads", any(l.startswith("Page loads on camera: 2 held") for l in lines), lines)
    check("the trace line says how the scenes joined",
          orchestrator.transitions_line(segments, 1) ==
          "1 scene clicks through on camera, 1 dissolve, 2 page loads held off screen",
          orchestrator.transitions_line(segments, 1))
    return session, segments, durations


async def test_mux(session, segments, durations):
    print("muxing: no loading frame, the dissolve, exact lengths")
    clips = []
    for i, (segment, duration) in enumerate(zip(segments, durations)):
        audio = silence(WORK / f"voice_{i}.wav", duration - 0.4)
        clip = WORK / f"scene_{i + 1:02d}.mp4"
        previous = await last_frame(clips[-1], WORK / f"last_{i}.png") if segment.opening == "cut" else None
        await mux_scene_clip(session, audio, clip, segment.start, segment.duration, focus=segment.focus,
                             holds=segment.holds, dissolve_from=previous)
        clips.append(clip)

    bare = WORK / "scene_02_unheld.mp4"
    await mux_scene_clip(session, WORK / "voice_1.wav", bare, segments[1].start, segments[1].duration)
    bare_colors = frame_colors(bare)
    shown = sum(map(magenta, bare_colors))
    first_seen = next((i / FPS for i, rgb in enumerate(bare_colors) if magenta(rgb)), None)
    lead = None if first_seen is None else segments[1].holds[0][0] - first_seen  # > 0: the video runs ahead
    check("without holds the loading screen reaches the clip (the test sees it)", shown >= 10,
          f"{shown} frames, first at {first_seen}s, the hold starts at {segments[1].holds[0][0]}s"
          + (f" (video ahead by {lead:.2f}s)" if lead is not None else ""))
    held = sum(map(magenta, frame_colors(clips[1])))
    check("with holds no loading frame survives", held == 0, f"{held} frames")
    spans = frozen_spans(clips[1])
    check("the clip is frozen where the loads were held", len(spans) >= 1 and all(
        any(abs(b - s) < 0.25 for s, _ in spans) for b, _ in segments[1].holds), (spans, segments[1].holds))

    colors3 = frame_colors(clips[2])
    end2 = frame_colors(clips[1])[-1]
    first3 = colors3[0]
    later3 = colors3[int(DISSOLVE_SEC * FPS) + 3]
    check("scene 3 starts on scene 2's last frame (the dissolve)", sum(abs(a - b) for a, b in zip(first3, end2)) <= 9,
          (first3, end2))
    middle = colors3[int(DISSOLVE_SEC * FPS / 2)]
    check("halfway through the dissolve both pages show", first3[2] - first3[0] < middle[2] - middle[0] <
          later3[2] - later3[0], (first3, middle, later3))
    check("…and once it is over only the blue /teams page does", later3[2] - later3[0] > 20, later3)
    lengths = [frame_count(clip) for clip in clips]
    expected = [round(d * FPS) for d in durations]
    check("every clip keeps its exact length (holds and the dissolve add no frames)", lengths == expected,
          (lengths, expected))
    final = WORK / "demo.mp4"
    await concat_scenes(clips, final)
    total = await probe_duration(final)
    check("the cut is as long as the scenes", abs(total - sum(durations)) < 0.1, f"{total:.2f}s")

    try:
        await last_frame(WORK / "missing.mp4", WORK / "never.png")
        check("last_frame on a missing clip raises", False)
    except Exception:
        check("last_frame on a missing clip raises", True)


async def main():
    serve()
    test_freeze_frames()
    session, segments, durations = await test_session()
    await test_mux(session, segments, durations)
    print(f"\n{sum(results)}/{len(results)} checks passed (work dir {WORK})")
    return 0 if all(results) else 1


sys.exit(asyncio.run(main()))
