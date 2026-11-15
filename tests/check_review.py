"""Unit checks for app/services/review.py (lane 7: scene review + gated retake).

Offline: contact sheets from synthetic scene clips (ffmpeg test pattern + tone, the pipeline's codec settings),
sanitizing of fake Gemini payloads, the retake gate.

    .venv/bin/python tests/check_review.py
"""
import asyncio
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the backend root (tests/..)
WORK = Path(tempfile.mkdtemp(prefix="demoday-review-"))
CLIPS = WORK / "clips"  # made by ensure_clips()
SCENE_RANGES = [(0.0, 10.0), (10.0, 9.4), (19.4, 9.4), (28.8, 9.8), (38.6, 8.9)]
os.environ.setdefault("STORAGE_DIR", str(WORK / "storage"))
sys.path.insert(0, BACKEND)

from app.models.schemas import Action, Scene  # noqa: E402
from app.services import review  # noqa: E402
from app.services.stitcher import concat_scenes, describe_media, frame_aligned, probe_duration  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail != '' else ''}")


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def ensure_clips():
    """Five scene clips (moving test pattern + a tone, H.264/AAC 48 kHz like mux_scene_clip) and a 5 s white clip."""
    CLIPS.mkdir(parents=True, exist_ok=True)
    encode = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac",
              "-b:a", "160k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", "-shortest"]
    for i, (_, t) in enumerate(SCENE_RANGES, 1):
        ffmpeg("-f", "lavfi", "-i", f"testsrc2=s=1920x1080:r=30:d={t:.3f}",
               "-f", "lavfi", "-i", f"sine=frequency={200 + 40 * i}:sample_rate=48000:duration={t:.3f}",
               *encode, str(CLIPS / f"scene_{i:02d}.mp4"))
    ffmpeg("-f", "lavfi", "-i", "color=c=white:s=1920x1080:r=30:d=5", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
           "-t", "5", *encode, str(CLIPS / "blank.mp4"))


def image_size(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    return tuple(int(x) for x in out.split(","))


# Scene plans from a real 47 s dashboard tour (overview, events, one event, rules, analytics); the clips are synthetic.
def sample_scenes():
    click = lambda href: Action(type="click", selector=f'a[href="{href}"]', dwell_ms=1500)  # noqa: E731
    scroll = Action(type="scroll", value="down", dwell_ms=1500)
    return [
        Scene(id="s1", title="Overview", route="/dashboard/warestack",
              narration="Warestack's overview shows deployment frequency, lead time and the pull requests that need attention.",
              actions=[scroll], duration_sec=10.0),
        Scene(id="s2", title="Events feed", route="/dashboard/warestack/monitoring",
              narration="Every workflow run and rule evaluation lands in the monitoring feed.",
              actions=[click("/dashboard/warestack/monitoring"), scroll], duration_sec=9.4),
        Scene(id="s3", title="Rule evaluation", route="/dashboard/warestack/monitoring/events/1",
              narration="Open an event to see why a rule fired and the evidence behind the finding.",
              actions=[click("/dashboard/warestack/monitoring/events/1")], duration_sec=9.4),
        Scene(id="s4", title="Rules catalog", route="/dashboard/warestack/rules",
              narration="The rules catalog lists every check you can enable per repository.",
              actions=[click("/dashboard/warestack/rules"), scroll], duration_sec=9.805),
        Scene(id="s5", title="Analytics", route="/dashboard/warestack/analytics",
              narration="Analytics tracks pull requests, deployments and lead time over the last thirty days.",
              actions=[click("/dashboard/warestack/analytics"), scroll], duration_sec=8.9),
    ]


def verdicts_of(spec):
    """[(verdict, confidence), ...] -> Verdict list (scene numbers 1..n)."""
    return [review.Verdict(scene=i, verdict=v, confidence=c, title=f"S{i}") for i, (v, c) in enumerate(spec, 1)]


async def main():
    ensure_clips()
    clips = [CLIPS / f"scene_{i:02d}.mp4" for i in range(1, 6)]
    work = Path(tempfile.mkdtemp(prefix="work-", dir=WORK))

    print("contact sheets")
    t0 = time.monotonic()
    sheets = await asyncio.gather(*(review.contact_sheet(c, d, work / f"sheet_{i:02d}.jpg")
                                    for i, (c, (_, d)) in enumerate(zip(clips, SCENE_RANGES), 1)))
    took = time.monotonic() - t0
    check("five 2x2 sheets produced in parallel", all(s.exists() for s in sheets), f"{took:.2f}s for 5 clips")
    for s in sheets:
        size = image_size(s)
        check(f"{s.name} is a 1280x720 JPEG of sane size", size == (1280, 720) and 15_000 < s.stat().st_size < 600_000,
              f"{size[0]}x{size[1]}, {s.stat().st_size // 1024} KB")
    three = await review.contact_sheet(clips[0], 10.0, work / "three.jpg", frames=3)
    check("a 3-frame sheet is one 1920-wide row", image_size(three) == (1920, 360), image_size(three))
    short = await review.contact_sheet(CLIPS / "blank.mp4", 5.0, work / "blank.jpg")
    check("the 5 s white clip gets a sheet too", image_size(short) == (1280, 720), image_size(short))

    print("prompt")
    scenes = sample_scenes()
    prompt = review.build_prompt("Warestack", scenes, [1, 2, 4, 5])
    check("prompt names the project, lists only the sheet scenes and the JSON shape",
          '"Warestack"' in prompt and "scenes 1, 2, 4, 5" in prompt and '3. "Rule evaluation"' not in prompt
          and '"verdict": "ok|blank|error_page|login_wall|static|mismatch"' in prompt)
    check("prompt carries narration and planned actions", "lead time and the pull requests" in prompt
          and 'click a[href="/dashboard/warestack/rules"]; scroll down' in prompt)

    print("sanitizing (fake Gemini)")
    captured = {}

    async def fake_generate_json(prompt, *, images=(), deep=False, timeout_sec=0, purpose=""):
        captured.update(prompt=prompt, images=list(images), purpose=purpose, timeout=timeout_sec)
        return {"scenes": [
            {"scene": 1, "verdict": "OK", "confidence": "0.9", "seen": "dashboard with four metric cards"},
            {"scene": 2, "verdict": "loading", "confidence": 85, "seen": "grey skeleton rows", "issue": "skeleton in all frames"},
            {"scene": 4, "verdict": "purple", "confidence": 0.5},
            {"scene": 9, "verdict": "ok", "confidence": 1},
            "garbage",
            {"scene": "scene 5", "verdict": "error page", "confidence": 0.95, "issue": "404 not found"},
            {"scene": 1, "verdict": "blank", "confidence": 1.0},  # duplicate: first wins
        ]}

    real = review.generate_json
    review.generate_json = fake_generate_json
    logs = []
    try:
        scenes = sample_scenes()
        verdicts = await review.review_scenes(clips, scenes, work, logs.append, project_name="Warestack")
    finally:
        review.generate_json = real
    got = [(v.scene, v.verdict, v.confidence) for v in verdicts]
    check("one verdict per scene, enums mapped, numbers clamped, missing -> unknown",
          got == [(1, "ok", 0.9), (2, "blank", 0.85), (3, "unknown", 0.0), (4, "unknown", 0.0), (5, "error_page", 0.95)], got)
    check("verdicts and seen notes written to the scenes",
          [s.verdict for s in scenes] == ["ok", "blank", "unknown", "unknown", "error_page"]
          and scenes[0].seen == "dashboard with four metric cards" and scenes[3].seen is None,
          [(s.verdict, s.seen) for s in scenes])
    check("one call with five inline JPEG sheets", len(captured["images"]) == 5
          and all(img[:3] == b"\xff\xd8\xff" for img in captured["images"]) and captured["purpose"] == "review footage",
          f"{len(captured['images'])} images, purpose={captured['purpose']!r}, timeout={captured['timeout']}")
    import inspect
    from app.services import gemini as _gemini
    expected = {"thinking": review.REVIEW_THINKING} if "thinking" in inspect.signature(_gemini.generate_json).parameters else {}
    check("thinking kwarg passed exactly when generate_json supports it", review._thinking_kwargs() == expected, review._thinking_kwargs())

    async def fake_with_thinking(prompt, *, images=(), deep=False, timeout_sec=0, purpose="", thinking=None):
        captured["thinking"] = thinking
        return [{"verdict": "ok", "confidence": 0.9}] * 5

    review.generate_json = fake_with_thinking
    try:
        await review.review_scenes(clips, sample_scenes(), work, lambda m: None)
        check("thinking='minimal' is forwarded once generate_json accepts it", captured.get("thinking") == "minimal",
              captured.get("thinking"))
    finally:
        review.generate_json = real
    check("sheets live under job_dir/review", sorted(p.name for p in (work / "review").iterdir()) ==
          [f"scene_{i:02d}.jpg" for i in range(1, 6)])
    check("summary string", review.summarize(verdicts) ==
          "1/5 pass · scene 2 looks blank (0.85) · scene 3 looks unreviewed (0.00) · scene 4 looks unreviewed (0.00) · +1 more",
          review.summarize(verdicts))
    check("all-pass summary", review.summarize(verdicts_of([("ok", 0.9)] * 5)) == "5/5 scenes pass")
    check("step extra is plain dicts", review.as_extra(verdicts)["scenes"][1] ==
          {"scene": 2, "verdict": "blank", "confidence": 0.85, "seen": "grey skeleton rows",
           "issue": "skeleton in all frames", "title": "Events feed"})
    check("log lines: checking + per-flag lines", logs[0].startswith("Review: Gemini is checking 5 scenes")
          and any(l.startswith("Review: scene 2 'Events feed' looks blank (confidence 0.85): skeleton") for l in logs), logs)
    bare = review.sanitize([{"verdict": "ok", "confidence": 0.8}, {"verdict": "sign-in", "confidence": "high"}], scenes[:2])
    check("bare list payload numbered by position; 'sign-in' -> login_wall, 'high' -> 0.9",
          [(v.scene, v.verdict, v.confidence) for v in bare] == [(1, "ok", 0.8), (2, "login_wall", 0.9)])
    keyed = review.sanitize({"1": {"verdict": "static", "confidence": 0.7}, "2": {"verdict": "mismatch", "confidence": "70%"}},
                            scenes[:2])
    check("dict keyed by scene number", [(v.verdict, v.confidence) for v in keyed] == [("static", 0.7), ("mismatch", 0.7)])
    junk = review.sanitize("not json at all", scenes[:2])
    check("junk payload -> all unknown", [v.verdict for v in junk] == ["unknown", "unknown"])

    async def failing(*a, **k):
        raise RuntimeError("503 overloaded")

    review.generate_json = failing
    try:
        scenes = sample_scenes()
        try:
            await review.review_scenes(clips, scenes, work, lambda m: None)
            check("Gemini failure propagates (so _optional reports 'Review skipped')", False)
        except RuntimeError as exc:
            check("Gemini failure propagates (so _optional reports 'Review skipped')", "503" in str(exc), exc)
        check("scenes untouched after a failed review", all(s.verdict is None for s in scenes))
    finally:
        review.generate_json = real
    try:
        await review.review_scenes([work / "missing.mp4"], sample_scenes()[:1], work, lambda m: None)
        check("no extractable frames -> raises", False)
    except Exception as exc:
        check("no extractable frames -> raises", "no frames" in str(exc), exc)

    print("retake gate")
    table = [
        ("all ok", [("ok", 0.9)] * 5, None),
        ("one blank at 0.85 -> that index", [("ok", 0.9), ("ok", 0.9), ("ok", 0.8), ("blank", 0.85), ("ok", 0.9)], 3),
        ("one error page at exactly 0.7 -> index", [("ok", 0.9), ("error_page", 0.7), ("ok", 0.9)], 1),
        ("blank at 0.6 -> None", [("ok", 0.9), ("blank", 0.6), ("ok", 0.9)], None),
        ("two blanks -> None", [("blank", 0.9), ("ok", 0.9), ("blank", 0.9)], None),
        ("blank 0.9 + blank 0.5 -> None", [("blank", 0.9), ("ok", 0.9), ("blank", 0.5)], None),
        ("login wall -> None", [("ok", 0.9), ("login_wall", 0.95), ("ok", 0.9)], None),
        ("login wall elsewhere + blank -> None", [("login_wall", 0.9), ("ok", 0.9), ("blank", 0.9)], None),
        ("static / mismatch are never retaken", [("static", 0.95), ("mismatch", 0.95), ("ok", 0.9)], None),
        ("unknown next to one blank -> blank's index", [("unknown", 0.0), ("blank", 0.9), ("ok", 0.9)], 1),
        ("empty -> None", [], None),
    ]
    for label, spec, expected in table:
        got = review.should_retake(verdicts_of(spec))
        check(label, got == expected, f"got {got}")
    check("pick_retake is the same gate", review.pick_retake is review.should_retake)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


sys.exit(asyncio.run(main()))
