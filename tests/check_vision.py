"""Unit checks for the Gemini vision brief: vision.describe_pages (sanitizing, write-back, retry, timeout),
vision.thumbnails, the planner's insight lines plus plan_rationale/why, and the screenshot hooks on probe_app_url.

Offline by default: Gemini is monkeypatched (the key is a dummy, so a missed patch fails loudly); ffmpeg is used
for the thumbnail and shrink checks when present. LIVE=1 adds one real probe of https://fastapi.tiangolo.com/ with
screenshots, one real describe_pages call and the thumbnails, printing latencies.

Run: .venv/bin/python tests/check_vision.py        LIVE=1 .venv/bin/python tests/check_vision.py
"""
import asyncio
import base64
import difflib
import importlib.machinery
import importlib.util
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the backend root (tests/..)
HERE = Path(__file__).resolve().parent
LIVE = os.environ.get("LIVE", "").strip().lower() in ("1", "true", "yes")
TMP = Path(tempfile.mkdtemp(prefix="demoday-vision-unit-"))
os.environ.setdefault("STORAGE_DIR", str(TMP / "storage"))
os.environ.setdefault("BRIEF_MAX_PAGES", "4")  # the live-run setting
if not LIVE:
    os.environ["GEMINI_API_KEY"] = "offline-test-key"  # every Gemini entry point is monkeypatched below
sys.path.insert(0, BACKEND)

from app.services import gemini, images, ingest, planner, vision  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + str(detail)[:400] if detail else ''}")


# --------------------------------------------------------------------------- fixtures


def make_jpeg(path: Path, color: str = "0x3366cc") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if FFMPEG:
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={color}:s=640x360",
                        "-frames:v", "1", "-q:v", "5", str(path)], check=True)
    else:
        path.write_bytes(b"\xff\xd8not-really-a-jpeg")
    return str(path)


def fake_app_info(n_pages=3, missing=(), shots=True):
    root = TMP / f"shots-{n_pages}-{len(missing)}-{int(shots)}"
    pages = []
    for i in range(1, n_pages + 1):
        page = {"path": f"/p{i}", "url": f"https://x.dev/p{i}", "title": f"Page {i}", "link_text": f"P{i}",
                "open_selector": f'a[href="/p{i}"]', "headings": [f"H{i}a", f"H{i}b"], "buttons": ["Go"], "inputs": []}
        if shots and i not in missing:
            page["screenshot"] = make_jpeg(root / f"page_{i:02d}.jpg")
        pages.append(page)
    info = {"url": "https://x.dev/", "status": 200, "title": "X - The X app", "description": "", "headings": ["Welcome"],
            "links": [], "buttons": [], "inputs": [], "login_form": False, "pages": pages}
    if shots:
        info["screenshot"] = make_jpeg(root / "page_00.jpg")
    return info


class FakeGemini:
    """Stands in for gemini.generate_json: records calls, answers from a queue (an Exception raises, a coroutine
    function is awaited)."""

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    async def generate_json(self, prompt, *, images=(), deep=False, timeout_sec=90, purpose="gemini"):
        self.calls.append({"prompt": prompt, "images": len(images), "purpose": purpose, "timeout": timeout_sec,
                           "deep": deep, "bytes": sum(len(i) for i in images)})
        answer = self.responses.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        if callable(answer):
            return await answer()
        return answer


async def fake_shrink(path, max_width=1280, quality=6):
    return b"\xff\xd8" + str(path).encode()


async def run_vision(app_info, *responses, log=None, shrink=fake_shrink, timeout=None):
    """describe_pages with Gemini (and optionally the shrink) monkeypatched; returns (rows_or_exception, fake)."""
    fake = FakeGemini(*responses)
    saved = gemini.generate_json, images.shrink_jpeg
    gemini.generate_json = fake.generate_json
    if shrink is not None:
        images.shrink_jpeg = shrink
    try:
        coro = vision.describe_pages(app_info, log or (lambda m: None))
        outcome = await (asyncio.wait_for(coro, timeout) if timeout else coro)
    except BaseException as exc:  # noqa: BLE001 - the test inspects it
        outcome = exc
    finally:
        gemini.generate_json, images.shrink_jpeg = saved
    return outcome, fake


GOOD_ANSWER = {"pages": [
    {"index": 1, "purpose": "**Framework** landing page", "demo_moment": "the feature list, top", "quality": "Good"},
    {"index": 3, "purpose": "x" * 300, "demo_moment": "y" * 300, "quality": "weird"},
    {"index": 2, "purpose": "Sign-in form", "demo_moment": "", "quality": "LOGIN"},
    {"index": 4, "purpose": "Reports table linking https://example.com/foo/bar", "demo_moment": "the ranking table, middle",
     "quality": "busy"},
]}


# --------------------------------------------------------------------------- offline checks


async def vision_checks():
    print("vision.describe_pages")
    info = fake_app_info(3)
    rows, fake = await run_vision(info, GOOD_ANSWER)
    check("one Gemini call with every screenshot inline", len(fake.calls) == 1 and fake.calls[0]["images"] == 4
          and fake.calls[0]["purpose"] == "understand pages" and fake.calls[0]["timeout"] == 20, fake.calls)
    prompt = fake.calls[0]["prompt"] if fake.calls else ""
    check("prompt lists the pages in order with their titles",
          'screenshots of 4 pages of the live app, in this order: 1) / "X - The X app"  2) /p1 "Page 1"  3) /p2 "Page 2"  4) /p3 "Page 3".' in prompt,
          prompt[:300])
    check("prompt carries the brief's rules verbatim", '"quality": "good|busy|empty|error|login"' in prompt
          and "A small Sample data pill in a corner and a green cursor are expected, not issues." in prompt
          and "never guess features from the URL and never invent anything" in prompt)
    check("rows in page order with the local screenshot path", isinstance(rows, list)
          and [r["path"] for r in rows] == ["/", "/p1", "/p2", "/p3"] and all(r.get("screenshot") for r in rows), rows)
    landing = info.get("landing_insight") or {}
    check("landing insight written, markdown stripped, quality normalized",
          landing.get("purpose") == "Framework landing page" and landing.get("demo_moment") == "the feature list, top"
          and landing.get("quality") == "good" and landing.get("source") == "screenshot", landing)
    check("index 2 -> pages[0]: login flag kept, empty moment allowed",
          info["pages"][0].get("insight", {}).get("quality") == "login" and info["pages"][0]["insight"]["demo_moment"] == "",
          info["pages"][0].get("insight"))
    p2 = info["pages"][1].get("insight", {})
    check("index 3 -> pages[1]: purpose capped at 120, moment at 140, unknown quality -> good",
          len(p2.get("purpose", "")) == 120 and len(p2.get("demo_moment", "")) == 140 and p2.get("quality") == "good", p2)
    p3 = info["pages"][2].get("insight", {})
    check("index 4 -> pages[2]: URL reduced to its domain (TTS-safe), busy kept",
          p3.get("purpose") == "Reports table linking example.com" and p3.get("quality") == "busy", p3)
    check("row carries title/purpose/demo_moment/quality for the step event",
          rows and {"path", "title", "purpose", "demo_moment", "quality", "source"} <= set(rows[0]) and rows[0]["title"] == "X - The X app",
          rows[0] if rows else None)
    line = vision.summary_line(rows)
    check("summary line matches the brief's format",
          line.startswith("Gemini saw: / = Framework landing page (best moment: the feature list, top) · /p1 = Sign-in form [flagged: login] · /p2 = xxx"),
          line[:200])

    info = fake_app_info(9)
    answer = {"pages": [{"index": i, "purpose": f"purpose {i}", "demo_moment": f"moment {i}", "quality": "good"} for i in range(1, 8)]}
    rows, fake = await run_vision(info, answer)
    check("at most 7 screenshots go to Gemini (landing + first 6 pages)", fake.calls[0]["images"] == 7
          and "screenshots of 7 pages" in fake.calls[0]["prompt"] and isinstance(rows, list) and len(rows) == 7, fake.calls[0]["images"])
    check("pages beyond the cap get no insight", all("insight" not in p for p in info["pages"][6:])
          and info["pages"][5]["insight"]["purpose"] == "purpose 7")

    info = fake_app_info(3, missing=(2,))
    rows, fake = await run_vision(info, {"pages": [{"index": i, "purpose": f"p{i}", "quality": "good"} for i in range(1, 4)]})
    check("a page without a screenshot is skipped and the indexes stay aligned", fake.calls[0]["images"] == 3
          and "insight" not in info["pages"][1] and info["pages"][2]["insight"]["purpose"] == "p3"
          and info["pages"][0]["insight"]["purpose"] == "p2" and info["landing_insight"]["purpose"] == "p1",
          [p.get("insight") for p in info["pages"]])

    info = fake_app_info(2)
    outcome, fake = await run_vision(info, RuntimeError("503 UNAVAILABLE: The model is overloaded"))
    check("a Gemini failure raises (for the orchestrator's _optional) and writes nothing",
          isinstance(outcome, RuntimeError) and len(fake.calls) == 1 and "landing_insight" not in info
          and all("insight" not in p for p in info["pages"]), repr(outcome))

    info = fake_app_info(2)
    logs = []
    outcome, fake = await run_vision(info, ValueError("400 INVALID_ARGUMENT: Unsupported MIME type image/jpeg"),
                                     {"pages": [{"index": 1, "purpose": "landing"}, {"index": 2, "purpose": "page one"}]},
                                     log=logs.append)
    check("an image refusal earns one text-only retry over the headings", len(fake.calls) == 2 and fake.calls[1]["images"] == 0
          and "Screenshots are not available" in fake.calls[1]["prompt"] and 'headings: H1a | H1b' in fake.calls[1]["prompt"]
          and fake.calls[1]["purpose"] == "understand pages (headings)", [c["purpose"] for c in fake.calls])
    check("retry insights are tagged source=headings and logged", isinstance(outcome, list) and outcome[0]["source"] == "headings"
          and info["landing_insight"]["source"] == "headings" and any("declined the screenshots" in m for m in logs), logs)

    info = fake_app_info(2)
    outcome, fake = await run_vision(info, RuntimeError("Unsupported MIME"), RuntimeError("still broken"))
    check("a failing retry raises too (no third call)", isinstance(outcome, RuntimeError) and len(fake.calls) == 2)

    async def slow():
        await asyncio.sleep(3)
        return GOOD_ANSWER

    info = fake_app_info(2)
    t = time.monotonic()
    outcome, fake = await run_vision(info, slow, timeout=0.3)
    check("a timeout (the orchestrator's wait_for) cancels the call promptly and writes nothing",
          isinstance(outcome, asyncio.TimeoutError) and time.monotonic() - t < 1.5 and "landing_insight" not in info, repr(outcome))

    info = fake_app_info(2)
    outcome, _ = await run_vision(info, {"pages": []})
    check("an empty answer raises so the trace shows the fallback chip", isinstance(outcome, ValueError), repr(outcome))

    info = fake_app_info(1)
    rows, _ = await run_vision(info, {"pages": [{"index": 0, "purpose": "A"}, {"index": 1, "purpose": "B"}]})
    check("0-based indexes are tolerated", info["landing_insight"]["purpose"] == "A" and info["pages"][0]["insight"]["purpose"] == "B")

    info = fake_app_info(1)
    rows, _ = await run_vision(info, [{"purpose": "A"}, {"purpose": "B"}])
    check("a bare list without indexes maps positionally", info["landing_insight"]["purpose"] == "A"
          and info["pages"][0]["insight"]["purpose"] == "B")

    info = fake_app_info(1)
    rows, _ = await run_vision(info, {"result": {"pages": [{"index": 1, "purpose": "A"}, {"index": 1, "purpose": "dup"},
                                                           {"index": 2, "purpose": "  "}, "junk"]}})
    check("wrapped answers unwrap; duplicates and blank purposes are dropped", info["landing_insight"]["purpose"] == "A"
          and "insight" not in info["pages"][0] and isinstance(rows, list) and len(rows) == 1, rows)

    info = fake_app_info(2, shots=False)
    rows, fake = await run_vision(info, GOOD_ANSWER)
    check("no screenshots -> no call, empty result", rows == [] and not fake.calls)

    if FFMPEG:
        info = fake_app_info(2)
        info["pages"][0]["screenshot"] = str(TMP / "does-not-exist.jpg")
        rows, fake = await run_vision(info, {"pages": [{"index": 1, "purpose": "A"}, {"index": 2, "purpose": "B"}]}, shrink=None)
        check("real shrink: an unreadable screenshot is dropped, the rest still go (bytes shrink)", fake.calls[0]["images"] == 2
              and 0 < fake.calls[0]["bytes"] < 200_000 and info["pages"][1]["insight"]["purpose"] == "B"
              and "insight" not in info["pages"][0], fake.calls[0]["bytes"])

    print("vision.thumbnails")
    info = fake_app_info(2)
    rows, _ = await run_vision(info, {"pages": [{"index": i, "purpose": f"p{i}"} for i in range(1, 4)]})
    rows.append({"path": "/missing", "title": "", "purpose": "x", "demo_moment": "", "quality": "good", "source": "screenshot",
                 "screenshot": str(TMP / "missing.jpg")})
    rows.append({"path": "/noshot", "title": "", "purpose": "x", "demo_moment": "", "quality": "good", "source": "screenshot"})
    if FFMPEG:
        t = time.monotonic()
        await vision.thumbnails(rows)
        ms = (time.monotonic() - t) * 1000
        thumbs = [base64.b64decode(r["thumb"]) for r in rows[:3] if r.get("thumb")]
        check("thumb is a base64 JPEG on every screenshotted row", len(thumbs) == 3 and all(t[:2] == b"\xff\xd8" for t in thumbs),
              f"{[len(t) for t in thumbs]} bytes in {ms:.0f} ms")
        check("the local path never reaches the event; a missing file just loses its thumb",
              all("screenshot" not in r for r in rows) and "thumb" not in rows[3] and "thumb" not in rows[4])
    else:
        check("ffmpeg missing: thumbnail check skipped", True)


def planner_checks():
    print("planner: site map + prompt")
    page = {"path": "/p1", "title": "Page 1", "open_selector": 'a[href="/p1"]', "headings": ["H1a", "H1b"], "buttons": ["Go"], "inputs": []}
    plain = planner._site_map_lines({"pages": [page]})
    check("site-map line without an insight is unchanged",
          plain == '- route /p1 - "Page 1"; shows: H1a | H1b; buttons: Go; open with: click a[href="/p1"]', plain)
    seen = dict(page, insight={"purpose": "interactive API reference", "demo_moment": "the Try it out button, top", "quality": "good"})
    line = planner._site_map_lines({"pages": [seen]})
    check("site-map line gains Gemini saw + best moment before 'open with'",
          line == '- route /p1 - "Page 1"; shows: H1a | H1b; buttons: Go; Gemini saw: "interactive API reference"; '
                  'best moment: "the Try it out button, top"; open with: click a[href="/p1"]', line)
    for quality, words in (("login", "a sign-in form"), ("error", "an error page"), ("empty", "almost no content")):
        flagged = dict(page, insight={"purpose": "form", "demo_moment": "", "quality": quality})
        line = planner._site_map_lines({"pages": [flagged]})
        check(f"{quality} page carries the AVOID flag", f'; Gemini saw: "form"; AVOID: in the brief this page showed {words}, so do not film it; open with:' in line, line)
        check(f"{quality} page is excluded from the tour count", planner._flagged(flagged) and not planner._flagged(seen))
    busy = dict(page, insight={"purpose": "dense table", "demo_moment": "", "quality": "busy"})
    check("busy pages are not flagged", "AVOID" not in planner._site_map_lines({"pages": [busy]}) and not planner._flagged(busy))
    check("no landing line without an insight", planner._landing_insight_line({}) == "" and planner._landing_insight_line({"landing_insight": {"purpose": ""}}) == "")
    check("landing line with an insight", planner._landing_insight_line({"landing_insight": {"purpose": "framework landing page", "demo_moment": "the feature list, top"}})
          == "- What the landing page shows (Gemini looked at a screenshot): framework landing page; best moment: the feature list, top\n")

    print("planner: _sanitize_storyboard")
    raw = {"project_name": "X", "tagline": "t", "plan_rationale": "**Opening** on /benchmarks because the README leads with performance.",
           "scenes": [{"id": "s", "title": "A", "route": "/", "narration": "Hello there.", "why": "Shows *speed* claims are real",
                       "actions": [{"type": "scroll", "value": "down"}]},
                      {"id": "s", "title": "B", "route": "/p1", "narration": "Second.", "actions": []}]}
    sb = planner._sanitize_storyboard(raw, "X")
    check("plan_rationale kept and cleaned", sb.plan_rationale == "Opening on /benchmarks because the README leads with performance.", sb.plan_rationale)
    check("scene why kept and cleaned; missing why is None", sb.scenes[0].why == "Shows speed claims are real" and sb.scenes[1].why is None,
          [s.why for s in sb.scenes])
    long = planner._sanitize_storyboard(dict(raw, plan_rationale="r" * 500, scenes=[dict(raw["scenes"][0], why="w" * 500)]), "X")
    check("caps: rationale 200, why 160", len(long.plan_rationale) == 200 and len(long.scenes[0].why) == 160)
    none = planner._sanitize_storyboard({"scenes": raw["scenes"][1:]}, "X")
    check("without the keys both are None", none.plan_rationale is None and none.scenes[0].why is None)
    tpl = planner._template_storyboard({"name": "X", "description": "d"}, {"title": "X", "links": []})
    check("template storyboard unchanged (no rationale)", tpl.plan_rationale is None and all(s.why is None for s in tpl.scenes))


async def prompt_checks():
    print("planner: generate_storyboard prompt")
    captured = []
    canned = {"project_name": "X", "tagline": "t", "plan_rationale": "Opening on /p1 because the README leads with it",
              "scenes": [{"id": "s", "title": "A", "route": "/", "narration": "Hi.", "why": "proves the landing hook",
                          "actions": [{"type": "wait"}]}]}

    async def capture(prompt, deep=False, purpose="gemini"):
        captured.append(prompt)
        return canned

    saved = planner._generate_json
    planner._generate_json = capture
    try:
        repo = {"name": "x", "readme": "# x", "dependencies": [], "routes": [], "language": "Python"}

        def pages(flag_first):
            out = []
            for i in range(1, 5):
                p = {"path": f"/p{i}", "title": f"Page {i}", "open_selector": f'a[href="/p{i}"]', "headings": [f"H{i}"], "buttons": [], "inputs": [],
                     "insight": {"purpose": f"purpose {i}", "demo_moment": f"moment {i}", "quality": "login" if (flag_first and i == 1) else "good"}}
                out.append(p)
            return out

        info = {"url": "https://x.dev/", "title": "X", "description": "", "headings": ["Welcome"], "links": [], "buttons": [], "inputs": [],
                "pages": pages(True), "landing_insight": {"purpose": "framework landing page", "demo_moment": "the feature list, top", "quality": "good"}}
        sb, note = await planner.generate_storyboard(repo, info, 60, "energetic_tech")
        prompt = captured[-1]
        check("PROJECT block carries the landing insight",
              "- What the landing page shows (Gemini looked at a screenshot): framework landing page; best moment: the feature list, top\n- Page routes found" in prompt)
        check("tour rule counts only unflagged pages (3 of 4 -> 4-5 scenes, at least 4 pages)",
              "Write 4-5 scenes as a tour" in prompt and "at least 4 distinct pages" in prompt)
        check("flagged page stays on the site map with AVOID", 'route /p1 - "Page 1"; shows: H1; Gemini saw: "purpose 1"; best moment: "moment 1"; AVOID: in the brief this page showed a sign-in form, so do not film it; open with' in prompt)
        check("JSON shape asks for plan_rationale and why",
              '"plan_rationale": "...", "scenes": [{"id": "scene-1",' in prompt and '"route": "/", "why": "...", "narration": "..."' in prompt
              and '"plan_rationale": one sentence, spoken to the founder, on why these pages in this order' in prompt
              and 'Each scene\'s "why": <= 20 words on what this\n   scene proves about the product.' in prompt)
        check("storyboard carries Gemini's rationale and why", sb.plan_rationale == canned["plan_rationale"] and sb.scenes[0].why == "proves the landing hook",
              (sb.plan_rationale, sb.scenes[0].why))
        info["pages"] = pages(False)
        await planner.generate_storyboard(repo, info, 60, "energetic_tech")
        check("4 unflagged pages -> 5-6 scenes", "Write 5-6 scenes as a tour" in captured[-1] and "at least 4 distinct pages" in captured[-1])
        for p in info["pages"]:
            p.pop("insight")
        info.pop("landing_insight")
        await planner.generate_storyboard(repo, info, 60, "energetic_tech")
        bare = captured[-1]
        check("without insights the prompt has no Gemini lines but still asks for the rationale",
              "Gemini saw" not in bare and "What the landing page shows" not in bare and "AVOID" not in bare and '"plan_rationale"' in bare)
    finally:
        planner._generate_json = saved


async def ingest_checks():
    print("ingest hooks")
    sig = inspect.signature(ingest.probe_app_url)
    check("probe_app_url gains keyword-only shots_dir and on_page (default None)",
          all(sig.parameters[n].kind is inspect.Parameter.KEYWORD_ONLY and sig.parameters[n].default is None for n in ("shots_dir", "on_page")),
          str(sig))
    check("positional callers unchanged", [p.name for p in sig.parameters.values()][:3] == ["app_url", "storage_state", "replayer"])
    sig2 = inspect.signature(ingest._explore_pages)
    check("_explore_pages accepts shots_dir/on_page", {"shots_dir", "on_page"} <= set(sig2.parameters) and sig2.parameters["shots_dir"].default is None)
    check("BRIEF_MAX_PAGES comes from the environment", ingest.BRIEF_MAX_PAGES == int(os.environ["BRIEF_MAX_PAGES"]), ingest.BRIEF_MAX_PAGES)
    check("_screenshot is a no-op without shots_dir", await ingest._screenshot(None, None, 0) is None)


async def live_checks():
    print("LIVE: probe https://fastapi.tiangolo.com/ with screenshots")
    shots = TMP / "shots"
    explored = []

    def on_page(p):
        explored.append(p["path"])
        print(f"    Explored {p['path']}: '{p['title']}' ({len(p['headings'])} headings, {len(p['buttons'])} buttons)")

    t = time.monotonic()
    info = await ingest.probe_app_url("https://fastapi.tiangolo.com/", shots_dir=shots, on_page=on_page)
    probe_s = time.monotonic() - t
    files = sorted(shots.glob("page_*.jpg"))
    check(f"probe finished in {probe_s:.1f}s with {len(info['pages'])} pages explored", not info.get("error") and info["pages"], info.get("error"))
    check("page_00.jpg (landing) + one JPEG per explored page", info.get("screenshot") and Path(info["screenshot"]).name == "page_00.jpg"
          and len(files) == len(info["pages"]) + 1 and all(p.get("screenshot") for p in info["pages"]),
          [f"{f.name} {f.stat().st_size // 1024} KB" for f in files])
    check("on_page fired once per explored page, in order", explored == [p["path"] for p in info["pages"]], explored)

    print("LIVE: vision.describe_pages (one Gemini call)")
    t = time.monotonic()
    try:
        rows = await asyncio.wait_for(vision.describe_pages(info, lambda m: print("    log:", m)), 20)
    except Exception as exc:  # noqa: BLE001
        rows = exc
    vision_s = time.monotonic() - t
    ok = isinstance(rows, list) and rows
    check(f"describe_pages answered in {vision_s:.1f}s", ok, repr(rows) if not ok else f"{len(rows)} pages described")
    if ok:
        check("qualities valid, purposes within caps", all(r["quality"] in vision.QUALITIES and 0 < len(r["purpose"]) <= 120 and len(r["demo_moment"]) <= 140 for r in rows))
        check("insights written onto app_info", info.get("landing_insight", {}).get("purpose") and any(p.get("insight") for p in info["pages"]))
        print("    " + vision.summary_line(rows))
        t = time.monotonic()
        await vision.thumbnails(rows)
        check(f"thumbnails in {(time.monotonic() - t) * 1000:.0f} ms", all(r.get("thumb") and "screenshot" not in r for r in rows),
              [len(r.get("thumb", "")) for r in rows])
        print("  site map lines the planner now sees:")
        for line in planner._site_map_lines(info).splitlines():
            print("    " + line[:230])
        print("  " + planner._landing_insight_line(info).strip())


async def main():
    await vision_checks()
    planner_checks()
    await prompt_checks()
    await ingest_checks()
    if LIVE:
        await live_checks()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    shutil.rmtree(TMP, ignore_errors=True)
    return 0 if all(results) else 1


sys.exit(asyncio.run(main()))
