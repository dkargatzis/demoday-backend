"""Offline checks for app/services/bookends.py (fal.ai FLUX title cards).

Covers: _image_arguments per model, the cover prompt (product name never sent to the image model), _generate_image on a
fake fal_client + a local HTTP file served without an extension (size cap, empty file, ledger row, 1920x1080 crop),
prepare_bookends(mode="cards") on a fake FLUX PNG (cards rendered BEFORE the image provider runs, one image reused,
notes, shortened intro line, 320 px thumb), the FLUX-failure / timeout / no-key fallbacks, personalized `overrides`,
`video=False` never touching the video endpoint, the cover poster (the painted intro card in cards mode,
prepare_poster for runs without title cards, `paint=False`), and make_bookend composing a card into a 3 s h264/aac
MP4. `LIVE=1` adds one real FLUX call + a composed intro.mp4 in the work dir.

    .venv/bin/python tests/check_bookends.py
"""
import asyncio
import base64
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import wave
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the backend root (tests/..)
WORK = Path(tempfile.mkdtemp(prefix="demoday-bookends-"))
os.environ.update({"STORAGE_DIR": str(WORK / "storage"), "ALLOW_PRIVATE_URLS": ""})
sys.path.insert(0, BACKEND)

from app.models.schemas import JobStatus  # noqa: E402
from app.services import bookends, ledger  # noqa: E402
from app.services.stitcher import describe_media, frame_aligned, make_bookend  # noqa: E402

results = []
REAL = {name: getattr(bookends, name) for name in
        ("_render_cards", "_generate_image", "_generate_clip", "get_secret", "synthesize_scene_audio")}
FAL_IMAGE_TIMEOUT_SEC = bookends.FAL_IMAGE_TIMEOUT_SEC


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + str(detail)[:300] if detail else ''}")


def restore():
    for name, fn in REAL.items():
        setattr(bookends, name, fn)
    bookends.FAL_IMAGE_TIMEOUT_SEC = FAL_IMAGE_TIMEOUT_SEC


def silent_wav(path: Path, seconds: float) -> float:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"\x00" * (int(48000 * seconds) * 4))
    return seconds


class FakeTTS:
    """Replaces synthesize_scene_audio: records the spoken text and writes a silent WAV sized to it."""

    def __init__(self):
        self.texts = {}

    async def __call__(self, text, output_path, voice_style="energetic_tech"):
        self.texts[Path(output_path).stem] = text
        return silent_wav(Path(output_path), round(0.5 + 0.055 * len(text), 2)), "fake TTS"


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def make_image(path: Path, size: str, colour: str = "0x1b2a4a"):
    ffmpeg("-f", "lavfi", "-i", f"color=c={colour}:s={size}", "-frames:v", "1", "-update", "1", str(path))


def dims(path) -> tuple:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.strip()
    w, h = out.split(",")[:2]
    return int(w), int(h)


def install_fake_fal(responses, calls):
    """A fal_client stand-in: subscribe records the call and pops the next canned response (or raises it)."""
    mod = types.ModuleType("fal_client")

    class AsyncClient:
        def __init__(self, key=None, **kw):
            calls.append(("client", key))

        async def subscribe(self, application, arguments, **kw):
            calls.append((application, arguments, kw))
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    mod.AsyncClient = AsyncClient
    sys.modules["fal_client"] = mod


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def norm(bookend, job_dir: Path):
    rel = lambda p: p.relative_to(job_dir).as_posix() if p else None  # noqa: E731
    return (bookend.kind, rel(bookend.narration), bookend.narration_sec, rel(bookend.overlay), rel(bookend.background),
            rel(bookend.video), bookend.note)


ARGS = ("FastAPI", "Fast, modern Python web framework", "https://fastapi.tiangolo.com",
        ["https://github.com/fastapi/fastapi"], "energetic_tech")
VISUALS = {"cover": "soft light trails in deep blue and emerald, shallow depth of field", "intro": None, "outro": None}


async def main():
    print("image arguments")
    a = bookends._image_arguments("fal-ai/flux/schnell", "p")
    check("schnell: 4 steps, one landscape_16_9 image",
          a == {"prompt": "p", "image_size": "landscape_16_9", "num_inference_steps": 4, "num_images": 1}, a)
    a = bookends._image_arguments("fal-ai/flux-pro/v1.1", "p")
    check("flux-pro v1.1: no step count", a == {"prompt": "p", "image_size": "landscape_16_9", "num_images": 1}, a)
    a = bookends._image_arguments("fal-ai/recraft/v3/text-to-image", "p")
    check("other models: prompt + image_size only", a == {"prompt": "p", "image_size": "landscape_16_9"}, a)
    check("FAL_IMAGE_MODEL defaults to schnell",
          bookends.FAL_IMAGE_MODEL == os.getenv("FAL_IMAGE_MODEL", "fal-ai/flux/schnell"), bookends.FAL_IMAGE_MODEL)
    check("Bookend.thumb defaults to None",
          bookends.Bookend("intro", Path("a"), 1.0, Path("b"), Path("c")).thumb is None)

    print("cover prompt")
    p = bookends.cover_prompt("An abstract view of FastAPI's domain: glowing request paths", "FastAPI")
    check("the product name never reaches the image model", "fastapi" not in p.lower(), p)
    check("keeps Gemini's art direction, the house style and the no-text rule",
          p.startswith("An abstract view of domain: glowing request paths") and bookends.COVER_STYLE in p
          and p.endswith(bookends.NO_TEXT), p)
    check("a visual that is only the name falls back to the template",
          bookends.cover_prompt("FastAPI", "FastAPI").startswith(bookends.TEMPLATE_VISUALS["cover"]))
    check("no visual -> template", bookends.cover_prompt(None).startswith(bookends.TEMPLATE_VISUALS["cover"]))

    print("fake image provider: fake fal_client + local HTTP (files served without an extension)")
    serve = WORK / "serve"
    serve.mkdir()
    make_image(serve / "wide.jpg", "1024x576")
    (serve / "wide.jpg").rename(serve / "wide")
    make_image(serve / "square.png", "800x800")
    (serve / "square.png").rename(serve / "square")
    (serve / "empty").write_bytes(b"")
    (serve / "huge").write_bytes(b"\0" * (bookends.MAX_IMAGE_BYTES + 1))
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(serve)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    calls, responses, logs = [], [], []
    install_fake_fal(responses, calls)
    job = JobStatus(job_id="t", status="running")
    ledger.bind(job)
    gen = WORK / "gen"
    gen.mkdir()
    responses.append({"images": [{"url": f"{base}/wide", "width": 1024, "height": 576, "content_type": "image/jpeg"}],
                      "seed": 1, "has_nsfw_concepts": [False]})
    out = await bookends._generate_image("prompt text", gen / "cover.png", "fake-key", logs.append)
    check("returns the destination PNG", out == gen / "cover.png" and out.exists())
    check("a 1024x576 JPEG served without an extension becomes a 1920x1080 PNG", dims(out) == (1920, 1080), dims(out))
    check("the raw download is removed", not any(p.name.startswith("cover_raw") for p in gen.iterdir()), list(gen.iterdir()))
    app_id, arguments, kw = calls[-1]
    check("fal called with the model's arguments and the client timeout",
          app_id == bookends.FAL_IMAGE_MODEL and arguments == bookends._image_arguments(bookends.FAL_IMAGE_MODEL, "prompt text")
          and kw.get("client_timeout") == bookends.FAL_IMAGE_TIMEOUT_SEC and calls[-2] == ("client", "fake-key"), calls)
    row = job.tool_calls[-1]
    check("ledger row: fal / cover art / model / ok",
          (row.tool, row.purpose, row.model, row.ok) == ("fal", "cover art", bookends.FAL_IMAGE_MODEL, True)
          and row.ms is not None, row)
    check("log line names the source size", logs and logs[-1].startswith("Cover art ready in ")
          and logs[-1].endswith("(1024x576, scaled to 1920x1080)"), logs)
    responses.append({"images": [{"url": f"{base}/square"}]})
    out = await bookends._generate_image("p", gen / "square.png", "k")
    check("a square image is scaled and centre-cropped to 1920x1080", dims(out) == (1920, 1080), dims(out))
    for name, expect in (("empty", "empty image"), ("huge", "unexpectedly large")):
        responses.append({"images": [{"url": f"{base}/{name}"}]})
        try:
            await bookends._generate_image("p", gen / f"{name}.png", "k")
            check(f"{name} download rejected", False)
        except RuntimeError as exc:
            check(f"{name} download rejected", expect in str(exc), exc)
    check("failed downloads are ledger fallbacks", job.tool_calls[-1].ok is False and job.tool_calls[-2].ok is False)
    responses.append({"images": []})
    try:
        await bookends._generate_image("p", gen / "none.png", "k")
        check("a response without images is rejected", False)
    except RuntimeError as exc:
        check("a response without images is rejected", "unexpected fal.ai response" in str(exc), exc)
    responses.append(RuntimeError("queue timed out after 90s"))
    try:
        await bookends._generate_image("p", gen / "err.png", "k")
        check("a fal error propagates", False)
    except RuntimeError as exc:
        check("a fal error propagates", str(exc) == "queue timed out after 90s" and job.tool_calls[-1].ok is False, exc)
    check("6 ledger rows for 6 fal calls", len(job.tool_calls) == 6, len(job.tool_calls))
    sys.modules.pop("fal_client", None)
    ledger.bind(None)
    server.shutdown()

    print("prepare_bookends(mode='cards') on a fake FLUX image")
    fake_flux = WORK / "fake_flux.png"
    make_image(fake_flux, "1920x1080", "0x0f4c3a")
    order, captured, captured_kwargs = [], {}, {}

    async def render(cards, job_dir, **kwargs):
        order.append("render:start")
        captured.clear()
        captured.update(cards)
        captured_kwargs.clear()
        captured_kwargs.update(kwargs)
        paths = await REAL["_render_cards"](cards, job_dir, **kwargs)
        order.append("render:done")
        return paths

    async def fake_image(prompt, dest, key, log=None):
        order.append("image")
        shutil.copy(fake_flux, dest)
        if log:
            log("Cover art ready in 0.0s (fake)")
        return dest

    async def must_not(*args, **kwargs):
        raise AssertionError("fal endpoint called")

    def fake_key(name):
        return "fake-key" if name == "FAL_KEY" else ""

    bookends._render_cards, bookends._generate_image, bookends._generate_clip = render, fake_image, must_not
    bookends.get_secret, bookends.synthesize_scene_audio = fake_key, (tts := FakeTTS())
    job_dir = WORK / "cards"
    job_dir.mkdir()
    logs = []
    t0 = time.monotonic()
    b = await bookends.prepare_bookends(job_dir, *ARGS, dict(VISUALS), logs.append, mode="cards")
    elapsed = time.monotonic() - t0
    check("cards are rendered before the image provider runs", order == ["render:start", "render:done", "image"], order)
    check("one image, reused as both backgrounds",
          b["intro"].background == job_dir / "cover.png" == b["outro"].background, (b["intro"].background, b["outro"].background))
    check("no video in cards mode", b["intro"].video is None and b["outro"].video is None)
    check("notes name the FLUX model",
          b["intro"].note == f"fal.ai FLUX background ({bookends.FAL_IMAGE_MODEL})" == b["outro"].note, b["intro"].note)
    check("intro line shortened to the name", tts.texts.get("intro") == "Introducing FastAPI.", tts.texts)
    check("outro line unchanged", tts.texts.get("outro") == "Try FastAPI today at fastapi dot tiangolo dot com.", tts.texts)
    check("narration durations come from the WAVs",
          abs(b["intro"].narration_sec - (0.5 + 0.055 * len("Introducing FastAPI."))) < 0.01, b["intro"].narration_sec)
    check("overlays are 1920x1080 PNGs", dims(b["intro"].overlay) == (1920, 1080) == dims(b["outro"].overlay))
    thumb = base64.b64decode(b["intro"].thumb or "")
    (WORK / "thumb.jpg").write_bytes(thumb)
    check("intro thumb is a 320x180 JPEG (base64)", thumb[:2] == b"\xff\xd8" and dims(WORK / "thumb.jpg") == (320, 180),
          f"{len(thumb)} bytes")
    check("outro carries no thumb", b["outro"].thumb is None)
    check("cover_card.jpg composed for the trace", (job_dir / "cover_card.jpg").exists())
    check("painted cards: the intro card is the poster, a 1280x720 cover.jpg",
          b["intro"].painted and b["outro"].painted and b["intro"].poster == job_dir / "cover.jpg"
          and (job_dir / "cover.jpg").read_bytes()[:2] == b"\xff\xd8" and dims(job_dir / "cover.jpg") == (1280, 720),
          (b["intro"].poster, b["intro"].painted))
    check("the outro carries no poster", b["outro"].poster is None)
    check("log: the painting line", any(line.startswith(f"Cover art: fal.ai FLUX ({bookends.FAL_IMAGE_MODEL}) painting "
                                                      "'soft light trails in deep blue and emerald") and
                                       line.endswith("for the title cards") for line in logs), logs)
    check("log: intro/outro notes", f"Intro: {b['intro'].note}" in logs and f"Outro: {b['outro'].note}" in logs, logs)
    check("default card text: Introducing / project / tagline",
          captured["intro"] == '<div class="kicker">Introducing</div><h1>FastAPI</h1><p>Fast, modern Python web framework</p>',
          captured.get("intro"))
    check("default outro card: host + repo", captured["outro"] ==
          '<div class="kicker">Try it now</div><h1>FastAPI</h1><p>fastapi.tiangolo.com<small>github.com/fastapi/fastapi</small></p>',
          captured.get("outro"))
    print(f"  (cards mode offline run: {elapsed:.1f}s including one Chromium render)")

    print("make_bookend composes the card (background_png + overlay + narration)")
    duration = frame_aligned(max(3.0, b["intro"].narration_sec + 0.4 + 0.6))
    t0 = time.monotonic()
    await make_bookend(job_dir / "intro.mp4", b["intro"].narration, b["intro"].overlay, duration, "in",
                       video_path=None, background_png=b["intro"].background, voice_delay_sec=0.4)
    media = await describe_media(job_dir / "intro.mp4")
    check(f"intro.mp4 is h264/aac 1920x1080 and {duration:.2f}s (encoded in {time.monotonic() - t0:.1f}s)",
          media["video_codec"] == "h264" and media["audio_codec"] == "aac" and (media["width"], media["height"]) == (1920, 1080)
          and abs(media["duration"] - duration) < 0.15, media)

    print("cards mode fallbacks")

    async def failing(prompt, dest, key, log=None):
        raise RuntimeError("queue timed out after 90s\nsecond line")

    bookends._generate_image = failing
    job_dir2 = WORK / "cards_fail"
    job_dir2.mkdir()
    logs = []
    b = await bookends.prepare_bookends(job_dir2, *ARGS, dict(VISUALS), logs.append, mode="cards")
    check("FLUX failure -> gradient backgrounds", b["intro"].background == job_dir2 / "intro_background.png"
          and b["outro"].background == job_dir2 / "outro_background.png", (b["intro"].background, b["outro"].background))
    check("note carries the first line of the reason",
          b["intro"].note == "typeset title on a gradient (FLUX unavailable: queue timed out after 90s)" == b["outro"].note,
          b["intro"].note)
    check("thumb still rendered from the gradient card", bool(b["intro"].thumb))
    check("FLUX failure -> not painted, no poster, no cover.jpg",
          not b["intro"].painted and b["intro"].poster is None and not (job_dir2 / "cover.jpg").exists())
    check("log: the fallback line", "Intro: typeset title on a gradient (FLUX unavailable: queue timed out after 90s)" in logs, logs)

    async def slow(prompt, dest, key, log=None):
        await asyncio.sleep(30)
        return dest

    bookends._generate_image, bookends.FAL_IMAGE_TIMEOUT_SEC = slow, -29.8  # wait_for(..., timeout + 30) -> 0.2 s
    job_dir3 = WORK / "cards_slow"
    job_dir3.mkdir()
    logs = []
    t0 = time.monotonic()
    b = await bookends.prepare_bookends(job_dir3, *ARGS, dict(VISUALS), logs.append, mode="cards")
    check("a hung image call times out into the gradient card",
          b["intro"].note == "typeset title on a gradient (FLUX unavailable: TimeoutError)"
          and b["intro"].background == job_dir3 / "intro_background.png" and time.monotonic() - t0 < 15, (b["intro"].note, logs))
    bookends.FAL_IMAGE_TIMEOUT_SEC = FAL_IMAGE_TIMEOUT_SEC

    bookends._generate_image, bookends.get_secret = must_not, lambda name: ""
    job_dir4 = WORK / "cards_nokey"
    job_dir4.mkdir()
    logs = []
    b = await bookends.prepare_bookends(job_dir4, *ARGS, dict(VISUALS), logs.append, mode="cards")
    check("no FAL_KEY -> gradient card, no fal call",
          b["intro"].note == "typeset title on a gradient (FAL_KEY not set)" == b["outro"].note
          and b["intro"].background == job_dir4 / "intro_background.png" and "Cover art:" not in " ".join(logs), (b["intro"].note, logs))
    check("no FAL_KEY -> no poster", not b["intro"].painted and b["intro"].poster is None
          and not (job_dir4 / "cover.jpg").exists())

    print("overrides (personalized cards)")
    overrides = {
        "intro": {"kicker": "Made for Acme & Co", "title": "FastAPI", "subtitle": "A 60-second tour for Maria",
                  "small": "Head of Platform at Acme & Co", "line": "Hi Maria. This is FastAPI, made for Acme and Co."},
        "outro": {"kicker": "Your next step", "title": "Book 15 minutes", "subtitle": "cal.com/founder/15min",
                  "small": "Made for Acme & Co · FastAPI",
                  "line": "If this would help at Acme and Co, grab fifteen minutes with us at the link on screen."},
        "cover": {"title": "ignored"},
    }
    bookends._generate_image, bookends.get_secret, bookends.synthesize_scene_audio = fake_image, fake_key, (tts := FakeTTS())
    job_dir5 = WORK / "cards_overrides"
    job_dir5.mkdir()
    logs = []
    b = await bookends.prepare_bookends(job_dir5, *ARGS, dict(VISUALS), logs.append, mode="cards", overrides=overrides,
                                        video=False)
    check("intro card typeset from the override, HTML-escaped", captured["intro"] ==
          '<div class="kicker">Made for Acme &amp; Co</div><h1>FastAPI</h1>'
          '<p>A 60-second tour for Maria<small>Head of Platform at Acme &amp; Co</small></p>', captured.get("intro"))
    check("outro card typeset from the override", captured["outro"] ==
          '<div class="kicker">Your next step</div><h1>Book 15 minutes</h1>'
          '<p>cal.com/founder/15min<small>Made for Acme &amp; Co · FastAPI</small></p>', captured.get("outro"))
    check("spoken lines come from the overrides",
          tts.texts == {"intro": overrides["intro"]["line"], "outro": overrides["outro"]["line"]}, tts.texts)
    check("unknown kinds ignored; FLUX background still used with video=False",
          b["intro"].background == job_dir5 / "cover.png" and b["intro"].note.startswith("fal.ai FLUX background"), b["intro"].note)
    tts = FakeTTS()
    bookends.synthesize_scene_audio = tts
    job_dir6 = WORK / "cards_partial"
    job_dir6.mkdir()
    b = await bookends.prepare_bookends(job_dir6, *ARGS, dict(VISUALS), logs.append, mode="cards",
                                        overrides={"intro": {"kicker": "Made for Acme"}})
    check("partial override: kicker only, title defaults to the project, line unchanged",
          captured["intro"] == '<div class="kicker">Made for Acme</div><h1>FastAPI</h1>'
          and tts.texts["intro"] == "Introducing FastAPI." and "Try it now" in captured["outro"], (captured.get("intro"), tts.texts))

    print("video=False never calls the video endpoint (mode='video', FAL_KEY set)")
    bookends._generate_clip = bookends._generate_image = must_not
    job_dir7 = WORK / "video_false"
    job_dir7.mkdir()
    logs = []
    b = await bookends.prepare_bookends(job_dir7, *ARGS, dict(VISUALS), logs.append, overrides=overrides, video=False)
    check("gradient cards, no clip, the 'no AI video requested' note",
          b["intro"].video is None and b["intro"].background == job_dir7 / "intro_background.png"
          and b["intro"].note == "typeset title card (no AI video requested)" == b["outro"].note, (b["intro"].note, logs))
    check("log line", "Intro: typeset title card (no AI video requested)" in logs, logs)
    check("no thumb in video mode", b["intro"].thumb is None)

    print("paint=False (FEATURE_COVER=0): the cards stay on the gradient even with a key")
    order.clear()
    bookends._render_cards, bookends._generate_image, bookends._generate_clip = render, fake_image, must_not
    bookends.get_secret, bookends.synthesize_scene_audio = fake_key, FakeTTS()
    jd = WORK / "cards_unpainted"
    jd.mkdir()
    logs = []
    b = await bookends.prepare_bookends(jd, *ARGS, dict(VISUALS), logs.append, mode="cards", paint=False)
    check("no image call, the 'FEATURE_COVER is off' note, no poster",
          "image" not in order and b["intro"].note == "typeset title on a gradient (FEATURE_COVER is off)" == b["outro"].note
          and not b["intro"].painted and b["intro"].poster is None and not (jd / "cover.jpg").exists(), (order, b["intro"].note))

    print("prepare_poster (a run without title cards)")
    prompts = []

    async def fake_image_prompt(prompt, dest, key, log=None):
        prompts.append(prompt)
        return await fake_image(prompt, dest, key, log)

    order.clear()
    bookends._generate_image = fake_image_prompt
    jd = WORK / "poster"
    jd.mkdir()
    logs = []
    t0 = time.monotonic()
    p = await bookends.prepare_poster(jd, "FastAPI", "Fast, modern Python web framework.", VISUALS["cover"], logs.append)
    took = time.monotonic() - t0
    check("the title is typeset before the image provider runs", order == ["render:start", "render:done", "image"], order)
    check("cover.jpg is a 1280x720 JPEG", p.path == jd / "cover.jpg" and p.path.read_bytes()[:2] == b"\xff\xd8"
          and dims(p.path) == (1280, 720) == (bookends.POSTER_WIDTH, bookends.POSTER_HEIGHT), f"{dims(p.path)}, {took:.1f}s")
    check("default title: 'Product demo', the name, the tagline without its full stop",
          set(captured) == {"poster"} and '<div class="kicker">Product demo</div>' in captured["poster"]
          and "<h1>FastAPI</h1>" in captured["poster"] and "<p>Fast, modern Python web framework</p>" in captured["poster"],
          captured)
    check("the image model gets the art direction, never the product name",
          len(prompts) == 1 and prompts[0].startswith(VISUALS["cover"]) and "FastAPI" not in prompts[0], prompts)
    (WORK / "poster_thumb.jpg").write_bytes(base64.b64decode(p.thumb or ""))
    check("note names the model; the trace thumb is 320x180",
          p.note == f"fal.ai FLUX poster ({bookends.FAL_IMAGE_MODEL})" and dims(WORK / "poster_thumb.jpg") == (320, 180), p.note)
    check("log: the painting line and the ready line",
          f"Cover art: fal.ai FLUX ({bookends.FAL_IMAGE_MODEL}) painting '{VISUALS['cover']}' for the poster" in logs
          and any(line.startswith("Cover poster ready: 1280x720, ") and line.endswith(" kB") for line in logs), logs)

    order.clear()
    jd = WORK / "poster_card"
    jd.mkdir()
    card = {"kicker": "Made for Acme", "title": "FastAPI", "subtitle": "A 60-second tour for Maria",
            "small": "Head of Platform at Acme", "line": "never typeset"}
    p = await bookends.prepare_poster(jd, "FastAPI", "ignored tagline", None, logs.append, card=card)
    check("a personalized card is typeset instead (kicker, title, subtitle, small)",
          '<div class="kicker">Made for Acme</div>' in captured["poster"] and "<small>Head of Platform at Acme</small>" in captured["poster"]
          and "ignored tagline" not in captured["poster"] and "never typeset" not in captured["poster"] and p.path.exists(), captured)
    check("no art direction -> the template visual", prompts[-1].startswith(bookends.TEMPLATE_VISUALS["cover"]), prompts[-1][:80])

    order.clear()
    bookends.get_secret = lambda name: ""
    jd = WORK / "poster_nokey"
    jd.mkdir()
    try:
        await bookends.prepare_poster(jd, "FastAPI", "t", None, logs.append)
        check("no FAL_KEY raises before Chromium or fal", False)
    except RuntimeError as exc:
        check("no FAL_KEY raises before Chromium or fal", str(exc) == "FAL_KEY not set" and order == [] and not any(jd.iterdir()),
              (exc, order))
    bookends.get_secret, bookends._generate_image = fake_key, failing
    jd = WORK / "poster_fail"
    jd.mkdir()
    try:
        await bookends.prepare_poster(jd, "FastAPI", "t", None, logs.append)
        check("a FLUX failure propagates (the orchestrator records the reason, skips the step)", False)
    except RuntimeError as exc:
        check("a FLUX failure propagates (the orchestrator records the reason, skips the step)",
              str(exc).startswith("queue timed out after 90s") and not (jd / "cover.jpg").exists(), exc)

    print("the product's own screens as the backdrop")

    def pixel(path, x, y, fmt="rgb24", size=3):
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"crop=1:1:{x}:{y}", "-f", "rawvideo",
                              "-pix_fmt", fmt, "-"], capture_output=True, check=True).stdout
        return tuple(out[:size])

    def row_alpha(path, y):
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"crop=1920:1:0:{y}", "-f", "rawvideo",
                              "-pix_fmt", "rgba", "-"], capture_output=True, check=True).stdout
        return max(out[3::4]) if out else 0

    shot_a, shot_b = WORK / "shot_a.jpg", WORK / "shot_b.jpg"
    for path, colour in ((shot_a, "0xd02020"), (shot_b, "0x2050d0")):
        ffmpeg("-f", "lavfi", "-i", f"color=c={colour}:s=1920x1080", "-frames:v", "1", "-q:v", "3", "-update", "1", str(path))
    order.clear()
    bookends._render_cards, bookends._generate_image, bookends._generate_clip = render, fake_image, must_not
    bookends.get_secret, bookends.synthesize_scene_audio = fake_key, FakeTTS()
    jd = WORK / "cards_product"
    jd.mkdir()
    logs = []
    b = await bookends.prepare_bookends(jd, *ARGS, dict(VISUALS), logs.append, mode="cards",
                                        product_shots=[(shot_a, "/"), (shot_b, "/rules")], window_host="acme.dev",
                                        theme_color="#5E6AD2")
    check("the image provider is never called when the product's screens exist", "image" not in order, order)
    check("each card names the screen it frames", b["intro"].note == "the product's own screen (/) in a browser frame"
          and b["outro"].note == "the product's own screen (/rules) in a browser frame", (b["intro"].note, b["outro"].note))
    check("the address bar, the theme glow and both screens reach the renderer", captured_kwargs.get("host") == "acme.dev"
          and captured_kwargs.get("glow") == "rgba(94,106,210,.42)" and set(captured_kwargs.get("shots") or {}) == {"intro", "outro"},
          captured_kwargs)
    intro_px, outro_px, top_px = (pixel(b["intro"].background, 960, 760), pixel(b["outro"].background, 960, 760),
                                  pixel(b["intro"].background, 960, 150))
    check("the intro window shows the first screen (red)", intro_px[0] > 150 and max(intro_px[1:]) < 90, intro_px)
    check("the outro window shows the second screen (blue)", outro_px[2] > 150 and outro_px[0] < 90, outro_px)
    check("the stage is dark above the window, where the title sits", max(top_px) < 60, top_px)
    check("the title is typeset above the window, nothing over the screen",
          row_alpha(b["intro"].overlay, 215) > 200 and row_alpha(b["intro"].overlay, 820) == 0,
          (row_alpha(b["intro"].overlay, 215), row_alpha(b["intro"].overlay, 820)))
    check("framed counts as painted: the intro card is the 1280x720 poster", b["intro"].painted and b["outro"].painted
          and b["intro"].poster == jd / "cover.jpg" and dims(jd / "cover.jpg") == (1280, 720))
    order.clear()
    jd = WORK / "cards_unreadable"
    jd.mkdir()
    b = await bookends.prepare_bookends(jd, *ARGS, dict(VISUALS), logs.append, mode="cards",
                                        product_shots=[(WORK / "missing.jpg", "/")])
    check("an unreadable screenshot falls back to the FLUX still", "image" in order
          and b["intro"].note == f"fal.ai FLUX background ({bookends.FAL_IMAGE_MODEL})", (order, b["intro"].note))
    order.clear()
    bookends.get_secret = lambda name: ""
    jd = WORK / "poster_product"
    jd.mkdir()
    p = await bookends.prepare_poster(jd, "FastAPI", "Fast, modern Python web framework.", None, logs.append,
                                      product_shot=(shot_a, "/docs"), window_host="acme.dev")
    check("the poster frames the product's screen with no key and no image call",
          p.path == jd / "cover.jpg" and dims(p.path) == (1280, 720) and "image" not in order
          and p.note == "the product's own screen (/docs) in a browser frame" and p.thumb, (p.note, order))
    check("theme colours: a real colour glows, white, black, grey and junk fall back to emerald",
          bookends.glow_colour("#5E6AD2") == "rgba(94,106,210,.42)" and bookends.glow_colour("#0a0") == "rgba(0,170,0,.42)"
          and {bookends.glow_colour(c) for c in ("#ffffff", "#000", "#888888", "", "teal", "#12")} == {bookends.HOUSE_GLOW})
    restore()

    if os.getenv("LIVE") == "1":
        print("LIVE: one real FLUX background + a composed intro card (make_bookend)")
        sys.modules.pop("fal_client", None)
        from app.config import get_secret as real_secret
        key = real_secret("FAL_KEY")
        check("FAL_KEY configured", bool(key))
        live = WORK / "flux_live"
        live.mkdir(exist_ok=True)
        job = JobStatus(job_id="live", status="running")
        ledger.bind(job)
        logs = []
        prompt = bookends.cover_prompt("soft light trails in deep blue and emerald, shallow depth of field, premium "
                                       "technology launch mood", "FastAPI")
        t0 = time.monotonic()
        png = await bookends._generate_image(prompt, live / "cover.png", key, logs.append)
        t_flux = time.monotonic() - t0
        row = job.tool_calls[-1]
        print(f"  {bookends.FAL_IMAGE_MODEL}: {t_flux:.1f}s end to end (ledger {row.ms} ms, ok={row.ok}); {logs}")
        check("live: a 1920x1080 PNG", dims(png) == (1920, 1080), dims(png))
        cards = {"intro": bookends._title_content("Introducing", "FastAPI", "Fast, modern Python web framework")}
        t0 = time.monotonic()
        paths = await bookends._render_cards(cards, live)
        t_cards = time.monotonic() - t0
        wav = live / "intro.wav"
        duration = frame_aligned(max(3.0, silent_wav(wav, 1.4) + 0.4 + 0.6))
        t0 = time.monotonic()
        await make_bookend(live / "intro.mp4", wav, paths["intro"]["overlay"], duration, "in", background_png=png,
                           voice_delay_sec=0.4)
        t_enc = time.monotonic() - t0
        media = await describe_media(live / "intro.mp4")
        ffmpeg("-ss", "1.5", "-i", str(live / "intro.mp4"), "-frames:v", "1", "-update", "1", str(live / "intro_frame.jpg"))
        check("live: intro.mp4 is h264/aac 1920x1080", media["video_codec"] == "h264" and media["audio_codec"] == "aac"
              and (media["width"], media["height"]) == (1920, 1080) and abs(media["duration"] - duration) < 0.15, media)
        print(f"  cards {t_cards:.1f}s · encode {t_enc:.1f}s · {media['duration']:.2f}s clip\n"
              f"  output: {live / 'intro.mp4'}\n  background: {png}\n  frame at 1.5s: {live / 'intro_frame.jpg'}")
        ledger.bind(None)

    print(f"\n{sum(results)}/{len(results)} checks passed (work dir {WORK})")
    return 0 if all(results) else 1


sys.exit(asyncio.run(main()))
