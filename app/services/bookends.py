"""Intro and outro ("bookends") for the demo video.

Each bookend is a typeset title card (real text rendered by the browser, never drawn by a model) on top of
- `mode="cards"`: one fal.ai FLUX still painted from Gemini's one-line art direction and reused for both cards, or
- `mode="video"`: a short fal.ai text-to-video clip per card (opt-in through ai_clips),
plus one narrated line in the same voice as the demo. Without fal.ai, or when it fails, the card sits on a slowly
zooming gradient, so the demo never depends on it. The cards are rendered FIRST, before any fal.ai wait, so Chromium
never runs during the timed browser recording that overlaps this work.
"""
import asyncio
import base64
import html
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import httpx
from playwright.async_api import async_playwright

from app.config import get_secret
from app.services import ledger
from app.services.audio import synthesize_scene_audio
from app.services.recorder import launch_chromium
from app.services.stitcher import HEIGHT, WIDTH, run_ffmpeg

logger = logging.getLogger("demoday.bookends")

# Seedance 1.0 Pro Fast: native 1080p 16:9, ~$0.25 per 5 s clip. Kling ("fal-ai/kling-video/v3/pro/text-to-video")
# and Veo endpoints also work; fal's Veo endpoints are deprecated as of November 2026.
FAL_VIDEO_MODEL = os.getenv("FAL_VIDEO_MODEL", "fal-ai/bytedance/seedance/v1/pro/fast/text-to-video")
CLIP_SECONDS = 5
FAL_VIDEO_TIMEOUT_SEC = 300
MAX_VIDEO_BYTES = 150 * 1024 * 1024
# FLUX schnell: 1024x576 JPEG in ~2 s warm (4 steps). "fal-ai/flux-pro/v1.1" is the slower, richer option.
FAL_IMAGE_MODEL = os.getenv("FAL_IMAGE_MODEL", "fal-ai/flux/schnell")
FAL_IMAGE_TIMEOUT_SEC = 90
MAX_IMAGE_BYTES = 20 * 1024 * 1024
NO_TEXT = ("No text, letters, numbers, logos, watermarks, screens, user interfaces, people, faces or hands anywhere "
           "in the frame.")
NEGATIVE = "text, letters, words, numbers, logo, watermark, user interface, screen, people, face, hands, blur, distortion"
COVER_STYLE = "Soft dark cinematic abstract background, subtle depth, empty centre."
TEMPLATE_VISUALS = {
    "intro": ("Slow cinematic dolly shot through soft glowing light trails and floating abstract geometric shapes in "
              "deep blue and emerald, shallow depth of field, premium technology launch mood."),
    "outro": ("Slow pull-back from a softly glowing abstract light sculpture in deep blue and emerald, gentle "
              "particles drifting, calm premium technology launch mood, fading towards darkness."),
    "cover": ("Soft glowing light trails and floating abstract geometric shapes in deep blue and emerald, shallow "
              "depth of field, premium technology launch mood."),
}


@dataclass
class Bookend:
    kind: str                      # "intro" or "outro"
    narration: Path
    narration_sec: float
    overlay: Path                  # transparent PNG with the typeset title
    background: Path               # opaque PNG used when there is no AI clip (the FLUX still in cards mode)
    video: Optional[Path] = None   # the AI clip, when generation succeeded
    note: str = ""
    thumb: Optional[str] = None    # base64 320 px JPEG of the finished intro card, for the one-off step event only


def _video_arguments(prompt: str) -> Dict:
    model = FAL_VIDEO_MODEL
    if "seedance" in model:
        return {"prompt": prompt, "duration": str(CLIP_SECONDS), "resolution": "1080p", "aspect_ratio": "16:9"}
    if "kling" in model:
        return {"prompt": prompt, "duration": str(CLIP_SECONDS), "aspect_ratio": "16:9", "negative_prompt": NEGATIVE,
                "generate_audio": False}
    if "veo" in model:
        return {"prompt": prompt, "duration": "6s", "resolution": "1080p", "aspect_ratio": "16:9",
                "negative_prompt": NEGATIVE, "generate_audio": False}
    return {"prompt": prompt, "aspect_ratio": "16:9"}


def _image_arguments(model: str, prompt: str) -> Dict:
    """Request body per image model: FLUX schnell is tuned for 4 steps; dev/pro take their own step defaults."""
    if "schnell" in model:
        return {"prompt": prompt, "image_size": "landscape_16_9", "num_inference_steps": 4, "num_images": 1}
    if "flux" in model:
        return {"prompt": prompt, "image_size": "landscape_16_9", "num_images": 1}
    return {"prompt": prompt, "image_size": "landscape_16_9"}


def cover_prompt(visual: Optional[str], project: str = "") -> str:
    """The image prompt: Gemini's art direction (or the template), the house style and the no-text rule.

    The product name never goes in (FLUX draws the words it is given); the browser-typeset overlay stays the only
    text on the card, so the name can never be misspelt.
    """
    text = " ".join((visual or "").split())
    if project.strip():
        text = " ".join(re.sub(re.escape(project.strip()) + r"(?:'s)?", "", text, flags=re.IGNORECASE).split())
    return f"{text or TEMPLATE_VISUALS['cover']} {COVER_STYLE} {NO_TEXT}"


async def _download(url: str, dest: Path, max_bytes: int, what: str) -> None:
    """Streams `url` into `dest` with a size cap. The file is later decoded by content, never by the URL's extension."""
    if url.startswith("data:"):  # fal.ai inlines small results as data URIs in sync mode
        payload = base64.b64decode(url.split(",", 1)[1])
        if len(payload) > max_bytes:
            raise RuntimeError(f"the generated {what} is unexpectedly large")
        if not payload:
            raise RuntimeError(f"fal.ai returned an empty {what}")
        dest.write_bytes(payload)
        return
    size = 0
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as http:
        async with http.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as out:
                async for chunk in resp.aiter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        raise RuntimeError(f"the generated {what} is unexpectedly large")
                    out.write(chunk)
    if size == 0:
        raise RuntimeError(f"fal.ai returned an empty {what}")


async def _generate_clip(prompt: str, dest: Path, key: str) -> Path:
    import fal_client

    client = fal_client.AsyncClient(key=key)
    async with ledger.call("fal", "AI clip", FAL_VIDEO_MODEL):
        result = await client.subscribe(FAL_VIDEO_MODEL, arguments=_video_arguments(f"{prompt} {NO_TEXT}"),
                                        client_timeout=FAL_VIDEO_TIMEOUT_SEC)
        video = (result or {}).get("video") or {}
        url = video.get("url") if isinstance(video, dict) else None
        if not url:
            raise RuntimeError(f"unexpected fal.ai response: {str(result)[:160]}")
        await _download(url, dest, MAX_VIDEO_BYTES, "video")
    return dest


async def _generate_image(prompt: str, dest: Path, key: str, log=None) -> Path:
    """One fal.ai text-to-image call -> `dest`, an opaque 1920x1080 PNG.

    The image is scaled and centre-cropped by FFmpeg (one frame, ~0.2 s) so make_bookend's plain `scale=1920:1080`
    never distorts a non-16:9 picture. The download is streamed with a size cap and decoded by content.
    """
    import fal_client

    started = time.monotonic()
    client = fal_client.AsyncClient(key=key)
    raw = dest.with_name(dest.stem + "_raw.bin")
    async with ledger.call("fal", "cover art", FAL_IMAGE_MODEL):
        result = await client.subscribe(FAL_IMAGE_MODEL, arguments=_image_arguments(FAL_IMAGE_MODEL, prompt),
                                        client_timeout=FAL_IMAGE_TIMEOUT_SEC)
        images = (result or {}).get("images") or []
        image = images[0] if images and isinstance(images[0], dict) else {}
        url = image.get("url")
        if not url:
            raise RuntimeError(f"unexpected fal.ai response: {str(result)[:160]}")
        await _download(url, raw, MAX_IMAGE_BYTES, "image")
    await run_ffmpeg(["-i", str(raw), "-vf", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
                      f"crop={WIDTH}:{HEIGHT}", "-frames:v", "1", "-update", "1", str(dest)], timeout=60)
    raw.unlink(missing_ok=True)
    if log:
        source = f"{image['width']}x{image['height']}, " if image.get("width") and image.get("height") else ""
        log(f"Cover art ready in {time.monotonic() - started:.1f}s ({source}scaled to {WIDTH}x{HEIGHT})")
    return dest


TITLE_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
html, body { margin: 0; width: 1920px; height: 1080px; background: __BACKGROUND__; overflow: hidden; }
.scrim { position: absolute; inset: 0;
  background: radial-gradient(ellipse 62% 54% at center, rgba(0,0,0,.68) 0%, rgba(0,0,0,.42) 50%, rgba(0,0,0,0) 85%); }
.wrap { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;
  padding: 0 180px; text-align: center; color: #fff;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Helvetica, Arial, sans-serif; }
.kicker { font-size: 30px; letter-spacing: 9px; text-transform: uppercase; color: #6ee7b7; font-weight: 600;
  margin-bottom: 30px; text-shadow: 0 2px 18px rgba(0,0,0,.6); }
h1 { font-size: 120px; line-height: 1.04; margin: 0; font-weight: 800; letter-spacing: -2px;
  text-shadow: 0 6px 44px rgba(0,0,0,.65); }
p { font-size: 46px; line-height: 1.3; margin: 30px 0 0; font-weight: 500; color: rgba(255,255,255,.9);
  text-shadow: 0 4px 26px rgba(0,0,0,.65); }
small { display: block; font-size: 32px; margin-top: 18px; color: rgba(255,255,255,.7); font-weight: 500; }
</style></head><body><div class="scrim"></div><div class="wrap">__CONTENT__</div></body></html>"""
OPAQUE_BACKGROUND = ("radial-gradient(1100px 650px at 28% 30%, #0f4c3a 0%, rgba(8,9,12,0) 70%), "
                     "radial-gradient(900px 600px at 75% 75%, #0c2f4a 0%, rgba(8,9,12,0) 70%), #08090c")


def _title_content(kicker: str, title: str, subtitle: str, small: str = "") -> str:
    parts = [f'<div class="kicker">{html.escape(kicker)}</div>', f"<h1>{html.escape(title)}</h1>"]
    if subtitle:
        parts.append(f"<p>{html.escape(subtitle)}{f'<small>{html.escape(small)}</small>' if small else ''}</p>")
    return "".join(parts)


async def _render_cards(cards: Dict[str, str], job_dir: Path) -> Dict[str, Dict[str, Path]]:
    """Renders each card twice: a transparent overlay for the AI clip and an opaque fallback background."""
    paths: Dict[str, Dict[str, Path]] = {}
    async with async_playwright() as pw:
        browser = await launch_chromium(pw)
        try:
            page = await browser.new_page(viewport={"width": 1920, "height": 1080})
            for kind, content in cards.items():
                paths[kind] = {}
                for variant, background, omit in (("overlay", "transparent", True),
                                                  ("background", OPAQUE_BACKGROUND, False)):
                    markup = content if variant == "overlay" else ""
                    await page.set_content(TITLE_HTML.replace("__BACKGROUND__", background)
                                           .replace("__CONTENT__", markup))
                    path = job_dir / f"{kind}_{variant}.png"
                    await page.screenshot(path=str(path), omit_background=omit)
                    paths[kind][variant] = path
        finally:
            await browser.close()
    return paths


async def _card_thumb(background: Path, overlay: Path, dest: Path, width: int = 320) -> Optional[str]:
    """The finished card (background + typeset overlay) as a `width`-pixel base64 JPEG for the agent trace."""
    try:
        await run_ffmpeg(["-i", str(background), "-i", str(overlay), "-filter_complex",
                          f"[0:v]scale={WIDTH}:{HEIGHT}[bg];[bg][1:v]overlay=0:0,scale={width}:-2,format=yuv420p[v]",
                          "-map", "[v]", "-q:v", "6", "-frames:v", "1", "-update", "1", str(dest)], timeout=60)
        return base64.b64encode(dest.read_bytes()).decode("ascii")
    except Exception as exc:  # a missing thumbnail only costs the trace its picture
        logger.warning("card thumbnail failed: %s", exc)
        return None


def _spoken(text: str) -> str:
    """Names and hosts as a voice should say them: 'shadcn/ui' -> 'shadcn ui', 'ui.shadcn.com' -> 'ui dot shadcn dot com'."""
    return " ".join(text.replace("/", " ").replace("_", " ").split())


def _reason(exc: BaseException) -> str:
    return str(exc).strip().splitlines()[0][:140] if str(exc).strip() else type(exc).__name__


async def prepare_bookends(job_dir: Path, project: str, tagline: str, app_url: str, repo_urls, voice_style: str,
                           visuals: Dict[str, Optional[str]], log, mode: str = "video",
                           overrides: Optional[Dict[str, Dict[str, str]]] = None,
                           video: bool = True) -> Dict[str, Bookend]:
    """Title cards, narration and the fal.ai media for the intro and outro.

    `mode="video"`: one fal.ai text-to-video clip per card when FAL_KEY is set and `video` is True (today's AI clips);
    `video=False` never calls the video endpoint (a personalized run without ai_clips: typeset cards, no credits).
    `mode="cards"`: one fal.ai FLUX still from `visuals["cover"]` (Gemini's art direction; `intro` or the template
    when absent) is the background of both cards; the video endpoint is never called.
    `overrides` replaces a card's text and spoken line per kind: {"intro": {"kicker", "title", "subtitle", "small",
    "line"}}, e.g. "Made for Acme". The cards are rendered before any fal.ai wait (see the module docstring).
    """
    host = urlparse(app_url).netloc or app_url
    repo = urlparse(repo_urls[0]).path.strip("/") if repo_urls else ""
    tagline = tagline.strip().rstrip(".")
    short_tagline = tagline if 0 < len(tagline.split()) <= 12 else ""
    lines = {
        "intro": f"Introducing {_spoken(project)}." + (f" {short_tagline}." if short_tagline else ""),
        "outro": f"Try {_spoken(project)} today at {host.replace('.', ' dot ')}.",
    }
    if mode == "cards":  # a ~3 s card: the tagline is on the card, not in the voice
        lines["intro"] = f"Introducing {_spoken(project)}."
    cards = {
        "intro": _title_content("Introducing", project, tagline[:110]),
        "outro": _title_content("Try it now", project, host, f"github.com/{repo}" if repo else ""),
    }
    for kind, card in (overrides or {}).items():
        if kind in cards:
            lines[kind] = card.get("line") or lines[kind]
            cards[kind] = _title_content(card.get("kicker") or "", card.get("title") or project,
                                         card.get("subtitle") or "", card.get("small") or "")
    key = get_secret("FAL_KEY")

    # Chromium first, before any fal.ai wait: this task overlaps the narration and the timed recording.
    card_paths = await _render_cards(cards, job_dir)

    async def clip(kind: str) -> Optional[Path]:
        if not key or not video:
            return None
        prompt = (visuals.get(kind) or TEMPLATE_VISUALS[kind]).strip()
        return await asyncio.wait_for(_generate_clip(prompt, job_dir / f"{kind}_ai.mp4", key),
                                      timeout=FAL_VIDEO_TIMEOUT_SEC + 30)

    async def cover() -> Optional[Path]:
        if not key:
            return None
        visual = " ".join((visuals.get("cover") or visuals.get("intro") or TEMPLATE_VISUALS["cover"]).split())
        log(f"Cover art: fal.ai {'FLUX' if 'flux' in FAL_IMAGE_MODEL else 'image model'} ({FAL_IMAGE_MODEL}) painting "
            f"'{visual[:100].rstrip('. ')}' for the title cards")
        return await asyncio.wait_for(_generate_image(cover_prompt(visual, project), job_dir / "cover.png", key, log),
                                      timeout=FAL_IMAGE_TIMEOUT_SEC + 30)

    if mode == "cards":
        shared = asyncio.create_task(cover())  # one image, reused for both cards
        tasks = {"intro": shared, "outro": shared}
    else:
        tasks = {kind: asyncio.create_task(clip(kind)) for kind in ("intro", "outro")}
    try:
        narrations = await asyncio.gather(*(synthesize_scene_audio(lines[k], job_dir / f"{k}.wav", voice_style)
                                            for k in ("intro", "outro")))
    except BaseException:
        for task in set(tasks.values()):
            task.cancel()
        raise

    bookends: Dict[str, Bookend] = {}
    for kind, (duration, _engine) in zip(("intro", "outro"), narrations):
        media, background, note = None, card_paths[kind]["background"], ""
        try:
            media = await tasks[kind]
            if mode == "cards":
                if media:
                    background = media
                    note = f"fal.ai {'FLUX' if 'flux' in FAL_IMAGE_MODEL else 'image'} background ({FAL_IMAGE_MODEL})"
                else:
                    note = "typeset title on a gradient (FAL_KEY not set)"
            elif media:
                note = f"AI clip from {FAL_VIDEO_MODEL}"
            else:
                note = "typeset title card (FAL_KEY not set)" if video else "typeset title card (no AI video requested)"
        except Exception as exc:
            if mode == "cards":
                if kind == "intro":
                    logger.warning("cover art failed: %s", exc)
                note = f"typeset title on a gradient (FLUX unavailable: {_reason(exc)})"
            else:
                logger.warning("AI %s clip failed: %s", kind, exc)
                note = f"typeset title card (AI clip unavailable: {_reason(exc)})"
        bookends[kind] = Bookend(kind=kind, narration=job_dir / f"{kind}.wav", narration_sec=duration,
                                 overlay=card_paths[kind]["overlay"], background=background,
                                 video=media if mode != "cards" else None, note=note)
        log(f"{kind.capitalize()}: {note}")
    if mode == "cards":
        intro = bookends["intro"]
        intro.thumb = await _card_thumb(intro.background, intro.overlay, job_dir / "cover_card.jpg")
    return bookends
