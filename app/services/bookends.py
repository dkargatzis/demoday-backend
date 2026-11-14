"""AI intro and outro ("bookends") for the demo video.

Each bookend is a short fal.ai text-to-video clip with a typeset title on top (real text rendered by the browser,
never drawn by the model) and one narrated line in the same voice as the demo. If the clip can't be generated, the
bookend falls back to the typeset title on a slowly zooming gradient, so the demo never depends on it.
"""
import asyncio
import html
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import httpx
from playwright.async_api import async_playwright

from app.config import get_secret
from app.services.audio import synthesize_scene_audio
from app.services.recorder import launch_chromium

logger = logging.getLogger("demoday.bookends")

# Seedance 1.0 Pro Fast: native 1080p 16:9, ~$0.25 per 5 s clip. Kling ("fal-ai/kling-video/v3/pro/text-to-video")
# and Veo endpoints also work; fal's Veo endpoints are deprecated as of November 2026.
FAL_VIDEO_MODEL = os.getenv("FAL_VIDEO_MODEL", "fal-ai/bytedance/seedance/v1/pro/fast/text-to-video")
CLIP_SECONDS = 5
FAL_VIDEO_TIMEOUT_SEC = 300
MAX_VIDEO_BYTES = 150 * 1024 * 1024
NO_TEXT = ("No text, letters, numbers, logos, watermarks, screens, user interfaces, people, faces or hands anywhere "
           "in the frame.")
NEGATIVE = "text, letters, words, numbers, logo, watermark, user interface, screen, people, face, hands, blur, distortion"
TEMPLATE_VISUALS = {
    "intro": ("Slow cinematic dolly shot through soft glowing light trails and floating abstract geometric shapes in "
              "deep blue and emerald, shallow depth of field, premium technology launch mood."),
    "outro": ("Slow pull-back from a softly glowing abstract light sculpture in deep blue and emerald, gentle "
              "particles drifting, calm premium technology launch mood, fading towards darkness."),
}


@dataclass
class Bookend:
    kind: str                      # "intro" or "outro"
    narration: Path
    narration_sec: float
    overlay: Path                  # transparent PNG with the typeset title
    background: Path               # opaque PNG used when there is no AI clip
    video: Optional[Path] = None   # the AI clip, when generation succeeded
    note: str = ""


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


async def _generate_clip(prompt: str, dest: Path, key: str) -> Path:
    import fal_client

    client = fal_client.AsyncClient(key=key)
    result = await client.subscribe(FAL_VIDEO_MODEL, arguments=_video_arguments(f"{prompt} {NO_TEXT}"),
                                    client_timeout=FAL_VIDEO_TIMEOUT_SEC)
    video = (result or {}).get("video") or {}
    url = video.get("url") if isinstance(video, dict) else None
    if not url:
        raise RuntimeError(f"unexpected fal.ai response: {str(result)[:160]}")
    size = 0
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as http:
        async with http.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as out:
                async for chunk in resp.aiter_bytes():
                    size += len(chunk)
                    if size > MAX_VIDEO_BYTES:
                        raise RuntimeError("the generated video is unexpectedly large")
                    out.write(chunk)
    if size == 0:
        raise RuntimeError("fal.ai returned an empty video")
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


def _spoken(text: str) -> str:
    """Names and hosts as a voice should say them: 'shadcn/ui' -> 'shadcn ui', 'ui.shadcn.com' -> 'ui dot shadcn dot com'."""
    return " ".join(text.replace("/", " ").replace("_", " ").split())


async def prepare_bookends(job_dir: Path, project: str, tagline: str, app_url: str, repo_urls, voice_style: str,
                           visuals: Dict[str, Optional[str]], log) -> Dict[str, Bookend]:
    """Narration, title cards and (when fal.ai is configured) the AI clips for the intro and outro, in parallel."""
    host = urlparse(app_url).netloc or app_url
    repo = urlparse(repo_urls[0]).path.strip("/") if repo_urls else ""
    tagline = tagline.strip().rstrip(".")
    short_tagline = tagline if 0 < len(tagline.split()) <= 12 else ""
    lines = {
        "intro": f"Introducing {_spoken(project)}." + (f" {short_tagline}." if short_tagline else ""),
        "outro": f"Try {_spoken(project)} today at {host.replace('.', ' dot ')}.",
    }
    cards = {
        "intro": _title_content("Introducing", project, tagline[:110]),
        "outro": _title_content("Try it now", project, host, f"github.com/{repo}" if repo else ""),
    }
    key = get_secret("FAL_KEY")

    async def clip(kind: str) -> Optional[Path]:
        if not key:
            return None
        prompt = (visuals.get(kind) or TEMPLATE_VISUALS[kind]).strip()
        return await asyncio.wait_for(_generate_clip(prompt, job_dir / f"{kind}_ai.mp4", key),
                                      timeout=FAL_VIDEO_TIMEOUT_SEC + 30)

    tasks = {kind: asyncio.create_task(clip(kind)) for kind in ("intro", "outro")}
    try:
        narrations = await asyncio.gather(*(synthesize_scene_audio(lines[k], job_dir / f"{k}.wav", voice_style)
                                            for k in ("intro", "outro")))
        card_paths = await _render_cards(cards, job_dir)
    except BaseException:
        for task in tasks.values():
            task.cancel()
        raise

    bookends: Dict[str, Bookend] = {}
    for kind, (duration, _engine) in zip(("intro", "outro"), narrations):
        video, note = None, ""
        try:
            video = await tasks[kind]
            note = f"AI clip from {FAL_VIDEO_MODEL}" if video else "typeset title card (FAL_KEY not set)"
        except Exception as exc:
            logger.warning("AI %s clip failed: %s", kind, exc)
            reason = str(exc).strip().splitlines()[0][:140] if str(exc).strip() else type(exc).__name__
            note = f"typeset title card (AI clip unavailable: {reason})"
        bookends[kind] = Bookend(kind=kind, narration=job_dir / f"{kind}.wav", narration_sec=duration,
                                 overlay=card_paths[kind]["overlay"], background=card_paths[kind]["background"],
                                 video=video, note=note)
        log(f"{kind.capitalize()}: {note}")
    return bookends
