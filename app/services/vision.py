"""Gemini vision brief: one multimodal look at the pages the browser explored, so the planner knows what each page
really shows (and which ones are a sign-in wall, an error page or empty) before it writes the storyboard.

`describe_pages` makes exactly ONE Gemini call with up to VISION_MAX_PAGES shrunk screenshots (the landing page first,
then the explored pages) and writes the answers back onto `app_info`: `app_info['landing_insight']` and
`app_info['pages'][i]['insight']`, each `{purpose, demo_moment, quality, source}`, which planner.py turns into the
"Gemini saw" site-map lines and the AVOID flags. It returns the described pages for the Understand step's one-off event
(`thumbnails` adds the base64 `thumb` the agent trace renders). It raises on timeout or failure so the orchestrator's
`_optional` wrapper logs one line and plans from the headings alone; nothing here can take the demo down.
"""
import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from app.services import gemini, images
from app.services.planner import _clean_narration

logger = logging.getLogger("demoday.vision")

VISION_MAX_PAGES = 7      # inline JPEGs in the one call (landing page + explored pages)
VISION_MAX_WIDTH = 1280   # px; a 1920x1080 viewport shot shrinks to ~100-150 KB
VISION_TIMEOUT_SEC = 20   # the orchestrator wraps the whole step in the same cap
PURPOSE_MAX_CHARS = 120
MOMENT_MAX_CHARS = 140
QUALITIES = ("good", "busy", "empty", "error", "login")
FLAGGED = ("empty", "error", "login")  # not worth a scene; planner.FLAGGED_PAGES words them for the storyboard prompt
# A Gemini refusal of the inline images (not a timeout, not an outage) earns one text-only retry over the headings.
_IMAGE_REFUSAL = re.compile(r"image|mime|unsupported|modality|inline[_ ]data", re.IGNORECASE)

PROMPT_RULES = (
    'For EACH page return {"pages": [{"index": 1, "purpose": "<= 12 words: what this page is for", '
    '"demo_moment": "<= 15 words: the single most demo-worthy element visible and where it sits (top/middle/bottom)", '
    '"quality": "good|busy|empty|error|login"}]}. Describe only what is visible in the screenshot; never guess features '
    'from the URL and never invent anything. A small Sample data pill in a corner and a green cursor are expected, not '
    'issues. Use empty, error or login when the page shows almost no content, an error page, or a sign-in form.'
)
HEADINGS_RULES = (
    'For EACH page return {"pages": [{"index": 1, "purpose": "<= 12 words: what this page is for", '
    '"demo_moment": "<= 15 words: the single most demo-worthy element the headings name and where it likely sits '
    '(top/middle/bottom)", "quality": "good|busy|empty|error|login"}]}. Describe only what the title and headings say; '
    'never guess features from the URL and never invent anything. Use empty, error or login when the headings show '
    'almost no content, an error page, or a sign-in form.'
)


@dataclass
class _Entry:
    path: str
    title: str
    headings: List[str]
    screenshot: str
    target: Dict[str, Any]   # where the insight is written: app_info (landing page) or app_info['pages'][i]
    landing: bool


async def describe_pages(app_info: Dict[str, Any], log: Callable[[str], None]) -> List[Dict[str, Any]]:
    """Describes the screenshotted pages in one Gemini call and writes the insights onto `app_info`.

    Returns one row per described page, in page order: `{path, title, purpose, demo_moment, quality, source,
    screenshot}` (`thumbnails` swaps `screenshot` for `thumb`). Returns [] without a call when nothing was
    screenshotted; raises when Gemini fails, times out or describes none of the pages.
    """
    entries = _entries(app_info)
    if not entries:
        return []
    name, url = _app_name(app_info), app_info.get("url", "")
    source = "screenshot"
    try:
        shots = await _shrunk(entries)  # drops the entries whose screenshot could not be re-encoded
        if not shots:
            raise RuntimeError("no screenshot could be prepared for Gemini")
        entries = [entry for entry, _ in shots]
        data = await gemini.generate_json(_prompt(name, url, entries), images=[jpeg for _, jpeg in shots],
                                          timeout_sec=VISION_TIMEOUT_SEC, purpose="understand pages")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if not _IMAGE_REFUSAL.search(str(exc)):
            raise
        log(f"Gemini declined the screenshots ({_short(exc)}); describing the pages from their headings instead")
        source = "headings"
        data = await gemini.generate_json(_prompt(name, url, entries, headings=True),
                                          timeout_sec=VISION_TIMEOUT_SEC, purpose="understand pages (headings)")
    rows: List[Dict[str, Any]] = []
    for entry, insight in zip(entries, _sanitize(data, len(entries))):
        if insight is None:
            continue
        insight["source"] = source
        entry.target["landing_insight" if entry.landing else "insight"] = insight
        rows.append({"path": entry.path, "title": entry.title, **insight, "screenshot": entry.screenshot})
    if not rows:
        raise ValueError("Gemini described none of the pages")
    return rows


async def thumbnails(rows: List[Dict[str, Any]]) -> None:
    """Replaces each row's local `screenshot` path with a 320 px base64 JPEG `thumb` for the one-off step event
    (never for JobStatus). A row whose thumbnail fails just loses it."""
    paths = [row.pop("screenshot", None) for row in rows]
    results = await asyncio.gather(*(images.thumbnail_b64(path) if path else _none() for path in paths),
                                   return_exceptions=True)
    for row, path, result in zip(rows, paths, results):
        if isinstance(result, BaseException):
            if isinstance(result, asyncio.CancelledError):
                raise result
            logger.warning("thumbnail for %s failed: %s", path, _short(result))
        elif result:
            row["thumb"] = result


def summary_line(rows: List[Dict[str, Any]]) -> str:
    """'Gemini saw: / = framework landing page (best moment: the feature list, top) · /docs = ...' for the job log."""
    parts = []
    for row in rows:
        if not row.get("purpose"):
            continue
        part = f"{row['path']} = {row['purpose']}"
        if row.get("demo_moment"):
            part += f" (best moment: {row['demo_moment']})"
        if row.get("quality") in FLAGGED:
            part += f" [flagged: {row['quality']}]"
        parts.append(part)
    return "Gemini saw: " + " · ".join(parts) if parts else "Gemini saw nothing it could describe"


# --------------------------------------------------------------------------- internals


def _entries(app_info: Dict[str, Any]) -> List[_Entry]:
    """The landing page then the explored pages, those with a screenshot, at most VISION_MAX_PAGES."""
    entries: List[_Entry] = []
    if app_info.get("screenshot"):
        entries.append(_Entry(_path_of(app_info.get("url", "")), str(app_info.get("title") or ""),
                              list(app_info.get("headings") or []), str(app_info["screenshot"]), app_info, True))
    for page in app_info.get("pages") or []:
        if page.get("screenshot"):
            entries.append(_Entry(str(page.get("path") or "/"), str(page.get("title") or ""),
                                  list(page.get("headings") or []), str(page["screenshot"]), page, False))
    return entries[:VISION_MAX_PAGES]


async def _shrunk(entries: List[_Entry]) -> List[tuple]:
    """(entry, shrunk JPEG bytes) for every entry whose screenshot re-encodes; the others are logged and dropped."""
    results = await asyncio.gather(*(images.shrink_jpeg(entry.screenshot, VISION_MAX_WIDTH) for entry in entries),
                                   return_exceptions=True)
    shots = []
    for entry, result in zip(entries, results):
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, BaseException):
            logger.warning("could not shrink %s: %s", entry.screenshot, _short(result))
            continue
        shots.append((entry, result))
    return shots


def _prompt(name: str, url: str, entries: List[_Entry], headings: bool = False) -> str:
    count = len(entries)
    if headings:
        listing = "  ".join(f'{i}) {e.path} "{_quote(e.title)}" headings: {" | ".join(e.headings[:6]) or "(none)"}'
                            for i, e in enumerate(entries, 1))
        return (f'You are preparing a demo-video brief for "{name}" ({url}). Screenshots are not available, so here '
                f'are the title and headings of {count} page{"s" if count != 1 else ""} of the live app, in this '
                f'order: {listing}. {HEADINGS_RULES}')
    listing = "  ".join(f'{i}) {e.path} "{_quote(e.title)}"' for i, e in enumerate(entries, 1))
    return (f'You are preparing a demo-video brief for "{name}" ({url}). Attached are screenshots of {count} '
            f'page{"s" if count != 1 else ""} of the live app, in this order: {listing}. {PROMPT_RULES}')


def _sanitize(data: Any, count: int) -> List[Optional[Dict[str, str]]]:
    """One insight per entry (None when Gemini skipped it), matched by the 1-based `index` (0-based tolerated) with a
    positional fallback; strings are cleaned and capped, unknown qualities become good."""
    raw_pages: Any = data
    if isinstance(data, dict):
        raw_pages = data.get("pages")
        if not isinstance(raw_pages, list):  # {"result": {"pages": [...]}}, {"page_insights": [...]}, or one page
            raw_pages = next((v for v in data.values() if isinstance(v, list)), None)
            if raw_pages is None:
                inner = next((v for v in data.values() if isinstance(v, dict) and isinstance(v.get("pages"), list)), None)
                raw_pages = inner["pages"] if inner else ([data] if "purpose" in data else [])
    if not isinstance(raw_pages, list):
        raw_pages = []
    items = [item for item in raw_pages if isinstance(item, dict)]
    indexes = [_as_int(item.get("index")) for item in items]
    base = 0 if any(i == 0 for i in indexes) else 1  # the prompt asks for 1-based; tolerate 0-based answers
    out: List[Optional[Dict[str, str]]] = [None] * count
    for position, (item, index) in enumerate(zip(items, indexes)):
        slot = index - base if index is not None and 0 <= index - base < count else position
        if slot >= count or out[slot] is not None:
            continue
        purpose = _clean_narration(item.get("purpose"))[:PURPOSE_MAX_CHARS]
        if not purpose:
            continue
        quality = str(item.get("quality") or "").strip().lower()
        out[slot] = {"purpose": purpose, "demo_moment": _clean_narration(item.get("demo_moment"))[:MOMENT_MAX_CHARS],
                     "quality": quality if quality in QUALITIES else "good"}
    return out


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None and not isinstance(value, bool) else None
    except (TypeError, ValueError):
        return None


def _app_name(app_info: Dict[str, Any]) -> str:
    title = str(app_info.get("title") or "").strip()
    lead = re.split(r"\s+[-|–—:]\s+", title, maxsplit=1)[0].strip()
    return (lead[:60] or urlparse(app_info.get("url", "")).netloc or "the app")


def _path_of(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return path + ("?" + parsed.query if parsed.query else "")


def _quote(text: str) -> str:
    return re.sub(r"\s+", " ", text).replace('"', "'").strip()[:80]


def _short(exc: BaseException) -> str:
    text = str(exc).strip()
    return (text.splitlines()[0] if text else type(exc).__name__)[:160]


async def _none() -> None:
    return None
