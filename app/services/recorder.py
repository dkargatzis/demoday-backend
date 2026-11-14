"""Headless Chromium recording of a storyboard, timed to each scene's narration.

All scenes are recorded in ONE continuous browser session (one context, one page), so state carries
over between scenes like a human-driven demo: text typed in scene 2 is still there in scene 3. A
scene's footage window opens only after its page has loaded and settled, so blank frames and loading
spinners between scenes never reach the final video; the muxer cuts each window out of the session.
"""
import asyncio
import logging
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence
from urllib.parse import urljoin, urlparse

from playwright.async_api import Browser, Locator, Page, Playwright, async_playwright
from playwright.async_api import Error as PlaywrightError

from app.models.schemas import Action, Scene
from app.services.netguard import guard_context

logger = logging.getLogger("demoday.recorder")

VIEWPORT = {"width": 1920, "height": 1080}
NAV_TIMEOUT_MS = 20_000
SELECTOR_TIMEOUT_MS = 1_500  # pages are already settled; a missing element shouldn't eat on-camera time
ACTION_TIMEOUT_MS = 5_000
MAX_TYPED_CHARS = 160

LogFn = Callable[[str], None]

# Injected into every page: a visible cursor + click ripple (headless recordings have no pointer),
# hidden scrollbars, and target=_blank links forced into the recorded tab.
INIT_SCRIPT = r"""
(() => {
  if (window.top !== window) return;
  const install = () => {
    if (document.getElementById("__demoday_cursor")) return;
    const style = document.createElement("style");
    style.textContent = `
      #__demoday_cursor { position: fixed; left: 0; top: 0; width: 22px; height: 22px; margin: -11px 0 0 -11px;
        border-radius: 50%; background: rgba(16,185,129,.35); border: 2px solid rgba(16,185,129,.95);
        box-shadow: 0 0 0 4px rgba(16,185,129,.18); pointer-events: none; z-index: 2147483647;
        opacity: 0; transition: transform .1s ease-out, opacity .2s; }
      #__demoday_cursor.down { transform: scale(.7); }
      .__demoday_ripple { position: fixed; width: 22px; height: 22px; margin: -11px 0 0 -11px; border-radius: 50%;
        border: 3px solid rgba(16,185,129,.9); pointer-events: none; z-index: 2147483646;
        animation: __demoday_ripple .6s ease-out forwards; }
      @keyframes __demoday_ripple { to { transform: scale(3.2); opacity: 0; } }
      html { scrollbar-width: none; }
      ::-webkit-scrollbar { display: none; }
    `;
    document.documentElement.appendChild(style);
    const dot = document.createElement("div");
    dot.id = "__demoday_cursor";
    document.documentElement.appendChild(dot);
    document.addEventListener("mousemove", (e) => {
      dot.style.left = e.clientX + "px";
      dot.style.top = e.clientY + "px";
      dot.style.opacity = "1";
    }, true);
    document.addEventListener("mousedown", (e) => {
      dot.classList.add("down");
      const ripple = document.createElement("div");
      ripple.className = "__demoday_ripple";
      ripple.style.left = e.clientX + "px";
      ripple.style.top = e.clientY + "px";
      document.documentElement.appendChild(ripple);
      setTimeout(() => ripple.remove(), 700);
    }, true);
    document.addEventListener("mouseup", () => dot.classList.remove("down"), true);
    document.addEventListener("click", (e) => {
      const link = e.target && e.target.closest ? e.target.closest("a[target]") : null;
      if (link) link.removeAttribute("target");
    }, true);
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", install);
  else install();
})();
"""

# Smooth-scrolls the document, or the largest scrollable container when the document itself can't scroll (SPAs).
SCROLL_SCRIPT = r"""
([mode, amount]) => {
  const root = document.scrollingElement || document.documentElement;
  let el = root;
  if (root.scrollHeight - root.clientHeight < 40) {
    let best = null;
    for (const node of document.querySelectorAll("body *")) {
      if (node.clientHeight < 200 || node.scrollHeight - node.clientHeight < 40) continue;
      const overflowY = getComputedStyle(node).overflowY;
      if (overflowY !== "auto" && overflowY !== "scroll") continue;
      if (!best || node.clientHeight > best.clientHeight) best = node;
    }
    if (best) el = best;
  }
  const view = el === root ? window.innerHeight : el.clientHeight;
  const opts = { behavior: "smooth" };
  if (mode === "top") el.scrollTo({ top: 0, ...opts });
  else if (mode === "bottom") el.scrollTo({ top: el.scrollHeight, ...opts });
  else el.scrollBy({ top: (mode === "up" ? -1 : 1) * (amount || Math.round(view * 0.7)), ...opts });
}
"""

# Brings an element into view only when needed; returns true if it had to scroll.
REVEAL_SCRIPT = r"""
(el) => {
  const r = el.getBoundingClientRect();
  if (r.top >= 80 && r.bottom <= window.innerHeight - 80) return false;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  return true;
}
"""

# True when a real click at the element's center would land on it: not transparent, not hidden,
# not covered by another element, not clipped away inside a collapsed menu.
HIT_TEST_SCRIPT = r"""
(el) => {
  if (el.checkVisibility && !el.checkVisibility({ opacityProperty: true, visibilityProperty: true })) return false;
  const r = el.getBoundingClientRect();
  const x = r.left + r.width / 2, y = r.top + r.height / 2;
  if (x < 0 || y < 0 || x >= window.innerWidth || y >= window.innerHeight) return false;
  const hit = document.elementFromPoint(x, y);
  return !!hit && (hit === el || el.contains(hit));
}
"""


# The browser may be signed in to the user's real account, so sign-out and destructive controls are never clicked,
# whatever the storyboard says. (Logging out usually also invalidates the saved session on the server.)
RISKY_LABEL = re.compile(
    r"\b(log\s?out|sign\s?out|logout|signout|delete|remove|destroy|deactivate|close account|disable account|"
    r"cancel (?:subscription|plan|account|membership)|unsubscribe|revoke|reset|archive|transfer|purchase|buy|pay|"
    r"checkout|upgrade|downgrade|leave (?:team|workspace|organi[sz]ation))\b",
    re.IGNORECASE,
)
RISKY_URL = re.compile(r"(log[-_]?out|sign[-_]?out|/delete|/destroy|/unsubscribe|/checkout|/billing)", re.IGNORECASE)
ELEMENT_LABEL_SCRIPT = (
    "el => [el.innerText, el.getAttribute('aria-label'), el.getAttribute('title'), el.value, el.getAttribute('href')]"
    ".filter(Boolean).join(' ').slice(0, 300)"
)


def is_risky(label: str = "", url: str = "") -> bool:
    return bool((label and RISKY_LABEL.search(label)) or (url and RISKY_URL.search(url)))


@dataclass
class SceneSegment:
    start: float     # seconds into the session recording where the scene's footage begins
    duration: float  # seconds of footage for the scene (narration + tail)


def resolve_url(app_url: str, route: Optional[str]) -> Optional[str]:
    """Resolves a storyboard route against the app URL. Returns None for anything that isn't http(s)."""
    route = (route or "").strip()
    if route in ("", "/"):
        return app_url
    base = app_url if app_url.endswith("/") else app_url + "/"
    resolved = urljoin(base, route)
    parsed = urlparse(resolved)
    return resolved if parsed.scheme in ("http", "https") and parsed.netloc else None


def _same_page(current: str, target: str) -> bool:
    a, b = urlparse(current), urlparse(target)
    return (a.scheme, a.netloc, a.path.rstrip("/"), a.query) == (b.scheme, b.netloc, b.path.rstrip("/"), b.query)


def _link_href(selector: Optional[str]) -> Optional[str]:
    """The href of an `a[href="..."]` selector (the format the app inventory produces), else None."""
    match = re.fullmatch(r'\s*a\[href=(["\'])(.+?)\1\]\s*', selector or "")
    return match.group(2).replace('\\"', '"') if match else None


def _short(exc: BaseException) -> str:
    text = str(exc).strip()
    return (text.splitlines()[0] if text else type(exc).__name__)[:160]


async def launch_chromium(pw: Playwright) -> Browser:
    try:
        return await pw.chromium.launch(headless=True)
    except PlaywrightError as exc:
        raise RuntimeError(
            "Could not launch headless Chromium. Run `python -m playwright install chromium` inside the "
            f"backend virtualenv. ({_short(exc)})"
        ) from exc


async def record_session(
    scenes: Sequence[Scene],
    app_url: str,
    durations: Sequence[float],
    output_path: Path,
    log: LogFn,
    on_scene: Optional[Callable[[int], None]] = None,
    storage_state: Optional[Dict] = None,
) -> List[SceneSegment]:
    """Records every scene in one 1920x1080 session and returns each scene's footage window.

    `storage_state` (a saved sign-in session) makes the browser start signed in to the app.
    """
    output_path = Path(output_path)
    video_dir = Path(tempfile.mkdtemp(prefix="pw-video-", dir=output_path.parent))
    segments: List[SceneSegment] = []
    raw_path = None
    try:
        async with async_playwright() as pw:
            browser = await launch_chromium(pw)
            try:
                context = await browser.new_context(
                    viewport=VIEWPORT,
                    screen=VIEWPORT,
                    device_scale_factor=1,
                    locale="en-US",
                    record_video_dir=str(video_dir),
                    record_video_size=VIEWPORT,
                    storage_state=storage_state,
                    service_workers="block",  # keeps every request visible to the network guard
                )
                await guard_context(context)
                await context.add_init_script(INIT_SCRIPT)
                page = await context.new_page()
                session_start = time.monotonic()  # Playwright starts recording when the page is created
                video = page.video
                raw_path = await video.path() if video else None
                director = _Director(page, app_url, log)

                for idx, (scene, duration) in enumerate(zip(scenes, durations)):
                    if on_scene:
                        on_scene(idx)
                    await director.open_scene(idx + 1, scene.route, first=(idx == 0))
                    start = time.monotonic()
                    await director.run(idx + 1, scene.actions, deadline=start + duration)
                    remaining = start + duration - time.monotonic()
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                    segments.append(SceneSegment(start=start - session_start, duration=duration))

                await context.close()  # flushes the .webm to disk
            finally:
                await browser.close()

        if not raw_path or not Path(raw_path).exists():
            raise RuntimeError("Playwright did not produce a session recording")
        shutil.move(str(raw_path), output_path)
    finally:
        shutil.rmtree(video_dir, ignore_errors=True)
    return segments


class _Director:
    """Performs storyboard actions on one page with on-camera pacing; never raises for a bad action."""

    def __init__(self, page: Page, app_url: str, log: LogFn):
        self.page = page
        self.app_url = app_url
        self.log = log
        self.pointer_shown = False
        self.landed: Dict[str, str] = {}  # requested URL -> URL after redirects

    async def open_scene(self, scene_no: int, route: str, first: bool) -> None:
        target = resolve_url(self.app_url, route)
        if target is None:
            self.log(f"Scene {scene_no}: ignoring unsupported route {route!r}")
            target = self.app_url
        if first or not _same_page(self.page.url, self.landed.get(target, target)):
            await self._goto(target, scene_no)
            self.landed[target] = self.page.url
        await self._settle()

    async def run(self, scene_no: int, actions: Sequence[Action], deadline: float) -> None:
        for position, action in enumerate(actions):
            if time.monotonic() >= deadline - 0.25:
                skipped = len(actions) - position
                self.log(f"Scene {scene_no}: narration finished, {skipped} remaining action(s) skipped")
                return
            try:
                done = await self._perform(scene_no, action, deadline)
            except PlaywrightError as exc:
                self.log(f"Scene {scene_no}: {action.type} skipped ({_short(exc)})")
                done = False
            if done:
                dwell = min(action.dwell_ms / 1000.0, deadline - time.monotonic())
                if dwell > 0:
                    await asyncio.sleep(dwell)

    async def _perform(self, scene_no: int, action: Action, deadline: float) -> bool:
        if action.type == "wait":
            return True
        if action.type == "navigate":
            url = resolve_url(self.app_url, action.value)
            if url is None:
                self.log(f"Scene {scene_no}: navigate skipped (unsupported URL {action.value!r})")
                return False
            if is_risky(url=url):
                self.log(f"Scene {scene_no}: skipped opening {url} (sign-out and destructive pages are never opened)")
                return False
            await self._goto(url, scene_no)
            await self._quiet_network(min(2500, self._budget_ms(deadline)))
            return True
        if action.type == "scroll":
            return await self._scroll(scene_no, action)

        target = await self._find(action.selector)
        if target is None:
            # A link that isn't on this page still tells us where the scene wants to go.
            href = _link_href(action.selector) if action.type == "click" else None
            url = resolve_url(self.app_url, href) if href else None
            if url and is_risky(url=url):
                self.log(f"Scene {scene_no}: skipped {href!r} (sign-out and destructive actions are never clicked)")
                return False
            if url:
                self.log(f"Scene {scene_no}: link {href!r} not on screen, navigating to it directly")
                await self._goto(url, scene_no)
                await self._quiet_network(min(2500, self._budget_ms(deadline)))
                return True
            self.log(f"Scene {scene_no}: selector {action.selector!r} not found, {action.type} skipped")
            return False
        if action.type == "click":
            label = await target.evaluate(ELEMENT_LABEL_SCRIPT)
            if is_risky(label=label, url=label):
                self.log(f"Scene {scene_no}: skipped clicking {label.strip()[:60]!r} "
                         "(sign-out and destructive actions are never clicked)")
                return False
        await self._point_at(target)
        if not await target.evaluate(HIT_TEST_SCRIPT):
            self.log(f"Scene {scene_no}: {action.selector!r} is hidden or covered, {action.type} skipped")
            return False
        if action.type == "click":
            await target.click(timeout=min(ACTION_TIMEOUT_MS, self._budget_ms(deadline)))
            await self._after_interaction(deadline)
            return True
        # type
        text = action.value or ""
        submit = text.endswith("\n")
        text = text.rstrip("\n")[:MAX_TYPED_CHARS]
        await target.click(timeout=ACTION_TIMEOUT_MS)
        await target.fill("", timeout=ACTION_TIMEOUT_MS)
        delay = max(20, min(80, 1400 // max(1, len(text))))
        await target.press_sequentially(text, delay=delay)
        if submit:
            await target.press("Enter")
            await self._after_interaction(deadline)
        return True

    async def _goto(self, url: str, scene_no: int) -> None:
        try:
            response = await self.page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
            if response is not None and response.status >= 400:
                self.log(f"Scene {scene_no}: {url} answered HTTP {response.status}")
        except PlaywrightError as exc:
            reason = "private network addresses are blocked" if "ERR_BLOCKED_BY_CLIENT" in str(exc) else _short(exc)
            self.log(f"Scene {scene_no}: navigation to {url} failed ({reason})")

    async def _settle(self) -> None:
        """Best-effort wait until the page looks finished: loaded, network quiet, web fonts ready."""
        try:
            await self.page.wait_for_load_state("load", timeout=8000)
        except PlaywrightError:
            pass
        await self._quiet_network(2500)
        try:
            await asyncio.wait_for(
                self.page.evaluate("document.fonts ? document.fonts.ready.then(() => true) : true"), 2
            )
        except (PlaywrightError, asyncio.TimeoutError):
            pass
        await asyncio.sleep(0.3)

    async def _quiet_network(self, timeout_ms: int) -> None:
        try:
            await self.page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except PlaywrightError:
            pass

    async def _after_interaction(self, deadline: float) -> None:
        # If the interaction started a navigation, let the next page paint before continuing.
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=min(4000, self._budget_ms(deadline)))
        except PlaywrightError:
            pass

    async def _find(self, selector: Optional[str]) -> Optional[Locator]:
        if not selector:
            return None
        locator = self.page.locator(selector).filter(visible=True).first
        try:
            await locator.wait_for(state="visible", timeout=SELECTOR_TIMEOUT_MS)
        except PlaywrightError:  # missing element or invalid selector syntax
            return None
        return locator

    async def _point_at(self, target: Locator) -> None:
        """Glides the visible cursor onto the element so viewers can follow the interaction."""
        try:
            if await target.evaluate(REVEAL_SCRIPT):
                await self._wait_until_still(target)
            box = await target.bounding_box()
        except PlaywrightError:
            return
        if not box:
            return
        if not self.pointer_shown:
            await self.page.mouse.move(VIEWPORT["width"] * 0.5, VIEWPORT["height"] * 0.55)
            self.pointer_shown = True
        await self.page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=20)
        await asyncio.sleep(0.25)

    @staticmethod
    async def _wait_until_still(target: Locator, max_wait: float = 1.5) -> None:
        """Waits for a smooth scroll to finish: the element's position is unchanged across two polls."""
        last = None
        for _ in range(int(max_wait / 0.1)):
            await asyncio.sleep(0.1)
            box = await target.bounding_box()
            position = (round(box["x"]), round(box["y"])) if box else None
            if position == last:
                return
            last = position

    async def _scroll(self, scene_no: int, action: Action) -> bool:
        value = (action.value or "down").strip().lower()
        selector = action.selector
        if not selector and value not in ("down", "up", "top", "bottom") and not value.lstrip("-").isdigit():
            selector = action.value  # e.g. {"type": "scroll", "value": "#features"}
        if selector:
            target = await self._find(selector)
            if target is None:
                self.log(f"Scene {scene_no}: scroll target {selector!r} not found, skipped")
                return False
            await target.evaluate("el => el.scrollIntoView({ behavior: 'smooth', block: 'center' })")
            return True
        amount = None
        if value.lstrip("-").isdigit():
            amount = abs(int(value))
            value = "up" if value.startswith("-") else "down"
        await self.page.evaluate(SCROLL_SCRIPT, [value, amount])
        return True

    @staticmethod
    def _budget_ms(deadline: float) -> int:
        return max(500, int((deadline - time.monotonic()) * 1000))
