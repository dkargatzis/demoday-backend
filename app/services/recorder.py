"""Headless Chromium recording of a storyboard, timed to each scene's narration.

All scenes are recorded in ONE continuous browser session (one context, one page), so state carries
over between scenes like a human-driven demo: text typed in scene 2 is still there in scene 3. A
scene's footage window opens only after its page has loaded and settled, so blank frames and loading
spinners between scenes never reach the final video; the muxer cuts each window out of the session.

Frames are kept clean (consent banners accepted or hidden, chat widgets hidden, skeleton loaders waited
out), clicks get a highlight ring, and every click or typed field is logged as a focus event with its
timestamp so the muxer can zoom toward it.
"""
import asyncio
import logging
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
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
LOADER_WAIT_SEC = 4.0
CLICK_ZOOM, TYPE_ZOOM = 1.45, 1.55
CLICK_HOLD_SEC = 1.6

LogFn = Callable[[str], None]
FocusTuple = Tuple[float, float, float, float, float]  # (t, cx, cy, zoom, hold)

# Injected into every page: a visible cursor + click ripple (headless recordings have no pointer), a highlight
# ring the recorder shows before clicks, an optional "Sample data" label, hidden scrollbars, and target=_blank
# links forced into the recorded tab. Everything it adds carries data-demoday so the cleanup skips it.
INIT_SCRIPT = r"""
(() => {
  if (window.top !== window) return;
  const install = () => {
    if (document.getElementById("__demoday_cursor")) return;
    const style = document.createElement("style");
    style.setAttribute("data-demoday", "");
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
      #__demoday_ring { position: fixed; pointer-events: none; z-index: 2147483645; border-radius: 10px;
        border: 3px solid rgba(16,185,129,.95); box-shadow: 0 0 0 6px rgba(16,185,129,.22), 0 0 28px rgba(16,185,129,.45);
        opacity: 0; transition: opacity .18s ease-out, left .22s ease-out, top .22s ease-out, width .22s ease-out, height .22s ease-out; }
      #__demoday_ring.on { opacity: 1; animation: __demoday_ring_pulse 1.1s ease-in-out infinite; }
      @keyframes __demoday_ring_pulse { 50% { box-shadow: 0 0 0 10px rgba(16,185,129,.12), 0 0 34px rgba(16,185,129,.55); } }
      #__demoday_pill { position: fixed; left: 18px; bottom: 18px; z-index: 2147483644; pointer-events: none;
        padding: 6px 12px; border-radius: 999px; background: rgba(9,9,11,.78); color: #d4d4d8;
        font: 600 14px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif; letter-spacing: .3px;
        border: 1px solid rgba(255,255,255,.14); backdrop-filter: blur(6px); }
      #__demoday_pill b { color: #6ee7b7; font-weight: 700; margin-right: 6px; }
      html { scrollbar-width: none; }
      ::-webkit-scrollbar { display: none; }
    `;
    document.documentElement.appendChild(style);
    const make = (id) => { const el = document.createElement("div"); el.id = id; el.setAttribute("data-demoday", "");
      document.documentElement.appendChild(el); return el; };
    const dot = make("__demoday_cursor");
    const ring = make("__demoday_ring");
    if (window.__DEMODAY_STAGED__) { const pill = make("__demoday_pill"); pill.innerHTML = "<b>●</b>Sample data"; }
    document.addEventListener("mousemove", (e) => {
      dot.style.left = e.clientX + "px";
      dot.style.top = e.clientY + "px";
      dot.style.opacity = "1";
    }, true);
    document.addEventListener("mousedown", (e) => {
      dot.classList.add("down");
      const ripple = document.createElement("div");
      ripple.className = "__demoday_ripple";
      ripple.setAttribute("data-demoday", "");
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
    window.__demoday = {
      highlight(el) {
        const r = el.getBoundingClientRect();
        const pad = 6;
        ring.style.left = (r.left - pad) + "px"; ring.style.top = (r.top - pad) + "px";
        ring.style.width = (r.width + 2 * pad) + "px"; ring.style.height = (r.height + 2 * pad) + "px";
        ring.classList.add("on");
      },
      unhighlight() { ring.classList.remove("on"); },
    };
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", install);
  else install();
})();
"""

# Accepts or hides consent banners, hides chat/assistant widgets and promo pop-ups. Returns what it did.
DECLUTTER_SCRIPT = r"""
() => {
  const result = { accepted: 0, hidden: 0 };
  const CONSENT = /cookie|consent|gdpr|privacy preferences|we use cookies|tracking/i;
  const ACCEPT = /^(accept( all)?( cookies)?|allow( all)?( cookies)?|agree|i agree|i accept|ok(ay)?|got it|i understand|understood|continue|dismiss|close)$/i;
  const PROMO = /newsletter|subscribe|special offer|discount|limited time|sign up for updates|join our (list|community)/i;
  const CLOSE = /^(close|dismiss|no thanks|not now|maybe later|×|✕|x)$/i;
  const WIDGET = /intercom|crisp|drift|hubspot|tawk|zendesk|zopim|freshchat|freshworks|tidio|olark|livechat|gorgias|chatwoot|kommunicate|userlike|helpscout|beacon|kapa|ask-?ai|askai|chatbot|chat-?widget|widget-?launcher|calendly|feedback-?button|hotjar|usersnap|\bfab\b/i;
  const label = (el) => ((el.innerText || el.textContent || "") + " " + (el.getAttribute("aria-label") || "")).replace(/\s+/g, " ").trim();
  const hide = (el) => { if (!el.hasAttribute("data-demoday-hidden")) { el.setAttribute("data-demoday-hidden", ""); el.style.setProperty("display", "none", "important"); result.hidden++; } };
  const all = [];
  const walk = (root) => { for (const el of root.querySelectorAll("*")) { all.push(el); if (el.shadowRoot) walk(el.shadowRoot); } };
  walk(document);
  const vw = window.innerWidth, vh = window.innerHeight;
  for (const el of all) {
    if (el.hasAttribute("data-demoday") || el.closest("[data-demoday]")) continue;
    const s = getComputedStyle(el);
    if (s.position !== "fixed" && s.position !== "sticky") continue;
    if (s.display === "none" || s.visibility === "hidden") continue;
    const r = el.getBoundingClientRect();
    if (r.width < 24 || r.height < 24) continue;
    const text = label(el);
    const ident = (el.id + " " + String(el.className || "") + " " + (el.getAttribute("src") || "") + " " + (el.getAttribute("aria-label") || "") + " " + (el.getAttribute("title") || "") + " " + (el.tagName === "IFRAME" ? (el.getAttribute("name") || "") : ""));
    const buttons = () => Array.from(el.querySelectorAll("button, a, [role=button], input[type=button], input[type=submit]"));
    // 1. consent banners: accept when there is a plain accept button, otherwise hide
    if (text.length < 4000 && CONSENT.test(text) && r.height >= 40 && (r.width >= 240)) {
      const accept = buttons().find((b) => ACCEPT.test(label(b)));
      if (accept) { accept.click(); result.accepted++; } else hide(el);
      continue;
    }
    // 2. chat, assistant, scheduling and feedback widgets, by vendor or role
    if (WIDGET.test(ident) || (el.tagName === "IFRAME" && WIDGET.test(el.getAttribute("src") || ""))) { hide(el); continue; }
    // 3. promo pop-ups: close or hide
    if (s.position === "fixed" && PROMO.test(text) && r.width * r.height > vw * vh * 0.12 && text.length < 3000) {
      const close = buttons().find((b) => CLOSE.test(label(b)));
      if (close) close.click(); else hide(el);
      result.hidden++;
      continue;
    }
    // 4. small floating buttons anchored to a bottom corner (chat bubbles, "ask AI", back-to-top)
    const atBottom = vh - r.bottom <= 48, atSide = r.left <= 48 || vw - r.right <= 48;
    if (s.position === "fixed" && atBottom && atSide && r.width <= 420 && r.height <= 420 && !/nav|menu|footer|toolbar|tab/i.test(ident) && !el.querySelector("input, textarea, select") && !/^(nav|footer|header)$/i.test(el.tagName)) {
      hide(el);
    }
  }
  return result;
}
"""

# Number of visible loading indicators still on the page (skeletons, spinners, progress bars).
LOADERS_SCRIPT = r"""
() => {
  const sel = '[class*="skeleton" i], [class*="spinner" i], [class*="shimmer" i], [class*="loading" i], [class*="placeholder-glow" i], .animate-pulse, [role="progressbar"], [aria-busy="true"]';
  let n = 0;
  for (const el of document.querySelectorAll(sel)) {
    if (el.closest("[data-demoday]")) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4 || r.bottom < 0 || r.top > window.innerHeight) continue;
    const s = getComputedStyle(el);
    if (s.display === "none" || s.visibility === "hidden" || +s.opacity < 0.05) continue;
    n++;
  }
  return n;
}
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

HIGHLIGHT_SCRIPT = "el => window.__demoday && window.__demoday.highlight(el)"
UNHIGHLIGHT_SCRIPT = "() => window.__demoday && window.__demoday.unhighlight()"


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
    focus: List[FocusTuple] = field(default_factory=list)  # zoom targets, times relative to the scene start


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


async def wait_for_loaders(page: Page, timeout_sec: float = LOADER_WAIT_SEC) -> int:
    """Waits until no skeleton/spinner is visible (or the timeout passes); returns how many are left."""
    deadline = time.monotonic() + timeout_sec
    remaining = 0
    while True:
        try:
            remaining = int(await page.evaluate(LOADERS_SCRIPT))
        except PlaywrightError:
            return 0
        if remaining == 0 or time.monotonic() >= deadline:
            return remaining
        await asyncio.sleep(0.25)


async def settle_page(page: Page) -> None:
    """Best-effort wait until the page looks finished: loaded, network quiet, fonts ready, loaders gone."""
    try:
        await page.wait_for_load_state("load", timeout=8000)
    except PlaywrightError:
        pass
    try:
        await page.wait_for_load_state("networkidle", timeout=2500)
    except PlaywrightError:
        pass
    try:
        await asyncio.wait_for(page.evaluate("document.fonts ? document.fonts.ready.then(() => true) : true"), 2)
    except (PlaywrightError, asyncio.TimeoutError):
        pass
    await wait_for_loaders(page)
    await asyncio.sleep(0.3)


async def declutter_page(page: Page) -> Dict[str, int]:
    """Accepts/hides consent banners and hides chat widgets and pop-ups; safe to call repeatedly."""
    try:
        result = await page.evaluate(DECLUTTER_SCRIPT)
        return result if isinstance(result, dict) else {}
    except PlaywrightError:
        return {}


async def record_session(
    scenes: Sequence[Scene],
    app_url: str,
    durations: Sequence[float],
    output_path: Path,
    log: LogFn,
    on_scene: Optional[Callable[[int], None]] = None,
    storage_state: Optional[Dict] = None,
    replayer=None,
) -> List[SceneSegment]:
    """Records every scene in one 1920x1080 session and returns each scene's footage window.

    `storage_state` (a saved sign-in session) makes the browser start signed in to the app. `replayer`
    (a fixtures.FixtureReplayer) answers the app's API calls with staged sample data, labelled on screen.
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
                    bypass_csp=True,  # so the injected cursor, ring and label render on strict-CSP sites
                )
                await guard_context(context)
                if replayer is not None:
                    await replayer.install(context)  # registered last, so it answers before the guard
                    await context.add_init_script("window.__DEMODAY_STAGED__ = true;")
                await context.add_init_script(INIT_SCRIPT)
                page = await context.new_page()
                session_start = time.monotonic()  # Playwright starts recording when the page is created
                video = page.video
                raw_path = await video.path() if video else None
                director = _Director(page, app_url, log, clock=lambda: time.monotonic() - session_start)

                for idx, (scene, duration) in enumerate(zip(scenes, durations)):
                    if on_scene:
                        on_scene(idx)
                    await director.open_scene(idx + 1, scene.route, first=(idx == 0))
                    start = time.monotonic()
                    await director.run(idx + 1, scene.actions, deadline=start + duration)
                    remaining = start + duration - time.monotonic()
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                    scene_start = start - session_start
                    segments.append(SceneSegment(start=scene_start, duration=duration,
                                                 focus=director.focus_for(scene_start, duration)))

                cleaned = director.cleaned
                if cleaned["accepted"] or cleaned["hidden"]:
                    log(f"Cleaned frames: {cleaned['accepted']} consent banner(s) accepted, "
                        f"{cleaned['hidden']} overlay(s) hidden")
                if replayer is not None:
                    log(f"Sample data answered {replayer.hits} request(s) during the recording")
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

    def __init__(self, page: Page, app_url: str, log: LogFn, clock: Callable[[], float]):
        self.page = page
        self.app_url = app_url
        self.log = log
        self.clock = clock
        self.pointer_shown = False
        self.landed: Dict[str, str] = {}  # requested URL -> URL after redirects
        self.focus: List[FocusTuple] = []  # (session time, cx, cy, zoom, hold)
        self.releases: List[float] = []    # session times when the view changed (navigation, scroll)
        self.cleaned = {"accepted": 0, "hidden": 0}
        page.on("framenavigated", lambda frame: frame == page.main_frame and self.releases.append(self.clock()))

    def focus_for(self, scene_start: float, duration: float) -> List[FocusTuple]:
        """Focus events inside a scene, re-based to the scene and cut short by the next view change."""
        out = []
        for t, cx, cy, zoom, hold in self.focus:
            if not scene_start <= t < scene_start + duration:
                continue
            end = min(t + hold, scene_start + duration)
            for release in self.releases:
                if t + 0.05 < release < end:
                    end = release
            if end - t >= 0.4:
                out.append((t - scene_start, cx, cy, zoom, end - t))
        return out

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
            finally:
                await self._unhighlight()
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
            await self._settle(budget_ms=min(4000, self._budget_ms(deadline)))
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
                await self._settle(budget_ms=min(4000, self._budget_ms(deadline)))
                return True
            self.log(f"Scene {scene_no}: selector {action.selector!r} not found, {action.type} skipped")
            return False
        if action.type == "click":
            label = await target.evaluate(ELEMENT_LABEL_SCRIPT)
            if is_risky(label=label, url=label):
                self.log(f"Scene {scene_no}: skipped clicking {label.strip()[:60]!r} "
                         "(sign-out and destructive actions are never clicked)")
                return False
        centre = await self._point_at(target)
        if not await target.evaluate(HIT_TEST_SCRIPT):
            self.log(f"Scene {scene_no}: {action.selector!r} is hidden or covered, {action.type} skipped")
            return False
        if action.type == "click":
            if centre:
                self.focus.append((self.clock(), centre[0], centre[1], CLICK_ZOOM, CLICK_HOLD_SEC))
            await target.click(timeout=min(ACTION_TIMEOUT_MS, self._budget_ms(deadline)))
            await self._after_interaction(deadline)
            return True
        # type
        text = action.value or ""
        submit = text.endswith("\n")
        text = text.rstrip("\n")[:MAX_TYPED_CHARS]
        delay = max(20, min(80, 1400 // max(1, len(text))))
        if centre:
            self.focus.append((self.clock(), centre[0], centre[1], TYPE_ZOOM, len(text) * delay / 1000 + 0.8))
        await target.click(timeout=ACTION_TIMEOUT_MS)
        await target.fill("", timeout=ACTION_TIMEOUT_MS)
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

    async def _settle(self, budget_ms: Optional[int] = None) -> None:
        """Waits for the page to finish, then clears banners and widgets (twice: some appear late)."""
        if budget_ms is None:
            await settle_page(self.page)
        else:
            try:
                await self.page.wait_for_load_state("networkidle", timeout=max(500, budget_ms - 1500))
            except PlaywrightError:
                pass
            await wait_for_loaders(self.page, timeout_sec=max(0.3, budget_ms / 1000 - 1.0))
        await self._declutter()
        if budget_ms is None:
            await asyncio.sleep(0.8)
            await self._declutter()

    async def _declutter(self) -> None:
        result = await declutter_page(self.page)
        self.cleaned["accepted"] += int(result.get("accepted", 0))
        self.cleaned["hidden"] += int(result.get("hidden", 0))

    async def _after_interaction(self, deadline: float) -> None:
        # If the interaction started a navigation, let the next page paint and settle before continuing.
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=min(4000, self._budget_ms(deadline)))
        except PlaywrightError:
            pass
        await wait_for_loaders(self.page, timeout_sec=min(2.5, self._budget_ms(deadline) / 1000))
        await self._declutter()

    async def _find(self, selector: Optional[str]) -> Optional[Locator]:
        if not selector:
            return None
        locator = self.page.locator(selector).filter(visible=True).first
        try:
            await locator.wait_for(state="visible", timeout=SELECTOR_TIMEOUT_MS)
        except PlaywrightError:  # missing element or invalid selector syntax
            return None
        return locator

    async def _point_at(self, target: Locator) -> Optional[Tuple[float, float]]:
        """Glides the visible cursor onto the element and rings it; returns the element's centre."""
        try:
            if await target.evaluate(REVEAL_SCRIPT):
                await self._wait_until_still(target)
                self.releases.append(self.clock())
            box = await target.bounding_box()
        except PlaywrightError:
            return None
        if not box:
            return None
        if not self.pointer_shown:
            await self.page.mouse.move(VIEWPORT["width"] * 0.5, VIEWPORT["height"] * 0.55)
            self.pointer_shown = True
        cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        try:
            await target.evaluate(HIGHLIGHT_SCRIPT)
        except PlaywrightError:
            pass
        await self.page.mouse.move(cx, cy, steps=20)
        await asyncio.sleep(0.35)
        return cx, cy

    async def _unhighlight(self) -> None:
        try:
            await self.page.evaluate(UNHIGHLIGHT_SCRIPT)
        except PlaywrightError:
            pass

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
        self.releases.append(self.clock())
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
