"""Live sign-in capture: a server-side browser the user drives from the dashboard.

The dashboard shows a live JPEG stream of the page and forwards clicks, typing and scrolling. When the user is
signed in, the browser's storage state (cookies, localStorage and IndexedDB, where Firebase Auth and similar SDKs
keep the signed-in user) is saved encrypted for that user and app origin. Whatever the user types goes straight into
their app's page; passwords are never stored.
"""
import asyncio
import base64
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple
from urllib.parse import urlparse

from playwright.async_api import BrowserContext, Page, async_playwright
from playwright.async_api import Error as PlaywrightError

from app.services.netguard import BlockedURL, ensure_public_url, guard_context
from app.services.recorder import launch_chromium

logger = logging.getLogger("demoday.capture")

VIEWPORT = {"width": 1280, "height": 800}
IDLE_TIMEOUT_SEC = 300
MAX_LIFETIME_SEC = 900
SPECIAL_KEYS = {
    "Enter", "Tab", "Shift+Tab", "Backspace", "Delete", "Escape", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
    "Home", "End", "PageUp", "PageDown", "ControlOrMeta+a",
}


def origin_of(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}".lower()


def unpack_login(blob: bytes) -> Tuple[Dict, Dict]:
    """(playwright storage_state, {origin: sessionStorage items}) from a saved login, old or new format."""
    import json

    data = json.loads(blob)
    if "cookies" in data or "origins" in data:  # first format: the bare storage state
        return data, {}
    return data.get("storage_state") or {}, data.get("session_storage") or {}


def indexed_db_records(storage_state: Dict) -> int:
    """Records across the IndexedDB snapshot (Firebase Auth keeps the signed-in user in `firebaseLocalStorageDb`)."""
    return sum(len(store.get("records") or []) for origin in storage_state.get("origins") or []
               for db in origin.get("indexedDB") or [] for store in db.get("stores") or [])


def describe_login(storage_state: Dict, session_storage: Dict) -> str:
    """'17 cookies, 2 localStorage keys, 1 IndexedDB record, 4 sessionStorage keys' for the job log."""
    origins = storage_state.get("origins") or []
    counts = [(len(storage_state.get("cookies") or []), "cookie"),
              (sum(len(o.get("localStorage") or []) for o in origins), "localStorage key"),
              (indexed_db_records(storage_state), "IndexedDB record"),
              (sum(len(v) for v in session_storage.values()), "sessionStorage key")]
    return ", ".join(f"{n} {label}{'' if n == 1 else 's'}" for n, label in counts if n) or "nothing saved"


def session_storage_script(session_storage: Dict) -> str:
    """Init script that seeds sessionStorage for matching origins (once per tab)."""
    import json

    return ("(() => { const data = " + json.dumps(session_storage) + "; const items = data[location.origin];"
            " if (!items || sessionStorage.getItem('__demoday_seeded')) return;"
            " for (const [k, v] of Object.entries(items)) sessionStorage.setItem(k, v);"
            " sessionStorage.setItem('__demoday_seeded', '1'); })();")


@dataclass
class CaptureSession:
    id: str
    user_id: str
    app_url: str
    origin: str
    playwright: Any
    browser: Any
    context: BrowserContext
    page: Optional[Page] = None
    cdp: Any = None
    frame: Optional[bytes] = None
    frame_seq: int = 0
    url: str = ""
    notice: Optional[str] = None
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    created: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class CaptureManager:
    def __init__(self) -> None:
        self.sessions: Dict[str, CaptureSession] = {}
        self._tasks: Set[asyncio.Task] = set()
        self._reaper: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------ lifecycle

    def get(self, capture_id: str, user_id: str) -> Optional[CaptureSession]:
        session = self.sessions.get(capture_id)
        return session if session and session.user_id == user_id else None

    def is_open(self, session: CaptureSession) -> bool:
        return self.sessions.get(session.id) is session

    async def start(self, user_id: str, app_url: str) -> CaptureSession:
        await ensure_public_url(app_url)
        for other in [s for s in self.sessions.values() if s.user_id == user_id]:
            await self.close(other)  # one live browser per user
        pw = await async_playwright().start()
        try:
            browser = await launch_chromium(pw)
            context = await browser.new_context(viewport=VIEWPORT, screen=VIEWPORT, device_scale_factor=1,
                                                locale="en-US", service_workers="block")
            await guard_context(context)
        except Exception:
            await pw.stop()
            raise
        session = CaptureSession(id=uuid.uuid4().hex, user_id=user_id, app_url=app_url, origin=origin_of(app_url),
                                 playwright=pw, browser=browser, context=context)
        self.sessions[session.id] = session
        context.on("page", lambda page: self._spawn(self._attach(session, page)))  # popups (OAuth, SSO)
        page = await context.new_page()
        await self._attach(session, page)
        try:
            await page.goto(app_url, wait_until="domcontentloaded", timeout=30_000)
        except PlaywrightError as exc:
            session.notice = f"The page didn't finish loading: {_short(exc)}"
        self._status(session)
        if self._reaper is None or self._reaper.done():
            self._reaper = asyncio.create_task(self._reap())
        return session

    async def finish(self, session: CaptureSession) -> Tuple[Dict, str]:
        state = await session.context.storage_state(indexed_db=True)
        last_url = session.page.url if session.page else session.app_url
        await self.close(session)
        return state, last_url

    async def close(self, session: CaptureSession) -> None:
        if self.sessions.pop(session.id, None) is None:
            return
        session.changed.set()  # wakes the stream so it can end
        for closer in (session.context.close, session.browser.close, session.playwright.stop):
            try:
                await closer()
            except Exception:  # already gone
                pass

    async def shutdown(self) -> None:
        for session in list(self.sessions.values()):
            await self.close(session)

    async def _reap(self) -> None:
        while self.sessions:
            await asyncio.sleep(15)
            now = time.monotonic()
            for session in list(self.sessions.values()):
                if now - session.last_activity > IDLE_TIMEOUT_SEC or now - session.created > MAX_LIFETIME_SEC:
                    logger.info("closing idle sign-in capture %s", session.id)
                    await self.close(session)

    # ------------------------------------------------------------------ screencast

    async def _attach(self, session: CaptureSession, page: Page) -> None:
        """Streams `page` (the newest tab, e.g. an OAuth popup) instead of the previous one."""
        async with session.lock:
            if session.page is page or not self.is_open(session) or page.is_closed():
                return
            if session.cdp is not None:
                try:
                    await session.cdp.send("Page.stopScreencast")
                    await session.cdp.detach()
                except PlaywrightError:
                    pass
            cdp = await session.context.new_cdp_session(page)
            session.page, session.cdp = page, cdp
            cdp.on("Page.screencastFrame", lambda params: self._spawn(self._on_frame(session, cdp, params)))
            page.on("framenavigated", lambda frame: frame == page.main_frame and self._status(session))
            page.on("close", lambda _: self._spawn(self._on_page_closed(session, page)))
            await cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 60, "everyNthFrame": 1,
                                                    "maxWidth": VIEWPORT["width"], "maxHeight": VIEWPORT["height"]})
            self._status(session)

    async def _on_frame(self, session: CaptureSession, cdp: Any, params: Dict) -> None:
        try:
            await cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})
        except PlaywrightError:
            return
        if cdp is session.cdp:
            session.frame = base64.b64decode(params["data"])
            session.frame_seq += 1
            session.changed.set()

    async def _on_page_closed(self, session: CaptureSession, page: Page) -> None:
        if session.page is page and self.is_open(session):
            remaining = [p for p in session.context.pages if not p.is_closed()]
            if remaining:
                session.page = None
                await self._attach(session, remaining[-1])

    def _status(self, session: CaptureSession) -> None:
        if session.page is not None and not session.page.is_closed():
            session.url = session.page.url
        session.changed.set()

    # ------------------------------------------------------------------ input

    async def handle_input(self, session: CaptureSession, message: Dict) -> None:
        page = session.page
        if page is None or page.is_closed():
            return
        session.last_activity = time.monotonic()
        kind = message.get("type")
        try:
            if kind == "click":
                x = min(max(float(message.get("x", 0)), 0), VIEWPORT["width"] - 1)
                y = min(max(float(message.get("y", 0)), 0), VIEWPORT["height"] - 1)
                await page.mouse.click(x, y)
            elif kind == "text":
                text = str(message.get("text", ""))[:500]
                if text:
                    await page.keyboard.type(text)
            elif kind == "key":
                key = str(message.get("key", ""))
                if key in SPECIAL_KEYS:
                    await page.keyboard.press(key)
            elif kind == "wheel":
                await page.mouse.wheel(0, min(max(float(message.get("dy", 0)), -2000), 2000))
            elif kind == "back":
                await page.go_back(wait_until="domcontentloaded", timeout=15_000)
            elif kind == "navigate":
                url = str(message.get("url", "")).strip()
                await ensure_public_url(url)
                await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        except BlockedURL as exc:
            self._notify(session, str(exc))
        except (PlaywrightError, ValueError) as exc:
            self._notify(session, f"That didn't work: {_short(exc)}")

    def _notify(self, session: CaptureSession, message: str) -> None:
        session.notice = message
        session.changed.set()

    def _spawn(self, coroutine) -> None:
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(lambda t: t.cancelled() or not t.exception()
                               or logger.warning("capture task failed: %s", t.exception()))


def _short(exc: BaseException) -> str:
    text = str(exc).strip()
    return (text.splitlines()[0] if text else type(exc).__name__)[:160]


CAPTURES = CaptureManager()
