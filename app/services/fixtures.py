"""Staged sample data: capture an app's JSON API responses, improve them with Gemini, replay them while recording.

A fixture set belongs to one user and one app origin. During a probe or recording, matching requests are answered
from the set instead of the network, so dashboards show curated, well-populated data. Recordings that use it carry a
"Sample data" label.
"""
import hashlib
import json
import logging
import re
import time
import uuid
from datetime import date
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from playwright.async_api import BrowserContext, Error as PlaywrightError, Request, Route, async_playwright

from app import db
from app.config import GEMINI_MODEL, get_secret

logger = logging.getLogger("demoday.fixtures")

MAX_BODY_BYTES = 300_000
MAX_ENTRIES = 60
MAX_SET_BYTES = 2_000_000
PROMPT_BODY_CHARS = 9_000
PROMPT_TOTAL_CHARS = 70_000


# --------------------------------------------------------------------------- storage


def _row_to_set(row, with_entries: bool = True) -> Dict[str, Any]:
    entries = json.loads(row["entries"])
    out = {"id": row["id"], "origin": row["origin"], "polished": bool(row["polished"]), "brief": row["brief"],
           "updated_at": row["updated_at"], "entry_count": len(entries),
           "enabled_count": sum(1 for e in entries if e.get("enabled", True))}
    if with_entries:
        out["entries"] = entries
    return out


def list_sets(user_id: str) -> List[Dict[str, Any]]:
    rows = db.query("SELECT * FROM fixtures WHERE user_id = ? ORDER BY updated_at DESC", (user_id,))
    return [_row_to_set(r, with_entries=False) for r in rows]


def get_set(user_id: str, set_id: str) -> Optional[Dict[str, Any]]:
    row = db.one("SELECT * FROM fixtures WHERE id = ? AND user_id = ?", (set_id, user_id))
    return _row_to_set(row) if row else None


def get_set_for_origin(user_id: str, origin: str) -> Optional[Dict[str, Any]]:
    row = db.one("SELECT * FROM fixtures WHERE user_id = ? AND origin = ?", (user_id, origin))
    return _row_to_set(row) if row else None


def save_set(user_id: str, origin: str, entries: List[Dict], polished: bool, brief: str = "") -> Dict[str, Any]:
    payload = json.dumps(entries)
    if len(payload) > MAX_SET_BYTES:
        raise ValueError("The sample data set is too large (2 MB limit)")
    db.execute(
        "INSERT INTO fixtures (id, user_id, origin, entries, polished, brief, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (user_id, origin) DO UPDATE SET entries = excluded.entries, polished = excluded.polished, "
        "brief = excluded.brief, updated_at = excluded.updated_at",
        (uuid.uuid4().hex, user_id, origin, payload, int(polished), brief[:500], time.time()),
    )
    return get_set_for_origin(user_id, origin)


def delete_set(user_id: str, set_id: str) -> bool:
    return db.execute("DELETE FROM fixtures WHERE id = ? AND user_id = ?", (set_id, user_id)) > 0


def validate_entries(entries: Any) -> List[Dict]:
    """Normalizes user-edited entries; raises ValueError with a plain message on bad input."""
    if not isinstance(entries, list):
        raise ValueError("entries must be a list")
    out = []
    for i, raw in enumerate(entries[:MAX_ENTRIES]):
        if not isinstance(raw, dict) or not isinstance(raw.get("url"), str):
            raise ValueError(f"entry {i + 1} needs a url")
        body = raw.get("body")
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except ValueError as exc:
                raise ValueError(f"entry {i + 1} ({raw['url'][:60]}): body is not valid JSON ({exc})") from exc
        method = str(raw.get("method", "GET")).upper()
        out.append({"url": raw["url"], "method": method, "status": int(raw.get("status", 200)),
                    "body": body, "post_data": raw.get("post_data") or None,
                    "enabled": bool(raw.get("enabled", True)), "source": raw.get("source", "edited")})
    return out


# --------------------------------------------------------------------------- capture


# Analytics, feature flags and error tracking: never worth replaying, and replaying flags can change the app.
NOISE_URL = re.compile(r"posthog|/flags/?(\?|$)|/decide/?(\?|$)|/e/?\?|segment\.io|sentry|analytics|mixpanel|amplitude|"
                       r"hotjar|intercom|datadog|googletagmanager|google-analytics|/collect\b|/batch\b|/track\b|"
                       r"launchdarkly|statsig|split\.io|/telemetry|/metrics\b|/events\b", re.IGNORECASE)
# Session/auth responses are left disabled by default: replaying a stale one signs the app out.
AUTH_URL = re.compile(r"/auth\b|/session|/login|/logout|/token|/oauth|/whoami|/me/?(\?|$)|/users?/me|/account\b|/csrf",
                      re.IGNORECASE)


def is_noise(url: str) -> bool:
    return bool(NOISE_URL.search(url))


def is_auth(url: str) -> bool:
    return bool(AUTH_URL.search(url))


def _key(method: str, url: str, post_data: Optional[str]) -> str:
    parsed = urlparse(url)
    base = f"{method.upper()} {parsed.scheme}://{parsed.netloc}{parsed.path}"
    if post_data:
        base += " " + hashlib.sha1(post_data.encode("utf-8", "replace")).hexdigest()[:12]
    return base


async def capture_fixtures(app_url: str, storage_state: Optional[Dict], settle, declutter) -> List[Dict]:
    """Loads the app (signed in when a session is given) and records its JSON API responses.

    `settle` and `declutter` are the recorder's page helpers, so the capture sees the same page the recording will.
    """
    from app.services.netguard import guard_context
    from app.services.recorder import (VIEWPORT, is_risky, launch_chromium, session_storage_script,
                                       split_session_storage)

    entries: Dict[str, Dict] = {}
    storage_state, session_storage = split_session_storage(storage_state)

    async def on_response(response) -> None:
        request = response.request
        if request.resource_type not in ("xhr", "fetch") or not 200 <= response.status < 300:
            return
        if "json" not in (response.headers.get("content-type") or "").lower():
            return
        try:
            text = await response.text()
        except PlaywrightError:
            return
        if len(text) > MAX_BODY_BYTES:
            return
        try:
            body = json.loads(text)
        except ValueError:
            return
        if is_noise(request.url):
            return
        post_data = request.post_data if request.method in ("POST", "PUT", "PATCH") else None
        key = _key(request.method, request.url, post_data)
        auth = is_auth(request.url)
        entries[key] = {"url": request.url.split("#", 1)[0], "method": request.method, "status": response.status,
                        "body": body, "post_data": post_data, "enabled": not auth,
                        "source": "captured (auth endpoint, off by default)" if auth else "captured"}

    async with async_playwright() as pw:
        browser = await launch_chromium(pw)
        try:
            context = await browser.new_context(viewport=VIEWPORT, locale="en-US", storage_state=storage_state,
                                                service_workers="block")
            await guard_context(context)
            if session_storage:
                await context.add_init_script(session_storage_script(session_storage))
            page = await context.new_page()
            page.on("response", lambda r: _spawn(on_response(r)))
            await page.goto(app_url, wait_until="domcontentloaded", timeout=30_000)
            await _collect_page(page, settle, declutter)
            # The demo will visit other pages of the app too, so follow its navigation links (same origin).
            try:
                links = await page.evaluate(NAV_LINKS_SCRIPT)
            except PlaywrightError:
                links = []
            visited = {urlparse(page.url).path.rstrip("/") or "/"}
            for link in links:
                if len(visited) > MAX_CAPTURE_PAGES or len(entries) >= MAX_ENTRIES:
                    break
                path = urlparse(link).path.rstrip("/") or "/"
                if path in visited or is_risky(url=link):
                    continue
                visited.add(path)
                try:
                    await page.goto(link, wait_until="domcontentloaded", timeout=20_000)
                    await _collect_page(page, settle, declutter)
                except PlaywrightError:
                    continue
        finally:
            await browser.close()
    return list(entries.values())[:MAX_ENTRIES]


MAX_CAPTURE_PAGES = 6
NAV_LINKS_SCRIPT = """
() => {
  const origin = location.origin, out = [], seen = new Set();
  const scope = document.querySelector("nav, header, aside, [role=navigation]") || document;
  for (const a of [...scope.querySelectorAll("a[href]"), ...document.querySelectorAll("a[href]")]) {
    let u; try { u = new URL(a.getAttribute("href"), location.href); } catch { continue; }
    if (u.origin !== origin || u.hash && u.pathname === location.pathname) continue;
    const key = u.pathname.replace(/\\/$/, "") || "/";
    if (seen.has(key)) continue;
    seen.add(key); out.push(u.origin + u.pathname + u.search);
  }
  return out.slice(0, 12);
}
"""


async def _collect_page(page, settle, declutter) -> None:
    await settle(page)
    await declutter(page)
    for _ in range(3):  # lazy-loaded sections below the fold
        await page.evaluate("window.scrollBy({top: window.innerHeight * 0.8, behavior: 'instant'})")
        await page.wait_for_timeout(600)
    try:
        await page.wait_for_load_state("networkidle", timeout=4000)
    except PlaywrightError:
        pass
    await page.wait_for_timeout(600)


_pending = set()


def _spawn(coro) -> None:
    import asyncio

    task = asyncio.ensure_future(coro)
    _pending.add(task)
    task.add_done_callback(_pending.discard)


# --------------------------------------------------------------------------- Gemini polish


def _trim_for_prompt(value: Any, depth: int = 0) -> Any:
    """Keeps the shape but shortens long lists so the prompt stays small."""
    if isinstance(value, list):
        return [_trim_for_prompt(v, depth + 1) for v in value[:4]]
    if isinstance(value, dict):
        return {k: _trim_for_prompt(v, depth + 1) for k, v in list(value.items())[:60]}
    if isinstance(value, str) and len(value) > 300:
        return value[:300] + "…"
    return value


def conform(original: Any, candidate: Any) -> Any:
    """Returns `candidate` with every list that was empty in `original` emptied again (their item shape is unknown)."""
    if isinstance(original, list):
        return [] if not original else candidate
    if isinstance(original, dict) and isinstance(candidate, dict):
        return {k: (conform(original[k], v) if k in original else v) for k, v in candidate.items()}
    return candidate


def same_shape(original: Any, candidate: Any) -> bool:
    """True when `candidate` has the same JSON structure as `original`: same keys for objects, and every list
    item shaped like some item of the original list (lists often mix shapes, e.g. a stat with "–" for no data)."""
    if original is None or candidate is None:
        return True
    if isinstance(original, dict):
        return isinstance(candidate, dict) and set(candidate) == set(original) and all(
            same_shape(original[k], candidate[k]) for k in original)
    if isinstance(original, list):
        if not isinstance(candidate, list):
            return False
        samples = original[:25]
        return not samples or all(any(same_shape(sample, item) for sample in samples) for item in candidate)
    if isinstance(original, bool) or isinstance(candidate, bool):
        return isinstance(original, bool) and isinstance(candidate, bool)
    if isinstance(original, (int, float)):
        return isinstance(candidate, (int, float))
    return isinstance(candidate, type(original))


async def polish_with_gemini(entries: List[Dict], origin: str, brief: str = "") -> Tuple[List[Dict], str]:
    """Asks Gemini for richer sample data with the same schema. Entries it can't improve safely stay as captured."""
    from app.services.planner import _generate_json

    if not get_secret("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set on the server")
    enabled = [e for e in entries if e.get("enabled", True)]
    if not enabled:
        raise RuntimeError("No captured endpoints to improve")
    shown, used = [], 0
    for e in enabled:
        snippet = json.dumps(_trim_for_prompt(e["body"]), ensure_ascii=False)[:PROMPT_BODY_CHARS]
        if used + len(snippet) > PROMPT_TOTAL_CHARS:
            break
        used += len(snippet)
        shown.append({"method": e["method"], "url": e["url"], "body": snippet})
    prompt = f"""You are preparing demo data for a product video of the web app at {origin}.
Below are real JSON API responses the app's dashboard loads (long lists are trimmed to a few items).
Rewrite each response as polished SAMPLE DATA for the demo:
- Keep EXACTLY the same JSON schema: same keys, same value types, same nesting, same enum-like values.
- Make it well populated and realistic: lists with 6-12 items where a dashboard would show a list, healthy metrics
  that tell a growth story, recent dates (today is {date.today().isoformat()}), consistent ids and references
  between endpoints.
- Use fictional but plausible names (people, companies, repositories). Never real customers or real people.
- Keep authentication, session, feature-flag, configuration and pagination fields unchanged.
- A list that is empty in the captured data must stay empty: its item shape is unknown, and the app would render
  invented fields as "undefined".
{('- The user asks: ' + brief.strip()) if brief.strip() else ''}
Return JSON only: {{"responses": [{{"method": "GET", "url": "...", "body": <the rewritten JSON>}}]}}
Include every endpoint below, with the same method and url.

ENDPOINTS
{json.dumps(shown, ensure_ascii=False, indent=1)}
"""
    data = await _generate_json(prompt)
    responses = data.get("responses") if isinstance(data, dict) else None
    if not isinstance(responses, list):
        raise RuntimeError("Gemini returned no responses")
    by_key = {_key(r.get("method", "GET"), r.get("url", ""), None): r.get("body") for r in responses
              if isinstance(r, dict)}
    improved, kept = 0, 0
    out = []
    for e in entries:
        candidate = by_key.get(_key(e["method"], e["url"], None))
        if isinstance(candidate, str):
            try:
                candidate = json.loads(candidate)
            except ValueError:
                candidate = None
        if candidate is not None:
            candidate = conform(e["body"], candidate)
        if e.get("enabled", True) and candidate is not None and same_shape(e["body"], candidate):
            out.append({**e, "body": candidate, "source": f"gemini ({GEMINI_MODEL})"})
            improved += 1
        else:
            out.append(e)
            kept += 1
    if improved == 0:
        raise RuntimeError("Gemini's data didn't match the captured schema, so nothing was changed")
    return out, f"{improved} endpoint(s) rewritten by Gemini" + (f", {kept} kept as captured" if kept else "")


# --------------------------------------------------------------------------- replay


class FixtureReplayer:
    """Answers matching requests from a fixture set; everything else falls through to the next handler."""

    def __init__(self, entries: List[Dict]):
        self.by_key: Dict[str, Dict] = {}
        self.by_path: Dict[str, Dict] = {}
        for e in entries:
            if not e.get("enabled", True) or is_noise(e["url"]):
                continue
            self.by_key[_key(e["method"], e["url"], e.get("post_data"))] = e
            self.by_path.setdefault(_key(e["method"], e["url"], None), e)
        self.hits = 0

    def match(self, request: Request) -> Optional[Dict]:
        post_data = request.post_data if request.method in ("POST", "PUT", "PATCH") else None
        return self.by_key.get(_key(request.method, request.url, post_data)) or \
            self.by_path.get(_key(request.method, request.url, None))

    async def install(self, context: BrowserContext) -> None:
        # Registered after the network guard, so it runs first; unmatched requests fall back to the guard.
        await context.route("**/*", self._handle)

    async def _handle(self, route: Route) -> None:
        request = route.request
        cors = {"access-control-allow-origin": request.headers.get("origin", "*"),
                "access-control-allow-credentials": "true",
                "access-control-allow-headers": request.headers.get("access-control-request-headers", "*"),
                "access-control-allow-methods": "GET,POST,PUT,PATCH,DELETE,OPTIONS"}
        if request.method == "OPTIONS":
            target = request.headers.get("access-control-request-method", "GET")
            if self.by_path.get(_key(target, request.url, None)):
                await route.fulfill(status=204, headers=cors)
                return
        entry = self.match(request)
        if entry is None:
            await route.fallback()
            return
        self.hits += 1
        await route.fulfill(status=entry.get("status", 200), content_type="application/json",
                            body=json.dumps(entry["body"]), headers={**cors, "cache-control": "no-store"})
