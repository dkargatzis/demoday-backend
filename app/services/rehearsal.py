"""The dry run: every step of the script performed once in a headless browser, without a camera, before any voice
or recording time is spent.

It opens each scene's page the way the recorder will (same viewport, saved login, sample-data replay, network
guard) and checks every action: the page answers, the element is there, visible and not covered, nothing risky.
It follows same-site links so later steps are reached for real, but it never clicks other elements or types: a
button or a form can change data in the real account, and the recording is the one run that may do that. A step
after such a button cannot be reached in a rehearsal; it is reported as unverified, never as a failure.

`apply` drops the steps that would fail on camera (missing, hidden or covered elements, risky targets, pages that
do not load), so the recorder does not spend narration time on them; everything else stays as planned.
"""
import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright

from app.models.schemas import Scene
from app.services.netguard import guard_context
from app.services.recorder import (ELEMENT_LABEL_SCRIPT, HIT_TEST_SCRIPT, NAV_TIMEOUT_MS, VIEWPORT, _link_href, _same_page,
                                   declutter_page, is_risky, launch_chromium, resolve_url, session_storage_script,
                                   split_session_storage, wait_for_loaders)

logger = logging.getLogger("demoday.rehearsal")

FIND_TIMEOUT_MS = 2_500     # a little more patience than the recorder: nothing is on camera yet
REHEARSAL_BUDGET_SEC = 75   # the whole dry run; steps not reached by then are reported as unverified
DROP = ("missing", "hidden", "risky", "failed")  # what the recorder would skip on camera anyway
LINK_SCRIPT = "el => { const a = el.closest('a[href]'); return a ? a.href : null; }"
PASSWORD_SCRIPT = "() => Array.from(document.querySelectorAll('input[type=password]')).some(i => i.offsetParent !== null)"


def _result(scene: int, index: int, action, status: str, note: str = "") -> Dict[str, Any]:
    target = action.selector or action.value or ""
    return {"scene": scene, "action": index, "type": action.type, "target": str(target)[:120], "status": status, "note": note}


async def dry_run(scenes: Sequence[Scene], app_url: str, log: Optional[Callable[[str], None]] = None,
                  storage_state: Optional[Dict] = None, replayer=None) -> List[Dict[str, Any]]:
    """One result per action, in order: status ok | not_clicked | unverified | missing | hidden | risky | failed."""
    results: List[Dict[str, Any]] = []
    started = time.monotonic()
    playwright_state, session_storage = split_session_storage(storage_state)
    async with async_playwright() as pw:
        browser = await launch_chromium(pw)
        try:
            context = await browser.new_context(viewport=VIEWPORT, locale="en-US", storage_state=playwright_state,
                                                service_workers="block", bypass_csp=True)
            await guard_context(context)
            if session_storage:
                await context.add_init_script(session_storage_script(session_storage))
            if replayer is not None:
                await replayer.install(context)
            page = await context.new_page()

            async def goto(url: str) -> str:
                """'' when the page loaded; otherwise why not."""
                try:
                    response = await page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
                except PlaywrightError as exc:
                    return (str(exc).strip().splitlines() or ["navigation failed"])[0][:120]
                if response is not None and response.status >= 400:
                    return f"HTTP {response.status}"
                try:
                    await page.wait_for_load_state("networkidle", timeout=3000)
                except PlaywrightError:
                    pass
                await wait_for_loaders(page, timeout_sec=2.0)
                await declutter_page(page)
                if await page.evaluate(PASSWORD_SCRIPT):
                    return "landed on a sign-in form"
                return ""

            for scene_no, scene in enumerate(scenes, 1):
                blocked = False  # an earlier button or form in this scene was not performed
                if time.monotonic() - started > REHEARSAL_BUDGET_SEC:
                    results += [_result(scene_no, i, a, "unverified", "out of rehearsal time") for i, a in enumerate(scene.actions)]
                    continue
                target = resolve_url(app_url, scene.route) or app_url
                if scene_no == 1 or not _same_page(page.url, target):
                    problem = await goto(target)
                    if problem:
                        if log:
                            log(f"Dry run: scene {scene_no} page {scene.route} did not load ({problem})")
                        results += [_result(scene_no, i, a, "failed", f"page {scene.route}: {problem}")
                                    for i, a in enumerate(scene.actions)]
                        continue
                for index, action in enumerate(scene.actions):
                    if time.monotonic() - started > REHEARSAL_BUDGET_SEC:
                        results.append(_result(scene_no, index, action, "unverified", "out of rehearsal time"))
                        continue
                    if action.type in ("wait", "scroll") and not action.selector:
                        results.append(_result(scene_no, index, action, "ok"))
                        continue
                    if action.type == "navigate":
                        url = resolve_url(app_url, action.value)
                        if url is None or is_risky(url=url):
                            results.append(_result(scene_no, index, action, "risky" if url else "failed",
                                                   "sign-out or destructive page" if url else "unsupported URL"))
                            continue
                        problem = await goto(url)
                        results.append(_result(scene_no, index, action, "failed" if problem else "ok", problem))
                        blocked = blocked and bool(problem)
                        continue
                    locator = page.locator(action.selector or "").filter(visible=True).first
                    try:
                        await locator.wait_for(state="visible", timeout=FIND_TIMEOUT_MS)
                        found = True
                    except PlaywrightError:
                        found = False
                    if not found:
                        href = _link_href(action.selector) if action.type == "click" else None
                        url = resolve_url(app_url, href) if href else None
                        if url and not is_risky(url=url):  # the recorder opens a link that is not on screen directly
                            problem = await goto(url)
                            results.append(_result(scene_no, index, action, "failed" if problem else "ok",
                                                   problem or "opened directly (not on screen)"))
                        else:
                            results.append(_result(scene_no, index, action, "unverified" if blocked else "missing",
                                                   "may appear after an earlier step" if blocked else "not on the page"))
                        continue
                    try:
                        label = await locator.evaluate(ELEMENT_LABEL_SCRIPT)
                        covered = not await locator.evaluate(HIT_TEST_SCRIPT)
                        link = await locator.evaluate(LINK_SCRIPT) if action.type == "click" else None
                    except PlaywrightError as exc:
                        results.append(_result(scene_no, index, action, "missing", (str(exc).splitlines() or [""])[0][:100]))
                        continue
                    if action.type == "click" and is_risky(label=label, url=link or label):
                        results.append(_result(scene_no, index, action, "risky", f"{label.strip()[:60]!r}"))
                    elif covered:
                        results.append(_result(scene_no, index, action, "hidden", "covered by another element"))
                    elif action.type == "click" and link and resolve_url(app_url, link) and not is_risky(url=link):
                        problem = await goto(link)  # a same-site link: followed, so later steps are reached for real
                        results.append(_result(scene_no, index, action, "failed" if problem else "ok", problem))
                    elif action.type in ("click", "type"):
                        blocked = True  # a button or a form: never performed in a rehearsal
                        results.append(_result(scene_no, index, action, "not_clicked" if action.type == "click" else "ok",
                                               "found; buttons and forms run only in the recording"))
                    else:
                        results.append(_result(scene_no, index, action, "ok"))
        finally:
            await browser.close()
    return results


def apply(scenes: Sequence[Scene], results: Sequence[Dict[str, Any]], log: Optional[Callable[[str], None]] = None) -> int:
    """Drops the actions the rehearsal showed would fail on camera; returns how many were dropped."""
    drop = {(r["scene"], r["action"]) for r in results if r["status"] in DROP}
    for scene_no, scene in enumerate(scenes, 1):
        kept = [a for i, a in enumerate(scene.actions) if (scene_no, i) not in drop]
        for i, action in enumerate(scene.actions):
            if (scene_no, i) in drop and log:
                why = next(r for r in results if (r["scene"], r["action"]) == (scene_no, i))
                log(f"Dry run: scene {scene_no} {action.type} {why['target']!r} dropped ({why['status']}: {why['note'] or 'n/a'})")
        scene.actions = kept
    return len(drop)


def summary(results: Sequence[Dict[str, Any]], dropped: int = 0) -> str:
    counts: Dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    parts = [f"{counts.get('ok', 0)} landed"]
    for status, words in (("not_clicked", "found (clicked only on camera)"), ("unverified", "unverified"),
                          ("missing", "missing"), ("hidden", "covered"), ("risky", "risky"), ("failed", "failed")):
        if counts.get(status):
            noun = ("button " if counts[status] == 1 else "buttons ") if status == "not_clicked" else ""
            parts.append(f"{counts[status]} {noun}{words}")
    return f"{len(results)} steps rehearsed: " + ", ".join(parts) + (f"; {dropped} dropped" if dropped else "")
