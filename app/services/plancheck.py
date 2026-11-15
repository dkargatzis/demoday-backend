"""Pre-camera plan check: the storyboard is verified against the live app's inventory before any voice or camera
time is spent, Gemini is asked to revise only the scenes with problems, and whatever is still invalid is dropped
deterministically. Narration, titles and `why` are never changed here, so no new claims can enter the video.

Everything is pure and fail-soft: `verify`, `notes`, `drop_invalid` and `describe` are arithmetic on the storyboard
and never raise; `repair` makes one bounded Gemini call (the orchestrator wraps it in its optional-step timeout) and
returns None when the answer is unusable.
"""
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple
from urllib.parse import urljoin, urlparse

from app.models.schemas import Action, Scene, Storyboard
from app.services.gemini import generate_json
from app.services.planner import (BOOKEND_ALLOWANCE_SEC, TAIL_ALLOWANCE_SEC, WORDS_PER_SECOND, _inventory_lines,
                                  _sanitize_action, _site_map_lines)
from app.services.recorder import _link_href, is_risky, resolve_url

logger = logging.getLogger("demoday.plancheck")

REPAIR_TIMEOUT_SEC = 15
WORD_TOLERANCE = 0.35  # narration outside +-35 % of the per-scene word budget is noted, never repaired
MAX_ACTIONS = 8  # same cap as the planner's sanitizer
IssueKind = Literal["selector", "route", "risky"]
REASONS = {"selector": "selector not in the inventory", "route": "unknown route", "risky": "sign-out or destructive"}


@dataclass
class Issue:
    scene: int  # 1-based scene number
    kind: IssueKind
    text: str  # one sentence for the log and the repair prompt
    action: Optional[int] = None  # index into Scene.actions; None when the scene's own route is the problem

    @property
    def reason(self) -> str:
        return REASONS[self.kind]


# --------------------------------------------------------------------------- verify


def _norm_path(target: str, base: str = "https://app/") -> str:
    """'/features' for '/features/', 'features/' or 'https://host/features/?x=1#top' (query kept, fragment dropped)."""
    parsed = urlparse(urljoin(base, (target or "").strip()))
    path = parsed.path.rstrip("/") or "/"
    return path + ("?" + parsed.query if parsed.query else "")


def _canon(selector: Optional[str]) -> str:
    return " ".join((selector or "").replace("'", '"').split())


def _known(app_info: dict, repo_routes: Sequence[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    """(known routes, known selectors, inventory input selectors) from the inventory, the site map and the repo."""
    app_url = app_info.get("url") or ""
    pages = [p for p in app_info.get("pages") or [] if isinstance(p, dict)]
    links = [l for l in app_info.get("links") or [] if isinstance(l, dict)]
    buttons = [b for b in app_info.get("buttons") or [] if isinstance(b, dict)]
    inputs = [i for i in app_info.get("inputs") or [] if isinstance(i, dict)]
    routes = {"/", _norm_path(app_url)} if app_url else {"/"}
    routes |= {_norm_path(p["path"]) for p in pages if p.get("path")}
    routes |= {_norm_path(l["path"]) for l in links if l.get("path")}
    routes |= {_norm_path(r) for r in repo_routes or () if isinstance(r, str) and r}
    selectors = {_canon(x["selector"]) for x in [*links, *buttons, *inputs] if x.get("selector")}
    selectors |= {_canon(p["open_selector"]) for p in pages if p.get("open_selector")}
    return routes, selectors, {_canon(i["selector"]) for i in inputs if i.get("selector")}


def _route_known(target: str, app_url: str, routes: Set[str]) -> bool:
    """A route or href is known when it resolves (as the recorder resolves it) to the app's host and a known path."""
    url = resolve_url(app_url, target) if app_url else None
    if not url:
        return False
    parsed = urlparse(url)
    if parsed.netloc != urlparse(app_url).netloc:
        return False  # the recorder would leave the app
    return _norm_path(url) in routes


def verify(storyboard: Storyboard, app_info: dict, repo_routes: Sequence[str] = ()) -> List[Issue]:
    """Issues that would waste camera time or leave the app: a click/type whose selector is not in the inventory
    (an `a[href]` to a known route is fine: the recorder opens it directly), a type into a non-input, a scene route
    or navigate target outside the known routes, and sign-out/destructive routes or actions. Never raises."""
    try:
        return _verify(storyboard, app_info, repo_routes)
    except Exception:  # noqa: BLE001 - an odd inventory shape must never block the pipeline
        logger.warning("plan check could not run; the plan is used as planned", exc_info=True)
        return []


def _verify(storyboard: Storyboard, app_info: dict, repo_routes: Sequence[str]) -> List[Issue]:
    app_url = app_info.get("url") or ""
    routes, selectors, inputs = _known(app_info, repo_routes)
    issues: List[Issue] = []
    for number, scene in enumerate(storyboard.scenes, start=1):
        route = (scene.route or "/").strip() or "/"
        if is_risky(url=route):
            issues.append(Issue(number, "risky", f"scene {number} starts on {route}, a sign-out or destructive page"))
        elif not _route_known(route, app_url, routes):
            issues.append(Issue(number, "route", f"scene {number} starts on {route}, which is not a known route"))
        for index, action in enumerate(scene.actions):
            issue = _check_action(number, index, action, app_url, routes, selectors, inputs)
            if issue:
                issues.append(issue)
    return issues


def _check_action(number: int, index: int, action: Action, app_url: str, routes: Set[str], selectors: Set[str],
                  inputs: Set[str]) -> Optional[Issue]:
    where = f"scene {number}, action {index + 1}"
    if action.type == "navigate":
        target = (action.value or "").strip()
        if is_risky(url=target):
            return Issue(number, "risky", f"{where} navigates to {target}, a sign-out or destructive page", index)
        if not _route_known(target, app_url, routes):
            return Issue(number, "route", f"{where} navigates to {target}, which is not a known route", index)
        return None
    if action.type not in ("click", "type"):
        return None
    selector = (action.selector or "").strip()
    if is_risky(label=selector, url=selector):
        return Issue(number, "risky", f"{where} {'clicks' if action.type == 'click' else 'types into'} {selector}, "
                                      "a sign-out or destructive control", index)
    if action.type == "type":
        if _canon(selector) in inputs:
            return None
        return Issue(number, "selector", f"{where} types into {selector}, which is not an inventory input", index)
    if _canon(selector) in selectors:
        return None
    href = _link_href(selector)
    if href:
        if _route_known(href, app_url, routes):
            return None  # not necessarily on this page, but the recorder navigates to a known route directly
        return Issue(number, "selector", f"{where} clicks {selector}, a link outside the app's known pages", index)
    return Issue(number, "selector", f"{where} clicks {selector}, which is not in the inventory", index)


def notes(storyboard: Storyboard, target_seconds: Optional[int] = None, voice_style: str = "energetic_tech",
          ai_clips: bool = False) -> List[str]:
    """Observations that are logged but never repaired: narration far from the per-scene word budget and runs of
    consecutive scenes on the same route (Gemini misses both routinely without the video suffering)."""
    try:
        return _notes(storyboard, target_seconds, voice_style, ai_clips)
    except Exception:  # noqa: BLE001
        logger.warning("plan notes skipped", exc_info=True)
        return []


def _notes(storyboard: Storyboard, target_seconds: Optional[int], voice_style: str, ai_clips: bool) -> List[str]:
    scenes = storyboard.scenes
    out: List[str] = []
    if target_seconds and scenes:
        narrated = target_seconds - TAIL_ALLOWANCE_SEC - (BOOKEND_ALLOWANCE_SEC if ai_clips else 0)
        budget = max(30, int(narrated * WORDS_PER_SECOND.get(voice_style, 2.0))) / len(scenes)
        for number, scene in enumerate(scenes, start=1):
            words = len(scene.narration.split())
            if abs(words - budget) > WORD_TOLERANCE * budget:
                out.append(f"scene {number} narration is {words} words against a budget of about {budget:.0f} "
                           f"({'long' if words > budget else 'short'}; the voice timing decides, not repaired)")
    run: List[int] = []
    for number, scene in enumerate(scenes, start=1):
        if run and _norm_path(scenes[run[-1] - 1].route) == _norm_path(scene.route):
            run.append(number)
            continue
        if len(run) > 1:
            out.append(_same_route_note(run, scenes[run[0] - 1].route))
        run = [number]
    if len(run) > 1:
        out.append(_same_route_note(run, scenes[run[0] - 1].route))
    return out


def _same_route_note(numbers: List[int], route: str) -> str:
    return (f"scenes {_join(numbers)} all start on {route} (the planner was asked for a different page per scene; "
            "not repaired)")


def _join(numbers: Sequence[int]) -> str:
    items = [str(n) for n in numbers]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def summarize(issues: Sequence[Issue]) -> str:
    """'2 issue(s): scene 3, action 2 clicks ..., which is not in the inventory; scene 5 starts on ...'."""
    return f"{len(issues)} issue{'' if len(issues) == 1 else 's'}: " + "; ".join(issue.text for issue in issues)


# --------------------------------------------------------------------------- repair (one Gemini call)


def _scene_json(scene: Scene) -> Dict[str, Any]:
    return {"id": scene.id, "title": scene.title, "route": scene.route, "narration": scene.narration,
            "actions": [a.model_dump(exclude_none=True) for a in scene.actions]}


def repair_prompt(storyboard: Storyboard, issues: Sequence[Issue], app_info: dict, repo_routes: Sequence[str] = ()) -> str:
    routes, _, _ = _known(app_info, repo_routes)
    broken = sorted({issue.scene for issue in issues})
    ids = ", ".join(storyboard.scenes[n - 1].id for n in broken if 0 < n <= len(storyboard.scenes))
    problems = "\n".join(f"{i + 1}) {issue.text}" for i, issue in enumerate(issues))
    site_map = _site_map_lines(app_info)
    return f"""The storyboard below, for a demo video of {storyboard.project_name}, failed these checks against the live app's
verified inventory:
{problems}
Fix ONLY the listed problems and keep everything else identical: the same scenes in the same order, with the same
titles and narration (they are not part of your answer). Use only selectors and routes from the inventory and site
map below; when nothing in the inventory fits, use a scroll or wait instead of guessing.

INTERACTIVE ELEMENTS ON THE LIVE APP (selectors are verified, use them verbatim; "(on /x)" means the link appears on that page)
{_inventory_lines(app_info)}

SITE MAP (each route is reachable; "shows" lists the page's headings)
{site_map or "- (only the landing page was reachable)"}
Known routes: {", ".join(sorted(routes))}

CURRENT STORYBOARD
{json.dumps([_scene_json(s) for s in storyboard.scenes], ensure_ascii=False)}

Actions: {{"type": "navigate", "value": "/path"}} | {{"type": "click", "selector": "<selector from the list>"}} |
{{"type": "type", "selector": "<input selector>", "value": "text"}} | {{"type": "scroll", "value": "down" | "up" | "top" | "bottom"}} |
{{"type": "wait"}}; each may set "dwell_ms" (500-4000). "route" is the path where the scene starts.
Respond with JSON only, containing only the scenes you changed ({ids}), each with its id and its full corrected
"route" and "actions" list:
{{"scenes": [{{"id": "scene-3", "route": "/", "actions": [{{"type": "scroll", "value": "down", "dwell_ms": 1500}}]}}]}}
"""


async def repair(storyboard: Storyboard, issues: Sequence[Issue], app_info: dict, log, repo_routes: Sequence[str] = ()
                 ) -> Optional[Storyboard]:
    """One Gemini call (JSON, no thinking, REPAIR_TIMEOUT_SEC) that revises only the scenes with issues.

    Returns a new Storyboard in which only those scenes' `route` and `actions` may differ from the original (every
    other field, including narration, title and why, is copied), or None when Gemini returned nothing usable or
    changed nothing. A failing or slow call raises, so the orchestrator's optional-step wrapper logs it and marks
    the step; the deterministic `drop_invalid` then has the last word either way.
    """
    if not issues:
        return None
    broken = sorted({issue.scene for issue in issues if 0 < issue.scene <= len(storyboard.scenes)})
    started = time.monotonic()
    data = await generate_json(repair_prompt(storyboard, issues, app_info, repo_routes),
                               timeout_sec=REPAIR_TIMEOUT_SEC, purpose="repair plan")
    elapsed = time.monotonic() - started
    repaired, revised = merge_repair(storyboard, data, broken)
    if not revised:
        log(f"Plan check: Gemini returned the plan unchanged ({elapsed:.1f}s); dropping the invalid actions instead")
        return None
    log(f"Plan check: Gemini revised scene{'' if len(revised) == 1 else 's'} {_join(revised)} ({elapsed:.1f}s); "
        "narration unchanged")
    return repaired


def merge_repair(original: Storyboard, data: Any, broken: Sequence[int]) -> Tuple[Storyboard, List[int]]:
    """Applies Gemini's answer to the `broken` scenes only (matched by id, else by position) and returns
    (the new storyboard, the 1-based numbers of the scenes whose route or actions changed)."""
    if isinstance(data, dict) and "scenes" not in data:  # unwrap {"storyboard": {...}}
        data = next((v for v in data.values() if isinstance(v, dict) and "scenes" in v), data)
    raw_scenes = data.get("scenes") if isinstance(data, dict) else data
    if not isinstance(raw_scenes, list):
        raise ValueError("repair answer has no scenes list")
    entries = [raw for raw in raw_scenes if isinstance(raw, dict)]
    repaired = original.model_copy(deep=True)
    total = len(original.scenes)
    by_id = {scene.id: n for n, scene in enumerate(original.scenes, start=1)}
    revised: List[int] = []
    for position, raw in enumerate(entries):
        number = by_id.get(str(raw.get("id") or "").strip())
        if number is None and isinstance(raw.get("scene"), int):
            number = raw["scene"]
        if number is None and len(entries) == total:
            number = position + 1  # the full storyboard came back, in order
        if number is None and len(entries) == len(broken):
            number = broken[position]  # only the broken scenes came back, in order
        if number not in broken:
            continue  # unlisted scenes keep their plan whatever the answer says
        scene = repaired.scenes[number - 1]
        if raw.get("route") not in (None, ""):
            scene.route = str(raw["route"]).strip() or scene.route
        if "actions" in raw and isinstance(raw["actions"], list):
            scene.actions = [a for a in (_sanitize_action(x) for x in raw["actions"]) if a][:MAX_ACTIONS]
        before = original.scenes[number - 1]
        if (scene.route != before.route or scene.actions != before.actions) and number not in revised:
            revised.append(number)
    return repaired, sorted(revised)


def changed_scenes(before: Storyboard, after: Storyboard) -> List[int]:
    """1-based numbers of the scenes whose route or actions differ (for the Check plan row)."""
    return [n for n, (a, b) in enumerate(zip(before.scenes, after.scenes), start=1)
            if a.route != b.route or a.actions != b.actions]


# --------------------------------------------------------------------------- deterministic last word


def drop_invalid(storyboard: Storyboard, issues: Sequence[Issue], log=None) -> int:
    """Removes the offending click/type/navigate actions (never turned into waits, which would lose the recorder's
    a[href] rescue) and moves a scene on an unknown or risky route to the previous scene's route (or '/').
    Mutates in place; returns the number of changes (actions removed + routes rewritten). Never raises."""
    log = log or (lambda message: None)
    try:
        return _drop_invalid(storyboard, issues, log)
    except Exception:  # noqa: BLE001
        logger.warning("drop_invalid skipped", exc_info=True)
        return 0


def _drop_invalid(storyboard: Storyboard, issues: Sequence[Issue], log) -> int:
    by_scene: Dict[int, List[Issue]] = {}
    for issue in issues:
        by_scene.setdefault(issue.scene, []).append(issue)
    changes, dropped_notes, previous_route = 0, [], "/"
    for number, scene in enumerate(storyboard.scenes, start=1):
        scene_issues = by_scene.get(number, [])
        route_issue = next((i for i in scene_issues if i.action is None), None)
        if route_issue is not None:
            old = scene.route
            scene.route = previous_route
            changes += 1
            log(f"Plan check: scene {number} now starts on {previous_route} instead of {old} ({route_issue.reason})")
        bad = {i.action: i for i in scene_issues if i.action is not None and 0 <= i.action < len(scene.actions)}
        for index in sorted(bad, reverse=True):
            scene.actions.pop(index)
            changes += 1
        if bad:
            reasons = sorted({bad[i].reason for i in bad})
            dropped_notes.append(f"scene {number}: {', '.join(reasons)}")
        previous_route = scene.route
    dropped = sum(len([i for i in by_scene.get(n, []) if i.action is not None]) for n in by_scene)
    if dropped_notes:
        log(f"Plan check: {dropped} action{'' if dropped == 1 else 's'} dropped ({'; '.join(dropped_notes)})")
    return changes


def describe(verified: int, issues: Sequence[Issue], revised: Sequence[int] = (), repair_sec: Optional[float] = None,
             left: Sequence[Issue] = (), fixed: int = 0, reason: Optional[str] = None) -> str:
    """The Check plan row's detail: '11 actions verified against the inventory, 0 issues',
    '2 issues → Gemini revised scenes 3 and 5 (6.2 s) · 0 left' or '2 issues → 2 fixed deterministically (no Gemini key)'."""
    if not issues:
        return f"{verified} action{'' if verified == 1 else 's'} verified against the inventory, 0 issues"
    head = f"{len(issues)} issue{'' if len(issues) == 1 else 's'} → "
    if revised:
        head += f"Gemini revised scene{'' if len(revised) == 1 else 's'} {_join(revised)}"
        head += f" ({repair_sec:.1f} s)" if repair_sec is not None else ""
        head += f" · {len(left)} left"
        if left:
            head += f", {fixed} fixed deterministically"
        return head
    head += f"{fixed} fixed deterministically"
    return head + (f" ({reason})" if reason else "")
