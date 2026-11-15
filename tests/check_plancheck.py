"""Offline checks for the pre-camera plan check (plancheck.py) and the pacing fit (recorder.fit_dwells).

Run: .venv/bin/python tests/check_plancheck.py                    (offline, fake Gemini)
     LIVE=1 .venv/bin/python tests/check_plancheck.py             (adds ONE real Gemini repair call, prints its latency)
"""
import asyncio
import copy
import os
import sys
import tempfile
import time

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the backend root (tests/..)
os.environ.setdefault("STORAGE_DIR", tempfile.mkdtemp(prefix="demoday-plancheck-"))
sys.path.insert(0, BACKEND)

from app.models.schemas import Action, Scene, Storyboard  # noqa: E402
from app.services import plancheck, recorder  # noqa: E402
from app.services.recorder import fit_dwells  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + str(detail)[:300] if detail != '' else ''}")


# --------------------------------------------------------------------------- fixtures (the FastAPI preset's inventory shape)

APP = {
    "url": "https://fastapi.tiangolo.com/", "status": 200, "title": "FastAPI - FastAPI",
    "description": "FastAPI framework, high performance, easy to learn", "headings": ["FastAPI", "Sponsors"],
    "login_form": False,
    "links": [
        {"text": "Features", "path": "/features/", "selector": 'a[href="features/"]', "nav": True},
        {"text": "Learn", "path": "/learn/", "selector": 'a[href="learn/"]', "nav": True},
        {"text": "Sponsors", "path": "/", "selector": 'a[href="#sponsors"]', "nav": False},
        {"text": "Benchmarks", "path": "/benchmarks/", "selector": 'a[href="../benchmarks/"]', "nav": False, "on": "/features/"},
    ],
    "buttons": [{"text": "Netflix", "selector": '[id="fo-tab-netflix"]'},
                {"text": "Search", "selector": 'role=button[name="Search"]'}],
    "inputs": [{"label": "Search", "selector": 'input[name="query"]'}],
    "pages": [
        {"path": "/features/", "url": "https://fastapi.tiangolo.com/features/", "title": "Features - FastAPI",
         "link_text": "Features", "open_selector": 'a[href="features/"]',
         "headings": ["Features", "FastAPI features", "Pydantic features"], "buttons": [], "inputs": []},
        {"path": "/learn/", "url": "https://fastapi.tiangolo.com/learn/", "title": "Learn - FastAPI",
         "link_text": "Learn", "open_selector": 'a[href="learn/"]', "headings": ["Learn"], "buttons": [], "inputs": []},
    ],
}
REPO_ROUTES = ["/", "/docs"]


def act(kind, selector=None, value=None, dwell=1500):
    return Action(type=kind, selector=selector, value=value, dwell_ms=dwell)


def storyboard():
    """Scene 1 clean; scene 2 a[href] rescue + unknown button; scene 3 unknown route + unknown input;
    scene 4 a sign-out link + a destructive navigate; scene 5 a dwell plan far longer than any narration."""
    return Storyboard(project_name="FastAPI", tagline="APIs at the speed of thought", scenes=[
        Scene(id="scene-1", title="The Challenge", route="/", narration="Building APIs in Python used to mean boilerplate.",
              why="hook", actions=[act("scroll", value="down"), act("click", '[id="fo-tab-netflix"]')]),
        Scene(id="scene-2", title="High Performance", route="/features/", narration="FastAPI is fast to run and fast to code.",
              why="features page", actions=[act("click", 'a[href="https://fastapi.tiangolo.com/benchmarks/"]'),
                                            act("click", 'role=button[name="Try it"]')]),
        Scene(id="scene-3", title="Search", route="/pricing/enterprise", narration="Search the docs for anything.",
              actions=[act("type", 'input[name="query"]', "FastAPI\n"), act("type", 'input[name="email"]', "me@x.dev")]),
        Scene(id="scene-4", title="Learn", route="/learn/", narration="The learn section walks you through it.",
              actions=[act("click", 'a[href="/logout"]'), act("navigate", value="/account/delete"), act("scroll", value="down")]),
        Scene(id="scene-5", title="Call to Action", route="/", narration="Try it today.",
              actions=[act("wait", dwell=4000), act("wait", dwell=4000), act("wait", dwell=4000)]),
    ])


def issue_keys(issues):
    return sorted((i.scene, i.kind, i.action) for i in issues)


# --------------------------------------------------------------------------- a stand-in for _Director.run's deadline logic

# Measured on the e2e logs: what each action really costs on camera before its dwell (independent of fit_dwells' model).
REAL_COST = {"button": 0.75, "link_on_screen": 1.3, "link_off_screen": 2.7, "scroll": 0.05, "wait": 0.0, "navigate": 2.4}


def simulate_skips(scene, clip, off_screen_links=()):
    """Replays run(): an action starts only while more than 0.25 s remain; dwells are clamped to the deadline."""
    t, skipped = 0.0, 0
    for position, action in enumerate(scene.actions):
        if t >= clip - 0.25:
            return len(scene.actions) - position
        if action.type == "click":
            href = recorder._link_href(action.selector)
            kind = "button" if not href else ("link_off_screen" if href in off_screen_links else "link_on_screen")
        else:
            kind = action.type
        t += REAL_COST[kind]
        t += max(0.0, min(action.dwell_ms / 1000, clip - t))
    return skipped


# The scene plans of a real job (faaa21276443): Gemini's dwells vs the measured Kokoro clips.
E2E_SCENES = [
    ("/", [("scroll", None, "down", 1800), ("click", '[id="fo-tab-netflix"]', None, 1800)]),
    ("/", [("click", 'a[href="https://fastapi.tiangolo.com/features/"]', None, 1800), ("scroll", None, "down", 1800)]),
    ("/features/", [("click", 'a[href="https://fastapi.tiangolo.com/benchmarks/"]', None, 1500), ("scroll", None, "down", 1300)]),
    ("/benchmarks/", [("click", 'a[href="https://fastapi.tiangolo.com/learn/"]', None, 1600), ("scroll", None, "down", 1600)]),
    ("/learn/", [("click", 'a[href="https://fastapi.tiangolo.com/#sponsors"]', None, 1200), ("wait", None, None, 1200)]),
]
E2E_CLIPS = [4.0, 5.23, 3.0, 4.9, 3.37]
E2E_OFF_SCREEN = {"https://fastapi.tiangolo.com/benchmarks/", "https://fastapi.tiangolo.com/#sponsors"}  # per the log


def e2e_storyboard():
    return [Scene(id=f"scene-{i + 1}", title=f"Scene {i + 1}", route=route, narration="x",
                  actions=[Action(type=k, selector=s, value=v, dwell_ms=d) for k, s, v, d in actions])
            for i, (route, actions) in enumerate(E2E_SCENES)]


async def main():
    print("verify")
    sb = storyboard()
    issues = plancheck.verify(sb, APP, REPO_ROUTES)
    expected = sorted([(2, "selector", 1), (3, "route", None), (3, "selector", 1), (4, "risky", 0), (4, "risky", 1)])
    check("finds exactly the bad selector, the unknown route, the unknown input and the two risky actions",
          issue_keys(issues) == expected, issue_keys(issues))
    check("an a[href] link to a known route that is not on the scene's page is NOT an issue (the recorder opens it directly)",
          not any(i.scene == 2 and i.action == 0 for i in issues))
    check("the too-long dwell plan is not a plan-check issue (that is fit_dwells' job)", not any(i.scene == 5 for i in issues))
    check("issue texts read as sentences", all(i.text.startswith("scene ") for i in issues), [i.text for i in issues][:2])
    check("a clean storyboard has no issues",
          plancheck.verify(Storyboard(project_name="x", tagline="y", scenes=[sb.scenes[0]]), APP, REPO_ROUTES) == [])
    external = Scene(id="scene-1", title="t", route="/", narration="n", actions=[act("click", 'a[href="https://twitter.com/fastapi"]')])
    found = plancheck.verify(Storyboard(project_name="x", tagline="y", scenes=[external]), APP, REPO_ROUTES)
    check("a link to another site is a selector issue (the recorder would leave the app)",
          issue_keys(found) == [(1, "selector", 0)] and "outside" in found[0].text, [i.text for i in found])
    nav = Scene(id="scene-1", title="t", route="/docs", narration="n", actions=[act("navigate", value="/nowhere"),
                                                                                act("navigate", value="/learn")])
    found = plancheck.verify(Storyboard(project_name="x", tagline="y", scenes=[nav]), APP, REPO_ROUTES)
    check("a repo route is known, a navigate to an unknown route is an issue, a known one (slash-insensitive) is not",
          issue_keys(found) == [(1, "route", 0)], issue_keys(found))
    quotes = Scene(id="scene-1", title="t", route="/", narration="n", actions=[act("click", "role=button[name='Search']")])
    check("selector matching ignores quote style", plancheck.verify(Storyboard(project_name="x", tagline="y", scenes=[quotes]), APP) == [])
    check("verify never raises on a broken inventory", plancheck.verify(sb, {"links": "oops", "pages": None}) == [] or True)

    print("notes (logged only)")
    long_sb = storyboard()
    long_sb.scenes[0].narration = " ".join(["word"] * 80)
    long_sb.scenes[1].route = "/"  # scenes 1 and 2 then share "/"
    lines = plancheck.notes(long_sb, 60, "energetic_tech")
    check("an 80-word scene against a ~23-word budget is noted", any("scene 1 narration is 80 words" in n for n in lines), lines)
    check("consecutive scenes on the same route are noted once", sum("all start on /" in n for n in lines) == 1
          and any("scenes 1 and 2 all start on /" in n for n in lines), lines)
    check("notes are not issues", issue_keys(plancheck.verify(long_sb, APP, REPO_ROUTES)) == sorted(
        [(2, "selector", 1), (3, "route", None), (3, "selector", 1), (4, "risky", 0), (4, "risky", 1)]))
    check("summarize reads as one line", plancheck.summarize(issues).startswith("5 issues: scene 2, action 2 clicks"),
          plancheck.summarize(issues))

    print("drop_invalid")
    sb = storyboard()
    logs = []
    changes = plancheck.drop_invalid(sb, plancheck.verify(sb, APP, REPO_ROUTES), logs.append)
    check("returns actions removed + routes rewritten", changes == 5, changes)
    check("scene 2 keeps its navigation link and loses only the unknown button",
          [a.selector for a in sb.scenes[1].actions] == ['a[href="https://fastapi.tiangolo.com/benchmarks/"]'])
    check("scene 3 moves to the previous scene's route and keeps the inventory input",
          sb.scenes[2].route == "/features/" and [a.selector for a in sb.scenes[2].actions] == ['input[name="query"]'],
          (sb.scenes[2].route, [a.selector for a in sb.scenes[2].actions]))
    check("scene 4 loses both risky actions and keeps the scroll", [a.type for a in sb.scenes[3].actions] == ["scroll"])
    check("clean scenes are untouched", len(sb.scenes[0].actions) == 2 and len(sb.scenes[4].actions) == 3)
    check("narration never changes", [s.narration for s in sb.scenes] == [s.narration for s in storyboard().scenes])
    check("re-verify is clean", plancheck.verify(sb, APP, REPO_ROUTES) == [])
    check("one line per route move and one for the dropped actions",
          len(logs) == 2 and "scene 3 now starts on /features/" in logs[0] and logs[1].startswith("Plan check: 4 actions dropped"), logs)
    first = Storyboard(project_name="x", tagline="y", scenes=[Scene(id="scene-1", title="t", route="/nope", narration="n")])
    plancheck.drop_invalid(first, plancheck.verify(first, APP))
    check("an unknown route on scene 1 becomes /", first.scenes[0].route == "/")

    print("fit_dwells")
    logs = []
    fits = Scene(id="s", title="t", route="/", narration="n", actions=[act("scroll", value="down"), act("click", '[id="x"]')])
    check("a plan that fits is untouched", fit_dwells([fits], [6.0], logs.append) == 0
          and [a.dwell_ms for a in fits.actions] == [1500, 1500] and not logs)
    trimmed = copy.deepcopy(fits)
    n = fit_dwells([trimmed], [4.0], logs.append)
    check("dwells over the clip are scaled down (3.0 s planned, 2.3 s available)",
          n == 1 and [a.dwell_ms for a in trimmed.actions] == [1150, 1150], [a.dwell_ms for a in trimmed.actions])
    check("...with one log line", len(logs) == 1 and "trimmed 3.0→2.3s" in logs[0], logs)
    logs.clear()
    drop = Scene(id="s", title="t", route="/features/", narration="n", actions=[
        act("click", 'a[href="https://fastapi.tiangolo.com/benchmarks/"]'), act("scroll", value="down", dwell=1300)])
    n = fit_dwells([drop], [3.0], logs.append)
    check("a 3 s clip cannot hold a page change plus a scroll: the trailing action is dropped, the click keeps a readable dwell",
          n == 1 and len(drop.actions) == 1 and drop.actions[0].type == "click" and drop.actions[0].dwell_ms == 500,
          (len(drop.actions), [a.dwell_ms for a in drop.actions]))
    check("...the log says so", "1 trailing action dropped" in logs[0], logs)
    logs.clear()
    spread = Scene(id="s", title="t", route="/", narration="n", actions=[act("wait", dwell=1000), act("scroll", value="down", dwell=1000)])
    n = fit_dwells([spread], [8.0], logs.append)
    check("a plan that fills under 60 % of the clip is spread to 90 % of it",
          n == 1 and [a.dwell_ms for a in spread.actions] == [3240, 3240] and "spread 2.0→6.5s" in logs[0],
          ([a.dwell_ms for a in spread.actions], logs))
    floor = Scene(id="s", title="t", route="/", narration="n", actions=[act("click", '[id="x"]', dwell=2000), act("wait", dwell=0)])
    fit_dwells([floor], [1.5])
    check("a trimmed click never drops under the zoom's minimum hold (MIN_HOLD_SEC)",
          [a.dwell_ms for a in floor.actions] == [350], [a.dwell_ms for a in floor.actions])
    zero = Scene(id="s", title="t", route="/", narration="n", actions=[act("wait", dwell=0), act("click", '[id="x"]', dwell=2000)])
    fit_dwells([zero], [3.0])
    check("a zero dwell stays zero under a trim (available 1.6 s for 2.0 s planned)",
          [a.dwell_ms for a in zero.actions] == [0, 1600], [a.dwell_ms for a in zero.actions])
    cap = Scene(id="s", title="t", route="/", narration="n", actions=[act("wait", dwell=1000)])
    fit_dwells([cap], [40.0])
    check("a spread dwell never exceeds the schema maximum", cap.actions[0].dwell_ms == 15000, cap.actions[0].dwell_ms)
    typed = Scene(id="s", title="t", route="/", narration="n", actions=[act("type", 'input[name="query"]', "FastAPI tutorial\n", 2000)])
    cost, path = recorder.action_cost_sec(typed.actions[0], "/")
    check("typing costs the recorder's per-character delay plus the submit wait", abs(cost - (0.6 + 16 * 0.08 + 1.0)) < 1e-6, cost)
    robust = [Scene(id="s", title="t", route="/", narration="n", actions=[act("wait")]),
              Scene(id="s", title="t", route="/", narration="n", actions=[]), fits]
    try:
        n = fit_dwells(robust, [None, 5.0, -1.0, 9.9], None)
        check("never raises (bad clip values, no actions, missing log) and counts only adjusted scenes", n == 0, n)
    except Exception as exc:  # noqa: BLE001
        check("never raises", False, repr(exc))

    print("fit_dwells replay of a real job (faaa21276443)")
    scenes = e2e_storyboard()
    before = {i + 1: simulate_skips(s, c, E2E_OFF_SCREEN) for i, (s, c) in enumerate(zip(scenes, E2E_CLIPS))}
    check("as planned, the recorder skips one action in scenes 3 and 5 (the two 'remaining action(s) skipped' lines)",
          {k: v for k, v in before.items() if v} == {3: 1, 5: 1}, before)
    logs = []
    retimed = fit_dwells(scenes, E2E_CLIPS, logs.append)
    after = {i + 1: simulate_skips(s, c, E2E_OFF_SCREEN) for i, (s, c) in enumerate(zip(scenes, E2E_CLIPS))}
    check("after fit_dwells no action is skipped", not any(after.values()), after)
    check("every retimed scene gets a log line", retimed == len(logs) and retimed >= 2, (retimed, len(logs)))
    for line in logs:
        print("      " + line)

    print("repair (fake Gemini)")
    captured = {}

    async def fake_generate_json(prompt, **kwargs):
        captured["prompt"], captured["kwargs"] = prompt, kwargs
        return captured["answer"]

    real_generate_json = plancheck.generate_json
    plancheck.generate_json = fake_generate_json
    try:
        sb = storyboard()
        issues = plancheck.verify(sb, APP, REPO_ROUTES)
        captured["answer"] = {"scenes": [
            {"id": "scene-2", "title": "CHANGED", "narration": "CHANGED CLAIM", "route": "/features/",
             "actions": [{"type": "click", "selector": 'a[href="https://fastapi.tiangolo.com/benchmarks/"]', "dwell_ms": 2000},
                         {"type": "scroll", "value": "down", "dwell_ms": 1500}]},
            {"id": "scene-3", "route": "/learn/", "actions": [{"type": "type", "selector": 'input[name="query"]', "value": "FastAPI\n"}]},
            {"id": "scene-4", "route": "/learn/", "actions": [{"type": "scroll", "value": "down", "dwell_ms": 1500}]},
            {"id": "scene-1", "route": "/docs", "actions": []},  # not listed as broken: must be ignored
            {"id": "scene-9", "route": "/", "actions": []},      # unknown id: ignored
        ]}
        logs = []
        repaired = await plancheck.repair(sb, issues, APP, logs.append, REPO_ROUTES)
        check("the prompt lists only the problems, with the inventory and site map blocks",
              "1) scene 2, action 2 clicks role=button[name=\"Try it\"]" in captured["prompt"]
              and "INTERACTIVE ELEMENTS" in captured["prompt"] and 'a[href="../benchmarks/"]' in captured["prompt"]
              and "route /features/" in captured["prompt"] and "(scene-2, scene-3, scene-4)" in captured["prompt"])
        check("one JSON call, 15 s, ledger purpose 'repair plan'",
              captured["kwargs"] == {"timeout_sec": 15, "purpose": "repair plan"}, captured["kwargs"])
        check("a storyboard comes back with the same scene count", repaired is not None and len(repaired.scenes) == 5)
        check("narration, title and why are the originals even though Gemini changed them",
              repaired.scenes[1].narration == sb.scenes[1].narration and repaired.scenes[1].title == "High Performance"
              and repaired.scenes[1].why == "features page")
        check("only the broken scenes changed: scene 1 keeps its route, scene 2 has the new actions",
              repaired.scenes[0].route == "/" and len(repaired.scenes[0].actions) == 2
              and [a.type for a in repaired.scenes[1].actions] == ["click", "scroll"] and repaired.scenes[2].route == "/learn/")
        check("the original is untouched (the orchestrator swaps objects)", len(sb.scenes[1].actions) == 2 and sb.scenes[2].route == "/pricing/enterprise")
        check("re-verify is clean", plancheck.verify(repaired, APP, REPO_ROUTES) == [])
        check("changed_scenes reports 2, 3 and 4", plancheck.changed_scenes(sb, repaired) == [2, 3, 4])
        check("the log line names the revised scenes and says the narration is unchanged",
              len(logs) == 1 and logs[0].startswith("Plan check: Gemini revised scenes 2, 3 and 4 (") and "narration unchanged" in logs[0], logs)
        check("describe after a clean repair", plancheck.describe(11, issues, revised=[2, 3, 4], repair_sec=6.2, left=[])
              == "5 issues → Gemini revised scenes 2, 3 and 4 (6.2 s) · 0 left")

        full = [{"title": "x", "narration": "y", "route": s.route, "actions": [a.model_dump() for a in s.actions]} for s in sb.scenes]
        full[1]["actions"] = [{"type": "scroll", "value": "down"}]
        full[2]["route"] = "/"
        full[3]["actions"] = []
        full[0]["route"] = "/nowhere"  # scene 1 is not broken: ignored even in a full answer
        captured["answer"] = {"storyboard": {"scenes": full}}
        repaired = await plancheck.repair(sb, issues, APP, logs.append, REPO_ROUTES)
        check("a full storyboard answer without ids is matched by position and only the broken scenes are merged",
              repaired is not None and repaired.scenes[0].route == "/" and [a.type for a in repaired.scenes[1].actions] == ["scroll"]
              and repaired.scenes[2].route == "/" and repaired.scenes[3].actions == [])

        captured["answer"] = {"scenes": [{"id": "scene-2", "route": "/features/", "actions": [a.model_dump() for a in sb.scenes[1].actions]}]}
        logs.clear()
        check("an unchanged answer returns None with one log line",
              await plancheck.repair(sb, issues, APP, logs.append, REPO_ROUTES) is None and len(logs) == 1 and "unchanged" in logs[0], logs)
        captured["answer"] = "garbage"
        try:
            await plancheck.repair(sb, issues, APP, logs.append, REPO_ROUTES)
            check("an unusable shape raises (the orchestrator's _optional logs it and drop_invalid takes over)", False)
        except ValueError as exc:
            check("an unusable shape raises (the orchestrator's _optional logs it and drop_invalid takes over)", True, exc)
        check("no issues: no call", await plancheck.repair(sb, [], APP, logs.append) is None)
        sb_count = len(sb.scenes)
        captured["answer"] = {"scenes": [{"id": "scene-3", "route": "/learn/", "actions": [{"type": "wait"}]} for _ in range(3)]}
        repaired = await plancheck.repair(sb, issues, APP, logs.append, REPO_ROUTES)
        check("the scene count can never change", repaired is not None and len(repaired.scenes) == sb_count)
    finally:
        plancheck.generate_json = real_generate_json

    print("describe")
    two = issues[:2]
    check("no issues", plancheck.describe(11, []) == "11 actions verified against the inventory, 0 issues")
    check("no Gemini", plancheck.describe(11, two, fixed=2, reason="no Gemini key") == "2 issues → 2 fixed deterministically (no Gemini key)")
    check("partial repair", plancheck.describe(11, two, revised=[3], repair_sec=4.0, left=two[:1], fixed=1)
          == "2 issues → Gemini revised scene 3 (4.0 s) · 1 left, 1 fixed deterministically")

    if os.environ.get("LIVE") == "1":
        print("LIVE: one real Gemini repair of a deliberately broken scene")
        from app.config import GEMINI_MODEL, get_secret
        if not get_secret("GEMINI_API_KEY"):
            check("GEMINI_API_KEY configured", False)
        else:
            sb = storyboard()
            sb.scenes[2].route = "/learn/"  # keep two problems: scene 2's unknown button and scene 4's risky actions
            sb.scenes[2].actions = sb.scenes[2].actions[:1]
            issues = plancheck.verify(sb, APP, REPO_ROUTES)
            print("      issues:", plancheck.summarize(issues))
            t = time.monotonic()
            logs = []
            try:
                repaired = await asyncio.wait_for(plancheck.repair(sb, issues, APP, logs.append, REPO_ROUTES), 15)
                took = time.monotonic() - t
                for line in logs:
                    print("      " + line)
                check(f"{GEMINI_MODEL} answered within the 15 s budget", repaired is not None, f"{took:.1f}s")
                if repaired is not None:
                    left = plancheck.verify(repaired, APP, REPO_ROUTES)
                    check("narration and titles unchanged",
                          [(s.title, s.narration) for s in repaired.scenes] == [(s.title, s.narration) for s in sb.scenes])
                    check("the revised scenes pass the deterministic check", left == [], plancheck.summarize(left) if left else "")
                    for n in plancheck.changed_scenes(sb, repaired):
                        s = repaired.scenes[n - 1]
                        print(f"      scene {n}: route={s.route} actions={[(a.type, a.selector or a.value) for a in s.actions]}")
                    print(f"      LIVE repair latency: {took:.1f}s")
            except Exception as exc:  # noqa: BLE001
                check("live repair", False, f"{type(exc).__name__}: {exc} after {time.monotonic() - t:.1f}s")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


sys.exit(asyncio.run(main()))
