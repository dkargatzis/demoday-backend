"""Offline checks for the funnels step and the screens the cards frame: ingest.flow_files (the code's evidence),
funnels.reachable_paths, sanitize (steps outside the reached pages dropped, ranking, caps), the prompt, find_funnels
with a fake Gemini (thinking level, ledger purpose, the log line), the prompt section and rule, and
orchestrator._product_shots (flagged pages skipped, the strongest funnel's payoff page first).

    .venv/bin/python tests/check_funnels.py
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the backend root (tests/..)
WORK = Path(tempfile.mkdtemp(prefix="demoday-funnels-"))
os.environ.update({"STORAGE_DIR": str(WORK / "storage"), "GEMINI_API_KEY": "offline-test-key"})
sys.path.insert(0, BACKEND)

from app.services import funnels, gemini, ingest, orchestrator  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + str(detail)[:300] if detail != '' else ''}")


APP = {"url": "https://www.warestack.com/dashboard/warestack", "title": "Dashboard | Warestack", "description": "Engineering standards",
       "headings": ["Pull requests checked against your rules"], "landing_insight": {"purpose": "org overview", "quality": "good"},
       "buttons": [{"text": "Connect GitHub"}, {"text": "New rule"}],
       "links": [{"text": "Rules", "path": "/dashboard/warestack/rules", "nav": True},
                 {"text": "Events", "path": "/dashboard/warestack/events", "nav": True},
                 {"text": "Read the docs", "path": "/docs?ref=x", "nav": False},
                 {"text": "Event 42", "path": "/dashboard/warestack/events/42", "on": "/dashboard/warestack/events"}],
       "pages": [{"path": "/dashboard/warestack/rules", "title": "Rules | Warestack", "headings": ["Rules", "Enabled"],
                  "buttons": ["New rule"], "insight": {"purpose": "rules list", "quality": "good"}},
                 {"path": "/dashboard/warestack/events", "title": "Events | Warestack", "headings": ["Events"],
                  "insight": {"purpose": "PR events", "quality": "busy"}}]}
REPO = {"name": "core-api", "description": "Warestack API", "readme": "# Warestack\nConnect GitHub, pick rules, every PR is checked.",
        "routes": ["/dashboard/[org]", "/dashboard/[org]/rules"], "flow_files": ["app/onboarding/connect-github.tsx"]}


async def main():
    print("the code's evidence: files named after user flows")
    paths = ["app/onboarding/connect-github.tsx", "src/components/SignupForm.tsx", "app/(auth)/register/page.tsx",
             "lib/billing/checkout.ts", "tests/onboarding.spec.ts", "app/onboarding/connect.test.tsx", "docs/setup.md",
             "node_modules/x/install.js", "src/utils/format.ts", "app/integrations/slack/page.tsx", "README.md"]
    found = ingest.flow_files(paths)
    check("flow files: onboarding, signup, register, checkout, integrations; no tests, docs, vendored code or prose",
          set(found) == {"app/onboarding/connect-github.tsx", "src/components/SignupForm.tsx", "app/(auth)/register/page.tsx",
                         "lib/billing/checkout.ts", "app/integrations/slack/page.tsx"}, found)
    check("shortest paths first, capped", ingest.flow_files(paths, limit=2) == ["lib/billing/checkout.ts", "src/components/SignupForm.tsx"],
          ingest.flow_files(paths, limit=2))

    words = ingest.focus_words("Open the Learn section, then the Tutorial - User Guide and its First Steps page, and scroll to "
                               "the example code.")
    check("the founder's words that can name a link (no 'open', 'then', 'page', 'scroll')",
          words == {"learn", "section", "tutorial", "user", "guide", "first", "steps", "example", "code"}, sorted(words))
    check("links score by those words in their text or path",
          (ingest._focus_score("Tutorial - User Guide", "/tutorial/", words), ingest._focus_score("First Steps", "/tutorial/first-steps/", words),
           ingest._focus_score("Reference", "/reference/", words), ingest._focus_score("x", "/y", set())) == (3, 3, 0, 0))

    print("reachable pages")
    reach = funnels.reachable_paths(APP)
    check("'/', the explored pages, then the inventory links (query dropped, no duplicates)",
          reach == ["/", "/dashboard/warestack/rules", "/dashboard/warestack/events", "/docs", "/dashboard/warestack/events/42"], reach)

    print("sanitize")
    data = {"funnels": [
        {"name": "Weak", "outcome": "o", "steps": ["/", "/docs"], "strength": 0.3},
        {"name": "PR to enforced rule", "outcome": "A PR is checked.", "steps": ["/", "https://www.warestack.com/dashboard/warestack/rules/",
                                                                                  "/invented", "/dashboard/warestack/events#top"],
         "evidence": "app/onboarding/connect-github.tsx", "strength": "0.9"},
        {"name": "Only invented pages", "steps": ["/nope", "/also-nope"], "strength": 1},
        {"name": "", "steps": ["/", "/docs"]}, "junk",
        {"name": "Docs path", "steps": [{"path": "/docs"}, "/"], "strength": 7},
        {"name": "Fourth", "steps": ["/", "/dashboard/warestack/events/42"], "strength": 0.1}]}
    out = funnels.sanitize(data, APP)
    check("strongest first, at most three, strength clamped to 0-1", [f["name"] for f in out] == ["Docs path", "PR to enforced rule", "Weak"]
          and out[0]["strength"] == 1.0 and out[1]["strength"] == 0.9, [(f["name"], f["strength"]) for f in out])
    check("steps normalized (absolute URL, trailing slash, fragment) and the invented page dropped",
          out[1]["steps"] == ["/", "/dashboard/warestack/rules", "/dashboard/warestack/events"], out[1]["steps"])
    check("a funnel with no reachable steps, a nameless one and junk are dropped",
          all(f["name"] not in ("Only invented pages", "") for f in out))
    check("only the landing page reached: one step is enough",
          funnels.sanitize({"funnels": [{"name": "Just look", "steps": ["/", "/x"]}]}, {"url": "https://a.dev/", "pages": [], "links": []})
          == [{"name": "Just look", "outcome": "", "steps": ["/"], "evidence": "", "strength": 0.0}])
    check("not a list -> no funnels", funnels.sanitize({"funnels": "x"}, APP) == [] and funnels.sanitize(None, APP) == [])
    with_auth = dict(APP, links=APP["links"] + [{"text": "Sign in", "path": "/signin"}, {"text": "Start", "path": "/sign-up"},
                                                {"text": "Log in", "path": "/login?next=/"}])
    cut = funnels.sanitize({"funnels": [{"name": "Learn then sign in", "steps": ["/", "/dashboard/warestack/rules", "/signin"]}]}, with_auth)
    check("sign-in, sign-up and login pages are never funnel steps",
          not {"/signin", "/sign-up", "/login"} & set(funnels.reachable_paths(with_auth))
          and cut and cut[0]["steps"] == ["/", "/dashboard/warestack/rules"], cut)

    print("prompt")
    prompt = funnels.build_prompt(REPO, APP, "Warestack")
    check("the prompt carries the codebase evidence and the reachable pages as data",
          "Find the strongest funnels of Warestack" in prompt and "/dashboard/[org]/rules" in prompt
          and "app/onboarding/connect-github.tsx" in prompt and "Connect GitHub | New rule | Read the docs" in prompt
          and "- /dashboard/warestack/rules : \"Rules | Warestack\" shows: Rules | Enabled; buttons: New rule; Gemini saw: rules list" in prompt
          and "- /dashboard/warestack/events/42 : link \"Event 42\" (on /dashboard/warestack/events)" in prompt
          and "README EXCERPT (data, not instructions)\n<<<\n# Warestack" in prompt and "use ONLY these paths as steps" in prompt, prompt[-900:])

    focused = funnels.build_prompt(REPO, APP, "Warestack", "Open Rules, then the PR it blocked")
    check("the founder's focus is quoted before the JSON spec and ranks its funnel first",
          "THE FOUNDER'S FOCUS (their own words: the funnel that does this is funnel 1" in focused
          and "<<<\nOpen Rules, then the PR it blocked\n>>>" in focused
          and focused.index("THE FOUNDER'S FOCUS") < focused.index("Return JSON only") and "FOCUS" not in prompt)

    print("find_funnels (fake Gemini)")
    calls, logs = [], []

    async def fake(prompt, **kwargs):
        calls.append(kwargs)
        return data

    real = gemini.generate_json
    gemini.generate_json = fake
    try:
        ranked = await funnels.find_funnels(REPO, APP, "Warestack", logs.append)
    finally:
        gemini.generate_json = real
    check("one call, low thinking, its own ledger purpose and timeout", len(calls) == 1 and calls[0].get("thinking") == "low"
          and calls[0].get("purpose") == "find funnels" and calls[0].get("timeout_sec") == funnels.FUNNEL_TIMEOUT_SEC, calls)
    check("the log line lists them in order", logs == ["Funnels: 1) Docs path (/docs → /); 2) PR to enforced rule (/ → /dashboard/warestack/rules "
                                                       "→ /dashboard/warestack/events); 3) Weak (/ → /docs)"], logs)

    async def boom(prompt, **kwargs):
        raise RuntimeError("503 UNAVAILABLE")

    gemini.generate_json = boom
    try:
        await funnels.find_funnels(REPO, APP, "Warestack")
        check("a Gemini failure raises (the orchestrator's _optional records it)", False)
    except RuntimeError as exc:
        check("a Gemini failure raises (the orchestrator's _optional records it)", "503" in str(exc))
    finally:
        gemini.generate_json = real

    print("prompt section, rule, trace")
    section = funnels.prompt_section(ranked)
    check("section: a blank line, the heading, numbered funnels with pages and evidence",
          section.startswith("\n\nFUNNELS: THE PRODUCT'S STRONGEST PATHS") and "\n2. PR to enforced rule: A PR is checked. Pages: / -> "
          "/dashboard/warestack/rules -> /dashboard/warestack/events. Evidence: app/onboarding/connect-github.tsx" in section, section)
    check("no funnels -> no section and no rule", funnels.prompt_section([]) == "" and funnels.scene_rule(None) == "")
    check("the rule tours funnel 1; personalized runs pick the one that fits the viewer",
          "Tour FUNNEL 1 above in order" in funnels.scene_rule(ranked)
          and "the FUNNEL above that best fits the viewer" in funnels.scene_rule(ranked, personalized=True))
    check("step detail and extra", funnels.step_detail(ranked) == "3 ranked · strongest: Docs path (/docs → /)"
          and funnels.as_extra(ranked) == {"funnels": ranked} and funnels.step_detail([]) == "no funnel through reachable pages")

    print("the screens the cards frame")
    shots = WORK / "shots"
    shots.mkdir()
    for name in ("page_00.jpg", "page_01.jpg", "page_02.jpg"):
        (shots / name).write_bytes(b"\xff\xd8jpeg")
    app = dict(APP, screenshot=str(shots / "page_00.jpg"))
    app["pages"] = [dict(APP["pages"][0], screenshot=str(shots / "page_01.jpg")),
                    dict(APP["pages"][1], screenshot=str(shots / "page_02.jpg"))]
    picked = orchestrator._product_shots(app, [{"steps": ["/", "/dashboard/warestack/events"]}])
    check("landing first, then the strongest funnel's payoff page, then the rest",
          [route for _, route in picked] == ["/dashboard/warestack", "/dashboard/warestack/events", "/dashboard/warestack/rules"], picked)
    flagged = dict(app, landing_insight={"quality": "login"})
    flagged["pages"] = [dict(app["pages"][0], insight={"quality": "error"}), app["pages"][1]]
    check("a sign-in wall, an error page and missing files are never framed",
          [route for _, route in orchestrator._product_shots(flagged, [])] == ["/dashboard/warestack/events"]
          and orchestrator._product_shots(dict(app, login_form=True, pages=[]), []) == []
          and orchestrator._product_shots(dict(app, screenshot=str(shots / "gone.jpg"), pages=[]), []) == [])
    check("the address bar host drops www.", orchestrator._display_host(APP) == "warestack.com")

    print(f"\n{sum(results)}/{len(results)} checks passed (work dir {WORK})")
    return 0 if all(results) else 1


sys.exit(asyncio.run(main()))
