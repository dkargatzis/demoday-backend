"""Offline checks for frozen scripts and the dry run.

The dry run runs real Chromium against a fake app on 127.0.0.1:8779: a link that is followed, a button found but not
clicked, the step after it unverified, a missing and a covered element, a sign-out link, a page that answers 404 and
one that lands on a sign-in form. `apply` drops what would fail on camera. Frozen scripts: POST /api/pipeline/start
with `from_job` (owner only, videos only, one narration line per scene) and the script's reset.

    .venv/bin/python tests/check_rehearsal.py
"""
import asyncio
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the backend root (tests/..)
WORK = Path(tempfile.mkdtemp(prefix="demoday-rehearsal-"))
sys.path.insert(0, BACKEND)

import dotenv  # noqa: E402

dotenv.load_dotenv = lambda *args, **kwargs: False  # hermetic: no local .env
for name in ("CORS_ORIGINS", "WORK_DIR", "SIGNUP_CODE", "GEMINI_API_KEY", "FAL_KEY", "DB_BACKUP_PATH", "GCS_BUCKET", "STATIC_DIR"):
    os.environ.pop(name, None)
os.environ.update({"STORAGE_DIR": str(WORK / "storage"), "DB_PATH": str(WORK / "demoday.db"), "COOKIE_SECURE": "false",
                   "ALLOW_PRIVATE_URLS": "true"})

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import db  # noqa: E402
from app.main import app  # noqa: E402
from app.models.schemas import Action, JobStatus, LaunchKit, Scene, Storyboard  # noqa: E402
from app.services import orchestrator, rehearsal  # noqa: E402

results = []
SITE = "http://127.0.0.1:8779"


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + str(detail)[:400] if detail != '' else ''}")


def page(body: str, status: int = 200) -> HTMLResponse:
    return HTMLResponse(f"<!doctype html><html><head><title>Fake</title></head><body>{body}</body></html>", status_code=status)


site = FastAPI()


@site.get("/")
async def home():
    return page('<h1>Home</h1><a href="/rules">Rules</a><button id="enable">Enable</button>'
                '<a href="/logout">Sign out</a>')


@site.get("/rules")
async def rules():
    return page('<h1>Rules</h1><input id="search" placeholder="Search"><a href="/rules/42">Rule 42</a>'
                '<div style="position:relative"><button id="covered">Hidden</button>'
                '<div style="position:absolute;inset:0;background:#fff"></div></div>')


@site.get("/rules/42")
async def rule():
    return page("<h1>Rule 42</h1>")


@site.get("/broken")
async def broken():
    return page("<h1>Gone</h1>", status=404)


@site.get("/signin")
async def signin():
    return page('<form><input type="email"><input type="password"></form>')


threading.Thread(target=uvicorn.run, args=(site,), kwargs={"host": "127.0.0.1", "port": 8779, "log_level": "error"},
                 daemon=True).start()


def scenes():
    return [
        Scene(id="scene-1", title="Home", route="/", narration="Home.", actions=[
            Action(type="click", selector='a[href="/rules"]'),            # a link: followed
            Action(type="click", selector='a[href="/rules/42"]'),          # on /rules now: followed
        ]),
        Scene(id="scene-2", title="Home again", route="/", narration="Enable.", actions=[
            Action(type="click", selector="#enable"),                     # a button: found, not clicked
            Action(type="click", selector="#modal-save"),                 # after it: unverified
        ]),
        Scene(id="scene-3", title="Rules", route="/rules", narration="Rules.", actions=[
            Action(type="click", selector="#nope"),                       # missing: dropped
            Action(type="scroll", value="down"),
            Action(type="type", selector="#search", value="SOC 2"),       # found, never typed
            Action(type="click", selector="#covered"),                    # covered: dropped
        ]),
        Scene(id="scene-4", title="Out", route="/", narration="Out.", actions=[
            Action(type="click", selector='a[href="/logout"]'),            # risky: dropped
            Action(type="navigate", value="/broken"),                     # 404: dropped
        ]),
        Scene(id="scene-5", title="Wall", route="/signin", narration="Wall.", actions=[Action(type="wait")]),
    ]


async def rehearsal_checks():
    print("dry run against the fake app")
    for _ in range(50):
        try:
            import httpx
            httpx.get(SITE + "/", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    script = scenes()
    logs = []
    t0 = time.monotonic()
    found = await rehearsal.dry_run(script, SITE + "/", logs.append)
    took = time.monotonic() - t0
    status = [(r["scene"], r["action"], r["status"]) for r in found]
    check("links are followed so later steps are reached for real", status[:2] == [(1, 0, "ok"), (1, 1, "ok")], status[:2])
    check("a button is found but never clicked; the step after it is unverified, not a failure",
          status[2:4] == [(2, 0, "not_clicked"), (2, 1, "unverified")], status[2:4])
    check("missing and covered elements; a form field is found but never typed into",
          status[4:8] == [(3, 0, "missing"), (3, 1, "ok"), (3, 2, "ok"), (3, 3, "hidden")], status[4:8])
    check("a sign-out link is risky; a page that answers 404 fails", status[8:10] == [(4, 0, "risky"), (4, 1, "failed")]
          and "HTTP 404" in found[9]["note"], found[8:10])
    check("a page that lands on a sign-in form fails its steps", status[10:] == [(5, 0, "failed")]
          and "sign-in form" in found[10]["note"], found[10:])
    check(f"the rehearsal is quick ({took:.1f}s)", took < 40)
    dropped = rehearsal.apply(script, found, logs.append)
    kept = [[a.selector or a.value or a.type for a in s.actions] for s in script]
    check("apply drops missing, covered, risky and failed steps; keeps the rest",
          dropped == 5 and kept == [['a[href="/rules"]', 'a[href="/rules/42"]'], ["#enable", "#modal-save"], ["down", "#search"], [], []],
          kept)
    check("each drop is logged with its reason", any(line.startswith("Dry run: scene 3 click '#covered' dropped (hidden") for line in logs), logs)
    check("summary", rehearsal.summary(found, dropped) == "11 steps rehearsed: 4 landed, 1 button found (clicked only on camera), "
          "1 unverified, 1 missing, 1 covered, 1 risky, 2 failed; 5 dropped", rehearsal.summary(found, dropped))


def script_checks():
    print("frozen scripts")
    source = Storyboard(project_name="Warestack", tagline="t", total_duration_sec=57.0, intro_sec=4.6, cover="x", soundtrack="y",
                        scenes=[Scene(id="scene-1", title="A", route="/", narration="Line one.", duration_sec=9.1,
                                      verdict="ok", seen="dashboard", actions=[Action(type="scroll", value="down")]),
                                Scene(id="scene-2", title="B", route="/rules", narration="Line two.", duration_sec=8.0)])
    fresh = orchestrator._script_storyboard({"storyboard": source, "from_job": "abc"}, ["", "A better second line."])
    check("the script keeps pages, steps and lines; replaces a line; clears every measured or produced field",
          [s.route for s in fresh.scenes] == ["/", "/rules"] and fresh.scenes[0].narration == "Line one."
          and fresh.scenes[1].narration == "A better second line." and fresh.scenes[0].actions[0].value == "down"
          and fresh.total_duration_sec == 0 and fresh.intro_sec == 0 and fresh.cover is None and fresh.soundtrack is None
          and fresh.scenes[0].verdict is None and fresh.scenes[0].duration_sec == 0 and source.scenes[1].narration == "Line two.")

    started = []

    async def fake_run(job_id, req, user_id=None, script=None):
        started.append((req.from_job, script))
        orchestrator.JOBS[job_id].status = "completed"

    real, orchestrator.run_pipeline_task = orchestrator.run_pipeline_task, fake_run
    try:
        with TestClient(app) as alice, TestClient(app) as bob:
            ids = {}
            for client, who in ((alice, "alice"), (bob, "bob")):
                r = client.post("/api/auth/signup", json={"email": f"{who}@example.com", "password": "script-pass-1"})
                ids[who] = r.json()["id"]
            for job_id, kind in (("aaaaaaaaaaaa", "video"), ("bbbbbbbbbbbb", "analysis")):
                db.execute("INSERT INTO jobs (id, user_id, created_at) VALUES (?, ?, ?)", (job_id, ids["alice"], time.time()))
                orchestrator.JOBS[job_id] = JobStatus(job_id=job_id, status="completed", kind=kind, app_url="https://x.dev",
                                                      storyboard=source if kind == "video" else None,
                                                      launch_kit=LaunchKit(readme_markdown="r", devpost_pitch="d", x_thread=["t"],
                                                                           linkedin_post="l"))
            body = {"app_url": "https://x.dev", "from_job": "aaaaaaaaaaaa", "narration": ["", "Better."]}
            r = alice.post("/api/pipeline/start", json=body)
            check("the owner re-records a finished video from its script", r.status_code == 202 and started
                  and started[-1][0] == "aaaaaaaaaaaa" and started[-1][1]["storyboard"] is source
                  and started[-1][1]["launch_kit"] is not None, (r.status_code, r.text[:200]))
            check("another user's job is a 404", bob.post("/api/pipeline/start", json=body).status_code == 404)
            r = alice.post("/api/pipeline/start", json=dict(body, from_job="bbbbbbbbbbbb", narration=None))
            check("an analysis has no script to re-record", r.status_code == 422 and "no script" in r.json()["detail"], r.text)
            r = alice.post("/api/pipeline/start", json=dict(body, narration=["only one"]))
            check("one narration line per scene", r.status_code == 422 and r.json()["detail"].startswith("narration: 2 lines expected"), r.text)
            r = alice.post("/api/pipeline/start", json=dict(body, from_job="not-hex"))
            check("a malformed job id is refused by validation", r.status_code == 422)
    finally:
        orchestrator.run_pipeline_task = real


asyncio.run(rehearsal_checks())
script_checks()
print(f"\n{sum(results)}/{len(results)} checks passed (work dir {WORK})")
sys.exit(0 if all(results) else 1)
