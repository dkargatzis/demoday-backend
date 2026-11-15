"""Launch Kit v2 contract checks (backend issue #3) against a running server: 422 messages, 202 + jobId, typed SSE
with replay, the jobs endpoints, and the cover poster (no intro/outro cards on a generic run). Runs one real
15-second job, so the server needs GEMINI_API_KEY (and FAL_KEY for the poster checks).

    .venv/bin/python tests/live_contract.py [base-url] [signup-code]     # default http://127.0.0.1:8000
"""
import asyncio
import json
import sys
import time

import httpx

API = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")  # optional: base URL, signup code
CODE = sys.argv[2] if len(sys.argv) > 2 else None
ORIGIN = API if API.startswith("https://") else "http://localhost:3000"
APP = "https://fastapi.tiangolo.com/"
RUN = str(int(time.time()))
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + str(detail)[:240] if detail != '' else ''}")


def jpeg_size(data: bytes):
    """(width, height) from a JPEG's SOF marker, or None."""
    i = 2
    while data[:2] == b"\xff\xd8" and i + 9 < len(data) and data[i] == 0xFF:
        marker, length = data[i + 1], int.from_bytes(data[i + 2:i + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
        i += 2 + length
    return None


def client():
    return httpx.AsyncClient(base_url=API, timeout=60, headers={"Origin": ORIGIN})


async def read_events(c, job_id, last_event_id=None, path="/api/pipeline/events/{}"):
    """All SSE messages of the stream until the server closes it: [{id, event, data}]."""
    headers = {"Last-Event-ID": str(last_event_id)} if last_event_id is not None else {}
    messages, current = [], {}
    async with c.stream("GET", path.format(job_id), headers=headers, timeout=None) as resp:
        assert resp.status_code == 200, resp.status_code
        async for line in resp.aiter_lines():
            if line == "":
                if "data" in current:
                    messages.append({"id": int(current.get("id", 0)), "event": current.get("event", "message"),
                                     "data": json.loads(current["data"])})
                current = {}
            elif not line.startswith(":"):
                key, _, value = line.partition(":")
                current[key] = value.strip()
    return messages


async def main():
    async with client() as alice, client() as bob:
        for c, name in ((alice, "alice"), (bob, "bob")):
            r = await c.post("/api/auth/signup", json={"email": f"{name}{RUN}@example.com", "password": "contract-password-1",
                                                       **({"signup_code": CODE} if CODE else {})})
            assert r.status_code == 200, r.text

        print("validation (422 with a sentence in `message`)")
        r = await alice.post("/api/pipeline/start", json={"app_url": APP, "goals": ["press"]})
        check("unknown goal", r.status_code == 422 and r.json().get("message") ==
              "Unknown goal 'press'. Valid goals: social, community, investors, customers.", r.json().get("message"))
        check("detail keeps Pydantic's errors", isinstance(r.json().get("detail"), list) and r.json()["detail"][0]["loc"][-1] == "goals")
        r = await alice.post("/api/pipeline/start", json={"app_url": APP, "language": "x1-23456789"})
        check("unknown language", r.status_code == 422 and r.json()["message"].startswith("Unknown language 'x1-23456789'"), r.json().get("message"))
        r = await alice.post("/api/pipeline/start", json={"app_url": APP, "target_seconds": 5})
        check("field errors read as a sentence", r.status_code == 422 and r.json()["message"] == "target_seconds: input should be greater than or equal to 15.", r.json().get("message"))
        r = await alice.post("/api/pipeline/start", json={"app_url": "ftp://example.com/"})
        check("non-http URL", r.status_code == 422 and r.json()["message"].endswith("."), r.json().get("message"))
        r = await alice.get("/api/jobs", params={"limit": 0})
        check("query validation too", r.status_code == 422 and r.json()["message"].startswith("limit:"), r.json().get("message"))

        print("start (202, jobId, echoed options)")
        r = await alice.post("/api/pipeline/start", json={"app_url": APP, "repo_urls": [], "target_seconds": 15,
                                                          "goals": ["social", "SOCIAL ", "investors"], "language": "EN"})
        job = r.json()
        job_id = job.get("job_id")
        check("202 with jobId == job_id", r.status_code == 202 and job.get("jobId") == job_id and job_id, {k: job.get(k) for k in ("jobId", "job_id", "status")})
        check("options echo normalized goals and language",
              job.get("options") == {"goals": ["social", "investors"], "requestedLanguage": "en", "language": "en"}, job.get("options"))
        check("app_url and created_at on the job", job.get("app_url") == APP and isinstance(job.get("created_at"), float))
        r = await alice.get(f"/api/jobs/{job_id}")
        check("GET /api/jobs/{id} while running", r.status_code == 200 and r.json()["status"] in ("queued", "running"), r.json().get("status"))
        check("bob can't read it", (await bob.get(f"/api/jobs/{job_id}")).status_code == 404)
        running = await alice.get("/api/jobs")
        check("running jobs are not listed yet", running.status_code == 200 and job_id not in [j["id"] for j in running.json()], running.json())

        print("typed event stream")
        t0 = time.monotonic()
        events = await read_events(alice, job_id)
        elapsed = time.monotonic() - t0
        names = [e["event"] for e in events]
        ids = [e["id"] for e in events]
        check("first event is `job` with id and mode", names[0] == "job" and events[0]["data"]["id"] == job_id
              and events[0]["data"]["mode"].get("videoSource") == "playwright", events[0]["data"])
        check("ids are 1..n without gaps", ids == list(range(1, len(ids) + 1)), ids[:5] + ["..."] + ids[-3:])
        check("only the documented event names", set(names) <= {"job", "step", "log", "status", "done", "error"}, sorted(set(names)))
        check(f"last event is `done` ({elapsed:.0f}s)", names[-1] == "done" and events[-1]["data"]["status"] == "completed", names[-1])
        done = events[-1]["data"]
        check("done carries the full job", done.get("video_url") and done.get("launch_kit") and done.get("storyboard"), list(done))
        steps = {}
        for e in events:
            if e["event"] == "step":
                steps.setdefault(e["data"]["id"], []).append(e["data"]["status"])
        check("steps: repos skipped (no repos), the rest running -> done",
              steps.get("repos") == ["skipped"] and steps.get("ai_clips") == ["skipped"]
              and all(steps.get(s) == ["running", "done"] for s in ("scrape", "storyboard", "launchpack", "narration", "record", "video")), steps)
        step_fields = [e["data"] for e in events if e["event"] == "step"]
        check("step events carry id, status, label", all({"id", "status", "label"} <= set(s) for s in step_fields))
        sb = done["storyboard"]
        check("a generic run has no intro or outro cards", sb.get("intro_sec") == 0 and sb.get("outro_sec") == 0,
              {k: sb.get(k) for k in ("intro_sec", "outro_sec")})
        if (await alice.get("/api/health")).json().get("fal"):
            framed = (sb.get("cover") or "").startswith("the product's own screen")
            check("cover: running -> done, the product's own screen framed (or FLUX from Gemini's art direction)",
                  steps.get("cover") == ["running", "done"] and done.get("cover_url") == f"/api/pipeline/cover/{job_id}"
                  and (framed or ((sb.get("cover") or "").startswith("fal.ai") and sb.get("cover_visual"))),
                  {"steps": steps.get("cover"), "cover_url": done.get("cover_url"), "cover": sb.get("cover"),
                   "visual": sb.get("cover_visual")})
            r = await alice.get(f"/api/pipeline/cover/{job_id}")
            check("GET /api/pipeline/cover/{id}: a 1280x720 JPEG for its owner", r.status_code == 200
                  and r.headers.get("content-type") == "image/jpeg" and jpeg_size(r.content) == (1280, 720),
                  (r.status_code, r.headers.get("content-type"), jpeg_size(r.content), len(r.content)))
            check("bob can't fetch the cover", (await bob.get(f"/api/pipeline/cover/{job_id}")).status_code == 404)
        else:
            check("no FAL_KEY: the cover frames the product's screen, or is skipped with the reason",
                  (steps.get("cover") == ["running", "done"] and done.get("cover_url"))
                  or (steps.get("cover") == ["skipped"] and sb.get("cover") == "skipped: FAL_KEY not set"), steps.get("cover"))
        check("funnels: ranked from the code and the live app, or skipped with a reason",
              steps.get("funnels") in (["running", "done"], ["running", "error"]), steps.get("funnels"))
        logs = [e["data"]["message"] for e in events if e["event"] == "log"]
        check("log events mirror job.logs", logs and logs == done["logs"][1:], f"{len(logs)} vs {len(done['logs'])}")
        statuses = [e["data"] for e in events if e["event"] == "status"]
        check("status snapshots are JobStatus objects with rising progress",
              statuses and all(s["job_id"] == job_id for s in statuses)
              and [s["progress"] for s in statuses] == sorted(s["progress"] for s in statuses), [s["progress"] for s in statuses][-6:])
        check("the last status snapshot equals done", statuses[-1] == done)
        r = await alice.get(f"/api/jobs/{job_id}")
        check("GET /api/jobs/{id} returns the same object as `done`", r.status_code == 200 and r.json() == done)
        legacy = await alice.get(f"/api/pipeline/status/{job_id}")
        check("older status alias still answers", legacy.status_code == 200 and legacy.json()["job_id"] == job_id)

        print("replay with Last-Event-ID")
        n = ids[-1]
        tail = await read_events(alice, job_id, last_event_id=n - 3)
        check("replays exactly the missed events", [e["id"] for e in tail] == [n - 2, n - 1, n] and tail[-1]["event"] == "done", [e["id"] for e in tail])
        check("nothing after the last id", await read_events(alice, job_id, last_event_id=n) == [])
        check("an id beyond the end is harmless", await read_events(alice, job_id, last_event_id=n + 50) == [])
        full = await read_events(alice, job_id, last_event_id=0)
        check("Last-Event-ID: 0 replays everything", [e["id"] for e in full] == ids)
        alias = await read_events(alice, job_id, path="/api/pipeline/stream/{}")
        check("older stream alias carries the same events", [e["id"] for e in alias] == ids)

        print("jobs list")
        r = await alice.get("/api/jobs")
        rows = r.json()
        mine = next((row for row in rows if row["id"] == job_id), None)
        check("finished job listed with camelCase fields", mine and mine["url"] == APP and mine["status"] == "completed"
              and mine["createdAt"].endswith("+00:00") and mine["productName"] == done["storyboard"]["project_name"]
              and mine.get("coverUrl") == done.get("cover_url"), mine)
        check("bob's list is empty", (await bob.get("/api/jobs")).json() == [])

        print("failure path")
        r = await alice.post("/api/pipeline/start", json={"app_url": "http://169.254.169.254/latest/", "repo_urls": []})
        bad_id = r.json()["jobId"]
        events = await read_events(alice, bad_id)
        names = [e["event"] for e in events]
        check("a blocked URL ends with `error`, never `done`", names[-1] == "error" and "done" not in names
              and "private" in events[-1]["data"]["message"].lower() or "blocked" in events[-1]["data"]["message"].lower(), events[-1]["data"])
        check("the last status snapshot says failed", [e for e in events if e["event"] == "status"][-1]["data"]["status"] == "failed")
        r = await alice.get("/api/jobs")
        check("failed jobs are listed with status failed", any(row["id"] == bad_id and row["status"] == "failed" for row in r.json()))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


asyncio.run(main())
