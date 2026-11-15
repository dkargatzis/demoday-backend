"""One personalized demo end to end against a running server: audience research, "Made for" FLUX cards (the intro
card is the poster), the outreach email and LinkedIn DM, the customer row in /api/jobs. Needs GEMINI_API_KEY and
FAL_KEY on the server; reads https://www.python.org for the brief.

    .venv/bin/python tests/live_personalized.py [base-url] [signup-code]     # default http://127.0.0.1:8000
"""
import json
import sys
import time

import httpx

API = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
CODE = sys.argv[2] if len(sys.argv) > 2 else None
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + str(detail)[:260] if detail != '' else ''}")


def jpeg_size(data: bytes):
    """(width, height) from a JPEG's SOF marker, or None."""
    i = 2
    while data[:2] == b"\xff\xd8" and i + 9 < len(data) and data[i] == 0xFF:
        marker, length = data[i + 1], int.from_bytes(data[i + 2:i + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
        i += 2 + length
    return None


c = httpx.Client(base_url=API, timeout=60, headers={"Origin": API if API.startswith("https://") else "http://localhost:3000"})
c.post("/api/auth/signup", json={"email": f"persona{int(time.time())}@example.com", "password": "persona-password-1",
                                 **({"signup_code": CODE} if CODE else {})}).raise_for_status()
audience = {"company": "Python Software Foundation", "website": "https://www.python.org",
            "persona": "Head of Platform at a 40-person team, drowning in pull request reviews",
            "first_name": "Maria", "goal": "book_call", "cta_url": "https://cal.com/demo/15min"}
r = c.post("/api/pipeline/start", json={"app_url": "https://fastapi.tiangolo.com/", "repo_urls": [], "target_seconds": 30,
                                        "audience": audience})
job = r.json()
check("202 with the pending brief", r.status_code == 202 and (job.get("audience") or {}).get("source") == "pending", job.get("audience"))
job_id = job["jobId"]
t0 = time.monotonic()
while True:
    time.sleep(5)
    job = c.get(f"/api/jobs/{job_id}").json()
    if job["status"] in ("completed", "failed") or time.monotonic() - t0 > 900:
        break
print(f"job {job_id}: {job['status']} after {time.monotonic() - t0:.0f}s | error: {job.get('error')}")
brief = job.get("audience") or {}
steps = {s["id"]: s for s in job.get("steps", [])}
sb = job.get("storyboard") or {}
kit = job.get("launch_kit") or {}
check("completed", job["status"] == "completed", job.get("error"))
check("audience step done with research detail", steps.get("audience", {}).get("status") == "done", steps.get("audience"))
check("brief researched the prospect's site", brief.get("pages_read") and "python.org" in (brief.get("source") or ""),
      {k: brief.get(k) for k in ("source", "pages_read", "what_they_do")})
check("facts are tagged and assumptions labelled", isinstance(brief.get("facts"), list) and isinstance(brief.get("assumptions"), list),
      {"facts": brief.get("facts", [])[:2], "assumptions": brief.get("assumptions", [])[:2]})
check("cover step done (FLUX title cards)", steps.get("cover", {}).get("status") == "done", steps.get("cover"))
check("intro and outro cards in the video", (sb.get("intro_sec") or 0) > 0 and (sb.get("outro_sec") or 0) > 0,
      {"intro_sec": sb.get("intro_sec"), "outro_sec": sb.get("outro_sec"), "cover": sb.get("cover")})
check("ai_clips stays skipped (no fal video credits)", steps.get("ai_clips", {}).get("status") == "skipped")
cover = c.get(f"/api/pipeline/cover/{job_id}")
check("the poster is the painted intro card: a 1280x720 JPEG", job.get("cover_url") == f"/api/pipeline/cover/{job_id}"
      and cover.status_code == 200 and jpeg_size(cover.content) == (1280, 720),
      (job.get("cover_url"), cover.status_code, jpeg_size(cover.content)))
named = sum(1 for s in sb.get("scenes", []) if "Python Software Foundation" in s["narration"] or "Maria" in s["narration"])
check("narration speaks to the viewer", named >= 1, f"named in {named} of {len(sb.get('scenes', []))} scenes")
email = kit.get("outreach_email") or {}
check("outreach email with exactly one demo-link placeholder", email.get("subject") and (email.get("body") or "").count("{{DEMO_LINK}}") == 1,
      {"subject": email.get("subject")})
check("LinkedIn DM present", bool(kit.get("linkedin_dm")), (kit.get("linkedin_dm") or "")[:120])
rows = c.get("/api/jobs").json()
row = next((r for r in rows if r["id"] == job_id), {})
check("jobs list shows the customer row", row.get("customer") == "Python Software Foundation" and row.get("durationSec")
      and row.get("coverUrl") == job.get("cover_url"), row)
for line in job["logs"]:
    if line.startswith(("Customer", "Researching", "python.org", "Audience brief", "Intro card", "Personalized", "Cover art", "Tools used", "Warning")):
        print("   log:", line[:200])
print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
