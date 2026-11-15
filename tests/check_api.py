"""Offline checks for what Launch Kit calls over HTTP: CORS for the dashboard (3000) and Launch Kit (3780), the cover
poster endpoint (session, ownership, 404 until painted, JPEG with a cache header), `coverUrl` in `GET /api/jobs`,
`_publish` keeping cover.jpg next to demo.mp4, and the planner asking for the poster's art direction on every run
without adding title cards or changing anything else in the prompt.

No keys, no network: a temporary storage dir and database, FastAPI's TestClient. The developer's .env is not read.

    .venv/bin/python tests/check_api.py
"""
import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the backend root (tests/..)
WORK = Path(tempfile.mkdtemp(prefix="demoday-api-"))
sys.path.insert(0, BACKEND)

import dotenv  # noqa: E402

dotenv.load_dotenv = lambda *args, **kwargs: False  # hermetic: app.config must not pick up a local .env
for name in ("CORS_ORIGINS", "WORK_DIR", "SIGNUP_CODE", "GEMINI_API_KEY", "FAL_KEY", "DB_BACKUP_PATH", "GCS_BUCKET",
             "STATIC_DIR"):
    os.environ.pop(name, None)
os.environ.update({"STORAGE_DIR": str(WORK / "storage"), "DB_PATH": str(WORK / "demoday.db"), "COOKIE_SECURE": "false"})

from fastapi.testclient import TestClient  # noqa: E402

from app import db  # noqa: E402
from app.config import storage_root  # noqa: E402
from app.main import ORIGINS, app  # noqa: E402
from app.models.schemas import JobStatus, Scene, Storyboard  # noqa: E402
from app.services import orchestrator, planner  # noqa: E402

results = []
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"\xff\xd9"


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + str(detail)[:300] if detail != '' else ''}")


def cors_checks():
    print("CORS")
    local = ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3780", "http://127.0.0.1:3780"]
    check("default origins: the dashboard and Launch Kit", ORIGINS == local, ORIGINS)
    example = (Path(BACKEND) / ".env.example").read_text()
    check(".env.example lists the same origins", f"CORS_ORIGINS={','.join(local)}\n" in example)
    c = TestClient(app)
    for origin in ("http://localhost:3780", "http://127.0.0.1:3780"):
        r = c.options("/api/jobs", headers={"Origin": origin, "Access-Control-Request-Method": "GET"})
        check(f"preflight from {origin} allowed with credentials", r.status_code == 200
              and r.headers.get("access-control-allow-origin") == origin
              and r.headers.get("access-control-allow-credentials") == "true", dict(r.headers))
    r = c.options("/api/jobs", headers={"Origin": "http://localhost:9999", "Access-Control-Request-Method": "GET"})
    check("an unknown origin is refused", "access-control-allow-origin" not in r.headers, r.status_code)


def cover_endpoint_checks():
    print("GET /api/pipeline/cover/{job_id}")
    alice, bob, anon = TestClient(app), TestClient(app), TestClient(app)
    users = {}
    for client, who in ((alice, "alice"), (bob, "bob")):
        r = client.post("/api/auth/signup", json={"email": f"{who}@example.com", "password": "cover-password-1"})
        assert r.status_code == 200, r.text
        users[who] = r.json()["id"]
    job_id = "c0ffee000001"
    db.execute("INSERT INTO jobs (id, user_id, created_at) VALUES (?, ?, ?)", (job_id, users["alice"], time.time()))
    orchestrator.JOBS[job_id] = JobStatus(
        job_id=job_id, status="completed", progress=100, stage="completed", app_url="https://x.dev", created_at=time.time(),
        video_url=f"/api/pipeline/video/{job_id}",
        storyboard=Storyboard(project_name="X", tagline="t", cover="fal.ai FLUX poster (fal-ai/flux/schnell)",
                              scenes=[Scene(id="scene-1", title="Hook", route="/", narration="Hello.")]))
    url = f"/api/pipeline/cover/{job_id}"
    check("no session -> 401", anon.get(url).status_code == 401)
    r = bob.get(url)
    check("another user's job -> 404 Job not found", r.status_code == 404 and r.json()["detail"] == "Job not found", r.text)
    r = alice.get(url)
    check("no poster yet -> 404 Cover not found", r.status_code == 404 and r.json()["detail"] == "Cover not found", r.text)
    check("a malformed id -> 404", alice.get("/api/pipeline/cover/not-a-job").status_code == 404)
    (storage_root() / job_id).mkdir(parents=True, exist_ok=True)
    (storage_root() / job_id / "cover.jpg").write_bytes(JPEG)
    r = alice.get(url)
    check("the owner gets the JPEG with a private cache header", r.status_code == 200 and r.content == JPEG
          and r.headers["content-type"] == "image/jpeg" and r.headers.get("cache-control") == "private, max-age=86400",
          (r.status_code, dict(r.headers)))
    check("bob still gets 404 once it exists", bob.get(url).status_code == 404)
    orchestrator.JOBS[job_id].cover_url = url
    rows = alice.get("/api/jobs").json()
    check("GET /api/jobs carries coverUrl", [row.get("coverUrl") for row in rows if row["id"] == job_id] == [url], rows)
    check("GET /api/jobs/{id} carries cover_url", alice.get(f"/api/jobs/{job_id}").json().get("cover_url") == url)
    spec = app.openapi()
    content = spec["paths"].get(url.replace(job_id, "{job_id}"), {}).get("get", {}).get("responses", {}).get("200", {})
    check("documented in OpenAPI as image/jpeg; version 1.5.0", "image/jpeg" in content.get("content", {})
          and spec["info"]["version"] == "1.5.0", spec["info"]["version"])


def publish_checks():
    print("_publish keeps cover.jpg next to demo.mp4")
    job_id = "abc123abc123"
    same = storage_root() / job_id  # local default: WORK_DIR unset, so the scratch dir is the result dir
    (same / "shots").mkdir(parents=True)
    for name, data in (("demo.mp4", b"video"), ("cover.jpg", JPEG), ("job.json", b"{}"), ("scene_01.mp4", b"x"),
                       ("cover.png", b"x"), ("cover_card.jpg", b"x"), ("shots/a.jpg", b"x")):
        (same / name).write_bytes(data)
    out = orchestrator._publish(same, job_id, same / "demo.mp4", same / "cover.jpg")
    check("same dir: demo.mp4, cover.jpg and job.json stay, every intermediate goes",
          out == same / "demo.mp4" and sorted(p.name for p in same.iterdir()) == ["cover.jpg", "demo.mp4", "job.json"],
          sorted(p.name for p in same.iterdir()))
    (same / "cover.jpg").write_bytes(b"half-written")
    orchestrator._publish(same, job_id, same / "demo.mp4", None)
    check("same dir, no poster: a leftover cover.jpg is removed", not (same / "cover.jpg").exists())
    scratch = WORK / "scratch" / "def456def456"  # Cloud Run: WORK_DIR on local disk, storage on the bucket
    scratch.mkdir(parents=True)
    for name, data in (("demo.mp4", b"video2"), ("cover.jpg", b"jpeg2"), ("intro.wav", b"x")):
        (scratch / name).write_bytes(data)
    out = orchestrator._publish(scratch, "def456def456", scratch / "demo.mp4", scratch / "cover.jpg")
    result = storage_root() / "def456def456"
    check("separate scratch dir: both copied into storage, the scratch dir removed",
          out == result / "demo.mp4" and out.read_bytes() == b"video2" and (result / "cover.jpg").read_bytes() == b"jpeg2"
          and not scratch.exists() and sorted(p.name for p in result.iterdir()) == ["cover.jpg", "demo.mp4"])


async def planner_checks():
    print("planner: the poster's art direction")
    captured = []

    async def capture(prompt, **kwargs):
        captured.append(prompt)
        raise RuntimeError("captured")

    repo = {"name": "demo", "description": "A demo app.", "readme": "# demo", "dependencies": [], "routes": [], "repos": []}
    info = {"url": "https://x.dev", "title": "Demo", "description": "", "headings": ["Welcome"], "links": [],
            "buttons": [], "inputs": [], "pages": []}
    saved, os.environ["GEMINI_API_KEY"] = planner._generate_json, "offline-test-key"
    planner._generate_json = capture
    try:
        prompts = {}
        for label, kwargs in (("plain", {}), ("cover", {"want_cover": True}), ("cards", {"title_cards": True}),
                              ("cards+cover", {"title_cards": True, "want_cover": True}),
                              ("clips+cover", {"ai_clips": True, "want_cover": True})):
            storyboard, note = await planner.generate_storyboard(repo, info, 60, "energetic_tech", "", **kwargs)
            prompts[label] = captured[-1]
        check("a failed Gemini call still gives the template storyboard", storyboard.scenes and "template" in note, note)
    finally:
        planner._generate_json = saved
        os.environ.pop("GEMINI_API_KEY", None)
    plain, cover = prompts["plain"], prompts["cover"]
    captured.clear()
    check("without want_cover nothing asks for cover_visual", '"cover_visual"' not in plain)
    start = cover.index('\n   "cover_visual": one or two sentences')
    end = cover.index("user interfaces or people.", start) + len("user interfaces or people.")
    stripped = (cover[:start] + cover[end:]).replace(' "cover_visual": "...",', "", 1)
    check("want_cover adds the JSON key and one rule, and changes nothing else (same word budget, no title cards)",
          cover.count('"cover_visual"') == 2 and "behind its typeset title" in cover and "7. The video opens" not in cover
          and stripped == plain)
    check("with title cards, rule 7 asks once: one image serves the cards and the poster",
          prompts["cards+cover"] == prompts["cards"] and prompts["cards"].count('"cover_visual"') == 2
          and "7. The video opens" in prompts["cards"] and "behind its typeset title" not in prompts["cards"])
    check("with AI clips, the poster's art direction sits next to the shot descriptions",
          '"intro_visual"' in prompts["clips+cover"] and prompts["clips+cover"].count('"cover_visual"') == 2)
    print("planner: the brand, not the repository")
    cases = [({"name": "core-api"}, {"title": "Engineering standards for GitHub | Warestack", "url": "https://www.warestack.com/x"}, "Warestack"),
             ({"name": "fastapi"}, {"title": "FastAPI", "url": "https://fastapi.tiangolo.com/"}, "FastAPI"),
             ({"name": "ui"}, {"title": "shadcn/ui - The Foundation for your Design System", "url": "https://ui.shadcn.com/"}, "shadcn/ui"),
             ({}, {"title": "Linear – Plan and build products", "url": "https://linear.app/"}, "Linear"),
             ({"name": "demoday-backend"}, {"title": "DemoDay Launch Studio", "url": "https://demoday-1.europe-west1.run.app/"}, "DemoDay Launch Studio"),
             ({"name": "core-api"}, {"title": "Home", "url": "https://x.dev/", "site_name": "Acme Cloud"}, "Acme Cloud"),
             ({"name": "y"}, {"title": "GitHub - x/y", "url": "https://github.com/x/y", "site_name": "GitHub"}, "y"),
             ({"name": "frontend"}, {"title": "Dashboard", "url": "https://acme-ops.vercel.app/"}, "Acme-ops")]
    got = [planner.project_name(repo, app) for repo, app, _ in cases]
    check("site name, the title segment matching the domain, then a non-generic repo name, then the domain",
          got == [want for _, _, want in cases], got)
    funnels = [{"name": "PR to enforced standard", "outcome": "A pull request is checked against the org's rules.",
                "steps": ["/", "/rules"], "evidence": "README; app/dashboard/[org]/rules", "strength": 0.9}]
    saved, os.environ["GEMINI_API_KEY"] = planner._generate_json, "offline-test-key"
    planner._generate_json = capture
    try:
        await planner.generate_storyboard(repo, info, 60, "energetic_tech", "", funnels=funnels)
        with_funnels = captured[-1]
        await planner.generate_storyboard(repo, info, 60, "energetic_tech", "", funnels=[])
        no_funnels = captured[-1]
    finally:
        planner._generate_json = saved
        os.environ.pop("GEMINI_API_KEY", None)
    check("funnels: the block sits after the site map, before the rules, and rule 1 tours funnel 1",
          with_funnels.index("SITE MAP") < with_funnels.index("FUNNELS: THE PRODUCT'S STRONGEST PATHS") < with_funnels.index("RULES")
          and "1. PR to enforced standard: A pull request is checked against the org's rules. Pages: / -> /rules." in with_funnels
          and "Tour FUNNEL 1 above in order" in with_funnels, with_funnels[-900:])
    check("no funnels, no block and no rule (the prompt is unchanged)", no_funnels == plain and "FUNNEL" not in no_funnels)
    sb = planner._sanitize_storyboard({"cover_visual": "  glowing   request paths ", "scenes": [
        {"title": "Hook", "route": "/", "narration": "Hello there."}]}, "X")
    check("cover_visual is cleaned up by the sanitizer", sb.cover_visual == "glowing request paths", sb.cover_visual)


async def main():
    cors_checks()
    cover_endpoint_checks()
    publish_checks()
    await planner_checks()
    print(f"\n{sum(results)}/{len(results)} checks passed (work dir {WORK})")
    return 0 if all(results) else 1


sys.exit(asyncio.run(main()))
