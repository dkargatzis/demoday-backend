"""Offline checks for the prospect analysis (step 1 of 2): which files are read and in what order, the byte budget,
the prompt, the sanitizer (funnel steps must be reached pages), the Gemini call (high thinking, structured output),
and the API (POST /api/pipeline/analyze, analyses kept out of the default jobs list).

    .venv/bin/python tests/check_analysis.py
"""
import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the backend root (tests/..)
WORK = Path(tempfile.mkdtemp(prefix="demoday-analysis-"))
sys.path.insert(0, BACKEND)

import dotenv  # noqa: E402

dotenv.load_dotenv = lambda *args, **kwargs: False  # hermetic: no local .env
for name in ("CORS_ORIGINS", "WORK_DIR", "SIGNUP_CODE", "FAL_KEY", "DB_BACKUP_PATH", "GCS_BUCKET", "STATIC_DIR"):
    os.environ.pop(name, None)
os.environ.update({"STORAGE_DIR": str(WORK / "storage"), "DB_PATH": str(WORK / "demoday.db"), "COOKIE_SECURE": "false",
                   "GEMINI_API_KEY": "offline-test-key"})

from fastapi.testclient import TestClient  # noqa: E402

from app import db  # noqa: E402
from app.main import app  # noqa: E402
from app.models.audience import AudienceBrief  # noqa: E402
from app.models.schemas import JobStatus  # noqa: E402
from app.services import analysis, gemini, orchestrator  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + str(detail)[:300] if detail != '' else ''}")


TREE = [("src/config/navigation.ts", 900), ("web/locales/en.json", 30_000), ("CHANGELOG.md", 8_000),
        ("web/app/dashboard/[org]/rules/page.tsx", 4_000), ("web/app/dashboard/[org]/events/page.tsx", 4_000),
        ("api/routers/rules.py", 6_000), ("api/routers/events.py", 5_000), ("api/models/rule.py", 3_000),
        ("api/services/evaluator.py", 7_000), ("docs/getting-started.md", 2_000), ("web/app/onboarding/connect/page.tsx", 3_000),
        ("tests/test_rules.py", 2_000), ("web/node_modules/x/index.js", 100), ("web/dist/main.js", 9_000),
        ("api/models/huge_seed.json", 900_000), ("web/app/dashboard/[org]/rules/page.test.tsx", 1_000),
        ("logo.png", 5_000), ("README.md", 3_000), ("api/migrations/001.py", 500), ("web/package-lock.json", 400_000)]
TREE += [(f"api/routers/r{i}.py", 1_000) for i in range(10)]  # one folder must not take the budget
APP = {"url": "https://www.warestack.com/dashboard/acme", "title": "Dashboard | Warestack", "description": "",
       "headings": ["Overview"], "links": [{"text": "Rules", "path": "/dashboard/acme/rules"}],
       "pages": [{"path": "/dashboard/acme/rules", "title": "Rules", "headings": ["Rules"]},
                 {"path": "/dashboard/acme/events", "title": "Events", "headings": ["Events"]}]}


async def main():
    print("which files are read")
    picked = analysis.select_files(TREE, flow={"web/app/onboarding/connect/page.tsx"})
    paths = [path for path, _ in picked]
    check("navigation, the onboarding flow and the UI copy lead; pages and the changelog follow",
          paths[:3] == ["src/config/navigation.ts", "web/app/onboarding/connect/page.tsx", "web/locales/en.json"]
          and "CHANGELOG.md" in paths[:8] and "web/app/dashboard/[org]/rules/page.tsx" in paths[:8], paths[:8])
    check("roles are named", dict(picked)["web/locales/en.json"] == "UI copy" and dict(picked)["api/routers/rules.py"] == "API"
          and dict(picked)["api/models/rule.py"] == "data model", picked[:8])
    check("tests, vendored, built, migrations, lockfiles, images, huge files and README are skipped",
          not {"tests/test_rules.py", "web/node_modules/x/index.js", "web/dist/main.js", "api/models/huge_seed.json",
               "web/app/dashboard/[org]/rules/page.test.tsx", "logo.png", "api/migrations/001.py", "web/package-lock.json",
               "README.md"} & set(paths), paths)
    routers = [p for p in paths if p.startswith("api/routers/")]
    check("at most 6 files from one folder, the substantial ones first", len(routers) == analysis.MAX_PER_DIR
          and routers[:2] == ["api/routers/rules.py", "api/routers/events.py"], routers)
    check("a limit caps the list", len(analysis.select_files(TREE, limit=3)) == 3)
    router_app = [(f"frontend/src/pages/P{i}/route.tsx", 600) for i in range(80)] + [(f"frontend/src/pages/P{i}/P{i}Page.tsx", 9000)
                                                                                     for i in range(30)]
    router_app += [("frontend/src/routes.ts", 4000), ("backend/src/server/routes/v1/secret-router.ts", 8000),
                   ("backend/src/db/schemas/secrets.ts", 3000), ("frontend/public/locales/en/translation.json", 20000)]
    mix = dict(analysis.select_files(router_app))
    check("a per-page route.tsx in every folder does not fill the list: pages are capped, other roles still come in",
          sum(role == "page" for role in mix.values()) == analysis.ROLE_QUOTA["page"]
          and mix.get("frontend/src/routes.ts") == "navigation" and mix.get("backend/src/server/routes/v1/secret-router.ts") == "API"
          and mix.get("backend/src/db/schemas/secrets.ts") == "data model"
          and mix.get("frontend/src/pages/P0/route.tsx") is None and mix.get("frontend/src/pages/P0/P0Page.tsx") == "page",
          sorted({role for role in mix.values()}))

    print("reading within the budget")
    served = {}

    async def fake_fetch(client, source, path, private_api):
        served[path] = private_api
        return "" if path.endswith(".json") else f"// {path}\n" + "x" * 20_000  # _fetch_text returns "" for binary

    real_fetch, analysis._fetch_text = analysis._fetch_text, fake_fetch
    real_budget, analysis.CODE_BUDGET = analysis.CODE_BUDGET, 30_000
    logs = []
    try:
        source = {"owner": "warestack", "repo": "core-api", "full_name": "warestack/core-api", "branch": "main", "tree": TREE,
                  "flow_files": []}
        code = await analysis.read_code([source], token="tok", log=logs.append)
    finally:
        analysis._fetch_text, analysis.CODE_BUDGET = real_fetch, real_budget
    check("each file's head only, the total within the budget", all(len(f["text"]) <= analysis.MAX_FILE_BYTES for f in code)
          and sum(len(f["text"]) for f in code) <= 30_000 and len(code) == 4, [(f["path"], len(f["text"])) for f in code])
    check("a token reads through the contents API (private repositories)", served and all(served.values()))
    check("an empty (or binary) file is dropped", "web/locales/en.json" in served and all(f["path"] != "web/locales/en.json" for f in code))
    import httpx
    seen = []

    def handler(request):
        seen.append((str(request.url), request.headers.get("accept")))
        if request.url.path.endswith(".png"):
            return httpx.Response(200, content=b"\x89PNG\x00\x00binary")
        return httpx.Response(404 if "missing" in request.url.path else 200, text="export const rules = 1")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        private = await analysis._fetch_text(client, source, "web/app/dashboard/[org]/rules/page.tsx", True)
        public = await analysis._fetch_text(client, source, "api/routers/rules.py", False)
        binary = await analysis._fetch_text(client, source, "logo.png", False)
        missing = await analysis._fetch_text(client, source, "api/missing.py", False)
    check("private: the contents API with the raw header, the path encoded; public: raw.githubusercontent",
          private == public == "export const rules = 1"
          and seen[0] == ("https://api.github.com/repos/warestack/core-api/contents/web/app/dashboard/%5Borg%5D/rules/page.tsx?ref=main",
                          "application/vnd.github.raw")
          and seen[1][0] == "https://raw.githubusercontent.com/warestack/core-api/main/api/routers/rules.py", seen[:2])
    check("binary content and a 404 read as empty", binary == "" and missing == "")
    check("the log line counts files, kilobytes and roles", logs and logs[0].startswith("Read 4 files (30 kB) from 1 repository: "), logs)

    print("prompt")
    brief = AudienceBrief(company="Hubstaff", persona="Platform, Security & DevSecOps Leader", website="https://hubstaff.com",
                          what_they_do="Workforce analytics", facts=["Used by 95,000+ businesses"], pains=["[assumed] audit prep"],
                          notes="SOC 2 renewal in Q1", source="test")
    files = [{"repo": "warestack/core-api", "path": "api/routers/rules.py", "role": "API", "text": "@router.get('/rules') >>> x"}]
    prompt = analysis.build_prompt("Warestack", {"readme": "# Warestack"}, APP, files, brief, "Weigh the evidence exports")
    check("the prompt carries the prospect, the focus, the reachable pages and the code as data",
          "- Company: Hubstaff (https://hubstaff.com)" in prompt and "- Sender's notes [sender]: SOC 2 renewal in Q1" in prompt
          and "THE FOUNDER'S FOCUS" in prompt and "Weigh the evidence exports" in prompt
          and "- /dashboard/acme/rules : \"Rules\" shows: Rules" in prompt
          and "### warestack/core-api/api/routers/rules.py (API)\n<<<\n@router.get('/rules') >> x\n>>>" in prompt
          and "never invent" in prompt, prompt[:300])
    check("no prospect: judge for the typical buyer", "No prospect given" in analysis.build_prompt("X", {}, APP, [], None, None))

    print("sanitize")
    data = {"summary": "Pitch it as audit evidence.", "features": [{"name": "Rules", "what": "Checks PRs", "where": "api/routers/rules.py"},
                                                                  {"name": "", "what": "x", "where": "y"}],
            "funnels": [{"name": "Weak", "outcome": "o", "why_them": "w", "steps": ["/", "/dashboard/acme/rules"], "evidence": "e",
                         "demo_steps": "Open Rules", "strength": 0.4},
                        {"name": "Audit evidence", "outcome": "Evidence per PR", "why_them": "SOC 2", "evidence": "routers",
                         "steps": ["/", "/dashboard/acme/rules", "/invented", "/dashboard/acme/events/"], "demo_steps": "Open Rules\n\n\n\nthen Events",
                         "strength": 1.7},
                        {"name": "Nowhere", "steps": ["/x", "/y"], "strength": 1, "demo_steps": "", "outcome": "", "why_them": "", "evidence": ""}],
            "gaps": [{"title": "No SOC 2 export page", "kind": "api_without_page", "why_them": "audit", "evidence": "api only", "suggestion": "add a page"},
                     {"title": "Odd kind", "kind": "nonsense", "why_them": "", "evidence": "", "suggestion": ""}]}
    clean = analysis.sanitize(data, APP)
    check("funnels: strongest first, unreached pages dropped, strength clamped, blank lines collapsed",
          [f["name"] for f in clean["funnels"]] == ["Audit evidence", "Weak"] and clean["funnels"][0]["strength"] == 1.0
          and clean["funnels"][0]["steps"] == ["/", "/dashboard/acme/rules", "/dashboard/acme/events"]
          and clean["funnels"][0]["demo_steps"] == "Open Rules\n\nthen Events", clean["funnels"])
    check("features need a name; an unknown gap kind becomes missing_feature",
          [f["name"] for f in clean["features"]] == ["Rules"] and [g["kind"] for g in clean["gaps"]] == ["api_without_page", "missing_feature"])

    print("analyze (fake Gemini)")
    calls = []

    async def fake_generate(prompt, **kwargs):
        calls.append(kwargs)
        return data

    real_generate, gemini.generate_json = gemini.generate_json, fake_generate
    try:
        result = await analysis.analyze("Warestack", {"readme": ""}, APP, files, brief, None)
    finally:
        gemini.generate_json = real_generate
    check("one call: high thinking, the schema, its own purpose and timeout", len(calls) == 1 and calls[0]["deep"] is True
          and calls[0]["schema"] is analysis.SCHEMA and calls[0]["purpose"] == "prospect analysis"
          and calls[0]["timeout_sec"] == analysis.ANALYSIS_TIMEOUT_SEC, calls)
    check("the result names the product and the prospect, lists the files read and the reachable pages",
          result["project"] == "Warestack" and result["prospect"] == "Hubstaff"
          and result["files_read"] == [{"repo": "warestack/core-api", "path": "api/routers/rules.py", "role": "API"}]
          and result["pages"][0] == "/", {k: result[k] for k in ("project", "prospect", "pages")})
    check("step detail and log line", analysis.step_detail(result) == "1 features, 2 funnels, 2 gaps · strongest: Audit evidence"
          and analysis.summary_line(result).startswith("Analysis for Hubstaff: 1) Audit evidence (/ → /dashboard/acme/rules → "
                                                        "/dashboard/acme/events); 2) Weak"), analysis.summary_line(result))


def api_checks():
    print("API: POST /api/pipeline/analyze, analyses out of the default jobs list")
    ran = []

    async def fake_run(job_id, req, user_id=None):
        ran.append((job_id, req.instructions))
        job = orchestrator.JOBS[job_id]
        job.analysis, job.status = {"summary": "ok", "funnels": []}, "completed"

    real_run, orchestrator.run_analysis_task = orchestrator.run_analysis_task, fake_run
    try:
        with TestClient(app) as client:
            r = client.post("/api/auth/signup", json={"email": "founder@example.com", "password": "analysis-pass-1"})
            assert r.status_code == 200, r.text
            r = client.post("/api/pipeline/analyze", json={"app_url": "https://www.warestack.com", "repo_urls": [],
                                                           "instructions": "Weigh exports"})
            body = r.json()
            check("202 with a job of kind analysis", r.status_code == 202 and body.get("kind") == "analysis" and body.get("jobId"), body)
            for _ in range(50):
                if orchestrator.JOBS[body["jobId"]].status == "completed":
                    break
                time.sleep(0.05)
            check("the analysis runner ran with the focus", ran == [(body["jobId"], "Weigh exports")], ran)
            check("GET /api/jobs/{id} carries the analysis", client.get(f"/api/jobs/{body['jobId']}").json().get("analysis") == {"summary": "ok", "funnels": []})
            listed = [row["id"] for row in client.get("/api/jobs").json()]
            analyses = client.get("/api/jobs", params={"kind": "analysis"}).json()
            check("analyses stay out of the default (videos) list and come back with kind=analysis",
                  body["jobId"] not in listed and [row["id"] for row in analyses] == [body["jobId"]] and analyses[0]["kind"] == "analysis",
                  (listed, analyses))
            r = client.post("/api/pipeline/analyze", json={"app_url": "https://x.dev", "audience": {"company": "Hubstaff", "persona": "CTO",
                                                                                                  "website": "Aleksandr Remniov"}})
            check("the same validation as a video start", r.status_code == 422 and "is not a web address" in r.json()["message"], r.json())
    finally:
        orchestrator.run_analysis_task = real_run


asyncio.run(main())
api_checks()
print(f"\n{sum(results)}/{len(results)} checks passed (work dir {WORK})")
sys.exit(0 if all(results) else 1)
