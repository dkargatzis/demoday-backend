"""Offline checks for the capability map (app/services/capabilities.py): route declarations across frameworks with
their mount prefixes, frontend API calls, matching by path core and method, coverage per area, deprecated routers
kept out of the gaps, which files are scanned, and the prompt section.

    .venv/bin/python tests/check_capabilities.py
"""
import os
import sys
import tempfile

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the backend root (tests/..)
os.environ.setdefault("STORAGE_DIR", tempfile.mkdtemp(prefix="demoday-capmap-"))
sys.path.insert(0, BACKEND)

from app.services import capabilities as cap  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + str(detail)[:400] if detail != '' else ''}")


FILES = {
    "main.py": '''from fastapi import FastAPI
from app.routers import rules, reports
app = FastAPI()
app.include_router(rules.router, prefix="/api/v1/rules")
app.include_router(reports.router, prefix="/api/v1/reports")''',
    "app/routers/rules.py": '''router = APIRouter()
@router.get("/")
async def list_rules(): ...
@router.get("/{rule_id}")
async def get_rule(rule_id: str): ...
@router.post("/{rule_id}/enable")
async def enable(rule_id: str): ...''',
    "app/routers/reports.py": '''router = APIRouter(prefix="/weekly")
@router.get("/{report_id}/export")
async def export(report_id): ...
@router.get("/")
async def weekly(): ...''',
    "app/views/admin.py": '''bp = Blueprint("admin", __name__, url_prefix="/admin")
@bp.route("/users", methods=["GET", "POST"])
def users(): ...''',
    "backend/src/routes/index.ts": '''import { registerProjectRouter } from "./project-router";
import { registerSecretRouter } from "./v4/secret-router";
export const registerRoutes = async (server) => {
  await server.register(
    async (projectRouter) => {
      await projectRouter.register(registerProjectRouter);
    },
    { prefix: "/projects" }
  );
  await server.register(registerSecretRouter, { prefix: "/secrets" });
};''',
    "backend/src/routes/project-router.ts": '''export const registerProjectRouter = async (server) => {
  server.route({ method: "GET", url: "/:projectId/users", handler: list });
  server.route({ url: "/:projectId/audit-logs", method: "GET", handler: logs });
};''',
    "backend/src/routes/v4/secret-router.ts": '''export const registerSecretRouter = async (server) => {
  server.route({ method: "POST", url: "/:secretName", handler: create });
};''',
    "backend/src/routes/v3/deprecated-secret-router.ts": '''export const registerSecretRouter = async (server) => {
  server.route({ method: "GET", url: "/raw", handler: raw });
};''',
    "server/app.js": '''import billingRouter from "./routes/billing";
app.use("/billing", billingRouter);''',
    "server/routes/billing.js": '''const router = express.Router();
router.get("/invoices", list);
export default router;''',
    "web/app/api/exports/route.ts": '''export async function GET(request: Request) { return Response.json([]); }''',
    "web/src/hooks/api/rules.ts": '''export const useRules = () => apiRequest.get(`/api/v1/rules`);
export const useRule = (id: string) => apiRequest.get<Rule>(`/api/v1/rules/${id}`);
export const enableRule = (id: string) => apiRequest.post(`/api/v1/rules/${id}/enable`);
export const projectUsers = (id: string) => fetch(`${API_URL}/api/v1/projects/${id}/users`);
export const createSecret = (name: string) => axios.post(`/api/v4/secrets/${name}`, {});''',
    "web/src/pages/Billing.tsx": '''const data = await fetch("/api/billing/invoices?page=1");''',
}


def main():
    print("paths")
    check("normalize: parameters of every style become :p, query and trailing slash dropped",
          [cap.normalize(p) for p in ("/api/v1/rules/{rule_id}/", "${API}/v1/rules/${id}?x=1", "/projects/:projectId/users",
                                      "/users/<int:uid>", "/app/[org]/rules")]
          == ["/api/v1/rules/:p", ":p/v1/rules/:p", "/projects/:p/users", "/users/:p", "/app/:p/rules"])
    check("core drops api, versions and leading parameters", cap.core("${API}/api/v2/rules/${id}") == ("rules", ":p")
          and cap.core("/") == ())

    files = [{"repo": "acme/app", "path": path, "text": text} for path, text in FILES.items()]
    mounts = cap.mount_prefixes(files)
    print("mounts")
    check("FastAPI include_router, Fastify grouped and direct mounts, Express use(), resolved through imports",
          mounts.get("app/routers/rules.py") == "/api/v1/rules" and mounts.get("app/routers/reports.py") == "/api/v1/reports"
          and mounts.get("backend/src/routes/project-router.ts") == "/projects"
          and mounts.get("backend/src/routes/v4/secret-router.ts") == "/secrets"
          and mounts.get("server/routes/billing.js") == "/billing", mounts)
    check("the same export name in a deprecated file is not mounted by mistake",
          "backend/src/routes/v3/deprecated-secret-router.ts" not in mounts)

    print("endpoints and calls")
    m = cap.build_map(files)
    uncalled = {(u["method"], u["path"]) for u in m["uncalled"]}
    check("every framework's routes are found: 13 endpoints", m["endpoints"] == 13, m["endpoints"])
    check("called: rules (list, one, enable), project users, the secret POST, billing invoices",
          m["called"] == 6, m)
    check("uncalled: the report export and weekly, Flask admin GET and POST, project audit logs, the Next.js handler",
          uncalled == {("GET", "/api/v1/reports/weekly/:p/export"), ("GET", "/api/v1/reports/weekly"), ("GET", "/admin/users"),
                       ("POST", "/admin/users"), ("GET", "/projects/:p/audit-logs"), ("GET", "/api/exports")}, sorted(uncalled))
    check("the deprecated router is counted, never a gap", m["retired"] == 1 and ("GET", "/raw") not in uncalled)
    areas = {a["area"]: (a["called"], a["endpoints"]) for a in m["areas"]}
    check("coverage per area", areas.get("rules") == (3, 3) and areas.get("reports") == (0, 2) and areas.get("admin") == (0, 2)
          and areas.get("projects") == (1, 2), areas)
    check("a call's method must fit: a GET call does not cover the POST", cap.build_map([
        {"repo": "r", "path": "app/routers/x.py", "text": '@router.post("/api/x")\nasync def x(): ...'},
        {"repo": "r", "path": "web/src/api/x.ts", "text": 'api.get("/api/x")'}])["called"] == 0)
    check("no frontend code: the map says so", cap.build_map(files[:4])["frontend_seen"] is False)

    print("which files are scanned")
    tree = [("app/routers/rules.py", 3000), ("web/src/hooks/api/rules.ts", 2000), ("web/src/components/Button.tsx", 900),
            ("web/app/api/exports/route.ts", 300), ("tests/test_rules.py", 900), ("backend/src/routes/index.ts", 9000),
            ("web/src/hooks/useTheme.ts", 400), ("docs/api.md", 2000), ("node_modules/x/router.js", 900)]
    picked = cap.scan_candidates(tree, limit=10)
    check("route files (biggest first) and the frontend's API files; tests, docs and vendored code skipped",
          picked[:3] == ["backend/src/routes/index.ts", "app/routers/rules.py", "web/app/api/exports/route.ts"]
          and "web/src/hooks/api/rules.ts" in picked and "tests/test_rules.py" not in picked
          and "docs/api.md" not in picked and "node_modules/x/router.js" not in picked, picked)
    check("a side that needs fewer files leaves its share", len(cap.scan_candidates(tree, limit=4)) == 4)

    print("prompt")
    section = cap.prompt_section(m)
    check("the prompt section states the counts, areas and the uncalled endpoints as measured facts",
          "CAPABILITY MAP (measured from the code, not guessed" in section and "- 13 endpoints; 6 called by the frontend; 6 with no caller found; 1 more in deprecated" in section
          and "  - GET /api/v1/reports/weekly/:p/export (acme/app/app/routers/reports.py)" in section, section)
    check("no frontend code: the prompt warns not to call everything a gap",
          "do not report that as a gap" in cap.prompt_section(cap.build_map(files[:4])))
    check("no endpoints: no section", cap.prompt_section({"endpoints": 0}) == "" and cap.prompt_section(None) == "")
    check("summary line", cap.summary_line(m).startswith("Capability map: 13 endpoints, 6 called by the frontend, 6 not, 1 deprecated"),
          cap.summary_line(m))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


sys.exit(main())
