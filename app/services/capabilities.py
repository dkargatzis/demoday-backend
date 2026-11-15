"""The capability map: what the backend can do against what the frontend uses, measured from the code, no model.

Endpoints come from route declarations (FastAPI/Flask decorators with their APIRouter/Blueprint prefixes, Express,
Fastify and Koa routes including `route({method, url})` objects, Next.js and SvelteKit route handlers); calls come
from API-looking path literals in the frontend (fetch, axios and API clients, template literals with `${...}`
parameters). Paths are compared by their core (the segments after any api/vN prefix, parameters as `:p`), so a
frontend that calls `${API}/v1/rules/${id}` matches a backend `@router.get("/{rule_id}")` under prefix `/rules`.
An endpoint with no matching call is the evidence for "the backend can, the app doesn't show it".
"""
import asyncio
import posixpath
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import httpx

from app.services.analysis import _SKIP, _fetch_text

MAX_SCAN_FILES = 300
MAX_SCAN_BYTES = 250_000  # scanned with patterns only, never sent to a model
METHODS = ("get", "post", "put", "patch", "delete")
_SOURCE = re.compile(r"\.(py|ts|tsx|js|jsx|mjs|vue|svelte|go|rb)$", re.IGNORECASE)
_UI_FILE = re.compile(r"\.(tsx|jsx|vue|svelte)$", re.IGNORECASE)
# Next.js / SvelteKit route handlers live inside the frontend folder but declare real endpoints.
_HANDLER_FILE = re.compile(r"(^|/)(app|pages/api|routes)/(.*/)?(route|\+server)\.(ts|js)$", re.IGNORECASE)
_FRONTEND_DIR = re.compile(r"(^|/)(frontend|web|webapp|client|clients|ui|dashboard|console|portal|apps/(web|app|dashboard|"
                           r"frontend|admin|console))/|(^|/)(hooks|components|composables|stores?)/", re.IGNORECASE)
_ROUTE_PATH = re.compile(r"(^|/)(api|apis|routers?|routes|endpoints?|controllers?|handlers?|views|urls|blueprints?|resources)(/|\.)"
                         r"|router|controller|routes|endpoints?|(^|/)(server|main|app|index|urls)\.(py|ts|js)$", re.IGNORECASE)
_CALL_PATH = re.compile(r"(^|/)(api|apis|client|clients|services?|queries|mutations|hooks|fetchers?|requests?|sdk|lib)(/|\.|-|_)"
                        r"|api|quer|mutation|client|service|fetch|request", re.IGNORECASE)
_PY_DECORATOR = re.compile(r"@(\w+)\.(get|post|put|patch|delete|route|api_route)\(\s*[rf]?[\"']([^\"']*)[\"']"
                           r"(?:[^)]*?methods\s*=\s*\[([^\]]*)\])?", re.IGNORECASE)
_PY_PREFIX = re.compile(r"(\w+)\s*=\s*(?:APIRouter|Blueprint)\(([^)]*)\)", re.IGNORECASE)
_PREFIX_ARG = re.compile(r"(?:prefix|url_prefix)\s*=\s*[rf]?[\"']([^\"']*)[\"']")
_JS_ROUTE = re.compile(r"\b(\w+)\.(get|post|put|patch|delete|all)\(\s*[\"'`]([^\"'`]+)[\"'`]", re.IGNORECASE)
_JS_ROUTE_OBJ = re.compile(r"method\s*:\s*\[?\s*[\"'`](GET|POST|PUT|PATCH|DELETE)[\"'`][^{}]{0,240}?url\s*:\s*[\"'`]([^\"'`]+)"
                           r"|url\s*:\s*[\"'`]([^\"'`]+)[\"'`][^{}]{0,240}?method\s*:\s*\[?\s*[\"'`](GET|POST|PUT|PATCH|DELETE)",
                           re.IGNORECASE | re.DOTALL)
_JS_EXPORT = re.compile(r"export\s+(?:const|let|async\s+function|function)\s+(\w+)")
_JS_REGISTER = re.compile(r"\b(?:register|use)\(\s*(\w+)\s*,\s*\{[^{}]*?prefix\s*:\s*[\"'`]([^\"'`]*)[\"'`]")
_JS_USE_PATH = re.compile(r"\.use\(\s*[\"'`](/[^\"'`]*)[\"'`]\s*,\s*(\w+)")  # express: app.use("/rules", rulesRouter)
_JS_IMPORT = re.compile(r"import\s*(?:(\w+)\s*,?\s*)?(?:\{([^}]*)\})?\s*from\s*[\"']([^\"']+)[\"']")
_PY_INCLUDE = re.compile(r"include_router\(\s*([\w.]+)\s*,[^)]*?prefix\s*=\s*[rf]?[\"']([^\"']*)[\"']")
_PY_BLUEPRINT = re.compile(r"register_blueprint\(\s*([\w.]+)\s*,[^)]*?url_prefix\s*=\s*[rf]?[\"']([^\"']*)[\"']")
_NEXT_HANDLER = re.compile(r"export\s+(?:async\s+)?(?:function|const)\s+(GET|POST|PUT|PATCH|DELETE)\b")
_CALL_LITERAL = re.compile(r"[\"'`]((?:\$\{[^}]{1,80}\})?/(?:api|v\d)(?:/[^\"'`\s?#)]*)?)(?:\?[^\"'`]*)?[\"'`]")
_CALL_CLIENT = re.compile(r"\b(?:fetch|axios|api|apiRequest|apiClient|client|http|request|\$fetch|ky)\s*(?:\.\s*(get|post|put|patch|delete))?\s*"
                          r"(?:<[^>()]{0,80}>)?\(\s*[\"'`]((?:\$\{[^}]{1,80}\})?/[^\"'`\s?#]+)", re.IGNORECASE)
_JS_CLIENT_OBJECTS = {"api", "apirequest", "apiclient", "client", "http", "axios", "request", "fetcher", "instance", "$fetch", "ky"}


def normalize(path: str) -> str:
    """'/api/v1/rules/{rule_id}/' and '${API}/v1/rules/${id}?x=1' -> '/api/v1/rules/:p' and ':p/v1/rules/:p'."""
    path = path.split("?", 1)[0].split("#", 1)[0].strip()
    path = re.sub(r"\$\{[^}]*\}|\{[^}]*\}|<[^>]*>|(?<=/):[A-Za-z_]\w*|\[\.{0,3}[^\]]*\]", ":p", path)
    path = re.sub(r"/{2,}", "/", path).rstrip("/") or "/"
    return path.lower()


def core(path: str) -> Tuple[str, ...]:
    """The segments that name the capability: leading api, vN and parameter segments dropped."""
    segments = [s for s in normalize(path).split("/") if s]
    while segments and (segments[0] in ("api", "apis", "rest", ":p") or re.fullmatch(r"v\d+", segments[0])):
        segments.pop(0)
    return tuple(segments)


def _join(prefix: str, path: str) -> str:
    return "/" + "/".join(s for s in (prefix.strip("/"), path.strip("/")) if s)


def _resolve_module(importer: str, module: str, known: Dict[str, Any]) -> Optional[str]:
    """'./secret-router' imported by routes/v4/index.ts -> 'routes/v4/secret-router.ts' when that file was scanned."""
    if not module.startswith("."):
        return None
    base = posixpath.normpath(posixpath.join(posixpath.dirname(importer), module))
    for candidate in (base, base + ".ts", base + ".js", base + ".tsx", base + "/index.ts", base + "/index.js"):
        if candidate in known:
            return candidate
    return None


def _group_mounts(text: str) -> List[Tuple[str, str]]:
    """Routers mounted inside a prefixed group: `register(async (r) => { await r.register(fn) }, { prefix: "/x" })`
    -> [(fn, "/x")] (an inner prefix adds to the group's)."""
    out: List[Tuple[str, str]] = []
    for m in re.finditer(r"\bregister\(\s*async\s*\(\s*\w+[^)]*\)\s*=>\s*\{", text):
        start, depth, i = m.end() - 1, 0, m.end() - 1
        while i < len(text):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                break
            i += 1
        group = re.match(r"\}\s*,\s*\{[^{}]*?prefix\s*:\s*[\"'`]([^\"'`]*)", text[i:i + 300])
        if not group:
            continue
        for fn, inner in re.findall(r"\.register\(\s*(\w+)\s*(?:,\s*\{[^{}]*?prefix\s*:\s*[\"'`]([^\"'`]*)[\"'`])?", text[start:i]):
            out.append((fn, _join(group.group(1), inner or "")))
    return out


def mount_prefixes(files: Sequence[Dict[str, str]]) -> Dict[str, str]:
    """Each router file's mount prefix, from where another file mounts it: Fastify `register(fn, {prefix})` and
    Express `use("/x", router)` resolved through the importer's imports (export names can repeat across files),
    FastAPI `include_router(module.router, prefix=)` and Flask `register_blueprint(bp, url_prefix=)` by module name.
    Nested mounts add up (index -> v1/index -> secret-router)."""
    known = {f["path"]: f for f in files}
    exporters: Dict[str, List[str]] = defaultdict(list)
    for f in files:
        for name in _JS_EXPORT.findall(f["text"]):
            exporters[name].append(f["path"])
    parent: Dict[str, Tuple[str, str]] = {}  # child file -> (mounting file, prefix)
    for f in files:
        imported: Dict[str, str] = {}
        for default, names, module in _JS_IMPORT.findall(f["text"]):
            target = _resolve_module(f["path"], module, known)
            for name in [default, *re.findall(r"(?:\w+\s+as\s+)?(\w+)", names or "")]:
                if name and target:
                    imported[name] = target
        mounts = (_JS_REGISTER.findall(f["text"]) + [(fn, prefix) for prefix, fn in _JS_USE_PATH.findall(f["text"])]
                  + _group_mounts(f["text"]))  # groups last: their outer prefix wins over an inner match
        for fn, prefix in mounts:
            child = imported.get(fn) or (exporters[fn][0] if len(exporters.get(fn, [])) == 1 else None)
            if child and child != f["path"]:
                parent[child] = (f["path"], prefix)
        for expr, prefix in _PY_INCLUDE.findall(f["text"]) + _PY_BLUEPRINT.findall(f["text"]):
            parts = expr.split(".")
            name = parts[-2] if len(parts) > 1 and parts[-1] in ("router", "bp", "blueprint", "api") else parts[-1]
            for path in known:
                stem = posixpath.splitext(posixpath.basename(path))[0]
                if path.endswith(".py") and stem in (name, name.replace("_router", ""), name.replace("_bp", "")):
                    parent.setdefault(path, (f["path"], prefix))
    prefixes: Dict[str, str] = {}
    for child in parent:
        chain, node, seen = [], child, set()
        while node in parent and node not in seen and len(chain) < 6:
            seen.add(node)
            node, prefix = parent[node]
            chain.append(prefix)
        prefixes[child] = "/" + "/".join(p.strip("/") for p in reversed(chain) if p.strip("/"))
    return prefixes


def endpoints_in(path: str, text: str, mount: str = "") -> List[Dict[str, str]]:
    """Route declarations in one file: [{method, path, file}], with the file's own router prefix and `mount` (where
    another file mounts it)."""
    found: List[Dict[str, str]] = []
    if path.endswith(".py"):
        prefixes = {}
        for var, args in _PY_PREFIX.findall(text):
            arg = _PREFIX_ARG.search(args)
            prefixes[var] = arg.group(1) if arg else ""
        for var, verb, route, methods in _PY_DECORATOR.findall(text):
            verbs = [m.strip(" \"'").lower() for m in methods.split(",") if m.strip(" \"'")] if methods else []
            if verb.lower() in METHODS:
                verbs = [verb.lower()]
            for method in verbs or ["get"]:
                found.append({"method": method.upper(), "path": _join(mount, _join(prefixes.get(var, ""), route)), "file": path})
        return found
    if re.search(r"(^|/)(app|src/app|pages/api|src/routes)/.*(route|\+server)\.(ts|js)$", path):
        url = re.sub(r"^.*?(?:^|/)(?:src/)?(?:app|pages|routes)/", "/", path)
        url = re.sub(r"/(route|\+server)\.(ts|js)$", "", url)
        url = re.sub(r"/\([^)]*\)", "", url)  # Next.js route groups
        for method in _NEXT_HANDLER.findall(text):
            found.append({"method": method.upper(), "path": url or "/", "file": path})
    prefix = mount
    for obj, verb, route in _JS_ROUTE.findall(text):
        if obj.lower() in _JS_CLIENT_OBJECTS or not route.startswith("/"):
            continue  # a client call (api.get("/x")) is a call, not a route
        found.append({"method": verb.upper() if verb.lower() != "all" else "ANY", "path": _join(prefix, route), "file": path})
    for m in _JS_ROUTE_OBJ.finditer(text):
        method, url = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
        if url and url.startswith("/"):
            found.append({"method": method.upper(), "path": _join(prefix, url), "file": path})
    return found


def calls_in(path: str, text: str) -> List[Tuple[str, str]]:
    """API paths a frontend file calls: [(method or '', normalized path)]."""
    out = set()
    for verb, url in _CALL_CLIENT.findall(text):
        out.add(((verb or "").upper(), normalize(url)))
    with_method = {url for method, url in out if method}
    for url in _CALL_LITERAL.findall(text):  # a bare literal (a URL constant) covers any method, unless a call gave one
        if normalize(url) not in with_method:
            out.add(("", normalize(url)))
    return sorted(out - {("", url) for url in with_method})


def _matches(endpoint_core: Tuple[str, ...], call_core: Tuple[str, ...]) -> bool:
    if not endpoint_core or not call_core:
        return False
    if endpoint_core == call_core:
        return True
    shorter, longer = sorted((endpoint_core, call_core), key=len)
    return len(shorter) >= 2 and longer[-len(shorter):] == shorter  # a prefix only one side knows (mount points)


_RETIRED = re.compile(r"deprecat|legacy|(^|/)old/", re.IGNORECASE)


def _area(path_core: Tuple[str, ...]) -> str:
    return next((s for s in path_core if s != ":p"), ":p")


def build_map(files: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    """The map from scanned files ([{repo, path, text}]): endpoints with their callers, coverage per area, the
    endpoints no frontend file calls."""
    endpoints, calls = [], []
    mounts = mount_prefixes(files)
    for f in files:
        frontend = bool(_UI_FILE.search(f["path"]) or _FRONTEND_DIR.search(f["path"]))
        if not frontend or _HANDLER_FILE.search(f["path"]):
            endpoints += [dict(e, repo=f["repo"]) for e in endpoints_in(f["path"], f["text"], mounts.get(f["path"], ""))]
        if frontend or not f["path"].endswith(".py"):
            calls += [(method, url, f"{f['repo']}/{f['path']}") for method, url in calls_in(f["path"], f["text"])
                      if frontend]
    seen, unique = set(), []
    for e in endpoints:
        key = (e["method"], normalize(e["path"]))
        if key not in seen and core(e["path"]):
            seen.add(key)
            unique.append(e)
    call_cores = [(method, core(url), where) for method, url, where in calls]
    by_area: Dict[str, Counter] = defaultdict(Counter)
    for e in unique:
        e_core = core(e["path"])
        callers = sorted({where for method, c, where in call_cores
                          if _matches(e_core, c) and (not method or e["method"] in (method, "ANY"))})
        e["callers"] = callers[:3]
        e["called"] = bool(callers)
        area = _area(e_core)
        by_area[area]["total"] += 1
        by_area[area]["called"] += e["called"]
    areas = sorted(({"area": a, "endpoints": c["total"], "called": c["called"]} for a, c in by_area.items()),
                   key=lambda a: (-a["endpoints"], a["area"]))
    # Deprecated or legacy routers are left uncalled on purpose: counted, never reported as a gap.
    retired = [e for e in unique if not e["called"] and _RETIRED.search(e["file"])]
    open_areas = {a["area"]: a["endpoints"] - a["called"] for a in areas}
    uncalled = sorted(({"method": e["method"], "path": normalize(e["path"]), "file": f"{e['repo']}/{e['file']}",
                        "area": _area(core(e["path"]))} for e in unique if not e["called"] and e not in retired),
                      key=lambda u: (-open_areas.get(u["area"], 0), u["area"], u["path"]))
    return {"endpoints": len(unique), "called": sum(e["called"] for e in unique), "calls": len({(m, u) for m, u, _ in calls}),
            "areas": areas[:30], "uncalled": uncalled[:80], "uncalled_total": len(uncalled), "retired": len(retired),
            "frontend_seen": bool(calls), "files_scanned": len(files)}


def scan_candidates(tree: Sequence[Tuple[str, int]], limit: int = MAX_SCAN_FILES) -> List[str]:
    """Source files where routes are declared (half the budget) and where the frontend calls the API (the other
    half), bigger files first in each (more routes or calls in them). UI files and frontend folders are never read
    as route files; a side that needs fewer files leaves its share to the other."""
    routes, callers = [], []
    for path, size in tree:
        if _SKIP.search(path) or not _SOURCE.search(path) or size <= 0 or size > MAX_SCAN_BYTES:
            continue
        frontend = bool(_UI_FILE.search(path) or _FRONTEND_DIR.search(path))
        if _HANDLER_FILE.search(path) or (not frontend and _ROUTE_PATH.search(path)):
            routes.append((-size, path))
        elif frontend and (_CALL_PATH.search(path) or _UI_FILE.search(path)):
            callers.append((0 if _CALL_PATH.search(path) else 1, -size, path))
    routes.sort()
    callers.sort()
    half = limit // 2
    take_routes = min(len(routes), max(half, limit - len(callers)))
    take_callers = min(len(callers), limit - take_routes)
    return [path for _, path in routes[:take_routes]] + [path for _, _, path in callers[:take_callers]]


async def scan(sources: Sequence[Dict[str, Any]], token: Optional[str] = None,
               log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Fetches the candidate files of every repository and builds the map."""
    headers = {"User-Agent": "DemoDay-Launch-Studio"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    per_repo = max(60, MAX_SCAN_FILES // max(1, len(sources)))
    wanted = [(source, path) for source in sources for path in scan_candidates(source.get("tree") or [], per_repo)]
    gate = asyncio.Semaphore(12)
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=headers) as client:
        async def fetch(source, path):
            async with gate:
                text = await _fetch_text(client, source, path, bool(token), limit=MAX_SCAN_BYTES)
            return {"repo": source.get("full_name") or source["repo"], "path": path, "text": text} if text else None
        files = [f for f in await asyncio.gather(*(fetch(*w) for w in wanted)) if f]
    result = build_map(files)
    if log is not None:
        log(summary_line(result))
    return result


def summary_line(m: Dict[str, Any]) -> str:
    if not m.get("endpoints"):
        return f"Capability map: no route declarations found in {m.get('files_scanned', 0)} files"
    top = ", ".join(f"{a['area']} {a['called']}/{a['endpoints']}" for a in m["areas"][:6])
    retired = f", {m['retired']} deprecated" if m.get("retired") else ""
    return (f"Capability map: {m['endpoints']} endpoints, {m['called']} called by the frontend, "
            f"{m.get('uncalled_total', m['endpoints'] - m['called'])} not{retired} ({top})")


def prompt_section(m: Optional[Dict[str, Any]]) -> str:
    """The map as facts for the analysis prompt ('' without endpoints)."""
    if not m or not m.get("endpoints"):
        return ""
    lines = ["", "CAPABILITY MAP (measured from the code, not guessed: backend endpoints and whether any frontend file calls them)",
             f"- {m['endpoints']} endpoints; {m['called']} called by the frontend; {m.get('uncalled_total', 0)} with no caller found"
             + (f"; {m['retired']} more in deprecated or legacy routers (not gaps)" if m.get("retired") else ""),
             "- Areas (called/total): " + "; ".join(f"{a['area']} {a['called']}/{a['endpoints']}" for a in m["areas"][:20])]
    if not m.get("frontend_seen"):
        lines.append("- No frontend code was given, so every endpoint reads as uncalled: do not report that as a gap.")
    else:
        lines.append("- Endpoints no frontend file calls (what the backend can do that the app does not show), biggest areas first:")
        per_area: Counter = Counter()
        for u in m["uncalled"]:
            if per_area[u["area"]] < 6 and sum(per_area.values()) < 48:
                per_area[u["area"]] += 1
                lines.append(f"  - {u['method']} {u['path']} ({u['file']})")
    return "\n".join(lines) + "\n"
