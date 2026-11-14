"""Context gathering: GitHub repositories (public or private) and a rendered inventory of the live app."""
import asyncio
import json
import re
import tomllib
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import httpx
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright

from app.config import GITHUB_API_URL, GITHUB_RAW_URL, get_secret
from app.services.netguard import guard_context
from app.services.recorder import VIEWPORT, is_risky, launch_chromium

GITHUB_API = GITHUB_API_URL
RAW_GITHUB = GITHUB_RAW_URL
MANIFESTS = ("package.json", "requirements.txt", "pyproject.toml")
MAX_REPOS = 5


# --------------------------------------------------------------------------- GitHub


def parse_github_url(repo_url: str) -> Optional[tuple]:
    match = re.search(r"github\.com[/:]([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", repo_url)
    if not match:
        return None
    repo = match.group(2)
    return match.group(1), repo[:-4] if repo.endswith(".git") else repo


async def extract_repos_context(repo_urls: List[str], token: Optional[str] = None) -> Dict[str, Any]:
    """Merges several repositories (for example a frontend and a backend) into one planning context.

    The first repository names the project; READMEs share the prompt budget; dependencies and routes are combined.
    """
    if not repo_urls:
        return {"name": "", "full_name": "", "description": "", "language": "", "readme": "", "dependencies": [],
                "routes": [], "repos": [], "notes": ["No repository given; planning from the live app only"]}
    urls = repo_urls[:MAX_REPOS]
    infos = await asyncio.gather(*(extract_github_context(url, token) for url in urls))
    merged = dict(infos[0])
    if len(infos) > 1:
        budget = max(1500, 6000 // len(infos))
        merged["readme"] = "\n\n".join(f"## {info.get('full_name') or url}\n{info['readme'][:budget]}"
                                       for url, info in zip(urls, infos) if info.get("readme"))
        merged["dependencies"] = _unique(d for info in infos for d in info.get("dependencies", []))[:30]
        merged["routes"] = _unique(r for info in infos for r in info.get("routes", []))[:30]
        merged["description"] = next((info["description"] for info in infos if info.get("description")), "")
        merged["language"] = ", ".join(_unique(info.get("language", "") for info in infos))
    merged["repos"] = [{"url": url, "full_name": info.get("full_name") or url, "private": info.get("private", False),
                        "description": info.get("description", ""), "language": info.get("language", "")}
                       for url, info in zip(urls, infos)]
    merged["notes"] = [note for info in infos for note in info.get("notes", [])]
    return merged


def _unique(items: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for item in items:
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


async def extract_github_context(repo_url: str, token: Optional[str] = None) -> Dict[str, Any]:
    """README, metadata, dependencies and page routes of one repository.

    `token` (the user's saved GitHub token) unlocks private repositories; the server's GITHUB_TOKEN is the fallback.
    """
    parsed = parse_github_url(repo_url)
    if not parsed:
        return {"name": "", "readme": "", "description": "", "dependencies": [], "routes": [],
                "notes": [f"{repo_url} is not a GitHub repository URL; continuing without its context"]}
    owner, repo = parsed
    full = f"{owner}/{repo}"
    token = token or get_secret("GITHUB_TOKEN") or None
    headers = {"User-Agent": "DemoDay-Launch-Studio", "Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    info: Dict[str, Any] = {"owner": owner, "repo": repo, "name": repo, "full_name": full, "private": False,
                            "description": "", "homepage": "", "language": "", "topics": [], "stars": 0,
                            "readme": "", "dependencies": [], "routes": [], "notes": []}

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=headers) as client:
        branch, token_rejected = None, False
        resp = await _get(client, f"{GITHUB_API}/repos/{owner}/{repo}")
        if resp is not None and resp.status_code == 401 and token:
            info["notes"].append(f"GitHub rejected the token for {full} (expired or revoked); update it in Settings")
            client.headers.pop("Authorization", None)
            token, token_rejected = None, True
            resp = await _get(client, f"{GITHUB_API}/repos/{owner}/{repo}")
        status = resp.status_code if resp is not None else 0
        meta = _json(resp) if status == 200 else None
        if meta:
            branch = meta.get("default_branch")
            info.update(
                name=meta.get("name") or repo,
                full_name=meta.get("full_name") or full,
                private=bool(meta.get("private")),
                description=meta.get("description") or "",
                homepage=meta.get("homepage") or "",
                language=meta.get("language") or "",
                topics=meta.get("topics") or [],
                stars=meta.get("stargazers_count") or 0,
            )
        elif status == 404:
            if token_rejected:
                reason = "the saved GitHub token was rejected"
            elif token:
                reason = "your GitHub token can't access it"
            else:
                reason = "no GitHub token is saved. Add one in Settings to include private repositories"
            info["notes"].append(f"{full} isn't visible: it doesn't exist, or it's private and {reason}")
            return info
        elif status in (403, 429):
            info["notes"].append("GitHub API rate limit reached"
                                 + ("" if token else "; add a GitHub token in Settings for higher limits"))
        else:
            info["notes"].append(f"Couldn't reach GitHub for {full}")

        readme_task = _fetch_readme(client, owner, repo, branch)
        tree_task = _get_json(client, f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1") if branch else None
        readme, tree = await asyncio.gather(readme_task, tree_task or _none())
        info["readme"] = _clean_readme(readme)

        paths = [item["path"] for item in (tree or {}).get("tree", []) if item.get("type") == "blob"]
        info["routes"] = derive_routes(paths)
        manifests = [m for m in MANIFESTS if m in paths] if paths else list(MANIFESTS)
        for branch_name in ([branch] if branch else ["main", "master"]):
            deps = await _fetch_dependencies(client, owner, repo, branch_name, manifests)
            if deps:
                info["dependencies"] = deps
                break
    return info


async def _none():
    return None


async def _get(client: httpx.AsyncClient, url: str) -> Optional[httpx.Response]:
    try:
        return await client.get(url)
    except httpx.HTTPError:
        return None


def _json(resp: httpx.Response) -> Optional[Dict]:
    try:
        data = resp.json()
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


async def _get_json(client: httpx.AsyncClient, url: str) -> Optional[Any]:
    try:
        resp = await client.get(url)
        if resp.status_code == 200:
            return resp.json()
    except (httpx.HTTPError, json.JSONDecodeError):
        pass
    return None


async def _get_text(client: httpx.AsyncClient, url: str, **kwargs) -> str:
    try:
        resp = await client.get(url, **kwargs)
        if resp.status_code == 200:
            return resp.text
    except httpx.HTTPError:
        pass
    return ""


async def _fetch_readme(client: httpx.AsyncClient, owner: str, repo: str, branch: Optional[str]) -> str:
    text = await _get_text(client, f"{GITHUB_API}/repos/{owner}/{repo}/readme",
                           headers={"Accept": "application/vnd.github.raw"})
    if text:
        return text
    for name in ([branch] if branch else []) + ["main", "master"]:
        text = await _get_text(client, f"{RAW_GITHUB}/{owner}/{repo}/{name}/README.md")
        if text:
            return text
    return ""


def _clean_readme(text: str, limit: int = 6000) -> str:
    """Drops HTML, images and badge clutter so the planner sees prose, not markup."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[\s*\]\([^)]*\)", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:limit]


def derive_routes(paths: List[str]) -> List[str]:
    """Page routes from Next.js (app/ and pages/ routers) and SvelteKit file conventions."""
    routes = set()
    for path in paths:
        if m := re.match(r"^(?:src/)?app/(?:(.*)/)?page\.(?:tsx|jsx|ts|js|mdx)$", path):
            segments = [s for s in (m.group(1) or "").split("/") if s]
        elif m := re.match(r"^(?:src/)?pages/(.+)\.(?:tsx|jsx|ts|js|mdx)$", path):
            if m.group(1).startswith("api/") or m.group(1).rsplit("/", 1)[-1].startswith("_"):
                continue
            segments = m.group(1).split("/")
            if segments[-1] == "index":
                segments = segments[:-1]
        elif m := re.match(r"^src/routes/(?:(.*)/)?\+page\.svelte$", path):
            segments = [s for s in (m.group(1) or "").split("/") if s]
        else:
            continue
        # Route groups "(marketing)" and parallel slots "@modal" don't appear in URLs.
        segments = [s for s in segments if not (s.startswith("(") and s.endswith(")")) and not s.startswith("@")]
        if any(s.startswith(("[", "_")) for s in segments):
            continue  # dynamic segments need real data to navigate to
        routes.add("/" + "/".join(segments))
    return sorted(routes, key=lambda r: (r.count("/"), r))[:25]


async def _fetch_dependencies(client, owner: str, repo: str, branch: str, manifests: List[str]) -> List[str]:
    deps: List[str] = []
    contents = await asyncio.gather(*(
        _get_text(client, f"{RAW_GITHUB}/{owner}/{repo}/{branch}/{name}") for name in manifests
    ))
    for name, text in zip(manifests, contents):
        if not text:
            continue
        try:
            if name == "package.json":
                pkg = json.loads(text)
                deps += list(pkg.get("dependencies", {})) + list(pkg.get("devDependencies", {}))
            elif name == "requirements.txt":
                deps += [_requirement_name(line) for line in text.splitlines()
                         if line.strip() and not line.lstrip().startswith(("#", "-"))]
            elif name == "pyproject.toml":
                data = tomllib.loads(text)
                deps += [_requirement_name(d) for d in data.get("project", {}).get("dependencies", [])]
                poetry = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
                deps += [d for d in poetry if d.lower() != "python"]
        except (ValueError, tomllib.TOMLDecodeError, AttributeError):
            continue
    seen, unique = set(), []
    for dep in deps:
        if dep and dep.lower() not in seen:
            seen.add(dep.lower())
            unique.append(dep)
    return unique[:25]


def _requirement_name(spec: str) -> str:
    return re.split(r"[\s\[<>=!~;@]", spec.strip(), maxsplit=1)[0]


# --------------------------------------------------------------------------- live app

INVENTORY_SCRIPT = r"""
() => {
  // Rendered, not transparent (opacity 0 menus count as "visible" to Playwright but can't be clicked),
  // and, when on screen, actually the top-most element at its center (catches clipped dropdowns).
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width <= 2 || r.height <= 2) return false;
    if (el.checkVisibility) {
      if (!el.checkVisibility({ opacityProperty: true, visibilityProperty: true, contentVisibilityAuto: true })) return false;
    } else {
      const s = window.getComputedStyle(el);
      if (s.visibility === "hidden" || s.display === "none" || s.opacity === "0") return false;
    }
    const x = r.left + r.width / 2, y = r.top + r.height / 2;
    if (x < 0 || y < 0 || x >= window.innerWidth || y >= window.innerHeight) return true;
    const hit = document.elementFromPoint(x, y);
    return !!hit && (hit === el || el.contains(hit));
  };
  const inNav = (el) => !!el.closest("nav, header, [role=navigation], [role=banner]");
  const text = (el) => (el.innerText || el.textContent || "").replace(/\s+/g, " ").trim().slice(0, 80);
  const pick = (selector, limit) => Array.from(document.querySelectorAll(selector)).filter(visible).slice(0, limit);
  const meta = (key) => {
    const el = document.querySelector(`meta[name="${key}"], meta[property="${key}"]`);
    return el ? (el.getAttribute("content") || "").trim() : "";
  };
  return {
    title: document.title || "",
    hasPassword: Array.from(document.querySelectorAll("input[type=password]")).some(visible),
    description: meta("description") || meta("og:description"),
    headings: pick("h1, h2", 12).map(text).filter(Boolean),
    links: pick("a[href]", 120).map((a) => ({
      text: text(a) || a.getAttribute("aria-label") || "", href: a.getAttribute("href") || "", abs: a.href,
      nav: inNav(a) })),
    buttons: pick("button, [role=button], input[type=submit]", 24).map((b) => ({
      text: text(b) || b.getAttribute("aria-label") || b.value || "", id: b.id || "" })),
    inputs: pick("input:not([type=hidden]):not([type=submit]):not([type=checkbox]):not([type=radio]), textarea", 12)
      .map((i) => ({ tag: i.tagName.toLowerCase(), id: i.id || "", name: i.getAttribute("name") || "",
        placeholder: i.getAttribute("placeholder") || "", aria: i.getAttribute("aria-label") || "" })),
  };
}
"""


def _quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


# IDs generated per render (React useId ":r1:" / "_r_1_" / "«r1»", React Aria, Radix, Headless UI, MUI, long counters)
# change on every page load, so a selector built from one never matches during recording.
_GENERATED_ID = re.compile(r":r\w+:|_r_\w+_|«r\w+»|react-aria\d+|radix-|headlessui-|^mui-|\d{5,}", re.IGNORECASE)


def _stable_id(value: str) -> bool:
    return bool(value) and not _GENERATED_ID.search(value)


async def probe_app_url(app_url: str, storage_state: Optional[Dict] = None) -> Dict[str, Any]:
    """Renders the app in headless Chromium (so SPAs work) and inventories what a demo could interact with.

    `storage_state` (a saved sign-in session) probes the app as the signed-in user. `login_form` reports a visible
    password field, which usually means a sign-in wall. Returns an `error` key when the page can't be loaded at all.
    """
    result: Dict[str, Any] = {"url": app_url, "status": 0, "title": "", "description": "", "headings": [],
                              "links": [], "buttons": [], "inputs": [], "login_form": False}
    async with async_playwright() as pw:
        browser = await launch_chromium(pw)
        try:
            context = await browser.new_context(viewport=VIEWPORT, locale="en-US", storage_state=storage_state,
                                                service_workers="block")
            await guard_context(context)
            page = await context.new_page()
            try:
                response = await page.goto(app_url, wait_until="domcontentloaded", timeout=25_000)
            except PlaywrightError as exc:
                message = str(exc).strip().splitlines()[0][:200]
                result["error"] = ("it redirected to a private network address, which isn't allowed"
                                   if "ERR_BLOCKED_BY_CLIENT" in message else message)
                return result
            result["status"] = response.status if response else 0
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except PlaywrightError:
                pass
            result["url"] = page.url
            raw = await page.evaluate(INVENTORY_SCRIPT)
        finally:
            await browser.close()

    origin = urlparse(result["url"]).netloc
    headings = [re.sub(r"[\s¶#🔗]+$", "", h) for h in raw["headings"]]  # drop permalink glyphs
    result.update(title=raw["title"].strip(), description=raw["description"], headings=[h for h in headings if h],
                  login_form=bool(raw.get("hasPassword")))

    seen_paths = set()
    for link in raw["links"]:
        href, label = link["href"].strip(), link["text"].strip()
        target = urlparse(link["abs"])
        if (not href or not label or '"' in label or href.startswith(("#", "javascript:", "mailto:", "tel:"))
                or target.scheme not in ("http", "https") or target.netloc != origin
                or is_risky(label=label, url=href)):  # never offer sign-out or destructive links to the planner
            continue
        path = target.path or "/"
        if target.query:
            path += "?" + target.query
        if path in seen_paths:
            continue
        seen_paths.add(path)
        result["links"].append({"text": label, "path": path, "selector": f'a[href="{_quote(href)}"]',
                                "nav": bool(link.get("nav"))})
        if len(result["links"]) >= 30:
            break

    for button in raw["buttons"]:
        label = button["text"].strip()
        if is_risky(label=label or button["id"]):
            continue
        if _stable_id(button["id"]):
            selector = f'[id="{_quote(button["id"])}"]'
        elif label and '"' not in label:
            selector = f'role=button[name="{label}"]'
        else:
            continue
        result["buttons"].append({"text": label or button["id"], "selector": selector})

    for field in raw["inputs"]:
        tag = field["tag"]
        for attr, key in (("id", "id"), ("name", "name"), ("placeholder", "placeholder"), ("aria-label", "aria")):
            if field[key] and (attr != "id" or _stable_id(field["id"])):
                selector = f'[id="{_quote(field["id"])}"]' if attr == "id" else f'{tag}[{attr}="{_quote(field[key])}"]'
                result["inputs"].append({"label": field["placeholder"] or field["aria"] or field["name"] or field["id"],
                                         "selector": selector})
                break
    return result
