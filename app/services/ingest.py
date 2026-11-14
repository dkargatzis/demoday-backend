"""Context gathering: GitHub repository metadata and a rendered inventory of the live app."""
import asyncio
import json
import re
import tomllib
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright

from app.config import get_secret
from app.services.recorder import VIEWPORT, launch_chromium

GITHUB_API = "https://api.github.com"
RAW_GITHUB = "https://raw.githubusercontent.com"
MANIFESTS = ("package.json", "requirements.txt", "pyproject.toml")


# --------------------------------------------------------------------------- GitHub


def parse_github_url(repo_url: str) -> Optional[tuple]:
    match = re.search(r"github\.com[/:]([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", repo_url)
    if not match:
        return None
    repo = match.group(2)
    return match.group(1), repo[:-4] if repo.endswith(".git") else repo


async def extract_github_context(repo_url: str) -> Dict[str, Any]:
    parsed = parse_github_url(repo_url)
    if not parsed:
        return {"name": "", "readme": "", "description": "", "dependencies": [], "routes": [],
                "notes": [f"{repo_url} is not a GitHub repository URL; continuing without repo context"]}
    owner, repo = parsed
    headers = {"User-Agent": "DemoDay-Launch-Studio", "Accept": "application/vnd.github+json"}
    token = get_secret("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    info: Dict[str, Any] = {"owner": owner, "repo": repo, "name": repo, "full_name": f"{owner}/{repo}",
                            "description": "", "homepage": "", "language": "", "topics": [], "stars": 0,
                            "readme": "", "dependencies": [], "routes": [], "notes": []}

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=headers) as client:
        branch = None
        meta = await _get_json(client, f"{GITHUB_API}/repos/{owner}/{repo}")
        if meta:
            branch = meta.get("default_branch")
            info.update(
                name=meta.get("name") or repo,
                full_name=meta.get("full_name") or info["full_name"],
                description=meta.get("description") or "",
                homepage=meta.get("homepage") or "",
                language=meta.get("language") or "",
                topics=meta.get("topics") or [],
                stars=meta.get("stargazers_count") or 0,
            )
        else:
            info["notes"].append("GitHub API unavailable (rate limit?): set GITHUB_TOKEN for richer repo context")

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


async def probe_app_url(app_url: str) -> Dict[str, Any]:
    """Renders the app in headless Chromium (so SPAs work) and inventories what a demo could interact with.

    Returns an `error` key when the page can't be loaded at all.
    """
    result: Dict[str, Any] = {"url": app_url, "status": 0, "title": "", "description": "", "headings": [],
                              "links": [], "buttons": [], "inputs": []}
    async with async_playwright() as pw:
        browser = await launch_chromium(pw)
        try:
            page = await browser.new_page(viewport=VIEWPORT, locale="en-US")
            try:
                response = await page.goto(app_url, wait_until="domcontentloaded", timeout=25_000)
            except PlaywrightError as exc:
                result["error"] = str(exc).strip().splitlines()[0][:200]
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
    result.update(title=raw["title"].strip(), description=raw["description"], headings=[h for h in headings if h])

    seen_paths = set()
    for link in raw["links"]:
        href, label = link["href"].strip(), link["text"].strip()
        target = urlparse(link["abs"])
        if (not href or not label or '"' in label or href.startswith(("#", "javascript:", "mailto:", "tel:"))
                or target.scheme not in ("http", "https") or target.netloc != origin):
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
        if button["id"]:
            selector = f'[id="{_quote(button["id"])}"]'
        elif label and '"' not in label:
            selector = f'role=button[name="{label}"]'
        else:
            continue
        result["buttons"].append({"text": label or button["id"], "selector": selector})

    for field in raw["inputs"]:
        tag = field["tag"]
        for attr, key in (("id", "id"), ("name", "name"), ("placeholder", "placeholder"), ("aria-label", "aria")):
            if field[key]:
                selector = f'[id="{_quote(field["id"])}"]' if attr == "id" else f'{tag}[{attr}="{_quote(field[key])}"]'
                result["inputs"].append({"label": field["placeholder"] or field["aria"] or field["name"] or field["id"],
                                         "selector": selector})
                break
    return result
