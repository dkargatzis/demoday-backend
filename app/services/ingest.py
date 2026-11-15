"""Context gathering: GitHub repositories (public or private) and a rendered inventory of the live app."""
import asyncio
import json
import logging
import os
import re
import time
import tomllib
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set
from urllib.parse import urlparse

import httpx
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright

from app.config import GITHUB_API_URL, GITHUB_RAW_URL, get_secret
from app.services.netguard import guard_context
from app.services.recorder import (VIEWPORT, declutter_page, is_risky, launch_chromium, session_storage_script,
                                   split_session_storage, wait_for_loaders)

logger = logging.getLogger("demoday.ingest")

GITHUB_API = GITHUB_API_URL
RAW_GITHUB = GITHUB_RAW_URL
MANIFESTS = ("package.json", "requirements.txt", "pyproject.toml")
MAX_REPOS = 5
BRIEF_MAX_PAGES = int(os.getenv("BRIEF_MAX_PAGES", "6"))  # pages explored beyond the landing page while preparing the brief
BRIEF_BUDGET_SEC = 25   # ...within this much time
FOCUS_EXTRA_PAGES = 2   # with the founder's steps, room to follow their links one level deeper
_FOCUS_STOPWORDS = frozenset("the and then open show page pages click its with from into onto scroll finish start the "
                             "for that this them there where what when your our you a an to of in on at by it is be".split())


def focus_words(text: Optional[str]) -> Set[str]:
    """The words of the founder's steps that can name a link ('Tutorial - User Guide' -> tutorial, user, guide)."""
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) >= 3 and w not in _FOCUS_STOPWORDS}


def _focus_score(label: str, route: str, words: Set[str]) -> int:
    """How many of the founder's words a link carries in its text or path."""
    if not words:
        return 0
    return len(words & set(re.findall(r"[a-z0-9]+", f"{label} {route}".lower())))
SHOT_TIMEOUT_MS = 8000  # a page screenshot for the vision brief never eats more of the budget than this
MAX_INVENTORY_LINKS = 40


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
        merged["flow_files"] = _unique(f for info in infos for f in info.get("flow_files", []))[:30]
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
        info["flow_files"] = flow_files(paths)
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


# Files whose names say "a user goes through this": the code-side evidence for the product's funnels.
_FLOW_WORDS = re.compile(r"onboard|sign-?up|register|welcome|get-?started|getting-?started|setup|connect|integrat|install|"
                         r"checkout|pricing|billing|subscri|upgrade|trial|invite|import|wizard|first-?run|activat|"
                         r"quick-?start|new-?project|create", re.IGNORECASE)
_SOURCE_FILE = re.compile(r"\.(tsx?|jsx?|vue|svelte|py|rb|go|php|ex|kt|swift)$")
_NOT_PRODUCT = re.compile(r"(^|/)(tests?|__tests__|spec|e2e|cypress|playwright|node_modules|dist|build|vendor|migrations|"
                          r"fixtures|mocks?|stories|\.github|scripts|docs?)/|\.(test|spec|stories|d)\.", re.IGNORECASE)


def flow_files(paths: List[str], limit: int = 25) -> List[str]:
    """Source files named after a user flow (onboarding, setup, integrations, checkout, invites...), shortest paths
    first: what the funnel analysis reads as the code's evidence of how users reach the product's outcome."""
    found = [p for p in paths if _SOURCE_FILE.search(p) and not _NOT_PRODUCT.search(p) and _FLOW_WORDS.search(p)]
    return sorted(found, key=lambda p: (p.count("/"), len(p)))[:limit]


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
    siteName: meta("og:site_name") || meta("application-name") || meta("apple-mobile-web-app-title"),
    themeColor: meta("theme-color"),
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
# Account pages are never worth a scene: the brief doesn't explore them and the planner isn't offered them as routes.
_ACCOUNT_PATH = re.compile(r"/(sign-?in|log-?in|sign-?up|register|auth|oauth|password|verify|account|settings|billing)(/|$|\?)",
                           re.IGNORECASE)


def _stable_id(value: str) -> bool:
    return bool(value) and not _GENERATED_ID.search(value)


def _scope_section(url: str) -> str:
    """'/dashboard' for a demo that targets a page deep inside the app; '' for a landing page."""
    path = urlparse(url).path.rstrip("/")
    return "/" + path.strip("/").split("/")[0] if path.count("/") >= 1 and path else ""


def _link_entries(raw_links: List[Dict], origin: str, seen_paths: set, on: Optional[str] = None) -> List[Dict]:
    """Inventory entries for the planner: same-origin, labelled, non-destructive links, one per path."""
    entries = []
    for link in raw_links:
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
        entry = {"text": label, "path": path, "selector": f'a[href="{_quote(href)}"]', "nav": bool(link.get("nav"))}
        if on:
            entry["on"] = on
        entries.append(entry)
    return entries


async def _screenshot(page, shots_dir: Optional[Path], index: int) -> Optional[str]:
    """Viewport JPEG of the page as the recording will show it (page_00 is the landing page), for the Gemini vision
    brief. Returns the file path, or None when no `shots_dir` was given or the capture failed."""
    if shots_dir is None:
        return None
    path = Path(shots_dir) / f"page_{index:02d}.jpg"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(path), type="jpeg", quality=50, timeout=SHOT_TIMEOUT_MS)
    except (PlaywrightError, OSError) as exc:
        logger.warning("screenshot %s failed: %s", path.name, (str(exc).strip() or type(exc).__name__).splitlines()[0][:120])
        return None
    return str(path)


async def _explore_pages(page, landing_url: str, raw_links: List[Dict], shots_dir: Optional[Path] = None,
                         on_page: Optional[Callable[[Dict], None]] = None,
                         focus: Optional[str] = None) -> List[Dict[str, Any]]:
    """Visits the app's own pages (navigation links first, inside the demo's section) so the brief can say what
    each one shows. Bounded by BRIEF_MAX_PAGES and BRIEF_BUDGET_SEC.

    With `shots_dir`, each page is also screenshotted (page_01.jpg, ...) for the Gemini vision brief and the entry
    gains `screenshot`; `on_page(entry)` is called after each page so the job log can report it live.

    With `focus` (the founder's steps), links whose text or path carry its words are visited first, and the matching
    links on those pages are followed one level deeper (FOCUS_EXTRA_PAGES more pages, same time budget), so a step
    like "open the Tutorial, then First Steps" reaches a page the navigation alone would not."""
    origin = urlparse(landing_url).netloc
    landing_path = urlparse(landing_url).path.rstrip("/") or "/"
    section = _scope_section(landing_url)
    words = focus_words(focus)
    seen = {landing_path}

    def candidates_from(links: List[Dict]) -> List[tuple]:
        found = []
        for link in links:
            href, label = link["href"].strip(), link["text"].strip()
            target = urlparse(link["abs"])
            path = target.path.rstrip("/") or "/"
            if (not href or not label or href.startswith(("#", "javascript:", "mailto:", "tel:"))
                    or target.scheme not in ("http", "https") or target.netloc != origin or path in seen
                    or is_risky(label=label, url=href) or (section and not path.startswith(section))
                    or _ACCOUNT_PATH.search(path)):
                continue
            seen.add(path)
            route = path + ("?" + target.query if target.query else "")
            found.append((link["abs"], route, href, label, _focus_score(label, route, words), bool(link.get("nav"))))
        return found

    queue = sorted(candidates_from(raw_links), key=lambda c: (-c[4], not c[5]))  # the founder's words, then navigation
    limit = BRIEF_MAX_PAGES + (FOCUS_EXTRA_PAGES if words else 0)
    pages: List[Dict[str, Any]] = []
    started = time.monotonic()
    while queue:
        abs_url, route, href, label, score, _nav = queue.pop(0)
        if len(pages) >= limit or time.monotonic() - started > BRIEF_BUDGET_SEC:
            break
        try:
            await page.goto(abs_url, wait_until="domcontentloaded", timeout=15_000)
            try:
                await page.wait_for_load_state("networkidle", timeout=3000)
            except PlaywrightError:
                pass
            await wait_for_loaders(page)
            await declutter_page(page)
            info = await page.evaluate(INVENTORY_SCRIPT)
        except PlaywrightError:
            continue
        landed = urlparse(page.url).path.rstrip("/") or "/"
        if landed == landing_path or any(landed == urlparse(p["url"]).path.rstrip("/") for p in pages):
            continue  # bounced back (a sign-in wall or an alias of a page already seen)
        headings = [re.sub(r"[\s¶#🔗]+$", "", h) for h in info["headings"]]
        pages.append({
            "path": route, "url": page.url, "title": info["title"].strip(), "link_text": label,
            "open_selector": f'a[href="{_quote(href)}"]',
            "headings": [h for h in headings if h][:6],
            "buttons": [b["text"].strip() for b in info["buttons"] if b["text"].strip() and not is_risky(label=b["text"])][:6],
            "inputs": [(i["placeholder"] or i["aria"] or i["name"]) for i in info["inputs"]
                       if (i["placeholder"] or i["aria"] or i["name"])][:4],
            "raw_links": info["links"],
        })
        shot = await _screenshot(page, shots_dir, len(pages))
        if shot:
            pages[-1]["screenshot"] = shot
        if score:  # a page the founder asked for: its own matching links come next (one level deeper)
            deeper = [c for c in candidates_from(info["links"]) if c[4]]
            queue[:0] = sorted(deeper, key=lambda c: -c[4])
        if on_page is not None:
            try:
                on_page(pages[-1])
            except Exception as exc:  # a log callback never stops the exploration
                logger.warning("on_page callback failed: %s", exc)
    return pages


async def probe_app_url(app_url: str, storage_state: Optional[Dict] = None, replayer=None, *,
                        shots_dir: Optional[Path] = None,
                        on_page: Optional[Callable[[Dict], None]] = None,
                        focus: Optional[str] = None) -> Dict[str, Any]:
    """Renders the app in headless Chromium (so SPAs work) and inventories what a demo could interact with.

    `storage_state` (a saved sign-in session) probes the app as the signed-in user. `login_form` reports a visible
    password field, which usually means a sign-in wall. Returns an `error` key when the page can't be loaded at all.
    With `shots_dir`, the landing page is saved as page_00.jpg (`screenshot`) and every explored page as
    page_NN.jpg (`pages[i]['screenshot']`) for the Gemini vision brief; `on_page` is called per explored page.
    """
    result: Dict[str, Any] = {"url": app_url, "status": 0, "title": "", "description": "", "headings": [],
                              "links": [], "buttons": [], "inputs": [], "login_form": False, "pages": []}
    async with async_playwright() as pw:
        browser = await launch_chromium(pw)
        try:
            playwright_state, session_storage = split_session_storage(storage_state)
            context = await browser.new_context(viewport=VIEWPORT, locale="en-US", storage_state=playwright_state,
                                                service_workers="block")
            await guard_context(context)
            if session_storage:
                await context.add_init_script(session_storage_script(session_storage))
            if replayer is not None:
                await replayer.install(context)
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
            await wait_for_loaders(page)
            await declutter_page(page)  # inventory the page as the recording will show it
            result["url"] = page.url
            raw = await page.evaluate(INVENTORY_SCRIPT)
            shot = await _screenshot(page, shots_dir, 0)
            if shot:
                result["screenshot"] = shot
            pages = [] if raw.get("hasPassword") else await _explore_pages(page, result["url"], raw["links"],
                                                                            shots_dir=shots_dir, on_page=on_page,
                                                                            focus=focus)
        finally:
            await browser.close()

    origin = urlparse(result["url"]).netloc
    headings = [re.sub(r"[\s¶#🔗]+$", "", h) for h in raw["headings"]]  # drop permalink glyphs
    result.update(title=raw["title"].strip(), description=raw["description"], headings=[h for h in headings if h],
                  login_form=bool(raw.get("hasPassword")), site_name=(raw.get("siteName") or "").strip()[:60],
                  theme_color=(raw.get("themeColor") or "").strip()[:30])

    seen_paths: set = set()
    result["links"] = _link_entries(raw["links"], origin, seen_paths)[:30]
    for explored in pages:  # links that only exist deeper in the app, so the planner can chain pages
        for entry in _link_entries(explored.pop("raw_links", []), origin, seen_paths, on=explored["path"]):
            if len(result["links"]) >= MAX_INVENTORY_LINKS:
                break
            result["links"].append(entry)
    result["pages"] = pages

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
