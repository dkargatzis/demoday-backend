"""Gemini planning: the scene-by-scene storyboard and the launch collateral, with template fallbacks."""
import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from google import genai
from google.genai import types

from app.config import GEMINI_MODEL, get_secret
from app.models.schemas import Action, LaunchKit, Scene, Storyboard

logger = logging.getLogger("demoday.planner")

GEMINI_TIMEOUT_SEC = 90
# Spoken words per second per voice preset (audio.VOICE_PRESETS), measured on real Gemini narration read by
# fal.ai Kokoro: sentence breaks and product names make it slower than plain prose. Erring low keeps videos
# at or under the requested length.
WORDS_PER_SECOND = {"energetic_tech": 2.0, "calm_expert": 1.7, "fast_founder": 2.1}
TAIL_ALLOWANCE_SEC = 2.0  # ~5 scenes x 0.4 s of breathing room added after each narration line
BOOKEND_ALLOWANCE_SEC = 9.0  # AI intro + outro, about 4.5 s each
ALLOWED_ACTIONS = {"navigate", "click", "type", "scroll", "wait"}
TONES = {
    "energetic_tech": "high-energy tech founder: upbeat, confident, crisp",
    "calm_expert": "calm technical expert: clear, measured, authoritative",
    "fast_founder": "rapid-fire hackathon pitch: short sentences, bold claims backed by real features",
}


# --------------------------------------------------------------------------- Gemini plumbing


async def _generate_json(prompt: str) -> Any:
    key = get_secret("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not configured")
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_SEC * 1000))
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            ),
            timeout=GEMINI_TIMEOUT_SEC + 10,
        )
    finally:
        await client.aio.aclose()
    text = (response.text or "").strip()
    if not text:
        raise ValueError("Gemini returned an empty response")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)  # tolerate fenced output
    # strict=False accepts raw newlines/tabs inside string values, which Gemini sometimes emits in long Markdown fields.
    return json.loads(text, strict=False)


def _failure_reason(exc: Exception) -> str:
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return message[:160]


# --------------------------------------------------------------------------- shared context


def project_name(repo_info: dict, app_info: dict) -> str:
    """Display name: the app title's leading segment when it names the repo as a whole word
    (repo "ui" + "shadcn/ui - The Foundation..." -> shadcn/ui; repo "fastapi" + "FastAPI - FastAPI" -> FastAPI),
    otherwise the repo name."""
    name = (repo_info.get("name") or "").strip()
    title = (app_info.get("title") or "").strip()
    lead = re.split(r"\s+[-|–—:]\s+", title, maxsplit=1)[0].strip()
    if name and lead and len(lead) <= 40:
        whole_word = r"(?<![A-Za-z0-9])" + re.escape(name.replace("-", " ")) + r"(?![A-Za-z0-9])"
        if re.search(whole_word, lead.replace("-", " "), re.IGNORECASE):
            return lead
    return name or lead or "This project"


def _summary(repo_info: dict, app_info: dict) -> str:
    text = (repo_info.get("description") or app_info.get("description") or "").strip()
    if not text:
        return ""
    first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0].strip()
    return first if first.endswith((".", "!", "?")) else first + "."


def _repo_urls(repo_info: dict, fallback: str = "") -> List[str]:
    urls = [repo["url"] for repo in repo_info.get("repos", []) if repo.get("url")]
    return urls or [u.strip() for u in (fallback or "").split(",") if u.strip()]


def _repo_lines(repo_info: dict, fallback: str = "") -> str:
    repos = repo_info.get("repos") or [{"url": u, "description": repo_info.get("description", "")}
                                       for u in _repo_urls(repo_info, fallback)]
    lines = [f"- Repository: {r['url']}" + (f" - {r['description']}" if r.get("description") else "") for r in repos]
    return "\n".join(lines) or "- Repository: none given"


def _pick_nav_link(links: List[dict]) -> Optional[dict]:
    """A primary-navigation link worth clicking on camera (skips home links and language switchers)."""
    def usable(link: dict) -> bool:
        path = link["path"].split("?")[0]
        return (path not in ("/", "") and 3 <= len(link["text"]) <= 30
                and not re.fullmatch(r"/[a-z]{2}(?:[-_][a-z]{2,4})?/?", path, re.IGNORECASE))
    candidates = [link for link in links if usable(link)]
    return next((link for link in candidates if link.get("nav")), candidates[0] if candidates else None)


def _readme_features(readme: str, limit: int = 5) -> List[str]:
    """The first bullet list after a line mentioning features (or the first bullet list at all)."""
    lines = readme.splitlines()
    start = next((i + 1 for i, line in enumerate(lines) if re.search(r"feature", line, re.IGNORECASE)), 0)
    bullets: List[str] = []
    for line in lines[start:]:
        bullet = re.match(r"^\s*[-*+]\s+(.*)", line)
        if bullet:
            text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", bullet.group(1))
            text = re.sub(r"[*_`]", "", text).strip()
            if 10 <= len(text) <= 220:
                bullets.append(text)
            if len(bullets) >= limit:
                break
        elif bullets and line.strip() and not line.startswith((" ", "\t")):
            break  # first list ended
    return bullets


def _clean_narration(text: Any) -> str:
    # TTS spells full URLs out character by character; say just the domain instead.
    text = re.sub(r"https?://(?:www\.)?([^/\s]+)\S*", r"\1", str(text or ""))
    text = re.sub(r"[*_#`>|]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _scope_rule(app_info: dict) -> str:
    """When the demo targets a page deep inside the app (a dashboard), keep the storyboard there."""
    path = urlparse(app_info.get("url") or "").path.rstrip("/")
    if path.count("/") < 1 or not path:
        return ""
    section = "/" + path.strip("/").split("/")[0]
    return (f" The demo is about the app section under {section}: stay on pages whose path starts with {section} "
            f"and never visit marketing pages (home, pricing, how-it-works, sign-in).")


def _inventory_lines(app_info: dict) -> str:
    lines = []
    for link in app_info.get("links", [])[:25]:
        lines.append(f'- link "{link["text"]}" -> path {link["path"]} | selector: {link["selector"]}')
    for button in app_info.get("buttons", [])[:12]:
        lines.append(f'- button "{button["text"]}" | selector: {button["selector"]}')
    for field in app_info.get("inputs", [])[:8]:
        lines.append(f'- input "{field["label"]}" | selector: {field["selector"]}')
    return "\n".join(lines) or "- (no interactive elements detected; use scroll and wait)"


# --------------------------------------------------------------------------- storyboard


async def generate_storyboard(
    repo_info: dict, app_info: dict, target_seconds: int = 60, voice_style: str = "energetic_tech",
    repo_url: str = "", ai_clips: bool = False,
) -> Tuple[Storyboard, str]:
    """Returns (storyboard, human-readable note on which planner produced it).

    With `ai_clips`, the intro/outro take part of the runtime and Gemini also describes the shots for them.
    """
    name = project_name(repo_info, app_info)
    narrated_sec = target_seconds - TAIL_ALLOWANCE_SEC - (BOOKEND_ALLOWANCE_SEC if ai_clips else 0)
    total_words = max(30, int(narrated_sec * WORDS_PER_SECOND.get(voice_style, 2.4)))
    bookend_rule = bookend_json = ""
    if ai_clips:
        bookend_rule = f"""
7. The video opens with an intro card that already introduces {name} by name and closes with an outro card showing
   the URL, so scene 1 starts with the problem and the last scene doesn't repeat the URL. Also write "intro_visual"
   and "outro_visual": one or two sentences each describing a 5-second cinematic shot that an AI video model will
   film behind those cards. Evoke the product's domain with abstract or atmospheric imagery and slow camera motion.
   They must contain no text, letters, logos, screens, user interfaces or people."""
        bookend_json = ' "intro_visual": "...", "outro_visual": "...",'
    prompt = f"""You are directing a {target_seconds}-second product demo video for a hackathon submission.
A headless browser (1920x1080) performs your actions on the LIVE app while a voiceover reads your narration.

PROJECT
- Name: {name}
{_repo_lines(repo_info, repo_url)}
- Language: {repo_info.get('language', '')}; dependencies: {', '.join(repo_info.get('dependencies', [])[:20])}
- Live app: {app_info.get('url', '')} - title "{app_info.get('title', '')}"; description: "{app_info.get('description', '')}"
- Headings on the landing page: {' | '.join(app_info.get('headings', [])[:8])}
- Page routes found in the repository: {', '.join(repo_info.get('routes', [])) or 'none detected'}

README EXCERPT
<<<
{repo_info.get('readme', '')[:3500]}
>>>

INTERACTIVE ELEMENTS ON THE LIVE LANDING PAGE (selectors are verified, use them verbatim)
{_inventory_lines(app_info)}

RULES
1. Write 4-5 scenes: hook, the problem or input, the core feature in action, the payoff, a closing call to action.
2. Narration is spoken aloud: plain conversational English, no emojis, markdown, URLs or code.
   About {total_words} words in total (roughly {total_words // 5} per scene). Tone: {TONES.get(voice_style, TONES['energetic_tech'])}.
3. "route" is the path where a scene starts; "/" means the app page itself ({app_info.get('url', '')}). Use only "/",
   paths from the list above, or repository routes. The browser keeps its state between scenes, so a scene can
   continue where the last one ended.{_scope_rule(app_info)}
4. Actions (each may set "dwell_ms", the pause after it, 500-4000):
   {{"type": "navigate", "value": "/path"}}
   {{"type": "click", "selector": "<selector from the list>"}}
   {{"type": "type", "selector": "<input selector>", "value": "text to type"}}  (end value with \\n to press Enter)
   {{"type": "scroll", "value": "down" | "up" | "top" | "bottom"}}
   {{"type": "wait"}}
   Only use selectors from the list. When unsure, scroll or wait instead of guessing.
   Prefer real interactions over scrolling: when the inventory offers navigation links, buttons or inputs, at least
   two scenes should click or type (the camera zooms toward every click, which is what makes the video feel alive).
   Keep each scene's total dwell close to its narration time (about 400 ms per word): the scene ends when its
   narration ends, and later actions are skipped.
5. The browser may be signed in to the user's real account: never sign out, delete, remove, cancel, purchase,
   or change settings. Show, don't modify.
6. Respond with JSON only:
{{"project_name": "{name}", "tagline": "one punchy line",{bookend_json} "scenes": [{{"id": "scene-1", "title": "...",
  "route": "/", "narration": "...", "actions": [{{"type": "scroll", "value": "down", "dwell_ms": 1500}}]}}]}}{bookend_rule}
"""
    if get_secret("GEMINI_API_KEY"):
        try:
            storyboard = _sanitize_storyboard(await _generate_json(prompt), name)
            return storyboard, f"Storyboard planned by Gemini ({GEMINI_MODEL})"
        except Exception as exc:
            logger.warning("Gemini storyboard failed: %s", exc)
            reason = f"Gemini ({GEMINI_MODEL}) failed: {_failure_reason(exc)}"
    else:
        reason = "GEMINI_API_KEY not set"
    return _template_storyboard(repo_info, app_info), f"{reason}: template storyboard fallback"


def _sanitize_storyboard(data: Any, fallback_name: str) -> Storyboard:
    if isinstance(data, list):
        data = {"scenes": data}
    if not isinstance(data, dict):
        raise ValueError("storyboard is not a JSON object")
    if "scenes" not in data:  # unwrap {"storyboard": {...}}
        data = next((v for v in data.values() if isinstance(v, dict) and "scenes" in v), data)
    scenes: List[Scene] = []
    for raw in data.get("scenes") or []:
        if not isinstance(raw, dict):
            continue
        narration = _clean_narration(raw.get("narration"))
        if not narration:
            continue
        actions = [a for a in (_sanitize_action(x) for x in raw.get("actions") or []) if a]
        route = str(raw.get("route") or "/").strip() or "/"
        scenes.append(Scene(
            id=f"scene-{len(scenes) + 1}",
            title=str(raw.get("title") or f"Scene {len(scenes) + 1}")[:80],
            route=route,
            narration=narration[:600],
            actions=actions[:8],
        ))
    if not scenes:
        raise ValueError("storyboard contained no usable scenes")
    return Storyboard(
        project_name=str(data.get("project_name") or fallback_name)[:80],
        tagline=_clean_narration(data.get("tagline"))[:140],
        scenes=scenes[:6],
        intro_visual=_clean_narration(data.get("intro_visual"))[:400] or None,
        outro_visual=_clean_narration(data.get("outro_visual"))[:400] or None,
    )


def _sanitize_action(raw: Any) -> Optional[Action]:
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("type", "")).strip().lower()
    if kind not in ALLOWED_ACTIONS:
        return None
    selector = str(raw["selector"]).strip() if raw.get("selector") not in (None, "") else None
    value = str(raw["value"]) if raw.get("value") not in (None, "") else None
    if (kind == "click" and not selector) or (kind == "type" and not (selector and value)) \
            or (kind == "navigate" and not value):
        return None
    try:
        dwell = int(float(raw.get("dwell_ms", 1500)))
    except (TypeError, ValueError):
        dwell = 1500
    return Action(type=kind, selector=selector, value=value, dwell_ms=max(0, min(dwell, 8000)))


def _template_storyboard(repo_info: dict, app_info: dict) -> Storyboard:
    """Deterministic, project-specific storyboard used when Gemini is unavailable."""
    name = project_name(repo_info, app_info)
    summary = _summary(repo_info, app_info)
    nav = _pick_nav_link(app_info.get("links", []))
    scenes = [
        Scene(id="scene-1", title="Hook", route="/",
              narration=f"Meet {name}. {summary}".strip(),
              actions=[Action(type="wait", dwell_ms=1200), Action(type="scroll", value="down", dwell_ms=1500)]),
        Scene(id="scene-2", title="A closer look", route="/",
              narration="Here is the live app, running in production right now. Let's scroll through what it offers.",
              actions=[Action(type="scroll", value="down", dwell_ms=1800),
                       Action(type="scroll", value="down", dwell_ms=1800)]),
    ]
    if nav:
        scenes.append(Scene(id="scene-3", title=nav["text"], route="/",
                            narration=f"Let's jump into {nav['text']} and see how it works.",
                            actions=[Action(type="scroll", value="top", dwell_ms=900),
                                     Action(type="click", selector=nav["selector"], dwell_ms=2200),
                                     Action(type="scroll", value="down", dwell_ms=1500)]))
    scenes.append(Scene(id=f"scene-{len(scenes) + 1}", title="Call to action", route="/",
                        narration=f"That's {name}. Check out the repository, give it a star, and try it yourself.",
                        actions=[Action(type="wait", dwell_ms=1500)]))
    tagline = summary.rstrip(".") if summary else f"{name}, live and ready to try"
    return Storyboard(project_name=name, tagline=tagline[:140], scenes=scenes)


# --------------------------------------------------------------------------- launch kit


async def generate_launch_kit(
    storyboard: Storyboard, repo_info: dict, app_info: dict, repo_url: str, app_url: str,
) -> Tuple[LaunchKit, str]:
    """Returns (launch kit, human-readable note on which generator produced it)."""
    name, tagline = storyboard.project_name, storyboard.tagline
    narration = "\n".join(f"{i + 1}. {s.title}: {s.narration}" for i, s in enumerate(storyboard.scenes))
    prompt = f"""Write launch collateral for a hackathon project. Be specific to THIS project and use only facts from
the context below; never invent metrics, users, or features.

PROJECT: {name} - {tagline}
Repositories: {', '.join(_repo_urls(repo_info, repo_url)) or 'none given'}
Live demo: {app_url}
Description: {repo_info.get('description', '') or app_info.get('description', '')}
Language: {repo_info.get('language', '')}; dependencies: {', '.join(repo_info.get('dependencies', [])[:20])}
README excerpt:
<<<
{repo_info.get('readme', '')[:3000]}
>>>
Demo video narration:
{narration}

Return JSON with exactly these keys:
- "readme_markdown": a polished GitHub README in Markdown: title and tagline, links to the live demo and repo,
  features, how it works, tech stack, and a quickstart using only commands evident from the context.
- "devpost_pitch": Markdown with the sections "## Inspiration", "## What it does", "## How we built it",
  "## Challenges we ran into", "## Accomplishments that we're proud of", "## What we learned", "## What's next".
- "x_thread": an array of 4-6 tweets, each under 270 characters; the first is the hook; include the live demo and
  repo links where natural; at most two hashtags in total.
- "linkedin_post": a professional announcement of 120-200 words in short paragraphs.
"""
    if get_secret("GEMINI_API_KEY"):
        try:
            return _sanitize_launch_kit(await _generate_json(prompt)), f"Launch kit written by Gemini ({GEMINI_MODEL})"
        except Exception as exc:
            logger.warning("Gemini launch kit failed: %s", exc)
            reason = f"Gemini ({GEMINI_MODEL}) failed: {_failure_reason(exc)}"
    else:
        reason = "GEMINI_API_KEY not set"
    kit = _template_launch_kit(storyboard, repo_info, app_info, repo_url, app_url)
    return kit, f"{reason}: template launch kit fallback"


def _sanitize_launch_kit(data: Any) -> LaunchKit:
    if not isinstance(data, dict):
        raise ValueError("launch kit is not a JSON object")
    thread = data.get("x_thread") or []
    if isinstance(thread, str):
        thread = [t.strip() for t in re.split(r"\n\s*\n", thread) if t.strip()]
    thread = [str(t).strip()[:280] for t in thread if str(t).strip()]
    kit = LaunchKit(
        readme_markdown=str(data.get("readme_markdown") or "").strip(),
        devpost_pitch=str(data.get("devpost_pitch") or "").strip(),
        x_thread=thread,
        linkedin_post=str(data.get("linkedin_post") or "").strip(),
    )
    if not (kit.readme_markdown and kit.devpost_pitch and kit.x_thread and kit.linkedin_post):
        raise ValueError("launch kit is missing fields")
    return kit


def _template_launch_kit(storyboard: Storyboard, repo_info: dict, app_info: dict,
                         repo_url: str, app_url: str) -> LaunchKit:
    name, tagline = storyboard.project_name, storyboard.tagline
    description = (repo_info.get("description") or app_info.get("description") or tagline).strip()
    deps = repo_info.get("dependencies", [])[:10]
    stack = ", ".join(([repo_info["language"]] if repo_info.get("language") else []) + deps) or "see the repository"
    highlights = _readme_features(repo_info.get("readme", "")) \
        or [h for h in app_info.get("headings", []) if 3 <= len(h) <= 80][:5]
    features = "\n".join(f"- {h}" for h in highlights) or f"- {description}"
    host = urlparse(app_url).netloc or app_url
    urls = _repo_urls(repo_info, repo_url)
    repo_url = urls[0] if urls else app_url  # primary link for posts
    clone = "".join(f"git clone {u}.git\n" for u in urls)
    readme = (
        f"# {name}\n\n> {tagline}\n\n"
        f"**Live demo:** {app_url}  \n**Source:** {', '.join(urls) or 'not public'}\n\n"
        f"## Overview\n{description}\n\n## Highlights\n{features}\n\n"
        f"## Tech stack\n{stack}\n\n"
        + (f"## Getting started\n```bash\n{clone}```\n"
           f"Then follow the setup steps in the repositories to run {name} locally.\n" if urls else "")
    )
    pitch = (
        f"## Inspiration\nWe wanted to build something people could pick up and use immediately. {name} is the result.\n\n"
        f"## What it does\n{description}\n\n{features}\n\n"
        f"## How we built it\nBuilt with {stack}, and deployed live at {host}.\n\n"
        f"## Challenges we ran into\nKeeping the experience fast and simple while shipping on a hackathon clock.\n\n"
        f"## Accomplishments that we're proud of\nA working product you can try right now at {app_url}.\n\n"
        f"## What we learned\nTight scope, real users, and a live demo beat slideware every time.\n\n"
        f"## What's next\nMore features, more feedback, and opening it up to more people. Follow along at {repo_url}."
    )
    thread = [
        f"Introducing {name}: {tagline} 🚀",
        f"What it does: {description}"[:270],
        f"Built with {stack}."[:270],
        f"Try the live demo: {app_url}",
        f"Star the repo and follow along: {repo_url} #buildinpublic",
    ]
    linkedin = (
        f"Excited to share {name}: {tagline}.\n\n{description}\n\n"
        f"It's live now at {app_url}" + (f", and the code is open at {', '.join(urls)}" if urls else "") + ".\n\n"
        f"Built with {stack}. Feedback is very welcome!"
    )
    return LaunchKit(readme_markdown=readme, devpost_pitch=pitch, x_thread=thread, linkedin_post=linkedin)
