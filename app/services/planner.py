import os
import json
from google import genai
from google.genai import types
from app.models.schemas import Storyboard, LaunchKit, Scene, Action

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

async def generate_storyboard(repo_info: dict, app_info: dict, target_seconds: int = 60) -> Storyboard:
    """Uses Gemini to plan an exact scene-by-scene demo storyboard."""
    client = get_gemini_client()
    project_name = repo_info.get("name") or app_info.get("title") or "GenAI Project"

    prompt = f"""You are the Demo Day Director. Produce a tight {target_seconds}-second video storyboard for a hackathon pitch demo.
Project Name: {project_name}
Repo Details: {repo_info.get('description', '')}
Dependencies: {', '.join(repo_info.get('dependencies', []))}
Live URL Title: {app_info.get('title', '')}
README Excerpt: {repo_info.get('readme', '')[:2000]}

Rules:
1. Divide demo into exactly 4-5 focused scenes (Hook, Problem/Input, AI Action, Magic Result/Export).
2. Each scene narration must be concise, punchy, conversational spoken English.
3. Actions must specify real Playwright browser steps (navigate, click, type, scroll, wait).
4. Output strictly valid JSON conforming to this schema:
{{
  "project_name": "{project_name}",
  "tagline": "A one-line punchy slogan",
  "scenes": [
    {{
      "id": "scene-1",
      "title": "Opening Hook",
      "route": "/",
      "narration": "What the voiceover says during this scene.",
      "actions": [
        {{"type": "navigate", "value": "/"}},
        {{"type": "wait", "dwell_ms": 1500}}
      ]
    }}
  ]
}}
"""
    if client:
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3
                )
            )
            data = json.loads(response.text)
            return Storyboard(**data)
        except Exception as e:
            print(f"Gemini fallback error: {e}")

    # Fallback deterministic template when API key is unset
    return Storyboard(
        project_name=project_name,
        tagline="Autonomous AI Launch Kit Generator",
        scenes=[
            Scene(
                id="scene-1",
                title="Hero Entry",
                route="/",
                narration=f"Meet {project_name}, built to streamline hackathon launches into a single automated pipeline.",
                actions=[Action(type="navigate", value="/"), Action(type="wait", dwell_ms=1800)]
            ),
            Scene(
                id="scene-2",
                title="Autonomous Input",
                route="/",
                narration="Provide any GitHub repository and deployed URL. The agent scans the route tree, dependencies, and live UI.",
                actions=[
                    Action(type="scroll", value="down", dwell_ms=1500),
                    Action(type="wait", dwell_ms=2000)
                ]
            ),
            Scene(
                id="scene-3",
                title="AI Orchestration",
                route="/",
                narration="Under the hood, Gemini drafts the narrative, fal synthesizes voiceovers, and Playwright records crisp 1080p browser footage.",
                actions=[Action(type="wait", dwell_ms=3000)]
            ),
            Scene(
                id="scene-4",
                title="Finished Deliverable",
                route="/",
                narration="In under 60 seconds, you get a stitched video demo, README documentation, and a complete social launch kit.",
                actions=[Action(type="scroll", value="top", dwell_ms=1500)]
            )
        ]
    )

async def generate_launch_kit(storyboard: Storyboard, repo_info: dict) -> LaunchKit:
    """Generates README, Devpost pitch, X thread, and LinkedIn announcement."""
    client = get_gemini_client()
    name = storyboard.project_name
    tagline = storyboard.tagline

    prompt = f"""Generate a complete Launch Kit for {name}: {tagline}.
Based on these demo scenes: {json.dumps([s.model_dump() for s in storyboard.scenes])}
Return strictly JSON:
{{
  "readme_markdown": "# Markdown with badges, features, and quickstart",
  "devpost_pitch": "### Inspiration\n### What it does\n### How we built it\n### Challenges\n### Accomplishments",
  "x_thread": ["Tweet 1 hook with link", "Tweet 2 technical architecture", "Tweet 3 demo callout"],
  "linkedin_post": "Engaging professional founder announcement with emojis and bullet points"
}}
"""
    if client:
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.4
                )
            )
            data = json.loads(response.text)
            return LaunchKit(**data)
        except Exception as e:
            print(f"Gemini launch kit fallback error: {e}")

    return LaunchKit(
        readme_markdown=f"""# {name} 🚀\n\n> {tagline}\n\n## Overview\n{name} autonomously navigates your web application, captures synchronized video footage, generates voiceovers, and prepares complete launch documentation.\n\n## Architecture\n- **Backend**: FastAPI, Pydantic AI, Playwright, FFmpeg\n- **Voice & Media**: fal.ai credits & Gemini 2.0\n- **Frontend**: Next.js App Router\n\n## Getting Started\n```bash\ngit clone <repo>\ncd demoday-backend && uvicorn app.main:app --reload\n```""",
        devpost_pitch=f"""### Inspiration\nHackathon teams spend 48 hours building groundbreaking tech, then scramble in the last 15 minutes to record buggy screen captures and hasty pitch decks. {name} turns that painful final hour into a 45-second automated task.\n\n### What it does\nGiven a repository and deployed URL, it extracts product context, drafts a storyboard, drives headless Playwright to record interactions, records fal.ai voiceovers, and muxes an MP4 demo.\n\n### How we built it\nFastAPI worker with headless Chromium and FFmpeg in Docker, Gemini 2.0 Flash for structured planning, and fal.ai for low-latency voice synthesis.""",
        x_thread=[
            f"1/ Excited to launch {name}: {tagline}! 🎥\n\nStop wasting your final hackathon hour recording broken demos. Point {name} at your repo and get a complete launch kit in 60s.",
            "2/ Powered by Pydantic AI, Gemini 2.0 for scene planning, Playwright for deterministic headless browser capture, and fal.ai for frame-accurate voiceovers.",
            "3/ Check out the live demo and full open-source repo: [link]"
        ],
        linkedin_post=f"We're thrilled to introduce {name}: {tagline}! 🎉\n\nEvery founder and hackathon participant knows the dread of recording last-minute demo videos. {name} automates the entire loop: browser interaction, voice synthesis, video muxing, and documentation generation.\n\nBuilt with FastAPI, Playwright, fal.ai, and Next.js."
    )
