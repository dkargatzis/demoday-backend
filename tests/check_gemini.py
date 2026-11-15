"""Offline checks for the one Gemini entry point (app/services/gemini.py) on a fake client: a malformed reply is asked
for once more (its own ledger row, "(retry)"), two in a row raise, a JSON schema reaches the API with the thinking
level, a schema or thinking level the model rejects falls back to the plain call, and the launch kit's schema.

    .venv/bin/python tests/check_gemini.py
"""
import asyncio
import json
import os
import sys
import tempfile
from types import SimpleNamespace

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the backend root (tests/..)
os.environ.update({"STORAGE_DIR": tempfile.mkdtemp(prefix="demoday-gemini-"), "GEMINI_API_KEY": "offline-test-key"})
sys.path.insert(0, BACKEND)

from app.models.schemas import JobStatus  # noqa: E402
from app.services import gemini, ledger, planner  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + str(detail)[:300] if detail != '' else ''}")


class FakeClient:
    """Stands in for genai.Client: replies come from `script` in order (text, or an exception to raise)."""
    script, configs = [], []

    def __init__(self, api_key=None, http_options=None):
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._generate), aclose=self._close)

    async def _generate(self, model, contents, config):
        FakeClient.configs.append(config)
        reply = FakeClient.script.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(text=reply)

    async def _close(self):
        pass


def run(script, **kwargs):
    FakeClient.script, FakeClient.configs = list(script), []
    job = JobStatus(job_id="g", status="running")
    ledger.bind(job)
    try:
        return asyncio.run(gemini.generate_json("prompt", **kwargs)), job, None
    except Exception as exc:  # noqa: BLE001
        return None, job, exc
    finally:
        ledger.bind(None)


def main():
    real = gemini.genai.Client
    gemini.genai.Client = FakeClient
    try:
        broken = '{"readme_markdown": "# X", "x_thread": ["Say "hi" to it"]}'
        data, job, exc = run([broken, '{"ok": true}'], purpose="launch kit")
        check("a malformed reply is asked for once more, and the second answer is used", data == {"ok": True} and exc is None, exc)
        check("each attempt is its own ledger row, the second labelled (retry)",
              [row.purpose for row in job.tool_calls] == ["launch kit", "launch kit (retry)"], [r.purpose for r in job.tool_calls])
        data, job, exc = run([broken, broken], purpose="launch kit")
        check("two malformed replies raise the parse error (the caller falls back)", isinstance(exc, json.JSONDecodeError), repr(exc))
        data, job, exc = run(["", "```json\n[1, 2]\n```"])
        check("an empty reply is retried too; fenced JSON is accepted", data == [1, 2], exc)

        schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
        data, job, exc = run(['{"a": "x"}'], schema=schema, thinking="low")
        first = FakeClient.configs[0]
        check("the schema and the thinking level reach the API together", data == {"a": "x"} and first.response_json_schema == schema
              and first.thinking_config.thinking_level is not None and first.response_mime_type == "application/json", first)
        data, job, exc = run([RuntimeError("400 INVALID_ARGUMENT: response_json_schema is not supported"),
                              RuntimeError("400 INVALID_ARGUMENT: response_json_schema is not supported"),
                              '{"a": "plain"}'], schema=schema, thinking="low")
        used = FakeClient.configs[-1]
        check("a rejected schema falls back to thinking without it, in one ledger row",
              data == {"a": "plain"} and used.response_json_schema is None and used.thinking_config is not None
              and len(job.tool_calls) == 1, [(c.response_json_schema is not None, c.thinking_config is not None) for c in FakeClient.configs])
        data, job, exc = run([RuntimeError("503 UNAVAILABLE")], schema=schema)
        check("other API errors are not swallowed", isinstance(exc, RuntimeError) and "503" in str(exc) and len(FakeClient.configs) == 1, exc)
        data, job, exc = run(['{"b": 1}'])
        check("no schema, no thinking: one plain JSON config", data == {"b": 1} and len(FakeClient.configs) == 1
              and FakeClient.configs[0].response_json_schema is None and FakeClient.configs[0].thinking_config is None)
    finally:
        gemini.genai.Client = real

    generic, personal = planner.launch_kit_schema(False), planner.launch_kit_schema(True)
    check("launch kit schema: the four kit fields, all required", generic["required"] == ["readme_markdown", "devpost_pitch",
          "x_thread", "linkedin_post"] and generic["properties"]["x_thread"] == {"type": "array", "items": {"type": "string"}})
    check("a personalized kit's schema adds the outreach email and the DM",
          personal["required"][-2:] == ["outreach_email", "linkedin_dm"]
          and personal["properties"]["outreach_email"]["required"] == ["subject", "body"])

    from app.models.audience import AudienceBrief
    from app.models.schemas import Scene, Storyboard
    seen = []

    async def fake_generate_json(prompt, **kwargs):
        seen.append(kwargs)
        return {"readme_markdown": "# X", "devpost_pitch": "## Inspiration", "x_thread": ["one"], "linkedin_post": "post",
                "outreach_email": {"subject": "A tour for Acme", "body": "Hi.\n\n{{DEMO_LINK}}\n\n[Your name]"},
                "linkedin_dm": "Hi {{DEMO_LINK}}"}

    real_planner = planner.generate_json
    planner.generate_json = fake_generate_json
    try:
        storyboard = Storyboard(project_name="X", tagline="t", scenes=[Scene(id="scene-1", title="Hook", route="/", narration="Hello.")])
        repo = {"name": "x", "readme": "", "dependencies": [], "routes": [], "repos": []}
        app = {"url": "https://x.dev", "title": "X", "headings": [], "links": [], "buttons": [], "inputs": [], "pages": []}
        _, generic_note = asyncio.run(planner.generate_launch_kit(storyboard, repo, app, "", "https://x.dev"))
        brief = AudienceBrief(company="Acme", persona="CTO at Acme", source="test")
        kit, personal_note = asyncio.run(planner.generate_launch_kit(storyboard, repo, app, "", "https://x.dev", audience=brief))
    finally:
        planner.generate_json = real_planner
    check("generate_launch_kit asks with the kit's schema (the outreach keys only for a customer)",
          [s.get("schema") for s in seen] == [planner.launch_kit_schema(False), planner.launch_kit_schema(True)]
          and all(s.get("purpose") == "launch kit" for s in seen), [s.get("purpose") for s in seen])
    check("both kits are written by Gemini, the outreach from its answer",
          generic_note.startswith("Launch kit written by Gemini") and personal_note.endswith("outreach email + LinkedIn DM for Acme")
          and kit.outreach_email.subject == "A tour for Acme", (generic_note, personal_note))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


sys.exit(main())
