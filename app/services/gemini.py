"""One Gemini entry point: JSON generation with optional inline images, deep thinking and the tool ledger.

Every Gemini call in the pipeline goes through `generate_json`, so the ledger sees all of them and the retry and
parsing rules live in one place.
"""
import asyncio
import json
import logging
import re
from typing import Any, List, Optional, Sequence

from google import genai
from google.genai import types

from app.config import GEMINI_MODEL, get_secret
from app.services import ledger

logger = logging.getLogger("demoday.gemini")
GEMINI_TIMEOUT_SEC = 90


def _parse(text: str) -> Any:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())  # tolerate fenced output
    if not text:
        raise ValueError("Gemini returned an empty response")
    # strict=False accepts raw newlines/tabs inside string values, which Gemini sometimes emits in long fields.
    return json.loads(text, strict=False)


def _configs(level: Optional[str], schema: Optional[dict]) -> List[types.GenerateContentConfig]:
    """Most specific first: thinking + schema, schema, thinking, plain. A model (or API version) that rejects one of
    them falls back to the next instead of failing the call."""
    out = []
    for with_schema in ((True, False) if schema else (False,)):
        for with_thinking in ((True, False) if level else (False,)):
            fields: dict = {"response_mime_type": "application/json"}
            if with_schema:
                fields["response_json_schema"] = schema
            if with_thinking:
                fields["thinking_config"] = types.ThinkingConfig(thinking_level=level)
            out.append(types.GenerateContentConfig(**fields))
    return out


async def _generate_text(key: str, contents: list, configs: List[types.GenerateContentConfig], timeout_sec: int,
                         purpose: str) -> str:
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=timeout_sec * 1000))
    try:
        async with ledger.call("gemini", purpose, GEMINI_MODEL):
            for config in configs:
                try:
                    response = await asyncio.wait_for(
                        client.aio.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config),
                        timeout=timeout_sec + 10,
                    )
                    break
                except Exception as exc:  # a config the model rejects (thinking levels, a schema): try the next one
                    message = str(exc).lower()
                    if config is configs[-1] or not any(word in message for word in ("thinking", "schema")):
                        raise
    finally:
        await client.aio.aclose()
    return response.text or ""


async def generate_json(prompt: str, *, images: Sequence[bytes] = (), deep: bool = False,
                        thinking: Optional[str] = None, timeout_sec: int = GEMINI_TIMEOUT_SEC,
                        purpose: str = "gemini", schema: Optional[dict] = None) -> Any:
    """Parsed JSON from Gemini for `prompt`, with `images` (JPEG bytes) shown before the text.

    `deep` asks for the model's highest thinking level; `thinking` sets another level ("minimal" for fast checks).
    Either falls back to the plain call when the model has no thinking levels. `schema` (a JSON Schema) constrains
    the answer to that shape, so long Markdown fields cannot break the JSON. A reply that still does not parse (or is
    empty) is asked for once more, as its own ledger row, before the error reaches the caller.
    """
    key = get_secret("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not configured")
    contents: list = [types.Part.from_bytes(data=image, mime_type="image/jpeg") for image in images]
    contents.append(prompt)
    configs = _configs("high" if deep else thinking, schema)
    for attempt in (1, 2):
        text = await _generate_text(key, contents, configs, timeout_sec, purpose if attempt == 1 else f"{purpose} (retry)")
        try:
            return _parse(text)
        except ValueError as exc:  # json.JSONDecodeError included
            if attempt == 2:
                raise
            where = getattr(exc, "pos", None)
            near = text[max(0, where - 80):where + 80] if isinstance(where, int) else text[:160]
            logger.warning("Gemini %s returned unusable JSON (%s) near %r; asking once more", purpose, exc, near)
    raise RuntimeError("unreachable")
