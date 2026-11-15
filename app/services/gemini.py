"""One Gemini entry point: JSON generation with optional inline images, deep thinking and the tool ledger.

Every Gemini call in the pipeline goes through `generate_json`, so the ledger sees all of them and the retry and
parsing rules live in one place.
"""
import asyncio
import json
import re
from typing import Any, Sequence

from google import genai
from google.genai import types

from app.config import GEMINI_MODEL, get_secret
from app.services import ledger

GEMINI_TIMEOUT_SEC = 90


async def generate_json(prompt: str, *, images: Sequence[bytes] = (), deep: bool = False,
                        timeout_sec: int = GEMINI_TIMEOUT_SEC, purpose: str = "gemini") -> Any:
    """Parsed JSON from Gemini for `prompt`, with `images` (JPEG bytes) shown before the text.

    `deep` asks for the model's highest thinking level and falls back to the plain call when the model has none.
    """
    key = get_secret("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not configured")
    contents: list = [types.Part.from_bytes(data=image, mime_type="image/jpeg") for image in images]
    contents.append(prompt)
    configs = [types.GenerateContentConfig(response_mime_type="application/json")]
    if deep:
        configs.insert(0, types.GenerateContentConfig(response_mime_type="application/json",
                                                      thinking_config=types.ThinkingConfig(thinking_level="high")))
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
                except Exception as exc:  # a model without thinking levels: fall back to the plain call
                    if config is configs[-1] or "thinking" not in str(exc).lower():
                        raise
    finally:
        await client.aio.aclose()
    text = (response.text or "").strip()
    if not text:
        raise ValueError("Gemini returned an empty response")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)  # tolerate fenced output
    # strict=False accepts raw newlines/tabs inside string values, which Gemini sometimes emits in long fields.
    return json.loads(text, strict=False)
