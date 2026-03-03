"""Google Gemini LLM generation provider."""

from __future__ import annotations

import logging
import os
import time

from generation.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3


class GeminiProvider(LLMProvider):
    """Google Gemini API-based text generation with retry and timeout."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ):
        from google import genai
        from dotenv import load_dotenv
        load_dotenv()

        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._client = genai.Client(api_key=api_key)
        self._model_name = model

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=config,
                )
                return response.text or ""
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(
                    "Gemini attempt %d/%d failed: %s — retrying in %ds",
                    attempt + 1, _MAX_RETRIES, e, wait,
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(wait)
                else:
                    raise
