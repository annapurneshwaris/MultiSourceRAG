"""OpenAI LLM generation provider (GPT-4o)."""

from __future__ import annotations

import logging
import time

from generation.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_TIMEOUT = 60  # seconds


class OpenAIProvider(LLMProvider):
    """OpenAI API-based text generation with retry and timeout."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ):
        import openai
        import config as cfg

        self._client = openai.OpenAI(
            api_key=api_key or cfg.OPENAI_API_KEY,
            timeout=_TIMEOUT,
        )
        self._model = model or cfg.LLM_MODEL_NAME

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str:
        import openai

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content or ""
            except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as e:
                wait = min(2 ** attempt * 2, 60)  # 2, 4, 8, 16, 32s
                logger.warning("OpenAI attempt %d/%d failed: %s -- retrying in %ds", attempt + 1, _MAX_RETRIES, e, wait)
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(wait)
                else:
                    raise
