"""Anthropic LLM generation provider (Claude)."""

from __future__ import annotations

import logging
import time

from generation.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_TIMEOUT = 30  # seconds


class AnthropicProvider(LLMProvider):
    """Anthropic API-based text generation with retry and timeout."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ):
        import anthropic
        import config as cfg

        self._client = anthropic.Anthropic(
            api_key=api_key or cfg.ANTHROPIC_API_KEY,
            timeout=_TIMEOUT,
        )
        self._model = model

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str:
        import anthropic

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except (anthropic.RateLimitError, anthropic.APITimeoutError, anthropic.APIConnectionError) as e:
                wait = 2 ** attempt
                logger.warning("Anthropic attempt %d/%d failed: %s — retrying in %ds", attempt + 1, _MAX_RETRIES, e, wait)
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(wait)
                else:
                    raise
