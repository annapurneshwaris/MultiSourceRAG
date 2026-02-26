"""Anthropic LLM generation provider (Claude)."""

from __future__ import annotations

from generation.providers.base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Anthropic API-based text generation."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ):
        import anthropic
        import config as cfg

        self._client = anthropic.Anthropic(api_key=api_key or cfg.ANTHROPIC_API_KEY)
        self._model = model

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
