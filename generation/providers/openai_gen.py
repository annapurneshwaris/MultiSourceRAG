"""OpenAI LLM generation provider (GPT-4o)."""

from __future__ import annotations

from generation.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI API-based text generation."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ):
        import openai
        import config as cfg

        self._client = openai.OpenAI(api_key=api_key or cfg.OPENAI_API_KEY)
        self._model = model or cfg.LLM_MODEL_NAME

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
