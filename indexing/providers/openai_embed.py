"""OpenAI embedding provider.

Uses text-embedding-3-small (1536 dim) with rate-limit retry.
"""

from __future__ import annotations

import time

import numpy as np

from indexing.providers.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI API-based embedding provider."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 64,
        max_retries: int = 3,
    ):
        import openai

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._batch_size = batch_size
        self._max_retries = max_retries
        # text-embedding-3-small = 1536, text-embedding-3-large = 3072
        self._dim = 1536 if "small" in model else 3072

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts in batches with retry on rate limit."""
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i: i + self._batch_size]
            embeddings = self._embed_with_retry(batch)
            all_embeddings.extend(embeddings)

            if i % (self._batch_size * 10) == 0 and i > 0:
                print(f"  Embedded {i}/{len(texts)} texts...")

        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        result = self._embed_with_retry([query])
        return np.array(result[0], dtype=np.float32)

    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(self._max_retries):
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                )
                return [e.embedding for e in response.data]
            except Exception as e:
                if "rate" in str(e).lower() and attempt < self._max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    @property
    def dimension(self) -> int:
        return self._dim
