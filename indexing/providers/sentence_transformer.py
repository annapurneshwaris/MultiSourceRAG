"""Sentence-transformers embedding provider (default).

Uses all-MiniLM-L6-v2 (384 dim) for fast, local embeddings.
"""

from __future__ import annotations

import numpy as np

from indexing.providers.base import EmbeddingProvider


class SentenceTransformerProvider(EmbeddingProvider):
    """Wraps sentence-transformers for batch embedding with progress."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self._dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts in batches with progress bar."""
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query (no progress bar)."""
        embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
        )
        return np.array(embedding[0], dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dim
