"""Abstract base class for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Interface for all embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            2-D numpy array of shape (len(texts), dimension).
        """

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Returns:
            1-D numpy array of shape (dimension,).
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
