"""Abstract base class for vector stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from processing.schemas import Chunk


class VectorStore(ABC):
    """Interface for all vector stores."""

    @abstractmethod
    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """Add embeddings and their associated chunks to the store.

        Args:
            embeddings: 2-D array of shape (n, dim).
            chunks: Corresponding Chunk objects (same length as embeddings).
        """

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_ids: Optional[set[int]] = None,
    ) -> list[tuple[Chunk, float]]:
        """Search for nearest neighbors.

        Args:
            query_embedding: 1-D array of shape (dim,).
            top_k: Number of results.
            filter_ids: If provided, only return chunks at these indices.

        Returns:
            List of (Chunk, score) tuples, sorted by descending score.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the store to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the store from disk."""

    @abstractmethod
    def __len__(self) -> int:
        """Return number of stored vectors."""
