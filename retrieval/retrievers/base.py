"""Abstract base class for per-source retrievers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from processing.schemas import Chunk


class SourceRetriever(ABC):
    """Interface for source-specific retrieval strategies."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 20,
        metadata_hints: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Retrieve relevant chunks for a query.

        Args:
            query: The user query text.
            query_embedding: Pre-computed query embedding.
            top_k: Number of results to return.
            metadata_hints: Optional metadata constraints from query parsing.

        Returns:
            List of (Chunk, score) tuples, sorted by descending score.
        """
