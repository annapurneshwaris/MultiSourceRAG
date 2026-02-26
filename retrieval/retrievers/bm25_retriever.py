"""BM25 baseline retriever — pure keyword search.

No metadata filtering. Used for BM25-only experiment configuration.
"""

from __future__ import annotations

import numpy as np

from processing.schemas import Chunk
from retrieval.retrievers.base import SourceRetriever
from indexing.bm25_index import BM25Index


class BM25Retriever(SourceRetriever):
    """Pure BM25 keyword search retriever."""

    def __init__(self, bm25_index: BM25Index):
        self._bm25 = bm25_index

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 20,
        metadata_hints: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Retrieve using BM25 keyword matching only."""
        return self._bm25.search(query, top_k=top_k)
