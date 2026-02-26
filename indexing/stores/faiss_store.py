"""FAISS vector store — primary store for HeteroRAG.

Uses IndexFlatIP with L2-normalized vectors (equivalent to cosine similarity).
Supports filtered search by overfetching and intersecting with allowed indices.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from indexing.stores.base import VectorStore
from processing.schemas import Chunk


class FAISSStore(VectorStore):
    """FAISS-backed vector store with metadata filtering support."""

    def __init__(self, dimension: int = 384):
        import faiss

        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)  # Inner product on L2-normed = cosine
        self._chunks: list[Chunk] = []

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """Add L2-normalized embeddings and chunks."""
        import faiss

        assert len(embeddings) == len(chunks), \
            f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks"

        # L2-normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)
        self._chunks.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_ids: Optional[set[int]] = None,
    ) -> list[tuple[Chunk, float]]:
        """Search with optional index filtering.

        When filter_ids is provided, overfetch 5x and intersect.
        """
        import faiss

        if len(self._chunks) == 0:
            return []

        # Normalize query
        qe = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(qe)

        if filter_ids is not None:
            # Overfetch to ensure enough filtered results
            overfetch_k = min(top_k * 5, len(self._chunks))
            scores, indices = self._index.search(qe, overfetch_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                if idx in filter_ids:
                    results.append((self._chunks[idx], float(score)))
                    if len(results) >= top_k:
                        break
            return results
        else:
            k = min(top_k, len(self._chunks))
            scores, indices = self._index.search(qe, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                results.append((self._chunks[idx], float(score)))
            return results

    def save(self, path: str) -> None:
        """Save FAISS index + chunks to directory."""
        import faiss

        os.makedirs(path, exist_ok=True)
        faiss.write_index(self._index, os.path.join(path, "faiss.index"))

        chunks_data = [c.to_dict() for c in self._chunks]
        with open(os.path.join(path, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        """Load FAISS index + chunks from directory."""
        import faiss

        self._index = faiss.read_index(os.path.join(path, "faiss.index"))
        self._dimension = self._index.d

        with open(os.path.join(path, "chunks.json"), "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        self._chunks = [Chunk.from_dict(d) for d in chunks_data]

    def get_chunks(self) -> list[Chunk]:
        """Return all stored chunks (for building metadata index)."""
        return self._chunks

    def __len__(self) -> int:
        return self._index.ntotal
