"""ChromaDB vector store — backup store with built-in metadata filtering.

Useful for experiments where native metadata filtering is needed.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from indexing.stores.base import VectorStore
from processing.schemas import Chunk


class ChromaStore(VectorStore):
    """ChromaDB-backed vector store with native metadata filtering."""

    def __init__(self, collection_name: str = "heterorag", persist_dir: str = ""):
        import chromadb

        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._chunks: list[Chunk] = []

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """Add embeddings and chunks to ChromaDB."""
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]

        # ChromaDB metadata must be flat (str, int, float, bool)
        metadatas = []
        for c in chunks:
            meta = {
                "source_type": c.source_type,
                "feature_area": c.feature_area,
                "source_id": c.source_id,
                "created_at": c.created_at,
            }
            # Flatten selected metadata fields
            for key in ["state", "os_platform", "verified", "item_type", "milestone"]:
                if key in c.metadata:
                    val = c.metadata[key]
                    if isinstance(val, bool):
                        meta[key] = val
                    elif isinstance(val, (int, float)):
                        meta[key] = val
                    else:
                        meta[key] = str(val)
            metadatas.append(meta)

        # ChromaDB has batch limits, add in batches of 5000
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            self._collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end].tolist(),
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )

        self._chunks.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_ids: Optional[set[int]] = None,
        where: Optional[dict] = None,
    ) -> list[tuple[Chunk, float]]:
        """Search ChromaDB with optional metadata filters.

        Args:
            query_embedding: 1-D array.
            top_k: Number of results.
            filter_ids: Not used for ChromaDB (use where instead).
            where: ChromaDB where clause, e.g. {"source_type": "bug"}.
        """
        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k,
        }
        if where:
            query_params["where"] = where

        results = self._collection.query(**query_params)

        # Map chunk_id to Chunk for fast lookup
        chunk_map = {c.chunk_id: c for c in self._chunks}

        output = []
        if results["ids"] and results["ids"][0]:
            distances = results["distances"][0] if results["distances"] else [0] * len(results["ids"][0])
            for chunk_id, distance in zip(results["ids"][0], distances):
                if chunk_id in chunk_map:
                    # ChromaDB returns distance; convert to similarity
                    score = 1.0 - distance
                    output.append((chunk_map[chunk_id], score))

        return output

    def save(self, path: str) -> None:
        """ChromaDB auto-persists if PersistentClient was used."""
        pass

    def load(self, path: str) -> None:
        """Re-initialize with persist directory."""
        import chromadb

        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name="heterorag",
            metadata={"hnsw:space": "cosine"},
        )

    def __len__(self) -> int:
        return self._collection.count()
