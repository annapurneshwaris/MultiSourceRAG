"""Bug retriever — vector + metadata filter + authority boost.

Step 1: Metadata filter (feature_area, os_platform, state).
Step 2: Vector search over filtered set.
Step 3: Authority boost (verified, team_response, high_reactions).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from processing.schemas import Chunk
from retrieval.retrievers.base import SourceRetriever


class BugRetriever(SourceRetriever):
    """Vector search with metadata filtering and authority boosting for bugs."""

    def __init__(self, vector_store, metadata_index=None):
        self._store = vector_store
        self._meta_idx = metadata_index

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 20,
        metadata_hints: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Retrieve bug chunks with metadata filtering and authority boost."""

        # Build metadata constraints
        filter_ids = None
        if self._meta_idx:
            constraints = {"source_type": "bug"}

            if metadata_hints:
                if "feature_area" in metadata_hints:
                    constraints["feature_area"] = metadata_hints["feature_area"]
                if "os_platform" in metadata_hints:
                    constraints["os_platform"] = metadata_hints["os_platform"]
                if "state" in metadata_hints:
                    constraints["state"] = metadata_hints["state"]

            filter_ids = self._meta_idx.filter(constraints)

            # Fallback: if constrained set too small, relax to all bugs
            if len(filter_ids) < top_k:
                filter_ids = self._meta_idx.filter({"source_type": "bug"})

        # Vector search
        results = self._store.search(
            query_embedding,
            top_k=top_k,
            filter_ids=filter_ids,
        )

        # Authority boost
        boosted = []
        for chunk, score in results:
            boost = 1.0
            meta = chunk.metadata

            # Verified bugs are more authoritative
            if meta.get("verified"):
                boost *= 1.2

            # Team response indicates quality
            if meta.get("has_team_response"):
                boost *= 1.3

            # High-reaction bugs are important
            reactions = meta.get("total_reactions", 0)
            if reactions > 10:
                boost *= 1.1
            elif reactions > 50:
                boost *= 1.2

            boosted.append((chunk, score * boost))

        # Re-sort by boosted score
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted[:top_k]
