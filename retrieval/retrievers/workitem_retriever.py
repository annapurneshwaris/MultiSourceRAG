"""Work item retriever — vector + metadata + reaction boost.

Step 1: Metadata filter (feature_area, item_type, milestone).
Step 2: Vector search over filtered set.
Step 3: Reaction boost (log-scale), shipped bonus, current milestone bonus.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from processing.schemas import Chunk
from retrieval.retrievers.base import SourceRetriever


class WorkItemRetriever(SourceRetriever):
    """Vector search with reaction boosting for work items."""

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
        """Retrieve work item chunks with metadata filtering and reaction boost."""

        # Build metadata constraints
        filter_ids = None
        if self._meta_idx:
            constraints = {"source_type": "work_item"}

            if metadata_hints:
                if "feature_area" in metadata_hints:
                    constraints["feature_area"] = metadata_hints["feature_area"]
                if "item_type" in metadata_hints:
                    constraints["item_type"] = metadata_hints["item_type"]

            filter_ids = self._meta_idx.filter(constraints)

            # Fallback
            if len(filter_ids) < top_k:
                filter_ids = self._meta_idx.filter({"source_type": "work_item"})

        # Vector search
        results = self._store.search(
            query_embedding,
            top_k=top_k,
            filter_ids=filter_ids,
        )

        # Reaction + status boost
        boosted = []
        for chunk, score in results:
            boost = 1.0
            meta = chunk.metadata

            # Reaction boost (log-scale to avoid domination)
            reactions = meta.get("total_reactions", 0)
            if reactions > 0:
                boost *= 1.0 + 0.1 * math.log1p(reactions)

            # Shipped/closed items are confirmed
            if meta.get("state") == "closed":
                boost *= 1.2

            # Current/recent milestones
            milestone = meta.get("milestone", "")
            if milestone and any(y in milestone for y in ["2025", "2024"]):
                boost *= 1.1

            boosted.append((chunk, score * boost))

        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted[:top_k]
