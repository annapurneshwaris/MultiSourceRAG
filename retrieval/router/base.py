"""Abstract base class for source routers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SourceRouter(ABC):
    """Interface for all source routing strategies.

    A router predicts source-type boost weights given a query,
    controlling how much each source contributes to retrieval.
    """

    @abstractmethod
    def predict(
        self,
        query: str,
        query_embedding: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Predict source boost weights for a query.

        Args:
            query: The user query text.
            query_embedding: Optional pre-computed embedding vector.

        Returns:
            Dict mapping source type to boost weight.
            Keys: "doc", "bug", "work_item"
            Values: floats in [0.2, 1.0] (never fully exclude a source).
        """
