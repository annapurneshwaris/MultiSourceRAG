"""Inverted metadata index for FAISS filtering.

Since FAISS doesn't support native metadata filtering, this builds an
inverted index: field_name → {value → set of chunk indices}.
Used by per-source retrievers to narrow the search space before FAISS lookup.
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

from processing.schemas import Chunk

# Fields to index for filtering
FILTER_FIELDS = [
    "source_type",
    "feature_area",
    "state",
    "os_platform",
    "verified",
    "item_type",
]


class MetadataIndex:
    """Inverted index mapping metadata field values to chunk indices."""

    def __init__(self):
        # {field_name: {value: set[int]}}
        self._index: dict[str, dict[str, set[int]]] = {}

    def build(self, chunks: list[Chunk]) -> None:
        """Build the inverted index from a list of chunks.

        Args:
            chunks: List of Chunk objects. Index position = list position.
        """
        self._index = {}

        for idx, chunk in enumerate(chunks):
            # Index top-level fields
            self._add_entry("source_type", chunk.source_type, idx)
            self._add_entry("feature_area", chunk.feature_area, idx)

            # Index metadata fields
            for field in FILTER_FIELDS:
                if field in chunk.metadata:
                    value = chunk.metadata[field]
                    # Normalize booleans to strings for consistent lookup
                    if isinstance(value, bool):
                        value = str(value).lower()
                    self._add_entry(field, str(value), idx)

    def _add_entry(self, field: str, value: str, idx: int) -> None:
        if field not in self._index:
            self._index[field] = {}
        if value not in self._index[field]:
            self._index[field][value] = set()
        self._index[field][value].add(idx)

    def filter(self, constraints: dict[str, str | list[str]]) -> set[int]:
        """Return chunk indices matching ALL constraints (AND logic).

        Args:
            constraints: {field_name: value} or {field_name: [value1, value2]} (OR within field).

        Returns:
            Set of chunk indices matching all constraints.
        """
        if not constraints:
            return set()

        result: Optional[set[int]] = None

        for field, values in constraints.items():
            if field not in self._index:
                return set()  # No matches for unknown field

            # Support single value or list of values (OR within field)
            if isinstance(values, str):
                values = [values]

            field_matches: set[int] = set()
            for val in values:
                if val in self._index[field]:
                    field_matches.update(self._index[field][val])

            if result is None:
                result = field_matches
            else:
                result = result.intersection(field_matches)

            if not result:
                return set()

        return result or set()

    def get_values(self, field: str) -> list[str]:
        """Return all unique values for a given field."""
        if field not in self._index:
            return []
        return list(self._index[field].keys())

    def get_count(self, field: str, value: str) -> int:
        """Return number of chunks matching field=value."""
        return len(self._index.get(field, {}).get(value, set()))

    def save(self, path: str) -> None:
        """Save index to pickle file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._index, f)

    def load(self, path: str) -> None:
        """Load index from pickle file."""
        with open(path, "rb") as f:
            self._index = pickle.load(f)

    def stats(self) -> dict:
        """Return summary statistics."""
        return {
            field: {
                "unique_values": len(values),
                "total_entries": sum(len(s) for s in values.values()),
            }
            for field, values in self._index.items()
        }
