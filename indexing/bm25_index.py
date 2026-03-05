"""BM25 keyword search index — baseline retriever.

Wraps rank_bm25 for term-based retrieval. Used as BM25-only baseline
and as a fallback when vector search returns poor results.
"""

from __future__ import annotations

import os
import pickle
import re

from processing.schemas import Chunk


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


class BM25Index:
    """BM25 keyword search over chunks."""

    def __init__(self):
        self._bm25 = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        """Build BM25 index from chunks.

        Args:
            chunks: List of Chunk objects to index.
        """
        from rank_bm25 import BM25Okapi

        self._chunks = chunks
        corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """Search by BM25 relevance.

        Args:
            query: Search query string.
            top_k: Number of results.

        Returns:
            List of (Chunk, score) tuples, sorted by descending score.
        """
        if self._bm25 is None or not self._chunks:
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Get top-k indices
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self._chunks[idx], float(scores[idx])))

        return results

    def save(self, path: str) -> None:
        """Save BM25 index and chunks."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "bm25.pkl"), "wb") as f:
            pickle.dump(self._bm25, f)
        with open(os.path.join(path, "bm25_chunks.pkl"), "wb") as f:
            pickle.dump(self._chunks, f)

    def load(self, path: str) -> None:
        """Load BM25 index and chunks."""
        with open(os.path.join(path, "bm25.pkl"), "rb") as f:
            self._bm25 = pickle.load(f)
        with open(os.path.join(path, "bm25_chunks.pkl"), "rb") as f:
            self._chunks = pickle.load(f)

    def score_query(self, query: str) -> dict[str, float]:
        """Get BM25 scores for all chunks, keyed by chunk_id.

        Returns only chunks with score > 0.
        """
        if self._bm25 is None or not self._chunks:
            return {}

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        result = {}
        for idx, score in enumerate(scores):
            if score > 0:
                result[self._chunks[idx].chunk_id] = float(score)
        return result

    def __len__(self) -> int:
        return len(self._chunks)
