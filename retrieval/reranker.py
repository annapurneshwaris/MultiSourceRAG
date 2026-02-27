"""Diversity-constrained MMR re-ranker.

Scores: w1×relevance + w2×source_boost + w3×freshness + w4×authority - w5×redundancy
Ensures at least 1 chunk from each source with score > threshold.
Input: ~60 candidates → output: top 10.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from processing.schemas import Chunk


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _freshness_score(created_at: str) -> float:
    """Compute freshness score based on age. More recent = higher score."""
    if not created_at:
        return 0.5
    try:
        date = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        age_days = (datetime.now(date.tzinfo) - date).days
        # Decay: score = 1/(1 + age_days/365)
        return 1.0 / (1.0 + age_days / 365.0)
    except (ValueError, TypeError):
        return 0.5


def _authority_score(chunk: Chunk) -> float:
    """Compute authority score from metadata."""
    score = 0.5  # Base
    meta = chunk.metadata

    if chunk.source_type == "bug":
        if meta.get("verified"):
            score += 0.2
        if meta.get("has_team_response"):
            score += 0.2
        reactions = meta.get("total_reactions", 0)
        if reactions > 10:
            score += 0.1

    elif chunk.source_type == "work_item":
        reactions = meta.get("total_reactions", 0)
        if reactions > 0:
            score += 0.1 * min(1.0, math.log1p(reactions) / 5.0)
        if meta.get("state") == "closed":
            score += 0.15

    elif chunk.source_type == "doc":
        score += 0.2  # Docs are authoritative by default

    return min(1.0, score)


def rerank(
    candidates: list[tuple[Chunk, float]],
    source_boosts: dict[str, float],
    top_k: int = 10,
    embeddings: dict[str, np.ndarray] | None = None,
    w_relevance: float = 0.4,
    w_source_boost: float = 0.25,
    w_freshness: float = 0.1,
    w_authority: float = 0.1,
    w_redundancy: float = 0.15,
    diversity_threshold: float = 0.3,
) -> list[tuple[Chunk, float]]:
    """Diversity-constrained MMR re-ranking.

    Args:
        candidates: List of (Chunk, relevance_score) from retrievers.
        source_boosts: Router's source boost weights {"doc": 0.8, "bug": 0.5, ...}.
        top_k: Number of results to return.
        embeddings: Optional dict of chunk_id → embedding for redundancy computation.
        w_relevance: Weight for base relevance score.
        w_source_boost: Weight for source routing boost.
        w_freshness: Weight for temporal freshness.
        w_authority: Weight for authority signals.
        w_redundancy: Weight for redundancy penalty.
        diversity_threshold: Min score for a source to be force-included.

    Returns:
        Top-k re-ranked (Chunk, final_score) list.
    """
    if not candidates:
        return []

    # Normalize relevance scores to [0, 1]
    max_rel = max(score for _, score in candidates) if candidates else 1.0
    if max_rel == 0:
        max_rel = 1.0

    # Score each candidate
    scored = []
    for chunk, rel_score in candidates:
        norm_rel = rel_score / max_rel
        source_boost = source_boosts.get(chunk.source_type, 0.5)
        freshness = _freshness_score(chunk.created_at)
        authority = _authority_score(chunk)

        final = (
            w_relevance * norm_rel
            + w_source_boost * source_boost
            + w_freshness * freshness
            + w_authority * authority
        )

        scored.append({
            "chunk": chunk,
            "rel_score": rel_score,
            "final_score": final,
            "source_type": chunk.source_type,
        })

    # Sort by final score
    scored.sort(key=lambda x: x["final_score"], reverse=True)

    # Greedy MMR selection with diversity
    selected: list[tuple[Chunk, float]] = []
    selected_ids: list[str] = []

    for item in scored:
        if len(selected) >= top_k:
            break

        chunk = item["chunk"]
        score = item["final_score"]

        # Redundancy penalty — cosine similarity if embeddings available, else word overlap
        redundancy = 0.0
        if embeddings and chunk.chunk_id in embeddings:
            emb = embeddings[chunk.chunk_id]
            for prev_id in selected_ids:
                if prev_id in embeddings:
                    sim = _cosine_similarity(emb, embeddings[prev_id])
                    redundancy = max(redundancy, sim)
        else:
            chunk_words = set(chunk.text.lower().split()[:50])
            for prev_chunk, _ in selected:
                prev_words = set(prev_chunk.text.lower().split()[:50])
                if chunk_words and prev_words:
                    overlap = len(chunk_words & prev_words) / len(chunk_words | prev_words)
                    redundancy = max(redundancy, overlap)

        adjusted_score = score - w_redundancy * redundancy

        selected.append((chunk, adjusted_score))
        selected_ids.append(chunk.chunk_id)

    # Diversity enforcement: ensure at least 1 from each source above threshold
    source_types_present = {chunk.source_type for chunk, _ in selected}
    all_sources = {"doc", "bug", "work_item"}
    missing_sources = all_sources - source_types_present

    already_selected_ids = {chunk.chunk_id for chunk, _ in selected}

    for missing_src in missing_sources:
        # Find best candidate from this source not already selected
        for item in scored:
            if (
                item["source_type"] == missing_src
                and item["final_score"] > diversity_threshold
                and item["chunk"].chunk_id not in already_selected_ids
            ):
                if len(selected) >= top_k:
                    # Replace lowest-scored item
                    selected.sort(key=lambda x: x[1], reverse=True)
                    selected[-1] = (item["chunk"], item["final_score"])
                else:
                    selected.append((item["chunk"], item["final_score"]))
                already_selected_ids.add(item["chunk"].chunk_id)
                break

    # Final sort
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[:top_k]
