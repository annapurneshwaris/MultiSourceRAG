"""LinUCB adaptive source router — C1 contribution.

Contextual bandit that learns source utility from experience.
Uses query embeddings as context features and maintains per-source
(A, b) matrices for Upper Confidence Bound action selection.

Cold start: first COLD_START_THRESHOLD queries use equal boosts.
Exploration decay: alpha decays from ALPHA_INITIAL to ALPHA_MIN.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from retrieval.router.base import SourceRouter

SOURCES = ["doc", "bug", "work_item"]


class AdaptiveRouter(SourceRouter):
    """LinUCB contextual bandit router with exploration decay."""

    def __init__(
        self,
        feature_dim: int = 384,
        alpha_initial: float = 1.0,
        alpha_min: float = 0.3,
        alpha_decay: float = 0.99,
        cold_start_threshold: int = 50,
    ):
        self._feature_dim = feature_dim
        self._alpha = alpha_initial
        self._alpha_initial = alpha_initial
        self._alpha_min = alpha_min
        self._alpha_decay = alpha_decay
        self._cold_start = cold_start_threshold
        self._query_count = 0

        # Per-source LinUCB parameters
        # A_s: d×d matrix (initialized to identity)
        # b_s: d×1 vector (initialized to zeros)
        self._A: dict[str, np.ndarray] = {}
        self._b: dict[str, np.ndarray] = {}

        for src in SOURCES:
            self._A[src] = np.eye(feature_dim, dtype=np.float64)
            self._b[src] = np.zeros(feature_dim, dtype=np.float64)

        # Cache A_inv for efficiency (recomputed on update)
        self._A_inv: dict[str, np.ndarray] = {}
        for src in SOURCES:
            self._A_inv[src] = np.eye(feature_dim, dtype=np.float64)

    def predict(
        self,
        query: str,
        query_embedding: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Predict source boosts using LinUCB."""
        self._query_count += 1

        # Cold start: equal weights
        if self._query_count <= self._cold_start or query_embedding is None:
            return {"doc": 0.5, "bug": 0.5, "work_item": 0.5}

        x = query_embedding.astype(np.float64)

        boosts = {}
        for src in SOURCES:
            A_inv = self._A_inv[src]
            theta = A_inv @ self._b[src]  # Expected reward

            # UCB bonus for exploration
            ucb_bonus = self._alpha * np.sqrt(x @ A_inv @ x)

            # Score = expected reward + exploration bonus
            score = float(theta @ x + ucb_bonus)

            # Clamp to [0.2, 1.0]
            boosts[src] = max(0.2, min(1.0, score))

        return boosts

    def update(
        self,
        query_embedding: np.ndarray,
        utilities: dict[str, float],
    ) -> None:
        """Update router with observed utilities from a query.

        Args:
            query_embedding: The query's embedding vector.
            utilities: {source_type: utility_score} from UtilitySignalCollector.
        """
        x = query_embedding.astype(np.float64)

        for src in SOURCES:
            if src not in utilities:
                continue

            reward = utilities[src]

            # LinUCB update: A_s += x·x', b_s += reward·x
            self._A[src] += np.outer(x, x)
            self._b[src] += reward * x

            # Recompute inverse
            self._A_inv[src] = np.linalg.inv(self._A[src])

        # Decay exploration
        self._alpha = max(self._alpha_min, self._alpha * self._alpha_decay)

    def save_state(self, path: str) -> None:
        """Save router state to JSON + numpy files."""
        os.makedirs(path, exist_ok=True)

        state = {
            "query_count": self._query_count,
            "alpha": self._alpha,
            "feature_dim": self._feature_dim,
            "alpha_initial": self._alpha_initial,
            "alpha_min": self._alpha_min,
            "alpha_decay": self._alpha_decay,
            "cold_start": self._cold_start,
        }
        with open(os.path.join(path, "router_state.json"), "w") as f:
            json.dump(state, f, indent=2)

        for src in SOURCES:
            np.save(os.path.join(path, f"A_{src}.npy"), self._A[src])
            np.save(os.path.join(path, f"b_{src}.npy"), self._b[src])

    def load_state(self, path: str) -> None:
        """Load router state from disk."""
        with open(os.path.join(path, "router_state.json"), "r") as f:
            state = json.load(f)

        self._query_count = state["query_count"]
        self._alpha = state["alpha"]
        self._feature_dim = state["feature_dim"]

        for src in SOURCES:
            self._A[src] = np.load(os.path.join(path, f"A_{src}.npy"))
            self._b[src] = np.load(os.path.join(path, f"b_{src}.npy"))
            self._A_inv[src] = np.linalg.inv(self._A[src])

    def save_history(self, entry: dict, path: str) -> None:
        """Append a query-result entry to JSONL history file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        entry["timestamp"] = time.time()
        entry["query_count"] = self._query_count
        entry["alpha"] = self._alpha
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    @property
    def stats(self) -> dict:
        return {
            "query_count": self._query_count,
            "alpha": round(self._alpha, 4),
            "cold_start_remaining": max(0, self._cold_start - self._query_count),
        }
