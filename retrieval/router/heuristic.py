"""Keyword-based heuristic router — static baseline.

Uses keyword lists to boost source weights. Never learns or improves.
Used as a lower-bound baseline for paper comparison.
"""

from __future__ import annotations

import numpy as np

from retrieval.router.base import SourceRouter

# Keyword lists for each source type
DOC_KEYWORDS = [
    "how to", "how do i", "configure", "setup", "install", "setting",
    "documentation", "tutorial", "guide", "extension", "feature",
    "enable", "disable", "shortcut", "keybinding", "theme",
    "workspace", "user settings", "snippet", "language support",
]

BUG_KEYWORDS = [
    "error", "bug", "crash", "freeze", "not working", "broken",
    "issue", "problem", "fail", "exception", "hang", "slow",
    "regression", "unexpected", "wrong", "flicker", "glitch",
    "cannot", "unable", "doesn't work", "stopped working",
]

WORK_KEYWORDS = [
    "plan", "roadmap", "feature request", "milestone", "iteration",
    "when will", "planned", "upcoming", "future", "ship",
    "release", "version", "timeline", "priority", "vote",
    "request", "suggestion", "proposal", "track", "progress",
]


class HeuristicRouter(SourceRouter):
    """Static keyword-based router."""

    def predict(
        self,
        query: str,
        query_embedding: np.ndarray | None = None,
    ) -> dict[str, float]:
        query_lower = query.lower()

        doc_score = sum(1 for kw in DOC_KEYWORDS if kw in query_lower)
        bug_score = sum(1 for kw in BUG_KEYWORDS if kw in query_lower)
        work_score = sum(1 for kw in WORK_KEYWORDS if kw in query_lower)

        total = doc_score + bug_score + work_score
        if total == 0:
            return {"doc": 0.5, "bug": 0.5, "work_item": 0.5}

        # Normalize to [0.2, 1.0] range
        max_score = max(doc_score, bug_score, work_score, 1)
        return {
            "doc": max(0.2, doc_score / max_score),
            "bug": max(0.2, bug_score / max_score),
            "work_item": max(0.2, work_score / max_score),
        }
