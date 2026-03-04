"""Extract metadata filter hints from user queries.

Parses queries for feature area mentions, OS mentions, and status keywords
to narrow metadata filtering in per-source retrievers.
"""

from __future__ import annotations

import re

from processing.feature_area_map import FEATURE_AREA_MAP

# OS detection patterns
_OS_MAP = {
    "windows": ["windows", "win10", "win11", "windows 10", "windows 11"],
    "macos": ["macos", "mac os", "mac", "darwin", "osx", "macbook"],
    "linux": ["linux", "ubuntu", "debian", "fedora", "arch linux"],
}

# Status keywords
_STATUS_MAP = {
    "closed": ["fixed", "resolved", "closed", "patched", "shipped"],
    "open": ["open", "unresolved", "pending", "active"],
}

# All known area keywords from GitHub labels (NOT DOC_AREA_MAP — those contain
# generic words like "configure", "editing", "setup" that cause false matches)
_AREA_KEYWORDS: dict[str, str] = {}
for label, area in FEATURE_AREA_MAP.items():
    # Use the label itself as a keyword
    _AREA_KEYWORDS[label.replace("-", " ")] = area
    _AREA_KEYWORDS[label] = area


def extract_hints(query: str) -> dict:
    """Extract metadata hints from a user query.

    Returns:
        Dict with optional keys: feature_area, os_platform, state, item_type.
    """
    query_lower = query.lower()
    hints: dict = {}

    # Feature area detection (longest match first)
    sorted_keywords = sorted(_AREA_KEYWORDS.keys(), key=len, reverse=True)
    for keyword in sorted_keywords:
        if keyword in query_lower and len(keyword) >= 3:
            hints["feature_area"] = _AREA_KEYWORDS[keyword]
            break

    # OS detection
    for os_name, patterns in _OS_MAP.items():
        for pattern in patterns:
            if pattern in query_lower:
                hints["os_platform"] = os_name
                break
        if "os_platform" in hints:
            break

    # Status detection
    for state, keywords in _STATUS_MAP.items():
        for keyword in keywords:
            if keyword in query_lower:
                hints["state"] = state
                break
        if "state" in hints:
            break

    # Work item type hints
    if any(kw in query_lower for kw in ["roadmap", "plan", "iteration", "milestone"]):
        hints["item_type"] = "iteration_plan"
    elif any(kw in query_lower for kw in ["feature request", "suggestion", "vote"]):
        hints["item_type"] = "feature_request"

    return hints
