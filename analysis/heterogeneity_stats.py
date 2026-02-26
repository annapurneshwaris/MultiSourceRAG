"""Table 1: Dataset heterogeneity statistics.

Generates the core paper table showing vocabulary overlap, avg token length,
feature area distribution, and temporal span across all three sources.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def _avg_tokens(texts: list[str]) -> float:
    if not texts:
        return 0
    return sum(len(re.findall(r"\b\w+\b", t)) for t in texts) / len(texts)


def generate_table1(chunks_path: str = "data/processed/all_chunks.json") -> dict:
    """Generate Table 1: Heterogeneity Statistics."""
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    by_source = {"doc": [], "bug": [], "work_item": []}
    for c in chunks:
        by_source.get(c["source_type"], []).append(c)

    # Vocabulary per source
    vocabs = {}
    for src, src_chunks in by_source.items():
        all_words = set()
        for c in src_chunks:
            all_words |= _tokenize(c["text"])
        vocabs[src] = all_words

    # Pairwise overlap
    overlap = {}
    sources = list(vocabs.keys())
    for i, s1 in enumerate(sources):
        for s2 in sources[i + 1:]:
            inter = len(vocabs[s1] & vocabs[s2])
            union = len(vocabs[s1] | vocabs[s2])
            overlap[f"{s1}-{s2}"] = round(inter / union * 100, 1) if union > 0 else 0

    # Per-source stats
    stats = {}
    for src, src_chunks in by_source.items():
        texts = [c["text"] for c in src_chunks]
        areas = Counter(c["feature_area"] for c in src_chunks)

        stats[src] = {
            "count": len(src_chunks),
            "vocab_size": len(vocabs[src]),
            "avg_tokens": round(_avg_tokens(texts), 1),
            "top_areas": dict(areas.most_common(5)),
            "unknown_pct": round(areas.get("unknown", 0) / len(src_chunks) * 100, 1),
        }

    return {
        "per_source": stats,
        "vocab_overlap_pct": overlap,
        "total_chunks": len(chunks),
    }


if __name__ == "__main__":
    result = generate_table1()
    print(json.dumps(result, indent=2))
