"""Table 5: Per-query category breakdown (money table).

Shows results broken down by query category (how_to, debugging, error_diagnosis,
status_roadmap, config) — demonstrates which source types help which query types.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict


def generate_table5(results_path: str = "data/evaluation/ablation_results.json") -> dict:
    """Generate Table 5: Per-Category Breakdown."""
    if not os.path.exists(results_path):
        return {"error": "No results found."}

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Only DBW config
    dbw_results = [r for r in results if r.get("config") == "DBW" and "error" not in r]

    by_category = defaultdict(list)
    for r in dbw_results:
        by_category[r.get("category", "unknown")].append(r)

    table = {}
    for category, cat_results in sorted(by_category.items()):
        n = len(cat_results)
        if n == 0:
            continue

        # Source distribution in results
        source_counts = defaultdict(int)
        for r in cat_results:
            for c in r.get("reranked_chunks", []):
                source_counts[c["source_type"]] += 1
        total = sum(source_counts.values())

        # Citation distribution
        cite_counts = defaultdict(int)
        for r in cat_results:
            for src, ids in r.get("citations", {}).items():
                cite_counts[src] += len(ids)

        table[category] = {
            "n_queries": n,
            "source_distribution": {
                src: round(cnt / total * 100, 1) if total > 0 else 0
                for src, cnt in source_counts.items()
            },
            "citation_distribution": dict(cite_counts),
            "avg_time_ms": round(
                sum(r.get("timing", {}).get("total_ms", 0) for r in cat_results) / n, 1
            ),
        }

    return {"categories": table}


if __name__ == "__main__":
    result = generate_table5()
    print(json.dumps(result, indent=2))
