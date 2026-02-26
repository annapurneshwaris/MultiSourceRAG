"""Figure 3: Source distribution by query category.

Stacked bar chart showing which sources contribute to each query category.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict


def generate_figure3(
    results_path: str = "data/evaluation/ablation_results.json",
    output_path: str = "data/evaluation/fig3_source_distribution.json",
) -> dict:
    """Generate data for Figure 3."""
    if not os.path.exists(results_path):
        return {"error": "No ablation results found."}

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    dbw_results = [r for r in results if r.get("config") == "DBW" and "error" not in r]

    by_category = defaultdict(lambda: defaultdict(int))
    for r in dbw_results:
        cat = r.get("category", "unknown")
        for c in r.get("reranked_chunks", []):
            by_category[cat][c["source_type"]] += 1

    # Normalize to percentages
    data = {}
    for cat, sources in sorted(by_category.items()):
        total = sum(sources.values())
        data[cat] = {
            src: round(cnt / total * 100, 1) if total > 0 else 0
            for src, cnt in sources.items()
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return data


if __name__ == "__main__":
    result = generate_figure3()
    print(json.dumps(result, indent=2))
