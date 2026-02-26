"""Table 3: Baseline comparison results.

Compares HeteroRAG against BM25, naive concat, and single-source baselines.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict


def generate_table3(results_path: str = "data/evaluation/ablation_results.json") -> dict:
    """Generate Table 3: Baseline Comparison."""
    if not os.path.exists(results_path):
        return {"error": "No results found."}

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Key configs for baseline comparison
    baseline_configs = ["D", "B", "W", "BM25", "DBW"]

    by_config = defaultdict(list)
    for r in results:
        if r.get("config") in baseline_configs and "error" not in r:
            by_config[r["config"]].append(r)

    table = {}
    for config, config_results in by_config.items():
        n = len(config_results)
        if n == 0:
            continue

        table[config] = {
            "n_queries": n,
            "avg_time_ms": round(
                sum(r.get("timing", {}).get("total_ms", 0) for r in config_results) / n, 1
            ),
            "avg_retrieved": round(
                sum(r.get("retrieved_count", 0) for r in config_results) / n, 1
            ),
        }

    return {"baselines": table}


if __name__ == "__main__":
    result = generate_table3()
    print(json.dumps(result, indent=2))
