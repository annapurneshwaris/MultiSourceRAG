"""Table 2: Source ablation results.

Generates the main ablation table showing RA, CSAS, MSUR for all 7+ configs.
This is the centerpiece table demonstrating multi-source value.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict


def generate_table2(results_path: str = "data/evaluation/ablation_results.json") -> dict:
    """Generate Table 2: Ablation Results."""
    if not os.path.exists(results_path):
        return {"error": "No ablation results found. Run evaluation.ablation_runner first."}

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Group by config
    by_config: dict[str, list] = defaultdict(list)
    for r in results:
        if "error" not in r:
            by_config[r["config"]].append(r)

    table = {}
    for config, config_results in by_config.items():
        # Count source types in retrieved chunks
        source_counts = defaultdict(int)
        for r in config_results:
            for c in r.get("reranked_chunks", []):
                source_counts[c["source_type"]] += 1

        total_chunks = sum(source_counts.values())
        timing_avg = sum(
            r.get("timing", {}).get("total_ms", 0) for r in config_results
        ) / len(config_results)

        table[config] = {
            "n_queries": len(config_results),
            "avg_retrieved": round(
                sum(r.get("retrieved_count", 0) for r in config_results) / len(config_results), 1
            ),
            "avg_time_ms": round(timing_avg, 1),
            "source_distribution": {
                src: round(cnt / total_chunks * 100, 1) if total_chunks > 0 else 0
                for src, cnt in source_counts.items()
            },
        }

    return {"configs": table}


if __name__ == "__main__":
    result = generate_table2()
    print(json.dumps(result, indent=2))
