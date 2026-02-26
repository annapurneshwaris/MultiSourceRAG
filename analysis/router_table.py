"""Table 4: Router comparison results.

Compares heuristic, adaptive (LinUCB), and LLM zero-shot routers.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict


def generate_table4(results_path: str = "data/evaluation/ablation_results.json") -> dict:
    """Generate Table 4: Router Comparison."""
    if not os.path.exists(results_path):
        return {"error": "No results found."}

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    by_router = defaultdict(list)
    for r in results:
        if "error" not in r and r.get("config") == "DBW":
            router = r.get("router_type", "heuristic")
            by_router[router].append(r)

    table = {}
    for router, router_results in by_router.items():
        n = len(router_results)
        if n == 0:
            continue

        # Average boosts
        avg_boosts = defaultdict(float)
        for r in router_results:
            for src, boost in r.get("source_boosts", {}).items():
                avg_boosts[src] += boost
        avg_boosts = {src: round(v / n, 3) for src, v in avg_boosts.items()}

        table[router] = {
            "n_queries": n,
            "avg_boosts": dict(avg_boosts),
            "avg_time_ms": round(
                sum(r.get("timing", {}).get("route_ms", 0) for r in router_results) / n, 1
            ),
        }

    return {"routers": table}


if __name__ == "__main__":
    result = generate_table4()
    print(json.dumps(result, indent=2))
