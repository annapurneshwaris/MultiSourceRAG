"""Table 4: Router comparison results.

Compares heuristic, adaptive (LinUCB), and LLM zero-shot routers on DBW config.
Shows quality metrics alongside routing behavior.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from evaluation.metrics import compute_csas, compute_msur


def _load_judge_scores(judge_path: str) -> dict[tuple[str, str, str], dict]:
    """Load LLM judge scores keyed by (query_id, config, router_type)."""
    if not os.path.exists(judge_path):
        return {}
    with open(judge_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        (d["query_id"], d["config"], d.get("router_type", "heuristic")): d
        for d in data
    }


def generate_table4(
    results_path: str = "data/evaluation/ablation_results.json",
    judge_path: str = "data/evaluation/judge_scores.json",
) -> dict:
    """Generate Table 4: Router Comparison with quality metrics."""
    if not os.path.exists(results_path):
        return {"error": "No results found."}

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    judge_scores = _load_judge_scores(judge_path)

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

        # RA from judge scores
        ra_scores = []
        for r in router_results:
            key = (r.get("query_id", ""), "DBW", router)
            if key in judge_scores:
                ra_scores.append(judge_scores[key].get("ra", 0.0))

        # CSAS
        csas_scores = []
        for r in router_results:
            citations = r.get("citations", {})
            expected = r.get("expected_sources", [])
            if expected:
                csas = compute_csas(citations, expected)
                csas_scores.append(csas.f1)

        # MSUR
        msur = compute_msur(router_results)

        # Average boosts
        avg_boosts = defaultdict(float)
        for r in router_results:
            for src, boost in r.get("source_boosts", {}).items():
                avg_boosts[src] += boost
        avg_boosts = {src: round(v / n, 3) for src, v in avg_boosts.items()}

        table[router] = {
            "n_queries": n,
            "avg_ra": round(sum(ra_scores) / len(ra_scores), 3) if ra_scores else None,
            "avg_csas": round(sum(csas_scores) / len(csas_scores), 3) if csas_scores else None,
            "msur": round(msur, 3),
            "avg_boosts": dict(avg_boosts),
            "avg_route_ms": round(
                sum(r.get("timing", {}).get("route_ms", 0) for r in router_results) / n, 1
            ),
            "avg_total_ms": round(
                sum(r.get("timing", {}).get("total_ms", 0) for r in router_results) / n, 1
            ),
        }

    return {"routers": table}


if __name__ == "__main__":
    result = generate_table4()
    print(json.dumps(result, indent=2))
