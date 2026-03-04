"""Table 4: Router comparison results.

Compares heuristic, adaptive (LinUCB), and LLM zero-shot routers on DBW config.
Shows quality metrics alongside routing behavior.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from evaluation.metrics import compute_csas, compute_msur


JUDGE_FILES = [
    "data/evaluation/judge_scores_gpt4o.json",
    "data/evaluation/judge_scores_claude.json",
    "data/evaluation/judge_scores_gemini.json",
]


def _load_judge_scores_averaged() -> dict[tuple[str, str, str], dict]:
    """Load all 3 judge scores and average RA, keyed by (query_id, config, router_type)."""
    from collections import defaultdict as _dd
    ra_by_key: dict[tuple, list] = _dd(list)
    any_scores: dict[tuple, dict] = {}
    for jf in JUDGE_FILES:
        if not os.path.exists(jf):
            continue
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            key = (d["query_id"], d["config"], d.get("router_type", "heuristic"))
            if d.get("ra") is not None:
                ra_by_key[key].append(d["ra"])
            any_scores[key] = d
    result = {}
    for key in any_scores:
        entry = dict(any_scores[key])
        if key in ra_by_key:
            entry["ra"] = sum(ra_by_key[key]) / len(ra_by_key[key])
        result[key] = entry
    return result


def generate_table4(
    results_path: str = "data/evaluation/ablation_results.json",
) -> dict:
    """Generate Table 4: Router Comparison with quality metrics."""
    if not os.path.exists(results_path):
        return {"error": "No results found."}

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    judge_scores = _load_judge_scores_averaged()

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
