"""Table 5: Per-query category breakdown (money table).

Shows results broken down by query category (how_to, debugging, error_diagnosis,
status_roadmap, config) — demonstrates which source types help which query types.
Includes RA, CSAS per category.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from evaluation.metrics import compute_csas, compute_msur


def _load_judge_scores(judge_path: str) -> dict[tuple[str, str], dict]:
    """Load LLM judge scores keyed by (query_id, config)."""
    if not os.path.exists(judge_path):
        return {}
    with open(judge_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {(d["query_id"], d["config"]): d for d in data}


def generate_table5(
    results_path: str = "data/evaluation/ablation_results.json",
    judge_path: str = "data/evaluation/judge_scores.json",
) -> dict:
    """Generate Table 5: Per-Category Breakdown with quality metrics."""
    if not os.path.exists(results_path):
        return {"error": "No results found."}

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    judge_scores = _load_judge_scores(judge_path)

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

        # RA from judge scores
        ra_scores = []
        for r in cat_results:
            key = (r.get("query_id", ""), "DBW")
            if key in judge_scores:
                ra_scores.append(judge_scores[key].get("ra", 0.0))

        # CSAS per query
        csas_scores = []
        for r in cat_results:
            citations = r.get("citations", {})
            expected = r.get("expected_sources", [])
            if expected:
                csas = compute_csas(citations, expected)
                csas_scores.append(csas.f1)

        # MSUR for this category
        msur = compute_msur(cat_results)

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
            "avg_ra": round(sum(ra_scores) / len(ra_scores), 3) if ra_scores else None,
            "avg_csas": round(sum(csas_scores) / len(csas_scores), 3) if csas_scores else None,
            "msur": round(msur, 3),
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
