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


JUDGE_FILES = [
    "data/evaluation/judge_scores_gpt4o.json",
    "data/evaluation/judge_scores_claude.json",
    "data/evaluation/judge_scores_gemini.json",
]


def _load_judge_scores_averaged() -> dict[tuple[str, str], dict]:
    """Load all 3 judge scores and average RA, keyed by (query_id, config)."""
    from collections import defaultdict as _dd
    ra_by_key: dict[tuple, list] = _dd(list)
    any_scores: dict[tuple, dict] = {}
    for jf in JUDGE_FILES:
        if not os.path.exists(jf):
            continue
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            key = (d["query_id"], d["config"])
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


def generate_table5(
    results_path: str = "data/evaluation/ablation_results.json",
) -> dict:
    """Generate Table 5: Per-Category Breakdown with quality metrics."""
    if not os.path.exists(results_path):
        return {"error": "No results found."}

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    judge_scores = _load_judge_scores_averaged()

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
