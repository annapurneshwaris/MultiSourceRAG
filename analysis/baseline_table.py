"""Table 3: Baseline comparison results.

Compares HeteroRAG (DBW) against BM25, Naive, and single-source baselines.
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


def generate_table3(
    results_path: str = "data/evaluation/ablation_results.json",
    judge_path: str = "data/evaluation/judge_scores.json",
) -> dict:
    """Generate Table 3: Baseline Comparison with RA, CSAS, MSUR."""
    if not os.path.exists(results_path):
        return {"error": "No results found."}

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    judge_scores = _load_judge_scores(judge_path)

    baseline_configs = ["D", "B", "W", "BM25", "Naive", "DBW"]

    by_config = defaultdict(list)
    for r in results:
        if r.get("config") in baseline_configs and "error" not in r:
            by_config[r["config"]].append(r)

    table = {}
    for config, config_results in by_config.items():
        n = len(config_results)
        if n == 0:
            continue

        # RA from judge scores
        ra_scores = []
        for r in config_results:
            key = (r.get("query_id", ""), config)
            if key in judge_scores:
                ra_scores.append(judge_scores[key].get("ra", 0.0))

        # CSAS
        csas_scores = []
        for r in config_results:
            citations = r.get("citations", {})
            expected = r.get("expected_sources", [])
            if expected:
                csas = compute_csas(citations, expected)
                csas_scores.append(csas.f1)

        # MSUR
        msur = compute_msur(config_results)

        table[config] = {
            "n_queries": n,
            "avg_ra": round(sum(ra_scores) / len(ra_scores), 3) if ra_scores else None,
            "avg_csas": round(sum(csas_scores) / len(csas_scores), 3) if csas_scores else None,
            "msur": round(msur, 3),
            "avg_time_ms": round(
                sum(r.get("timing", {}).get("total_ms", 0) for r in config_results) / n, 1
            ),
            "avg_retrieved": round(
                sum(r.get("retrieved_count", 0) for r in config_results) / n, 1
            ),
        }

    # Compute delta DBW vs best single-source
    if "DBW" in table:
        dbw_ra = table["DBW"].get("avg_ra")
        single_ras = [
            table[c].get("avg_ra") for c in ["D", "B", "W"]
            if c in table and table[c].get("avg_ra") is not None
        ]
        if dbw_ra is not None and single_ras:
            best_single = max(single_ras)
            table["_delta_vs_best_single"] = round(dbw_ra - best_single, 3)

    return {"baselines": table}


if __name__ == "__main__":
    result = generate_table3()
    print(json.dumps(result, indent=2))
