"""Table 2: Source ablation results.

Generates the main ablation table showing RA, CSAS, MSUR for all 7+ configs.
This is the centerpiece table demonstrating multi-source value.
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


def generate_table2(
    results_path: str = "data/evaluation/ablation_results.json",
    judge_path: str = "data/evaluation/judge_scores.json",
) -> dict:
    """Generate Table 2: Ablation Results with RA, CSAS, MSUR."""
    if not os.path.exists(results_path):
        return {"error": "No ablation results found. Run evaluation.ablation_runner first."}

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    judge_scores = _load_judge_scores(judge_path)

    # Group by config
    by_config: dict[str, list] = defaultdict(list)
    for r in results:
        if "error" not in r:
            by_config[r["config"]].append(r)

    table = {}
    for config, config_results in by_config.items():
        n = len(config_results)

        # RA scores from judge (if available)
        ra_scores = []
        for r in config_results:
            key = (r.get("query_id", ""), config)
            if key in judge_scores:
                ra_scores.append(judge_scores[key].get("ra", 0.0))

        # CSAS: compute from citations vs expected_sources
        csas_scores = []
        for r in config_results:
            citations = r.get("citations", {})
            expected = r.get("expected_sources", [])
            if expected:
                csas = compute_csas(citations, expected)
                csas_scores.append(csas.f1)

        # MSUR: multi-source utilization
        msur = compute_msur(config_results)

        # Source distribution in retrieved chunks
        source_counts = defaultdict(int)
        for r in config_results:
            for c in r.get("reranked_chunks", []):
                source_counts[c["source_type"]] += 1
        total_chunks = sum(source_counts.values())

        timing_avg = sum(
            r.get("timing", {}).get("total_ms", 0) for r in config_results
        ) / n

        table[config] = {
            "n_queries": n,
            "avg_ra": round(sum(ra_scores) / len(ra_scores), 3) if ra_scores else None,
            "avg_csas": round(sum(csas_scores) / len(csas_scores), 3) if csas_scores else None,
            "msur": round(msur, 3),
            "avg_retrieved": round(
                sum(r.get("retrieved_count", 0) for r in config_results) / n, 1
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
