"""Table 2: Source ablation results.

Generates the main ablation table showing RA, CSAS, MSUR for all 7+ configs.
This is the centerpiece table demonstrating multi-source value.
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
    ra_by_key: dict[tuple, list[float]] = _dd(list)
    rci_by_key: dict[tuple, list[int]] = _dd(list)
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
            if d.get("rci") is not None:
                rci_by_key[key].append(d["rci"])
            any_scores[key] = d
    # Build averaged dict
    result = {}
    for key in any_scores:
        entry = dict(any_scores[key])
        if key in ra_by_key:
            entry["ra"] = sum(ra_by_key[key]) / len(ra_by_key[key])
        if key in rci_by_key:
            entry["rci"] = round(sum(rci_by_key[key]) / len(rci_by_key[key]), 2)
        result[key] = entry
    return result


def generate_table2(
    results_path: str = "data/evaluation/ablation_results.json",
) -> dict:
    """Generate Table 2: Ablation Results with RA, CSAS, MSUR."""
    if not os.path.exists(results_path):
        return {"error": "No ablation results found. Run evaluation.ablation_runner first."}

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    judge_scores = _load_judge_scores_averaged()

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
