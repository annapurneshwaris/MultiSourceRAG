"""STEP 1: Build master data frame — single source of truth for all paper tables.

Output: paper/FINAL_NUMBERS.json
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42
np.random.seed(SEED)

METADATA = {
    "generated": datetime.now().strftime("%Y-%m-%d"),
    "dbw_filter": "router_type=heuristic for Tables 2/3/5, all variants for Table 4",
    "judges": ["gpt4o", "claude", "gemini"],
    "ra_method": "3-judge average of (RCI + AS + VM) / 6",
    "bootstrap_resamples": 10000,
    "random_seed": SEED,
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")


def is_error_entry(result):
    """Check if an ablation result is an error entry."""
    if "error" in result:
        return True
    answer = result.get("answer", "")
    if "429" in answer or "Generation error" in answer:
        return True
    if len(answer) < 20:
        return True
    return False


def compute_csas(citations, expected_sources):
    """Compute CSAS F1 from citations dict and expected source list."""
    if not expected_sources:
        return 0.0

    # What sources were actually cited
    cited_sources = set()
    if isinstance(citations, dict):
        for src_type, ids in citations.items():
            if ids:  # non-empty list
                # Map source type names
                if src_type in ("doc", "bug", "work_item"):
                    cited_sources.add(src_type)

    expected_set = set(expected_sources)

    if not cited_sources and not expected_set:
        return 1.0
    if not cited_sources or not expected_set:
        return 0.0

    tp = len(cited_sources & expected_set)
    precision = tp / len(cited_sources) if cited_sources else 0
    recall = tp / len(expected_set) if expected_set else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_msur(reranked_chunks):
    """Compute MSUR: fraction of {doc, bug, work_item} present in chunks."""
    if not reranked_chunks:
        return 0.0
    all_source_types = {"doc", "bug", "work_item"}
    present = set()
    for c in reranked_chunks:
        st = c.get("source_type", "")
        if st in all_source_types:
            present.add(st)
    return len(present) / len(all_source_types)


def paired_bootstrap(a_scores, b_scores, n_resamples=10000, seed=42):
    """Paired bootstrap test. Returns p-value and 95% CI of delta."""
    rng = np.random.RandomState(seed)
    a = np.array(a_scores)
    b = np.array(b_scores)
    observed_delta = np.mean(a) - np.mean(b)

    n = len(a)
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        deltas[i] = np.mean(a[idx]) - np.mean(b[idx])

    # Two-sided p-value
    if observed_delta >= 0:
        p = np.mean(deltas <= 0)
    else:
        p = np.mean(deltas >= 0)

    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))

    return float(observed_delta), float(p), ci_lower, ci_upper


def cohens_d(a_scores, b_scores):
    """Compute Cohen's d effect size."""
    a = np.array(a_scores)
    b = np.array(b_scores)
    n1, n2 = len(a), len(b)
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def paired_t_test(a_scores, b_scores):
    """Paired t-test. Returns t-statistic and p-value."""
    from scipy import stats
    t_stat, p_val = stats.ttest_rel(a_scores, b_scores)
    return float(t_stat), float(p_val)


def main():
    print("=" * 60)
    print("STEP 1: Building Master Data Frame")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 1. Load all data
    # ---------------------------------------------------------------
    print("\n[1/5] Loading data...")
    ablation = load_json("data/evaluation/ablation_results.json")
    print(f"  Ablation results: {len(ablation)}")

    judges = {}
    for name in ["gpt4o", "claude", "gemini"]:
        judges[name] = load_json(f"data/evaluation/judge_scores_{name}.json")
        print(f"  Judge {name}: {len(judges[name])} scores")

    # ---------------------------------------------------------------
    # 2. Build judge score lookup: (query_id, config, router_type) → averaged scores
    # ---------------------------------------------------------------
    print("\n[2/5] Building 3-judge averaged scores...")

    # Collect per-key scores from each judge
    judge_scores_by_key = defaultdict(lambda: {"ra": [], "rci": [], "as": [], "vm": []})

    for name, jdata in judges.items():
        for entry in jdata:
            key = (entry["query_id"], entry["config"], entry.get("router_type", "heuristic"))
            judge_scores_by_key[key]["ra"].append(entry["ra"])
            judge_scores_by_key[key]["rci"].append(entry["rci"])
            judge_scores_by_key[key]["as"].append(entry["as"])
            judge_scores_by_key[key]["vm"].append(entry["vm"])

    # Average across judges
    averaged_scores = {}
    for key, scores in judge_scores_by_key.items():
        averaged_scores[key] = {
            "ra": round(np.mean(scores["ra"]), 6),
            "rci": round(np.mean(scores["rci"]), 6),
            "as": round(np.mean(scores["as"]), 6),
            "vm": round(np.mean(scores["vm"]), 6),
            "n_judges": len(scores["ra"]),
        }

    print(f"  Unique (query_id, config, router_type) entries: {len(averaged_scores)}")

    # ---------------------------------------------------------------
    # 3. Merge ablation results with judge scores
    # ---------------------------------------------------------------
    print("\n[3/5] Merging ablation + judge scores...")

    master = []
    skipped_error = 0
    skipped_no_judge = 0

    for result in ablation:
        if is_error_entry(result):
            skipped_error += 1
            continue

        key = (result["query_id"], result["config"], result.get("router_type", "heuristic"))
        if key not in averaged_scores:
            skipped_no_judge += 1
            continue

        scores = averaged_scores[key]
        csas = compute_csas(result.get("citations", {}), result.get("expected_sources", []))
        msur = compute_msur(result.get("reranked_chunks", []))

        # Source distribution in reranked chunks
        source_dist = {"doc": 0, "bug": 0, "work_item": 0}
        for c in result.get("reranked_chunks", []):
            st = c.get("source_type", "")
            if st in source_dist:
                source_dist[st] += 1
        n_chunks = sum(source_dist.values())
        source_pct = {k: round(v / n_chunks, 4) if n_chunks > 0 else 0 for k, v in source_dist.items()}

        # Citation counts
        citation_counts = {}
        for src_type, ids in result.get("citations", {}).items():
            citation_counts[src_type] = len(ids) if isinstance(ids, list) else 0

        entry = {
            "query_id": result["query_id"],
            "config": result["config"],
            "router_type": result.get("router_type", "heuristic"),
            "category": result.get("category", ""),
            "expected_sources": result.get("expected_sources", []),
            "expected_area": result.get("expected_area", ""),
            "difficulty": result.get("difficulty", ""),
            # Judge scores (3-judge averaged)
            "ra": scores["ra"],
            "rci": scores["rci"],
            "as": scores["as"],
            "vm": scores["vm"],
            "n_judges": scores["n_judges"],
            # Computed metrics
            "csas": round(csas, 6),
            "msur": round(msur, 6),
            # Retrieval info
            "retrieved_count": result.get("retrieved_count", 0),
            "reranked_count": len(result.get("reranked_chunks", [])),
            "source_boosts": result.get("source_boosts", {}),
            "source_distribution": source_dist,
            "source_pct": source_pct,
            "citation_counts": citation_counts,
            # Timing
            "timing": result.get("timing", {}),
            # Answer excerpt
            "answer_excerpt": result.get("answer", "")[:300],
        }
        master.append(entry)

    print(f"  Master entries: {len(master)}")
    print(f"  Skipped (errors): {skipped_error}")
    print(f"  Skipped (no judge): {skipped_no_judge}")

    # ---------------------------------------------------------------
    # 4. Compute headline results for verification
    # ---------------------------------------------------------------
    print("\n[4/5] Computing headline results...")

    # Filter to heuristic only for main comparisons
    heuristic = [e for e in master if e["router_type"] == "heuristic"]

    # Group by config
    by_config = defaultdict(list)
    for e in heuristic:
        by_config[e["config"]].append(e)

    # Print counts
    for cfg in ["DBW", "D", "B", "W", "DB", "DW", "BW", "BM25", "Naive"]:
        entries = by_config.get(cfg, [])
        avg_ra = np.mean([e["ra"] for e in entries]) if entries else 0
        avg_csas = np.mean([e["csas"] for e in entries]) if entries else 0
        print(f"  {cfg:6s}: n={len(entries):3d}, RA={avg_ra:.4f}, CSAS={avg_csas:.4f}")

    # Key comparisons
    def get_paired_scores(config_a, config_b, metric="ra"):
        """Get paired scores for two configs matching on query_id."""
        a_by_qid = {e["query_id"]: e[metric] for e in by_config[config_a]}
        b_by_qid = {e["query_id"]: e[metric] for e in by_config[config_b]}
        common = sorted(set(a_by_qid.keys()) & set(b_by_qid.keys()))
        return [a_by_qid[q] for q in common], [b_by_qid[q] for q in common], common

    # DBW vs D
    a, b, common = get_paired_scores("DBW", "D", "ra")
    delta, p, ci_lo, ci_hi = paired_bootstrap(a, b)
    d_effect = cohens_d(a, b)
    print(f"\n  DBW vs D (RA): delta={delta:.4f}, p={p:.4f}, CI=[{ci_lo:.4f}, {ci_hi:.4f}], d={d_effect:.3f}, n_paired={len(common)}")

    # DBW vs BM25
    a, b, common = get_paired_scores("DBW", "BM25", "ra")
    delta_bm25, p_bm25, ci_lo_bm25, ci_hi_bm25 = paired_bootstrap(a, b)
    print(f"  DBW vs BM25 (RA): delta={delta_bm25:.4f}, p={p_bm25:.4f}")

    # CSAS comparisons
    a, b, common = get_paired_scores("DBW", "D", "csas")
    delta_csas_d, p_csas_d, _, _ = paired_bootstrap(a, b)
    print(f"  DBW vs D (CSAS): delta={delta_csas_d:.4f}, p={p_csas_d:.6f}")

    a, b, common = get_paired_scores("DBW", "BM25", "csas")
    delta_csas_bm25, p_csas_bm25, _, _ = paired_bootstrap(a, b)
    print(f"  DBW vs BM25 (CSAS): delta={delta_csas_bm25:.4f}, p={p_csas_bm25:.6f}")

    # ---------------------------------------------------------------
    # 5. Save master + headline
    # ---------------------------------------------------------------
    print("\n[5/5] Saving FINAL_NUMBERS.json...")

    dbw_ra = round(np.mean([e["ra"] for e in by_config["DBW"]]), 3)
    d_ra = round(np.mean([e["ra"] for e in by_config["D"]]), 3)
    bm25_ra = round(np.mean([e["ra"] for e in by_config["BM25"]]), 3)

    headline = {
        "dbw_heuristic_ra": dbw_ra,
        "d_ra": d_ra,
        "bm25_ra": bm25_ra,
        "dbw_vs_d_ra_delta": round(delta, 3),
        "dbw_vs_d_ra_p": round(p, 4),
        "dbw_vs_bm25_ra_delta": round(delta_bm25, 3),
        "dbw_vs_bm25_ra_p": round(p_bm25, 4),
        "dbw_vs_d_csas_delta": round(delta_csas_d, 3),
        "dbw_vs_d_csas_p": round(p_csas_d, 6),
        "dbw_vs_bm25_csas_delta": round(delta_csas_bm25, 3),
        "dbw_vs_bm25_csas_p": round(p_csas_bm25, 6),
    }

    output = {
        "_metadata": METADATA,
        "headline_results": headline,
        "master_entries": master,
        "summary": {
            "total_entries": len(master),
            "heuristic_entries": len(heuristic),
            "configs": {cfg: len(entries) for cfg, entries in by_config.items()},
            "error_entries_excluded": skipped_error,
            "no_judge_entries_excluded": skipped_no_judge,
        },
    }

    save_json(output, "paper/FINAL_NUMBERS.json")

    # Print headline for verification
    print("\n" + "=" * 60)
    print("HEADLINE RESULTS (verify these match expectations)")
    print("=" * 60)
    for k, v in headline.items():
        print(f"  {k}: {v}")

    return output


if __name__ == "__main__":
    main()
