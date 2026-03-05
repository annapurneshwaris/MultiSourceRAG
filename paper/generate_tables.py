"""STEP 2: Generate Tables 1-8 from master data frame.

Reads: paper/FINAL_NUMBERS.json, data files
Outputs: paper/tables/table{1..8}_*.json
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42
np.random.seed(SEED)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")


def paired_bootstrap(a_scores, b_scores, n_resamples=10000, seed=42):
    rng = np.random.RandomState(seed)
    a = np.array(a_scores, dtype=float)
    b = np.array(b_scores, dtype=float)
    observed_delta = float(np.mean(a) - np.mean(b))
    n = len(a)
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        deltas[i] = np.mean(a[idx]) - np.mean(b[idx])
    if observed_delta >= 0:
        p = float(np.mean(deltas <= 0))
    else:
        p = float(np.mean(deltas >= 0))
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    return observed_delta, p, ci_lower, ci_upper


def paired_t_test(a_scores, b_scores):
    from scipy import stats
    t_stat, p_val = stats.ttest_rel(a_scores, b_scores)
    return float(t_stat), float(p_val)


def cohens_d(a_scores, b_scores):
    a = np.array(a_scores, dtype=float)
    b = np.array(b_scores, dtype=float)
    n1, n2 = len(a), len(b)
    var1 = float(np.var(a, ddof=1))
    var2 = float(np.var(b, ddof=1))
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def get_paired(by_config, config_a, config_b, metric="ra"):
    a_by_qid = {e["query_id"]: e[metric] for e in by_config[config_a]}
    b_by_qid = {e["query_id"]: e[metric] for e in by_config[config_b]}
    common = sorted(set(a_by_qid.keys()) & set(b_by_qid.keys()))
    return [a_by_qid[q] for q in common], [b_by_qid[q] for q in common], common


METADATA = {
    "generated": datetime.now().strftime("%Y-%m-%d"),
    "dbw_filter": "router_type=heuristic for Tables 2/3/5, all variants for Table 4",
    "judges": ["gpt4o", "claude", "gemini"],
    "ra_method": "3-judge average of (RCI + AS + VM) / 6",
    "bootstrap_resamples": 10000,
    "random_seed": SEED,
}


# ===================================================================
# TABLE 1: Dataset Statistics
# ===================================================================
def generate_table1():
    print("\n--- TABLE 1: Dataset Statistics ---")

    stats = load_json("data/processed/processing_stats.json")

    # Load chunk files for token distribution
    sources = {
        "doc": load_json("data/processed/doc_chunks.json"),
        "bug": load_json("data/processed/bug_chunks.json"),
        "work_item": load_json("data/processed/workitem_chunks.json"),
    }

    rows = []
    for src_key, label, repo, structure in [
        ("doc", "Documentation", "microsoft/vscode-docs", "Unstructured"),
        ("bug", "Bug Reports", "microsoft/vscode", "Semi-structured"),
        ("work_item", "Work Items", "microsoft/vscode", "Structured"),
    ]:
        chunks = sources[src_key]
        # Compute token lengths (approximate: words)
        lengths = []
        for c in chunks:
            text = c.get("text", "")
            tok_count = len(text.split())
            lengths.append(tok_count)
        lengths = np.array(lengths) if lengths else np.array([0])

        # Raw items from stats
        raw_count_map = {
            "doc": 402,  # from processing stats / known
            "bug": 13913,
            "work_item": 3729,
        }

        # Feature area coverage
        fa_unknown = stats.get("feature_area_coverage", {}).get(f"{src_key}_unknown", 0)
        n_chunks = len(chunks)
        fa_coverage = round((n_chunks - fa_unknown) / n_chunks * 100, 1) if n_chunks > 0 else 0

        row = {
            "source": label,
            "repository": repo,
            "structure": structure,
            "raw_items": raw_count_map.get(src_key, 0),
            "chunks": n_chunks,
            "min_tokens": int(np.min(lengths)),
            "p25_tokens": int(np.percentile(lengths, 25)),
            "median_tokens": int(np.median(lengths)),
            "p75_tokens": int(np.percentile(lengths, 75)),
            "max_tokens": int(np.max(lengths)),
            "feature_area_coverage_pct": fa_coverage,
        }
        rows.append(row)

    # Total row
    all_chunks = sum(r["chunks"] for r in rows)
    all_raw = sum(r["raw_items"] for r in rows)

    # Unique feature areas
    all_areas = set()
    for src_key in ["doc", "bug", "work_item"]:
        for area_name, count in stats.get(f"{src_key}_areas", {}).items():
            if area_name != "unknown":
                all_areas.add(area_name)

    table1 = {
        "_metadata": METADATA,
        "rows": rows,
        "total": {
            "raw_items": all_raw,
            "chunks": all_chunks,
        },
        "additional": {
            "date_range": "2023-01-01 to 2025-12-31",
            "unique_feature_areas": len(all_areas),
            "feature_areas_list": sorted(all_areas),
            "query_bank": {
                "total": 250,
                "how_to": 60,
                "debugging": 75,
                "error_diagnosis": 50,
                "status_roadmap": 40,
                "config": 25,
            },
            "tree_nodes": stats.get("tree_nodes", 0),
        },
    }

    save_json(table1, "paper/tables/table1_dataset_stats.json")
    return table1


# ===================================================================
# TABLE 2: Source Ablation Results
# ===================================================================
def generate_table2(master):
    print("\n--- TABLE 2: Source Ablation Results ---")

    heuristic = [e for e in master if e["router_type"] == "heuristic"]
    by_config = defaultdict(list)
    for e in heuristic:
        by_config[e["config"]].append(e)

    config_order = ["DBW", "DB", "DW", "BW", "D", "B", "W", "BM25", "Naive"]
    rows = []

    for cfg in config_order:
        entries = by_config.get(cfg, [])
        if not entries:
            continue

        ra_vals = [e["ra"] for e in entries]
        csas_vals = [e["csas"] for e in entries]
        msur_vals = [e["msur"] for e in entries]
        retrieved_vals = [e["retrieved_count"] for e in entries]
        total_ms_vals = [e["timing"].get("total_ms", 0) for e in entries]

        row = {
            "config": cfg,
            "n": len(entries),
            "csas": round(float(np.mean(csas_vals)), 4),
            "csas_std": round(float(np.std(csas_vals)), 4),
            "ra": round(float(np.mean(ra_vals)), 4),
            "ra_std": round(float(np.std(ra_vals)), 4),
            "msur": round(float(np.mean(msur_vals)), 4),
            "retrieved": round(float(np.mean(retrieved_vals)), 1),
            "latency_ms": round(float(np.mean(total_ms_vals)), 0),
        }
        rows.append(row)

    # Bold best values
    best = {
        "csas": max(rows, key=lambda r: r["csas"])["config"],
        "ra": max(rows, key=lambda r: r["ra"])["config"],
        "msur": max(rows, key=lambda r: r["msur"])["config"],
    }

    # Significance markers vs D (paired bootstrap)
    sig_markers = {}
    for cfg in config_order:
        if cfg == "D":
            sig_markers[cfg] = ""
            continue
        a, b, common = get_paired(by_config, cfg, "D", "ra")
        if not common:
            sig_markers[cfg] = ""
            continue
        _, p, _, _ = paired_bootstrap(a, b)
        if p < 0.001:
            sig_markers[cfg] = "***"
        elif p < 0.01:
            sig_markers[cfg] = "**"
        elif p < 0.05:
            sig_markers[cfg] = "*"
        else:
            sig_markers[cfg] = ""

    for row in rows:
        row["ra_sig_vs_d"] = sig_markers.get(row["config"], "")

    table2 = {
        "_metadata": METADATA,
        "rows": rows,
        "best_values": best,
        "significance_note": "* p<0.05, ** p<0.01, *** p<0.001 vs D (paired bootstrap, 10K resamples)",
    }

    save_json(table2, "paper/tables/table2_ablation.json")
    return table2


# ===================================================================
# TABLE 3: Significance Tests
# ===================================================================
def generate_table3(master):
    print("\n--- TABLE 3: Significance Tests ---")

    heuristic = [e for e in master if e["router_type"] == "heuristic"]
    by_config = defaultdict(list)
    for e in heuristic:
        by_config[e["config"]].append(e)

    comparisons_list = [
        ("DBW", "D"), ("DBW", "B"), ("DBW", "W"),
        ("DBW", "DB"), ("DBW", "DW"), ("DBW", "BW"),
        ("DBW", "BM25"), ("DBW", "Naive"), ("BM25", "D"),
    ]

    results = []
    all_ra_p = []
    all_csas_p = []

    for cfg_a, cfg_b in comparisons_list:
        entry = {"comparison": f"{cfg_a} vs {cfg_b}"}

        for metric in ["ra", "csas"]:
            a, b, common = get_paired(by_config, cfg_a, cfg_b, metric)
            if len(common) < 10:
                entry[metric] = {"error": f"too few paired samples: {len(common)}"}
                continue

            delta, boot_p, ci_lo, ci_hi = paired_bootstrap(a, b)
            d_eff = cohens_d(a, b)
            t_stat, t_p = paired_t_test(a, b)

            entry[metric] = {
                "delta": round(delta, 4),
                "bootstrap_p": round(boot_p, 6),
                "ci_lower": round(ci_lo, 4),
                "ci_upper": round(ci_hi, 4),
                "effect_size_cohens_d": round(d_eff, 3),
                "t_test_p": round(t_p, 6),
                "significant_at_05": boot_p < 0.05,
                "n_paired": len(common),
            }

            if metric == "ra":
                all_ra_p.append((f"{cfg_a} vs {cfg_b}", boot_p))
            else:
                all_csas_p.append((f"{cfg_a} vs {cfg_b}", boot_p))

        results.append(entry)

    # Holm-Bonferroni correction
    def holm_bonferroni(p_values_list):
        """Apply Holm-Bonferroni correction. Input: list of (label, p)."""
        sorted_ps = sorted(p_values_list, key=lambda x: x[1])
        m = len(sorted_ps)
        corrected = {}
        for rank, (label, p) in enumerate(sorted_ps):
            corrected_p = min(1.0, p * (m - rank))
            corrected[label] = {
                "original_p": round(p, 6),
                "corrected_p": round(corrected_p, 6),
                "significant_after_correction": corrected_p < 0.05,
            }
        return corrected

    holm_ra = holm_bonferroni(all_ra_p)
    holm_csas = holm_bonferroni(all_csas_p)

    # Add Holm correction results to entries
    for entry in results:
        comp = entry["comparison"]
        if "ra" in entry and isinstance(entry["ra"], dict) and "delta" in entry["ra"]:
            h = holm_ra.get(comp, {})
            entry["ra"]["holm_corrected_p"] = h.get("corrected_p", None)
            entry["ra"]["significant_after_holm"] = h.get("significant_after_correction", False)
        if "csas" in entry and isinstance(entry["csas"], dict) and "delta" in entry["csas"]:
            h = holm_csas.get(comp, {})
            entry["csas"]["holm_corrected_p"] = h.get("corrected_p", None)
            entry["csas"]["significant_after_holm"] = h.get("significant_after_correction", False)

    table3 = {
        "_metadata": METADATA,
        "comparisons": results,
        "holm_bonferroni_ra": holm_ra,
        "holm_bonferroni_csas": holm_csas,
        "significance_markers": "* p<0.05, ** p<0.01, *** p<0.001 (uncorrected). † significant after Holm correction.",
    }

    save_json(table3, "paper/tables/table3_significance.json")
    return table3


# ===================================================================
# TABLE 4: Router Comparison
# ===================================================================
def generate_table4(master):
    print("\n--- TABLE 4: Router Comparison ---")

    dbw = [e for e in master if e["config"] == "DBW"]
    by_rt = defaultdict(list)
    for e in dbw:
        by_rt[e["router_type"]].append(e)

    rows = []
    for rt in ["heuristic", "adaptive", "llm_zeroshot"]:
        entries = by_rt.get(rt, [])
        if not entries:
            continue

        ra_vals = [e["ra"] for e in entries]
        csas_vals = [e["csas"] for e in entries]
        msur_vals = [e["msur"] for e in entries]
        route_ms = [e["timing"].get("route_ms", 0) for e in entries]
        total_ms = [e["timing"].get("total_ms", 0) for e in entries]

        # Average source boosts
        avg_boosts = {"doc": [], "bug": [], "work_item": []}
        for e in entries:
            for src in avg_boosts:
                avg_boosts[src].append(e.get("source_boosts", {}).get(src, 0))

        row = {
            "router_type": rt,
            "n": len(entries),
            "ra": round(float(np.mean(ra_vals)), 4),
            "ra_std": round(float(np.std(ra_vals)), 4),
            "csas": round(float(np.mean(csas_vals)), 4),
            "msur": round(float(np.mean(msur_vals)), 4),
            "route_ms_mean": round(float(np.mean(route_ms)), 1),
            "route_ms_p50": round(float(np.median(route_ms)), 1),
            "total_ms_mean": round(float(np.mean(total_ms)), 0),
            "avg_source_boosts": {k: round(float(np.mean(v)), 4) for k, v in avg_boosts.items()},
        }
        rows.append(row)

    table4 = {
        "_metadata": METADATA,
        "_note": "Adaptive and LLM-zero-shot results use pre-fix reranker. Heuristic uses fixed reranker with hybrid BM25 signal and router-aware diversity. Direct RA comparison across routers is confounded by reranker version. Table primarily demonstrates routing latency tradeoffs and source boost distributions.",
        "recommendation": "Present in paper as supplementary/appendix with caveat, OR reframe as 'Impact of Routing Strategy and Reranker Logic' showing reranker matters more than router choice.",
        "rows": rows,
    }

    save_json(table4, "paper/tables/table4_router.json")
    return table4


# ===================================================================
# TABLE 5: Per-Category Breakdown
# ===================================================================
def generate_table5(master):
    print("\n--- TABLE 5: Per-Category Breakdown ---")

    heuristic = [e for e in master if e["router_type"] == "heuristic"]
    by_config = defaultdict(list)
    for e in heuristic:
        by_config[e["config"]].append(e)

    categories = ["how_to", "debugging", "error_diagnosis", "status_roadmap", "config"]
    configs_compare = ["DBW", "D", "B", "W", "BM25"]

    # 5a: DBW heuristic per category
    dbw_entries = by_config["DBW"]
    dbw_by_cat = defaultdict(list)
    for e in dbw_entries:
        dbw_by_cat[e["category"]].append(e)

    table5a = []
    for cat in categories:
        entries = dbw_by_cat.get(cat, [])
        if not entries:
            continue

        # Dominant source
        src_counts = {"doc": 0, "bug": 0, "work_item": 0}
        for e in entries:
            for src, cnt in e.get("source_distribution", {}).items():
                src_counts[src] = src_counts.get(src, 0) + cnt
        total_chunks = sum(src_counts.values())
        dominant = max(src_counts, key=src_counts.get) if total_chunks > 0 else "none"
        dominant_pct = round(src_counts[dominant] / total_chunks * 100, 1) if total_chunks > 0 else 0

        # Citation leader
        cite_counts = {"doc": 0, "bug": 0, "work_item": 0}
        for e in entries:
            for src, cnt in e.get("citation_counts", {}).items():
                cite_counts[src] = cite_counts.get(src, 0) + cnt
        cite_leader = max(cite_counts, key=cite_counts.get) if sum(cite_counts.values()) > 0 else "none"

        row = {
            "category": cat,
            "n": len(entries),
            "ra": round(float(np.mean([e["ra"] for e in entries])), 4),
            "csas": round(float(np.mean([e["csas"] for e in entries])), 4),
            "msur": round(float(np.mean([e["msur"] for e in entries])), 4),
            "dominant_source": dominant,
            "dominant_source_pct": dominant_pct,
            "citation_leader": cite_leader,
        }
        table5a.append(row)

    # 5b: Cross-config per category
    table5b = []
    for cat in categories:
        row = {"category": cat}
        for cfg in configs_compare:
            entries = [e for e in by_config.get(cfg, []) if e["category"] == cat]
            row[f"{cfg}_ra"] = round(float(np.mean([e["ra"] for e in entries])), 4) if entries else None
            row[f"{cfg}_n"] = len(entries)

        # Delta: DBW minus best single-source
        single_ras = []
        for cfg in ["D", "B", "W"]:
            val = row.get(f"{cfg}_ra")
            if val is not None:
                single_ras.append(val)
        best_single = max(single_ras) if single_ras else 0
        dbw_val = row.get("DBW_ra", 0) or 0
        row["delta_vs_best_single"] = round(dbw_val - best_single, 4)
        table5b.append(row)

    # 5c: Per-category significance (DBW vs D)
    table5c = []
    for cat in categories:
        dbw_cat = [e for e in by_config["DBW"] if e["category"] == cat]
        d_cat = [e for e in by_config["D"] if e["category"] == cat]

        dbw_by_qid = {e["query_id"]: e["ra"] for e in dbw_cat}
        d_by_qid = {e["query_id"]: e["ra"] for e in d_cat}
        common = sorted(set(dbw_by_qid.keys()) & set(d_by_qid.keys()))

        if len(common) >= 5:
            a = [dbw_by_qid[q] for q in common]
            b = [d_by_qid[q] for q in common]
            delta, p, ci_lo, ci_hi = paired_bootstrap(a, b)
            row = {
                "category": cat,
                "n_paired": len(common),
                "dbw_ra": round(float(np.mean(a)), 4),
                "d_ra": round(float(np.mean(b)), 4),
                "delta": round(delta, 4),
                "bootstrap_p": round(p, 4),
                "ci_lower": round(ci_lo, 4),
                "ci_upper": round(ci_hi, 4),
                "significant": p < 0.05,
            }
        else:
            row = {"category": cat, "n_paired": len(common), "error": "too few paired samples"}
        table5c.append(row)

    table5 = {
        "_metadata": METADATA,
        "table5a_dbw_per_category": table5a,
        "table5b_cross_config": table5b,
        "table5c_per_category_significance": table5c,
    }

    save_json(table5, "paper/tables/table5_per_category.json")
    return table5


# ===================================================================
# TABLE 6: Inter-Judge Agreement
# ===================================================================
def generate_table6():
    print("\n--- TABLE 6: Inter-Judge Agreement ---")
    from scipy import stats as scipy_stats

    judges = {}
    for name in ["gpt4o", "claude", "gemini"]:
        data = load_json(f"data/evaluation/judge_scores_{name}.json")
        judges[name] = {(e["query_id"], e["config"], e.get("router_type", "heuristic")): e for e in data}

    pairs = [("gpt4o", "claude"), ("gpt4o", "gemini"), ("claude", "gemini")]

    def quadratic_weighted_kappa(a, b, max_val):
        """Compute quadratic weighted Cohen's kappa."""
        # Build confusion matrix
        n_cat = max_val + 1
        confusion = np.zeros((n_cat, n_cat), dtype=float)
        for ai, bi in zip(a, b):
            ai_int = min(max(int(round(ai)), 0), max_val)
            bi_int = min(max(int(round(bi)), 0), max_val)
            confusion[ai_int][bi_int] += 1

        n = confusion.sum()
        if n == 0:
            return 0.0

        # Weight matrix (quadratic)
        weights = np.zeros((n_cat, n_cat))
        for i in range(n_cat):
            for j in range(n_cat):
                weights[i][j] = (i - j) ** 2 / (max_val ** 2) if max_val > 0 else 0

        # Expected matrix
        row_sum = confusion.sum(axis=1)
        col_sum = confusion.sum(axis=0)
        expected = np.outer(row_sum, col_sum) / n

        observed_weighted = np.sum(weights * confusion) / n
        expected_weighted = np.sum(weights * expected) / n

        if expected_weighted == 0:
            return 1.0
        return float(1 - observed_weighted / expected_weighted)

    def bootstrap_ci(a_vals, b_vals, func, n_boot=10000, seed=42):
        rng = np.random.RandomState(seed)
        a = np.array(a_vals)
        b = np.array(b_vals)
        n = len(a)
        boots = []
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            try:
                boots.append(func(a[idx], b[idx]))
            except Exception:
                pass
        if not boots:
            return 0, 0
        return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    pair_results = []
    for j1, j2 in pairs:
        common_keys = sorted(set(judges[j1].keys()) & set(judges[j2].keys()))
        ra_1 = [judges[j1][k]["ra"] for k in common_keys]
        ra_2 = [judges[j2][k]["ra"] for k in common_keys]
        rci_1 = [judges[j1][k]["rci"] for k in common_keys]
        rci_2 = [judges[j2][k]["rci"] for k in common_keys]
        as_1 = [judges[j1][k]["as"] for k in common_keys]
        as_2 = [judges[j2][k]["as"] for k in common_keys]
        vm_1 = [judges[j1][k]["vm"] for k in common_keys]
        vm_2 = [judges[j2][k]["vm"] for k in common_keys]

        pearson_r, _ = scipy_stats.pearsonr(ra_1, ra_2)
        spearman_rho, _ = scipy_stats.spearmanr(ra_1, ra_2)

        # Bootstrap CIs for correlations
        def pearson_func(a, b):
            return scipy_stats.pearsonr(a, b)[0]
        def spearman_func(a, b):
            return scipy_stats.spearmanr(a, b)[0]

        pearson_ci = bootstrap_ci(ra_1, ra_2, pearson_func)
        spearman_ci = bootstrap_ci(ra_1, ra_2, spearman_func)

        kappa_rci = quadratic_weighted_kappa(rci_1, rci_2, 2)
        kappa_as = quadratic_weighted_kappa(as_1, as_2, 2)
        kappa_vm = quadratic_weighted_kappa(vm_1, vm_2, 2)

        pair_results.append({
            "pair": f"{j1} vs {j2}",
            "n_paired": len(common_keys),
            "pearson_r": round(pearson_r, 4),
            "pearson_ci": [round(pearson_ci[0], 4), round(pearson_ci[1], 4)],
            "spearman_rho": round(spearman_rho, 4),
            "spearman_ci": [round(spearman_ci[0], 4), round(spearman_ci[1], 4)],
            "kappa_rci": round(kappa_rci, 4),
            "kappa_as": round(kappa_as, 4),
            "kappa_vm": round(kappa_vm, 4),
        })

    # Averages
    avg = {
        "pearson_r": round(float(np.mean([r["pearson_r"] for r in pair_results])), 4),
        "spearman_rho": round(float(np.mean([r["spearman_rho"] for r in pair_results])), 4),
        "kappa_rci": round(float(np.mean([r["kappa_rci"] for r in pair_results])), 4),
        "kappa_as": round(float(np.mean([r["kappa_as"] for r in pair_results])), 4),
        "kappa_vm": round(float(np.mean([r["kappa_vm"] for r in pair_results])), 4),
    }

    table6 = {
        "_metadata": METADATA,
        "pairs": pair_results,
        "averages": avg,
    }

    save_json(table6, "paper/tables/table6_inter_judge.json")
    return table6


# ===================================================================
# TABLE 7: Metric Definitions & Judge Rubric
# ===================================================================
def generate_table7():
    print("\n--- TABLE 7: Metric Definitions ---")

    table7 = {
        "_metadata": METADATA,
        "metrics": [
            {
                "name": "RA (Resolution Accuracy)",
                "formula": "(RCI + AS + VM) / 6",
                "range": "[0, 1]",
                "components": [
                    {"name": "RCI (Technical Depth)", "scale": "0-2", "description": "0=no mechanism, 1=correct area, 2=specific mechanism"},
                    {"name": "AS (Actionable Steps)", "scale": "0-2", "description": "0=no steps, 1=vague, 2=specific commands/settings"},
                    {"name": "VM (Version Match)", "scale": "0-2", "description": "0=wrong version, 1=general, 2=version-specific"},
                ],
            },
            {
                "name": "CSAS (Cross-Source Attribution Score)",
                "formula": "F1(precision, recall) of source-type citations vs expected sources",
                "range": "[0, 1]",
                "description": "Measures whether the answer cites the correct type of source",
            },
            {
                "name": "MSUR (Multi-Source Utilization Rate)",
                "formula": "avg fraction of {doc, bug, work_item} present in top-k per query",
                "range": "[0, 1]",
                "description": "Higher = more source diversity in retrieved chunks",
            },
        ],
        "judge_setup": {
            "models": ["GPT-4o (2024-11-20)", "Claude Sonnet 4", "Gemini 2.5 Flash"],
            "prompt_type": "Depth-based RCI with 4 few-shot examples per dimension",
            "temperature": 0.0,
            "scoring": "3-judge average",
            "root_cause_taxonomy": ["configuration", "bug_or_issue", "gap_or_missing", "unknown"],
        },
    }

    save_json(table7, "paper/tables/table7_metric_definitions.json")
    return table7


# ===================================================================
# TABLE 8: Latency Breakdown
# ===================================================================
def generate_table8(master):
    print("\n--- TABLE 8: Latency Breakdown ---")

    dbw_h = [e for e in master if e["config"] == "DBW" and e["router_type"] == "heuristic"]

    stages = ["route_ms", "retrieve_ms", "rerank_ms", "generate_ms", "total_ms"]
    stage_labels = {
        "route_ms": "Routing",
        "retrieve_ms": "Retrieval",
        "rerank_ms": "Reranking",
        "generate_ms": "Generation",
        "total_ms": "Total",
    }

    rows = []
    for stage in stages:
        vals = [e["timing"].get(stage, 0) for e in dbw_h if e["timing"].get(stage, 0) > 0]
        if not vals:
            continue
        arr = np.array(vals)
        rows.append({
            "stage": stage_labels.get(stage, stage),
            "stage_key": stage,
            "mean_ms": round(float(np.mean(arr)), 1),
            "p50_ms": round(float(np.median(arr)), 1),
            "p95_ms": round(float(np.percentile(arr, 95)), 1),
            "min_ms": round(float(np.min(arr)), 1),
            "max_ms": round(float(np.max(arr)), 1),
            "n": len(vals),
        })

    # Embed stage (separate since it's step 0)
    embed_vals = [e["timing"].get("embed_ms", 0) for e in dbw_h if e["timing"].get("embed_ms", 0) > 0]
    if embed_vals:
        arr = np.array(embed_vals)
        rows.insert(0, {
            "stage": "Embedding",
            "stage_key": "embed_ms",
            "mean_ms": round(float(np.mean(arr)), 1),
            "p50_ms": round(float(np.median(arr)), 1),
            "p95_ms": round(float(np.percentile(arr, 95)), 1),
            "min_ms": round(float(np.min(arr)), 1),
            "max_ms": round(float(np.max(arr)), 1),
            "n": len(embed_vals),
        })

    table8 = {
        "_metadata": METADATA,
        "config": "DBW heuristic",
        "n": len(dbw_h),
        "rows": rows,
    }

    save_json(table8, "paper/tables/table8_latency.json")
    return table8


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("=" * 60)
    print("STEP 2: Generating Tables 1-8")
    print("=" * 60)

    # Load master
    final = load_json("paper/FINAL_NUMBERS.json")
    master = final["master_entries"]
    print(f"Master entries: {len(master)}")

    generate_table1()
    generate_table2(master)
    generate_table3(master)
    generate_table4(master)
    generate_table5(master)
    generate_table6()
    generate_table7()
    generate_table8(master)

    print("\n" + "=" * 60)
    print("All 8 tables generated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
