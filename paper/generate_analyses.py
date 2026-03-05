"""STEP 4: Generate Analyses 1-5.

Reads: paper/FINAL_NUMBERS.json
Outputs: paper/analyses/analysis{1..5}_*.json
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

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
    return observed_delta, p


# ===================================================================
# ANALYSIS 1: Citation Error Taxonomy
# ===================================================================
def generate_analysis1(master):
    print("\n--- ANALYSIS 1: Citation Error Taxonomy ---")

    dbw_h = [e for e in master if e["config"] == "DBW" and e["router_type"] == "heuristic"]

    # Find queries where RA >= 0.7 but CSAS < 0.5
    high_ra_low_csas = [e for e in dbw_h if e["ra"] >= 0.7 and e["csas"] < 0.5]
    print(f"  High RA (>=0.7), Low CSAS (<0.5): {len(high_ra_low_csas)} queries")

    errors = []
    error_type_counts = defaultdict(int)

    for e in high_ra_low_csas:
        expected = set(e.get("expected_sources", []))
        # Determine actually cited sources
        cited = set()
        for src, cnt in e.get("citation_counts", {}).items():
            if cnt > 0:
                cited.add(src)

        missing = expected - cited
        unexpected = cited - expected

        # Classify error type
        if missing and not unexpected:
            error_type = "expected_source_missing"
        elif unexpected and not missing:
            error_type = "wrong_source_cited"
        elif len(cited) == 3 and len(expected) <= 1:
            error_type = "over_attribution"
        elif missing and unexpected:
            error_type = "wrong_source_cited"
        elif not cited:
            error_type = "no_citations"
        else:
            error_type = "other"

        error_type_counts[error_type] += 1
        errors.append({
            "query_id": e["query_id"],
            "query_text": e.get("answer_excerpt", "")[:100] + "...",
            "category": e["category"],
            "ra": e["ra"],
            "csas": e["csas"],
            "expected_sources": list(expected),
            "cited_sources": list(cited),
            "missing_sources": list(missing),
            "unexpected_sources": list(unexpected),
            "error_type": error_type,
        })

    # Summary table
    error_summary = []
    for etype in ["expected_source_missing", "wrong_source_cited", "over_attribution", "no_citations", "other"]:
        count = error_type_counts.get(etype, 0)
        if count > 0:
            example = next((e for e in errors if e["error_type"] == etype), None)
            error_summary.append({
                "type": etype,
                "count": count,
                "example_query_id": example["query_id"] if example else None,
            })

    analysis1 = {
        "_metadata": METADATA,
        "total_high_ra_low_csas": len(high_ra_low_csas),
        "error_types": error_summary,
        "detailed_errors": errors[:30],  # Top 30
    }

    save_json(analysis1, "paper/analyses/analysis1_citation_errors.json")
    return analysis1


# ===================================================================
# ANALYSIS 2: Query Difficulty Split
# ===================================================================
def generate_analysis2(master):
    print("\n--- ANALYSIS 2: Query Difficulty Split ---")

    heuristic = [e for e in master if e["router_type"] == "heuristic"]
    by_config = defaultdict(list)
    for e in heuristic:
        by_config[e["config"]].append(e)

    # D-only RA as difficulty proxy
    d_by_qid = {e["query_id"]: e["ra"] for e in by_config["D"]}
    dbw_by_qid = {e["query_id"]: e["ra"] for e in by_config["DBW"]}

    common = sorted(set(d_by_qid.keys()) & set(dbw_by_qid.keys()))
    d_scores = np.array([d_by_qid[q] for q in common])

    # Split into quartiles based on D RA
    quartile_bounds = [0, np.percentile(d_scores, 25), np.percentile(d_scores, 50),
                       np.percentile(d_scores, 75), 1.01]
    labels = ["Very Hard (Q1)", "Hard (Q2)", "Medium (Q3)", "Easy (Q4)"]

    quartiles = []
    for i in range(4):
        lo, hi = quartile_bounds[i], quartile_bounds[i + 1]
        qids = [q for q in common if lo <= d_by_qid[q] < hi]
        if not qids:
            continue

        d_vals = [d_by_qid[q] for q in qids]
        dbw_vals = [dbw_by_qid[q] for q in qids]

        delta, p = paired_bootstrap(dbw_vals, d_vals)

        quartiles.append({
            "label": labels[i],
            "d_ra_range": f"{lo:.2f}-{hi:.2f}",
            "n": len(qids),
            "d_ra": round(float(np.mean(d_vals)), 4),
            "dbw_ra": round(float(np.mean(dbw_vals)), 4),
            "delta": round(delta, 4),
            "p_value": round(p, 4),
            "significant": p < 0.05,
        })

    analysis2 = {
        "_metadata": METADATA,
        "quartiles": quartiles,
        "interpretation": "Positive delta means DBW outperforms D. Largest advantage expected on hardest queries.",
    }

    save_json(analysis2, "paper/analyses/analysis2_query_difficulty.json")
    return analysis2


# ===================================================================
# ANALYSIS 3: Retrieval Source Overlap
# ===================================================================
def generate_analysis3(master):
    print("\n--- ANALYSIS 3: Retrieval Source Overlap ---")

    dbw_h = [e for e in master if e["config"] == "DBW" and e["router_type"] == "heuristic"]

    sources_per_query = {"1_source": 0, "2_sources": 0, "3_sources": 0}
    total_sources = []

    for e in dbw_h:
        present = set()
        for src, cnt in e.get("source_distribution", {}).items():
            if cnt > 0:
                present.add(src)
        n = len(present)
        total_sources.append(n)
        key = f"{n}_source" if n == 1 else f"{n}_sources"
        if key in sources_per_query:
            sources_per_query[key] += 1

    n_total = len(dbw_h)

    analysis3 = {
        "_metadata": METADATA,
        "sources_per_query": {
            k: {"count": v, "pct": round(v / n_total * 100, 1)} for k, v in sources_per_query.items()
        },
        "avg_sources_per_query": round(float(np.mean(total_sources)), 2),
        "n": n_total,
        "interpretation": f"{sources_per_query['2_sources'] + sources_per_query['3_sources']} of {n_total} queries ({round((sources_per_query['2_sources'] + sources_per_query['3_sources']) / n_total * 100, 1)}%) use chunks from 2+ sources, confirming multi-source retrieval is active",
    }

    save_json(analysis3, "paper/analyses/analysis3_source_overlap.json")
    return analysis3


# ===================================================================
# ANALYSIS 4: Router Confidence vs Performance
# ===================================================================
def generate_analysis4(master):
    print("\n--- ANALYSIS 4: Router Confidence vs Performance ---")

    dbw_h = [e for e in master if e["config"] == "DBW" and e["router_type"] == "heuristic"]

    # Compute confidence for each query
    for e in dbw_h:
        boosts = e.get("source_boosts", {})
        vals = sorted(boosts.values(), reverse=True)
        if len(vals) >= 2:
            e["_confidence"] = vals[0] - vals[1]
        else:
            e["_confidence"] = 0

    # Sort by confidence and split into terciles
    dbw_h.sort(key=lambda x: x["_confidence"])
    n = len(dbw_h)
    t1 = n // 3
    t2 = 2 * n // 3

    groups = [
        ("Low confidence", dbw_h[:t1]),
        ("Medium confidence", dbw_h[t1:t2]),
        ("High confidence", dbw_h[t2:]),
    ]

    terciles = []
    for label, entries in groups:
        confs = [e["_confidence"] for e in entries]
        terciles.append({
            "label": label,
            "n": len(entries),
            "avg_ra": round(float(np.mean([e["ra"] for e in entries])), 4),
            "avg_csas": round(float(np.mean([e["csas"] for e in entries])), 4),
            "confidence_range": f"{min(confs):.4f}-{max(confs):.4f}",
            "avg_confidence": round(float(np.mean(confs)), 4),
        })

    analysis4 = {
        "_metadata": METADATA,
        "terciles": terciles,
    }

    save_json(analysis4, "paper/analyses/analysis4_router_confidence.json")
    return analysis4


# ===================================================================
# ANALYSIS 5: Qualitative Case Studies
# ===================================================================
def generate_analysis5(master):
    print("\n--- ANALYSIS 5: Qualitative Case Studies ---")

    heuristic = [e for e in master if e["router_type"] == "heuristic"]
    by_config = defaultdict(list)
    for e in heuristic:
        by_config[e["config"]].append(e)

    dbw_by_qid = {e["query_id"]: e for e in by_config["DBW"]}
    d_by_qid = {e["query_id"]: e for e in by_config["D"]}
    bm25_by_qid = {e["query_id"]: e for e in by_config["BM25"]}

    cases = []

    # --- Case 1: Multi-source synthesis win ---
    # DBW RA >= 0.8 AND D RA <= 0.6 (gap >= 0.2)
    case1 = None
    for qid, dbw in dbw_by_qid.items():
        d = d_by_qid.get(qid)
        if d and dbw["ra"] >= 0.8 and d["ra"] <= 0.6:
            case1 = (qid, dbw, d)
            break
    # Relax if not found
    if not case1:
        best = None
        best_gap = 0
        for qid, dbw in dbw_by_qid.items():
            d = d_by_qid.get(qid)
            if d:
                gap = dbw["ra"] - d["ra"]
                if gap > best_gap:
                    best_gap = gap
                    best = (qid, dbw, d)
        case1 = best

    if case1:
        qid, dbw, d = case1
        bm25 = bm25_by_qid.get(qid, {})
        cases.append({
            "case_id": 1,
            "label": "Multi-source synthesis win",
            "query_id": qid,
            "category": dbw["category"],
            "dbw_ra": dbw["ra"],
            "d_ra": d["ra"],
            "bm25_ra": bm25.get("ra", None),
            "dbw_csas": dbw["csas"],
            "dbw_answer_excerpt": dbw.get("answer_excerpt", ""),
            "d_answer_excerpt": d.get("answer_excerpt", ""),
            "dbw_citations": dbw.get("citation_counts", {}),
            "dbw_source_distribution": dbw.get("source_distribution", {}),
            "source_boosts": dbw.get("source_boosts", {}),
            "narrative": f"DBW RA={dbw['ra']:.3f} vs D RA={d['ra']:.3f}: multi-source retrieval provides complementary information that single-source misses.",
        })

    # --- Case 2: Attribution quality win ---
    # DBW RA ≈ BM25 RA (within 0.05) but DBW CSAS >> BM25 CSAS (gap >= 0.3)
    case2 = None
    for qid, dbw in dbw_by_qid.items():
        bm25 = bm25_by_qid.get(qid)
        if not bm25:
            continue
        ra_diff = abs(dbw["ra"] - bm25["ra"])
        csas_diff = dbw["csas"] - bm25.get("csas", 0)
        if ra_diff <= 0.05 and csas_diff >= 0.3:
            case2 = (qid, dbw, bm25)
            break

    if not case2:
        best = None
        best_csas_gap = 0
        for qid, dbw in dbw_by_qid.items():
            bm25 = bm25_by_qid.get(qid)
            if not bm25:
                continue
            ra_diff = abs(dbw["ra"] - bm25["ra"])
            csas_diff = dbw["csas"] - bm25.get("csas", 0)
            if ra_diff <= 0.1 and csas_diff > best_csas_gap:
                best_csas_gap = csas_diff
                best = (qid, dbw, bm25)
        case2 = best

    if case2:
        qid, dbw, bm25 = case2
        d = d_by_qid.get(qid, {})
        cases.append({
            "case_id": 2,
            "label": "Attribution quality win",
            "query_id": qid,
            "category": dbw["category"],
            "dbw_ra": dbw["ra"],
            "d_ra": d.get("ra", None),
            "bm25_ra": bm25["ra"],
            "dbw_csas": dbw["csas"],
            "bm25_csas": bm25.get("csas", 0),
            "dbw_answer_excerpt": dbw.get("answer_excerpt", ""),
            "bm25_answer_excerpt": bm25.get("answer_excerpt", ""),
            "dbw_citations": dbw.get("citation_counts", {}),
            "source_boosts": dbw.get("source_boosts", {}),
            "narrative": f"Similar RA ({dbw['ra']:.3f} vs {bm25['ra']:.3f}) but DBW correctly attributes to expected sources (CSAS={dbw['csas']:.3f} vs {bm25.get('csas',0):.3f}).",
        })

    # --- Case 3: Perfect routing ---
    # Debugging query where source_boosts show bug > 0.6 and reranked chunks are 70%+ bugs
    case3 = None
    for qid, dbw in dbw_by_qid.items():
        if dbw["category"] != "debugging":
            continue
        boosts = dbw.get("source_boosts", {})
        dist = dbw.get("source_pct", {})
        if boosts.get("bug", 0) > 0.5 and dist.get("bug", 0) > 0.6:
            case3 = (qid, dbw)
            break

    if not case3:
        # Relax: any debugging query with bug > 0.4
        for qid, dbw in dbw_by_qid.items():
            if dbw["category"] != "debugging":
                continue
            boosts = dbw.get("source_boosts", {})
            dist = dbw.get("source_pct", {})
            if boosts.get("bug", 0) > 0.4 and dist.get("bug", 0) > 0.5:
                case3 = (qid, dbw)
                break

    if case3:
        qid, dbw = case3
        d = d_by_qid.get(qid, {})
        bm25 = bm25_by_qid.get(qid, {})
        cases.append({
            "case_id": 3,
            "label": "Perfect routing",
            "query_id": qid,
            "category": dbw["category"],
            "dbw_ra": dbw["ra"],
            "d_ra": d.get("ra", None),
            "bm25_ra": bm25.get("ra", None),
            "dbw_csas": dbw["csas"],
            "dbw_answer_excerpt": dbw.get("answer_excerpt", ""),
            "dbw_citations": dbw.get("citation_counts", {}),
            "dbw_source_distribution": dbw.get("source_distribution", {}),
            "source_boosts": dbw.get("source_boosts", {}),
            "narrative": f"Router correctly boosted bugs ({dbw.get('source_boosts',{}).get('bug',0):.2f}) for debugging query; {dbw.get('source_pct',{}).get('bug',0)*100:.0f}% of retrieved chunks are from bug reports.",
        })

    # --- Case 4: Status/roadmap data ceiling ---
    # ALL configs score RA < 0.4
    case4 = None
    for qid, dbw in dbw_by_qid.items():
        if dbw["category"] != "status_roadmap":
            continue
        d = d_by_qid.get(qid)
        bm25 = bm25_by_qid.get(qid)
        if d and bm25 and dbw["ra"] < 0.4 and d["ra"] < 0.4 and bm25["ra"] < 0.4:
            case4 = (qid, dbw, d, bm25)
            break

    if not case4:
        # Relax threshold to 0.5
        for qid, dbw in dbw_by_qid.items():
            if dbw["category"] != "status_roadmap":
                continue
            d = d_by_qid.get(qid)
            bm25 = bm25_by_qid.get(qid)
            if d and bm25 and dbw["ra"] < 0.5 and d["ra"] < 0.5 and bm25["ra"] < 0.5:
                case4 = (qid, dbw, d, bm25)
                break

    if case4:
        qid, dbw, d, bm25 = case4
        cases.append({
            "case_id": 4,
            "label": "Status/roadmap data ceiling",
            "query_id": qid,
            "category": dbw["category"],
            "dbw_ra": dbw["ra"],
            "d_ra": d["ra"],
            "bm25_ra": bm25["ra"],
            "dbw_csas": dbw["csas"],
            "dbw_answer_excerpt": dbw.get("answer_excerpt", ""),
            "dbw_citations": dbw.get("citation_counts", {}),
            "source_boosts": dbw.get("source_boosts", {}),
            "narrative": f"All configs fail (DBW={dbw['ra']:.3f}, D={d['ra']:.3f}, BM25={bm25['ra']:.3f}): query requires up-to-date roadmap data that our static dataset lacks.",
        })

    # --- Case 5: Diversity fix demonstration ---
    # how_to or debugging query where reranked chunks are 80%+ from one source
    case5 = None
    for qid, dbw in dbw_by_qid.items():
        if dbw["category"] not in ("how_to", "debugging", "config"):
            continue
        dist = dbw.get("source_pct", {})
        max_pct = max(dist.values()) if dist else 0
        if max_pct >= 0.8 and dbw["ra"] >= 0.7:
            case5 = (qid, dbw)
            break

    if not case5:
        for qid, dbw in dbw_by_qid.items():
            dist = dbw.get("source_pct", {})
            max_pct = max(dist.values()) if dist else 0
            if max_pct >= 0.7 and dbw["ra"] >= 0.6:
                case5 = (qid, dbw)
                break

    if case5:
        qid, dbw = case5
        d = d_by_qid.get(qid, {})
        bm25 = bm25_by_qid.get(qid, {})
        dominant = max(dbw.get("source_pct", {}), key=dbw.get("source_pct", {}).get)
        cases.append({
            "case_id": 5,
            "label": "Diversity fix demonstration",
            "query_id": qid,
            "category": dbw["category"],
            "dbw_ra": dbw["ra"],
            "d_ra": d.get("ra", None),
            "bm25_ra": bm25.get("ra", None),
            "dbw_csas": dbw["csas"],
            "dbw_answer_excerpt": dbw.get("answer_excerpt", ""),
            "dbw_citations": dbw.get("citation_counts", {}),
            "dbw_source_distribution": dbw.get("source_distribution", {}),
            "source_boosts": dbw.get("source_boosts", {}),
            "narrative": f"Fixed reranker correctly lets {dominant} dominate ({dbw.get('source_pct',{}).get(dominant,0)*100:.0f}%) when the router identifies a single-source query, instead of forcing artificial diversity.",
        })

    analysis5 = {
        "_metadata": METADATA,
        "cases": cases,
        "n_cases_found": len(cases),
    }

    save_json(analysis5, "paper/analyses/analysis5_case_studies.json")
    return analysis5


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("=" * 60)
    print("STEP 4: Generating Analyses 1-5")
    print("=" * 60)

    final = load_json("paper/FINAL_NUMBERS.json")
    master = final["master_entries"]

    generate_analysis1(master)
    generate_analysis2(master)
    generate_analysis3(master)
    generate_analysis4(master)
    generate_analysis5(master)

    print("\n" + "=" * 60)
    print("All 5 analyses generated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
