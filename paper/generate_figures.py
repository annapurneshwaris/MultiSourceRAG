"""STEP 3: Generate Figures 1-4.

Reads: paper/FINAL_NUMBERS.json, model files
Outputs: paper/figures/fig{1..4}_*.{json,mermaid}
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
    n = len(a)
    boots = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        boots[i] = np.mean(a[idx])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ===================================================================
# FIGURE 1: Architecture Diagram (Mermaid)
# ===================================================================
def generate_fig1():
    print("\n--- FIGURE 1: Architecture Diagram ---")

    mermaid = """graph TD
    Q["🔍 User Query"] --> EMB["Embedding<br/>(all-MiniLM-L6-v2, 384d)"]
    EMB --> ROUTER["Adaptive Router<br/>(LinUCB Bandit)"]
    EMB --> BM25S["BM25 Scorer<br/>(Keyword Signal)"]

    ROUTER -->|"boost=0.7"| DR["📄 Doc Retriever<br/>Tree Navigation + Vector Search"]
    ROUTER -->|"boost=0.2"| BR["🐛 Bug Retriever<br/>Vector + Metadata Filter"]
    ROUTER -->|"boost=0.1"| WR["📋 Work Item Retriever<br/>Vector + Metadata Filter"]

    subgraph "Source-Specific Chunking"
        DC["Docs: Hierarchical<br/>(H2/H3 split, breadcrumbs)"]
        BC["Bugs: Composite<br/>(title+body+errors+comments)"]
        WC["Work Items: Type-dependent<br/>(plans→sections, requests→single)"]
    end

    DC -.-> DR
    BC -.-> BR
    WC -.-> WR

    DR --> RERANK["Diversity-Constrained<br/>MMR Reranker"]
    BR --> RERANK
    WR --> RERANK
    BM25S --> RERANK

    RERANK -->|"Score = 0.4×semantic + 0.25×router<br/>+ 0.15×BM25 + 0.1×freshness<br/>+ 0.1×authority - 0.15×redundancy"| TOP10["Top-10 Chunks"]

    TOP10 --> GEN["LLM Generator<br/>(GPT-4o)"]
    GEN --> ANS["Answer with Citations<br/>[DOC-x] [BUG-y] [PLAN-z]"]

    ANS --> UTIL["Utility Signal<br/>(citation + rank + position)"]
    UTIL -->|"LinUCB update:<br/>A += x·x', b += u·x"| ROUTER

    style Q fill:#e1f5fe
    style ANS fill:#e8f5e9
    style ROUTER fill:#fff3e0
    style RERANK fill:#f3e5f5
    style GEN fill:#fce4ec
"""

    path = "paper/figures/fig1_architecture.mermaid"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(mermaid)
    print(f"  Saved: {path}")


# ===================================================================
# FIGURE 2: Router Learning Curve
# ===================================================================
def generate_fig2():
    print("\n--- FIGURE 2: Router Learning Curve ---")

    # Load router training data
    router_state = load_json("data/models/adaptive_router/router_state.json")
    training_stats = load_json("data/models/adaptive_router/training_stats.json")

    # Load history
    history_path = "data/models/adaptive_router/history.jsonl"
    history = []
    if os.path.exists(history_path):
        with open(history_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    history.append(json.loads(line))

    print(f"  Router state keys: {list(router_state.keys())}")
    print(f"  Training stats keys: {list(training_stats.keys())}")
    print(f"  History entries: {len(history)}")

    # Extract learning data from history
    # Each history entry has: query, config, boosts, utilities
    epochs_data = []
    window_size = 50  # Rolling window

    if history:
        # Group into windows
        for i in range(0, len(history), window_size):
            window = history[i:i + window_size]
            if not window:
                continue

            # Average utilities per source
            utils = {"doc": [], "bug": [], "work_item": []}
            boosts = {"doc": [], "bug": [], "work_item": []}
            for entry in window:
                for src in utils:
                    if "utilities" in entry and src in entry["utilities"]:
                        utils[src].append(entry["utilities"][src])
                    if "boosts" in entry and src in entry["boosts"]:
                        boosts[src].append(entry["boosts"][src])

            epoch = {
                "window_start": i,
                "window_end": min(i + window_size, len(history)),
                "n": len(window),
                "mean_utility": {k: round(float(np.mean(v)), 4) if v else 0 for k, v in utils.items()},
                "mean_boost": {k: round(float(np.mean(v)), 4) if v else 0 for k, v in boosts.items()},
            }
            epochs_data.append(epoch)

    fig2 = {
        "_metadata": METADATA,
        "router_state_summary": {
            "total_updates": router_state.get("total_updates", 0),
            "alpha": router_state.get("alpha", 0),
            "cold_start_complete": router_state.get("cold_start_complete", False),
        },
        "training_stats": training_stats,
        "learning_windows": epochs_data,
        "total_history_entries": len(history),
        "window_size": window_size,
    }

    save_json(fig2, "paper/figures/fig2_learning_curve.json")
    return fig2


# ===================================================================
# FIGURE 3: Source Distribution Heatmap
# ===================================================================
def generate_fig3(master):
    print("\n--- FIGURE 3: Source Distribution Heatmap ---")

    dbw_h = [e for e in master if e["config"] == "DBW" and e["router_type"] == "heuristic"]
    categories = ["how_to", "config", "debugging", "error_diagnosis", "status_roadmap"]

    heatmap = {}
    for cat in categories:
        entries = [e for e in dbw_h if e["category"] == cat]
        src_counts = {"doc": 0, "bug": 0, "work_item": 0}
        for e in entries:
            for src, cnt in e.get("source_distribution", {}).items():
                src_counts[src] = src_counts.get(src, 0) + cnt
        total = sum(src_counts.values())
        heatmap[cat] = {k: round(v / total * 100, 1) if total > 0 else 0 for k, v in src_counts.items()}

    fig3 = {
        "_metadata": METADATA,
        "_note": "Post-fix DBW heuristic data with router-aware diversity constraint",
        "source_distribution_pct": heatmap,
        "n_per_category": {cat: len([e for e in dbw_h if e["category"] == cat]) for cat in categories},
    }

    save_json(fig3, "paper/figures/fig3_source_distribution.json")
    return fig3


# ===================================================================
# FIGURE 4: RA Comparison Bar Chart
# ===================================================================
def generate_fig4(master):
    print("\n--- FIGURE 4: RA Comparison Bar Chart ---")

    heuristic = [e for e in master if e["router_type"] == "heuristic"]
    by_config = defaultdict(list)
    for e in heuristic:
        by_config[e["config"]].append(e)

    config_order = ["DBW", "D", "B", "W", "DB", "DW", "BW", "BM25", "Naive"]

    configs = []
    ra_means = []
    ci_lowers = []
    ci_uppers = []
    csas_means = []

    for cfg in config_order:
        entries = by_config.get(cfg, [])
        if not entries:
            continue

        ra_vals = [e["ra"] for e in entries]
        csas_vals = [e["csas"] for e in entries]
        mean_ra = float(np.mean(ra_vals))

        # Bootstrap CI for mean RA
        rng = np.random.RandomState(SEED)
        boots = []
        for _ in range(10000):
            idx = rng.randint(0, len(ra_vals), size=len(ra_vals))
            boots.append(np.mean([ra_vals[i] for i in idx]))
        ci_lo = float(np.percentile(boots, 2.5))
        ci_hi = float(np.percentile(boots, 97.5))

        configs.append(cfg)
        ra_means.append(round(mean_ra, 4))
        ci_lowers.append(round(ci_lo, 4))
        ci_uppers.append(round(ci_hi, 4))
        csas_means.append(round(float(np.mean(csas_vals)), 4))

    fig4 = {
        "_metadata": METADATA,
        "configs": configs,
        "ra": ra_means,
        "ci_lower": ci_lowers,
        "ci_upper": ci_uppers,
        "csas": csas_means,
    }

    save_json(fig4, "paper/figures/fig4_ra_comparison.json")
    return fig4


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("=" * 60)
    print("STEP 3: Generating Figures 1-4")
    print("=" * 60)

    final = load_json("paper/FINAL_NUMBERS.json")
    master = final["master_entries"]

    generate_fig1()
    generate_fig2()
    generate_fig3(master)
    generate_fig4(master)

    print("\n" + "=" * 60)
    print("All 4 figures generated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
