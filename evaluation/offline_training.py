"""Offline router training using ablation-derived utility signals.

Trains the adaptive router by replaying ablation results: for each query,
compute offline utility (RA drop when source is removed) and update LinUCB.

Usage:
    python -m evaluation.offline_training
    python -m evaluation.offline_training --epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def compute_offline_utilities(
    judge_path: str = "data/evaluation/judge_scores.json",
) -> dict[str, dict[str, float]]:
    """Compute per-query offline utility from ablation judge scores.

    For each query, utility(source) = max(0, RA_DBW - RA_without_source) / RA_DBW

    Returns:
        Dict mapping query_id -> {source_type: utility}.
    """
    if not os.path.exists(judge_path):
        return {}

    with open(judge_path, "r", encoding="utf-8") as f:
        scores = json.load(f)

    # Index by (query_id, config)
    by_key: dict[tuple[str, str], float] = {}
    for s in scores:
        by_key[(s["query_id"], s["config"])] = s.get("ra", 0.0)

    # For each query, compute utility per source
    query_ids = set(s["query_id"] for s in scores)
    utilities: dict[str, dict[str, float]] = {}

    # Source removal configs: removing D means config BW, etc.
    removal_map = {
        "doc": "BW",
        "bug": "DW",
        "work_item": "DB",
    }

    for qid in query_ids:
        ra_full = by_key.get((qid, "DBW"), 0.0)
        if ra_full <= 0:
            continue

        query_util = {}
        for source, ablated_config in removal_map.items():
            ra_without = by_key.get((qid, ablated_config), 0.0)
            drop = max(0.0, ra_full - ra_without)
            query_util[source] = drop / ra_full

        utilities[qid] = query_util

    return utilities


def train_offline(
    results_path: str = "data/evaluation/ablation_results.json",
    judge_path: str = "data/evaluation/judge_scores.json",
    epochs: int = 1,
    output_dir: str = "data/models/adaptive_router",
) -> dict:
    """Train adaptive router offline from ablation utility signals.

    Args:
        results_path: Path to ablation results (for query embeddings).
        judge_path: Path to judge scores (for RA values).
        epochs: Number of passes over the data.
        output_dir: Where to save trained router state.

    Returns:
        Training stats dict.
    """
    from retrieval.router.adaptive import AdaptiveRouter
    import config as cfg

    # Load utilities
    utilities = compute_offline_utilities(judge_path)
    if not utilities:
        print("No offline utilities available. Run judge_runner first.")
        return {"error": "no_judge_scores"}

    # Load results to get query embeddings
    if not os.path.exists(results_path):
        print("No ablation results. Run ablation_runner first.")
        return {"error": "no_results"}

    # We need embeddings for each query — re-embed them
    from indexing.providers.sentence_transformer import SentenceTransformerProvider
    embedder = SentenceTransformerProvider(
        model_name=cfg.EMBEDDING_MODEL_NAME,
        batch_size=cfg.EMBEDDING_BATCH_SIZE,
    )

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Get unique queries with their text
    query_texts: dict[str, str] = {}
    for r in results:
        if "error" not in r and r.get("query_id") in utilities:
            query_texts[r["query_id"]] = r.get("query", "")

    if not query_texts:
        print("No matching queries between results and utilities.")
        return {"error": "no_matching_queries"}

    # Embed all queries
    print(f"Embedding {len(query_texts)} queries...")
    qids = list(query_texts.keys())
    texts = [query_texts[qid] for qid in qids]
    embeddings = embedder.embed(texts)

    # Create and train router
    router = AdaptiveRouter(
        feature_dim=cfg.EMBEDDING_DIM,
        alpha_initial=cfg.ALPHA_INITIAL,
        alpha_min=cfg.ALPHA_MIN,
        alpha_decay=cfg.ALPHA_DECAY,
        cold_start_threshold=0,  # No cold start for offline training
    )

    total_updates = 0
    for epoch in range(epochs):
        for i, qid in enumerate(qids):
            if qid in utilities:
                emb = embeddings[i].astype(np.float64)
                router.update(emb, utilities[qid])
                total_updates += 1

        print(f"  Epoch {epoch + 1}/{epochs}: {total_updates} updates, alpha={router._alpha:.4f}")

    # Save trained router
    os.makedirs(output_dir, exist_ok=True)
    router.save_state(output_dir)

    stats = {
        "n_queries": len(qids),
        "epochs": epochs,
        "total_updates": total_updates,
        "final_alpha": float(router._alpha),
        "router_stats": router.stats,
    }

    stats_path = os.path.join(output_dir, "training_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Router trained and saved to {output_dir}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Offline router training from ablation data")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--results-path", type=str, default="data/evaluation/ablation_results.json")
    parser.add_argument("--judge-path", type=str, default="data/evaluation/judge_scores.json")
    args = parser.parse_args()

    train_offline(
        results_path=args.results_path,
        judge_path=args.judge_path,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
