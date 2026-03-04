"""Offline training for the adaptive router using ablation results + judge scores.

Trains LinUCB router by replaying query-utility pairs from evaluation data.
Uses ablation-based utility: how much does removing a source hurt RA?

Usage:
    python -m retrieval.router.train_offline --epochs 3
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

from indexing.providers.sentence_transformer import SentenceTransformerProvider
from retrieval.router.adaptive import AdaptiveRouter

SOURCES = ["doc", "bug", "work_item"]
# Map config letters to source lists
CONFIG_SOURCES = {
    "D": ["doc"], "B": ["bug"], "W": ["work_item"],
    "DB": ["doc", "bug"], "DW": ["doc", "work_item"], "BW": ["bug", "work_item"],
    "DBW": ["doc", "bug", "work_item"],
}
SAVE_DIR = "models/adaptive_router"


def _load_judge_scores() -> dict:
    """Load and average RA scores across all 3 judges, keyed by (query_id, config, router_type)."""
    judge_files = [
        "data/evaluation/judge_scores_gpt4o.json",
        "data/evaluation/judge_scores_claude.json",
        "data/evaluation/judge_scores_gemini.json",
    ]
    # Collect RA per key per judge
    ra_by_key: dict[tuple, list[float]] = {}
    for jf in judge_files:
        if not os.path.exists(jf):
            print(f"  Warning: {jf} not found, skipping")
            continue
        with open(jf, "r", encoding="utf-8") as f:
            scores = json.load(f)
        for s in scores:
            key = (s["query_id"], s["config"], s.get("router_type", "heuristic"))
            ra = s.get("ra")
            if ra is not None:
                ra_by_key.setdefault(key, []).append(ra)

    # Average across judges
    avg_ra: dict[tuple, float] = {}
    for key, vals in ra_by_key.items():
        avg_ra[key] = sum(vals) / len(vals)
    return avg_ra


def _compute_utilities(query_id: str, avg_ra: dict, router_type: str = "heuristic") -> dict[str, float]:
    """Compute per-source utility via ablation: utility_s = max(0, RA_DBW - RA_without_s) / RA_DBW."""
    ra_full = avg_ra.get((query_id, "DBW", router_type))
    if ra_full is None or ra_full <= 0:
        return {}

    # "Without source s" = the config that has the other two sources
    without_map = {
        "doc": "BW",       # DBW minus doc = BW
        "bug": "DW",       # DBW minus bug = DW
        "work_item": "DB", # DBW minus work_item = DB
    }

    utilities = {}
    for src, without_config in without_map.items():
        ra_without = avg_ra.get((query_id, without_config, router_type))
        if ra_without is not None:
            drop = ra_full - ra_without
            utilities[src] = max(0.0, drop / ra_full)
        else:
            # If we don't have the ablation config, use citation-based fallback
            utilities[src] = 0.0

    return utilities


def train(epochs: int = 3, seed: int = 42):
    """Train adaptive router offline from ablation data."""
    print(f"Loading judge scores...")
    avg_ra = _load_judge_scores()
    print(f"  Loaded {len(avg_ra)} averaged RA scores")

    # Get unique query IDs that have DBW results
    query_ids = sorted(set(qid for qid, cfg, rt in avg_ra if cfg == "DBW"))
    print(f"  {len(query_ids)} queries with DBW results")

    # Load queries for embedding
    with open("data/evaluation/ablation_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    # Build query_id -> query text map (from DBW config)
    query_map = {}
    for r in results:
        if r.get("config") == "DBW" and r["query_id"] in set(query_ids):
            query_map[r["query_id"]] = r["query"]

    print(f"  {len(query_map)} queries with text")

    # Embed all queries
    print("Embedding queries...")
    embedder = SentenceTransformerProvider()
    query_texts = [query_map[qid] for qid in query_ids if qid in query_map]
    query_ids_with_text = [qid for qid in query_ids if qid in query_map]
    embeddings = embedder.embed(query_texts)
    print(f"  Embedded {len(embeddings)} queries (dim={embeddings.shape[1]})")

    # Build training pairs
    training_data = []
    for i, qid in enumerate(query_ids_with_text):
        utilities = _compute_utilities(qid, avg_ra)
        if utilities:
            training_data.append((embeddings[i], utilities, qid))

    print(f"  {len(training_data)} training pairs with utility signals")

    # Show utility distribution
    for src in SOURCES:
        vals = [u[src] for _, u, _ in training_data if src in u]
        if vals:
            print(f"  {src}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}, "
                  f"nonzero={sum(1 for v in vals if v > 0)}/{len(vals)}")

    # Train router
    router = AdaptiveRouter(
        feature_dim=embeddings.shape[1],
        cold_start_threshold=0,  # No cold start for offline training
        alpha_initial=0.5,
        alpha_min=0.1,
        alpha_decay=0.995,
    )

    rng = random.Random(seed)

    for epoch in range(epochs):
        # Shuffle training data each epoch
        order = list(range(len(training_data)))
        rng.shuffle(order)

        epoch_utilities = {src: [] for src in SOURCES}
        for idx in order:
            emb, utilities, qid = training_data[idx]
            router.update(emb, utilities)
            for src in SOURCES:
                if src in utilities:
                    epoch_utilities[src].append(utilities[src])

        # Log epoch stats
        print(f"\n  Epoch {epoch + 1}/{epochs}:")
        print(f"    Alpha: {router._alpha:.4f}")
        print(f"    Query count: {router._query_count}")
        for src in SOURCES:
            vals = epoch_utilities[src]
            print(f"    {src}: mean_utility={np.mean(vals):.3f}")

        # Test predictions on a few queries
        test_indices = rng.sample(range(len(training_data)), min(3, len(training_data)))
        for ti in test_indices:
            emb, _, qid = training_data[ti]
            boosts = router.predict("", emb)
            print(f"    Sample {qid}: {', '.join(f'{s}={b:.3f}' for s, b in boosts.items())}")

    # Save trained router
    os.makedirs(SAVE_DIR, exist_ok=True)
    router.save_state(SAVE_DIR)
    print(f"\nRouter saved to {SAVE_DIR}/")
    print(f"Final stats: {router.stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train adaptive router offline")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    train(epochs=args.epochs, seed=args.seed)
