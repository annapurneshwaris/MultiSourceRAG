"""Run ablation experiments across all 13 configurations.

Supports checkpoint/resume for long-running evaluations.
Output: data/evaluation/ablation_results.json

Usage:
    python -m evaluation.ablation_runner
    python -m evaluation.ablation_runner --configs D,DBW --queries 10
"""

from __future__ import annotations

import argparse
import json
import os
import time

from evaluation.query_bank import load_queries, EvalQuery

# All 13 experiment configurations
ALL_CONFIGS = [
    "D", "B", "W",           # Single source
    "DB", "DW", "BW",        # Two sources
    "DBW",                     # Three sources (main)
    "BM25",                    # BM25 baseline
]

ROUTER_VARIANTS = ["heuristic", "adaptive", "llm_zeroshot"]

RESULTS_PATH = "data/evaluation/ablation_results.json"
CHECKPOINT_PATH = "data/evaluation/ablation_checkpoint.json"


def _load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return {"completed": []}


def _save_checkpoint(checkpoint: dict) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f)


def _save_results(results: list[dict]) -> None:
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_ablation(
    configs: list[str] | None = None,
    max_queries: int | None = None,
    generate: bool = True,
    router_type: str = "heuristic",
) -> list[dict]:
    """Run ablation experiments.

    Args:
        configs: List of config strings, or None for all.
        max_queries: Limit number of queries (for testing).
        generate: Whether to generate LLM answers.
        router_type: Which router to use.

    Returns:
        List of result dicts.
    """
    from retrieval.pipeline import RetrievalPipeline

    configs = configs or ALL_CONFIGS
    queries = load_queries()
    if max_queries:
        queries = queries[:max_queries]

    checkpoint = _load_checkpoint()
    completed_keys = set(tuple(k) for k in checkpoint["completed"])

    # Initialize pipeline
    print(f"Initializing pipeline (router={router_type})...")
    pipeline = RetrievalPipeline(router_type=router_type)

    all_results: list[dict] = []

    # Load existing results if any
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            all_results = json.load(f)

    total = len(configs) * len(queries)
    done = 0

    for config in configs:
        for query in queries:
            key = (query.query_id, config, router_type)
            if key in completed_keys:
                done += 1
                continue

            print(f"  [{done + 1}/{total}] {config} | {query.query_id}: {query.query_text[:60]}...")

            try:
                t0 = time.time()
                result = pipeline.process_query(
                    query=query.query_text,
                    config=config,
                    generate=generate,
                )

                result["query_id"] = query.query_id
                result["category"] = query.category
                result["expected_sources"] = query.expected_sources
                result["expected_area"] = query.expected_area
                result["difficulty"] = query.difficulty
                result["router_type"] = router_type
                result["eval_time"] = round(time.time() - t0, 2)

                all_results.append(result)

                # Checkpoint
                checkpoint["completed"].append(list(key))
                _save_checkpoint(checkpoint)

                # Save periodically
                if (done + 1) % 10 == 0:
                    _save_results(all_results)

            except Exception as e:
                print(f"    ERROR: {e}")
                all_results.append({
                    "query_id": query.query_id,
                    "config": config,
                    "router_type": router_type,
                    "error": str(e),
                })

            done += 1

    _save_results(all_results)
    print(f"\nDone: {done} experiments, saved to {RESULTS_PATH}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run HeteroRAG ablation experiments")
    parser.add_argument("--configs", type=str, default=None, help="Comma-separated configs")
    parser.add_argument("--queries", type=int, default=None, help="Max queries to evaluate")
    parser.add_argument("--router", type=str, default="heuristic", help="Router type")
    parser.add_argument("--no-generate", action="store_true", help="Skip LLM generation")
    args = parser.parse_args()

    configs = args.configs.split(",") if args.configs else None

    run_ablation(
        configs=configs,
        max_queries=args.queries,
        generate=not args.no_generate,
        router_type=args.router,
    )


if __name__ == "__main__":
    main()
