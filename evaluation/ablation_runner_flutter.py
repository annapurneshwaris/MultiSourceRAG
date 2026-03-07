"""Run ablation experiments on Flutter data.

Usage:
    python -m evaluation.ablation_runner_flutter
    python -m evaluation.ablation_runner_flutter --configs D,DBW,BM25 --queries 25
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

from evaluation.query_bank_flutter import load_queries, EvalQuery

logger = logging.getLogger(__name__)

ALL_CONFIGS = ["D", "B", "W", "DB", "DW", "BW", "DBW", "BM25", "Naive"]

RESULTS_PATH = "data/flutter/evaluation/ablation_results.json"
CHECKPOINT_PATH = "data/flutter/evaluation/ablation_checkpoint.json"

FLUTTER_INDEX_DIR = "data/flutter/indices/combined"
FLUTTER_BM25_DIR = "data/flutter/indices/bm25"


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
    """Run ablation experiments on Flutter data."""
    # Monkey-patch metadata_hints to use Flutter's feature area map
    import retrieval.metadata_hints_flutter as hints_flutter
    import retrieval.metadata_hints as hints_orig
    hints_orig.extract_hints = hints_flutter.extract_hints

    from retrieval.pipeline import RetrievalPipeline

    configs = configs or ALL_CONFIGS
    queries = load_queries()
    if max_queries:
        queries = queries[:max_queries]

    checkpoint = _load_checkpoint()
    completed_keys = set(tuple(k) for k in checkpoint["completed"])

    all_results: list[dict] = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            all_results = json.load(f)

    # Initialize pipeline with Flutter indices
    print(f"Initializing pipeline (router={router_type}, index={FLUTTER_INDEX_DIR})...")

    # Load Flutter BM25 index
    from indexing.bm25_index import BM25Index
    bm25 = None
    if os.path.exists(FLUTTER_BM25_DIR):
        bm25 = BM25Index()
        bm25.load(FLUTTER_BM25_DIR)

    # Use Gemini 2.5 Flash for generation (cheap + fast)
    from generation.providers.gemini_gen import GeminiProvider
    from generation.generator import Generator
    gemini_generator = Generator(llm_provider=GeminiProvider(model="gemini-2.5-flash"))

    pipeline = RetrievalPipeline(
        router_type=router_type,
        index_dir=FLUTTER_INDEX_DIR,
        bm25_index=bm25,
        generator=gemini_generator,
    )

    total = len(configs) * len(queries)
    done = 0

    for cfg in configs:
        for query in queries:
            key = (query.query_id, cfg, router_type)
            if key in completed_keys:
                done += 1
                continue

            print(f"  [{done + 1}/{total}] {cfg} | {query.query_id}: {query.query_text[:60]}...")

            try:
                t0 = time.time()
                result = pipeline.process_query(
                    query=query.query_text,
                    config=cfg,
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

                checkpoint["completed"].append(list(key))
                _save_checkpoint(checkpoint)

                if (done + 1) % 10 == 0:
                    _save_results(all_results)

                # Print source distribution for this result
                sources = [c["source_type"] for c in result.get("reranked_chunks", [])]
                source_dist = {s: sources.count(s) for s in set(sources)}
                print(f"    Retrieved: {source_dist} | Timing: {result.get('timing', {}).get('total_ms', 0)}ms")

            except Exception as e:
                logger.error(f"Error on {cfg} {query.query_id}: {e}")
                print(f"    ERROR: {e}")
                all_results.append({
                    "query_id": query.query_id,
                    "config": cfg,
                    "router_type": router_type,
                    "error": str(e),
                })
                checkpoint["completed"].append(list(key))
                _save_checkpoint(checkpoint)

            done += 1

    _save_results(all_results)
    print(f"\nDone: {done} experiments, saved to {RESULTS_PATH}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run Flutter ablation experiments")
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
