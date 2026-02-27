"""LLM judge batch runner — score all ablation results.

Runs LLM judge (GPT-4o and/or Claude) across ablation results to produce
RA scores for all query-config pairs. Supports checkpoint/resume.

Usage:
    python -m evaluation.judge_runner
    python -m evaluation.judge_runner --judge-model gpt-4o --max-results 50
    python -m evaluation.judge_runner --judge-model claude  # Cross-validation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

logger = logging.getLogger(__name__)

SCORES_PATH = "data/evaluation/judge_scores.json"
CHECKPOINT_PATH = "data/evaluation/judge_checkpoint.json"


def _load_checkpoint() -> set[tuple[str, str, str]]:
    """Load set of already-scored (query_id, config, judge_model) tuples."""
    if not os.path.exists(CHECKPOINT_PATH):
        return set()
    with open(CHECKPOINT_PATH, "r") as f:
        data = json.load(f)
    return set(tuple(k) for k in data.get("completed", []))


def _save_checkpoint(completed: set[tuple[str, str, str]]) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"completed": [list(k) for k in completed]}, f)


def _save_scores(scores: list[dict]) -> None:
    os.makedirs(os.path.dirname(SCORES_PATH), exist_ok=True)
    with open(SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)


def run_judge(
    results_path: str = "data/evaluation/ablation_results.json",
    judge_model: str | None = None,
    max_results: int | None = None,
) -> list[dict]:
    """Run LLM judge on ablation results.

    Args:
        results_path: Path to ablation results JSON.
        judge_model: "gpt-4o" or "claude" for cross-validation.
        max_results: Limit number of results to score.

    Returns:
        List of judge score dicts.
    """
    import config as cfg
    from evaluation.llm_judge import LLMJudge

    if judge_model is None:
        judge_model = cfg.JUDGE_MODEL

    if not os.path.exists(results_path):
        print(f"No results at {results_path}. Run ablation_runner first.")
        return []

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Filter to results that have answers
    scorable = [
        r for r in results
        if "error" not in r and r.get("answer") and r["answer"] != ""
    ]

    if max_results:
        scorable = scorable[:max_results]

    # Initialize judge
    if judge_model == "claude":
        from generation.providers.anthropic_gen import AnthropicProvider
        llm = AnthropicProvider()
        judge = LLMJudge(llm_provider=llm, judge_model="claude-sonnet-4-20250514")
    else:
        judge = LLMJudge(judge_model=judge_model)

    # Load existing scores and checkpoint
    all_scores: list[dict] = []
    if os.path.exists(SCORES_PATH):
        with open(SCORES_PATH, "r", encoding="utf-8") as f:
            all_scores = json.load(f)

    completed = _load_checkpoint()

    total = len(scorable)
    done = 0

    for r in scorable:
        query_id = r.get("query_id", "")
        config = r.get("config", "")
        key = (query_id, config, judge_model)

        if key in completed:
            done += 1
            continue

        print(f"  [{done + 1}/{total}] Scoring {config} | {query_id}...")

        try:
            # Build minimal chunk list for judge context
            chunks_for_judge = []
            for c_data in r.get("reranked_chunks", [])[:10]:
                from processing.schemas import Chunk
                chunk = Chunk(
                    chunk_id=c_data["chunk_id"],
                    source_type=c_data["source_type"],
                    source_id=c_data.get("source_id", ""),
                    source_url=c_data.get("source_url", ""),
                    text=c_data.get("text", ""),
                    text_with_context=c_data.get("text", ""),
                    feature_area=c_data.get("feature_area", ""),
                    created_at="", updated_at="",
                )
                chunks_for_judge.append((chunk, c_data.get("score", 0.0)))

            judge_result = judge.evaluate(
                query_id=query_id,
                query_text=r.get("query", ""),
                expected_sources=r.get("expected_sources", []),
                answer=r.get("answer", ""),
                chunks=chunks_for_judge,
                config=config,
            )

            score_dict = judge_result.to_dict()
            score_dict["router_type"] = r.get("router_type", "heuristic")
            all_scores.append(score_dict)

            completed.add(key)
            _save_checkpoint(completed)

            # Save periodically
            if (done + 1) % 10 == 0:
                _save_scores(all_scores)

            # Rate limit
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"Judge error on {query_id}/{config}: {e}")
            print(f"    ERROR: {e}")

        done += 1

    _save_scores(all_scores)
    print(f"\nDone: scored {done} results with {judge_model}, saved to {SCORES_PATH}")
    return all_scores


def main():
    parser = argparse.ArgumentParser(description="Run LLM judge on ablation results")
    parser.add_argument("--judge-model", type=str, default=None, help="Judge model (gpt-4o-dated or claude)")
    parser.add_argument("--max-results", type=int, default=None, help="Max results to score")
    parser.add_argument("--results-path", type=str, default="data/evaluation/ablation_results.json")
    args = parser.parse_args()

    run_judge(
        results_path=args.results_path,
        judge_model=args.judge_model,
        max_results=args.max_results,
    )


if __name__ == "__main__":
    main()
