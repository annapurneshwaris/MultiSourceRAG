"""LLM judge batch runner — score all ablation results.

Runs LLM judge (GPT-4o, Claude, Gemini) across ablation results to produce
RA scores for all query-config pairs. Supports checkpoint/resume.
Each judge writes to a separate file for inter-judge comparison.

Usage:
    python -m evaluation.judge_runner                          # GPT-4o (default)
    python -m evaluation.judge_runner --judge-model claude     # Claude cross-validation
    python -m evaluation.judge_runner --judge-model gemini     # Gemini cross-validation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

logger = logging.getLogger(__name__)

# Per-judge output files for inter-judge comparison
JUDGE_OUTPUT_MAP = {
    "gpt-4o": "data/evaluation/judge_scores_gpt4o.json",
    "claude": "data/evaluation/judge_scores_claude.json",
    "gemini": "data/evaluation/judge_scores_gemini.json",
}
JUDGE_CHECKPOINT_MAP = {
    "gpt-4o": "data/evaluation/judge_checkpoint_gpt4o.json",
    "claude": "data/evaluation/judge_checkpoint_claude.json",
    "gemini": "data/evaluation/judge_checkpoint_gemini.json",
}
# Legacy combined path (kept for backward compat)
SCORES_PATH = "data/evaluation/judge_scores.json"
CHECKPOINT_PATH = "data/evaluation/judge_checkpoint.json"


def _resolve_paths(judge_key: str) -> tuple[str, str]:
    """Return (scores_path, checkpoint_path) for a given judge key."""
    scores = JUDGE_OUTPUT_MAP.get(judge_key, SCORES_PATH)
    checkpoint = JUDGE_CHECKPOINT_MAP.get(judge_key, CHECKPOINT_PATH)
    return scores, checkpoint


def _load_checkpoint(checkpoint_path: str) -> set[tuple[str, str, str]]:
    """Load set of already-scored (query_id, config, judge_model) tuples."""
    if not os.path.exists(checkpoint_path):
        return set()
    with open(checkpoint_path, "r") as f:
        data = json.load(f)
    return {tuple(k) for k in data.get("completed", [])}


def _save_checkpoint(completed: set[tuple[str, str, str]], checkpoint_path: str) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump({"completed": [list(k) for k in completed]}, f)


def _save_scores(scores: list[dict], scores_path: str) -> None:
    os.makedirs(os.path.dirname(scores_path), exist_ok=True)
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)


def _init_judge(judge_model: str):
    """Create LLMJudge instance for the given model key."""
    from evaluation.llm_judge import LLMJudge

    if judge_model == "claude":
        from generation.providers.anthropic_gen import AnthropicProvider
        return LLMJudge(llm_provider=AnthropicProvider(), judge_model="claude-sonnet-4-20250514")
    if judge_model == "gemini":
        from generation.providers.gemini_gen import GeminiProvider
        return LLMJudge(llm_provider=GeminiProvider(), judge_model="gemini-2.5-flash")
    # Default: OpenAI
    return LLMJudge(judge_model=judge_model)


def _build_chunks_for_judge(result: dict) -> list[tuple]:
    """Extract chunk objects from a result dict for judge evaluation."""
    from processing.schemas import Chunk
    chunks = []
    for c_data in result.get("reranked_chunks", [])[:10]:
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
        chunks.append((chunk, c_data.get("score", 0.0)))
    return chunks


def run_judge(
    results_path: str = "data/evaluation/ablation_results.json",
    judge_model: str | None = None,
    max_results: int | None = None,
) -> list[dict]:
    """Run LLM judge on ablation results.

    Args:
        results_path: Path to ablation results JSON.
        judge_model: "gpt-4o", "claude", or "gemini".
        max_results: Limit number of results to score.

    Returns:
        List of judge score dicts.
    """
    import config as cfg

    if judge_model is None:
        judge_model = cfg.JUDGE_MODEL

    # Determine judge key for file paths
    judge_key = "gpt-4o" if "gpt" in judge_model else judge_model
    scores_path, checkpoint_path = _resolve_paths(judge_key)

    if not os.path.exists(results_path):
        print(f"No results at {results_path}. Run ablation_runner first.")
        return []

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    scorable = [
        r for r in results
        if "error" not in r and r.get("answer") and r["answer"] != ""
    ]
    if max_results:
        scorable = scorable[:max_results]

    judge = _init_judge(judge_model)

    # Load existing scores and checkpoint
    all_scores: list[dict] = []
    if os.path.exists(scores_path):
        with open(scores_path, "r", encoding="utf-8") as f:
            all_scores = json.load(f)

    completed = _load_checkpoint(checkpoint_path)
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
            judge_result = judge.evaluate(
                query_id=query_id,
                query_text=r.get("query", ""),
                expected_sources=r.get("expected_sources", []),
                answer=r.get("answer", ""),
                chunks=_build_chunks_for_judge(r),
                config=config,
            )

            score_dict = judge_result.to_dict()
            score_dict["router_type"] = r.get("router_type", "heuristic")
            all_scores.append(score_dict)

            completed.add(key)
            _save_checkpoint(completed, checkpoint_path)

            if (done + 1) % 10 == 0:
                _save_scores(all_scores, scores_path)

            time.sleep(0.5)

        except Exception as e:
            logger.error("Judge error on %s/%s: %s", query_id, config, e)
            print(f"    ERROR: {e}")

        done += 1

    _save_scores(all_scores, scores_path)
    print(f"\nDone: scored {done} results with {judge_model}, saved to {scores_path}")
    return all_scores


def main():
    parser = argparse.ArgumentParser(description="Run LLM judge on ablation results")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Judge model: gpt-4o (default), claude, or gemini")
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
