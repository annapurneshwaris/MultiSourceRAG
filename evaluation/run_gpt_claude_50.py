"""Re-run GPT-4o and Claude on 50-query subset with v2 prompt."""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUBSET_PATH = os.path.join(BASE, "data", "evaluation", "test_subset_50.json")


def run_judge(judge_name, judge_model, provider_factory, out_path):
    from evaluation.llm_judge import LLMJudge
    from evaluation.judge_runner import _build_chunks_for_judge

    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        subset = json.load(f)

    scores = []
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            scores = json.load(f)
        print(f"Loaded {len(scores)} existing {judge_name} scores")

    scored_keys = {(s["query_id"], s["config"]) for s in scores}
    remaining = [r for r in subset if (r.get("query_id"), r.get("config")) not in scored_keys]

    if not remaining:
        print(f"{judge_name}: all 50 already scored")
        return

    print(f"Running {judge_name} on {len(remaining)} remaining queries...")
    judge = LLMJudge(llm_provider=provider_factory(), judge_model=judge_model)

    for i, r in enumerate(remaining):
        qid = r.get("query_id", "")
        cfg = r.get("config", "DBW")
        print(f"  {judge_name} [{i+1}/{len(remaining)}] {cfg}|{qid}")
        try:
            result = judge.evaluate(
                query_id=qid,
                query_text=r.get("query", ""),
                expected_sources=r.get("expected_sources", []),
                answer=r.get("answer", ""),
                chunks=_build_chunks_for_judge(r),
                config=cfg,
            )
            d = result.to_dict()
            d["router_type"] = r.get("router_type", "heuristic")
            scores.append(d)

            if (i + 1) % 10 == 0:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(scores, f, indent=2, ensure_ascii=False)
                print(f"    Saved ({len(scores)} total)")

            time.sleep(0.3)
        except Exception as e:
            print(f"    ERROR: {e}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"{judge_name} done: {len(scores)} scores")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=["gpt4o", "claude"], required=True)
    args = parser.parse_args()

    if args.judge == "gpt4o":
        from generation.providers.openai_gen import OpenAIProvider
        run_judge(
            "GPT-4o", "gpt-4o-2024-11-20",
            lambda: OpenAIProvider(),
            os.path.join(BASE, "data", "evaluation", "judge_scores_v2_gpt4o.json"),
        )
    else:
        from generation.providers.anthropic_gen import AnthropicProvider
        run_judge(
            "Claude", "claude-sonnet-4-20250514",
            lambda: AnthropicProvider(),
            os.path.join(BASE, "data", "evaluation", "judge_scores_v2_claude.json"),
        )


if __name__ == "__main__":
    main()
