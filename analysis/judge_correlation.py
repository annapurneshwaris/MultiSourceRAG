"""Table 6: Inter-judge agreement matrix.

Computes pairwise quadratic weighted kappa (ordinal dimensions) and
Spearman rho with bootstrap 95% CIs across all LLM judges.
Optionally includes human annotations if available.

Output: 3x3 pairwise matrix (GPT-4o, Claude, Gemini) + optional human row.
"""

from __future__ import annotations

import json
import os

from evaluation.correlation import compute_inter_judge_matrix, compute_pairwise_agreement

# Resolve paths relative to repo root
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

JUDGE_FILES = {
    "gpt-4o": os.path.join(BASE, "data", "evaluation", "judge_scores_gpt4o.json"),
    "claude": os.path.join(BASE, "data", "evaluation", "judge_scores_claude.json"),
    "gemini": os.path.join(BASE, "data", "evaluation", "judge_scores_gemini.json"),
}
HUMAN_FILE = os.path.join(BASE, "data", "evaluation", "annotations.json")
LEGACY_GPT4O = os.path.join(BASE, "data", "evaluation", "judge_scores.json")


def _load_scores(path: str) -> list[dict] | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_table6() -> dict:
    """Generate Table 6: Inter-judge agreement matrix."""
    judge_scores: dict[str, list[dict]] = {}

    for name, path in JUDGE_FILES.items():
        scores = _load_scores(path)
        if scores is None and name == "gpt-4o":
            scores = _load_scores(LEGACY_GPT4O)
        if scores:
            judge_scores[name] = scores

    if len(judge_scores) < 2:
        available = list(judge_scores.keys())
        return {"error": f"Need at least 2 judges. Available: {available}"}

    result = compute_inter_judge_matrix(judge_scores)

    # Add human annotations if available
    human_scores = _load_scores(HUMAN_FILE)
    if human_scores:
        result["human_comparisons"] = []
        for judge_name, judge_data in judge_scores.items():
            pair = compute_pairwise_agreement(
                human_scores, judge_data,
                label_a="human", label_b=judge_name,
            )
            result["human_comparisons"].append(pair)

    return result


if __name__ == "__main__":
    result = generate_table6()
    print(json.dumps(result, indent=2, default=str))
