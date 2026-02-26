"""Table 6: Judge correlation analysis.

Shows human-LLM judge agreement (Pearson rho, Cohen's kappa).
"""

from __future__ import annotations

import json
import os

from evaluation.correlation import compute_judge_correlation


def generate_table6(
    annotations_path: str = "data/evaluation/annotations.json",
    judge_results_path: str = "data/evaluation/judge_results.json",
) -> dict:
    """Generate Table 6: Judge Correlation."""
    if not os.path.exists(annotations_path) or not os.path.exists(judge_results_path):
        return {"error": "Need both annotations.json and judge_results.json"}

    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    with open(judge_results_path, "r", encoding="utf-8") as f:
        judge_results = json.load(f)

    return compute_judge_correlation(annotations, judge_results)


if __name__ == "__main__":
    result = generate_table6()
    print(json.dumps(result, indent=2, default=str))
