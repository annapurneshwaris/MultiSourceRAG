"""Inter-judge and human-LLM judge correlation analysis.

Computes agreement between any two raters (LLM judges or human annotators)
across all evaluation dimensions (RCI, AS, VM, RA).

Supports:
- Inter-judge: GPT-4o vs Claude vs Gemini (pairwise)
- Human-LLM: human annotations vs any LLM judge (future)
"""

from __future__ import annotations

from evaluation.significance import cohens_kappa, pearson_correlation, spearman_correlation

DIMENSIONS = ["rci", "as", "vm"]


def _build_score_map(scores: list[dict]) -> dict[tuple[str, str], dict]:
    """Key scores by (query_id, config)."""
    result = {}
    for s in scores:
        key = (s["query_id"], s["config"])
        result[key] = s
    return result


def _extract_paired(
    map_a: dict[tuple[str, str], dict],
    map_b: dict[tuple[str, str], dict],
) -> dict[str, tuple[list[float], list[float]]]:
    """Extract paired score lists for all dimensions from two score maps.

    Returns:
        Dict mapping dimension name -> (scores_a, scores_b) lists.
    """
    common_keys = set(map_a.keys()) & set(map_b.keys())

    paired: dict[str, tuple[list, list]] = {
        dim: ([], []) for dim in DIMENSIONS + ["ra"]
    }

    for key in common_keys:
        a, b = map_a[key], map_b[key]
        for dim in DIMENSIONS:
            a_key = "as" if dim == "as" else dim
            val_a = a.get(a_key, 0)
            val_b = b.get(a_key, 0)
            if val_a is not None and val_b is not None:
                paired[dim][0].append(float(val_a))
                paired[dim][1].append(float(val_b))

        ra_a = a.get("ra")
        ra_b = b.get("ra")
        if ra_a is not None and ra_b is not None:
            paired["ra"][0].append(float(ra_a))
            paired["ra"][1].append(float(ra_b))

    return paired


def compute_pairwise_agreement(
    scores_a: list[dict],
    scores_b: list[dict],
    label_a: str = "judge_a",
    label_b: str = "judge_b",
) -> dict:
    """Compute Cohen's kappa and Spearman rho between two sets of judge scores.

    Works for any two raters: LLM-LLM or human-LLM.

    Args:
        scores_a: Score dicts from rater A.
        scores_b: Score dicts from rater B.
        label_a: Display label for rater A.
        label_b: Display label for rater B.

    Returns:
        Dict with per-dimension kappa, spearman, pearson, and n_paired.
    """
    map_a = _build_score_map(scores_a)
    map_b = _build_score_map(scores_b)
    paired = _extract_paired(map_a, map_b)

    result = {
        "rater_a": label_a,
        "rater_b": label_b,
        "n_paired": len(paired["ra"][0]),
        "dimensions": {},
    }

    for dim in DIMENSIONS + ["ra"]:
        vals_a, vals_b = paired[dim]
        if len(vals_a) < 3:
            continue

        dim_result = {}

        # Spearman rank correlation (primary for paper)
        dim_result["spearman"] = spearman_correlation(vals_a, vals_b)

        # Pearson correlation
        dim_result["pearson"] = pearson_correlation(vals_a, vals_b)

        # Cohen's kappa (discretize to integer bins for ordinal data)
        int_a = [round(v) for v in vals_a]
        int_b = [round(v) for v in vals_b]
        if len(set(int_a)) > 1 or len(set(int_b)) > 1:
            dim_result["kappa"] = cohens_kappa(int_a, int_b)

        result["dimensions"][dim] = dim_result

    return result


def compute_inter_judge_matrix(
    judge_scores: dict[str, list[dict]],
) -> dict:
    """Compute full pairwise agreement matrix across all judges.

    Args:
        judge_scores: Dict mapping judge name -> list of score dicts.
            e.g. {"gpt-4o": [...], "claude": [...], "gemini": [...]}

    Returns:
        Dict with pairwise comparisons and summary statistics.
    """
    judges = list(judge_scores.keys())
    comparisons = []

    for i in range(len(judges)):
        for j in range(i + 1, len(judges)):
            pair = compute_pairwise_agreement(
                judge_scores[judges[i]],
                judge_scores[judges[j]],
                label_a=judges[i],
                label_b=judges[j],
            )
            comparisons.append(pair)

    # Summary: average kappa and spearman across all pairs for RA
    avg_kappa = []
    avg_spearman = []
    for comp in comparisons:
        ra_dim = comp.get("dimensions", {}).get("ra", {})
        if "kappa" in ra_dim:
            avg_kappa.append(ra_dim["kappa"])
        if "spearman" in ra_dim:
            avg_spearman.append(ra_dim["spearman"]["correlation"])

    return {
        "judges": judges,
        "pairwise": comparisons,
        "summary": {
            "avg_ra_kappa": sum(avg_kappa) / len(avg_kappa) if avg_kappa else None,
            "avg_ra_spearman": sum(avg_spearman) / len(avg_spearman) if avg_spearman else None,
            "n_pairs": len(comparisons),
        },
    }
