"""Inter-judge and human-LLM judge correlation analysis.

Computes agreement between any two raters (LLM judges or human annotators)
across all evaluation dimensions (RCI, AS, VM, RA).

Supports:
- Inter-judge: GPT-4o vs Claude vs Gemini (pairwise)
- Human-LLM: human annotations vs any LLM judge (future)

Metrics:
- Quadratic weighted kappa for ordinal dimensions (RCI, AS, VM: 0-2 scale)
- Spearman rho + Pearson r with bootstrap 95% CIs for all dimensions
- Kappa is NOT computed for RA (continuous [0,1] — discretizing is misleading)
"""

from __future__ import annotations

from evaluation.significance import (
    pearson_correlation,
    quadratic_weighted_kappa,
    spearman_correlation,
)

# Ordinal dimensions (0-2 scale) — suitable for weighted kappa
ORDINAL_DIMENSIONS = ["rci", "as", "vm"]


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

    Only includes a data point if BOTH raters have a non-None value for
    that dimension. Never defaults missing values to 0.

    Returns:
        Dict mapping dimension name -> (scores_a, scores_b) lists.
    """
    common_keys = set(map_a.keys()) & set(map_b.keys())

    paired: dict[str, tuple[list, list]] = {
        dim: ([], []) for dim in ORDINAL_DIMENSIONS + ["ra"]
    }

    for key in common_keys:
        a, b = map_a[key], map_b[key]

        # Skip entries where either judge had a parse error
        if a.get("reasoning") == "Parse error" or b.get("reasoning") == "Parse error":
            continue

        for dim in ORDINAL_DIMENSIONS:
            a_key = "as" if dim == "as" else dim
            # Only include if key exists in BOTH and values are not None
            if a_key in a and a_key in b:
                val_a, val_b = a[a_key], b[a_key]
                if val_a is not None and val_b is not None:
                    paired[dim][0].append(float(val_a))
                    paired[dim][1].append(float(val_b))

        # RA: only include if present in both
        if "ra" in a and "ra" in b:
            ra_a, ra_b = a["ra"], b["ra"]
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
    """Compute agreement between two sets of judge scores.

    - Quadratic weighted kappa for ordinal dimensions (RCI, AS, VM)
    - Spearman rho + Pearson r with bootstrap CIs for all dimensions
    - Kappa is NOT computed for RA (continuous float, not ordinal)

    Args:
        scores_a: Score dicts from rater A.
        scores_b: Score dicts from rater B.
        label_a: Display label for rater A.
        label_b: Display label for rater B.

    Returns:
        Dict with per-dimension agreement metrics and n_paired.
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

    for dim in ORDINAL_DIMENSIONS + ["ra"]:
        vals_a, vals_b = paired[dim]
        if len(vals_a) < 3:
            continue

        dim_result = {}

        # Spearman rank correlation (primary metric, with bootstrap CI)
        dim_result["spearman"] = spearman_correlation(vals_a, vals_b)

        # Pearson correlation (secondary, with bootstrap CI)
        dim_result["pearson"] = pearson_correlation(vals_a, vals_b)

        # Quadratic weighted kappa — only for ordinal dimensions (0-2)
        # NOT for RA which is continuous [0,1]
        if dim in ORDINAL_DIMENSIONS:
            int_a = [int(round(v)) for v in vals_a]
            int_b = [int(round(v)) for v in vals_b]
            # Both raters must have variance for kappa to be meaningful
            if len(set(int_a)) > 1 and len(set(int_b)) > 1:
                dim_result["weighted_kappa"] = quadratic_weighted_kappa(int_a, int_b)

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

    # Summary: average spearman and weighted kappa across all pairs
    avg_spearman = []
    avg_kappa_rci = []
    avg_kappa_as = []
    avg_kappa_vm = []
    for comp in comparisons:
        dims = comp.get("dimensions", {})
        ra = dims.get("ra", {})
        if "spearman" in ra and ra["spearman"].get("correlation") is not None:
            avg_spearman.append(ra["spearman"]["correlation"])
        for dim_name, kappa_list in [("rci", avg_kappa_rci), ("as", avg_kappa_as), ("vm", avg_kappa_vm)]:
            d = dims.get(dim_name, {})
            if "weighted_kappa" in d:
                kappa_list.append(d["weighted_kappa"])

    return {
        "judges": judges,
        "pairwise": comparisons,
        "summary": {
            "avg_ra_spearman": sum(avg_spearman) / len(avg_spearman) if avg_spearman else None,
            "avg_rci_weighted_kappa": sum(avg_kappa_rci) / len(avg_kappa_rci) if avg_kappa_rci else None,
            "avg_as_weighted_kappa": sum(avg_kappa_as) / len(avg_kappa_as) if avg_kappa_as else None,
            "avg_vm_weighted_kappa": sum(avg_kappa_vm) / len(avg_kappa_vm) if avg_kappa_vm else None,
            "n_pairs": len(comparisons),
        },
    }
