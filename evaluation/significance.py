"""Statistical significance tests for evaluation results.

- Paired bootstrap (10K resamples)
- Paired t-test
- Cohen's kappa (unweighted + quadratic weighted)
- Pearson correlation (with bootstrap CI)
- Spearman correlation (with bootstrap CI)
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def paired_bootstrap(
    scores_a: list[float],
    scores_b: list[float],
    n_resamples: int = 10000,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap significance test.

    Tests whether system A is significantly better than system B.

    Returns:
        Dict with p_value, mean_diff, ci_lower, ci_upper, significant.
    """
    assert len(scores_a) == len(scores_b), "Score lists must have same length"
    n = len(scores_a)
    arr_a = np.array(scores_a)
    arr_b = np.array(scores_b)

    observed_diff = np.mean(arr_a) - np.mean(arr_b)

    # Bootstrap
    rng = np.random.default_rng(42)
    diffs = np.zeros(n_resamples)
    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        diffs[i] = np.mean(arr_a[indices]) - np.mean(arr_b[indices])

    p_value = np.mean(diffs <= 0) if observed_diff > 0 else np.mean(diffs >= 0)
    ci_lower = np.percentile(diffs, 100 * alpha / 2)
    ci_upper = np.percentile(diffs, 100 * (1 - alpha / 2))

    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "significant": p_value < alpha,
    }


def paired_t_test(
    scores_a: list[float],
    scores_b: list[float],
) -> dict:
    """Paired t-test between two systems."""
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


def cohens_kappa(
    ratings_a: list[int],
    ratings_b: list[int],
) -> float:
    """Compute Cohen's kappa (unweighted) for inter-annotator agreement."""
    assert len(ratings_a) == len(ratings_b)
    n = len(ratings_a)

    agreements = sum(1 for a, b in zip(ratings_a, ratings_b) if a == b)
    p_o = agreements / n

    unique_vals = set(ratings_a) | set(ratings_b)
    p_e = 0.0
    for val in unique_vals:
        p_a = sum(1 for r in ratings_a if r == val) / n
        p_b = sum(1 for r in ratings_b if r == val) / n
        p_e += p_a * p_b

    if p_e == 1.0:
        return 1.0

    return (p_o - p_e) / (1.0 - p_e)


def quadratic_weighted_kappa(
    ratings_a: list[int],
    ratings_b: list[int],
) -> float:
    """Compute quadratic weighted Cohen's kappa for ordinal scales.

    Appropriate for ordinal data (e.g., 0-1-2 ratings) where
    disagreement by 2 levels is worse than by 1 level.
    """
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(ratings_a, ratings_b, weights="quadratic"))


def _has_variance(scores: list[float]) -> bool:
    """Check if a list has at least 2 distinct values."""
    return len(set(scores)) >= 2


def _bootstrap_ci(
    scores_a: list[float],
    scores_b: list[float],
    stat_fn,
    n_resamples: int = 10000,
    alpha: float = 0.05,
) -> dict:
    """Compute bootstrap 95% CI for a correlation statistic.

    Args:
        stat_fn: callable(a, b) -> float (e.g., spearmanr or pearsonr).
    """
    rng = np.random.default_rng(42)
    arr_a = np.array(scores_a)
    arr_b = np.array(scores_b)
    n = len(arr_a)

    boot_stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sa, sb = arr_a[idx], arr_b[idx]
        if _has_variance(sa.tolist()) and _has_variance(sb.tolist()):
            val, _ = stat_fn(sa, sb)
            if not np.isnan(val):
                boot_stats.append(val)

    if len(boot_stats) < 100:
        return {"ci_lower": None, "ci_upper": None}

    boot_stats = np.array(boot_stats)
    return {
        "ci_lower": float(np.percentile(boot_stats, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
    }


def pearson_correlation(
    scores_a: list[float],
    scores_b: list[float],
    bootstrap_ci: bool = True,
) -> dict:
    """Compute Pearson correlation with optional bootstrap 95% CI."""
    if not _has_variance(scores_a) or not _has_variance(scores_b):
        return {"correlation": None, "p_value": None, "significant": False,
                "note": "constant_input"}

    r, p_value = stats.pearsonr(scores_a, scores_b)
    result = {
        "correlation": float(r),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }

    if bootstrap_ci:
        ci = _bootstrap_ci(scores_a, scores_b, stats.pearsonr)
        result["ci_lower"] = ci["ci_lower"]
        result["ci_upper"] = ci["ci_upper"]

    return result


def spearman_correlation(
    scores_a: list[float],
    scores_b: list[float],
    bootstrap_ci: bool = True,
) -> dict:
    """Compute Spearman rank correlation with optional bootstrap 95% CI."""
    if not _has_variance(scores_a) or not _has_variance(scores_b):
        return {"correlation": None, "p_value": None, "significant": False,
                "note": "constant_input"}

    rho, p_value = stats.spearmanr(scores_a, scores_b)

    if np.isnan(rho):
        return {"correlation": None, "p_value": None, "significant": False,
                "note": "nan_result"}

    result = {
        "correlation": float(rho),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }

    if bootstrap_ci:
        ci = _bootstrap_ci(scores_a, scores_b, stats.spearmanr)
        result["ci_lower"] = ci["ci_lower"]
        result["ci_upper"] = ci["ci_upper"]

    return result
