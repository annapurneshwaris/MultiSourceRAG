"""Statistical significance tests for evaluation results.

- Paired bootstrap (10K resamples)
- Paired t-test
- Cohen's kappa (inter-annotator agreement)
- Pearson correlation (human-LLM judge)
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
    """Compute Cohen's kappa for inter-annotator agreement.

    Args:
        ratings_a: Ratings from annotator A.
        ratings_b: Ratings from annotator B (same items).

    Returns:
        Kappa value in [-1, 1]. >0.6 is substantial, >0.8 is excellent.
    """
    assert len(ratings_a) == len(ratings_b)
    n = len(ratings_a)

    # Count agreements
    agreements = sum(1 for a, b in zip(ratings_a, ratings_b) if a == b)
    p_o = agreements / n  # Observed agreement

    # Expected agreement by chance
    unique_vals = set(ratings_a) | set(ratings_b)
    p_e = 0.0
    for val in unique_vals:
        p_a = sum(1 for r in ratings_a if r == val) / n
        p_b = sum(1 for r in ratings_b if r == val) / n
        p_e += p_a * p_b

    if p_e == 1.0:
        return 1.0

    return (p_o - p_e) / (1.0 - p_e)


def pearson_correlation(
    scores_a: list[float],
    scores_b: list[float],
) -> dict:
    """Compute Pearson correlation between two sets of scores."""
    r, p_value = stats.pearsonr(scores_a, scores_b)
    return {
        "correlation": float(r),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


def spearman_correlation(
    scores_a: list[float],
    scores_b: list[float],
) -> dict:
    """Compute Spearman rank correlation between two sets of scores."""
    rho, p_value = stats.spearmanr(scores_a, scores_b)
    return {
        "correlation": float(rho),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }
