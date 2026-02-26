"""Human-LLM judge correlation analysis.

Computes agreement between human annotators and LLM judges
across all evaluation dimensions (RCI, AS, VM, RA).
"""

from __future__ import annotations

import json
import os

from evaluation.significance import cohens_kappa, pearson_correlation


def compute_judge_correlation(
    human_annotations: list[dict],
    judge_results: list[dict],
) -> dict:
    """Compute correlation between human and LLM judge scores.

    Args:
        human_annotations: List of human annotation dicts.
        judge_results: List of LLM judge result dicts.

    Returns:
        Dict with per-dimension correlations and overall agreement.
    """
    # Match by (query_id, config)
    human_map = {}
    for ann in human_annotations:
        key = (ann["query_id"], ann["config"])
        human_map.setdefault(key, []).append(ann)

    judge_map = {}
    for jr in judge_results:
        key = (jr["query_id"], jr["config"])
        judge_map[key] = jr

    # Collect paired scores
    paired_rci_h, paired_rci_j = [], []
    paired_as_h, paired_as_j = [], []
    paired_vm_h, paired_vm_j = [], []
    paired_ra_h, paired_ra_j = [], []

    for key, h_anns in human_map.items():
        if key not in judge_map:
            continue

        j = judge_map[key]
        # Average human scores if multiple annotators
        avg_rci = sum(a["rci"] for a in h_anns) / len(h_anns)
        avg_as = sum(a["as"] for a in h_anns) / len(h_anns)
        avg_vm = sum(a["vm"] for a in h_anns) / len(h_anns)
        avg_ra = sum(a.get("ra", (a["rci"] + a["as"] + a["vm"]) / 6.0) for a in h_anns) / len(h_anns)

        paired_rci_h.append(avg_rci)
        paired_rci_j.append(j["rci"])
        paired_as_h.append(avg_as)
        paired_as_j.append(j["as"])
        paired_vm_h.append(avg_vm)
        paired_vm_j.append(j["vm"])
        paired_ra_h.append(avg_ra)
        paired_ra_j.append(j.get("ra", (j["rci"] + j["as"] + j["vm"]) / 6.0))

    result = {"n_paired": len(paired_ra_h)}

    if len(paired_ra_h) >= 3:
        result["rci_correlation"] = pearson_correlation(paired_rci_h, paired_rci_j)
        result["as_correlation"] = pearson_correlation(paired_as_h, paired_as_j)
        result["vm_correlation"] = pearson_correlation(paired_vm_h, paired_vm_j)
        result["ra_correlation"] = pearson_correlation(paired_ra_h, paired_ra_j)

        # Cohen's kappa (discretize to integer bins)
        rci_h_int = [round(v) for v in paired_rci_h]
        rci_j_int = [round(v) for v in paired_rci_j]
        if len(set(rci_h_int)) > 1 or len(set(rci_j_int)) > 1:
            result["rci_kappa"] = cohens_kappa(rci_h_int, rci_j_int)

    return result
