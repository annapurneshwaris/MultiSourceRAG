"""Resolution-centric evaluation metrics — C2 contribution.

RA (Resolution Adequacy): composite of RCI + AS + VM
CSAS (Cross-Source Attribution Score): citation precision per source
MSUR (Multi-Source Utilization Rate): source diversity in results
"""

from __future__ import annotations

from dataclasses import dataclass

# Root cause categories (collapsed from 9 to 4 for inter-judge reliability)
ROOT_CAUSE_CATEGORIES = [
    "configuration",    # settings, preferences, workspace config, user error
    "bug_or_issue",     # known bugs, extension conflicts, platform-specific, performance
    "gap_or_missing",   # missing features, documentation gaps, feature requests
    "unknown",          # cannot determine
]

# Mapping from legacy 9-category to new 4-category taxonomy
LEGACY_CATEGORY_MAP = {
    "configuration": "configuration",
    "user_error": "configuration",
    "extension_conflict": "bug_or_issue",
    "known_bug": "bug_or_issue",
    "platform_specific": "bug_or_issue",
    "performance": "bug_or_issue",
    "missing_feature": "gap_or_missing",
    "documentation_gap": "gap_or_missing",
    "unknown": "unknown",
}


@dataclass
class RAScore:
    """Resolution Adequacy composite score."""
    rci: int    # Root Cause Identification (0-2)
    as_: int    # Actionable Steps (0-2)
    vm: int     # Version Match (0-2)
    root_cause_category: str = "unknown"

    @property
    def ra(self) -> float:
        """RA = (RCI + AS + VM) / 6, in [0, 1]."""
        return (self.rci + self.as_ + self.vm) / 6.0


@dataclass
class CSASScore:
    """Cross-Source Attribution Score."""
    doc_cited: int = 0
    bug_cited: int = 0
    work_item_cited: int = 0
    doc_expected: bool = False
    bug_expected: bool = False
    work_item_expected: bool = False

    @property
    def precision(self) -> float:
        """Fraction of expected sources that were actually cited."""
        expected = sum([self.doc_expected, self.bug_expected, self.work_item_expected])
        if expected == 0:
            return 1.0

        cited_correctly = 0
        if self.doc_expected and self.doc_cited > 0:
            cited_correctly += 1
        if self.bug_expected and self.bug_cited > 0:
            cited_correctly += 1
        if self.work_item_expected and self.work_item_cited > 0:
            cited_correctly += 1

        return cited_correctly / expected

    @property
    def recall(self) -> float:
        """Fraction of cited sources that were expected."""
        total_cited = sum([
            1 if self.doc_cited > 0 else 0,
            1 if self.bug_cited > 0 else 0,
            1 if self.work_item_cited > 0 else 0,
        ])
        if total_cited == 0:
            return 0.0

        cited_correctly = 0
        if self.doc_expected and self.doc_cited > 0:
            cited_correctly += 1
        if self.bug_expected and self.bug_cited > 0:
            cited_correctly += 1
        if self.work_item_expected and self.work_item_cited > 0:
            cited_correctly += 1

        return cited_correctly / total_cited

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


def compute_ra(rci: int, as_: int, vm: int) -> float:
    """Compute RA composite score."""
    return (rci + as_ + vm) / 6.0


def compute_csas(
    citations: dict[str, list[str]],
    expected_sources: list[str],
) -> CSASScore:
    """Compute CSAS from citations and expected sources.

    Args:
        citations: {"doc": [...], "bug": [...], "work_item": [...]}
        expected_sources: ["doc", "bug", "work_item"]
    """
    return CSASScore(
        doc_cited=len(citations.get("doc", [])),
        bug_cited=len(citations.get("bug", [])),
        work_item_cited=len(citations.get("work_item", [])),
        doc_expected="doc" in expected_sources,
        bug_expected="bug" in expected_sources,
        work_item_expected="work_item" in expected_sources,
    )


def compute_msur(results: list[dict]) -> float:
    """Compute Multi-Source Utilization Rate across a set of results.

    MSUR = average fraction of source types represented in top-k results.

    Args:
        results: List of pipeline result dicts (from pipeline.process_query).

    Returns:
        MSUR score in [0, 1]. Higher = better source diversity.
    """
    if not results:
        return 0.0

    all_sources = {"doc", "bug", "work_item"}
    utilization_scores = []

    for result in results:
        chunks = result.get("reranked_chunks", [])
        sources_present = set(c["source_type"] for c in chunks)
        utilization = len(sources_present & all_sources) / len(all_sources)
        utilization_scores.append(utilization)

    return sum(utilization_scores) / len(utilization_scores)
