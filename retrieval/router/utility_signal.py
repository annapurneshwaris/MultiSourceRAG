"""Utility signal computation for adaptive router feedback.

Two modes:
- Online: quick signal from citations + retrieval quality + position
- Offline: ablation-based RA drop (for paper evaluation)
"""

from __future__ import annotations


class UtilitySignalCollector:
    """Computes per-source utility signals for router learning."""

    def compute_online(
        self,
        source_type: str,
        was_cited: bool,
        retrieval_rank: int,
        total_retrieved: int,
    ) -> float:
        """Compute online utility signal for a single source.

        Formula: 0.4 × was_cited + 0.3 × retrieval_quality + 0.3 × source_position

        Args:
            source_type: "doc", "bug", or "work_item".
            was_cited: Whether this source was cited in the answer.
            retrieval_rank: Best rank position of this source's chunk (1-indexed).
            total_retrieved: Total number of chunks retrieved.

        Returns:
            Utility score in [0, 1].
        """
        # Citation signal (binary)
        citation_signal = 1.0 if was_cited else 0.0

        # Retrieval quality: inverse rank (higher = better)
        if retrieval_rank > 0 and total_retrieved > 0:
            retrieval_quality = 1.0 - (retrieval_rank - 1) / total_retrieved
        else:
            retrieval_quality = 0.0

        # Source position: how early this source appears in results
        position_score = retrieval_quality  # Same as rank-based for now

        utility = (
            0.4 * citation_signal
            + 0.3 * retrieval_quality
            + 0.3 * position_score
        )

        return min(1.0, max(0.0, utility))

    def compute_offline(
        self,
        ra_full: float,
        ra_without_source: float,
    ) -> float:
        """Compute offline utility from ablation study.

        Formula: max(0, RA_full - RA_without_source) / RA_full

        Args:
            ra_full: RA score with all sources.
            ra_without_source: RA score without this source.

        Returns:
            Utility score in [0, 1]. Higher = source is more important.
        """
        if ra_full <= 0:
            return 0.0
        drop = ra_full - ra_without_source
        return max(0.0, drop / ra_full)
