"""Tests for the retrieval module: routers, reranker, metadata_hints."""

import numpy as np
import pytest

from processing.schemas import Chunk
from retrieval.router.heuristic import HeuristicRouter
from retrieval.router.adaptive import AdaptiveRouter
from retrieval.router.utility_signal import UtilitySignalCollector
from retrieval.reranker import rerank, _freshness_score, _authority_score
from retrieval.metadata_hints import extract_hints


def _make_chunk(chunk_id, source_type, feature_area="unknown", text="text", **meta):
    return Chunk(
        chunk_id=chunk_id, source_type=source_type, source_id=chunk_id,
        source_url=f"https://example.com/{chunk_id}",
        text=text, text_with_context=f"[{source_type}] {text}",
        feature_area=feature_area,
        created_at="2024-06-01", updated_at="2024-06-01",
        metadata=meta,
    )


# ===== Heuristic Router =====

class TestHeuristicRouter:
    def test_doc_query(self):
        router = HeuristicRouter()
        boosts = router.predict("How do I configure the terminal settings?")
        assert boosts["doc"] >= boosts["bug"]
        assert boosts["doc"] >= boosts["work_item"]

    def test_bug_query(self):
        router = HeuristicRouter()
        boosts = router.predict("VS Code crashes with error when opening files")
        assert boosts["bug"] >= boosts["work_item"]

    def test_work_item_query(self):
        router = HeuristicRouter()
        boosts = router.predict("What features are planned for the next release milestone?")
        assert boosts["work_item"] >= boosts["doc"]

    def test_neutral_query(self):
        router = HeuristicRouter()
        boosts = router.predict("VS Code")
        assert boosts["doc"] == boosts["bug"] == boosts["work_item"] == 0.5

    def test_all_boosts_in_range(self):
        router = HeuristicRouter()
        for query in ["configure settings", "error crash", "roadmap plan", "random stuff"]:
            boosts = router.predict(query)
            for v in boosts.values():
                assert 0.2 <= v <= 1.0


# ===== Adaptive Router =====

class TestAdaptiveRouter:
    def test_cold_start_equal_weights(self):
        router = AdaptiveRouter(feature_dim=16, cold_start_threshold=5)
        emb = np.random.randn(16).astype(np.float32)
        boosts = router.predict("test query", query_embedding=emb)
        assert boosts == {"doc": 0.5, "bug": 0.5, "work_item": 0.5}

    def test_cold_start_without_embedding(self):
        router = AdaptiveRouter(feature_dim=16, cold_start_threshold=5)
        boosts = router.predict("test query", query_embedding=None)
        assert boosts == {"doc": 0.5, "bug": 0.5, "work_item": 0.5}

    def test_update_changes_state(self):
        router = AdaptiveRouter(feature_dim=16, cold_start_threshold=0)
        emb = np.random.randn(16).astype(np.float64)
        router.update(emb, {"doc": 0.9, "bug": 0.1, "work_item": 0.3})
        assert router._query_count == 0  # count only increments on predict

    def test_alpha_decay(self):
        router = AdaptiveRouter(feature_dim=16, alpha_initial=1.0, alpha_decay=0.9, cold_start_threshold=0)
        emb = np.random.randn(16).astype(np.float64)
        router.update(emb, {"doc": 0.5, "bug": 0.5, "work_item": 0.5})
        assert router._alpha < 1.0

    def test_save_load_state(self, tmp_path):
        router = AdaptiveRouter(feature_dim=16, cold_start_threshold=0)
        emb = np.random.randn(16).astype(np.float64)
        # Update a few times
        for _ in range(3):
            router.update(emb, {"doc": 0.8, "bug": 0.2, "work_item": 0.5})

        save_dir = str(tmp_path / "router")
        router.save_state(save_dir)

        router2 = AdaptiveRouter(feature_dim=16)
        router2.load_state(save_dir)
        assert router2._alpha == pytest.approx(router._alpha)

    def test_stats(self):
        router = AdaptiveRouter(feature_dim=16, cold_start_threshold=10)
        stats = router.stats
        assert "query_count" in stats
        assert "alpha" in stats
        assert stats["cold_start_remaining"] == 10


# ===== Utility Signal =====

class TestUtilitySignal:
    def test_online_cited(self):
        collector = UtilitySignalCollector()
        score = collector.compute_online("doc", was_cited=True, retrieval_rank=1, total_retrieved=10)
        assert 0.5 < score <= 1.0

    def test_online_not_cited(self):
        collector = UtilitySignalCollector()
        score = collector.compute_online("bug", was_cited=False, retrieval_rank=10, total_retrieved=10)
        assert score < 0.5

    def test_offline_positive_drop(self):
        collector = UtilitySignalCollector()
        score = collector.compute_offline(ra_full=0.8, ra_without_source=0.5)
        assert score == pytest.approx(0.375)

    def test_offline_no_drop(self):
        collector = UtilitySignalCollector()
        score = collector.compute_offline(ra_full=0.8, ra_without_source=0.8)
        assert score == 0.0

    def test_offline_zero_full(self):
        collector = UtilitySignalCollector()
        score = collector.compute_offline(ra_full=0.0, ra_without_source=0.0)
        assert score == 0.0


# ===== Reranker =====

class TestReranker:
    @pytest.fixture
    def sample_candidates(self):
        return [
            (_make_chunk("d1", "doc", "terminal", verified=False), 0.9),
            (_make_chunk("b1", "bug", "terminal", verified=True, has_team_response=True), 0.85),
            (_make_chunk("w1", "work_item", "terminal", state="closed", total_reactions=20), 0.7),
            (_make_chunk("d2", "doc", "editor"), 0.6),
            (_make_chunk("b2", "bug", "editor", verified=False), 0.5),
        ]

    def test_rerank_returns_top_k(self, sample_candidates):
        boosts = {"doc": 0.8, "bug": 0.6, "work_item": 0.4}
        results = rerank(sample_candidates, boosts, top_k=3)
        assert len(results) == 3

    def test_rerank_scores_descending(self, sample_candidates):
        boosts = {"doc": 0.5, "bug": 0.5, "work_item": 0.5}
        results = rerank(sample_candidates, boosts, top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_empty(self):
        results = rerank([], {"doc": 0.5, "bug": 0.5, "work_item": 0.5}, top_k=5)
        assert results == []

    def test_freshness_recent(self):
        score = _freshness_score("2026-01-01T00:00:00+00:00")
        assert score > 0.5

    def test_freshness_empty(self):
        assert _freshness_score("") == 0.5

    def test_authority_verified_bug(self):
        chunk = _make_chunk("b", "bug", verified=True, has_team_response=True)
        score = _authority_score(chunk)
        assert score > 0.7

    def test_authority_doc(self):
        chunk = _make_chunk("d", "doc")
        score = _authority_score(chunk)
        assert score >= 0.7  # Docs get +0.2 base boost


# ===== Metadata Hints =====

class TestMetadataHints:
    def test_extract_feature_area(self):
        hints = extract_hints("terminal shell integration not working")
        assert hints.get("feature_area") == "terminal"

    def test_extract_os(self):
        hints = extract_hints("VS Code crashes on Windows 11")
        assert hints.get("os_platform") == "windows"

    def test_extract_os_macos(self):
        hints = extract_hints("Keyboard shortcuts not working on macOS")
        assert hints.get("os_platform") == "macos"

    def test_extract_state(self):
        hints = extract_hints("Is this bug fixed yet?")
        assert hints.get("state") == "closed"

    def test_extract_item_type_roadmap(self):
        hints = extract_hints("What's on the roadmap for next iteration?")
        assert hints.get("item_type") == "iteration_plan"

    def test_extract_empty(self):
        hints = extract_hints("hello")
        assert hints == {} or "feature_area" not in hints
