"""Tests for the evaluation module: metrics, query_bank, significance."""

import pytest

from evaluation.metrics import (
    compute_ra, compute_csas, compute_msur,
    RAScore, CSASScore, ROOT_CAUSE_CATEGORIES,
)
from evaluation.query_bank import (
    load_queries, SEED_QUERIES, stratify_by_category, get_category_distribution,
    EvalQuery,
)


# ===== RA Score =====

class TestRAScore:
    def test_perfect_score(self):
        assert compute_ra(rci=2, as_=2, vm=2) == 1.0

    def test_zero_score(self):
        assert compute_ra(rci=0, as_=0, vm=0) == 0.0

    def test_partial_score(self):
        assert compute_ra(rci=1, as_=1, vm=1) == pytest.approx(0.5)

    def test_ra_dataclass(self):
        score = RAScore(rci=2, as_=1, vm=0)
        assert score.ra == pytest.approx(0.5)

    def test_root_cause_categories_exist(self):
        assert "configuration" in ROOT_CAUSE_CATEGORIES
        assert "known_bug" in ROOT_CAUSE_CATEGORIES
        assert "unknown" in ROOT_CAUSE_CATEGORIES
        assert len(ROOT_CAUSE_CATEGORIES) == 9


# ===== CSAS Score =====

class TestCSASScore:
    def test_perfect_precision(self):
        csas = compute_csas(
            citations={"doc": ["d1"], "bug": ["b1"], "work_item": []},
            expected_sources=["doc", "bug"],
        )
        assert csas.precision == 1.0

    def test_zero_precision(self):
        csas = compute_csas(
            citations={"doc": [], "bug": [], "work_item": ["w1"]},
            expected_sources=["doc", "bug"],
        )
        assert csas.precision == 0.0

    def test_partial_precision(self):
        csas = compute_csas(
            citations={"doc": ["d1"], "bug": [], "work_item": []},
            expected_sources=["doc", "bug"],
        )
        assert csas.precision == 0.5

    def test_recall(self):
        csas = compute_csas(
            citations={"doc": ["d1"], "bug": ["b1"], "work_item": ["w1"]},
            expected_sources=["doc"],
        )
        assert csas.recall == pytest.approx(1 / 3)  # 1 correct out of 3 cited

    def test_f1(self):
        csas = compute_csas(
            citations={"doc": ["d1"], "bug": [], "work_item": []},
            expected_sources=["doc"],
        )
        assert csas.f1 == 1.0  # Perfect precision and recall

    def test_no_expected(self):
        csas = compute_csas(
            citations={"doc": ["d1"], "bug": [], "work_item": []},
            expected_sources=[],
        )
        assert csas.precision == 1.0  # Vacuously true


# ===== MSUR =====

class TestMSUR:
    def test_perfect_msur(self):
        results = [
            {"reranked_chunks": [
                {"source_type": "doc"}, {"source_type": "bug"}, {"source_type": "work_item"}
            ]},
        ]
        assert compute_msur(results) == 1.0

    def test_single_source_msur(self):
        results = [
            {"reranked_chunks": [{"source_type": "doc"}, {"source_type": "doc"}]},
        ]
        assert compute_msur(results) == pytest.approx(1 / 3)

    def test_empty_results(self):
        assert compute_msur([]) == 0.0

    def test_average_across_results(self):
        results = [
            {"reranked_chunks": [{"source_type": "doc"}, {"source_type": "bug"}, {"source_type": "work_item"}]},
            {"reranked_chunks": [{"source_type": "doc"}]},
        ]
        # (3/3 + 1/3) / 2 = 2/3
        assert compute_msur(results) == pytest.approx(2 / 3)


# ===== Query Bank =====

class TestQueryBank:
    def test_seed_queries_exist(self):
        assert len(SEED_QUERIES) >= 250

    def test_seed_queries_have_required_fields(self):
        for q in SEED_QUERIES:
            assert "query_id" in q
            assert "query_text" in q
            assert "category" in q
            assert "expected_sources" in q
            assert "expected_area" in q

    def test_load_queries_returns_seed(self):
        queries = load_queries("nonexistent_path.json")
        assert len(queries) == len(SEED_QUERIES)
        assert all(isinstance(q, EvalQuery) for q in queries)

    def test_stratify_by_category(self):
        queries = [EvalQuery.from_dict(q) for q in SEED_QUERIES]
        groups = stratify_by_category(queries)
        assert "how_to" in groups
        assert "debugging" in groups
        assert len(groups) >= 4

    def test_category_distribution(self):
        queries = [EvalQuery.from_dict(q) for q in SEED_QUERIES]
        dist = get_category_distribution(queries)
        assert sum(dist.values()) == len(queries)

    def test_eval_query_roundtrip(self):
        q = EvalQuery(
            query_id="test_1", query_text="Test?", category="how_to",
            expected_sources=["doc"], expected_area="editor",
            difficulty="easy", ground_truth_notes="Test note",
        )
        d = q.to_dict()
        restored = EvalQuery.from_dict(d)
        assert restored.query_id == "test_1"
        assert restored.ground_truth_notes == "Test note"


# ===== Significance =====

class TestSignificance:
    def test_paired_bootstrap(self):
        from evaluation.significance import paired_bootstrap
        scores_a = [0.8, 0.7, 0.9, 0.85, 0.75]
        scores_b = [0.5, 0.4, 0.6, 0.55, 0.45]
        result = paired_bootstrap(scores_a, scores_b)
        assert result["observed_diff"] > 0
        assert result["significant"]
        assert "p_value" in result

    def test_paired_bootstrap_same_scores(self):
        from evaluation.significance import paired_bootstrap
        scores = [0.5, 0.6, 0.5, 0.6, 0.5]
        result = paired_bootstrap(scores, scores)
        assert result["observed_diff"] == 0.0

    def test_cohens_kappa_perfect(self):
        from evaluation.significance import cohens_kappa
        ratings = [0, 1, 2, 0, 1, 2]
        kappa = cohens_kappa(ratings, ratings)
        assert kappa == 1.0

    def test_cohens_kappa_partial(self):
        from evaluation.significance import cohens_kappa
        a = [0, 1, 2, 1, 0]
        b = [0, 1, 1, 1, 0]
        kappa = cohens_kappa(a, b)
        assert 0 < kappa < 1.0

    def test_pearson_correlation(self):
        from evaluation.significance import pearson_correlation
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.1, 2.1, 2.9, 4.0, 5.1]
        result = pearson_correlation(a, b)
        assert result["correlation"] > 0.95
        assert result["significant"]
