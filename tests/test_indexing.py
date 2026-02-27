"""Tests for the indexing module: metadata_index, bm25, stores."""

import numpy as np
import pytest

from processing.schemas import Chunk
from indexing.metadata_index import MetadataIndex
from indexing.bm25_index import BM25Index


def _make_chunk(chunk_id, source_type, feature_area, text="sample text", **meta):
    return Chunk(
        chunk_id=chunk_id,
        source_type=source_type,
        source_id=chunk_id,
        source_url=f"https://example.com/{chunk_id}",
        text=text,
        text_with_context=f"[{source_type.upper()}] {text}",
        feature_area=feature_area,
        created_at="2024-01-01",
        updated_at="2024-01-01",
        metadata=meta,
    )


# ===== Metadata Index =====

class TestMetadataIndex:
    @pytest.fixture
    def sample_chunks(self):
        return [
            _make_chunk("c0", "doc", "terminal"),
            _make_chunk("c1", "doc", "editor"),
            _make_chunk("c2", "bug", "terminal", state="open", verified=True),
            _make_chunk("c3", "bug", "editor", state="closed", verified=False),
            _make_chunk("c4", "work_item", "terminal", item_type="feature_request"),
            _make_chunk("c5", "work_item", "editor", item_type="iteration_plan"),
        ]

    def test_build_and_filter_source_type(self, sample_chunks):
        idx = MetadataIndex()
        idx.build(sample_chunks)
        docs = idx.filter({"source_type": "doc"})
        assert docs == {0, 1}

    def test_filter_feature_area(self, sample_chunks):
        idx = MetadataIndex()
        idx.build(sample_chunks)
        terminal = idx.filter({"feature_area": "terminal"})
        assert terminal == {0, 2, 4}

    def test_filter_combined_and(self, sample_chunks):
        idx = MetadataIndex()
        idx.build(sample_chunks)
        result = idx.filter({"source_type": "bug", "feature_area": "terminal"})
        assert result == {2}

    def test_filter_list_values_or(self, sample_chunks):
        idx = MetadataIndex()
        idx.build(sample_chunks)
        result = idx.filter({"source_type": ["doc", "bug"]})
        assert result == {0, 1, 2, 3}

    def test_filter_empty_constraints(self, sample_chunks):
        idx = MetadataIndex()
        idx.build(sample_chunks)
        assert idx.filter({}) == set()

    def test_filter_nonexistent_field(self, sample_chunks):
        idx = MetadataIndex()
        idx.build(sample_chunks)
        assert idx.filter({"nonexistent": "value"}) == set()

    def test_get_values(self, sample_chunks):
        idx = MetadataIndex()
        idx.build(sample_chunks)
        source_types = idx.get_values("source_type")
        assert "doc" in source_types
        assert "bug" in source_types
        assert "work_item" in source_types

    def test_get_count(self, sample_chunks):
        idx = MetadataIndex()
        idx.build(sample_chunks)
        assert idx.get_count("source_type", "doc") == 2
        assert idx.get_count("source_type", "bug") == 2

    def test_stats(self, sample_chunks):
        idx = MetadataIndex()
        idx.build(sample_chunks)
        stats = idx.stats()
        assert "source_type" in stats
        assert stats["source_type"]["unique_values"] == 3

    def test_save_load(self, sample_chunks, tmp_path):
        idx = MetadataIndex()
        idx.build(sample_chunks)

        save_path = str(tmp_path / "meta.pkl")
        idx.save(save_path)

        idx2 = MetadataIndex()
        idx2.load(save_path)
        assert idx2.filter({"source_type": "doc"}) == {0, 1}


# ===== BM25 Index =====

class TestBM25Index:
    @pytest.fixture
    def sample_chunks(self):
        return [
            _make_chunk("c0", "doc", "terminal", text="How to configure the integrated terminal in VS Code"),
            _make_chunk("c1", "doc", "editor", text="Basic editing features and keyboard shortcuts"),
            _make_chunk("c2", "bug", "terminal", text="Terminal crashes when using PowerShell on Windows"),
            _make_chunk("c3", "bug", "git", text="Git extension fails to detect repository changes"),
        ]

    def test_build_and_search(self, sample_chunks):
        bm25 = BM25Index()
        bm25.build(sample_chunks)
        results = bm25.search("terminal configure integrated", top_k=2)
        assert len(results) > 0
        # Top result should be terminal-related
        top_chunk, top_score = results[0]
        assert "terminal" in top_chunk.text.lower()

    def test_search_returns_scores(self, sample_chunks):
        bm25 = BM25Index()
        bm25.build(sample_chunks)
        results = bm25.search("terminal", top_k=4)
        # Scores should be descending
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_query(self, sample_chunks):
        bm25 = BM25Index()
        bm25.build(sample_chunks)
        results = bm25.search("", top_k=2)
        assert len(results) == 0

    def test_search_no_match(self, sample_chunks):
        bm25 = BM25Index()
        bm25.build(sample_chunks)
        results = bm25.search("xyznonexistentterm", top_k=2)
        assert len(results) == 0

    def test_len(self, sample_chunks):
        bm25 = BM25Index()
        bm25.build(sample_chunks)
        assert len(bm25) == 4

    def test_save_load(self, sample_chunks, tmp_path):
        bm25 = BM25Index()
        bm25.build(sample_chunks)

        save_dir = str(tmp_path / "bm25")
        bm25.save(save_dir)

        bm25_loaded = BM25Index()
        bm25_loaded.load(save_dir)
        results = bm25_loaded.search("terminal crashes powershell", top_k=2)
        assert len(results) > 0


# ===== FAISS Store =====

class TestFAISSStore:
    @pytest.fixture
    def sample_data(self):
        chunks = [
            _make_chunk("c0", "doc", "terminal", text="Terminal config"),
            _make_chunk("c1", "bug", "editor", text="Editor crash"),
            _make_chunk("c2", "work_item", "git", text="Git feature"),
        ]
        # Random embeddings (384 dim)
        embeddings = np.random.randn(3, 384).astype(np.float32)
        return chunks, embeddings

    def test_add_and_search(self, sample_data):
        from indexing.stores.faiss_store import FAISSStore
        chunks, embeddings = sample_data
        store = FAISSStore(dimension=384)
        store.add(embeddings, chunks)
        assert len(store) == 3

        query = np.random.randn(384).astype(np.float32)
        results = store.search(query, top_k=2)
        assert len(results) == 2
        assert all(isinstance(r[0], Chunk) for r in results)
        assert all(isinstance(r[1], float) for r in results)

    def test_search_with_filter(self, sample_data):
        from indexing.stores.faiss_store import FAISSStore
        chunks, embeddings = sample_data
        store = FAISSStore(dimension=384)
        store.add(embeddings, chunks)

        query = np.random.randn(384).astype(np.float32)
        results = store.search(query, top_k=3, filter_ids={0, 2})
        assert len(results) <= 2
        for chunk, _ in results:
            assert chunk.chunk_id in ("c0", "c2")

    def test_search_empty_store(self):
        from indexing.stores.faiss_store import FAISSStore
        store = FAISSStore(dimension=384)
        query = np.random.randn(384).astype(np.float32)
        results = store.search(query, top_k=5)
        assert results == []

    def test_save_load(self, sample_data, tmp_path):
        from indexing.stores.faiss_store import FAISSStore
        chunks, embeddings = sample_data
        store = FAISSStore(dimension=384)
        store.add(embeddings, chunks)

        save_dir = str(tmp_path / "faiss_test")
        store.save(save_dir)

        store2 = FAISSStore()
        store2.load(save_dir)
        assert len(store2) == 3

        query = np.random.randn(384).astype(np.float32)
        results = store2.search(query, top_k=2)
        assert len(results) == 2
