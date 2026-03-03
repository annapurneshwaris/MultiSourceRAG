"""7-step retrieval pipeline — THE central orchestration file.

Steps:
1. Route: predict source boosts
2. Retrieve: per-source retrieval with metadata filtering
3. Re-rank: diversity-constrained MMR
4. Generate: LLM answer with citations
5. Compute utility: online utility signals
6. Update router: LinUCB parameter update (adaptive only)
7. Persist: save router state + history
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import numpy as np

from processing.schemas import Chunk
from retrieval.metadata_hints import extract_hints

logger = logging.getLogger(__name__)
from retrieval.reranker import rerank

# Source type lists for config parsing
_SOURCE_MAP = {
    "D": ["doc"],
    "B": ["bug"],
    "W": ["work_item"],
}


def _parse_config(config: str) -> list[str]:
    """Parse config string like 'DBW' into source type list.

    Configs: D, B, W, DB, DW, BW, DBW, BM25, Naive,
             HeteroRAG, HeteroRAG-H, HeteroRAG-L
    """
    if config.upper() in ("BM25", "NAIVE", "HETERORAG", "HETERORAG-H", "HETERORAG-L"):
        return ["doc", "bug", "work_item"]

    sources = []
    for char in config.upper():
        if char in _SOURCE_MAP:
            sources.extend(_SOURCE_MAP[char])
    return sources or ["doc", "bug", "work_item"]


class RetrievalPipeline:
    """Main HeteroRAG retrieval pipeline."""

    def __init__(
        self,
        vector_store=None,
        metadata_index=None,
        bm25_index=None,
        embedder=None,
        generator=None,
        router_type: str = "adaptive",
        index_dir: str = "data/indices/combined",
    ):
        """Initialize pipeline components.

        Args:
            vector_store: Pre-loaded vector store, or None to load from index_dir.
            metadata_index: Pre-loaded metadata index, or None to load.
            bm25_index: Pre-loaded BM25 index, or None to load.
            embedder: Embedding provider, or None to create default.
            generator: Generation module, or None to create on demand.
            router_type: "adaptive", "heuristic", or "llm_zeroshot".
            index_dir: Directory containing FAISS + metadata indices.
        """
        import config as cfg

        # Embedding provider
        if embedder is None:
            from indexing.providers.sentence_transformer import SentenceTransformerProvider
            self._embedder = SentenceTransformerProvider(
                model_name=cfg.EMBEDDING_MODEL_NAME,
                batch_size=cfg.EMBEDDING_BATCH_SIZE,
            )
        else:
            self._embedder = embedder

        # Vector store
        if vector_store is None:
            from indexing.stores.faiss_store import FAISSStore
            self._store = FAISSStore()
            self._store.load(index_dir)
        else:
            self._store = vector_store

        # Metadata index
        if metadata_index is None:
            from indexing.metadata_index import MetadataIndex
            self._meta_idx = MetadataIndex()
            meta_path = os.path.join(index_dir, "metadata.pkl")
            if os.path.exists(meta_path):
                self._meta_idx.load(meta_path)
        else:
            self._meta_idx = metadata_index

        # BM25 index
        if bm25_index is None:
            from indexing.bm25_index import BM25Index
            bm25_dir = os.path.join("data", "indices", "bm25")
            if os.path.exists(bm25_dir):
                self._bm25 = BM25Index()
                self._bm25.load(bm25_dir)
            else:
                self._bm25 = None
        else:
            self._bm25 = bm25_index

        # Per-source retrievers
        from retrieval.retrievers.doc_retriever import DocRetriever
        from retrieval.retrievers.bug_retriever import BugRetriever
        from retrieval.retrievers.workitem_retriever import WorkItemRetriever

        self._retrievers = {
            "doc": DocRetriever(self._store, self._meta_idx),
            "bug": BugRetriever(self._store, self._meta_idx),
            "work_item": WorkItemRetriever(self._store, self._meta_idx),
        }

        if self._bm25:
            from retrieval.retrievers.bm25_retriever import BM25Retriever
            self._retrievers["bm25"] = BM25Retriever(self._bm25)

        # Router
        self._router_type = router_type
        if router_type == "adaptive":
            from retrieval.router.adaptive import AdaptiveRouter
            self._router = AdaptiveRouter(
                feature_dim=cfg.EMBEDDING_DIM,
                alpha_initial=cfg.ALPHA_INITIAL,
                alpha_min=cfg.ALPHA_MIN,
                alpha_decay=cfg.ALPHA_DECAY,
                cold_start_threshold=cfg.COLD_START_THRESHOLD,
            )
            # Load saved state if available
            router_state_dir = os.path.join("data", "models", "adaptive_router")
            if os.path.exists(os.path.join(router_state_dir, "router_state.json")):
                self._router.load_state(router_state_dir)
        elif router_type == "llm_zeroshot":
            from retrieval.router.llm_zeroshot import LLMZeroShotRouter
            self._router = LLMZeroShotRouter()
        else:
            from retrieval.router.heuristic import HeuristicRouter
            self._router = HeuristicRouter()

        # Generator (lazy loaded)
        self._generator = generator

        # Utility signal
        from retrieval.router.utility_signal import UtilitySignalCollector
        self._utility = UtilitySignalCollector()

    def process_query(
        self,
        query: str,
        config: str = "DBW",
        router_type: str | None = None,
        top_k: int = 10,
        generate: bool = True,
    ) -> dict:
        """Process a query through the full 7-step pipeline.

        Args:
            query: User query text.
            config: Source config string (D, B, W, DB, DW, BW, DBW, BM25, etc.)
            router_type: Override router type for this query.
            top_k: Number of final results.
            generate: Whether to generate LLM answer.

        Returns:
            Full result dict with keys: query, config, source_boosts,
            retrieved_chunks, reranked_chunks, answer, citations,
            timing, router_stats.
        """
        timing = {}
        t_start = time.time()

        import config as cfg

        # Parse config
        config_upper = config.upper()
        active_sources = _parse_config(config)
        is_bm25 = config_upper == "BM25"
        is_naive = config_upper == "NAIVE"

        # Step 0: Embed query
        t0 = time.time()
        query_embedding = self._embedder.embed_query(query)
        timing["embed_ms"] = round((time.time() - t0) * 1000, 1)

        # Step 0.5: Extract metadata hints
        hints = extract_hints(query)

        # Step 1: Route — predict source boosts
        t0 = time.time()
        if is_bm25 or is_naive:
            # No routing for baselines — equal weights
            source_boosts = {"doc": 0.5, "bug": 0.5, "work_item": 0.5}
        else:
            router = self._router
            if router_type and router_type != self._router_type:
                if router_type == "heuristic":
                    from retrieval.router.heuristic import HeuristicRouter
                    router = HeuristicRouter()
                elif router_type == "llm_zeroshot":
                    from retrieval.router.llm_zeroshot import LLMZeroShotRouter
                    router = LLMZeroShotRouter()
            source_boosts = router.predict(query, query_embedding)

        # Normalize boosts so they sum to 1 (consistent reranker weighting)
        boost_sum = sum(source_boosts.values())
        if boost_sum > 0:
            source_boosts = {k: v / boost_sum for k, v in source_boosts.items()}

        timing["route_ms"] = round((time.time() - t0) * 1000, 1)

        # Step 2: Retrieve — per-source retrieval
        t0 = time.time()
        all_candidates: list[tuple[Chunk, float]] = []

        if is_bm25 and "bm25" in self._retrievers:
            all_candidates = self._retrievers["bm25"].retrieve(
                query, query_embedding, top_k=top_k * 3
            )
        elif is_naive:
            # Naive: retrieve from all sources, no metadata hints
            per_source_k = cfg.TOP_K_PER_SOURCE
            for source in active_sources:
                if source in self._retrievers:
                    results = self._retrievers[source].retrieve(
                        query, query_embedding, top_k=per_source_k,
                    )
                    all_candidates.extend(results)
        else:
            per_source_k = cfg.TOP_K_PER_SOURCE
            for source in active_sources:
                if source in self._retrievers:
                    results = self._retrievers[source].retrieve(
                        query, query_embedding,
                        top_k=per_source_k,
                        metadata_hints=hints,
                    )
                    all_candidates.extend(results)

        timing["retrieve_ms"] = round((time.time() - t0) * 1000, 1)

        # Step 3: Re-rank — diversity-constrained MMR
        t0 = time.time()

        if is_naive:
            # Naive baseline: sort by relevance only, no diversity enforcement
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            reranked = all_candidates[:top_k]
        else:
            # Build chunk embeddings for cosine-based redundancy detection
            chunk_embeddings = {}
            if all_candidates:
                cand_texts = [c.text_with_context or c.text for c, _ in all_candidates]
                cand_embs = self._embedder.embed(cand_texts)
                for (chunk, _), emb in zip(all_candidates, cand_embs):
                    chunk_embeddings[chunk.chunk_id] = emb

            reranked = rerank(
                candidates=all_candidates,
                source_boosts=source_boosts,
                top_k=top_k,
                embeddings=chunk_embeddings,
                w_relevance=cfg.W_RELEVANCE,
                w_source_boost=cfg.W_SOURCE_BOOST,
                w_freshness=cfg.W_FRESHNESS,
                w_authority=cfg.W_AUTHORITY,
                w_redundancy=cfg.W_REDUNDANCY,
            )
        timing["rerank_ms"] = round((time.time() - t0) * 1000, 1)

        # Step 4: Generate — LLM answer with citations
        answer = ""
        citations: dict = {}
        if generate and reranked:
            t0 = time.time()
            try:
                if self._generator is None:
                    from generation.generator import Generator
                    self._generator = Generator()
                answer, citations = self._generator.generate(query, reranked)
            except Exception as e:
                answer = f"[Generation error: {e}]"
                citations = {}
            timing["generate_ms"] = round((time.time() - t0) * 1000, 1)

        # Step 5-7: Compute utility, update router, persist (adaptive only)
        if (
            self._router_type == "adaptive"
            and hasattr(self._router, "update")
            and citations
        ):
            # Compute per-source utility
            utilities = {}
            for source in active_sources:
                n_source = sum(1 for c, _ in reranked if c.source_type == source)
                cited_ids = set(citations.get(source, []))
                was_cited = len(cited_ids) > 0
                n_cited = len(cited_ids)

                # Best rank for this source
                best_rank = 0
                for rank, (chunk, _) in enumerate(reranked, 1):
                    if chunk.source_type == source:
                        best_rank = rank
                        break

                utilities[source] = self._utility.compute_online(
                    source_type=source,
                    was_cited=was_cited,
                    retrieval_rank=best_rank,
                    total_retrieved=len(reranked),
                    n_cited_chunks=n_cited,
                    n_source_chunks=n_source,
                )

            # Update router
            self._router.update(query_embedding, utilities)

            # Persist state
            router_state_dir = os.path.join("data", "models", "adaptive_router")
            self._router.save_state(router_state_dir)
            self._router.save_history(
                {"query": query, "config": config, "boosts": source_boosts, "utilities": utilities},
                os.path.join(router_state_dir, "history.jsonl"),
            )

        timing["total_ms"] = round((time.time() - t_start) * 1000, 1)

        # Build result
        return {
            "query": query,
            "config": config,
            "active_sources": active_sources,
            "source_boosts": source_boosts,
            "metadata_hints": hints,
            "retrieved_count": len(all_candidates),
            "reranked_chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "source_type": c.source_type,
                    "source_id": c.source_id,
                    "source_url": c.source_url,
                    "feature_area": c.feature_area,
                    "text": c.text[:2000],  # ~500 tokens; matches CHUNK_MAX_CHARS
                    "score": round(s, 4),
                }
                for c, s in reranked
            ],
            "answer": answer,
            "citations": citations,
            "timing": timing,
            "router_stats": self._router.stats if hasattr(self._router, "stats") else {},
        }
