"""Build all vector and keyword indices.

Usage:
    python -m indexing.run_indexing
    python -m indexing.run_indexing --configs combined   # build only combined index
"""

from __future__ import annotations

import argparse
import json
import os
import time

from processing.schemas import Chunk
from indexing.metadata_index import MetadataIndex
from indexing.bm25_index import BM25Index

INDICES_DIR = os.path.join("data", "indices")
DATA_DIR = os.path.join("data", "processed")

# Index configurations: name → source_type filter (None = all)
INDEX_CONFIGS = {
    "combined": None,
    "docs_only": ["doc"],
    "bugs_only": ["bug"],
    "workitems_only": ["work_item"],
    "docs_bugs": ["doc", "bug"],
    "docs_workitems": ["doc", "work_item"],
    "bugs_workitems": ["bug", "work_item"],
}


def _load_chunks() -> list[Chunk]:
    path = os.path.join(DATA_DIR, "all_chunks.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Chunk.from_dict(d) for d in data]


def _filter_chunks(chunks: list[Chunk], source_types: list[str] | None) -> list[Chunk]:
    if source_types is None:
        return chunks
    return [c for c in chunks if c.source_type in source_types]


def _get_embedder():
    """Create embedding provider based on config."""
    import config as cfg

    if cfg.EMBEDDING_PROVIDER == "openai":
        from indexing.providers.openai_embed import OpenAIEmbeddingProvider
        return OpenAIEmbeddingProvider(
            api_key=cfg.OPENAI_API_KEY,
            batch_size=cfg.EMBEDDING_BATCH_SIZE,
        )
    else:
        from indexing.providers.sentence_transformer import SentenceTransformerProvider
        return SentenceTransformerProvider(
            model_name=cfg.EMBEDDING_MODEL_NAME,
            batch_size=cfg.EMBEDDING_BATCH_SIZE,
        )


def _build_vector_index(
    config_name: str,
    chunks: list[Chunk],
    embedder,
) -> None:
    """Build FAISS + metadata index for one config."""
    from indexing.stores.faiss_store import FAISSStore

    index_dir = os.path.join(INDICES_DIR, config_name)

    print(f"  Embedding {len(chunks)} chunks...")
    texts = [c.text_with_context for c in chunks]
    embeddings = embedder.embed(texts)

    print(f"  Building FAISS index (dim={embedder.dimension})...")
    store = FAISSStore(dimension=embedder.dimension)
    store.add(embeddings, chunks)
    store.save(index_dir)

    print(f"  Building metadata index...")
    meta_idx = MetadataIndex()
    meta_idx.build(chunks)
    meta_idx.save(os.path.join(index_dir, "metadata.pkl"))

    # Save stats
    stats = meta_idx.stats()
    stats["chunk_count"] = len(chunks)
    stats["embedding_dim"] = embedder.dimension
    with open(os.path.join(index_dir, "index_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Saved to {index_dir} ({len(chunks)} vectors)")


def _build_bm25_index(chunks: list[Chunk]) -> None:
    """Build BM25 keyword index over all chunks."""
    bm25_dir = os.path.join(INDICES_DIR, "bm25")

    print(f"  Building BM25 index over {len(chunks)} chunks...")
    bm25 = BM25Index()
    bm25.build(chunks)
    bm25.save(bm25_dir)
    print(f"  Saved to {bm25_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HeteroRAG indices")
    parser.add_argument(
        "--configs",
        type=str,
        default="all",
        help="Comma-separated config names, or 'all' for everything",
    )
    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Skip BM25 index building",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("HeteroRAG Indexing Pipeline")
    print("=" * 60)
    start = time.time()

    # Load all chunks
    print("\n[1/3] Loading chunks...")
    all_chunks = _load_chunks()
    print(f"  Loaded {len(all_chunks)} chunks")

    # Determine which configs to build
    if args.configs == "all":
        configs_to_build = list(INDEX_CONFIGS.keys())
    else:
        configs_to_build = [c.strip() for c in args.configs.split(",")]

    # Create embedder (shared across configs)
    print("\n[2/3] Initializing embedder...")
    embedder = _get_embedder()
    print(f"  Using {type(embedder).__name__} (dim={embedder.dimension})")

    # Pre-compute embeddings for all chunks once (reuse across configs)
    print("\n  Embedding ALL chunks (will be filtered per config)...")
    all_texts = [c.text_with_context for c in all_chunks]
    all_embeddings = embedder.embed(all_texts)
    print(f"  Done: {all_embeddings.shape}")

    # Build each config
    print("\n[3/3] Building indices...")
    for config_name in configs_to_build:
        if config_name not in INDEX_CONFIGS:
            print(f"  WARNING: Unknown config '{config_name}', skipping")
            continue

        source_filter = INDEX_CONFIGS[config_name]
        filtered_chunks = _filter_chunks(all_chunks, source_filter)

        if not filtered_chunks:
            print(f"  {config_name}: 0 chunks, skipping")
            continue

        print(f"\n  --- {config_name} ({len(filtered_chunks)} chunks) ---")

        # Get corresponding embeddings
        if source_filter is None:
            filtered_embeddings = all_embeddings
        else:
            indices = [i for i, c in enumerate(all_chunks) if c.source_type in source_filter]
            filtered_embeddings = all_embeddings[indices]

        # Build FAISS + metadata
        from indexing.stores.faiss_store import FAISSStore

        index_dir = os.path.join(INDICES_DIR, config_name)
        store = FAISSStore(dimension=embedder.dimension)
        store.add(filtered_embeddings.copy(), filtered_chunks)
        store.save(index_dir)

        meta_idx = MetadataIndex()
        meta_idx.build(filtered_chunks)
        meta_idx.save(os.path.join(index_dir, "metadata.pkl"))

        stats = meta_idx.stats()
        stats["chunk_count"] = len(filtered_chunks)
        stats["embedding_dim"] = embedder.dimension
        with open(os.path.join(index_dir, "index_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        print(f"    Saved {config_name}: {len(filtered_chunks)} vectors")

    # BM25 (over all chunks)
    if not args.skip_bm25:
        print(f"\n  --- BM25 (all {len(all_chunks)} chunks) ---")
        _build_bm25_index(all_chunks)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"  Built {len(configs_to_build)} vector indices + BM25")
    print(f"  Total vectors: {len(all_chunks)} × {embedder.dimension}d")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
