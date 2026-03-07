"""Orchestrator: run all three chunkers on Flutter data.

Usage:
    python -m processing.run_processing_flutter
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter

# Monkey-patch: swap the feature_area_map module BEFORE importing chunkers
import processing.feature_area_map_flutter as flutter_fam
import processing.feature_area_map as fam_module
fam_module.FEATURE_AREA_MAP = flutter_fam.FEATURE_AREA_MAP
fam_module.DOC_AREA_MAP = flutter_fam.DOC_AREA_MAP
fam_module.extract_feature_area = flutter_fam.extract_feature_area

from processing.doc_chunker import process_all as process_docs
from processing.bug_chunker import process_all as process_bugs
from processing.workitem_chunker import process_all as process_workitems


DATA_DIR = os.path.join("data", "flutter", "processed")
RAW_DIR = os.path.join("data", "flutter", "raw")


def _load_json(path: str) -> list | dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(data: list | dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {path} ({len(data) if isinstance(data, list) else 'dict'} items)")


def main() -> None:
    print("=" * 60)
    print("HeteroRAG Processing Pipeline (Flutter)")
    print("=" * 60)
    start = time.time()

    # --- Load raw data ---
    print("\n[1/5] Loading data...")
    docs = _load_json(os.path.join(RAW_DIR, "docs.json"))
    bugs = _load_json(os.path.join(RAW_DIR, "bugs.json"))
    workitems = _load_json(os.path.join(RAW_DIR, "workitems.json"))

    comments_path = os.path.join(RAW_DIR, "comments.json")
    if os.path.exists(comments_path):
        comments = _load_json(comments_path)
    else:
        print("  WARNING: comments.json not found, using empty dict")
        comments = {}

    print(f"  Loaded: {len(docs)} docs, {len(bugs)} bugs, "
          f"{len(workitems)} work items, {len(comments)} comment threads")

    # --- Process docs ---
    print("\n[2/5] Processing docs (hierarchical chunking + tree)...")
    t0 = time.time()
    doc_chunks, tree_nodes = process_docs(docs)
    print(f"  {len(doc_chunks)} chunks, {len(tree_nodes)} tree nodes "
          f"({time.time() - t0:.1f}s)")

    # --- Process bugs ---
    print("\n[3/5] Processing bugs (composite text chunking)...")
    t0 = time.time()
    bug_chunks = process_bugs(bugs, comments)
    print(f"  {len(bug_chunks)} chunks ({time.time() - t0:.1f}s)")

    # --- Process work items ---
    print("\n[4/5] Processing work items (type-dependent chunking)...")
    t0 = time.time()
    workitem_chunks = process_workitems(workitems)
    print(f"  {len(workitem_chunks)} chunks ({time.time() - t0:.1f}s)")

    # --- Save outputs ---
    print("\n[5/5] Saving outputs...")

    doc_chunk_dicts = [c.to_dict() for c in doc_chunks]
    bug_chunk_dicts = [c.to_dict() for c in bug_chunks]
    wi_chunk_dicts = [c.to_dict() for c in workitem_chunks]
    all_chunk_dicts = doc_chunk_dicts + bug_chunk_dicts + wi_chunk_dicts
    tree_node_dicts = [n.to_dict() for n in tree_nodes]

    _save_json(doc_chunk_dicts, os.path.join(DATA_DIR, "doc_chunks.json"))
    _save_json(bug_chunk_dicts, os.path.join(DATA_DIR, "bug_chunks.json"))
    _save_json(wi_chunk_dicts, os.path.join(DATA_DIR, "workitem_chunks.json"))
    _save_json(all_chunk_dicts, os.path.join(DATA_DIR, "all_chunks.json"))
    _save_json(tree_node_dicts, os.path.join(DATA_DIR, "doc_tree.json"))

    # --- Stats ---
    doc_areas = Counter(c.feature_area for c in doc_chunks)
    bug_areas = Counter(c.feature_area for c in bug_chunks)
    wi_areas = Counter(c.feature_area for c in workitem_chunks)
    wi_types = Counter(c.metadata.get("item_type", "unknown") for c in workitem_chunks)

    doc_tokens = [len(c.text) // 4 for c in doc_chunks]
    bug_tokens = [len(c.text) // 4 for c in bug_chunks]
    wi_tokens = [len(c.text) // 4 for c in workitem_chunks]

    stats = {
        "total_chunks": len(all_chunk_dicts),
        "doc_chunks": len(doc_chunks),
        "bug_chunks": len(bug_chunks),
        "workitem_chunks": len(workitem_chunks),
        "tree_nodes": len(tree_nodes),
        "doc_areas": dict(doc_areas.most_common()),
        "bug_areas": dict(bug_areas.most_common()),
        "workitem_areas": dict(wi_areas.most_common()),
        "workitem_types": dict(wi_types),
        "token_stats": {
            "doc_median": sorted(doc_tokens)[len(doc_tokens) // 2] if doc_tokens else 0,
            "bug_median": sorted(bug_tokens)[len(bug_tokens) // 2] if bug_tokens else 0,
            "wi_median": sorted(wi_tokens)[len(wi_tokens) // 2] if wi_tokens else 0,
            "doc_min": min(doc_tokens, default=0),
            "doc_max": max(doc_tokens, default=0),
            "bug_min": min(bug_tokens, default=0),
            "bug_max": max(bug_tokens, default=0),
            "wi_min": min(wi_tokens, default=0),
            "wi_max": max(wi_tokens, default=0),
        },
        "feature_area_coverage": {
            "doc_unknown": sum(1 for c in doc_chunks if c.feature_area == "unknown"),
            "bug_unknown": sum(1 for c in bug_chunks if c.feature_area == "unknown"),
            "wi_unknown": sum(1 for c in workitem_chunks if c.feature_area == "unknown"),
        },
        "processing_time_seconds": round(time.time() - start, 1),
    }

    _save_json(stats, os.path.join(DATA_DIR, "processing_stats.json"))

    # --- Summary ---
    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"    Docs:       {stats['doc_chunks']} chunks, "
          f"{stats['tree_nodes']} tree nodes")
    print(f"    Bugs:       {stats['bug_chunks']} chunks")
    print(f"    Work Items: {stats['workitem_chunks']} chunks "
          f"({dict(wi_types)})")
    print(f"  Feature area unknowns: "
          f"doc={stats['feature_area_coverage']['doc_unknown']}, "
          f"bug={stats['feature_area_coverage']['bug_unknown']}, "
          f"wi={stats['feature_area_coverage']['wi_unknown']}")
    print(f"  Token medians: doc={stats['token_stats']['doc_median']}, "
          f"bug={stats['token_stats']['bug_median']}, "
          f"wi={stats['token_stats']['wi_median']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
