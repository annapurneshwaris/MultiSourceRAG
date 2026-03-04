"""Documentation retriever — hybrid tree + vector search.

Step 1: Load doc tree, optionally narrow to candidate chunks via tree navigation.
Step 2: Vector search within narrowed set (or full doc set as fallback).
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from processing.schemas import Chunk, TreeNode
from retrieval.retrievers.base import SourceRetriever


class DocRetriever(SourceRetriever):
    """Hybrid tree-navigated + vector retrieval for documentation."""

    def __init__(
        self,
        vector_store,
        metadata_index=None,
        tree_path: str = "data/processed/doc_tree.json",
    ):
        self._store = vector_store
        self._meta_idx = metadata_index
        self._tree: list[TreeNode] = []
        self._node_map: dict[str, TreeNode] = {}

        # Load tree if available
        if os.path.exists(tree_path):
            with open(tree_path, "r", encoding="utf-8") as f:
                tree_data = json.load(f)
            self._tree = [TreeNode.from_dict(d) for d in tree_data]
            self._node_map = {n.node_id: n for n in self._tree}

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 20,
        metadata_hints: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Retrieve doc chunks with optional tree narrowing."""

        # Get doc-only filter indices
        filter_ids = None
        if self._meta_idx:
            constraints = {"source_type": "doc"}
            if metadata_hints and "feature_area" in metadata_hints:
                constraints["feature_area"] = metadata_hints["feature_area"]
            filter_ids = self._meta_idx.filter(constraints)

            if not filter_ids:
                # No match on area filter — fall back to all docs
                filter_ids = self._meta_idx.filter({"source_type": "doc"})

        # Vector search with filter
        results = self._store.search(
            query_embedding,
            top_k=top_k,
            filter_ids=filter_ids,
        )

        # Safety fallback: if filtered search returned nothing, retry with all docs
        if not results and filter_ids is not None:
            all_doc_ids = self._meta_idx.filter({"source_type": "doc"}) if self._meta_idx else None
            if all_doc_ids and all_doc_ids != filter_ids:
                results = self._store.search(query_embedding, top_k=top_k, filter_ids=all_doc_ids)

        return results

    def get_tree_summary(self, max_depth: int = 2) -> str:
        """Get tree structure summary for LLM navigation.

        Returns a text representation of the doc tree at specified depth.
        """
        if not self._tree:
            return "No documentation tree available."

        lines = []
        root_nodes = [n for n in self._tree if n.depth == 0]

        for root in root_nodes[:50]:  # Cap at 50 docs
            lines.append(f"- {root.title} ({len(root.chunk_ids)} chunks)")
            if max_depth >= 1:
                for child_id in root.children[:10]:
                    child = self._node_map.get(child_id)
                    if child:
                        lines.append(f"  - {child.title} ({len(child.chunk_ids)} chunks)")
                        if max_depth >= 2:
                            for gc_id in child.children[:5]:
                                gc = self._node_map.get(gc_id)
                                if gc:
                                    lines.append(f"    - {gc.title}")

        return "\n".join(lines)
