"""Universal data schemas for the HeteroRAG pipeline.

All shared data structures live here. This module has ZERO internal imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """Universal chunk schema — every chunk from every source conforms to this."""

    # Identity
    chunk_id: str               # "{source_type}_{source_id}_{chunk_idx}"
    source_type: str            # "doc" | "bug" | "work_item"
    source_id: str              # File path (docs) or issue number (bugs/work)
    source_url: str             # Full URL to original source

    # Content
    text: str                   # Raw chunk text, target 200-500 tokens
    text_with_context: str      # Prefixed for embedding (source-type metadata prepended)

    # Cross-source linking
    feature_area: str           # Normalized area: "terminal", "editor", "debug", etc.

    # Temporal
    created_at: str             # ISO 8601
    updated_at: str             # ISO 8601

    # Source-specific metadata (varies by source_type)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "text": self.text,
            "text_with_context": self.text_with_context,
            "feature_area": self.feature_area,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(**d)


@dataclass
class TreeNode:
    """For PageIndex-style hierarchical doc retrieval."""

    node_id: str                # e.g. "docs/terminal/configuration"
    title: str                  # e.g. "Configuration"
    summary: str                # LLM-generated 1-line summary (or heading text)
    depth: int                  # 0=root, 1=H1, 2=H2, 3=H3
    chunk_ids: list = field(default_factory=list)    # Chunks under this node
    children: list = field(default_factory=list)     # Child node IDs
    parent: str = ""            # Parent node ID

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "summary": self.summary,
            "depth": self.depth,
            "chunk_ids": self.chunk_ids,
            "children": self.children,
            "parent": self.parent,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TreeNode:
        return cls(**d)
