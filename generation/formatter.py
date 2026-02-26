"""Format retrieved chunks into prompt context."""

from __future__ import annotations

from processing.schemas import Chunk

# Source type to citation tag prefix
_TAG_MAP = {
    "doc": "DOC",
    "bug": "BUG",
    "work_item": "PLAN",
}


def _make_tag(chunk: Chunk) -> str:
    """Create citation tag like [DOC-doc_abc123_0]."""
    prefix = _TAG_MAP.get(chunk.source_type, "SRC")
    return f"[{prefix}-{chunk.chunk_id}]"


def format_chunks_for_prompt(
    chunks: list[tuple[Chunk, float]],
    max_context_chars: int = 12000,
) -> str:
    """Format ranked chunks into a prompt context block.

    Each chunk gets a citation tag, source type label, score, and truncated text.

    Args:
        chunks: List of (Chunk, score) tuples from re-ranker.
        max_context_chars: Maximum total context length.

    Returns:
        Formatted context string for the generation prompt.
    """
    parts: list[str] = []
    total_chars = 0

    for i, (chunk, score) in enumerate(chunks, 1):
        tag = _make_tag(chunk)
        source_label = chunk.source_type.upper().replace("_", " ")

        header = f"--- Chunk {i} {tag} [{source_label}] (score: {score:.3f}) ---"
        url_line = f"Source: {chunk.source_url}"
        area_line = f"Area: {chunk.feature_area}"

        # Truncate text if needed
        text = chunk.text
        remaining = max_context_chars - total_chars - len(header) - len(url_line) - len(area_line) - 20
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining] + "..."

        block = f"{header}\n{url_line}\n{area_line}\n{text}\n"
        parts.append(block)
        total_chars += len(block)

    return "\n".join(parts)
