"""Hierarchical chunking for VS Code documentation.

Splits 402 markdown docs by heading structure (H2/H3), merges small sections,
splits oversized ones, and builds a PageIndex-style TreeNode hierarchy.
"""

from __future__ import annotations

import re
import hashlib
from typing import Optional

from processing.schemas import Chunk, TreeNode
from processing.feature_area_map import extract_feature_area

# Rough token estimate: 1 token ≈ 4 chars (English prose + code)
_CHARS_PER_TOKEN = 4

CHUNK_MIN_CHARS = 100 * _CHARS_PER_TOKEN   # 400 chars ≈ 100 tokens
CHUNK_MAX_CHARS = 500 * _CHARS_PER_TOKEN   # 2000 chars ≈ 500 tokens

# Regex for heading lines
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Regex for fenced code blocks (to avoid splitting inside them)
_CODE_FENCE_RE = re.compile(r"^```", re.MULTILINE)


def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def _make_doc_url(file_path: str) -> str:
    """Convert file_path like 'docs/terminal/basics.md' to a VS Code docs URL."""
    # Strip leading 'docs/' and trailing '.md'
    path = file_path
    if path.startswith("docs/"):
        path = path[5:]
    if path.endswith(".md"):
        path = path[:-3]
    return f"https://code.visualstudio.com/docs/{path}"


def _split_by_headings(markdown: str) -> list[dict]:
    """Split markdown into sections by headings.

    Returns list of {level, title, content, line_start}.
    The first section may have level=0 (preamble before any heading).
    """
    sections: list[dict] = []
    heading_positions = []

    for m in _HEADING_RE.finditer(markdown):
        level = len(m.group(1))
        title = m.group(2).strip()
        heading_positions.append((m.start(), m.end(), level, title))

    if not heading_positions:
        # No headings — whole doc is one section
        return [{"level": 0, "title": "Content", "content": markdown.strip()}]

    # Preamble before first heading
    preamble = markdown[: heading_positions[0][0]].strip()
    if preamble:
        sections.append({"level": 0, "title": "Introduction", "content": preamble})

    # Each heading to next heading
    for i, (start, end, level, title) in enumerate(heading_positions):
        if i + 1 < len(heading_positions):
            content = markdown[end: heading_positions[i + 1][0]].strip()
        else:
            content = markdown[end:].strip()
        sections.append({"level": level, "title": title, "content": content})

    return sections


def _build_heading_path(sections: list[dict], section_idx: int) -> str:
    """Build breadcrumb path like 'Features > Terminal > Configuration'."""
    current = sections[section_idx]
    path_parts = [current["title"]]
    target_level = current["level"]

    # Walk backwards to find parent headings
    for i in range(section_idx - 1, -1, -1):
        if sections[i]["level"] < target_level and sections[i]["level"] > 0:
            path_parts.insert(0, sections[i]["title"])
            target_level = sections[i]["level"]

    return " > ".join(path_parts)


def _split_large_section(text: str, max_chars: int) -> list[str]:
    """Split oversized text into chunks at paragraph boundaries."""
    paragraphs = re.split(r"\n\n+", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    # If any chunk is still too large (e.g., huge code block), hard-split
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) > max_chars * 1.5:
            # Hard split at max_chars boundaries, trying line breaks
            lines = chunk.split("\n")
            buf = ""
            for line in lines:
                if len(buf) + len(line) + 1 > max_chars and buf:
                    final.append(buf.strip())
                    buf = line
                else:
                    buf = buf + "\n" + line if buf else line
            if buf.strip():
                final.append(buf.strip())
        else:
            final.append(chunk)

    return final


def _make_chunk_id(file_path: str, idx: int) -> str:
    # Stable hash from file path
    path_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
    return f"doc_{path_hash}_{idx}"


def process_all(docs: list[dict]) -> tuple[list[Chunk], list[TreeNode]]:
    """Process all docs into chunks and build a tree hierarchy.

    Args:
        docs: List of doc dicts from docs_processed.json.

    Returns:
        (chunks, tree_nodes) tuple.
    """
    all_chunks: list[Chunk] = []
    all_nodes: list[TreeNode] = []

    for doc in docs:
        file_path = doc.get("file_path", "")
        area = doc.get("area", "other")
        raw_md = doc.get("raw_markdown", "")
        page_title = doc.get("page_title") or doc.get("toc_title") or file_path
        created_at = doc.get("date_approved", "")
        source_url = _make_doc_url(file_path)

        # Normalize area through our map
        feature_area = extract_feature_area([area], source_type="doc")

        # Split markdown by headings
        sections = _split_by_headings(raw_md)

        # Build chunks from sections
        doc_chunks: list[Chunk] = []
        chunk_idx = 0
        pending_text = ""
        pending_title = ""

        for sec_i, section in enumerate(sections):
            text = section["content"]
            title = section["title"]
            heading_path = _build_heading_path(sections, sec_i)

            if not text.strip():
                continue

            # Merge small consecutive sections
            if _estimate_tokens(text) < CHUNK_MIN_CHARS // _CHARS_PER_TOKEN and pending_text:
                pending_text += f"\n\n## {title}\n{text}"
                continue
            elif _estimate_tokens(text) < CHUNK_MIN_CHARS // _CHARS_PER_TOKEN:
                pending_text = text
                pending_title = heading_path
                continue

            # Flush pending small section
            if pending_text:
                combined = pending_text + f"\n\n## {title}\n{text}"
                if _estimate_tokens(combined) <= CHUNK_MAX_CHARS // _CHARS_PER_TOKEN:
                    text = combined
                    heading_path = pending_title + " + " + heading_path
                else:
                    # Emit pending as its own chunk
                    chunk = _create_chunk(
                        file_path, chunk_idx, pending_text, pending_title,
                        feature_area, page_title, source_url, created_at
                    )
                    doc_chunks.append(chunk)
                    chunk_idx += 1
                pending_text = ""
                pending_title = ""

            # Split large sections
            if _estimate_tokens(text) > CHUNK_MAX_CHARS // _CHARS_PER_TOKEN:
                parts = _split_large_section(text, CHUNK_MAX_CHARS)
                for part_i, part in enumerate(parts):
                    suffix = f" (part {part_i + 1})" if len(parts) > 1 else ""
                    chunk = _create_chunk(
                        file_path, chunk_idx, part, heading_path + suffix,
                        feature_area, page_title, source_url, created_at
                    )
                    doc_chunks.append(chunk)
                    chunk_idx += 1
            else:
                chunk = _create_chunk(
                    file_path, chunk_idx, text, heading_path,
                    feature_area, page_title, source_url, created_at
                )
                doc_chunks.append(chunk)
                chunk_idx += 1

        # Flush final pending
        if pending_text:
            chunk = _create_chunk(
                file_path, chunk_idx, pending_text, pending_title,
                feature_area, page_title, source_url, created_at
            )
            doc_chunks.append(chunk)
            chunk_idx += 1

        all_chunks.extend(doc_chunks)

        # Build tree for this doc
        tree_nodes = _build_tree(file_path, page_title, sections, doc_chunks)
        all_nodes.extend(tree_nodes)

    return all_chunks, all_nodes


def _create_chunk(
    file_path: str,
    idx: int,
    text: str,
    heading_path: str,
    feature_area: str,
    page_title: str,
    source_url: str,
    created_at: str,
) -> Chunk:
    """Create a single doc Chunk with embedding prefix."""
    chunk_id = _make_chunk_id(file_path, idx)

    # Embedding prefix per v4 spec
    text_with_context = (
        f"[DOC] Area: {feature_area} | Topic: {heading_path} | "
        f"Path: {file_path}\n{text}"
    )

    return Chunk(
        chunk_id=chunk_id,
        source_type="doc",
        source_id=file_path,
        source_url=source_url,
        text=text,
        text_with_context=text_with_context,
        feature_area=feature_area,
        created_at=created_at,
        updated_at=created_at,
        metadata={
            "page_title": page_title,
            "heading_path": heading_path,
            "file_path": file_path,
        },
    )


def _build_tree(
    file_path: str,
    page_title: str,
    sections: list[dict],
    chunks: list[Chunk],
) -> list[TreeNode]:
    """Build TreeNode hierarchy for one doc file."""
    nodes: list[TreeNode] = []

    # Root node for the entire doc
    root_id = file_path.replace("/", "_").replace(".", "_")
    root = TreeNode(
        node_id=root_id,
        title=page_title,
        summary=f"Documentation page: {page_title}",
        depth=0,
        chunk_ids=[c.chunk_id for c in chunks],
        children=[],
        parent="",
    )
    nodes.append(root)

    # Track parent stack: list of (level, node_id)
    parent_stack: list[tuple[int, str]] = [(0, root_id)]

    for section in sections:
        level = section["level"]
        title = section["title"]
        if level == 0:
            continue  # Preamble handled by root

        node_id = f"{root_id}_h{level}_{hashlib.md5(title.encode()).hexdigest()[:6]}"

        # Find chunks belonging to this section (by heading_path match)
        section_chunk_ids = [
            c.chunk_id for c in chunks
            if title in c.metadata.get("heading_path", "")
        ]

        node = TreeNode(
            node_id=node_id,
            title=title,
            summary=title,  # Placeholder — LLM summaries added later if needed
            depth=level,
            chunk_ids=section_chunk_ids,
            children=[],
            parent="",
        )

        # Find parent: walk up stack to find first node with lower depth
        while len(parent_stack) > 1 and parent_stack[-1][0] >= level:
            parent_stack.pop()

        parent_node_id = parent_stack[-1][1]
        node.parent = parent_node_id

        # Add as child of parent
        for n in nodes:
            if n.node_id == parent_node_id:
                n.children.append(node_id)
                break

        parent_stack.append((level, node_id))
        nodes.append(node)

    return nodes
