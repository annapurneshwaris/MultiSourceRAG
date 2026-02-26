"""Parse citations from generated answers."""

from __future__ import annotations

import re

# Match [DOC-xxx], [BUG-xxx], [PLAN-xxx]
_CITATION_RE = re.compile(r"\[(DOC|BUG|PLAN)-([^\]]+)\]")

# Map citation prefix back to source type
_PREFIX_TO_SOURCE = {
    "DOC": "doc",
    "BUG": "bug",
    "PLAN": "work_item",
}


def extract_citations(answer: str) -> dict[str, list[str]]:
    """Extract citations from a generated answer.

    Args:
        answer: Generated text containing citations like [DOC-chunk_id].

    Returns:
        Dict mapping source type to list of chunk IDs.
        E.g. {"doc": ["doc_abc_0"], "bug": ["bug_123_0"], "work_item": []}
    """
    citations: dict[str, list[str]] = {
        "doc": [],
        "bug": [],
        "work_item": [],
    }

    for match in _CITATION_RE.finditer(answer):
        prefix = match.group(1)
        chunk_id = match.group(2)
        source_type = _PREFIX_TO_SOURCE.get(prefix, prefix.lower())
        if source_type in citations:
            if chunk_id not in citations[source_type]:
                citations[source_type].append(chunk_id)

    return citations
