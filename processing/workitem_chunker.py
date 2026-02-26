"""Type-dependent chunking for VS Code work items.

Three sub-types with different strategies:
- iteration_plan (116): split by H2/H3 sections → 5-20 chunks each
- plan_item (613): single chunk = title + body + completion ratio
- feature_request (3000): single chunk = title + first 300 tok + reactions + milestone
"""

from __future__ import annotations

import re
import math

from processing.schemas import Chunk
from processing.feature_area_map import extract_feature_area

_CHARS_PER_TOKEN = 4
_CHUNK_MAX_CHARS = 500 * _CHARS_PER_TOKEN  # ~500 tokens
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_CHECKBOX_RE = re.compile(r"- \[([ xX])\]")


def _truncate(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * _CHARS_PER_TOKEN
    if not text or len(text) <= max_chars:
        return text or ""
    cutoff = text[:max_chars].rfind(" ")
    if cutoff < max_chars // 2:
        cutoff = max_chars
    return text[:cutoff] + "..."


def _parse_completion_ratio(body: str) -> tuple[int, int]:
    """Count checked/total checkboxes in markdown body."""
    matches = _CHECKBOX_RE.findall(body)
    total = len(matches)
    checked = sum(1 for m in matches if m.lower() == "x")
    return checked, total


def _total_reactions(reactions: dict) -> int:
    return sum(reactions.get(k, 0) for k in ["+1", "heart", "rocket", "hooray"])


def _process_iteration_plan(wi: dict) -> list[Chunk]:
    """Split iteration plan by H2/H3 sections."""
    number = wi["number"]
    title = wi.get("title", "")
    body = wi.get("body", "") or ""
    labels = wi.get("labels", [])
    milestone = wi.get("milestone") or ""
    created_at = wi.get("created_at", "")
    closed_at = wi.get("closed_at", created_at)
    html_url = wi.get("html_url", f"https://github.com/microsoft/vscode/issues/{number}")
    reactions = wi.get("reactions", {}) or {}

    feature_area = extract_feature_area(labels, source_type="bug")

    checked, total = _parse_completion_ratio(body)
    completion = f"{checked}/{total}" if total > 0 else "N/A"

    # Split by headings
    sections: list[tuple[str, str]] = []
    heading_positions = list(_HEADING_RE.finditer(body))

    if not heading_positions:
        # No headings — single chunk
        sections.append((title, body))
    else:
        # Preamble
        preamble = body[: heading_positions[0].start()].strip()
        if preamble:
            sections.append((f"{title} - Overview", preamble))

        for i, m in enumerate(heading_positions):
            h_title = m.group(2).strip()
            start = m.end()
            end = heading_positions[i + 1].start() if i + 1 < len(heading_positions) else len(body)
            content = body[start:end].strip()
            if content:
                sections.append((h_title, content))

    chunks: list[Chunk] = []
    for idx, (sec_title, sec_text) in enumerate(sections):
        # Further split if too large
        if len(sec_text) > _CHUNK_MAX_CHARS:
            paragraphs = re.split(r"\n\n+", sec_text)
            buf = ""
            sub_idx = 0
            for para in paragraphs:
                if len(buf) + len(para) + 2 > _CHUNK_MAX_CHARS and buf:
                    chunks.append(_make_workitem_chunk(
                        number, f"{idx}_{sub_idx}", buf,
                        f"{title} > {sec_title} (part {sub_idx + 1})",
                        feature_area, "iteration_plan", milestone,
                        completion, reactions, html_url, created_at,
                        closed_at, labels,
                    ))
                    sub_idx += 1
                    buf = para
                else:
                    buf = buf + "\n\n" + para if buf else para
            if buf.strip():
                chunks.append(_make_workitem_chunk(
                    number, f"{idx}_{sub_idx}", buf,
                    f"{title} > {sec_title}" + (f" (part {sub_idx + 1})" if sub_idx > 0 else ""),
                    feature_area, "iteration_plan", milestone,
                    completion, reactions, html_url, created_at,
                    closed_at, labels,
                ))
        else:
            chunks.append(_make_workitem_chunk(
                number, str(idx), sec_text,
                f"{title} > {sec_title}" if sec_title != title else title,
                feature_area, "iteration_plan", milestone,
                completion, reactions, html_url, created_at,
                closed_at, labels,
            ))

    return chunks


def _process_plan_item(wi: dict) -> list[Chunk]:
    """Single chunk: title + body + completion ratio."""
    number = wi["number"]
    title = wi.get("title", "")
    body = wi.get("body", "") or ""
    labels = wi.get("labels", [])
    milestone = wi.get("milestone") or ""
    created_at = wi.get("created_at", "")
    closed_at = wi.get("closed_at", created_at)
    html_url = wi.get("html_url", f"https://github.com/microsoft/vscode/issues/{number}")
    reactions = wi.get("reactions", {}) or {}

    feature_area = extract_feature_area(labels, source_type="bug")

    checked, total = _parse_completion_ratio(body)
    completion = f"{checked}/{total}" if total > 0 else "N/A"

    text = f"Plan Item #{number}: {title}\n\n{body}"
    if total > 0:
        text += f"\n\nCompletion: {completion} tasks done"

    return [_make_workitem_chunk(
        number, "0", text, title, feature_area, "plan_item",
        milestone, completion, reactions, html_url, created_at,
        closed_at, labels,
    )]


def _process_feature_request(wi: dict) -> list[Chunk]:
    """Single chunk: title + first 300 tok + reactions + milestone."""
    number = wi["number"]
    title = wi.get("title", "")
    body = wi.get("body", "") or ""
    labels = wi.get("labels", [])
    milestone = wi.get("milestone") or ""
    created_at = wi.get("created_at", "")
    closed_at = wi.get("closed_at", created_at)
    html_url = wi.get("html_url", f"https://github.com/microsoft/vscode/issues/{number}")
    reactions = wi.get("reactions", {}) or {}
    state = wi.get("state", "open")

    feature_area = extract_feature_area(labels, source_type="bug")

    total_rx = _total_reactions(reactions)

    # Build text: title + truncated body + reactions context
    parts = [f"Feature Request #{number}: {title}"]

    truncated_body = _truncate(body, 300)
    if truncated_body:
        parts.append(truncated_body)

    if total_rx > 0:
        parts.append(f"Community support: {total_rx} reactions ({reactions.get('+1', 0)} upvotes)")

    if milestone:
        parts.append(f"Milestone: {milestone}")

    if state == "closed":
        parts.append("Status: shipped/closed")

    text = "\n\n".join(parts)

    return [_make_workitem_chunk(
        number, "0", text, title, feature_area, "feature_request",
        milestone, "N/A", reactions, html_url, created_at,
        closed_at, labels,
    )]


def _make_workitem_chunk(
    number: int,
    idx: str,
    text: str,
    heading: str,
    feature_area: str,
    item_type: str,
    milestone: str,
    completion: str,
    reactions: dict,
    html_url: str,
    created_at: str,
    closed_at: str,
    labels: list[str],
) -> Chunk:
    """Create a work item Chunk with embedding prefix."""
    total_rx = _total_reactions(reactions)

    text_with_context = (
        f"[PLAN] #{number} | Area: {feature_area} | Milestone: {milestone} | "
        f"Type: {item_type} | Reactions: {total_rx}\n{text}"
    )

    return Chunk(
        chunk_id=f"work_item_{number}_{idx}",
        source_type="work_item",
        source_id=str(number),
        source_url=html_url,
        text=text,
        text_with_context=text_with_context,
        feature_area=feature_area,
        created_at=created_at,
        updated_at=closed_at or created_at,
        metadata={
            "item_type": item_type,
            "milestone": milestone,
            "completion": completion,
            "total_reactions": total_rx,
            "state": "closed" if closed_at else "open",
            "labels": labels,
        },
    )


def process_all(workitems: list[dict]) -> list[Chunk]:
    """Process all work items into chunks.

    Args:
        workitems: List of work item dicts from workitems_processed.json.

    Returns:
        List of Chunk objects.
    """
    all_chunks: list[Chunk] = []

    for wi in workitems:
        wtype = wi.get("workitem_type", "unknown")

        if wtype == "iteration_plan":
            all_chunks.extend(_process_iteration_plan(wi))
        elif wtype == "plan_item":
            all_chunks.extend(_process_plan_item(wi))
        else:
            # feature_request and any unknown types
            all_chunks.extend(_process_feature_request(wi))

    return all_chunks
