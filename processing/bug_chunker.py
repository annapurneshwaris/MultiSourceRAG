"""Composite-text chunking for GitHub bug issues.

Builds a rich text representation per bug: title + body (truncated) + errors +
top team comment. Merges comments from raw comments file. Mostly 1 chunk per bug.
"""

from __future__ import annotations

import re
from typing import Optional

from processing.schemas import Chunk
from processing.feature_area_map import extract_feature_area

_CHARS_PER_TOKEN = 4
_BODY_MAX_CHARS = 400 * _CHARS_PER_TOKEN       # ~400 tokens
_COMMENT_MAX_CHARS = 200 * _CHARS_PER_TOKEN     # ~200 tokens
_CHUNK_MAX_CHARS = 800 * _CHARS_PER_TOKEN       # ~800 tokens (split threshold)

# Common OS patterns in bug bodies
_OS_PATTERNS = [
    (re.compile(r"windows\s*1[01]", re.IGNORECASE), "windows"),
    (re.compile(r"windows", re.IGNORECASE), "windows"),
    (re.compile(r"macos|mac\s*os|darwin|osx", re.IGNORECASE), "macos"),
    (re.compile(r"linux|ubuntu|debian|fedora|arch", re.IGNORECASE), "linux"),
]


def _detect_os(text: str) -> str:
    """Extract OS platform from bug body text."""
    for pattern, os_name in _OS_PATTERNS:
        if pattern.search(text):
            return os_name
    return "unknown"


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars at a word boundary."""
    if not text or len(text) <= max_chars:
        return text or ""
    # Find last space before limit
    cutoff = text[:max_chars].rfind(" ")
    if cutoff < max_chars // 2:
        cutoff = max_chars
    return text[:cutoff] + "..."


def _extract_errors(body: str, parsed: dict) -> str:
    """Extract error messages from parsed data or body text."""
    # Use parsed error_messages if available
    errors = parsed.get("error_messages", [])
    if errors:
        return "\n".join(str(e) for e in errors[:3])

    # Fallback: regex for common error patterns
    error_lines = []
    for line in body.split("\n"):
        line_strip = line.strip()
        if any(kw in line_strip.lower() for kw in ["error:", "exception:", "traceback", "stack trace", "enoent", "cannot"]):
            error_lines.append(line_strip)
            if len(error_lines) >= 3:
                break

    return "\n".join(error_lines)


def _get_best_team_comment(
    team_comments: list[dict],
    raw_comments: list[dict],
) -> str:
    """Get the most useful team comment for this bug."""
    # Prefer team_comments from processed data
    for tc in team_comments:
        body = tc.get("comment_body", "")
        if body and len(body) > 20:
            return _truncate(body, _COMMENT_MAX_CHARS)

    # Fallback to raw comments (MEMBER/COLLABORATOR)
    for rc in raw_comments:
        if rc.get("author_association") in ("MEMBER", "COLLABORATOR", "OWNER"):
            body = rc.get("comment_body", "")
            if body and len(body) > 20:
                return _truncate(body, _COMMENT_MAX_CHARS)

    return ""


def process_all(
    bugs: list[dict],
    comments: dict[str, list[dict]],
) -> list[Chunk]:
    """Process all bugs into chunks.

    Args:
        bugs: List of bug dicts from bugs_processed.json.
        comments: Dict mapping issue_number (str) -> list of comment dicts
                  from comments.json.

    Returns:
        List of Chunk objects.
    """
    all_chunks: list[Chunk] = []

    for bug in bugs:
        number = bug["number"]
        title = bug.get("title", "")
        body = bug.get("body", "") or ""
        state = bug.get("state", "unknown")
        labels = bug.get("labels", [])
        parsed = bug.get("parsed", {}) or {}
        team_comments_list = bug.get("team_comments", []) or []
        reactions = bug.get("reactions", {}) or {}
        created_at = bug.get("created_at", "")
        updated_at = bug.get("updated_at", created_at)
        html_url = bug.get("html_url", f"https://github.com/microsoft/vscode/issues/{number}")
        milestone = bug.get("milestone") or ""

        # Normalize feature area from labels
        feature_area = extract_feature_area(labels, source_type="bug")

        # Detect OS
        os_platform = _detect_os(body)
        parsed_os = parsed.get("os_version", "")
        if parsed_os and os_platform == "unknown":
            os_platform = _detect_os(parsed_os)

        # Extract version
        version = parsed.get("vscode_version", "")

        # Build composite text
        parts: list[str] = []
        parts.append(f"Bug #{number}: {title}")

        # Truncated body
        truncated_body = _truncate(body, _BODY_MAX_CHARS)
        if truncated_body:
            parts.append(truncated_body)

        # Error messages
        errors = _extract_errors(body, parsed)
        if errors:
            parts.append(f"Error messages:\n{errors}")

        # Steps to reproduce
        steps = parsed.get("steps_to_reproduce", "")
        if steps:
            parts.append(f"Steps: {_truncate(str(steps), 300 * _CHARS_PER_TOKEN)}")

        # Best team comment
        raw_comments = comments.get(str(number), [])
        team_comment = _get_best_team_comment(team_comments_list, raw_comments)
        if team_comment:
            parts.append(f"Team response: {team_comment}")

        composite_text = "\n\n".join(parts)

        # Verified status
        is_verified = "verified" in [l.lower() for l in labels]

        # Has team response
        has_team_response = bool(team_comment)

        # Total reactions
        total_reactions = sum(reactions.get(k, 0) for k in ["+1", "heart", "rocket"])

        # Build embedding prefix per v4 spec
        text_with_context = (
            f"[BUG] #{number} | Area: {feature_area} | Status: {state} | "
            f"Version: {version} | OS: {os_platform}\n{composite_text}"
        )

        # Split if composite is too large
        if len(composite_text) > _CHUNK_MAX_CHARS:
            # Split at paragraph boundaries
            halves = composite_text.split("\n\n")
            mid = len(halves) // 2
            part1 = "\n\n".join(halves[:mid])
            part2 = "\n\n".join(halves[mid:])

            for i, part_text in enumerate([part1, part2]):
                chunk = Chunk(
                    chunk_id=f"bug_{number}_{i}",
                    source_type="bug",
                    source_id=str(number),
                    source_url=html_url,
                    text=part_text,
                    text_with_context=(
                        f"[BUG] #{number} | Area: {feature_area} | Status: {state} | "
                        f"Version: {version} | OS: {os_platform}\n{part_text}"
                    ),
                    feature_area=feature_area,
                    created_at=created_at,
                    updated_at=updated_at,
                    metadata={
                        "state": state,
                        "os_platform": os_platform,
                        "version": version,
                        "verified": is_verified,
                        "has_team_response": has_team_response,
                        "total_reactions": total_reactions,
                        "milestone": milestone,
                        "labels": labels,
                    },
                )
                all_chunks.append(chunk)
        else:
            chunk = Chunk(
                chunk_id=f"bug_{number}_0",
                source_type="bug",
                source_id=str(number),
                source_url=html_url,
                text=composite_text,
                text_with_context=text_with_context,
                feature_area=feature_area,
                created_at=created_at,
                updated_at=updated_at,
                metadata={
                    "state": state,
                    "os_platform": os_platform,
                    "version": version,
                    "verified": is_verified,
                    "has_team_response": has_team_response,
                    "total_reactions": total_reactions,
                    "milestone": milestone,
                    "labels": labels,
                },
            )
            all_chunks.append(chunk)

    return all_chunks
