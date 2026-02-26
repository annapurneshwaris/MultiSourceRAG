"""Format pipeline results for Gradio display."""

from __future__ import annotations

# Color-coded citation styling
SOURCE_COLORS = {
    "doc": "#2196F3",      # Blue
    "bug": "#F44336",      # Red
    "work_item": "#4CAF50", # Green
}

SOURCE_LABELS = {
    "doc": "DOC",
    "bug": "BUG",
    "work_item": "PLAN",
}


def format_answer_with_citations(answer: str, citations: dict) -> str:
    """Color-code citations in the answer text."""
    result = answer

    for source_type, color in SOURCE_COLORS.items():
        label = SOURCE_LABELS[source_type]
        # Replace [DOC-xxx] with colored version
        import re
        pattern = rf"\[{label}-([^\]]+)\]"
        replacement = f'<span style="background-color: {color}22; color: {color}; font-weight: bold; padding: 1px 4px; border-radius: 3px;">[{label}-\\1]</span>'
        result = re.sub(pattern, replacement, result)

    return result


def format_source_boosts(boosts: dict) -> str:
    """Format source boosts as a visual bar chart."""
    lines = []
    for source, weight in sorted(boosts.items(), key=lambda x: x[1], reverse=True):
        color = SOURCE_COLORS.get(source, "#999")
        bar_width = int(weight * 200)
        label = SOURCE_LABELS.get(source, source.upper())
        lines.append(
            f'<div style="margin: 4px 0;">'
            f'<span style="display: inline-block; width: 60px; font-weight: bold; color: {color};">{label}</span>'
            f'<span style="display: inline-block; width: {bar_width}px; height: 20px; '
            f'background-color: {color}; border-radius: 3px; margin-right: 8px;"></span>'
            f'<span>{weight:.3f}</span></div>'
        )
    return "".join(lines)


def format_chunks_table(chunks: list[dict]) -> list[list[str]]:
    """Format chunks for Gradio Dataframe display."""
    rows = []
    for i, c in enumerate(chunks, 1):
        rows.append([
            str(i),
            SOURCE_LABELS.get(c["source_type"], c["source_type"]),
            c["feature_area"],
            c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
            f"{c['score']:.4f}",
            c["source_url"],
        ])
    return rows


def format_timing(timing: dict) -> str:
    """Format timing dict as a readable summary."""
    parts = []
    for key, ms in sorted(timing.items()):
        parts.append(f"{key}: {ms:.0f}ms")
    return " | ".join(parts)
