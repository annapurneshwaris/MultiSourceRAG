"""Chart generation for Gradio dashboard using plotly."""

from __future__ import annotations

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

SOURCE_COLORS = {
    "doc": "#2196F3",
    "bug": "#F44336",
    "work_item": "#4CAF50",
}


def source_distribution_chart(chunks: list[dict]) -> object | None:
    """Pie chart of source type distribution in results."""
    if not HAS_PLOTLY or not chunks:
        return None

    from collections import Counter
    counts = Counter(c["source_type"] for c in chunks)

    fig = go.Figure(data=[go.Pie(
        labels=list(counts.keys()),
        values=list(counts.values()),
        marker_colors=[SOURCE_COLORS.get(k, "#999") for k in counts.keys()],
    )])
    fig.update_layout(title="Source Distribution", height=300)
    return fig


def router_learning_curve(history_path: str = "models/adaptive_router/history.jsonl") -> object | None:
    """Line chart showing router alpha decay and source boosts over time."""
    if not HAS_PLOTLY:
        return None

    import json
    import os

    if not os.path.exists(history_path):
        return None

    entries = []
    with open(history_path, "r") as f:
        for line in f:
            entries.append(json.loads(line))

    if not entries:
        return None

    queries = list(range(len(entries)))
    alphas = [e.get("alpha", 1.0) for e in entries]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=queries, y=alphas, name="Alpha (exploration)"))

    for src in ["doc", "bug", "work_item"]:
        boosts = [e.get("boosts", {}).get(src, 0.5) for e in entries]
        fig.add_trace(go.Scatter(
            x=queries, y=boosts, name=f"{src} boost",
            line=dict(color=SOURCE_COLORS.get(src, "#999")),
        ))

    fig.update_layout(title="Router Learning Curve", xaxis_title="Queries", yaxis_title="Value", height=400)
    return fig


def compare_configs_chart(results: list[dict]) -> object | None:
    """Bar chart comparing timing across configs."""
    if not HAS_PLOTLY or not results:
        return None

    configs = [r.get("config", "?") for r in results]
    total_ms = [r.get("timing", {}).get("total_ms", 0) for r in results]
    retrieved = [r.get("retrieved_count", 0) for r in results]

    fig = go.Figure(data=[
        go.Bar(name="Total ms", x=configs, y=total_ms),
        go.Bar(name="Retrieved", x=configs, y=retrieved),
    ])
    fig.update_layout(title="Config Comparison", barmode="group", height=350)
    return fig
