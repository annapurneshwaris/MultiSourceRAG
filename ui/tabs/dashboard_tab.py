"""Dashboard tab — learning curves, source distribution charts."""

from __future__ import annotations

import gradio as gr
import httpx

from ui.charts import router_learning_curve, source_distribution_chart

API_BASE = "http://localhost:8000"


def load_dashboard():
    """Load dashboard data."""
    try:
        response = httpx.get(f"{API_BASE}/stats", timeout=10.0)
        stats = response.json()
    except Exception:
        stats = {}

    processing = stats.get("processing", {})
    index = stats.get("index_combined", {})

    summary = f"""### Dataset Statistics
- **Total chunks**: {processing.get('total_chunks', 'N/A')}
- **Doc chunks**: {processing.get('doc_chunks', 'N/A')}
- **Bug chunks**: {processing.get('bug_chunks', 'N/A')}
- **Work item chunks**: {processing.get('workitem_chunks', 'N/A')}
- **Tree nodes**: {processing.get('tree_nodes', 'N/A')}
- **Index vectors**: {index.get('chunk_count', 'N/A')}
- **Embedding dim**: {index.get('embedding_dim', 'N/A')}
"""

    # Router learning curve
    learning_fig = router_learning_curve()

    return summary, learning_fig


def create_tab():
    with gr.Tab("Dashboard"):
        gr.Markdown("## System Dashboard")

        refresh_btn = gr.Button("Refresh", variant="secondary")

        with gr.Row():
            stats_output = gr.Markdown()
            learning_output = gr.Plot(label="Router Learning Curve")

        refresh_btn.click(
            fn=load_dashboard,
            inputs=[],
            outputs=[stats_output, learning_output],
        )
