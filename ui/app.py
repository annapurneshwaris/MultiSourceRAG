"""Gradio 5-tab frontend for HeteroRAG.

Usage:
    python -m ui.app
    # Requires api.app running on port 8000
"""

from __future__ import annotations

import gradio as gr

from ui.tabs import query_tab, compare_tab, chunks_tab, eval_tab, dashboard_tab


def create_app() -> gr.Blocks:
    with gr.Blocks(
        title="HeteroRAG — Multi-Source RAG for VS Code",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# HeteroRAG\n"
            "**Multi-Source Heterogeneous RAG for VS Code Technical Support**\n\n"
            "Query across documentation, bug reports, and roadmap items with learned source routing."
        )

        query_tab.create_tab()
        compare_tab.create_tab()
        chunks_tab.create_tab()
        eval_tab.create_tab()
        dashboard_tab.create_tab()

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_port=7860, share=False)
