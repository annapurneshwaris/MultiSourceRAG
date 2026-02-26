"""Chunks tab — retrieved chunks inspector."""

from __future__ import annotations

import gradio as gr
import httpx

from ui.formatters import format_chunks_table

API_BASE = "http://localhost:8000"


def inspect_fn(query: str, config: str, top_k: int):
    """Retrieve chunks without generating answer."""
    try:
        response = httpx.post(f"{API_BASE}/query", json={
            "query": query,
            "config": config,
            "top_k": int(top_k),
            "generate": False,
        }, timeout=30.0)
        result = response.json()
    except Exception as e:
        return [], f"Error: {e}"

    chunks = result.get("reranked_chunks", [])
    table = format_chunks_table(chunks)

    info = (
        f"Retrieved: {result.get('retrieved_count', 0)} | "
        f"Reranked: {len(chunks)} | "
        f"Hints: {result.get('metadata_hints', {})}"
    )

    return table, info


def create_tab():
    with gr.Tab("Chunks"):
        gr.Markdown("## Retrieved Chunks Inspector")
        gr.Markdown("Inspect the raw chunks retrieved and re-ranked for a query (no LLM generation).")

        with gr.Row():
            query_input = gr.Textbox(label="Question", lines=2)
            config_input = gr.Dropdown(
                choices=["D", "B", "W", "DB", "DW", "BW", "DBW", "BM25"],
                value="DBW",
                label="Config",
            )
            top_k_input = gr.Slider(5, 30, value=10, step=1, label="Top K")

        submit_btn = gr.Button("Retrieve", variant="primary")

        info_output = gr.Textbox(label="Info", interactive=False)
        chunks_output = gr.Dataframe(
            headers=["#", "Source", "Area", "Text", "Score", "URL"],
            label="Retrieved Chunks",
        )

        submit_btn.click(
            fn=inspect_fn,
            inputs=[query_input, config_input, top_k_input],
            outputs=[chunks_output, info_output],
        )
