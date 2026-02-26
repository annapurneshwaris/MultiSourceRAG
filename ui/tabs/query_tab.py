"""Query tab — main query interface with source routing visualization."""

from __future__ import annotations

import gradio as gr
import httpx

from ui.formatters import format_answer_with_citations, format_source_boosts, format_timing

API_BASE = "http://localhost:8000"


def query_fn(query: str, config: str, router_type: str, top_k: int, generate: bool):
    """Call the API and format results."""
    try:
        response = httpx.post(f"{API_BASE}/query", json={
            "query": query,
            "config": config,
            "router_type": router_type if router_type != "default" else None,
            "top_k": int(top_k),
            "generate": generate,
        }, timeout=60.0)
        result = response.json()
    except Exception as e:
        return f"Error: {e}", "", "", ""

    # Format answer with colored citations
    answer_html = format_answer_with_citations(
        result.get("answer", "No answer generated"),
        result.get("citations", {}),
    )

    # Source boosts visualization
    boosts_html = format_source_boosts(result.get("source_boosts", {}))

    # Timing
    timing_text = format_timing(result.get("timing", {}))

    # Citations summary
    citations = result.get("citations", {})
    cite_parts = []
    for src, ids in citations.items():
        if ids:
            cite_parts.append(f"{src}: {len(ids)} citations")
    citations_text = " | ".join(cite_parts) if cite_parts else "No citations"

    return answer_html, boosts_html, timing_text, citations_text


def create_tab():
    with gr.Tab("Query"):
        gr.Markdown("## HeteroRAG Query Interface")

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Question",
                    placeholder="How do I configure the integrated terminal?",
                    lines=2,
                )
            with gr.Column(scale=1):
                config_input = gr.Dropdown(
                    choices=["D", "B", "W", "DB", "DW", "BW", "DBW", "BM25"],
                    value="DBW",
                    label="Source Config",
                )
                router_input = gr.Dropdown(
                    choices=["default", "heuristic", "adaptive", "llm_zeroshot"],
                    value="default",
                    label="Router",
                )
                top_k_input = gr.Slider(1, 20, value=10, step=1, label="Top K")
                generate_input = gr.Checkbox(value=True, label="Generate Answer")

        submit_btn = gr.Button("Submit", variant="primary")

        with gr.Row():
            with gr.Column(scale=2):
                answer_output = gr.HTML(label="Answer")
            with gr.Column(scale=1):
                boosts_output = gr.HTML(label="Source Routing")
                timing_output = gr.Textbox(label="Timing", interactive=False)
                citations_output = gr.Textbox(label="Citations", interactive=False)

        submit_btn.click(
            fn=query_fn,
            inputs=[query_input, config_input, router_input, top_k_input, generate_input],
            outputs=[answer_output, boosts_output, timing_output, citations_output],
        )
