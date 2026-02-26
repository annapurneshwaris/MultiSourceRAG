"""Compare tab — side-by-side config comparison (high paper value)."""

from __future__ import annotations

import gradio as gr
import httpx

from ui.formatters import format_answer_with_citations, format_source_boosts

API_BASE = "http://localhost:8000"


def compare_fn(query: str, configs_text: str):
    """Compare query across multiple configs."""
    configs = [c.strip() for c in configs_text.split(",") if c.strip()]
    if not configs:
        configs = ["D", "DB", "DBW"]

    try:
        response = httpx.post(f"{API_BASE}/query/compare", json={
            "query": query,
            "configs": configs,
        }, timeout=120.0)
        data = response.json()
        results = data.get("results", [])
    except Exception as e:
        return f"Error: {e}", ""

    # Build comparison HTML
    rows = []
    for result in results:
        config = result.get("config", "?")
        answer = format_answer_with_citations(
            result.get("answer", "N/A")[:500],
            result.get("citations", {}),
        )
        boosts = format_source_boosts(result.get("source_boosts", {}))
        timing = result.get("timing", {}).get("total_ms", 0)
        n_chunks = result.get("retrieved_count", 0)

        citations = result.get("citations", {})
        n_citations = sum(len(v) for v in citations.values())

        rows.append(f"""
        <div style="border: 1px solid #ddd; padding: 12px; margin: 8px; border-radius: 8px;">
            <h3>Config: {config}</h3>
            <div style="display: flex; gap: 16px; margin-bottom: 8px;">
                <span>Time: {timing:.0f}ms</span>
                <span>Retrieved: {n_chunks}</span>
                <span>Citations: {n_citations}</span>
            </div>
            <div style="margin-bottom: 8px;">{boosts}</div>
            <div>{answer}</div>
        </div>
        """)

    comparison_html = "".join(rows)
    return comparison_html


def create_tab():
    with gr.Tab("Compare"):
        gr.Markdown("## Side-by-Side Config Comparison")
        gr.Markdown("Compare how different source configurations answer the same query.")

        with gr.Row():
            query_input = gr.Textbox(
                label="Question",
                placeholder="How do I configure the integrated terminal?",
                lines=2,
            )
            configs_input = gr.Textbox(
                label="Configs (comma-separated)",
                value="D, DB, DBW, BM25",
            )

        submit_btn = gr.Button("Compare", variant="primary")

        comparison_output = gr.HTML(label="Comparison Results")

        submit_btn.click(
            fn=compare_fn,
            inputs=[query_input, configs_input],
            outputs=[comparison_output],
        )
