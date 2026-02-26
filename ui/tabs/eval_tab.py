"""Eval tab — human annotation interface."""

from __future__ import annotations

import gradio as gr
import httpx

API_BASE = "http://localhost:8000"


def submit_annotation(query_id, config, annotator_id, rci, as_, vm, category, notes):
    """Submit a human annotation."""
    try:
        response = httpx.post(f"{API_BASE}/annotate", json={
            "query_id": query_id,
            "config": config,
            "annotator_id": annotator_id,
            "rci": int(rci),
            "as": int(as_),
            "vm": int(vm),
            "root_cause_category": category,
            "notes": notes,
        }, timeout=10.0)
        result = response.json()
        ra = result.get("ra", 0)
        return f"Saved! RA = {ra:.3f}"
    except Exception as e:
        return f"Error: {e}"


def create_tab():
    with gr.Tab("Evaluate"):
        gr.Markdown("## Human Annotation Interface")
        gr.Markdown("Score answers on Resolution Adequacy dimensions (0-2 each).")

        with gr.Row():
            query_id_input = gr.Textbox(label="Query ID", placeholder="ht_001")
            config_input = gr.Textbox(label="Config", value="DBW")
            annotator_input = gr.Textbox(label="Annotator ID", placeholder="annotator_1")

        gr.Markdown("### Scores (0 = None, 1 = Partial, 2 = Full)")

        with gr.Row():
            rci_input = gr.Slider(0, 2, value=1, step=1, label="RCI (Root Cause)")
            as_input = gr.Slider(0, 2, value=1, step=1, label="AS (Actionable Steps)")
            vm_input = gr.Slider(0, 2, value=1, step=1, label="VM (Version Match)")

        category_input = gr.Dropdown(
            choices=[
                "configuration", "extension_conflict", "known_bug",
                "missing_feature", "platform_specific", "performance",
                "user_error", "documentation_gap", "unknown",
            ],
            value="unknown",
            label="Root Cause Category",
        )
        notes_input = gr.Textbox(label="Notes", lines=2)

        submit_btn = gr.Button("Submit Annotation", variant="primary")
        result_output = gr.Textbox(label="Result", interactive=False)

        submit_btn.click(
            fn=submit_annotation,
            inputs=[
                query_id_input, config_input, annotator_input,
                rci_input, as_input, vm_input, category_input, notes_input,
            ],
            outputs=[result_output],
        )
