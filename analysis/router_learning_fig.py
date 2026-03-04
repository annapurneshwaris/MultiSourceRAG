"""Figure 2: Router learning curve.

Shows alpha decay and per-source boost evolution over queries.
"""

from __future__ import annotations

import json
import os


def generate_figure2(
    history_path: str = "models/adaptive_router/history.jsonl",
    output_path: str = "data/evaluation/fig2_router_learning.json",
) -> dict:
    """Generate data for Figure 2."""
    if not os.path.exists(history_path):
        return {"error": "No router history found."}

    entries = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))

    data = {
        "queries": list(range(len(entries))),
        "alpha": [e.get("alpha", 1.0) for e in entries],
        "doc_boost": [e.get("boosts", {}).get("doc", 0.5) for e in entries],
        "bug_boost": [e.get("boosts", {}).get("bug", 0.5) for e in entries],
        "work_item_boost": [e.get("boosts", {}).get("work_item", 0.5) for e in entries],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data


if __name__ == "__main__":
    result = generate_figure2()
    print(f"Generated with {len(result.get('queries', []))} data points")
