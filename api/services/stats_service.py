"""Stats service — dataset and index statistics."""

from __future__ import annotations

import json
import os


class StatsService:
    """Service for providing system statistics."""

    def get_processing_stats(self) -> dict:
        path = os.path.join("data", "processed", "processing_stats.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_index_stats(self, config: str = "combined") -> dict:
        path = os.path.join("data", "indices", config, "index_stats.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_all_stats(self) -> dict:
        return {
            "processing": self.get_processing_stats(),
            "index_combined": self.get_index_stats("combined"),
        }
