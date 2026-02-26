"""Evaluation query bank — 250 curated queries across 5 categories.

Categories (per v4 spec):
- how_to (60): Configuration, setup, feature usage
- debugging (75): Error diagnosis, crash investigation
- error_diagnosis (50): Specific error messages, stack traces
- status_roadmap (40): Feature plans, milestones, release dates
- config (25): Settings, keybindings, workspace config
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalQuery:
    """A single evaluation query."""
    query_id: str
    query_text: str
    category: str  # how_to, debugging, error_diagnosis, status_roadmap, config
    expected_sources: list[str]  # ["doc", "bug", "work_item"]
    expected_area: str  # Normalized feature area
    difficulty: str  # easy, medium, hard
    ground_truth_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "category": self.category,
            "expected_sources": self.expected_sources,
            "expected_area": self.expected_area,
            "difficulty": self.difficulty,
            "ground_truth_notes": self.ground_truth_notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EvalQuery:
        return cls(**d)


# Seed queries for each category (will be expanded to 250)
SEED_QUERIES: list[dict] = [
    # --- how_to (24% = 60 queries) ---
    {"query_id": "ht_001", "query_text": "How do I configure the integrated terminal to use zsh on macOS?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "terminal", "difficulty": "easy"},
    {"query_id": "ht_002", "query_text": "How to set up remote SSH development in VS Code?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "remote", "difficulty": "easy"},
    {"query_id": "ht_003", "query_text": "How do I enable bracket pair colorization?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "ht_004", "query_text": "How to configure GitHub Copilot inline suggestions?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "copilot", "difficulty": "medium"},
    {"query_id": "ht_005", "query_text": "How do I use multi-cursor editing in VS Code?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "ht_006", "query_text": "How to set up a dev container for Python?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "remote", "difficulty": "medium"},
    {"query_id": "ht_007", "query_text": "How do I customize the sidebar and panel layout?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "workbench", "difficulty": "easy"},
    {"query_id": "ht_008", "query_text": "How to configure workspace-specific settings?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "ht_009", "query_text": "How do I set up debugging for a Node.js application?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "debug", "difficulty": "medium"},
    {"query_id": "ht_010", "query_text": "How to use Jupyter notebooks in VS Code?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "notebook", "difficulty": "medium"},

    # --- debugging (30% = 75 queries) ---
    {"query_id": "db_001", "query_text": "VS Code terminal is not rendering colors correctly on Windows", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "terminal", "difficulty": "medium"},
    {"query_id": "db_002", "query_text": "Git extension not detecting changes in workspace", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "git", "difficulty": "medium"},
    {"query_id": "db_003", "query_text": "VS Code freezes when opening large files", "category": "debugging", "expected_sources": ["bug"], "expected_area": "performance", "difficulty": "hard"},
    {"query_id": "db_004", "query_text": "Copilot suggestions appearing very slowly", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "copilot", "difficulty": "medium"},
    {"query_id": "db_005", "query_text": "Remote SSH connection keeps dropping", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "remote", "difficulty": "hard"},
    {"query_id": "db_006", "query_text": "Extension host terminated unexpectedly", "category": "debugging", "expected_sources": ["bug"], "expected_area": "extensions", "difficulty": "hard"},
    {"query_id": "db_007", "query_text": "IntelliSense not working for TypeScript in monorepo", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "languages", "difficulty": "hard"},
    {"query_id": "db_008", "query_text": "Search not finding results in workspace", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "search", "difficulty": "medium"},
    {"query_id": "db_009", "query_text": "Sticky scroll causing rendering glitches", "category": "debugging", "expected_sources": ["bug"], "expected_area": "editor", "difficulty": "medium"},
    {"query_id": "db_010", "query_text": "Debug console not showing output from Python scripts", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "debug", "difficulty": "medium"},

    # --- error_diagnosis (20% = 50 queries) ---
    {"query_id": "ed_001", "query_text": "Error: EACCES permission denied when installing extensions", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "extensions", "difficulty": "medium"},
    {"query_id": "ed_002", "query_text": "TypeError: Cannot read properties of undefined in output panel", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "workbench", "difficulty": "hard"},
    {"query_id": "ed_003", "query_text": "Git error: unsafe repository is owned by someone else", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "git", "difficulty": "medium"},
    {"query_id": "ed_004", "query_text": "EPERM operation not permitted when saving files on Windows", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "explorer", "difficulty": "medium"},
    {"query_id": "ed_005", "query_text": "SSH connection error: Permission denied (publickey)", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "remote", "difficulty": "medium"},

    # --- status_roadmap (16% = 40 queries) ---
    {"query_id": "sr_001", "query_text": "What is planned for VS Code Copilot in the next release?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "copilot", "difficulty": "medium"},
    {"query_id": "sr_002", "query_text": "When will native bracket pair colorization support custom colors?", "category": "status_roadmap", "expected_sources": ["work_item", "bug"], "expected_area": "editor", "difficulty": "medium"},
    {"query_id": "sr_003", "query_text": "Is there a plan to improve terminal performance?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "terminal", "difficulty": "medium"},
    {"query_id": "sr_004", "query_text": "What features were shipped in the January 2024 iteration?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "workbench", "difficulty": "easy"},
    {"query_id": "sr_005", "query_text": "Has the multi-root workspace support been improved recently?", "category": "status_roadmap", "expected_sources": ["work_item", "doc"], "expected_area": "workbench", "difficulty": "medium"},

    # --- config (10% = 25 queries) ---
    {"query_id": "cf_001", "query_text": "How to change the default font size in the editor?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "cf_002", "query_text": "What keybinding opens the command palette?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "cf_003", "query_text": "How to configure auto-save behavior?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "cf_004", "query_text": "How to set different settings for different file types?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "medium"},
    {"query_id": "cf_005", "query_text": "How to sync VS Code settings across machines?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
]


def load_queries(path: str = "data/evaluation/query_bank.json") -> list[EvalQuery]:
    """Load evaluation queries from file."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [EvalQuery.from_dict(d) for d in data]
    # Return seed queries if no saved bank exists
    return [EvalQuery.from_dict(q) for q in SEED_QUERIES]


def save_queries(queries: list[EvalQuery], path: str = "data/evaluation/query_bank.json") -> None:
    """Save evaluation queries to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = [q.to_dict() for q in queries]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def stratify_by_category(queries: list[EvalQuery]) -> dict[str, list[EvalQuery]]:
    """Group queries by category."""
    groups: dict[str, list[EvalQuery]] = {}
    for q in queries:
        groups.setdefault(q.category, []).append(q)
    return groups


def get_category_distribution(queries: list[EvalQuery]) -> dict[str, int]:
    """Return count per category."""
    from collections import Counter
    return dict(Counter(q.category for q in queries))
