"""Evaluation query bank for Flutter cross-project validation — 25 seed queries.

Categories (same distribution as VS Code):
- how_to (5): Configuration, setup, feature usage
- debugging (5): Error diagnosis, crash investigation
- error_diagnosis (5): Specific error messages, stack traces
- status_roadmap (5): Feature plans, milestones, release dates
- config (5): Settings, build config, CI/CD
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalQuery:
    """A single evaluation query."""
    query_id: str
    query_text: str
    category: str
    expected_sources: list[str]
    expected_area: str
    difficulty: str
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


SEED_QUERIES: list[dict] = [
    # =========================================================================
    # how_to (5 queries)
    # =========================================================================
    {
        "query_id": "flutter_ht_01",
        "query_text": "How do I create a custom widget in Flutter?",
        "category": "how_to",
        "expected_sources": ["doc"],
        "expected_area": "ui",
        "difficulty": "easy",
    },
    {
        "query_id": "flutter_ht_02",
        "query_text": "How to set up hot reload for Flutter development?",
        "category": "how_to",
        "expected_sources": ["doc"],
        "expected_area": "development",
        "difficulty": "easy",
    },
    {
        "query_id": "flutter_ht_03",
        "query_text": "How do I add platform-specific code for Android and iOS?",
        "category": "how_to",
        "expected_sources": ["doc"],
        "expected_area": "platform_views",
        "difficulty": "medium",
    },
    {
        "query_id": "flutter_ht_04",
        "query_text": "How to use the Navigator 2.0 API for routing?",
        "category": "how_to",
        "expected_sources": ["doc", "bug"],
        "expected_area": "navigation",
        "difficulty": "medium",
    },
    {
        "query_id": "flutter_ht_05",
        "query_text": "How do I implement a custom paint widget?",
        "category": "how_to",
        "expected_sources": ["doc"],
        "expected_area": "ui",
        "difficulty": "medium",
    },

    # =========================================================================
    # debugging (5 queries)
    # =========================================================================
    {
        "query_id": "flutter_db_01",
        "query_text": "My Flutter app crashes on startup with a RenderFlex overflow error",
        "category": "debugging",
        "expected_sources": ["bug", "doc"],
        "expected_area": "framework",
        "difficulty": "easy",
    },
    {
        "query_id": "flutter_db_02",
        "query_text": "Hot reload stopped working after updating to Flutter 3.x",
        "category": "debugging",
        "expected_sources": ["bug"],
        "expected_area": "tooling",
        "difficulty": "medium",
    },
    {
        "query_id": "flutter_db_03",
        "query_text": "The build fails with Gradle dependency resolution error on Android",
        "category": "debugging",
        "expected_sources": ["bug"],
        "expected_area": "android",
        "difficulty": "medium",
    },
    {
        "query_id": "flutter_db_04",
        "query_text": "Flutter web app shows blank screen in production build",
        "category": "debugging",
        "expected_sources": ["bug", "doc"],
        "expected_area": "web",
        "difficulty": "hard",
    },
    {
        "query_id": "flutter_db_05",
        "query_text": "setState is not updating the UI after async operation completes",
        "category": "debugging",
        "expected_sources": ["bug", "doc"],
        "expected_area": "framework",
        "difficulty": "easy",
    },

    # =========================================================================
    # error_diagnosis (5 queries)
    # =========================================================================
    {
        "query_id": "flutter_ed_01",
        "query_text": "Error: The method was called on null. Receiver: null",
        "category": "error_diagnosis",
        "expected_sources": ["bug"],
        "expected_area": "framework",
        "difficulty": "easy",
    },
    {
        "query_id": "flutter_ed_02",
        "query_text": "MissingPluginException when calling platform channel on iOS",
        "category": "error_diagnosis",
        "expected_sources": ["bug", "doc"],
        "expected_area": "ios",
        "difficulty": "medium",
    },
    {
        "query_id": "flutter_ed_03",
        "query_text": "Error running pod install for iOS build",
        "category": "error_diagnosis",
        "expected_sources": ["bug"],
        "expected_area": "tooling",
        "difficulty": "medium",
    },
    {
        "query_id": "flutter_ed_04",
        "query_text": "The getter length was called on null during ListView.builder",
        "category": "error_diagnosis",
        "expected_sources": ["bug", "doc"],
        "expected_area": "framework",
        "difficulty": "easy",
    },
    {
        "query_id": "flutter_ed_05",
        "query_text": "Unhandled Exception: PlatformException during camera initialization",
        "category": "error_diagnosis",
        "expected_sources": ["bug"],
        "expected_area": "plugins",
        "difficulty": "hard",
    },

    # =========================================================================
    # status_roadmap (5 queries)
    # =========================================================================
    {
        "query_id": "flutter_sr_01",
        "query_text": "What is the status of Impeller rendering engine on iOS?",
        "category": "status_roadmap",
        "expected_sources": ["work_item", "bug"],
        "expected_area": "engine",
        "difficulty": "medium",
    },
    {
        "query_id": "flutter_sr_02",
        "query_text": "When will Flutter support WebAssembly compilation?",
        "category": "status_roadmap",
        "expected_sources": ["work_item"],
        "expected_area": "web",
        "difficulty": "hard",
    },
    {
        "query_id": "flutter_sr_03",
        "query_text": "What features were shipped in Flutter 3.24?",
        "category": "status_roadmap",
        "expected_sources": ["work_item", "doc"],
        "expected_area": "reference",
        "difficulty": "medium",
    },
    {
        "query_id": "flutter_sr_04",
        "query_text": "Is there a roadmap for improving Flutter desktop support?",
        "category": "status_roadmap",
        "expected_sources": ["work_item"],
        "expected_area": "desktop",
        "difficulty": "hard",
    },
    {
        "query_id": "flutter_sr_05",
        "query_text": "What are the plans for Flutter package ecosystem improvements?",
        "category": "status_roadmap",
        "expected_sources": ["work_item"],
        "expected_area": "plugins",
        "difficulty": "hard",
    },

    # =========================================================================
    # config (5 queries)
    # =========================================================================
    {
        "query_id": "flutter_cf_01",
        "query_text": "How to configure Flutter build flavors for different environments?",
        "category": "config",
        "expected_sources": ["doc"],
        "expected_area": "tooling",
        "difficulty": "medium",
    },
    {
        "query_id": "flutter_cf_02",
        "query_text": "How do I set up custom font loading in Flutter?",
        "category": "config",
        "expected_sources": ["doc"],
        "expected_area": "ui",
        "difficulty": "easy",
    },
    {
        "query_id": "flutter_cf_03",
        "query_text": "How to configure ProGuard rules for Flutter Android release?",
        "category": "config",
        "expected_sources": ["doc", "bug"],
        "expected_area": "android",
        "difficulty": "medium",
    },
    {
        "query_id": "flutter_cf_04",
        "query_text": "How do I set up CI/CD for Flutter with GitHub Actions?",
        "category": "config",
        "expected_sources": ["doc"],
        "expected_area": "tooling",
        "difficulty": "medium",
    },
    {
        "query_id": "flutter_cf_05",
        "query_text": "How to configure Flutter deep linking on Android and iOS?",
        "category": "config",
        "expected_sources": ["doc", "bug"],
        "expected_area": "navigation",
        "difficulty": "hard",
    },
]


def _build_queries() -> list[EvalQuery]:
    queries = []
    for d in SEED_QUERIES:
        queries.append(EvalQuery(**d))
    return queries


_QUERIES: list[EvalQuery] | None = None


def load_queries() -> list[EvalQuery]:
    global _QUERIES
    if _QUERIES is None:
        _QUERIES = _build_queries()
    return _QUERIES


def print_stats():
    queries = load_queries()
    cats = Counter(q.category for q in queries)
    diffs = Counter(q.difficulty for q in queries)
    print(f"Total queries: {len(queries)}")
    print(f"Categories: {dict(cats)}")
    print(f"Difficulty: {dict(diffs)}")


if __name__ == "__main__":
    print_stats()
