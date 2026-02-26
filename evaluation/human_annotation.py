"""Human annotation data management.

Supports saving/loading annotations, tracking annotator progress,
and computing inter-annotator agreement.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class HumanAnnotation:
    """A single human evaluation annotation."""
    query_id: str
    config: str
    annotator_id: str
    rci: int            # 0-2
    as_: int            # 0-2
    vm: int             # 0-2
    root_cause_category: str
    notes: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "config": self.config,
            "annotator_id": self.annotator_id,
            "rci": self.rci,
            "as": self.as_,
            "vm": self.vm,
            "ra": (self.rci + self.as_ + self.vm) / 6.0,
            "root_cause_category": self.root_cause_category,
            "notes": self.notes,
            "timestamp": self.timestamp or datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> HumanAnnotation:
        return cls(
            query_id=d["query_id"],
            config=d["config"],
            annotator_id=d["annotator_id"],
            rci=d["rci"],
            as_=d["as"],
            vm=d["vm"],
            root_cause_category=d.get("root_cause_category", "unknown"),
            notes=d.get("notes", ""),
            timestamp=d.get("timestamp", ""),
        )


class AnnotationStore:
    """Manage human annotations."""

    def __init__(self, path: str = "data/evaluation/annotations.json"):
        self._path = path
        self._annotations: list[HumanAnnotation] = []
        if os.path.exists(path):
            self._load()

    def _load(self) -> None:
        with open(self._path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._annotations = [HumanAnnotation.from_dict(d) for d in data]

    def save(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        data = [a.to_dict() for a in self._annotations]
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add(self, annotation: HumanAnnotation) -> None:
        self._annotations.append(annotation)
        self.save()

    def get_progress(self, annotator_id: str) -> dict:
        """Get annotation progress for an annotator."""
        annotator_anns = [a for a in self._annotations if a.annotator_id == annotator_id]
        done_pairs = {(a.query_id, a.config) for a in annotator_anns}
        return {
            "completed": len(done_pairs),
            "annotations": annotator_anns,
        }

    def get_all(self) -> list[HumanAnnotation]:
        return self._annotations

    def get_by_query(self, query_id: str) -> list[HumanAnnotation]:
        return [a for a in self._annotations if a.query_id == query_id]
