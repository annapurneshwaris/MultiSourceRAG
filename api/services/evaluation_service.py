"""Evaluation service — handles annotation and judging."""

from __future__ import annotations

from evaluation.human_annotation import AnnotationStore, HumanAnnotation


class EvaluationService:
    """Service for managing human annotations."""

    def __init__(self):
        self._store = AnnotationStore()

    def add_annotation(
        self,
        query_id: str,
        config: str,
        annotator_id: str,
        rci: int,
        as_: int,
        vm: int,
        root_cause_category: str = "unknown",
        notes: str = "",
    ) -> dict:
        annotation = HumanAnnotation(
            query_id=query_id,
            config=config,
            annotator_id=annotator_id,
            rci=rci,
            as_=as_,
            vm=vm,
            root_cause_category=root_cause_category,
            notes=notes,
        )
        self._store.add(annotation)
        return annotation.to_dict()

    def get_progress(self, annotator_id: str) -> dict:
        return self._store.get_progress(annotator_id)

    def get_all_annotations(self) -> list[dict]:
        return [a.to_dict() for a in self._store.get_all()]
