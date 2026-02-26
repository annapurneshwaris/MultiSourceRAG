"""Query service — wraps the retrieval pipeline for API use."""

from __future__ import annotations

from retrieval.pipeline import RetrievalPipeline


class QueryService:
    """Singleton-like service wrapping the retrieval pipeline."""

    def __init__(self):
        self._pipeline: RetrievalPipeline | None = None

    def initialize(self, **kwargs) -> None:
        """Initialize the pipeline (called on app startup)."""
        self._pipeline = RetrievalPipeline(**kwargs)

    @property
    def pipeline(self) -> RetrievalPipeline:
        if self._pipeline is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self._pipeline

    def query(self, query: str, config: str = "DBW", **kwargs) -> dict:
        """Run a query through the pipeline."""
        return self.pipeline.process_query(query=query, config=config, **kwargs)

    def compare(self, query: str, configs: list[str], **kwargs) -> list[dict]:
        """Run the same query across multiple configs for comparison."""
        results = []
        for config in configs:
            result = self.pipeline.process_query(query=query, config=config, **kwargs)
            results.append(result)
        return results
