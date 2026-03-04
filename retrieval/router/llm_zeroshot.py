"""LLM zero-shot router — upper-bound baseline.

Prompts an LLM to score each source 0-10, then normalizes to [0.2, 1.0].
Accurate but expensive (~500ms per query). Used as upper-bound baseline.
"""

from __future__ import annotations

import json
import re

import numpy as np

from retrieval.router.base import SourceRouter

_PROMPT = """You are a source routing expert for a VS Code technical support RAG system.

Given a user query, score how relevant each information source is on a scale of 0-10:
- "doc": Official VS Code documentation (how-to guides, configuration, features)
- "bug": GitHub bug reports (errors, crashes, workarounds, confirmed issues)
- "work_item": Roadmap items (feature requests, iteration plans, milestones)

Respond with ONLY a JSON object: {{"doc": N, "bug": N, "work_item": N}}

Query: {query}"""


class LLMZeroShotRouter(SourceRouter):
    """LLM-based zero-shot source routing."""

    def __init__(self, llm_provider=None):
        """
        Args:
            llm_provider: A generation LLM provider with generate() method.
                         If None, will be initialized on first use.
        """
        self._llm = llm_provider

    def _get_llm(self):
        if self._llm is None:
            from generation.providers.openai_gen import OpenAIProvider
            self._llm = OpenAIProvider()
        return self._llm

    def predict(
        self,
        query: str,
        query_embedding: np.ndarray | None = None,
    ) -> dict[str, float]:
        llm = self._get_llm()

        prompt = _PROMPT.format(query=query)
        response = llm.generate(prompt, max_tokens=100, temperature=0.0)

        # Parse JSON from response
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r"\{[^}]+\}", response)
            if json_match:
                scores = json.loads(json_match.group())
            else:
                scores = json.loads(response)
        except (json.JSONDecodeError, AttributeError):
            # Fallback: equal weights
            return {"doc": 0.5, "bug": 0.5, "work_item": 0.5}

        # Normalize 0-10 scores to [0.2, 1.0]
        result = {}
        for key in ["doc", "bug", "work_item"]:
            raw = float(scores.get(key, 5))
            normalized = 0.2 + (raw / 10.0) * 0.8  # Maps 0→0.2, 10→1.0
            result[key] = round(min(1.0, max(0.2, normalized)), 3)

        return result
