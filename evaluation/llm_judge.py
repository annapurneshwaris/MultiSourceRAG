"""LLM-based automated evaluation judge.

Uses GPT-4o or Claude to score answers on RCI, AS, VM dimensions.
Supports cross-validation (GPT-4o + Claude) to avoid self-preference bias.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from generation.prompt_templates import LLM_JUDGE_PROMPT
from generation.formatter import format_chunks_for_prompt
from evaluation.metrics import RAScore


@dataclass
class JudgeResult:
    """Result from an LLM judge evaluation."""
    query_id: str
    config: str
    judge_model: str
    rci: int
    as_: int
    vm: int
    root_cause_category: str
    reasoning: str

    @property
    def ra(self) -> float:
        return (self.rci + self.as_ + self.vm) / 6.0

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "config": self.config,
            "judge_model": self.judge_model,
            "rci": self.rci,
            "as": self.as_,
            "vm": self.vm,
            "ra": self.ra,
            "root_cause_category": self.root_cause_category,
            "reasoning": self.reasoning,
        }


class LLMJudge:
    """Automated evaluation using LLM judges."""

    def __init__(self, llm_provider=None, judge_model: str = "gpt-4o"):
        """
        Args:
            llm_provider: LLMProvider instance.
            judge_model: Model name for logging.
        """
        if llm_provider is None:
            from generation.providers.openai_gen import OpenAIProvider
            self._llm = OpenAIProvider()
        else:
            self._llm = llm_provider
        self._judge_model = judge_model

    def evaluate(
        self,
        query_id: str,
        query_text: str,
        expected_sources: list[str],
        answer: str,
        chunks: list[tuple],
        config: str = "DBW",
    ) -> JudgeResult:
        """Evaluate a single answer using LLM judge.

        Args:
            query_id: Evaluation query ID.
            query_text: The original query.
            expected_sources: Expected source types.
            answer: Generated answer text.
            chunks: Retrieved (Chunk, score) pairs.
            config: Config that produced this answer.

        Returns:
            JudgeResult with scores.
        """
        # Format chunks for prompt
        chunks_text = format_chunks_for_prompt(chunks, max_context_chars=6000)

        prompt = LLM_JUDGE_PROMPT.format(
            query=query_text,
            expected_sources=", ".join(expected_sources),
            answer=answer,
            chunks=chunks_text,
        )

        # Gemini thinking models need more tokens for chain-of-thought
        max_tok = 2048 if "gemini" in self._judge_model else 500
        response = self._llm.generate(prompt, max_tokens=max_tok, temperature=0.0)

        # Strip markdown code fences (Gemini wraps JSON in ```json ... ```)
        cleaned = response.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ```)
            first_nl = cleaned.find("\n")
            if first_nl > 0:
                cleaned = cleaned[first_nl + 1:]
            # Remove closing fence
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        # Parse JSON response — use find/rfind to handle nested braces
        try:
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                scores = json.loads(cleaned[start:end])
            else:
                scores = json.loads(cleaned)
        except json.JSONDecodeError:
            scores = {"rci": 0, "as": 0, "vm": 0, "root_cause_category": "unknown", "reasoning": "Parse error"}

        return JudgeResult(
            query_id=query_id,
            config=config,
            judge_model=self._judge_model,
            rci=int(scores.get("rci", 0)),
            as_=int(scores.get("as", 0)),
            vm=int(scores.get("vm", 0)),
            root_cause_category=scores.get("root_cause_category", "unknown"),
            reasoning=scores.get("reasoning", ""),
        )
