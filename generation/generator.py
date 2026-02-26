"""Main generator — combines prompt template + formatter + LLM + citation extraction."""

from __future__ import annotations

from processing.schemas import Chunk
from generation.prompt_templates import GENERATION_PROMPT
from generation.formatter import format_chunks_for_prompt
from generation.citation_extractor import extract_citations


class Generator:
    """LLM-based answer generator with citation tracking."""

    def __init__(self, llm_provider=None):
        """
        Args:
            llm_provider: An LLMProvider instance. If None, creates default OpenAI.
        """
        if llm_provider is None:
            from generation.providers.openai_gen import OpenAIProvider
            self._llm = OpenAIProvider()
        else:
            self._llm = llm_provider

    def generate(
        self,
        query: str,
        chunks: list[tuple[Chunk, float]],
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> tuple[str, dict[str, list[str]]]:
        """Generate an answer with citations.

        Args:
            query: User question.
            chunks: Re-ranked (Chunk, score) list.
            max_tokens: Max generation tokens.
            temperature: LLM temperature.

        Returns:
            (answer_text, citations_dict) tuple.
        """
        # Format chunks into context
        context = format_chunks_for_prompt(chunks)

        # Build prompt
        prompt = GENERATION_PROMPT.format(context=context, query=query)

        # Generate
        answer = self._llm.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Extract citations
        citations = extract_citations(answer)

        return answer, citations
