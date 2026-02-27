"""Tests for the generation module: formatter, citation_extractor, prompt_templates."""

import pytest

from processing.schemas import Chunk
from generation.formatter import format_chunks_for_prompt, _make_tag
from generation.citation_extractor import extract_citations
from generation.prompt_templates import GENERATION_PROMPT, LLM_JUDGE_PROMPT


def _make_chunk(chunk_id, source_type, text="sample", feature_area="unknown"):
    return Chunk(
        chunk_id=chunk_id, source_type=source_type, source_id="test",
        source_url="https://example.com", text=text,
        text_with_context=f"[{source_type}] {text}",
        feature_area=feature_area,
        created_at="2024-01-01", updated_at="2024-01-01",
    )


# ===== Citation Tags =====

class TestCitationTags:
    def test_doc_tag(self):
        chunk = _make_chunk("doc_abc_0", "doc")
        assert _make_tag(chunk) == "[DOC-doc_abc_0]"

    def test_bug_tag(self):
        chunk = _make_chunk("bug_123_0", "bug")
        assert _make_tag(chunk) == "[BUG-bug_123_0]"

    def test_work_item_tag(self):
        chunk = _make_chunk("work_item_500_0", "work_item")
        assert _make_tag(chunk) == "[PLAN-work_item_500_0]"


# ===== Formatter =====

class TestFormatter:
    def test_format_basic(self):
        chunks = [
            (_make_chunk("doc_1", "doc", "Terminal configuration guide"), 0.95),
            (_make_chunk("bug_2", "bug", "Terminal crash on Windows"), 0.80),
        ]
        result = format_chunks_for_prompt(chunks)
        assert "[DOC-doc_1]" in result
        assert "[BUG-bug_2]" in result
        assert "Terminal configuration" in result
        assert "score: 0.950" in result

    def test_format_respects_max_context(self):
        chunks = [
            (_make_chunk(f"c_{i}", "doc", "x" * 500), float(1.0 - i * 0.1))
            for i in range(20)
        ]
        result = format_chunks_for_prompt(chunks, max_context_chars=2000)
        assert len(result) <= 3000  # Some tolerance for headers

    def test_format_empty(self):
        result = format_chunks_for_prompt([])
        assert result == ""


# ===== Citation Extractor =====

class TestCitationExtractor:
    def test_extract_doc_citations(self):
        answer = "According to [DOC-doc_abc_0], you should configure settings."
        citations = extract_citations(answer)
        assert "doc_abc_0" in citations["doc"]
        assert len(citations["bug"]) == 0

    def test_extract_bug_citations(self):
        answer = "This is a known issue [BUG-bug_123_0] that was fixed."
        citations = extract_citations(answer)
        assert "bug_123_0" in citations["bug"]

    def test_extract_plan_citations(self):
        answer = "This feature is planned [PLAN-work_item_500_0] for next release."
        citations = extract_citations(answer)
        assert "work_item_500_0" in citations["work_item"]

    def test_extract_multiple_citations(self):
        answer = (
            "The docs say [DOC-d1] and [DOC-d2]. "
            "Bug [BUG-b1] confirms this. "
            "Plan [PLAN-w1] addresses it."
        )
        citations = extract_citations(answer)
        assert len(citations["doc"]) == 2
        assert len(citations["bug"]) == 1
        assert len(citations["work_item"]) == 1

    def test_extract_no_citations(self):
        citations = extract_citations("This answer has no citations.")
        assert citations == {"doc": [], "bug": [], "work_item": []}

    def test_no_duplicate_citations(self):
        answer = "[DOC-d1] says X. Also [DOC-d1] confirms Y."
        citations = extract_citations(answer)
        assert len(citations["doc"]) == 1  # No duplicate


# ===== Prompt Templates =====

class TestPromptTemplates:
    def test_generation_prompt_has_placeholders(self):
        assert "{context}" in GENERATION_PROMPT
        assert "{query}" in GENERATION_PROMPT

    def test_generation_prompt_format(self):
        result = GENERATION_PROMPT.format(context="Test context", query="Test query")
        assert "Test context" in result
        assert "Test query" in result

    def test_judge_prompt_has_placeholders(self):
        assert "{query}" in LLM_JUDGE_PROMPT
        assert "{answer}" in LLM_JUDGE_PROMPT
        assert "{chunks}" in LLM_JUDGE_PROMPT
