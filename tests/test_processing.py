"""Tests for the processing module: schemas, feature_area_map, and chunkers."""

import pytest

from processing.schemas import Chunk, TreeNode
from processing.feature_area_map import (
    extract_feature_area, FEATURE_AREA_MAP, DOC_AREA_MAP,
)
from processing.doc_chunker import (
    _split_by_headings, _build_heading_path, _split_large_section,
    _make_doc_url, _estimate_tokens, process_all as process_docs,
)
from processing.bug_chunker import (
    _detect_os, _truncate, _extract_errors,
    process_all as process_bugs,
)
from processing.workitem_chunker import (
    _parse_completion_ratio, _total_reactions,
    process_all as process_workitems,
)


# ===== Chunk schema =====

class TestChunk:
    def test_create_chunk(self):
        chunk = Chunk(
            chunk_id="doc_abc_0",
            source_type="doc",
            source_id="docs/terminal/basics.md",
            source_url="https://code.visualstudio.com/docs/terminal/basics",
            text="Terminal basics content",
            text_with_context="[DOC] Area: terminal | Topic: Basics\nTerminal basics content",
            feature_area="terminal",
            created_at="2024-01-01",
            updated_at="2024-01-01",
        )
        assert chunk.chunk_id == "doc_abc_0"
        assert chunk.source_type == "doc"
        assert chunk.feature_area == "terminal"

    def test_chunk_to_dict_roundtrip(self):
        chunk = Chunk(
            chunk_id="bug_123_0", source_type="bug", source_id="123",
            source_url="https://github.com/microsoft/vscode/issues/123",
            text="Bug text", text_with_context="[BUG] #123\nBug text",
            feature_area="editor", created_at="2024-01-01", updated_at="2024-06-01",
            metadata={"state": "closed", "verified": True},
        )
        d = chunk.to_dict()
        restored = Chunk.from_dict(d)
        assert restored.chunk_id == chunk.chunk_id
        assert restored.metadata["state"] == "closed"
        assert restored.metadata["verified"] is True

    def test_chunk_default_metadata(self):
        chunk = Chunk(
            chunk_id="x", source_type="doc", source_id="x",
            source_url="", text="", text_with_context="",
            feature_area="unknown", created_at="", updated_at="",
        )
        assert chunk.metadata == {}


class TestTreeNode:
    def test_create_tree_node(self):
        node = TreeNode(
            node_id="root", title="Page", summary="A doc page",
            depth=0, chunk_ids=["c1", "c2"], children=["child1"],
        )
        assert node.depth == 0
        assert len(node.chunk_ids) == 2

    def test_tree_node_roundtrip(self):
        node = TreeNode(
            node_id="n1", title="Section", summary="Summary",
            depth=2, parent="root",
        )
        d = node.to_dict()
        restored = TreeNode.from_dict(d)
        assert restored.node_id == "n1"
        assert restored.parent == "root"


# ===== Feature area map =====

class TestFeatureAreaMap:
    def test_exact_label_match(self):
        assert extract_feature_area(["terminal"], source_type="bug") == "terminal"
        assert extract_feature_area(["editor-core"], source_type="bug") == "editor"
        assert extract_feature_area(["scm"], source_type="bug") == "git"

    def test_doc_area_match(self):
        assert extract_feature_area(["editing"], source_type="doc") == "editor"
        assert extract_feature_area(["sourcecontrol"], source_type="doc") == "git"
        assert extract_feature_area(["debugging"], source_type="doc") == "debug"

    def test_unknown_label(self):
        assert extract_feature_area(["nonexistent-label"], source_type="bug") == "unknown"
        assert extract_feature_area([], source_type="bug") == "unknown"

    def test_prefix_fallback(self):
        # "editor-something-new" should match "editor" prefix
        result = extract_feature_area(["editor-something-new"], source_type="bug")
        assert result == "editor"

    def test_multiple_labels_first_wins(self):
        result = extract_feature_area(["bug", "terminal", "editor-core"], source_type="bug")
        assert result == "terminal"  # First match in list

    def test_maps_not_empty(self):
        assert len(FEATURE_AREA_MAP) > 50
        assert len(DOC_AREA_MAP) > 15


# ===== Doc chunker =====

class TestDocChunker:
    def test_make_doc_url(self):
        assert _make_doc_url("docs/terminal/basics.md") == "https://code.visualstudio.com/docs/terminal/basics"
        assert _make_doc_url("docs/editor/codebasics.md") == "https://code.visualstudio.com/docs/editor/codebasics"

    def test_estimate_tokens(self):
        assert _estimate_tokens("word " * 100) == 125  # 500 chars / 4

    def test_split_by_headings(self):
        md = "# Title\nIntro\n## Section 1\nContent 1\n## Section 2\nContent 2"
        sections = _split_by_headings(md)
        assert len(sections) >= 2
        titles = [s["title"] for s in sections]
        assert "Title" in titles
        assert "Section 1" in titles

    def test_split_by_headings_no_headings(self):
        md = "Just plain text without any headings."
        sections = _split_by_headings(md)
        assert len(sections) == 1

    def test_build_heading_path(self):
        sections = [
            {"level": 1, "title": "Page", "content": ""},
            {"level": 2, "title": "Features", "content": ""},
            {"level": 3, "title": "Config", "content": ""},
        ]
        path = _build_heading_path(sections, 2)
        assert "Features" in path
        assert "Config" in path

    def test_split_large_section(self):
        large_text = ("word one. " * 30 + "\n\n" + "word two. " * 30 + "\n\n" + "word three. " * 30)
        parts = _split_large_section(large_text, 400)
        assert len(parts) >= 2

    def test_process_docs_basic(self):
        docs = [{
            "file_path": "docs/test/sample.md",
            "area": "editor",
            "raw_markdown": "# Test Page\nIntro text here.\n## Section A\nContent A with details.\n## Section B\nContent B with more details.",
            "page_title": "Test Page",
            "toc_title": "Test",
            "headings": [],
            "code_blocks": [],
            "internal_links": [],
            "date_approved": "2024-01-01",
        }]
        chunks, nodes = process_docs(docs)
        assert len(chunks) > 0
        assert len(nodes) > 0
        assert all(c.source_type == "doc" for c in chunks)
        assert all("[DOC]" in c.text_with_context for c in chunks)
        # Tree has a root node
        roots = [n for n in nodes if n.depth == 0]
        assert len(roots) == 1


# ===== Bug chunker =====

class TestBugChunker:
    def test_detect_os(self):
        assert _detect_os("Running Windows 11 Pro") == "windows"
        assert _detect_os("macOS Ventura 13.4") == "macos"
        assert _detect_os("Ubuntu 22.04 LTS") == "linux"
        assert _detect_os("No OS info here") == "unknown"

    def test_truncate(self):
        assert _truncate("short", 100) == "short"
        long_text = "word " * 100
        result = _truncate(long_text, 20)
        assert len(result) <= 25  # 20 + "..."
        assert result.endswith("...")

    def test_truncate_empty(self):
        assert _truncate("", 100) == ""
        assert _truncate(None, 100) == ""

    def test_extract_errors(self):
        body = "Something\nError: ENOENT file not found\nMore text"
        errors = _extract_errors(body, {})
        assert "ENOENT" in errors

    def test_extract_errors_from_parsed(self):
        errors = _extract_errors("", {"error_messages": ["TypeError: undefined"]})
        assert "TypeError" in errors

    def test_process_bugs_basic(self):
        bugs = [{
            "number": 12345,
            "title": "Editor crashes on large file",
            "body": "VS Code crashes when opening files > 100MB.\nError: out of memory",
            "state": "open",
            "labels": ["bug", "editor-core"],
            "parsed": {"vscode_version": "1.85.0", "os_version": "", "error_messages": [], "steps_to_reproduce": "", "extensions_list": []},
            "team_comments": [],
            "reactions": {"+1": 5, "-1": 0, "laugh": 0, "hooray": 0, "confused": 0, "heart": 0, "rocket": 0, "eyes": 0},
            "feature_areas": [],
            "created_at": "2024-01-15",
            "updated_at": "2024-02-01",
            "html_url": "https://github.com/microsoft/vscode/issues/12345",
            "milestone": None,
        }]
        chunks = process_bugs(bugs, {})
        assert len(chunks) >= 1
        assert chunks[0].source_type == "bug"
        assert chunks[0].feature_area == "editor"
        assert "[BUG]" in chunks[0].text_with_context
        assert chunks[0].metadata["state"] == "open"

    def test_process_bugs_with_comments(self):
        bugs = [{
            "number": 99,
            "title": "Test bug",
            "body": "Body text",
            "state": "closed",
            "labels": ["bug", "terminal"],
            "parsed": {},
            "team_comments": [{"comment_body": "Fixed in version 1.86", "comment_author": "dev", "author_association": "MEMBER", "created_at": "2024-01-20"}],
            "reactions": {"+1": 0, "-1": 0, "laugh": 0, "hooray": 0, "confused": 0, "heart": 0, "rocket": 0, "eyes": 0},
            "feature_areas": [],
            "created_at": "2024-01-01",
            "updated_at": "2024-01-20",
            "html_url": "https://github.com/microsoft/vscode/issues/99",
            "milestone": None,
        }]
        chunks = process_bugs(bugs, {})
        assert "Fixed in version" in chunks[0].text


# ===== Work item chunker =====

class TestWorkItemChunker:
    def test_parse_completion_ratio(self):
        body = "- [x] Done\n- [x] Also done\n- [ ] Not done\n- [ ] Also not"
        checked, total = _parse_completion_ratio(body)
        assert checked == 2
        assert total == 4

    def test_parse_completion_ratio_empty(self):
        checked, total = _parse_completion_ratio("No checkboxes here")
        assert checked == 0
        assert total == 0

    def test_total_reactions(self):
        reactions = {"+1": 10, "heart": 5, "rocket": 3, "hooray": 1, "-1": 2}
        assert _total_reactions(reactions) == 19

    def test_process_feature_request(self):
        items = [{
            "number": 500,
            "title": "Add dark mode toggle",
            "body": "Would be great to have a quick toggle for dark mode.",
            "workitem_type": "feature_request",
            "state": "open",
            "state_reason": None,
            "labels": ["feature-request", "workbench-layout"],
            "feature_areas": [],
            "milestone": "January 2025",
            "assignees": [],
            "created_at": "2024-06-01",
            "closed_at": None,
            "reactions": {"+1": 42, "-1": 0, "laugh": 0, "hooray": 0, "confused": 0, "heart": 8, "rocket": 3, "eyes": 0},
            "html_url": "https://github.com/microsoft/vscode/issues/500",
            "comments_count": 5,
            "parsed": {},
        }]
        chunks = process_workitems(items)
        assert len(chunks) == 1
        assert chunks[0].source_type == "work_item"
        assert chunks[0].feature_area == "workbench"
        assert "42" in chunks[0].text  # reactions mentioned
        assert "[PLAN]" in chunks[0].text_with_context

    def test_process_iteration_plan(self):
        items = [{
            "number": 900,
            "title": "January 2025 Iteration Plan",
            "body": "# Plan\n## Editor\n- [x] Feature A\n- [ ] Feature B\n## Terminal\n- [x] Feature C",
            "workitem_type": "iteration_plan",
            "state": "closed",
            "state_reason": None,
            "labels": ["iteration-plan"],
            "feature_areas": [],
            "milestone": "January 2025",
            "assignees": [],
            "created_at": "2025-01-01",
            "closed_at": "2025-02-01",
            "reactions": {"+1": 0, "-1": 0, "laugh": 0, "hooray": 0, "confused": 0, "heart": 0, "rocket": 0, "eyes": 0},
            "html_url": "https://github.com/microsoft/vscode/issues/900",
            "comments_count": 0,
            "parsed": {},
        }]
        chunks = process_workitems(items)
        assert len(chunks) >= 2  # Split by sections
        assert all(c.metadata["item_type"] == "iteration_plan" for c in chunks)

    def test_process_plan_item(self):
        items = [{
            "number": 700,
            "title": "Improve terminal reconnection",
            "body": "- [x] Step 1\n- [x] Step 2\n- [ ] Step 3",
            "workitem_type": "plan_item",
            "state": "open",
            "state_reason": None,
            "labels": ["plan-item", "terminal"],
            "feature_areas": [],
            "milestone": None,
            "assignees": [],
            "created_at": "2024-09-01",
            "closed_at": None,
            "reactions": {"+1": 0, "-1": 0, "laugh": 0, "hooray": 0, "confused": 0, "heart": 0, "rocket": 0, "eyes": 0},
            "html_url": "https://github.com/microsoft/vscode/issues/700",
            "comments_count": 0,
            "parsed": {},
        }]
        chunks = process_workitems(items)
        assert len(chunks) == 1
        assert "2/3" in chunks[0].text  # completion ratio
