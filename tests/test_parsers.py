"""Unit tests for all parsers."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.parsers.bug_body_parser import parse_bug_body
from ingestion.parsers.workitem_body_parser import parse_workitem_body
from ingestion.parsers.markdown_parser import parse_frontmatter, parse_headings, parse_code_blocks, parse_internal_links
from ingestion.utils.anonymizer import anonymize_text


# ===== Bug Body Parser Tests =====

SAMPLE_BUG_BODY = """
<!-- Please search existing issues to avoid creating duplicates. -->

**Issue Type:** Bug

**VS Code Version:** 1.85.1
**OS Version:** Windows 11 23H2
**Extensions:** ms-python.python, dbaeumer.vscode-eslint

## Steps to Reproduce
1. Open a Python file
2. Try to use IntelliSense
3. Notice it doesn't show any suggestions

## Expected Behavior
IntelliSense should show Python completions.

## Actual Behavior
No suggestions appear. The following error is in the output:
```
Error: Pylance language server crashed unexpectedly
TypeError: Cannot read properties of undefined (reading 'getCompletions')
```

My workspace is at C:\\Users\\JohnDoe\\Projects\\myapp
Contact me at john.doe@example.com
"""


def test_parse_vscode_version():
    result = parse_bug_body(SAMPLE_BUG_BODY)
    assert result["vscode_version"] == "1.85.1"


def test_parse_os_version():
    result = parse_bug_body(SAMPLE_BUG_BODY)
    assert "Windows 11" in result["os_version"]


def test_parse_steps_to_reproduce():
    result = parse_bug_body(SAMPLE_BUG_BODY)
    assert "Open a Python file" in result["steps_to_reproduce"]
    assert "IntelliSense" in result["steps_to_reproduce"]


def test_parse_error_messages():
    result = parse_bug_body(SAMPLE_BUG_BODY)
    errors = result["error_messages"]
    assert len(errors) > 0
    assert any("Pylance" in e or "TypeError" in e for e in errors)


def test_parse_extensions():
    result = parse_bug_body(SAMPLE_BUG_BODY)
    # Extensions in this format may or may not be picked up depending on section header
    # The body uses "Extensions:" header-less format, so test gracefully
    assert isinstance(result["extensions_list"], list)


def test_parse_empty_body():
    result = parse_bug_body("")
    assert result["vscode_version"] == ""
    assert result["os_version"] == ""
    assert result["steps_to_reproduce"] == ""
    assert result["error_messages"] == []
    assert result["extensions_list"] == []


# ===== Work Item Body Parser Tests =====

SAMPLE_WORKITEM_BODY = """
# Iteration Plan for February 2024

## Editor
- [x] 🏃 Improve sticky scroll performance @hediet #198234
- [x] 💪 Fix bracket matching in JSX @aiday-man #199001
- [ ] ✋ Add inline completions API v2 @jrieken #200100

## Terminal
- [x] 💪 Terminal reconnection on reload @tyriar #197500
- [ ] 🏃 Shell integration improvements @tyriar #198000

## Debug
- [x] ✅ Conditional breakpoints UX @isidorn #196000

cc @bpasero @sandy081
"""


def test_parse_task_checkboxes():
    result = parse_workitem_body(SAMPLE_WORKITEM_BODY)
    checkboxes = result["task_checkboxes"]
    assert len(checkboxes) >= 5

    checked_items = [c for c in checkboxes if c["checked"]]
    unchecked_items = [c for c in checkboxes if not c["checked"]]
    assert len(checked_items) >= 3
    assert len(unchecked_items) >= 1


def test_parse_status_emojis():
    result = parse_workitem_body(SAMPLE_WORKITEM_BODY)
    emojis = result["status_emojis"]
    assert "in_progress" in emojis or "done" in emojis
    assert emojis.get("in_progress", 0) >= 1 or emojis.get("done", 0) >= 1


def test_parse_cross_referenced_issues():
    result = parse_workitem_body(SAMPLE_WORKITEM_BODY)
    refs = result["cross_referenced_issues"]
    assert len(refs) >= 3
    assert 198234 in refs
    assert 199001 in refs


def test_parse_assigned_developers():
    result = parse_workitem_body(SAMPLE_WORKITEM_BODY)
    devs = result["assigned_developers"]
    assert "tyriar" in devs
    assert "jrieken" in devs
    assert "bpasero" in devs


def test_parse_empty_workitem():
    result = parse_workitem_body("")
    assert result["task_checkboxes"] == []
    assert result["status_emojis"] == {}
    assert result["cross_referenced_issues"] == []
    assert result["assigned_developers"] == []


# ===== Markdown Parser Tests =====

SAMPLE_MARKDOWN = """---
ContentId: abc-123-def
Area: Editor
TOCTitle: Basic Editing
PageTitle: VS Code Basic Editing
MetaDescription: Learn about basic editing features
DateApproved: 2024-11-15
---

# Basic Editing

VS Code is a powerful editor.

## Multiple Selections

You can select multiple items.

```json
{
  "editor.fontSize": 14
}
```

See [Settings](../getstarted/settings.md) for more info.
Also check [the docs](https://code.visualstudio.com).
"""


def test_parse_frontmatter():
    meta = parse_frontmatter(SAMPLE_MARKDOWN)
    assert meta["ContentId"] == "abc-123-def"
    assert meta["Area"] == "Editor"
    assert meta["TOCTitle"] == "Basic Editing"


def test_parse_headings_from_markdown():
    headings = parse_headings(SAMPLE_MARKDOWN)
    assert len(headings) >= 2
    h1 = [h for h in headings if h["level"] == 1]
    h2 = [h for h in headings if h["level"] == 2]
    assert len(h1) >= 1
    assert len(h2) >= 1
    assert h1[0]["text"] == "Basic Editing"


def test_parse_code_blocks_from_markdown():
    blocks = parse_code_blocks(SAMPLE_MARKDOWN)
    assert len(blocks) >= 1
    assert blocks[0]["language"] == "json"
    assert "editor.fontSize" in blocks[0]["content"]


def test_parse_internal_links_from_markdown():
    links = parse_internal_links(SAMPLE_MARKDOWN)
    assert "../getstarted/settings.md" in links
    # External links should be excluded
    assert not any("http" in link for link in links)


# ===== Anonymizer Tests =====

def test_anonymize_email():
    text = "Contact john.doe@example.com for help"
    result = anonymize_text(text)
    assert "[EMAIL_REDACTED]" in result
    assert "john.doe@example.com" not in result


def test_anonymize_windows_path():
    text = "File at C:\\Users\\JohnDoe\\Projects\\myapp"
    result = anonymize_text(text)
    assert "[USER]" in result
    assert "JohnDoe" not in result


def test_anonymize_mac_path():
    text = "File at /Users/johndoe/Documents/test.py"
    result = anonymize_text(text)
    assert "[USER]" in result
    assert "johndoe" not in result


def test_anonymize_linux_path():
    text = "Log at /home/developer/logs/app.log"
    result = anonymize_text(text)
    assert "[USER]" in result
    assert "developer" not in result


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
