"""Parse VS Code documentation markdown files — YAML frontmatter, headings, code blocks, links."""

import re
import yaml


# --- YAML Frontmatter ---

FRONTMATTER_PATTERN = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)

# --- Headings ---

HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

# --- Fenced Code Blocks ---

CODE_BLOCK_PATTERN = re.compile(r'^```(\w*)\n(.*?)^```', re.MULTILINE | re.DOTALL)

# --- Internal Links ---

INTERNAL_LINK_PATTERN = re.compile(r'\[([^\]]*)\]\(([^)]+)\)')


def parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from markdown content."""
    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        return {}

    try:
        meta = yaml.safe_load(match.group(1))
        return meta if isinstance(meta, dict) else {}
    except yaml.YAMLError:
        return {}


def parse_headings(content: str) -> list[dict]:
    """Extract heading hierarchy with line numbers."""
    headings = []
    for i, line in enumerate(content.split('\n'), start=1):
        match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if match:
            headings.append({
                "level": len(match.group(1)),
                "text": match.group(2).strip(),
                "line": i,
            })
    return headings


def parse_code_blocks(content: str) -> list[dict]:
    """Extract fenced code blocks with language and line number."""
    blocks = []
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('```'):
            language = line[3:].strip()
            code_lines = []
            start_line = i + 1
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code_lines.append(lines[i])
                i += 1
            blocks.append({
                "language": language or "text",
                "content": '\n'.join(code_lines),
                "line": start_line,
            })
        i += 1
    return blocks


def parse_internal_links(content: str) -> list[str]:
    """Extract internal relative links (not http/https)."""
    links = []
    for match in INTERNAL_LINK_PATTERN.finditer(content):
        href = match.group(2)
        if not href.startswith(('http://', 'https://', '#', 'mailto:')):
            links.append(href)
    return list(set(links))


def strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from content, return body only."""
    return FRONTMATTER_PATTERN.sub('', content, count=1)


def derive_area_from_path(relative_path: str) -> str:
    """Derive the topic area from the file's directory path.

    e.g., 'docs/editor/codebasics.md' → 'editor'
          'docs/copilot/chat/copilot-chat.md' → 'copilot'
          'api/extension-guides/tree-view.md' → 'api'
    """
    parts = relative_path.replace("\\", "/").split("/")
    # Path format: docs/<area>/... or api/<area>/...
    if len(parts) >= 2:
        # Use the first directory under docs/ or api/
        return parts[1] if parts[0] in ("docs", "api") and len(parts) >= 3 else parts[0]
    return "unknown"


def parse_markdown_file(file_path: str, relative_path: str) -> dict:
    """Parse a single markdown file into a structured document object."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    frontmatter = parse_frontmatter(content)
    body = strip_frontmatter(content)

    # Area: prefer frontmatter field, fall back to directory-derived area
    area = frontmatter.get("Area", "") or derive_area_from_path(relative_path)

    return {
        "file_path": relative_path,
        "content_id": frontmatter.get("ContentId", ""),
        "area": area,
        "toc_title": frontmatter.get("TOCTitle", ""),
        "page_title": frontmatter.get("PageTitle", ""),
        "meta_description": frontmatter.get("MetaDescription", ""),
        "date_approved": str(frontmatter.get("DateApproved", "")),
        "headings": parse_headings(body),
        "code_blocks": parse_code_blocks(body),
        "internal_links": parse_internal_links(body),
        "raw_markdown": body,
    }
