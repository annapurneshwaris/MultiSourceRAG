"""Source 1: Collect and parse Flutter documentation from flutter/website."""

import os
import sys
import json
import logging
import subprocess
import re
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config_flutter as config

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("collect_flutter_docs")

DOCS_DIR = "flutter-website"
TARGET_FOLDERS = ["src/content"]


def clone_docs_repo():
    """Clone flutter/website repo."""
    if os.path.exists(DOCS_DIR):
        logger.info(f"Repo already exists at {DOCS_DIR}, pulling latest...")
        subprocess.run(["git", "-C", DOCS_DIR, "pull", "--ff-only"], check=True)
        return

    logger.info("Cloning flutter/website (sparse checkout: src/content/)...")
    subprocess.run([
        "git", "clone", "--no-checkout", "--depth", "1",
        config.DOCS_CLONE_URL, DOCS_DIR,
    ], check=True)

    subprocess.run(["git", "-C", DOCS_DIR, "sparse-checkout", "set", "src/content"], check=True)
    subprocess.run(["git", "-C", DOCS_DIR, "checkout"], check=True)
    logger.info("Clone complete.")


def collect_markdown_files(limit: int = 0) -> list[str]:
    """Walk src/content/ and collect all .md file paths."""
    md_files = []
    for folder in TARGET_FOLDERS:
        folder_path = os.path.join(DOCS_DIR, folder)
        if not os.path.exists(folder_path):
            logger.warning(f"Folder not found: {folder_path}")
            continue
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.endswith('.md'):
                    md_files.append(os.path.join(root, f))
                    if limit and len(md_files) >= limit:
                        logger.info(f"Hit limit of {limit} files")
                        return md_files

    logger.info(f"Found {len(md_files)} markdown files")
    return md_files


FRONTMATTER_PATTERN = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)
HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)


def parse_frontmatter(content: str) -> dict:
    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        return {}
    try:
        meta = yaml.safe_load(match.group(1))
        return meta if isinstance(meta, dict) else {}
    except yaml.YAMLError:
        return {}


def strip_frontmatter(content: str) -> str:
    return FRONTMATTER_PATTERN.sub('', content, count=1)


def parse_headings(content: str) -> list[dict]:
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


def derive_area_from_path(relative_path: str) -> str:
    """Derive topic area from directory path.

    e.g., 'src/content/ui/widgets.md' -> 'ui'
          'src/content/testing/overview.md' -> 'testing'
    """
    parts = relative_path.replace("\\", "/").split("/")
    # Find 'content' in path and take next directory
    try:
        content_idx = parts.index("content")
        if content_idx + 1 < len(parts) - 1:  # At least one dir after 'content'
            return parts[content_idx + 1]
    except ValueError:
        pass
    return "unknown"


def parse_flutter_doc(file_path: str, relative_path: str) -> dict:
    """Parse a single Flutter doc markdown file."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    frontmatter = parse_frontmatter(content)
    body = strip_frontmatter(content)
    area = derive_area_from_path(relative_path)

    return {
        "file_path": relative_path,
        "content_id": "",
        "area": area,
        "toc_title": frontmatter.get("short-title", "") or frontmatter.get("title", ""),
        "page_title": frontmatter.get("title", ""),
        "meta_description": frontmatter.get("description", ""),
        "date_approved": "",
        "headings": parse_headings(body),
        "code_blocks": [],  # Skip detailed code block parsing for speed
        "internal_links": [],
        "raw_markdown": body,
    }


def collect_docs(limit: int = 0):
    """Main collection pipeline for Flutter documentation."""
    clone_docs_repo()

    md_files = collect_markdown_files(limit=limit)
    docs = []
    errors = 0

    for filepath in md_files:
        relative = os.path.relpath(filepath, DOCS_DIR).replace("\\", "/")
        try:
            doc = parse_flutter_doc(filepath, relative)
            docs.append(doc)
        except Exception as e:
            logger.error(f"Failed to parse {relative}: {e}")
            errors += 1

    # Save output
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    output_path = os.path.join(config.RAW_DATA_DIR, "docs.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

    logger.info(f"Collected {len(docs)} docs ({errors} errors). Saved to {output_path}")

    # Quick stats
    areas = {}
    for doc in docs:
        area = doc.get("area", "unknown") or "unknown"
        areas[area] = areas.get(area, 0) + 1
    logger.info(f"Areas distribution: {dict(sorted(areas.items(), key=lambda x: -x[1])[:15])}")

    # Sample output
    for doc in docs[:3]:
        logger.info(f"  Sample: {doc['file_path']} | area={doc['area']} | title={doc['page_title'][:60]}")

    return docs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max docs to collect (0=all)")
    args = parser.parse_args()
    collect_docs(limit=args.limit)
