"""Source 1: Collect and parse VS Code documentation from microsoft/vscode-docs."""

import os
import sys
import json
import logging
import subprocess

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from ingestion.parsers.markdown_parser import parse_markdown_file
from ingestion.utils.anonymizer import anonymize_text

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("collect_docs")

DOCS_DIR = "vscode-docs"
TARGET_FOLDERS = ["docs", "api"]


def clone_docs_repo():
    """Clone vscode-docs with sparse checkout (docs/ and api/ only)."""
    if os.path.exists(DOCS_DIR):
        logger.info(f"Repo already exists at {DOCS_DIR}, pulling latest...")
        subprocess.run(["git", "-C", DOCS_DIR, "pull", "--ff-only"], check=True)
        return

    logger.info("Cloning microsoft/vscode-docs (sparse checkout: docs/ + api/)...")
    subprocess.run([
        "git", "clone", "--no-checkout", "--depth", "1",
        config.DOCS_CLONE_URL, DOCS_DIR,
    ], check=True)

    subprocess.run(["git", "-C", DOCS_DIR, "sparse-checkout", "set"] + TARGET_FOLDERS, check=True)
    subprocess.run(["git", "-C", DOCS_DIR, "checkout"], check=True)
    logger.info("Clone complete.")


def collect_markdown_files() -> list[str]:
    """Walk target folders and collect all .md file paths."""
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

    logger.info(f"Found {len(md_files)} markdown files")
    return md_files


def collect_docs():
    """Main collection pipeline for documentation source."""
    clone_docs_repo()

    md_files = collect_markdown_files()
    docs = []
    errors = 0

    for filepath in md_files:
        relative = os.path.relpath(filepath, DOCS_DIR).replace("\\", "/")
        try:
            doc = parse_markdown_file(filepath, relative)

            # Anonymize content
            doc["raw_markdown"] = anonymize_text(doc["raw_markdown"])
            doc["meta_description"] = anonymize_text(doc.get("meta_description", ""))

            docs.append(doc)
        except Exception as e:
            logger.error(f"Failed to parse {relative}: {e}")
            errors += 1

    # Save output
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

    return docs


if __name__ == "__main__":
    collect_docs()
