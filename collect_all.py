"""Orchestrator: Run all data collection and post-processing pipelines."""

import os
import json
import argparse
import logging
from collections import Counter

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("collect_all")


def run_docs():
    from ingestion.collectors.collect_docs import collect_docs
    return collect_docs()


def run_bugs():
    from ingestion.collectors.collect_bugs import collect_bugs
    return collect_bugs()


def run_workitems():
    from ingestion.collectors.collect_workitems import collect_workitems
    return collect_workitems()


def run_comments():
    from ingestion.collectors.collect_comments import collect_comments
    return collect_comments()


def post_process():
    """Post-processing: merge comments into bugs, validate, compute stats."""
    logger.info("=" * 60)
    logger.info("POST-PROCESSING")
    logger.info("=" * 60)

    # --- Load raw data ---
    data = {}
    for source in ["docs", "bugs", "workitems", "comments"]:
        path = os.path.join(config.RAW_DATA_DIR, f"{source}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data[source] = json.load(f)
            logger.info(f"Loaded {source}: {len(data[source])} records")
        else:
            logger.warning(f"Missing: {path}")
            data[source] = [] if source != "comments" else {}

    # --- Merge comments into bugs ---
    comments = data["comments"]
    if isinstance(comments, dict) and data["bugs"]:
        merge_count = 0
        for bug in data["bugs"]:
            issue_key = str(bug["number"])
            if issue_key in comments:
                bug["team_comments"] = comments[issue_key]
                merge_count += 1
            else:
                bug["team_comments"] = []
        logger.info(f"Merged team comments into {merge_count} bugs")

    # --- Validate required fields ---
    validate_data(data)

    # --- Compute heterogeneity statistics ---
    compute_heterogeneity_stats(data)

    # --- Save processed data ---
    for source in ["docs", "bugs", "workitems"]:
        if data[source]:
            output_path = os.path.join(config.PROCESSED_DATA_DIR, f"{source}_processed.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data[source], f, indent=2, ensure_ascii=False)
            logger.info(f"Saved processed {source} to {output_path}")

    logger.info("Post-processing complete.")


def validate_data(data: dict):
    """Validate collected data for completeness."""
    logger.info("--- Validation ---")

    # Docs validation
    docs = data.get("docs", [])
    if docs:
        missing_area = sum(1 for d in docs if not d.get("area"))
        missing_content_id = sum(1 for d in docs if not d.get("content_id"))
        logger.info(f"Docs: {len(docs)} total | missing area: {missing_area} | missing content_id: {missing_content_id}")

    # Bugs validation
    bugs = data.get("bugs", [])
    if bugs:
        missing_body = sum(1 for b in bugs if not b.get("body"))
        has_version = sum(1 for b in bugs if b.get("parsed", {}).get("vscode_version"))
        has_os = sum(1 for b in bugs if b.get("parsed", {}).get("os_version"))
        has_steps = sum(1 for b in bugs if b.get("parsed", {}).get("steps_to_reproduce"))
        logger.info(f"Bugs: {len(bugs)} total | has_version: {has_version} | has_os: {has_os} | has_steps: {has_steps} | missing_body: {missing_body}")

    # Work items validation
    workitems = data.get("workitems", [])
    if workitems:
        type_counts = Counter(w.get("workitem_type", "unknown") for w in workitems)
        has_milestone = sum(1 for w in workitems if w.get("milestone"))
        has_checkboxes = sum(1 for w in workitems if w.get("parsed", {}).get("task_checkboxes"))
        logger.info(f"Work items: {len(workitems)} total | types: {dict(type_counts)} | has_milestone: {has_milestone} | has_checkboxes: {has_checkboxes}")


def compute_heterogeneity_stats(data: dict):
    """Compute structural heterogeneity statistics for paper Table 1."""
    logger.info("--- Heterogeneity Statistics (Paper Table 1) ---")

    stats = {}

    # --- Average text length per source ---
    for source_name, source_key, text_field in [
        ("Documentation", "docs", "raw_markdown"),
        ("Bug Reports", "bugs", "body"),
        ("Work Items", "workitems", "body"),
    ]:
        items = data.get(source_key, [])
        if items:
            lengths = [len(item.get(text_field, "")) for item in items]
            avg_len = sum(lengths) / len(lengths)
            stats[source_name] = {
                "count": len(items),
                "avg_text_length": round(avg_len),
                "min_text_length": min(lengths),
                "max_text_length": max(lengths),
            }
            logger.info(f"{source_name}: {len(items)} items, avg text length: {avg_len:.0f} chars")

    # --- Metadata field coverage ---
    bugs = data.get("bugs", [])
    if bugs:
        n = len(bugs)
        stats["Bug Reports"]["metadata_coverage"] = {
            "milestone": round(100 * sum(1 for b in bugs if b.get("milestone")) / n, 1),
            "vscode_version": round(100 * sum(1 for b in bugs if b.get("parsed", {}).get("vscode_version")) / n, 1),
            "os_version": round(100 * sum(1 for b in bugs if b.get("parsed", {}).get("os_version")) / n, 1),
            "feature_area": round(100 * sum(1 for b in bugs if b.get("feature_areas")) / n, 1),
            "reactions": round(100 * sum(1 for b in bugs if sum(b.get("reactions", {}).values()) > 0) / n, 1),
        }

    workitems = data.get("workitems", [])
    if workitems:
        n = len(workitems)
        stats["Work Items"]["metadata_coverage"] = {
            "milestone": round(100 * sum(1 for w in workitems if w.get("milestone")) / n, 1),
            "assignees": round(100 * sum(1 for w in workitems if w.get("assignees")) / n, 1),
            "feature_area": round(100 * sum(1 for w in workitems if w.get("feature_areas")) / n, 1),
            "task_checkboxes": round(100 * sum(1 for w in workitems if w.get("parsed", {}).get("task_checkboxes")) / n, 1),
            "reactions": round(100 * sum(1 for w in workitems if sum(w.get("reactions", {}).values()) > 0) / n, 1),
        }

    # --- Vocabulary overlap (Jaccard similarity) ---
    def get_vocabulary(items: list[dict], text_field: str, sample_size: int = 500) -> set:
        """Get unique words from a sample of items."""
        words = set()
        for item in items[:sample_size]:
            text = item.get(text_field, "") or ""
            words.update(text.lower().split())
        return words

    doc_vocab = get_vocabulary(data.get("docs", []), "raw_markdown")
    bug_vocab = get_vocabulary(data.get("bugs", []), "body")
    work_vocab = get_vocabulary(data.get("workitems", []), "body")

    if doc_vocab and bug_vocab:
        jaccard_doc_bug = len(doc_vocab & bug_vocab) / len(doc_vocab | bug_vocab)
        logger.info(f"Vocabulary overlap (Docs-Bugs): {jaccard_doc_bug:.3f}")

    if doc_vocab and work_vocab:
        jaccard_doc_work = len(doc_vocab & work_vocab) / len(doc_vocab | work_vocab)
        logger.info(f"Vocabulary overlap (Docs-WorkItems): {jaccard_doc_work:.3f}")

    if bug_vocab and work_vocab:
        jaccard_bug_work = len(bug_vocab & work_vocab) / len(bug_vocab | work_vocab)
        logger.info(f"Vocabulary overlap (Bugs-WorkItems): {jaccard_bug_work:.3f}")

    # --- Feature area distribution across sources ---
    all_areas = set()
    for source_key in ["bugs", "workitems"]:
        for item in data.get(source_key, []):
            all_areas.update(item.get("feature_areas", []))

    if all_areas:
        logger.info(f"Unique feature areas found: {len(all_areas)}")

    # Save stats
    stats_path = os.path.join(config.PROCESSED_DATA_DIR, "heterogeneity_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved heterogeneity stats to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="HeteroRAG Data Collection Pipeline")
    parser.add_argument(
        "--source",
        choices=["docs", "bugs", "workitems", "comments", "all", "process"],
        default="all",
        help="Which source to collect (default: all)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sources that already have output files",
    )
    args = parser.parse_args()

    logger.info(f"HeteroRAG Data Collection — Source: {args.source}")
    logger.info(f"Output directory: {config.RAW_DATA_DIR}")

    if args.source in ("docs", "all"):
        if args.skip_existing and os.path.exists(os.path.join(config.RAW_DATA_DIR, "docs.json")):
            logger.info("Skipping docs (file exists)")
        else:
            run_docs()

    if args.source in ("bugs", "all"):
        if args.skip_existing and os.path.exists(os.path.join(config.RAW_DATA_DIR, "bugs.json")):
            logger.info("Skipping bugs (file exists)")
        else:
            run_bugs()

    if args.source in ("workitems", "all"):
        if args.skip_existing and os.path.exists(os.path.join(config.RAW_DATA_DIR, "workitems.json")):
            logger.info("Skipping workitems (file exists)")
        else:
            run_workitems()

    if args.source in ("comments", "all"):
        if args.skip_existing and os.path.exists(os.path.join(config.RAW_DATA_DIR, "comments.json")):
            logger.info("Skipping comments (file exists)")
        else:
            run_comments()

    if args.source in ("all", "process"):
        post_process()


if __name__ == "__main__":
    main()
