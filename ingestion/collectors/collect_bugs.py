"""Source 2: Collect VS Code bug reports via GitHub Search API, paginated by month."""

import os
import sys
import json
import logging
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from ingestion.collectors.github_client import GitHubClient
from ingestion.parsers.bug_body_parser import parse_bug_body
from ingestion.utils.checkpoint import CheckpointManager
from ingestion.utils.anonymizer import anonymize_issue

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("collect_bugs")


def extract_feature_areas(labels: list[dict]) -> list[str]:
    """Extract feature area labels from issue labels."""
    label_names = [l["name"] for l in labels if isinstance(l, dict)]
    return [name for name in label_names if name in config.FEATURE_AREA_LABELS]


def transform_issue(issue: dict) -> dict:
    """Transform raw GitHub API issue into our schema."""
    labels_raw = issue.get("labels", [])
    label_names = [l["name"] for l in labels_raw if isinstance(l, dict)]

    result = {
        "number": issue["number"],
        "title": issue.get("title", ""),
        "body": issue.get("body", "") or "",
        "state": issue.get("state", ""),
        "state_reason": issue.get("state_reason", ""),
        "labels": label_names,
        "feature_areas": extract_feature_areas(labels_raw),
        "milestone": issue.get("milestone", {}).get("title", "") if issue.get("milestone") else "",
        "created_at": issue.get("created_at", ""),
        "closed_at": issue.get("closed_at", ""),
        "updated_at": issue.get("updated_at", ""),
        "comments_count": issue.get("comments", 0),
        "reactions": issue.get("reactions", {}),
        "html_url": issue.get("html_url", ""),
        "author_association": issue.get("author_association", ""),
        "user_login": issue.get("user", {}).get("login", "") if issue.get("user") else "",
    }

    # Clean reactions dict — keep only counts
    if isinstance(result["reactions"], dict):
        result["reactions"] = {
            k: v for k, v in result["reactions"].items()
            if k in ("+1", "-1", "laugh", "hooray", "confused", "heart", "rocket", "eyes")
        }

    # Parse structured fields from body
    result["parsed"] = parse_bug_body(result["body"])

    return result


def collect_bugs():
    """Main collection pipeline for bug reports."""
    client = GitHubClient()
    checkpoint = CheckpointManager(config.CHECKPOINT_DIR, "bugs")

    client.print_rate_limit()

    all_bugs = {}

    # Load previously collected bugs from checkpoint
    for month_label in checkpoint.state["completed"]:
        month_data = checkpoint.get_data(month_label)
        if month_data:
            for bug in month_data:
                all_bugs[bug["number"]] = bug

    logger.info(f"Starting bug collection. {len(all_bugs)} bugs already in checkpoint.")

    for date_range in tqdm(config.BUG_DATE_RANGES, desc="Collecting bugs by month"):
        month_label = date_range["label"]

        if checkpoint.is_done(month_label):
            logger.info(f"Skipping {month_label} (already done)")
            continue

        query = f'repo:{config.REPO_OWNER}/{config.REPO_NAME} is:issue label:bug created:{date_range["start"]}..{date_range["end"]}'
        logger.info(f"Searching: {month_label} ({date_range['start']} to {date_range['end']})")

        month_bugs = []
        for issue in client.search_issues(query):
            # Skip pull requests
            if "pull_request" in issue:
                continue

            bug = transform_issue(issue)
            bug = anonymize_issue(bug)
            month_bugs.append(bug)
            all_bugs[bug["number"]] = bug

        checkpoint.mark_done(month_label, month_bugs)
        logger.info(f"  {month_label}: {len(month_bugs)} bugs (total: {len(all_bugs)})")

    # Deduplicate and save
    bugs_list = sorted(all_bugs.values(), key=lambda x: x["number"])

    output_path = os.path.join(config.RAW_DATA_DIR, "bugs.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bugs_list, f, indent=2, ensure_ascii=False)

    logger.info(f"Collected {len(bugs_list)} unique bugs. Saved to {output_path}")

    # Quick stats
    states = {}
    for bug in bugs_list:
        s = bug["state"]
        states[s] = states.get(s, 0) + 1
    logger.info(f"State distribution: {states}")

    parsed_versions = sum(1 for b in bugs_list if b["parsed"]["vscode_version"])
    logger.info(f"Parsed vscode_version: {parsed_versions}/{len(bugs_list)} ({100 * parsed_versions // max(len(bugs_list), 1)}%)")

    return bugs_list


if __name__ == "__main__":
    collect_bugs()
