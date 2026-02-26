"""Tier 2: Collect team member comments for high-value verified closed bugs."""

import os
import sys
import json
import logging
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from ingestion.collectors.github_client import GitHubClient
from ingestion.utils.checkpoint import CheckpointManager
from ingestion.utils.anonymizer import anonymize_comment

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("collect_comments")


def filter_high_value_bugs(bugs: list[dict]) -> list[dict]:
    """Filter bugs that qualify for comment collection.

    Criteria: state=closed AND 'verified' in labels AND comments_count >= 2
    """
    qualified = []
    for bug in bugs:
        if (bug.get("state") == "closed"
                and "verified" in bug.get("labels", [])
                and bug.get("comments_count", 0) >= 2):
            qualified.append(bug)

    logger.info(f"Filtered {len(qualified)} high-value bugs from {len(bugs)} total")
    return qualified


def extract_team_comments(comments: list[dict]) -> list[dict]:
    """Filter and transform comments to keep only team member responses."""
    team_comments = []
    for comment in comments:
        association = comment.get("author_association", "")
        if association in config.COMMENT_FILTER["team_associations"]:
            team_comment = {
                "comment_body": comment.get("body", ""),
                "comment_author": comment.get("user", {}).get("login", "") if comment.get("user") else "",
                "author_association": association,
                "created_at": comment.get("created_at", ""),
            }
            team_comment = anonymize_comment(team_comment)
            team_comments.append(team_comment)
    return team_comments


def collect_comments():
    """Main collection pipeline for team comments."""
    # Load bugs
    bugs_path = os.path.join(config.RAW_DATA_DIR, "bugs.json")
    if not os.path.exists(bugs_path):
        logger.error(f"Bugs file not found: {bugs_path}. Run collect_bugs.py first.")
        return {}

    with open(bugs_path, "r", encoding="utf-8") as f:
        bugs = json.load(f)

    qualified_bugs = filter_high_value_bugs(bugs)

    client = GitHubClient()
    checkpoint = CheckpointManager(config.CHECKPOINT_DIR, "comments")

    client.print_rate_limit()

    all_comments = {}

    # Load existing from checkpoint
    for key in checkpoint.state["completed"]:
        data = checkpoint.get_data(key)
        if data:
            all_comments[key] = data

    logger.info(f"Starting comment collection. {len(all_comments)} issues already done.")

    batch_count = 0
    for bug in tqdm(qualified_bugs, desc="Fetching comments"):
        issue_key = str(bug["number"])

        if checkpoint.is_done(issue_key):
            continue

        try:
            raw_comments = client.get_issue_comments(bug["number"])
            team_comments = extract_team_comments(raw_comments)

            if team_comments:
                all_comments[issue_key] = team_comments

            checkpoint.mark_done(issue_key, team_comments)
            batch_count += 1

            # Log progress every 100 issues
            if batch_count % 100 == 0:
                logger.info(f"Progress: {batch_count} issues processed, {len(all_comments)} with team comments")

        except Exception as e:
            logger.error(f"Failed to fetch comments for #{bug['number']}: {e}")
            continue

    # Save output
    output_path = os.path.join(config.RAW_DATA_DIR, "comments.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_comments, f, indent=2, ensure_ascii=False)

    total_comments = sum(len(v) for v in all_comments.values())
    logger.info(f"Collected comments for {len(all_comments)} issues ({total_comments} team comments total). Saved to {output_path}")

    return all_comments


if __name__ == "__main__":
    collect_comments()
