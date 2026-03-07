"""Source 3: Collect Flutter work items (feature requests + priority items)."""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config_flutter as config
from ingestion.collectors.github_client_flutter import GitHubClient
from ingestion.utils.checkpoint import CheckpointManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("collect_flutter_workitems")


def extract_feature_areas(labels: list[dict]) -> list[str]:
    label_names = [l["name"] for l in labels if isinstance(l, dict)]
    return [name for name in label_names if name in config.FEATURE_AREA_LABELS]


def transform_workitem(issue: dict, workitem_type: str) -> dict:
    """Transform raw GitHub API issue into work item schema."""
    labels_raw = issue.get("labels", [])
    label_names = [l["name"] for l in labels_raw if isinstance(l, dict)]

    assignees = []
    for a in issue.get("assignees", []):
        if isinstance(a, dict) and a.get("login"):
            assignees.append(a["login"])

    result = {
        "number": issue["number"],
        "title": issue.get("title", ""),
        "body": issue.get("body", "") or "",
        "workitem_type": workitem_type,
        "state": issue.get("state", ""),
        "state_reason": issue.get("state_reason", ""),
        "labels": label_names,
        "feature_areas": extract_feature_areas(labels_raw),
        "milestone": issue.get("milestone", {}).get("title", "") if issue.get("milestone") else "",
        "assignees": assignees,
        "created_at": issue.get("created_at", ""),
        "closed_at": issue.get("closed_at", ""),
        "reactions": {},
        "html_url": issue.get("html_url", ""),
        "comments_count": issue.get("comments", 0),
    }

    raw_reactions = issue.get("reactions", {})
    if isinstance(raw_reactions, dict):
        result["reactions"] = {
            k: v for k, v in raw_reactions.items()
            if k in ("+1", "-1", "laugh", "hooray", "confused", "heart", "rocket", "eyes")
        }

    # Simple parsed body for compatibility
    result["parsed"] = {"tasks": [], "summary": ""}

    return result


def collect_by_label(client: GitHubClient, label: str, workitem_type: str,
                     checkpoint: CheckpointManager, max_results: int = 0,
                     use_date_pagination: bool = False) -> list[dict]:
    """Collect all issues with a specific label."""
    checkpoint_key = f"label_{label}"

    if checkpoint.is_done(checkpoint_key):
        items = checkpoint.get_data(checkpoint_key) or []
        logger.info(f"Skipping {label} (already done, {len(items)} items)")
        return items

    items = []

    if use_date_pagination:
        for year in range(2023, 2026):
            query = f'repo:{config.REPO_OWNER}/{config.REPO_NAME} is:issue label:"{label}" state:closed created:{year}-01-01..{year}-12-31'
            logger.info(f"Searching {label} for {year}...")
            for issue in client.search_issues(query, max_results=max_results):
                if "pull_request" in issue:
                    continue
                item = transform_workitem(issue, workitem_type)
                items.append(item)
                if max_results and len(items) >= max_results:
                    break
            if max_results and len(items) >= max_results:
                break
    else:
        query = f'repo:{config.REPO_OWNER}/{config.REPO_NAME} is:issue label:"{label}" state:closed'
        logger.info(f"Searching all {label}...")
        for issue in client.search_issues(query, max_results=max_results):
            if "pull_request" in issue:
                continue
            item = transform_workitem(issue, workitem_type)
            items.append(item)
            if max_results and len(items) >= max_results:
                break

    checkpoint.mark_done(checkpoint_key, items)
    logger.info(f"  {label}: {len(items)} items collected")
    return items


def collect_workitems(max_results: int = 0):
    """Main collection pipeline for Flutter work items."""
    client = GitHubClient()
    checkpoint = CheckpointManager(config.CHECKPOINT_DIR, "workitems")

    client.print_rate_limit()

    all_items = {}

    # 1. P0 issues (high priority, proxy for iteration plans)
    p0_items = collect_by_label(
        client, "P0", "iteration_plan", checkpoint,
        max_results=max_results,
    )
    for item in p0_items:
        all_items[item["number"]] = item

    # 2. Proposals (proxy for plan items)
    proposals = collect_by_label(
        client, "c: proposal", "plan_item", checkpoint,
        max_results=max_results,
    )
    for item in proposals:
        all_items[item["number"]] = item

    # 3. New features (large volume, use date pagination)
    feature_requests = collect_by_label(
        client, "c: new feature", "feature_request", checkpoint,
        max_results=max_results,
        use_date_pagination=True,
    )
    for item in feature_requests:
        all_items[item["number"]] = item

    # Deduplicate and save
    workitems_list = sorted(all_items.values(), key=lambda x: x["number"])

    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    output_path = os.path.join(config.RAW_DATA_DIR, "workitems.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(workitems_list, f, indent=2, ensure_ascii=False)

    logger.info(f"Collected {len(workitems_list)} unique work items. Saved to {output_path}")

    # Stats by type
    type_counts = {}
    for item in workitems_list:
        t = item["workitem_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    logger.info(f"Work item types: {type_counts}")

    # Milestone distribution
    from collections import Counter
    milestones = Counter(item.get("milestone", "") or "none" for item in workitems_list)
    logger.info(f"Top milestones: {dict(milestones.most_common(10))}")

    return workitems_list


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=0, help="Max items per label (0=all)")
    args = parser.parse_args()
    collect_workitems(max_results=args.max)
