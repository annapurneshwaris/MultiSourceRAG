"""Source 2: Collect Flutter bug reports via GitHub Search API."""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config_flutter as config
from ingestion.collectors.github_client_flutter import GitHubClient
from ingestion.utils.checkpoint import CheckpointManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("collect_flutter_bugs")


def extract_feature_areas(labels: list[dict]) -> list[str]:
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

    # Clean reactions dict
    if isinstance(result["reactions"], dict):
        result["reactions"] = {
            k: v for k, v in result["reactions"].items()
            if k in ("+1", "-1", "laugh", "hooray", "confused", "heart", "rocket", "eyes")
        }

    # Flutter bugs don't have the same structured body as VS Code
    # Simple parsed dict for compatibility
    result["parsed"] = _parse_flutter_bug_body(result["body"])

    return result


def _parse_flutter_bug_body(body: str) -> dict:
    """Simple body parser for Flutter bug reports."""
    parsed = {
        "steps_to_reproduce": "",
        "expected_results": "",
        "actual_results": "",
        "error_messages": [],
        "vscode_version": "",  # Not applicable but kept for schema compat
        "os_version": "",
        "flutter_version": "",
    }

    if not body:
        return parsed

    # Flutter bug template sections
    sections = {}
    current_section = ""
    current_lines = []

    for line in body.split("\n"):
        line_lower = line.strip().lower()
        # Detect section headers
        if line_lower.startswith("## steps to reproduce") or line_lower.startswith("**steps to reproduce**"):
            if current_section:
                sections[current_section] = "\n".join(current_lines)
            current_section = "steps"
            current_lines = []
        elif line_lower.startswith("## expected results") or line_lower.startswith("**expected results**"):
            if current_section:
                sections[current_section] = "\n".join(current_lines)
            current_section = "expected"
            current_lines = []
        elif line_lower.startswith("## actual results") or line_lower.startswith("**actual results**"):
            if current_section:
                sections[current_section] = "\n".join(current_lines)
            current_section = "actual"
            current_lines = []
        elif line_lower.startswith("## logs") or line_lower.startswith("**logs**"):
            if current_section:
                sections[current_section] = "\n".join(current_lines)
            current_section = "logs"
            current_lines = []
        else:
            current_lines.append(line)

    if current_section:
        sections[current_section] = "\n".join(current_lines)

    parsed["steps_to_reproduce"] = sections.get("steps", "")
    parsed["expected_results"] = sections.get("expected", "")
    parsed["actual_results"] = sections.get("actual", "")

    # Extract flutter version from body
    import re
    ver_match = re.search(r'Flutter\s+(\d+\.\d+\.\d+)', body)
    if ver_match:
        parsed["flutter_version"] = ver_match.group(1)

    # Extract error messages
    error_lines = []
    for line in body.split("\n"):
        line_strip = line.strip()
        if any(kw in line_strip.lower() for kw in ["error:", "exception:", "traceback", "════"]):
            error_lines.append(line_strip)
            if len(error_lines) >= 5:
                break
    parsed["error_messages"] = error_lines

    return parsed


def collect_bugs(max_results: int = 0, date_start: str = "", date_end: str = ""):
    """Main collection pipeline for Flutter bug reports."""
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

    # Use provided date range or default monthly ranges
    if date_start and date_end:
        date_ranges = [{"start": date_start, "end": date_end, "label": f"{date_start}_to_{date_end}"}]
    else:
        date_ranges = config.BUG_DATE_RANGES

    for date_range in date_ranges:
        month_label = date_range["label"]

        if checkpoint.is_done(month_label):
            logger.info(f"Skipping {month_label} (already done)")
            continue

        query = f'repo:{config.REPO_OWNER}/{config.REPO_NAME} is:issue label:"has reproducible steps" state:closed created:{date_range["start"]}..{date_range["end"]}'
        logger.info(f"Searching: {month_label} ({date_range['start']} to {date_range['end']})")

        month_bugs = []
        for issue in client.search_issues(query, max_results=max_results):
            if "pull_request" in issue:
                continue

            bug = transform_issue(issue)
            month_bugs.append(bug)
            all_bugs[bug["number"]] = bug

            if max_results and len(all_bugs) >= max_results:
                break

        checkpoint.mark_done(month_label, month_bugs)
        logger.info(f"  {month_label}: {len(month_bugs)} bugs (total: {len(all_bugs)})")

        if max_results and len(all_bugs) >= max_results:
            break

    # Deduplicate and save
    bugs_list = sorted(all_bugs.values(), key=lambda x: x["number"])

    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    output_path = os.path.join(config.RAW_DATA_DIR, "bugs.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bugs_list, f, indent=2, ensure_ascii=False)

    logger.info(f"Collected {len(bugs_list)} unique bugs. Saved to {output_path}")

    # Stats
    from collections import Counter
    label_counts = Counter()
    team_count = 0
    for bug in bugs_list:
        label_counts.update(bug["labels"])
        if bug.get("author_association") in ("MEMBER", "COLLABORATOR"):
            team_count += 1

    logger.info(f"Top 10 labels: {dict(label_counts.most_common(10))}")
    logger.info(f"Team authored: {team_count}/{len(bugs_list)}")

    return bugs_list


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=0, help="Max bugs to collect (0=all)")
    parser.add_argument("--start", type=str, default="", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="", help="End date YYYY-MM-DD")
    args = parser.parse_args()
    collect_bugs(max_results=args.max, date_start=args.start, date_end=args.end)
