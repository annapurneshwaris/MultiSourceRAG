"""Parse structured fields from VS Code work item issue bodies (iteration plans, plan items, feature requests)."""

import re


# Task checkboxes: - [x] Done task or - [ ] Pending task
CHECKBOX_PATTERN = re.compile(r'^[\s]*-\s+\[([ xX])\]\s+(.+)$', re.MULTILINE)

# Status emojis used in VS Code iteration plans
STATUS_EMOJIS = {
    '\U0001f3c3': 'in_progress',   # 🏃 runner
    '\U0001f4aa': 'done',          # 💪 flexed bicep
    '\u270b': 'blocked',           # ✋ raised hand
    '\U0001f44d': 'approved',      # 👍 thumbs up
    '\u2705': 'completed',         # ✅ check mark
    '\u274c': 'cancelled',         # ❌ cross mark
    '\U0001f51c': 'soon',          # 🔜 soon arrow
    '\u23f3': 'waiting',           # ⏳ hourglass
}

# Cross-referenced issues: #12345
ISSUE_REF_PATTERN = re.compile(r'(?<!\w)#(\d{3,6})(?!\w)')

# @username mentions
USERNAME_PATTERN = re.compile(r'(?<!\w)@([\w-]+)(?!\w)')


def parse_task_checkboxes(body: str) -> list[dict]:
    """Extract task checkbox items with their completion status."""
    if not body:
        return []

    checkboxes = []
    for match in CHECKBOX_PATTERN.finditer(body):
        checked = match.group(1).lower() == 'x'
        text = match.group(2).strip()
        # Clean up common prefixes/noise
        text = re.sub(r'^~(.+)~$', r'\1', text)  # Strikethrough
        if text and len(text) > 2:
            checkboxes.append({
                "text": text,
                "checked": checked,
            })
    return checkboxes


def parse_status_emojis(body: str) -> dict:
    """Count status emojis used in iteration plans."""
    if not body:
        return {}

    counts = {}
    for emoji, status in STATUS_EMOJIS.items():
        count = body.count(emoji)
        if count > 0:
            counts[status] = counts.get(status, 0) + count

    return counts


def parse_cross_referenced_issues(body: str) -> list[int]:
    """Extract referenced issue numbers (#NNNNN)."""
    if not body:
        return []

    refs = set()
    for match in ISSUE_REF_PATTERN.finditer(body):
        num = int(match.group(1))
        if num > 100:  # Filter out likely non-issue references
            refs.add(num)
    return sorted(refs)


def parse_assigned_developers(body: str) -> list[str]:
    """Extract @mentioned developers from work item body."""
    if not body:
        return []

    usernames = set()
    for match in USERNAME_PATTERN.finditer(body):
        username = match.group(1)
        # Filter out common non-person mentions
        if username.lower() not in ('github', 'dependabot', 'vscode', 'microsoft', 'here', 'param'):
            usernames.add(username)
    return sorted(usernames)


def parse_workitem_body(body: str) -> dict:
    """Parse all structured fields from a work item body."""
    return {
        "task_checkboxes": parse_task_checkboxes(body),
        "status_emojis": parse_status_emojis(body),
        "cross_referenced_issues": parse_cross_referenced_issues(body),
        "assigned_developers": parse_assigned_developers(body),
    }
