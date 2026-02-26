"""PII anonymization for collected data."""

import re


# Patterns to detect and redact
EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
USER_PATH_PATTERNS = [
    # Windows: C:\Users\JohnDoe\...
    re.compile(r'[A-Z]:\\Users\\[^\\]+', re.IGNORECASE),
    # macOS: /Users/johndoe/...
    re.compile(r'/Users/[^/\s]+'),
    # Linux: /home/johndoe/...
    re.compile(r'/home/[^/\s]+'),
]


def anonymize_text(text: str) -> str:
    """Redact PII from text content."""
    if not text:
        return text

    # Redact email addresses
    text = EMAIL_PATTERN.sub('[EMAIL_REDACTED]', text)

    # Normalize user-specific file paths
    for pattern in USER_PATH_PATTERNS:
        text = pattern.sub(lambda m: _normalize_path(m.group()), text)

    return text


def _normalize_path(path: str) -> str:
    """Replace username portion of file path."""
    if '\\Users\\' in path:
        parts = path.split('\\')
        idx = parts.index('Users')
        parts[idx + 1] = '[USER]'
        return '\\'.join(parts)
    elif '/Users/' in path:
        parts = path.split('/')
        idx = parts.index('Users')
        parts[idx + 1] = '[USER]'
        return '/'.join(parts)
    elif '/home/' in path:
        parts = path.split('/')
        idx = parts.index('home')
        parts[idx + 1] = '[USER]'
        return '/'.join(parts)
    return path


def anonymize_issue(issue: dict) -> dict:
    """Anonymize PII fields in an issue object (mutates in place)."""
    if issue.get("body"):
        issue["body"] = anonymize_text(issue["body"])
    if issue.get("title"):
        issue["title"] = anonymize_text(issue["title"])

    # Anonymize parsed fields if present
    parsed = issue.get("parsed", {})
    if parsed.get("steps_to_reproduce"):
        parsed["steps_to_reproduce"] = anonymize_text(parsed["steps_to_reproduce"])
    if parsed.get("error_messages"):
        parsed["error_messages"] = [anonymize_text(m) for m in parsed["error_messages"]]

    return issue


def anonymize_comment(comment: dict) -> dict:
    """Anonymize PII in a comment object (mutates in place)."""
    if comment.get("comment_body"):
        comment["comment_body"] = anonymize_text(comment["comment_body"])
    return comment
