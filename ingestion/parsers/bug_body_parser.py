"""Parse structured fields from VS Code bug report issue bodies."""

import re


# VS Code version: "Version: 1.85.1" or "VS Code Version: 1.85.1"
VSCODE_VERSION_PATTERNS = [
    re.compile(r'\*{0,2}(?:VS\s*Code\s+)?Version:?\*{0,2}\s*([\d.]+(?:-\w+)?)', re.IGNORECASE),
    re.compile(r'(?:VS\s*Code|Code(?:\s*-\s*OSS)?)\s+([\d]+\.[\d]+\.[\d]+)', re.IGNORECASE),
]

# OS version
OS_VERSION_PATTERNS = [
    re.compile(r'OS\s+Version:\s*(.+?)(?:\n|$)', re.IGNORECASE),
    re.compile(r'Operating\s+System:\s*(.+?)(?:\n|$)', re.IGNORECASE),
    # Common OS patterns in freeform text
    re.compile(r'(Windows\s+\d+(?:\s+\w+)?(?:\s+[\d.]+)?)', re.IGNORECASE),
    re.compile(r'(macOS\s+[\d.]+(?:\s+\w+)?)', re.IGNORECASE),
    re.compile(r'(Ubuntu\s+[\d.]+)', re.IGNORECASE),
    re.compile(r'(Linux\s+[\w\d.-]+)', re.IGNORECASE),
]

# Steps to reproduce — text between section header and next section
STEPS_PATTERN = re.compile(
    r'(?:Steps\s+to\s+Reproduce|Reproduction\s+Steps|How\s+to\s+reproduce)[:\s]*\n(.*?)(?=\n(?:##|Expected|Actual|Does this issue occur|---)|$)',
    re.IGNORECASE | re.DOTALL,
)

# Error messages — lines with Error:/Exception: or inside code blocks with stack traces
ERROR_LINE_PATTERN = re.compile(r'^.*(?:Error|Exception|ENOENT|EPIPE|EACCES|TypeError|ReferenceError|SyntaxError)[:\s].+$', re.MULTILINE | re.IGNORECASE)
ERROR_BLOCK_PATTERN = re.compile(r'```(?:\w*)\n(.*?(?:Error|Exception|Traceback|stack trace).*?)```', re.DOTALL | re.IGNORECASE)

# Extensions list
EXTENSIONS_PATTERN = re.compile(
    r'(?:Extensions|Extension\s+list|Installed\s+extensions)[:\s]*\n(.*?)(?=\n(?:##|---)|$)',
    re.IGNORECASE | re.DOTALL,
)
EXTENSION_NAME_PATTERN = re.compile(r'[\w.-]+\.[\w.-]+')


def parse_vscode_version(body: str) -> str:
    """Extract VS Code version from bug report body."""
    if not body:
        return ""
    for pattern in VSCODE_VERSION_PATTERNS:
        match = pattern.search(body)
        if match:
            return match.group(1).strip()
    return ""


def parse_os_version(body: str) -> str:
    """Extract OS version from bug report body."""
    if not body:
        return ""
    for pattern in OS_VERSION_PATTERNS:
        match = pattern.search(body)
        if match:
            return match.group(1).strip()
    return ""


def parse_steps_to_reproduce(body: str) -> str:
    """Extract steps to reproduce section."""
    if not body:
        return ""
    match = STEPS_PATTERN.search(body)
    if match:
        return match.group(1).strip()
    return ""


def parse_error_messages(body: str) -> list[str]:
    """Extract error messages and stack traces from body."""
    if not body:
        return []

    errors = set()

    # From code blocks containing errors
    for match in ERROR_BLOCK_PATTERN.finditer(body):
        block = match.group(1).strip()
        if len(block) < 2000:  # Skip overly long blocks
            errors.add(block)

    # Individual error lines
    for match in ERROR_LINE_PATTERN.finditer(body):
        line = match.group(0).strip()
        if len(line) < 500:
            errors.add(line)

    return list(errors)[:10]  # Cap at 10 to avoid noise


def parse_extensions(body: str) -> list[str]:
    """Extract list of installed extensions."""
    if not body:
        return []
    match = EXTENSIONS_PATTERN.search(body)
    if match:
        section = match.group(1)
        # Look for extension IDs like "publisher.extension-name"
        extensions = EXTENSION_NAME_PATTERN.findall(section)
        return list(set(extensions))[:50]
    return []


def parse_bug_body(body: str) -> dict:
    """Parse all structured fields from a bug report body."""
    return {
        "vscode_version": parse_vscode_version(body),
        "os_version": parse_os_version(body),
        "steps_to_reproduce": parse_steps_to_reproduce(body),
        "error_messages": parse_error_messages(body),
        "extensions_list": parse_extensions(body),
    }
