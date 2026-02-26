"""Feature-area normalization — maps ~60+ GitHub labels to ~20 normalized areas.

This is THE cross-source linking mechanism. Feature-area is the only field
that exists naturally across all three sources (docs Area, bugs/work labels).
"""

# Maps GitHub labels → normalized areas
FEATURE_AREA_MAP: dict[str, str] = {
    # Editor
    "editor-core": "editor",
    "editor-find": "editor",
    "editor-autocomplete": "editor",
    "editor-folding": "editor",
    "editor-hover": "editor",
    "editor-minimap": "editor",
    "editor-bracket-matching": "editor",
    "editor-color-picker": "editor",
    "editor-diff": "editor",
    "editor-indent-guides": "editor",
    "editor-merge-conflicts": "editor",
    "editor-sticky-scroll": "editor",
    "editor-symbols": "editor",
    "editor-wrapping": "editor",
    "editor-code-actions": "editor",
    "editor-commands": "editor",
    "editor-indent": "editor",
    "editor-inlay-hints": "editor",
    "editor-linked-editing": "editor",
    "editor-multicursor": "editor",
    "editor-rename": "editor",
    "editor-gpu": "editor",
    "diff-editor": "editor",
    "merge-editor": "editor",

    # Terminal
    "terminal": "terminal",
    "terminal-local": "terminal",
    "terminal-remote": "terminal",
    "terminal-shell-integration": "terminal",
    "terminal-external": "terminal",
    "terminal-profiles": "terminal",
    "terminal-tabs": "terminal",
    "terminal-suggest": "terminal",
    "integrated-terminal": "terminal",

    # Debug
    "debug": "debug",
    "debug-console": "debug",

    # Git / SCM
    "scm": "git",
    "git": "git",
    "github": "git",
    "source-control": "git",

    # Remote
    "remote": "remote",
    "remote-explorer": "remote",
    "wsl": "remote",
    "ssh": "remote",
    "dev-container": "remote",
    "devcontainer": "remote",
    "tunnels": "remote",

    # Search
    "search": "search",
    "search-replace": "search",

    # Notebook
    "notebook": "notebook",
    "notebook-output": "notebook",
    "notebook-kernel": "notebook",

    # Testing
    "testing": "testing",
    "test-cli": "testing",

    # Extensions
    "extensions": "extensions",
    "extension-host": "extensions",
    "marketplace": "extensions",

    # Settings / Config
    "settings": "settings",
    "keybindings": "settings",
    "settings-sync": "settings",

    # Workbench
    "workbench-editors": "workbench",
    "workbench-views": "workbench",
    "workbench-window": "workbench",
    "workbench-layout": "workbench",
    "workbench-status-bar": "workbench",
    "workbench-tabs": "workbench",
    "workbench-activity-bar": "workbench",
    "workbench-state": "workbench",
    "workbench-hot-exit": "workbench",
    "layout": "workbench",
    "panel": "workbench",
    "sidebar": "workbench",
    "statusbar": "workbench",
    "titlebar": "workbench",

    # Languages
    "languages": "languages",
    "languages-basic": "languages",
    "languages-typescript": "languages",
    "languages-json": "languages",
    "languages-html": "languages",
    "languages-css": "languages",
    "languages-markdown": "languages",
    "typescript": "languages",
    "javascript": "languages",
    "python": "languages",
    "java": "languages",
    "c++": "languages",
    "markdown": "languages",
    "json": "languages",
    "html": "languages",
    "css": "languages",

    # Tasks
    "tasks": "tasks",

    # Accessibility
    "accessibility": "accessibility",

    # Copilot / AI
    "copilot": "copilot",
    "chat": "copilot",
    "inline-chat": "copilot",
    "chat-terminal": "copilot",
    "chat-prompts": "copilot",

    # Comments
    "comments": "comments",

    # File explorer
    "explorer-fileoperations": "explorer",
    "file-explorer": "explorer",
    "explorer": "explorer",

    # Performance
    "performance": "performance",
    "freeze-slow-crash-leak": "performance",
    "electron": "performance",
}

# Maps docs Area YAML field → normalized area
DOC_AREA_MAP: dict[str, str] = {
    "copilot": "copilot",
    "editor": "editor",
    "editing": "editor",
    "terminal": "terminal",
    "debugging": "debug",
    "debugtest": "debug",
    "sourcecontrol": "git",
    "remote": "remote",
    "devcontainers": "remote",
    "containers": "remote",
    "search": "search",
    "notebooks": "notebook",
    "datascience": "notebook",
    "testing": "testing",
    "extensions": "extensions",
    "extension-guides": "extensions",
    "extension-capabilities": "extensions",
    "language-extensions": "extensions",
    "working-with-extensions": "extensions",
    "configure": "settings",
    "customization": "settings",
    "languages": "languages",
    "cpp": "languages",
    "csharp": "languages",
    "java": "languages",
    "python": "languages",
    "typescript": "languages",
    "nodejs": "languages",
    "intelligentapps": "copilot",
    "azure": "azure",
    "setup": "setup",
    "getstarted": "setup",
    "get-started": "setup",
    "introvideos": "setup",
    "references": "reference",
    "reference": "reference",
    "advanced-topics": "advanced",
    "supporting": "reference",
    "ux-guidelines": "workbench",
    "enterprise": "enterprise",
    "api": "extensions",
    "other": "other",
}


def extract_feature_area(labels: list[str], source_type: str = "bug") -> str:
    """Return best matching normalized area, or 'unknown'.

    For docs: use DOC_AREA_MAP on the Area YAML field.
    For bugs/work_items: scan labels list against FEATURE_AREA_MAP.
    """
    if source_type == "doc":
        for label in labels:
            normalized = label.lower().strip()
            if normalized in DOC_AREA_MAP:
                return DOC_AREA_MAP[normalized]
        return "unknown"

    # Bugs and work items: scan GitHub labels
    for label in labels:
        label_lower = label.lower().strip()
        if label_lower in FEATURE_AREA_MAP:
            return FEATURE_AREA_MAP[label_lower]

    # Prefix matching fallback
    for label in labels:
        label_lower = label.lower().strip()
        for key, area in FEATURE_AREA_MAP.items():
            prefix = key.split("-")[0]
            if label_lower.startswith(prefix) and len(prefix) >= 3:
                return area

    return "unknown"
