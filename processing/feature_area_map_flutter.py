"""Feature-area normalization for Flutter — maps GitHub labels to normalized areas.

Flutter label patterns:
- 'a: ...' = feature area
- 'f: ...' = framework area
- 'platform-...' = platform
- 'engine'/'framework'/'tool' = top-level
- 'p: ...' = package
- 'c: ...' = category
- 'e: ...' = engine sub-area
- 't: ...' = tool sub-area
"""

# Maps Flutter GitHub labels -> normalized areas
FEATURE_AREA_MAP: dict[str, str] = {
    # Feature area labels (a: ...)
    "a: text input": "text_input",
    "a: accessibility": "accessibility",
    "a: animation": "animation",
    "a: desktop": "desktop",
    "a: images": "images",
    "a: internationalization": "i18n",
    "a: mouse": "input",
    "a: platform-views": "platform_views",
    "a: quality": "quality",
    "a: tests": "testing",
    "a: annoyance": "quality",
    "a: browser": "web",
    "a: existing-apps": "integration",
    "a: fidelity": "quality",
    "a: plugins": "plugins",
    "a: size": "performance",
    "a: state management": "state_management",
    "a: typography": "text_input",

    # Framework area labels (f: ...)
    "f: material design": "material",
    "f: cupertino": "cupertino",
    "f: scrolling": "scrolling",
    "f: routes": "navigation",
    "f: gestures": "gestures",
    "f: focus": "focus",

    # Platform labels
    "platform-android": "android",
    "platform-ios": "ios",
    "platform-web": "web",
    "platform-windows": "windows",
    "platform-linux": "linux",
    "platform-macos": "macos",

    # Top-level areas
    "engine": "engine",
    "framework": "framework",
    "tool": "tooling",

    # Engine sub-areas
    "e: impeller": "engine",
    "e: dart": "dart",

    # Tool sub-areas
    "t: gradle": "tooling",
    "t: xcode": "tooling",

    # Category labels (c: ...)
    "c: performance": "performance",
    "c: crash": "crash",
    "c: regression": "regression",
    "c: new feature": "feature",
    "c: proposal": "feature",

    # Package labels (p: ...)
    "p: camera": "plugins",
    "p: webview": "plugins",
    "p: video_player": "plugins",
    "p: share": "plugins",
    "p: url_launcher": "plugins",
    "p: path_provider": "plugins",

    # Priority labels (used as work item types)
    "P0": "priority",
    "P1": "priority",
    "P2": "priority",

    # Type labels
    "type: bug": "bug",
    "type: feature request": "feature",
}

# Maps Flutter docs directory -> normalized area
DOC_AREA_MAP: dict[str, str] = {
    "ui": "ui",
    "testing": "testing",
    "data-and-backend": "data",
    "platform-integration": "platform_views",
    "packages-and-plugins": "plugins",
    "development": "development",
    "tools": "tooling",
    "perf": "performance",
    "deployment": "deployment",
    "accessibility-and-internationalization": "i18n",
    "add-to-app": "integration",
    "get-started": "setup",
    "release": "release",
    "cookbook": "cookbook",
    "codelabs": "setup",
    "resources": "reference",
    "reference": "reference",
    "ai": "ai",
    "app-architecture": "architecture",
    "assets": "ui",
    "navigation": "navigation",
    "design": "ui",
    "animation": "animation",
    "interactivity": "ui",
    "layout": "ui",
    "widgets": "ui",
    "security": "security",
    "gaming": "gaming",
    "news": "reference",
    "dash": "reference",
}

# All Flutter label names for matching
FEATURE_AREA_LABELS = list(FEATURE_AREA_MAP.keys())


def extract_feature_area(labels: list[str], source_type: str = "bug") -> str:
    """Return best matching normalized area, or 'unknown'.

    For docs: use DOC_AREA_MAP on the directory-derived area.
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

    # Prefix matching fallback for 'a:', 'f:', 'platform-' patterns
    for label in labels:
        label_lower = label.lower().strip()
        if label_lower.startswith("a: "):
            return "other_feature"
        if label_lower.startswith("f: "):
            return "other_framework"
        if label_lower.startswith("platform-"):
            return "other_platform"
        if label_lower.startswith("p: "):
            return "plugins"

    return "unknown"
