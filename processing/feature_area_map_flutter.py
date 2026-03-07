"""Feature-area normalization for Flutter — maps GitHub labels to normalized areas.

Flutter label patterns:
- 'a: ...' = feature area (MOST SPECIFIC — check first)
- 'f: ...' = framework area (SPECIFIC)
- 'platform-...' = platform (SPECIFIC)
- 'e: ...' = engine sub-area
- 'p: ...' = package
- 'team-...' = team ownership (FALLBACK — use as area proxy)
- 'c: ...' = category (GENERAL — check last)
- 'engine'/'framework'/'tool' = top-level

Priority: specific area (a:/f:/platform-/e:/p:) > team proxy > top-level > category
Skip non-area labels: P0-P6, r:*, triaged-*, d:*, infra:*
"""

# --- TIER 1: Specific area labels (highest priority) ---
# These directly identify what the issue is about

_TIER1_AREA_MAP: dict[str, str] = {
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
    "a: build": "tooling",
    "a: leak tracking": "testing",
    "a: debugging": "debugging",
    "a: error message": "debugging",
    "a: first hour": "setup",
    "a: widget previews": "ui",

    # Framework area labels (f: ...)
    "f: material design": "material",
    "f: cupertino": "cupertino",
    "f: scrolling": "scrolling",
    "f: routes": "navigation",
    "f: gestures": "gestures",
    "f: focus": "focus",
    "f: theming": "material",

    # Platform labels
    "platform-android": "android",
    "platform-ios": "ios",
    "platform-web": "web",
    "platform-windows": "windows",
    "platform-linux": "linux",
    "platform-macos": "macos",
    "platform-mac": "macos",

    # Engine sub-areas
    "e: impeller": "engine",
    "e: dart": "dart",
    "e: wasm": "web",
    "e: web_canvaskit": "web",
    "e: local-engine-development": "engine",
    "flutter-gpu": "engine",

    # Package labels (p: ...) — all map to plugins
    "p: camera": "plugins",
    "p: webview": "plugins",
    "p: video_player": "plugins",
    "p: share": "plugins",
    "p: url_launcher": "plugins",
    "p: path_provider": "plugins",
    "p: go_router": "navigation",
    "p: go_router_builder": "navigation",
    "p: two_dimensional_scrollables": "scrolling",
    "p: flutter_markdown": "ui",
    "p: google_fonts": "ui",
    "p: pigeon": "plugins",
    "p: in_app_purchase": "plugins",
    "p: shared_preferences": "plugins",
    "p: rfw": "ui",
    "p: interactive_media_ads": "plugins",
    "p: web_benchmarks": "performance",

    # Tool sub-areas
    "t: gradle": "tooling",
    "t: xcode": "tooling",
}

# --- TIER 2: Team ownership (use as area proxy when no Tier 1 match) ---

_TIER2_TEAM_MAP: dict[str, str] = {
    "team-design": "material",
    "team-framework": "framework",
    "team-engine": "engine",
    "team-tool": "tooling",
    "team-web": "web",
    "team-ios": "ios",
    "team-android": "android",
    "team-ecosystem": "plugins",
    "team-infra": "infra",
    "team-accessibility": "accessibility",
    "team-text-input": "text_input",
    "team-windows": "windows",
    "team-macos": "macos",
    "team-linux": "linux",
    "team-devexp": "tooling",
}

# --- TIER 3: Top-level and category (lowest priority) ---

_TIER3_GENERAL_MAP: dict[str, str] = {
    "engine": "engine",
    "framework": "framework",
    "tool": "tooling",
    "package": "plugins",

    # Category labels (c: ...)
    "c: performance": "performance",
    "c: crash": "crash",
    "c: regression": "regression",
    "c: new feature": "feature",
    "c: proposal": "feature",
    "c: flake": "testing",
    "c: tech-debt": "maintenance",
    "c: contributor-productivity": "maintenance",
    "c: rendering": "engine",
    "c: new widget": "ui",

    # Documentation labels
    "d: api docs": "documentation",
    "d: examples": "documentation",
    "d: docs/": "documentation",

    # Type labels
    "type: bug": "bug",
    "type: feature request": "feature",

    # Dependency
    "dependency: dart": "dart",
}

# Labels to SKIP (not useful for area classification)
_SKIP_LABELS = {
    "p0", "p1", "p2", "p3", "p4", "p5", "p6",
    "r: fixed", "r: solved", "r: duplicate", "r: invalid",
    "has reproducible steps", "good first issue",
    "waiting for customer response", "workaround available",
    "waiting for pr to land (fixed)", "has partial patch",
    "design doc", "refactor",
    ":hourglass_flowing_sand:", ":scroll:", ":star:",
    "design systems study",
}

# Combined map (all lowercase) for FEATURE_AREA_LABELS compatibility
FEATURE_AREA_MAP: dict[str, str] = {}
for _m in [_TIER1_AREA_MAP, _TIER2_TEAM_MAP, _TIER3_GENERAL_MAP]:
    for k, v in _m.items():
        FEATURE_AREA_MAP[k.lower()] = v

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
    For bugs/work_items: 3-tier priority scan:
        Tier 1: specific area (a:/f:/platform-/e:/p:) — most informative
        Tier 2: team ownership (team-*) — good proxy for area
        Tier 3: top-level / category (framework/engine/c:*) — fallback
    """
    if source_type == "doc":
        for label in labels:
            normalized = label.lower().strip()
            if normalized in DOC_AREA_MAP:
                return DOC_AREA_MAP[normalized]
        return "unknown"

    # Normalize all labels to lowercase once
    labels_lower = [l.lower().strip() for l in labels]

    # Skip non-area labels
    labels_lower = [l for l in labels_lower if l not in _SKIP_LABELS
                    and not l.startswith("triaged-")
                    and not l.startswith("fyi-")
                    and not l.startswith("infra:")
                    and not l.startswith("found in release:")
                    and not l.startswith("customer:")]

    # Tier 1: Specific area labels (a:, f:, platform-, e:, p:, t:)
    for label in labels_lower:
        if label in _TIER1_AREA_MAP:
            return _TIER1_AREA_MAP[label]

    # Tier 1 fallback: prefix matching for unmapped specific labels
    for label in labels_lower:
        if label.startswith("a: "):
            return "other_feature"
        if label.startswith("f: "):
            return "other_framework"
        if label.startswith("platform-"):
            return "other_platform"
        if label.startswith("p: "):
            return "plugins"
        if label.startswith("e: "):
            return "engine"
        if label.startswith("t: "):
            return "tooling"

    # Tier 2: Team labels as area proxy
    for label in labels_lower:
        if label in _TIER2_TEAM_MAP:
            return _TIER2_TEAM_MAP[label]

    # Tier 3: Top-level / category
    for label in labels_lower:
        if label in _TIER3_GENERAL_MAP:
            return _TIER3_GENERAL_MAP[label]

    return "unknown"
