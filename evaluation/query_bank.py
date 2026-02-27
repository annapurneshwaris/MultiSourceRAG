"""Evaluation query bank — 250 curated queries across 5 categories.

Categories (per v4 spec):
- how_to (60): Configuration, setup, feature usage
- debugging (75): Error diagnosis, crash investigation
- error_diagnosis (50): Specific error messages, stack traces
- status_roadmap (40): Feature plans, milestones, release dates
- config (25): Settings, keybindings, workspace config
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalQuery:
    """A single evaluation query."""
    query_id: str
    query_text: str
    category: str  # how_to, debugging, error_diagnosis, status_roadmap, config
    expected_sources: list[str]  # ["doc", "bug", "work_item"]
    expected_area: str  # Normalized feature area
    difficulty: str  # easy, medium, hard
    ground_truth_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "category": self.category,
            "expected_sources": self.expected_sources,
            "expected_area": self.expected_area,
            "difficulty": self.difficulty,
            "ground_truth_notes": self.ground_truth_notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EvalQuery:
        return cls(**d)


# 250 seed queries across 5 categories
SEED_QUERIES: list[dict] = [
    # =========================================================================
    # how_to (60 queries, 24%)
    # =========================================================================
    {"query_id": "ht_001", "query_text": "How do I configure the integrated terminal to use zsh on macOS?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "terminal", "difficulty": "easy"},
    {"query_id": "ht_002", "query_text": "How to set up remote SSH development in VS Code?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "remote", "difficulty": "easy"},
    {"query_id": "ht_003", "query_text": "How do I enable bracket pair colorization?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "ht_004", "query_text": "How to configure GitHub Copilot inline suggestions?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "copilot", "difficulty": "medium"},
    {"query_id": "ht_005", "query_text": "How do I use multi-cursor editing in VS Code?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "ht_006", "query_text": "How to set up a dev container for Python?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "remote", "difficulty": "medium"},
    {"query_id": "ht_007", "query_text": "How do I customize the sidebar and panel layout?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "workbench", "difficulty": "easy"},
    {"query_id": "ht_008", "query_text": "How to configure workspace-specific settings?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "ht_009", "query_text": "How do I set up debugging for a Node.js application?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "debug", "difficulty": "medium"},
    {"query_id": "ht_010", "query_text": "How to use Jupyter notebooks in VS Code?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "notebook", "difficulty": "medium"},
    {"query_id": "ht_011", "query_text": "How to configure VS Code to use a custom Python interpreter?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "languages", "difficulty": "easy"},
    {"query_id": "ht_012", "query_text": "How do I enable word wrap in the editor?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "ht_013", "query_text": "How to set up Git in VS Code for the first time?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "git", "difficulty": "easy"},
    {"query_id": "ht_014", "query_text": "How do I use the built-in diff editor?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "editor", "difficulty": "medium"},
    {"query_id": "ht_015", "query_text": "How to configure tasks for building a C++ project?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "tasks", "difficulty": "medium"},
    {"query_id": "ht_016", "query_text": "How do I install and manage extensions in VS Code?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "extensions", "difficulty": "easy"},
    {"query_id": "ht_017", "query_text": "How to use the timeline view to see file history?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "workbench", "difficulty": "medium"},
    {"query_id": "ht_018", "query_text": "How do I configure the minimap in the editor?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "ht_019", "query_text": "How to set up WSL integration with VS Code?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "remote", "difficulty": "medium"},
    {"query_id": "ht_020", "query_text": "How do I create custom code snippets?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "editor", "difficulty": "medium"},
    {"query_id": "ht_021", "query_text": "How to use the problems panel to view errors and warnings?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "workbench", "difficulty": "easy"},
    {"query_id": "ht_022", "query_text": "How do I configure proxy settings for VS Code?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "medium"},
    {"query_id": "ht_023", "query_text": "How to use the Source Control panel for staging commits?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "git", "difficulty": "easy"},
    {"query_id": "ht_024", "query_text": "How do I set up launch configurations for debugging?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "debug", "difficulty": "medium"},
    {"query_id": "ht_025", "query_text": "How to enable Emmet in HTML files?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "languages", "difficulty": "easy"},
    {"query_id": "ht_026", "query_text": "How do I use the command palette effectively?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "workbench", "difficulty": "easy"},
    {"query_id": "ht_027", "query_text": "How to set up a remote tunnel connection to VS Code?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "remote", "difficulty": "hard"},
    {"query_id": "ht_028", "query_text": "How do I configure live share for pair programming?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "extensions", "difficulty": "medium"},
    {"query_id": "ht_029", "query_text": "How to use the testing framework integration in VS Code?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "testing", "difficulty": "medium"},
    {"query_id": "ht_030", "query_text": "How do I configure editor font ligatures?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "ht_031", "query_text": "How to use the rename symbol feature across files?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "ht_032", "query_text": "How do I set up TypeScript project references in VS Code?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "languages", "difficulty": "hard"},
    {"query_id": "ht_033", "query_text": "How to configure file exclusion patterns for search?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "search", "difficulty": "easy"},
    {"query_id": "ht_034", "query_text": "How do I use breadcrumbs navigation in the editor?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "ht_035", "query_text": "How to configure the integrated terminal split view?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "terminal", "difficulty": "easy"},
    {"query_id": "ht_036", "query_text": "How do I use Git stash in VS Code?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "git", "difficulty": "medium"},
    {"query_id": "ht_037", "query_text": "How to enable and configure indent guides?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "ht_038", "query_text": "How do I use the merge editor for resolving Git conflicts?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "git", "difficulty": "medium"},
    {"query_id": "ht_039", "query_text": "How to configure the output panel for build tasks?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "tasks", "difficulty": "medium"},
    {"query_id": "ht_040", "query_text": "How do I use the peek definition feature?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "ht_041", "query_text": "How to set up Python virtual environments in VS Code?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "languages", "difficulty": "medium"},
    {"query_id": "ht_042", "query_text": "How do I configure the editor tab behavior?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "workbench", "difficulty": "easy"},
    {"query_id": "ht_043", "query_text": "How to use the refactoring features in VS Code?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "editor", "difficulty": "medium"},
    {"query_id": "ht_044", "query_text": "How do I use the interactive rebase editor for Git?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "git", "difficulty": "hard"},
    {"query_id": "ht_045", "query_text": "How to configure accessibility features in VS Code?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "accessibility", "difficulty": "medium"},
    {"query_id": "ht_046", "query_text": "How do I set up Docker development with VS Code?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "remote", "difficulty": "medium"},
    {"query_id": "ht_047", "query_text": "How to use the outline view for code navigation?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "workbench", "difficulty": "easy"},
    {"query_id": "ht_048", "query_text": "How do I configure the integrated terminal shell arguments?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "terminal", "difficulty": "medium"},
    {"query_id": "ht_049", "query_text": "How to use workspace trust and manage trusted folders?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "workbench", "difficulty": "medium"},
    {"query_id": "ht_050", "query_text": "How do I configure formatting on save?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "ht_051", "query_text": "How to use the GitHub Pull Requests extension in VS Code?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "git", "difficulty": "medium"},
    {"query_id": "ht_052", "query_text": "How do I set up Go development in VS Code?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "languages", "difficulty": "medium"},
    {"query_id": "ht_053", "query_text": "How to create and run build tasks from tasks.json?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "tasks", "difficulty": "medium"},
    {"query_id": "ht_054", "query_text": "How do I use the editor's find and replace with regex?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "ht_055", "query_text": "How to configure color themes and icon themes?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "workbench", "difficulty": "easy"},
    {"query_id": "ht_056", "query_text": "How do I debug a browser-based web application in VS Code?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "debug", "difficulty": "medium"},
    {"query_id": "ht_057", "query_text": "How to use the multi-root workspaces feature?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "workbench", "difficulty": "medium"},
    {"query_id": "ht_058", "query_text": "How do I configure the editor to use a specific EOL character?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "ht_059", "query_text": "How to set up Rust development with rust-analyzer?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "languages", "difficulty": "medium"},
    {"query_id": "ht_060", "query_text": "How do I use the VS Code CLI to open files and diff?", "category": "how_to", "expected_sources": ["doc"], "expected_area": "workbench", "difficulty": "easy"},

    # =========================================================================
    # debugging (75 queries, 30%)
    # =========================================================================
    {"query_id": "db_001", "query_text": "VS Code terminal is not rendering colors correctly on Windows", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "terminal", "difficulty": "medium"},
    {"query_id": "db_002", "query_text": "Git extension not detecting changes in workspace", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "git", "difficulty": "medium"},
    {"query_id": "db_003", "query_text": "VS Code freezes when opening large files", "category": "debugging", "expected_sources": ["bug"], "expected_area": "performance", "difficulty": "hard"},
    {"query_id": "db_004", "query_text": "Copilot suggestions appearing very slowly", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "copilot", "difficulty": "medium"},
    {"query_id": "db_005", "query_text": "Remote SSH connection keeps dropping", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "remote", "difficulty": "hard"},
    {"query_id": "db_006", "query_text": "Extension host terminated unexpectedly", "category": "debugging", "expected_sources": ["bug"], "expected_area": "extensions", "difficulty": "hard"},
    {"query_id": "db_007", "query_text": "IntelliSense not working for TypeScript in monorepo", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "languages", "difficulty": "hard"},
    {"query_id": "db_008", "query_text": "Search not finding results in workspace", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "search", "difficulty": "medium"},
    {"query_id": "db_009", "query_text": "Sticky scroll causing rendering glitches", "category": "debugging", "expected_sources": ["bug"], "expected_area": "editor", "difficulty": "medium"},
    {"query_id": "db_010", "query_text": "Debug console not showing output from Python scripts", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "debug", "difficulty": "medium"},
    {"query_id": "db_011", "query_text": "Terminal cursor disappears after resizing window", "category": "debugging", "expected_sources": ["bug"], "expected_area": "terminal", "difficulty": "medium"},
    {"query_id": "db_012", "query_text": "Auto-import not working for JavaScript modules", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "languages", "difficulty": "medium"},
    {"query_id": "db_013", "query_text": "VS Code consuming too much memory with multiple tabs open", "category": "debugging", "expected_sources": ["bug"], "expected_area": "performance", "difficulty": "hard"},
    {"query_id": "db_014", "query_text": "Keyboard shortcuts not working after extension update", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "settings", "difficulty": "medium"},
    {"query_id": "db_015", "query_text": "Diff editor showing wrong changes after Git merge", "category": "debugging", "expected_sources": ["bug"], "expected_area": "git", "difficulty": "hard"},
    {"query_id": "db_016", "query_text": "Code folding not working correctly for Python files", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "editor", "difficulty": "medium"},
    {"query_id": "db_017", "query_text": "WSL terminal not starting properly", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "remote", "difficulty": "medium"},
    {"query_id": "db_018", "query_text": "Breakpoints not being hit in debug session", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "debug", "difficulty": "hard"},
    {"query_id": "db_019", "query_text": "Editor scrolling performance is laggy", "category": "debugging", "expected_sources": ["bug"], "expected_area": "performance", "difficulty": "medium"},
    {"query_id": "db_020", "query_text": "File watcher not detecting external changes on Linux", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "workbench", "difficulty": "medium"},
    {"query_id": "db_021", "query_text": "Problems panel not updating after fixing lint errors", "category": "debugging", "expected_sources": ["bug"], "expected_area": "workbench", "difficulty": "medium"},
    {"query_id": "db_022", "query_text": "Copilot Chat not responding to prompts", "category": "debugging", "expected_sources": ["bug"], "expected_area": "copilot", "difficulty": "medium"},
    {"query_id": "db_023", "query_text": "Color picker not appearing for CSS color values", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "db_024", "query_text": "Terminal copy-paste not working on macOS", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "terminal", "difficulty": "medium"},
    {"query_id": "db_025", "query_text": "VS Code settings sync failing silently", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "settings", "difficulty": "medium"},
    {"query_id": "db_026", "query_text": "Notebook cells not executing on remote server", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "notebook", "difficulty": "hard"},
    {"query_id": "db_027", "query_text": "Extensions marketplace search returning no results", "category": "debugging", "expected_sources": ["bug"], "expected_area": "extensions", "difficulty": "medium"},
    {"query_id": "db_028", "query_text": "Git blame annotations disappearing intermittently", "category": "debugging", "expected_sources": ["bug"], "expected_area": "git", "difficulty": "medium"},
    {"query_id": "db_029", "query_text": "Editor hover tooltips not appearing for TypeScript", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "editor", "difficulty": "medium"},
    {"query_id": "db_030", "query_text": "Task runner failing with exit code 1 but no error message", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "tasks", "difficulty": "hard"},
    {"query_id": "db_031", "query_text": "VS Code window blank after resume from sleep", "category": "debugging", "expected_sources": ["bug"], "expected_area": "workbench", "difficulty": "hard"},
    {"query_id": "db_032", "query_text": "Emmet expansions not triggering in JSX files", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "languages", "difficulty": "medium"},
    {"query_id": "db_033", "query_text": "Terminal output garbled with special characters", "category": "debugging", "expected_sources": ["bug"], "expected_area": "terminal", "difficulty": "medium"},
    {"query_id": "db_034", "query_text": "Minimap not updating when editing large files", "category": "debugging", "expected_sources": ["bug"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "db_035", "query_text": "Dev container build failing with network errors", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "remote", "difficulty": "hard"},
    {"query_id": "db_036", "query_text": "Source Control view showing stale status", "category": "debugging", "expected_sources": ["bug"], "expected_area": "git", "difficulty": "medium"},
    {"query_id": "db_037", "query_text": "Editor inlay hints overlapping with code text", "category": "debugging", "expected_sources": ["bug"], "expected_area": "editor", "difficulty": "medium"},
    {"query_id": "db_038", "query_text": "Python debugger not stopping at conditional breakpoints", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "debug", "difficulty": "hard"},
    {"query_id": "db_039", "query_text": "File explorer not refreshing after creating new files", "category": "debugging", "expected_sources": ["bug"], "expected_area": "workbench", "difficulty": "easy"},
    {"query_id": "db_040", "query_text": "Multi-cursor selections getting lost on undo", "category": "debugging", "expected_sources": ["bug"], "expected_area": "editor", "difficulty": "medium"},
    {"query_id": "db_041", "query_text": "VS Code update fails with EPERM on Windows", "category": "debugging", "expected_sources": ["bug"], "expected_area": "workbench", "difficulty": "medium"},
    {"query_id": "db_042", "query_text": "GitHub authentication token keeps expiring in VS Code", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "git", "difficulty": "medium"},
    {"query_id": "db_043", "query_text": "Bracket matching highlighting wrong pairs", "category": "debugging", "expected_sources": ["bug"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "db_044", "query_text": "Remote container extension crashing on Apple Silicon", "category": "debugging", "expected_sources": ["bug"], "expected_area": "remote", "difficulty": "hard"},
    {"query_id": "db_045", "query_text": "Go to definition navigating to wrong file", "category": "debugging", "expected_sources": ["bug"], "expected_area": "editor", "difficulty": "medium"},
    {"query_id": "db_046", "query_text": "Terminal bell sound not working", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "terminal", "difficulty": "easy"},
    {"query_id": "db_047", "query_text": "VS Code crashes when opening very long lines", "category": "debugging", "expected_sources": ["bug"], "expected_area": "performance", "difficulty": "hard"},
    {"query_id": "db_048", "query_text": "Workspace recommendations not appearing for new users", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "extensions", "difficulty": "medium"},
    {"query_id": "db_049", "query_text": "Text selection highlight not visible with some themes", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "workbench", "difficulty": "easy"},
    {"query_id": "db_050", "query_text": "Format document changing line endings unexpectedly", "category": "debugging", "expected_sources": ["bug"], "expected_area": "editor", "difficulty": "medium"},
    {"query_id": "db_051", "query_text": "SSH file editing introduces extra blank lines", "category": "debugging", "expected_sources": ["bug"], "expected_area": "remote", "difficulty": "medium"},
    {"query_id": "db_052", "query_text": "Integrated terminal freezes after running npm install", "category": "debugging", "expected_sources": ["bug"], "expected_area": "terminal", "difficulty": "medium"},
    {"query_id": "db_053", "query_text": "Notebook kernel connection failing intermittently", "category": "debugging", "expected_sources": ["bug"], "expected_area": "notebook", "difficulty": "hard"},
    {"query_id": "db_054", "query_text": "Quick fix suggestions not appearing for ESLint errors", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "languages", "difficulty": "medium"},
    {"query_id": "db_055", "query_text": "Git merge editor losing changes after save", "category": "debugging", "expected_sources": ["bug"], "expected_area": "git", "difficulty": "hard"},
    {"query_id": "db_056", "query_text": "Tab completion not working in integrated terminal", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "terminal", "difficulty": "medium"},
    {"query_id": "db_057", "query_text": "Debugger variables panel showing stale values", "category": "debugging", "expected_sources": ["bug"], "expected_area": "debug", "difficulty": "medium"},
    {"query_id": "db_058", "query_text": "Explorer drag and drop moving files to wrong location", "category": "debugging", "expected_sources": ["bug"], "expected_area": "workbench", "difficulty": "medium"},
    {"query_id": "db_059", "query_text": "VS Code not recognizing .env files for environment variables", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "debug", "difficulty": "medium"},
    {"query_id": "db_060", "query_text": "Editor word wrap breaking in middle of words", "category": "debugging", "expected_sources": ["bug"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "db_061", "query_text": "Copilot completions appearing in comments when disabled", "category": "debugging", "expected_sources": ["bug"], "expected_area": "copilot", "difficulty": "medium"},
    {"query_id": "db_062", "query_text": "Code actions menu flickering and disappearing", "category": "debugging", "expected_sources": ["bug"], "expected_area": "editor", "difficulty": "medium"},
    {"query_id": "db_063", "query_text": "Test explorer not discovering tests in subdirectories", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "testing", "difficulty": "medium"},
    {"query_id": "db_064", "query_text": "Status bar items overlapping on small screens", "category": "debugging", "expected_sources": ["bug"], "expected_area": "workbench", "difficulty": "easy"},
    {"query_id": "db_065", "query_text": "Linked editing not working for HTML tag pairs", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "db_066", "query_text": "Terminal shell integration showing wrong working directory", "category": "debugging", "expected_sources": ["bug"], "expected_area": "terminal", "difficulty": "medium"},
    {"query_id": "db_067", "query_text": "Extensions taking too long to activate on startup", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "performance", "difficulty": "medium"},
    {"query_id": "db_068", "query_text": "Autocomplete showing irrelevant suggestions in Markdown", "category": "debugging", "expected_sources": ["bug"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "db_069", "query_text": "SSH remote server using too much disk space", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "remote", "difficulty": "medium"},
    {"query_id": "db_070", "query_text": "Git graph extension conflicting with built-in Git features", "category": "debugging", "expected_sources": ["bug"], "expected_area": "extensions", "difficulty": "medium"},
    {"query_id": "db_071", "query_text": "Editor not saving file when disk is full", "category": "debugging", "expected_sources": ["bug"], "expected_area": "workbench", "difficulty": "medium"},
    {"query_id": "db_072", "query_text": "Semantic highlighting flickering when typing fast", "category": "debugging", "expected_sources": ["bug"], "expected_area": "editor", "difficulty": "medium"},
    {"query_id": "db_073", "query_text": "Debug watch expressions not evaluating correctly", "category": "debugging", "expected_sources": ["bug"], "expected_area": "debug", "difficulty": "hard"},
    {"query_id": "db_074", "query_text": "Terminal font rendering blurry on high-DPI displays", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "terminal", "difficulty": "medium"},
    {"query_id": "db_075", "query_text": "VS Code not prompting to install recommended extensions", "category": "debugging", "expected_sources": ["bug", "doc"], "expected_area": "extensions", "difficulty": "easy"},

    # =========================================================================
    # error_diagnosis (50 queries, 20%)
    # =========================================================================
    {"query_id": "ed_001", "query_text": "Error: EACCES permission denied when installing extensions", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "extensions", "difficulty": "medium"},
    {"query_id": "ed_002", "query_text": "TypeError: Cannot read properties of undefined in output panel", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "workbench", "difficulty": "hard"},
    {"query_id": "ed_003", "query_text": "Git error: unsafe repository is owned by someone else", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "git", "difficulty": "medium"},
    {"query_id": "ed_004", "query_text": "EPERM operation not permitted when saving files on Windows", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "explorer", "difficulty": "medium"},
    {"query_id": "ed_005", "query_text": "SSH connection error: Permission denied (publickey)", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "remote", "difficulty": "medium"},
    {"query_id": "ed_006", "query_text": "Error: spawn ENOENT when running tasks on Windows", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "tasks", "difficulty": "medium"},
    {"query_id": "ed_007", "query_text": "ENOSPC: System limit for number of file watchers reached", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "workbench", "difficulty": "medium"},
    {"query_id": "ed_008", "query_text": "Error: Running the contributed command failed", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "extensions", "difficulty": "hard"},
    {"query_id": "ed_009", "query_text": "SyntaxError: Unexpected token in JSON settings file", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "ed_010", "query_text": "Git: fatal: not a git repository when opening workspace", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "git", "difficulty": "easy"},
    {"query_id": "ed_011", "query_text": "Error: command 'python.setInterpreter' not found", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "languages", "difficulty": "medium"},
    {"query_id": "ed_012", "query_text": "ENOMEM: not enough memory when opening project", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "performance", "difficulty": "hard"},
    {"query_id": "ed_013", "query_text": "Error: ETIMEOUT connecting to remote SSH host", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "remote", "difficulty": "medium"},
    {"query_id": "ed_014", "query_text": "Error: Cannot find module 'typescript' in workspace", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "languages", "difficulty": "medium"},
    {"query_id": "ed_015", "query_text": "Renderer process crashed with exit code 1", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "workbench", "difficulty": "hard"},
    {"query_id": "ed_016", "query_text": "Error: EBUSY: resource busy when deleting files", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "explorer", "difficulty": "medium"},
    {"query_id": "ed_017", "query_text": "Git: error: Your local changes would be overwritten by merge", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "git", "difficulty": "easy"},
    {"query_id": "ed_018", "query_text": "Debug adapter process has terminated unexpectedly", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "debug", "difficulty": "hard"},
    {"query_id": "ed_019", "query_text": "Error: Unable to resolve workspace folder variable", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "tasks", "difficulty": "medium"},
    {"query_id": "ed_020", "query_text": "OOM killed: VS Code server running out of memory on remote", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "remote", "difficulty": "hard"},
    {"query_id": "ed_021", "query_text": "Error: certificate has expired when connecting to marketplace", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "extensions", "difficulty": "medium"},
    {"query_id": "ed_022", "query_text": "Error: ENOENT no such file or directory in tasks", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "tasks", "difficulty": "medium"},
    {"query_id": "ed_023", "query_text": "ESLint error: failed to load config extends", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "languages", "difficulty": "medium"},
    {"query_id": "ed_024", "query_text": "Error: EXDEV rename not permitted across devices", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "explorer", "difficulty": "medium"},
    {"query_id": "ed_025", "query_text": "SSH: Could not establish connection to host", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "remote", "difficulty": "medium"},
    {"query_id": "ed_026", "query_text": "Error: Extension activation failed for Python extension", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "extensions", "difficulty": "medium"},
    {"query_id": "ed_027", "query_text": "Terminal: pty host was unable to resolve shell environment", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "terminal", "difficulty": "medium"},
    {"query_id": "ed_028", "query_text": "Error: Maximum call stack size exceeded in extension", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "extensions", "difficulty": "hard"},
    {"query_id": "ed_029", "query_text": "Jupyter: Kernel died with exit code 1", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "notebook", "difficulty": "hard"},
    {"query_id": "ed_030", "query_text": "Error: ECONNREFUSED connecting to debug port", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "debug", "difficulty": "medium"},
    {"query_id": "ed_031", "query_text": "Git push rejected: non-fast-forward updates", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "git", "difficulty": "easy"},
    {"query_id": "ed_032", "query_text": "Error: Unable to write to user settings (Unknown Error)", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "settings", "difficulty": "medium"},
    {"query_id": "ed_033", "query_text": "SIGTERM: Extension host exited with signal SIGTERM", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "extensions", "difficulty": "hard"},
    {"query_id": "ed_034", "query_text": "Error: No such device or address when opening terminal", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "terminal", "difficulty": "medium"},
    {"query_id": "ed_035", "query_text": "Error: EMFILE too many open files", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "workbench", "difficulty": "medium"},
    {"query_id": "ed_036", "query_text": "TypeScript server crashed 5 times in the last 3 minutes", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "languages", "difficulty": "hard"},
    {"query_id": "ed_037", "query_text": "Error: container build failed with exit code 125", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "remote", "difficulty": "hard"},
    {"query_id": "ed_038", "query_text": "Error: ECONNRESET when downloading extensions", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "extensions", "difficulty": "medium"},
    {"query_id": "ed_039", "query_text": "Git: error: pathspec did not match any files", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "git", "difficulty": "easy"},
    {"query_id": "ed_040", "query_text": "Error: The editor could not be opened because the file was not found", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "workbench", "difficulty": "easy"},
    {"query_id": "ed_041", "query_text": "Segmentation fault in VS Code on Linux Wayland", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "workbench", "difficulty": "hard"},
    {"query_id": "ed_042", "query_text": "Error: XHR failed when fetching extension details", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "extensions", "difficulty": "medium"},
    {"query_id": "ed_043", "query_text": "Error: Could not create temporary directory for download", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "workbench", "difficulty": "medium"},
    {"query_id": "ed_044", "query_text": "Python linting error: pylint is not installed", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "languages", "difficulty": "easy"},
    {"query_id": "ed_045", "query_text": "Error: Unable to open integrated terminal", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "terminal", "difficulty": "medium"},
    {"query_id": "ed_046", "query_text": "Shared process crashed with exit code 3221225477", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "workbench", "difficulty": "hard"},
    {"query_id": "ed_047", "query_text": "Error: socket hang up during remote development", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "remote", "difficulty": "medium"},
    {"query_id": "ed_048", "query_text": "Git: error: lock file already exists", "category": "error_diagnosis", "expected_sources": ["bug", "doc"], "expected_area": "git", "difficulty": "medium"},
    {"query_id": "ed_049", "query_text": "Error: Activating extension failed: command already registered", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "extensions", "difficulty": "hard"},
    {"query_id": "ed_050", "query_text": "Error: CORS policy blocking request from webview", "category": "error_diagnosis", "expected_sources": ["bug"], "expected_area": "workbench", "difficulty": "hard"},

    # =========================================================================
    # status_roadmap (40 queries, 16%)
    # =========================================================================
    {"query_id": "sr_001", "query_text": "What is planned for VS Code Copilot in the next release?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "copilot", "difficulty": "medium"},
    {"query_id": "sr_002", "query_text": "When will native bracket pair colorization support custom colors?", "category": "status_roadmap", "expected_sources": ["work_item", "bug"], "expected_area": "editor", "difficulty": "medium"},
    {"query_id": "sr_003", "query_text": "Is there a plan to improve terminal performance?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "terminal", "difficulty": "medium"},
    {"query_id": "sr_004", "query_text": "What features were shipped in the January 2024 iteration?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "workbench", "difficulty": "easy"},
    {"query_id": "sr_005", "query_text": "Has the multi-root workspace support been improved recently?", "category": "status_roadmap", "expected_sources": ["work_item", "doc"], "expected_area": "workbench", "difficulty": "medium"},
    {"query_id": "sr_006", "query_text": "What notebook improvements are planned for VS Code?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "notebook", "difficulty": "medium"},
    {"query_id": "sr_007", "query_text": "Is VS Code planning to add native support for Vim keybindings?", "category": "status_roadmap", "expected_sources": ["work_item", "bug"], "expected_area": "editor", "difficulty": "medium"},
    {"query_id": "sr_008", "query_text": "What accessibility improvements are in the roadmap?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "accessibility", "difficulty": "medium"},
    {"query_id": "sr_009", "query_text": "When will the new settings editor be available?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "settings", "difficulty": "medium"},
    {"query_id": "sr_010", "query_text": "What debugging improvements were made in the last quarter?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "debug", "difficulty": "easy"},
    {"query_id": "sr_011", "query_text": "Is there a plan to support multiple GitHub accounts?", "category": "status_roadmap", "expected_sources": ["work_item", "bug"], "expected_area": "git", "difficulty": "medium"},
    {"query_id": "sr_012", "query_text": "What extensions API improvements are planned?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "extensions", "difficulty": "hard"},
    {"query_id": "sr_013", "query_text": "Will VS Code add built-in support for AI code review?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "copilot", "difficulty": "medium"},
    {"query_id": "sr_014", "query_text": "What search improvements are in the current iteration plan?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "search", "difficulty": "medium"},
    {"query_id": "sr_015", "query_text": "Has the issue about slow startup been fixed?", "category": "status_roadmap", "expected_sources": ["work_item", "bug"], "expected_area": "performance", "difficulty": "medium"},
    {"query_id": "sr_016", "query_text": "What remote development features were shipped in 2024?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "remote", "difficulty": "easy"},
    {"query_id": "sr_017", "query_text": "Is there a roadmap for improving Python support?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "languages", "difficulty": "medium"},
    {"query_id": "sr_018", "query_text": "What testing framework improvements are planned?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "testing", "difficulty": "medium"},
    {"query_id": "sr_019", "query_text": "Will VS Code support workspaces in the web version?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "workbench", "difficulty": "hard"},
    {"query_id": "sr_020", "query_text": "What git features are planned for the next release?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "git", "difficulty": "medium"},
    {"query_id": "sr_021", "query_text": "Has the editor performance regression been addressed?", "category": "status_roadmap", "expected_sources": ["work_item", "bug"], "expected_area": "performance", "difficulty": "medium"},
    {"query_id": "sr_022", "query_text": "What localization improvements are in progress?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "workbench", "difficulty": "easy"},
    {"query_id": "sr_023", "query_text": "Is there a plan for better terminal profiles management?", "category": "status_roadmap", "expected_sources": ["work_item", "doc"], "expected_area": "terminal", "difficulty": "medium"},
    {"query_id": "sr_024", "query_text": "What Copilot features were completed in the December iteration?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "copilot", "difficulty": "easy"},
    {"query_id": "sr_025", "query_text": "Will VS Code add native support for database browsing?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "extensions", "difficulty": "medium"},
    {"query_id": "sr_026", "query_text": "What progress has been made on the new diff algorithm?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "editor", "difficulty": "medium"},
    {"query_id": "sr_027", "query_text": "Is there a plan for workspace-level extension recommendations?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "extensions", "difficulty": "medium"},
    {"query_id": "sr_028", "query_text": "What were the key themes in the February 2024 iteration plan?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "workbench", "difficulty": "easy"},
    {"query_id": "sr_029", "query_text": "Has the SSH reconnection reliability been improved?", "category": "status_roadmap", "expected_sources": ["work_item", "bug"], "expected_area": "remote", "difficulty": "medium"},
    {"query_id": "sr_030", "query_text": "What inline chat improvements are planned for Copilot?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "copilot", "difficulty": "medium"},
    {"query_id": "sr_031", "query_text": "Is there a plan to improve extension load time?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "performance", "difficulty": "medium"},
    {"query_id": "sr_032", "query_text": "What editor features were shipped in the March 2024 release?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "sr_033", "query_text": "Will VS Code add built-in profiling tools?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "performance", "difficulty": "hard"},
    {"query_id": "sr_034", "query_text": "What improvements are planned for the notebook experience?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "notebook", "difficulty": "medium"},
    {"query_id": "sr_035", "query_text": "Has the plan for improved breadcrumbs been completed?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "sr_036", "query_text": "What WSL improvements are in the current iteration?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "remote", "difficulty": "medium"},
    {"query_id": "sr_037", "query_text": "Is there a timeline for the settings profile sync feature?", "category": "status_roadmap", "expected_sources": ["work_item", "doc"], "expected_area": "settings", "difficulty": "medium"},
    {"query_id": "sr_038", "query_text": "What task runner improvements are planned?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "tasks", "difficulty": "medium"},
    {"query_id": "sr_039", "query_text": "Has the sticky scroll feature been stabilized?", "category": "status_roadmap", "expected_sources": ["work_item", "bug"], "expected_area": "editor", "difficulty": "easy"},
    {"query_id": "sr_040", "query_text": "What themes and icon improvements are planned?", "category": "status_roadmap", "expected_sources": ["work_item"], "expected_area": "workbench", "difficulty": "easy"},

    # =========================================================================
    # config (25 queries, 10%)
    # =========================================================================
    {"query_id": "cf_001", "query_text": "How to change the default font size in the editor?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "cf_002", "query_text": "What keybinding opens the command palette?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "cf_003", "query_text": "How to configure auto-save behavior?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "cf_004", "query_text": "How to set different settings for different file types?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "medium"},
    {"query_id": "cf_005", "query_text": "How to sync VS Code settings across machines?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "cf_006", "query_text": "How to customize the keybinding for toggling the terminal?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "cf_007", "query_text": "What setting controls the tab size for Python files?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "cf_008", "query_text": "How to configure file associations for custom file extensions?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "medium"},
    {"query_id": "cf_009", "query_text": "How to set the default formatter for JavaScript files?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "cf_010", "query_text": "How to disable telemetry in VS Code?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "cf_011", "query_text": "How to configure the editor to show whitespace characters?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "cf_012", "query_text": "How to set a custom color theme for the editor?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "cf_013", "query_text": "How to exclude folders from the file explorer?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "medium"},
    {"query_id": "cf_014", "query_text": "How to configure the terminal font family?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "cf_015", "query_text": "How to set up a custom keyboard shortcut for a command?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "medium"},
    {"query_id": "cf_016", "query_text": "How to configure line numbers in the editor?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "cf_017", "query_text": "How to set the default shell for the integrated terminal?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "cf_018", "query_text": "How to configure window title bar customization?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "medium"},
    {"query_id": "cf_019", "query_text": "How to set cursor style and blinking behavior?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "cf_020", "query_text": "How to configure breadcrumbs visibility in the editor?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "cf_021", "query_text": "How to set workspace-specific git settings?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "medium"},
    {"query_id": "cf_022", "query_text": "How to configure the sidebar position (left vs right)?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "cf_023", "query_text": "How to set the zoom level for VS Code?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "cf_024", "query_text": "How to configure compact folders in the explorer?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "easy"},
    {"query_id": "cf_025", "query_text": "How to set up editor rulers at specific column positions?", "category": "config", "expected_sources": ["doc"], "expected_area": "settings", "difficulty": "medium"},
]


def load_queries(path: str = "data/evaluation/query_bank.json") -> list[EvalQuery]:
    """Load evaluation queries from file."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [EvalQuery.from_dict(d) for d in data]
    # Return seed queries if no saved bank exists
    return [EvalQuery.from_dict(q) for q in SEED_QUERIES]


def save_queries(queries: list[EvalQuery], path: str = "data/evaluation/query_bank.json") -> None:
    """Save evaluation queries to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = [q.to_dict() for q in queries]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def stratify_by_category(queries: list[EvalQuery]) -> dict[str, list[EvalQuery]]:
    """Group queries by category."""
    groups: dict[str, list[EvalQuery]] = {}
    for q in queries:
        groups.setdefault(q.category, []).append(q)
    return groups


def get_category_distribution(queries: list[EvalQuery]) -> dict[str, int]:
    """Return count per category."""
    return dict(Counter(q.category for q in queries))
