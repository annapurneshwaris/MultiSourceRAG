"""Central configuration for HeteroRAG pipeline."""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# --- GitHub API ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_API_BASE = "https://api.github.com"
GITHUB_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
if GITHUB_TOKEN:
    GITHUB_HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

# Rate limits (authenticated)
SEARCH_RATE_LIMIT = 30       # requests per minute
REST_RATE_LIMIT = 5000       # requests per hour
SEARCH_SLEEP_BUFFER = 2.1    # seconds between search requests (safe margin)

# Retry settings
MAX_RETRIES = 3
BACKOFF_BASE = 2             # exponential backoff: 2^attempt seconds
BACKOFF_MAX = 60             # max wait between retries

# --- Target Repository ---
REPO_OWNER = "microsoft"
REPO_NAME = "vscode"
DOCS_REPO = "microsoft/vscode-docs"
DOCS_CLONE_URL = "https://github.com/microsoft/vscode-docs.git"

# --- Date Ranges ---
BUG_START_DATE = "2023-01-01"
BUG_END_DATE = "2025-12-31"


def generate_monthly_ranges(start_date: str, end_date: str) -> list[dict]:
    """Generate monthly date windows for paginated search queries."""
    ranges = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        # Last day of the month
        if current.month == 12:
            month_end = current.replace(day=31)
        else:
            month_end = current.replace(month=current.month + 1, day=1) - timedelta(days=1)

        if month_end > end:
            month_end = end

        ranges.append({
            "start": current.strftime("%Y-%m-%d"),
            "end": month_end.strftime("%Y-%m-%d"),
            "label": current.strftime("%Y-%m"),
        })
        # Move to first day of next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1, day=1)
        else:
            current = current.replace(month=current.month + 1, day=1)

    return ranges


BUG_DATE_RANGES = generate_monthly_ranges(BUG_START_DATE, BUG_END_DATE)

# --- Labels ---
BUG_LABELS = ["bug"]

WORKITEM_LABELS = {
    "iteration_plans": "iteration-plan",
    "plan_items": "plan-item",
    "feature_requests": "feature-request",
}

# Feature area labels for cross-source linking (~60+)
FEATURE_AREA_LABELS = [
    "editor", "terminal", "debug", "git", "search", "scm",
    "extensions", "workbench", "tasks", "keybindings", "snippets",
    "languages", "typescript", "javascript", "python", "java", "c++",
    "markdown", "json", "html", "css", "notebook", "testing",
    "remote", "wsl", "ssh", "devcontainer", "copilot", "chat",
    "settings", "themes", "icons", "accessibility", "l10n",
    "explorer", "outline", "breadcrumbs", "minimap", "diff-editor",
    "integrated-terminal", "output", "problems", "comments",
    "timeline", "scm", "source-control", "merge-editor",
    "authentication", "accounts", "proxy", "network",
    "install", "update", "cli", "electron", "performance",
    "ux", "layout", "panel", "sidebar", "statusbar", "titlebar",
    "editor-autocomplete", "editor-bracket-matching", "editor-code-actions",
    "editor-color-picker", "editor-commands", "editor-folding",
    "editor-hover", "editor-indent", "editor-inlay-hints",
    "editor-linked-editing", "editor-multicursor", "editor-rename",
    "editor-sticky-scroll", "editor-symbols", "editor-wrapping",
]

# --- Tier 2 Comment Filter ---
COMMENT_FILTER = {
    "state": "closed",
    "required_labels": ["verified"],
    "min_comments": 2,
    "team_associations": ["MEMBER", "COLLABORATOR", "OWNER"],
}

# --- Chunking ---
CHUNK_MIN_TOKENS = 100
CHUNK_MAX_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

# --- Embedding ---
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_MODEL", "sentence-transformers")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384              # Updated if model changes
EMBEDDING_BATCH_SIZE = 64

# --- LLM ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_PROVIDER = os.getenv("LLM_MODEL", "openai")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2048
JUDGE_MODEL = "gpt-4o-2024-11-20"        # Pinned snapshot for reproducibility
RANDOM_SEED = 42

# --- Vector Store ---
VECTOR_STORE = os.getenv("VECTOR_STORE", "faiss")

# --- Retrieval ---
TOP_K_PER_SOURCE = 20            # Retrieve 20 per source
TOP_K_FINAL = 10                 # Re-rank to top 10
MIN_BOOST = 0.2                  # Never fully exclude a source

# --- Adaptive Router ---
COLD_START_THRESHOLD = 50
ALPHA_INITIAL = 1.0
ALPHA_MIN = 0.3
ALPHA_DECAY = 0.99

# --- Re-ranker Weights ---
W_RELEVANCE = 0.4
W_SOURCE_BOOST = 0.25
W_FRESHNESS = 0.1
W_AUTHORITY = 0.1
W_REDUNDANCY = 0.15
W_BM25 = 0.15

# --- Output Paths ---
RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
CHECKPOINT_DIR = os.path.join("data", "checkpoints")
INDICES_DIR = os.path.join("data", "indices")
MODELS_DIR = os.path.join("data", "models")
EVAL_DIR = os.path.join("data", "evaluation")

ALL_DIRS = [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR, INDICES_DIR, MODELS_DIR, EVAL_DIR]


def init_paths():
    """Create output directories. Call once at startup, not at import time."""
    for d in ALL_DIRS:
        os.makedirs(d, exist_ok=True)


def set_seeds(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility across numpy, torch, and Python."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
