"""Central configuration for HeteroRAG pipeline — Flutter cross-project validation."""

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

# --- Target Repository (Flutter) ---
REPO_OWNER = "flutter"
REPO_NAME = "flutter"
DOCS_REPO = "flutter/website"
DOCS_CLONE_URL = "https://github.com/flutter/website.git"

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

# --- Labels (Flutter) ---
BUG_LABELS = ["has reproducible steps", "c: crash", "c: regression"]

WORKITEM_LABELS = {
    "iteration_plans": "P0",
    "plan_items": "c: proposal",
    "feature_requests": "c: new feature",
}

# Feature area labels for cross-source linking (Flutter)
FEATURE_AREA_LABELS = [
    "a: text input", "a: accessibility", "a: animation", "a: desktop",
    "a: images", "a: internationalization", "a: mouse", "a: platform-views",
    "a: quality", "a: tests",
    "f: material design", "f: cupertino", "f: scrolling", "f: routes",
    "f: gestures", "f: focus",
    "platform-android", "platform-ios", "platform-web",
    "platform-windows", "platform-linux", "platform-macos",
    "engine", "framework", "tool",
    "e: impeller", "t: gradle", "t: xcode",
    "c: performance", "c: crash", "c: regression",
    "p: camera", "p: webview", "p: video_player",
    "P0", "P1",
]

# --- Tier 2 Comment Filter ---
COMMENT_FILTER = {
    "state": "closed",
    "required_labels": [],  # Flutter doesn't use 'verified' label
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
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 64

# --- LLM ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_PROVIDER = os.getenv("LLM_MODEL", "openai")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2048
JUDGE_MODEL = "gpt-4o-2024-11-20"
RANDOM_SEED = 42

# --- Vector Store ---
VECTOR_STORE = os.getenv("VECTOR_STORE", "faiss")

# --- Retrieval ---
TOP_K_PER_SOURCE = 20
TOP_K_FINAL = 10
MIN_BOOST = 0.2

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

# --- Output Paths (Flutter subdirectory) ---
RAW_DATA_DIR = os.path.join("data", "flutter", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "flutter", "processed")
CHECKPOINT_DIR = os.path.join("data", "flutter", "checkpoints")
INDICES_DIR = os.path.join("data", "flutter", "indices")
MODELS_DIR = os.path.join("data", "flutter", "models")
EVAL_DIR = os.path.join("data", "flutter", "evaluation")

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
