"""Shared GitHub API client with rate limiting, retry, and logging."""

import time
import logging
import requests

import config

logger = logging.getLogger(__name__)


class GitHubClient:
    """GitHub API client with automatic rate limiting and retry logic."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(config.GITHUB_HEADERS)
        self._search_request_times: list[float] = []

    # --- Rate Limit Handling ---

    def _check_rate_limit(self, response: requests.Response):
        """Sleep if we're close to hitting rate limits."""
        remaining = int(response.headers.get("x-ratelimit-remaining", 999))
        reset_time = int(response.headers.get("x-ratelimit-reset", 0))

        if remaining <= 2:
            wait = max(reset_time - time.time(), 0) + 1
            logger.warning(f"Rate limit nearly exhausted ({remaining} left). Sleeping {wait:.0f}s until reset.")
            time.sleep(wait)

    def _throttle_search(self):
        """Enforce search API rate limit: max 30 requests/minute."""
        now = time.time()
        # Remove timestamps older than 60 seconds
        self._search_request_times = [t for t in self._search_request_times if now - t < 60]

        if len(self._search_request_times) >= config.SEARCH_RATE_LIMIT - 1:
            oldest = self._search_request_times[0]
            wait = 60 - (now - oldest) + 0.5
            if wait > 0:
                logger.info(f"Search rate limit: sleeping {wait:.1f}s")
                time.sleep(wait)

        self._search_request_times.append(time.time())

    # --- Request with Retry ---

    def _request(self, method: str, url: str, is_search: bool = False, **kwargs) -> requests.Response:
        """Make an API request with retry and backoff."""
        if is_search:
            self._throttle_search()

        for attempt in range(config.MAX_RETRIES + 1):
            try:
                response = self.session.request(method, url, timeout=30, **kwargs)

                if response.status_code == 200:
                    self._check_rate_limit(response)
                    return response

                if response.status_code in (403, 429):
                    # Rate limited or forbidden
                    retry_after = int(response.headers.get("retry-after", 0))
                    reset_time = int(response.headers.get("x-ratelimit-reset", 0))

                    if retry_after:
                        wait = retry_after + 1
                    elif reset_time:
                        wait = max(reset_time - time.time(), 0) + 1
                    else:
                        wait = min(config.BACKOFF_BASE ** (attempt + 1), config.BACKOFF_MAX)

                    logger.warning(f"Rate limited (HTTP {response.status_code}). Waiting {wait:.0f}s (attempt {attempt + 1})")
                    time.sleep(wait)
                    continue

                if response.status_code >= 500:
                    wait = min(config.BACKOFF_BASE ** (attempt + 1), config.BACKOFF_MAX)
                    logger.warning(f"Server error {response.status_code}. Retrying in {wait}s (attempt {attempt + 1})")
                    time.sleep(wait)
                    continue

                # Other client errors — don't retry
                response.raise_for_status()

            except requests.exceptions.Timeout:
                wait = min(config.BACKOFF_BASE ** (attempt + 1), config.BACKOFF_MAX)
                logger.warning(f"Request timeout. Retrying in {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
            except requests.exceptions.ConnectionError:
                wait = min(config.BACKOFF_BASE ** (attempt + 1), config.BACKOFF_MAX)
                logger.warning(f"Connection error. Retrying in {wait}s (attempt {attempt + 1})")
                time.sleep(wait)

        raise RuntimeError(f"Failed after {config.MAX_RETRIES + 1} attempts: {url}")

    # --- Public API Methods ---

    def search_issues(self, query: str, per_page: int = 100):
        """Search issues, yielding all results across pages.

        Yields individual issue dicts. Handles pagination automatically.
        GitHub Search API returns max 1000 results per query.
        """
        url = f"{config.GITHUB_API_BASE}/search/issues"
        page = 1
        total_yielded = 0

        while True:
            params = {"q": query, "per_page": per_page, "page": page}
            response = self._request("GET", url, is_search=True, params=params)
            data = response.json()

            total_count = data.get("total_count", 0)
            items = data.get("items", [])

            if not items:
                break

            for item in items:
                yield item
                total_yielded += 1

            logger.info(f"Search page {page}: got {len(items)} items ({total_yielded}/{total_count} total)")

            if total_yielded >= total_count or total_yielded >= 1000:
                break

            page += 1

    def get_issue_comments(self, issue_number: int, per_page: int = 100) -> list[dict]:
        """Fetch all comments for a specific issue."""
        url = f"{config.GITHUB_API_BASE}/repos/{config.REPO_OWNER}/{config.REPO_NAME}/issues/{issue_number}/comments"
        all_comments = []
        page = 1

        while True:
            params = {"per_page": per_page, "page": page}
            response = self._request("GET", url, params=params)
            comments = response.json()

            if not comments:
                break

            all_comments.extend(comments)

            if len(comments) < per_page:
                break

            page += 1

        return all_comments

    def get_rate_limit(self) -> dict:
        """Get current rate limit status."""
        url = f"{config.GITHUB_API_BASE}/rate_limit"
        response = self._request("GET", url)
        return response.json()

    def print_rate_limit(self):
        """Print current rate limit status to logger."""
        limits = self.get_rate_limit()
        core = limits["resources"]["core"]
        search = limits["resources"]["search"]
        logger.info(f"Rate limits — Core: {core['remaining']}/{core['limit']} | Search: {search['remaining']}/{search['limit']}")
