"""GitHub API client configured for Flutter repo."""

import time
import logging
import requests

import config_flutter as config

logger = logging.getLogger(__name__)


class GitHubClient:
    """GitHub API client with automatic rate limiting and retry logic."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(config.GITHUB_HEADERS)
        self._search_request_times: list[float] = []

    def _check_rate_limit(self, response: requests.Response):
        remaining = int(response.headers.get("x-ratelimit-remaining", 999))
        reset_time = int(response.headers.get("x-ratelimit-reset", 0))
        if remaining <= 2:
            wait = max(reset_time - time.time(), 0) + 1
            logger.warning(f"Rate limit nearly exhausted ({remaining} left). Sleeping {wait:.0f}s")
            time.sleep(wait)

    def _throttle_search(self):
        now = time.time()
        self._search_request_times = [t for t in self._search_request_times if now - t < 60]
        if len(self._search_request_times) >= config.SEARCH_RATE_LIMIT - 1:
            oldest = self._search_request_times[0]
            wait = 60 - (now - oldest) + 0.5
            if wait > 0:
                logger.info(f"Search rate limit: sleeping {wait:.1f}s")
                time.sleep(wait)
        self._search_request_times.append(time.time())

    def _request(self, method: str, url: str, is_search: bool = False, **kwargs) -> requests.Response:
        if is_search:
            self._throttle_search()

        for attempt in range(config.MAX_RETRIES + 1):
            try:
                response = self.session.request(method, url, timeout=30, **kwargs)

                if response.status_code == 200:
                    self._check_rate_limit(response)
                    return response

                if response.status_code in (403, 429):
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
                    logger.warning(f"Server error {response.status_code}. Retrying in {wait}s")
                    time.sleep(wait)
                    continue

                response.raise_for_status()

            except requests.exceptions.Timeout:
                wait = min(config.BACKOFF_BASE ** (attempt + 1), config.BACKOFF_MAX)
                logger.warning(f"Request timeout. Retrying in {wait}s")
                time.sleep(wait)
            except requests.exceptions.ConnectionError:
                wait = min(config.BACKOFF_BASE ** (attempt + 1), config.BACKOFF_MAX)
                logger.warning(f"Connection error. Retrying in {wait}s")
                time.sleep(wait)

        raise RuntimeError(f"Failed after {config.MAX_RETRIES + 1} attempts: {url}")

    def search_issues(self, query: str, per_page: int = 100, max_results: int = 0):
        """Search issues, yielding all results across pages."""
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
                if max_results and total_yielded >= max_results:
                    return

            logger.info(f"Search page {page}: got {len(items)} items ({total_yielded}/{total_count} total)")

            if total_yielded >= total_count or total_yielded >= 1000:
                break

            page += 1

    def get_issue_comments(self, issue_number: int, per_page: int = 100) -> list[dict]:
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
        url = f"{config.GITHUB_API_BASE}/rate_limit"
        response = self._request("GET", url)
        return response.json()

    def print_rate_limit(self):
        limits = self.get_rate_limit()
        core = limits["resources"]["core"]
        search = limits["resources"]["search"]
        logger.info(f"Rate limits - Core: {core['remaining']}/{core['limit']} | Search: {search['remaining']}/{search['limit']}")
