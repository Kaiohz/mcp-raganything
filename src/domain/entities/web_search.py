"""Web-search domain entities.

Pure Pydantic models describing the DuckDuckGo provider result mapping and
the per-result exploration outcome. No FastMCP, DDGS, Crawl4AI or HTTPX
imports are allowed in this module.
"""

from enum import StrEnum

from pydantic import BaseModel


class WebCrawlStatus(StrEnum):
    """Exploration status of a single search result."""

    NOT_EXPLORED = "not_explored"
    SUCCESS = "success"
    FAILED = "failed"


class WebSearchPage(BaseModel):
    """A provider search entry mapped to the domain (title, url, snippet)."""

    title: str
    url: str
    snippet: str


class WebCrawlOutcome(BaseModel):
    """Exploration outcome for one crawled (or not crawled) URL.

    Attributes:
        url: The URL that was selected for exploration (validated as safe).
        status: ``not_explored`` / ``success`` / ``failed``.
        final_url: Last validated URL when known, otherwise None.
        markdown: Crawl4AI Markdown for successful pages, "" when the page
            had no content. Never truncated unless ``truncated`` is True
            (``WEB_SEARCH_PAGE_MAX_CHARS`` cap). Always None unless success.
        truncated: True when the Markdown was cut at the configured
            per-page character cap (False when no cap applies).
        error: Safe, concise failure reason (no provider internals).
    """

    url: str
    status: WebCrawlStatus
    final_url: str | None = None
    markdown: str | None = None
    truncated: bool = False
    error: str | None = None


class WebSearchResult(BaseModel):
    """One public search result together with its exploration outcome."""

    title: str
    url: str
    snippet: str
    crawl: WebCrawlOutcome


class WebSearchReport(BaseModel):
    """Complete web-search report in stable provider (DuckDuckGo) order."""

    query: str
    results: list[WebSearchResult]
