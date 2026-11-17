"""Outbound port for asynchronous page exploration (Crawl4AI)."""

from abc import ABC, abstractmethod

from domain.entities.web_search import WebCrawlOutcome


class WebCrawlerPort(ABC):
    """Boundary for exploring a bounded, pre-validated list of URLs.

    Implementations own the crawler lifecycle (lazy shared instance, per-batch
    dispatchers, process-wide concurrency ceiling, finite timeouts) and return
    one outcome per requested URL, aligned to the input order.
    """

    @abstractmethod
    async def crawl(self, urls: list[str]) -> list[WebCrawlOutcome]:
        """Crawl the given safe URLs in one streaming batch.

        Results are restored to the original input order. Successful pages
        carry the complete Crawl4AI Markdown (never truncated); failures carry
        a safe, concise reason and no Markdown.

        Args:
            urls: URL-safe destinations selected by the use case (in order).

        Returns:
            One :class:`WebCrawlOutcome` per requested URL, input order.
        """

    @abstractmethod
    async def close(self) -> None:
        """Close the shared crawler if it was created. Must be idempotent."""
