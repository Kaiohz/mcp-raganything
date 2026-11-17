"""Crawl4AI exploration adapter (pinned ``crawl4ai==0.9.5``).

Owns the crawler lifecycle and process-local crawl capacity:

- One lazy shared ``AsyncWebCrawler`` per process; the first start is protected
  by an asyncio lock and Chromium is never started at import/startup.
- A fresh ``MemoryAdaptiveDispatcher`` per ``arun_many`` batch, sharing one
  process-wide ``RateLimiter`` for per-domain backoff.
- A process-wide semaphore held across the complete batch (batch startup and
  full stream consumption).
- ``CrawlerRunConfig`` with ``stream=True``, ``CacheMode.BYPASS``,
  ``check_robots_txt=True``, a finite page timeout and no shared session id.
- An overall exploration deadline: pages completed before the deadline are
  preserved, unfinished pages fail safely, and successful Markdown is never
  truncated (no character/output-size guard exists anywhere here).
- Final-URL validation through the URL-safety port before any successful
  outcome is returned — a public-to-private redirect can never leak content.

Residual risk: with the pinned Crawl4AI/Playwright API, redirects followed by
the browser cannot be pre-validated per hop and browser sub-requests cannot be
intercepted; the final URL is validated fail-closed instead. Network-level
egress restrictions remain the recommended defense in depth.
"""

import asyncio
import logging
import time
from collections import deque

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher, RateLimiter

from domain.entities.web_search import WebCrawlOutcome, WebCrawlStatus
from domain.errors.messages import ErrorMessage
from domain.logging.messages import LogMessage
from domain.ports.url_safety import UrlSafetyPort
from domain.ports.web_crawler import WebCrawlerPort

logger = logging.getLogger(__name__)


def _sanitized_crawl_error(error_message: object) -> str:
    """Translate a provider crawl failure into a safe, concise reason.

    Provider error text may contain secrets, stack traces or raw internals;
    only fixed sentences derived from local policy detection are emitted.
    """
    text = str(error_message or "").lower()
    if "robots" in text:
        return ErrorMessage.WEB_SEARCH_CRAWL_ROBOTS
    if "timeout" in text or "timed out" in text:
        return ErrorMessage.WEB_SEARCH_CRAWL_TIMED_OUT
    return ErrorMessage.WEB_SEARCH_CRAWL_FAILED


def _extract_markdown(result) -> str:
    """Return the complete selected Crawl4AI Markdown representation.

    Uses ``markdown.raw_markdown`` (the pinned 0.9.5 representation); falling
    back to ``str(markdown)`` (which yields raw_markdown) only when the
    attribute is absent. Never slices or truncates; ``""`` stays ``""``.
    """
    markdown = getattr(result, "markdown", None)
    raw = getattr(markdown, "raw_markdown", None)
    if raw is None:
        return str(markdown)
    return raw


class Crawl4AIWebCrawlerAdapter(WebCrawlerPort):
    """Bounded, process-local exploration of pre-validated URLs."""

    def __init__(
        self,
        *,
        max_concurrent_batches: int,
        max_sessions_per_batch: int,
        memory_threshold_percent: int,
        page_timeout_ms: int,
        overall_timeout_seconds: float,
        rate_limit_min_delay_seconds: float,
        rate_limit_max_delay_seconds: float,
        url_safety: UrlSafetyPort,
    ) -> None:
        self._max_sessions = max_sessions_per_batch
        self._memory_threshold = memory_threshold_percent
        self._page_timeout_ms = page_timeout_ms
        self._overall_timeout = overall_timeout_seconds
        self._url_safety = url_safety
        self._rate_limiter = RateLimiter(
            base_delay=(
                rate_limit_min_delay_seconds,
                rate_limit_max_delay_seconds,
            )
        )
        self._batch_slots = asyncio.Semaphore(max_concurrent_batches)
        self._crawler: AsyncWebCrawler | None = None
        self._crawler_lock = asyncio.Lock()

    async def crawl(self, urls: list[str]) -> list[WebCrawlOutcome]:
        """Explore one streaming batch under the process-wide ceiling.

        Args:
            urls: URL-safe destinations in search-result order.

        Returns:
            One outcome per URL, restored to the input order. Completed
            pages survive overall-deadline and per-page failures.
        """
        async with self._batch_slots:
            return await self._crawl_batch(list(urls))

    async def _crawl_batch(self, urls: list[str]) -> list[WebCrawlOutcome]:
        # Slot-based bookkeeping (one outcome per request index, keyed by
        # position, not by URL string) so duplicate URLs can never collapse.
        outcomes: list[WebCrawlOutcome | None] = [None] * len(urls)
        pending: deque[tuple[int, str]] = deque(enumerate(urls))
        started = time.monotonic()
        stream = None
        try:
            async with asyncio.timeout(self._overall_timeout):
                crawler = await self._ensure_crawler()
                dispatcher = MemoryAdaptiveDispatcher(
                    memory_threshold_percent=self._memory_threshold,
                    max_session_permit=self._max_sessions,
                    rate_limiter=self._rate_limiter,
                )
                run_config = CrawlerRunConfig(
                    stream=True,
                    cache_mode=CacheMode.BYPASS,
                    check_robots_txt=True,
                    page_timeout=self._page_timeout_ms,
                )
                stream = await crawler.arun_many(
                    list(urls), config=run_config, dispatcher=dispatcher
                )
                async for result in stream:
                    await self._record(result, outcomes, pending)
        except TimeoutError:
            logger.warning(
                LogMessage.CRAWL_BATCH_TIMEOUT, self._overall_timeout, len(pending)
            )
        finally:
            if stream is not None:
                try:
                    await stream.aclose()
                except Exception:
                    logger.exception(LogMessage.CRAWL_CLOSE_FAILED)
        # Anything still unresolved (deadline hit, or provider ended early)
        # fails safely so successful pages survive. Every request index is
        # either already recorded (removed from ``pending``) or still pending.
        for index, _url in pending:
            outcomes[index] = self._failed_outcome(
                urls[index], ErrorMessage.WEB_SEARCH_CRAWL_TIMED_OUT
            )
        resolved = [
            outcome
            if outcome is not None
            else self._failed_outcome(url, ErrorMessage.WEB_SEARCH_CRAWL_FAILED)
            for outcome, url in zip(outcomes, urls, strict=True)
        ]
        succeeded = sum(
            1 for outcome in resolved if outcome.status == WebCrawlStatus.SUCCESS
        )
        failed = sum(
            1 for outcome in resolved if outcome.status == WebCrawlStatus.FAILED
        )
        logger.info(
            LogMessage.CRAWL_BATCH_COMPLETED,
            succeeded,
            failed,
            (time.monotonic() - started) * 1000,
        )
        return resolved

    async def _ensure_crawler(self) -> AsyncWebCrawler:
        """Create and start the lazy shared crawler exactly once."""
        async with self._crawler_lock:
            if self._crawler is None:
                crawler = AsyncWebCrawler()
                try:
                    await crawler.start()
                except Exception:
                    logger.exception(LogMessage.CRAWLER_START_FAILED)
                    raise
                self._crawler = crawler
                logger.info(LogMessage.CRAWLER_STARTED)
            return self._crawler

    async def _record(
        self,
        result,
        outcomes: list[WebCrawlOutcome | None],
        pending: deque[tuple[int, str]],
    ) -> None:
        """Map one streamed result onto its requested URL slot."""
        slot = self._match_requested(result, pending)
        if slot is None:
            return
        index, requested = slot
        if not getattr(result, "success", False):
            reason = _sanitized_crawl_error(getattr(result, "error_message", None))
            outcomes[index] = self._failed_outcome(requested, reason)
            return
        # crawl4ai contract (async_webcrawler.py / CrawlResult): ``url`` is the
        # REQUESTED url and ``redirected_url`` is the browser's FINAL url after
        # redirects / JS navigation. The final destination is the one that
        # must be re-validated before any successful Markdown is returned —
        # validating ``url`` would only re-check the already-vetted original.
        final_url = str(
            getattr(result, "redirected_url", "") or getattr(result, "url", "") or ""
        )
        if not final_url:
            # Fail closed when the final destination cannot be determined.
            outcomes[index] = self._failed_outcome(
                requested, ErrorMessage.WEB_SEARCH_CRAWL_FAILED
            )
            return
        final_reason = await self._url_safety.validate(final_url)
        if final_reason is not None:
            logger.info(LogMessage.URL_SAFETY_REJECTED, "unsafe final url")
            outcomes[index] = self._failed_outcome(
                requested,
                ErrorMessage.WEB_SEARCH_CRAWL_REDIRECTED
                if final_url != requested
                else final_reason,
            )
            return
        outcomes[index] = WebCrawlOutcome(
            url=requested,
            status=WebCrawlStatus.SUCCESS,
            final_url=final_url,
            markdown=_extract_markdown(result),
            error=None,
        )

    @staticmethod
    def _failed_outcome(url: str, reason: str) -> WebCrawlOutcome:
        """Build a failed outcome keeping only the last validated URL."""
        return WebCrawlOutcome(
            url=url,
            status=WebCrawlStatus.FAILED,
            final_url=url,
            markdown=None,
            error=reason,
        )

    @staticmethod
    def _match_requested(
        result,
        pending: deque[tuple[int, str]],
    ) -> tuple[int, str] | None:
        """Find the still-open request slot matching a streamed result.

        Exact matches on the crawled URLs are preferred (``url`` = requested,
        ``redirected_url`` = final); when a redirect hides the original URL,
        the earliest open slot is assumed. Slots carry their request index so
        duplicate URLs stay position-independent.
        """
        for candidate in (
            str(getattr(result, "url", "") or ""),
            str(getattr(result, "redirected_url", "") or ""),
        ):
            if not candidate:
                continue
            for position, (index, url) in enumerate(pending):
                if url == candidate:
                    del pending[position]
                    return index, url
        if pending:
            return pending.popleft()
        return None

    async def close(self) -> None:
        """Close the shared crawler if it was created. Idempotent."""
        crawler, self._crawler = self._crawler, None
        if crawler is None:
            return
        try:
            await crawler.close()
        except Exception:
            logger.exception(LogMessage.CRAWLER_CLOSE_FAILED)
            return
        logger.info(LogMessage.CRAWLER_CLOSED)
