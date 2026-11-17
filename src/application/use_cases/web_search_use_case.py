"""Web-search use case: DuckDuckGo search plus bounded page exploration.

Owns input validation, provider-result ordering, URL safety gating, the
exploration selection policy (at most ``min(explore_top_n, usable results)``
URL-safe pages, unsafe pages fail locally without consuming a slot and are
never crawled), and partial-failure assembly. The tool layer stays thin.
"""

import asyncio
import logging
import time

from domain.entities.web_search import (
    WebCrawlOutcome,
    WebCrawlStatus,
    WebSearchPage,
    WebSearchReport,
    WebSearchResult,
)
from domain.errors.messages import ErrorMessage
from domain.errors.web import WebSearchValidationError
from domain.logging.messages import LogMessage
from domain.ports.url_safety import UrlSafetyPort
from domain.ports.web_crawler import WebCrawlerPort
from domain.ports.web_search import WebSearchPort

logger = logging.getLogger(__name__)

_MAX_RESULTS_LIMIT = 10
_EXPLORE_LIMIT = 5


class WebSearchUseCase:
    """Orchestrate provider search and safe, bounded page exploration."""

    def __init__(
        self,
        search: WebSearchPort,
        crawler: WebCrawlerPort,
        url_safety: UrlSafetyPort,
    ) -> None:
        self._search = search
        self._crawler = crawler
        self._url_safety = url_safety

    async def execute(
        self, query: str, max_results: int = 5, explore_top_n: int = 3
    ) -> WebSearchReport:
        """Run the validated search and exploration flow.

        Args:
            query: Raw user query (trimmed here; blanks are rejected).
            max_results: Provider result bound (1..10).
            explore_top_n: How many URL-safe results to explore (0..5). When
                0, the crawler is never created, started or invoked.

        Returns:
            The complete report in DuckDuckGo order with per-result outcomes.

        Raises:
            WebSearchValidationError: For blank queries or out-of-range
                counts (no external provider is contacted).
        """
        if not isinstance(query, str) or not query.strip():
            logger.info(LogMessage.WEB_SEARCH_VALIDATION_REJECTED)
            raise WebSearchValidationError(ErrorMessage.WEB_SEARCH_INVALID_QUERY)
        cleaned = query.strip()
        if (
            not isinstance(max_results, int)
            or not 1 <= max_results <= _MAX_RESULTS_LIMIT
        ):
            raise WebSearchValidationError(ErrorMessage.WEB_SEARCH_INVALID_MAX_RESULTS)
        if (
            not isinstance(explore_top_n, int)
            or not 0 <= explore_top_n <= _EXPLORE_LIMIT
        ):
            raise WebSearchValidationError(
                ErrorMessage.WEB_SEARCH_INVALID_EXPLORE_TOP_N
            )

        started = time.monotonic()
        pages = await self._search.search(cleaned, max_results)
        if explore_top_n == 0:
            # Search-only: no URL validation, no crawler slot, no Chromium.
            results = [
                WebSearchResult(
                    title=page.title,
                    url=page.url,
                    snippet=page.snippet,
                    crawl=WebCrawlOutcome(
                        url=page.url, status=WebCrawlStatus.NOT_EXPLORED
                    ),
                )
                for page in pages
            ]
            logger.info(
                LogMessage.WEB_SEARCH_COMPLETED,
                len(results),
                0,
                (time.monotonic() - started) * 1000,
            )
            return WebSearchReport(query=cleaned, results=results)
        results, selected = await self._select(pages, explore_top_n)
        if selected:
            await self._explore(results, selected)
        logger.info(
            LogMessage.WEB_SEARCH_COMPLETED,
            len(results),
            len(selected),
            (time.monotonic() - started) * 1000,
        )
        return WebSearchReport(query=cleaned, results=results)

    async def _select(
        self, pages: list[WebSearchPage], explore_top_n: int
    ) -> tuple[list[WebSearchResult], list[str]]:
        """Gate every result through the URL policy and pick the explored set.

        Unsafe URLs fail locally (never crawled, no capacity consumed); safe
        URLs fill at most ``explore_top_n`` slots in provider order; the rest
        stay ``not_explored``. The safety checks are independent I/O, so they
        run concurrently while preserving result order. A check that raises
        (e.g. a malformed hostname failing IDNA encoding) fails only its own
        result; it never aborts the search.
        """
        reasons = await asyncio.gather(
            *(self._url_safety.validate(page.url) for page in pages),
            return_exceptions=True,
        )
        results: list[WebSearchResult] = []
        selected: list[str] = []
        for page, checked in zip(pages, reasons, strict=True):
            # Normalize each check into a safe rejection reason (or None):
            # a raised exception must fail only its own result, never the
            # whole search, and its text never reaches the response.
            reason: str | None
            if isinstance(checked, BaseException):
                logger.info(LogMessage.URL_SAFETY_REJECTED, "validation error")
                reason = ErrorMessage.WEB_SEARCH_DNS_FAILED
            elif checked is not None:
                logger.info(LogMessage.URL_SAFETY_REJECTED, "unsafe result")
                reason = checked
            else:
                reason = None
            if reason is not None:
                results.append(
                    WebSearchResult(
                        title=page.title,
                        url=page.url,
                        snippet=page.snippet,
                        crawl=WebCrawlOutcome(
                            url=page.url,
                            status=WebCrawlStatus.FAILED,
                            error=reason,
                        ),
                    )
                )
                continue
            if len(selected) < explore_top_n:
                selected.append(page.url)
            results.append(
                WebSearchResult(
                    title=page.title,
                    url=page.url,
                    snippet=page.snippet,
                    crawl=WebCrawlOutcome(
                        url=page.url, status=WebCrawlStatus.NOT_EXPLORED
                    ),
                )
            )
        return results, selected

    async def _explore(
        self, results: list[WebSearchResult], selected: list[str]
    ) -> None:
        """Fill the selected results' outcomes, preserving result order."""
        outcomes = await self._crawler.crawl(list(selected))
        slot = 0
        for result in results:
            if slot < len(outcomes) and result.url == selected[slot]:
                result.crawl = outcomes[slot]
                slot += 1
