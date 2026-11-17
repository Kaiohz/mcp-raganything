"""DuckDuckGo search adapter (pinned ``ddgs==9.16.1``).

Runs the synchronous DDGS SDK on a dedicated bounded ``ThreadPoolExecutor``
(never the default executor, never the event loop) behind a process-wide
semaphore, with an outer asyncio timeout around the complete search
operation. A timed-out (or cancelled) awaitable does not release capacity
while its worker thread is still running: release happens only when the
thread actually finishes, so a saturated process never starts work above its
configured ceiling.

``DDGS`` is imported as a module-level symbol so tests can patch the adapter
locally without touching the library namespace.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from domain.entities.web_search import WebSearchPage
from domain.errors.messages import ErrorMessage
from domain.errors.web import WebSearchProviderError, WebSearchTimeoutError
from domain.logging.messages import LogMessage
from domain.ports.web_search import WebSearchPort

logger = logging.getLogger(__name__)


def _mark_completed(loop: asyncio.AbstractEventLoop, event: asyncio.Event) -> None:
    """Thread-safe completion hook for a blocking search worker."""

    def _set() -> None:
        if not loop.is_closed():
            event.set()

    if not loop.is_closed():
        loop.call_soon_threadsafe(_set)


class DDGSSearchAdapter(WebSearchPort):
    """Search DuckDuckGo through DDGS with bounded, dedicated capacity.

    Attributes:
        _timeout: Outer timeout (seconds) around the complete search op.
        _executor: Dedicated thread pool (``WEB_SEARCH_EXECUTOR_WORKERS``).
        _semaphore: Process-wide cap on active DDGS calls in flight.
    """

    def __init__(
        self,
        timeout_seconds: float,
        max_concurrency: int,
        executor_workers: int,
    ) -> None:
        self._timeout = timeout_seconds
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=executor_workers,
            thread_name_prefix="raganything-ddgs",
        )
        self._semaphore = asyncio.Semaphore(max_concurrency)
        # Strong references for fire-and-forget release tasks scheduled after
        # a timeout, so the interpreter cannot garbage-collect a still-pending
        # task before it releases the semaphore (each task removes itself).
        self._pending_releases: set[asyncio.Task[None]] = set()

    async def search(self, query: str, max_results: int) -> list[WebSearchPage]:
        """Run one bounded DuckDuckGo search for the validated query.

        Args:
            query: Trimmed, non-empty user query.
            max_results: Requested result count (1..10).

        Returns:
            Provider pages in DuckDuckGo order; entries without a usable URL
            are dropped.

        Raises:
            WebSearchTimeoutError: Outer timeout (capacity stays accounted
                for until the worker thread really finishes).
            WebSearchProviderError: Sanitized provider failure.
        """
        try:
            pages = await self._search_within_deadline(query, max_results)
        except WebSearchTimeoutError:
            logger.warning(LogMessage.WEB_SEARCH_TIMED_OUT, self._timeout)
            raise
        except TimeoutError:
            # ``asyncio.timeout`` raises the builtin TimeoutError; translate it
            # into the dedicated, sanitized domain error so the tool surfaces
            # the specific timeout wording (never a generic failure + traceback).
            logger.warning(LogMessage.WEB_SEARCH_TIMED_OUT, self._timeout)
            raise WebSearchTimeoutError(ErrorMessage.WEB_SEARCH_TIMED_OUT) from None
        except WebSearchProviderError:
            logger.warning(LogMessage.WEB_SEARCH_PROVIDER_FAILED)
            raise
        return pages

    async def _search_within_deadline(
        self, query: str, max_results: int
    ) -> list[WebSearchPage]:
        loop = asyncio.get_running_loop()
        async with asyncio.timeout(self._timeout):
            await self._semaphore.acquire()
            future = self._executor.submit(self._search_blocking, query, max_results)
            completed = asyncio.Event()
            future.add_done_callback(lambda _f: _mark_completed(loop, completed))
            try:
                # Plain-future completion signal: cancelling the awaiting task
                # must never cancel (nor early-release) the worker thread.
                await completed.wait()
            except asyncio.CancelledError:
                # Outer timeout or caller cancellation while the worker thread
                # may still be running: release capacity only on true
                # completion, then let asyncio.timeout translate the cancel.
                self._release_when_completed(completed)
                raise
            error = future.exception()
            if error is not None:
                self._semaphore.release()
                # The provider exception is never forwarded/logged in detail.
                raise WebSearchProviderError(
                    ErrorMessage.WEB_SEARCH_PROVIDER_UNAVAILABLE
                ) from None
            raw_entries = future.result()
            self._semaphore.release()
            return self._map_entries(raw_entries)

    def _release_when_completed(self, completed: asyncio.Event) -> None:
        """Schedule the semaphore release for actual worker completion."""
        pending_releases = self._pending_releases

        async def _release() -> None:
            try:
                await completed.wait()
                self._semaphore.release()
            finally:
                pending_releases.discard(task)

        task = asyncio.get_running_loop().create_task(_release())
        pending_releases.add(task)

    @staticmethod
    def _search_blocking(query: str, max_results: int) -> list[dict]:
        """Synchronous DDGS call executed inside the dedicated executor.

        A fresh DDGS client is created per search operation; the explicit
        ``backend="duckduckgo"`` and requested ``max_results`` are the only
        search parameters passed to the provider.

        Empty-result mapping (ddgs 9.16.1 evidence: ``BaseSearchEngine.request``
        returns ``None`` for ANY non-200 response — including DuckDuckGo's HTTP
        202 anti-bot challenge page — and ``DDGS._search_sync`` then raises
        ``DDGSException("No results found.")`` with no distinguishing signal
        when every engine returns nothing). The adapter therefore keeps the
        genuine-empty ``[]`` mapping (AC-22) and logs a warning so operators
        know an empty result set may also mean the provider is challenging or
        unreachable from this network; connection and timeout failures surface
        as distinct sanitized provider errors instead.
        """
        client = DDGS()
        try:
            results = client.text(query, backend="duckduckgo", max_results=max_results)
            if not results:
                logger.warning(LogMessage.WEB_SEARCH_PROVIDER_NO_RESULTS, max_results)
            return results
        except DDGSException as exc:
            if "no results found" in str(exc).lower():
                logger.warning(LogMessage.WEB_SEARCH_PROVIDER_NO_RESULTS, max_results)
                return []
            raise

    @staticmethod
    def _map_entries(raw_entries: list[dict]) -> list[WebSearchPage]:
        """Map provider entries, dropping those without a usable URL."""
        pages: list[WebSearchPage] = []
        for entry in raw_entries:
            if not isinstance(entry, dict):
                continue
            url = str(entry.get("href") or "").strip()
            if not url:
                continue
            pages.append(
                WebSearchPage(
                    title=str(entry.get("title") or ""),
                    url=url,
                    snippet=str(entry.get("body") or ""),
                )
            )
        return pages

    def close(self) -> None:
        """Shut down the dedicated executor. Idempotent, non-blocking."""
        self._executor.shutdown(wait=False)
