"""Outbound port for the public-web search provider (DuckDuckGo via DDGS)."""

from abc import ABC, abstractmethod

from domain.entities.web_search import WebSearchPage


class WebSearchPort(ABC):
    """Boundary for provider text search used by the web-search use case."""

    @abstractmethod
    async def search(self, query: str, max_results: int) -> list[WebSearchPage]:
        """Search the provider for the validated query.

        Implementations must run the synchronous provider SDK off the event
        loop inside a bounded window and map provider entries (``title``,
        ``href``, ``body``) to :class:`WebSearchPage`, silently dropping
        entries without a usable URL. Provider ordering must be preserved.

        Args:
            query: Non-empty, trimmed user query (never logged).
            max_results: Requested provider result bound (1..10).

        Returns:
            Provider results in provider order.

        Raises:
            WebSearchTimeoutError: If the bounded operation timed out.
            WebSearchProviderError: For sanitized provider failures.
        """
