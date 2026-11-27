"""MCP ``web_search`` tool (server ``Web``, mounted at ``/web/mcp``).

Thin FastMCP inbound adapter: validates nothing itself besides translation —
all business behavior lives in :class:`WebSearchUseCase`. Exactly one public
tool is exposed; separate ``crawl_page`` tools or RAG indexing do not exist
in this feature. Domain failures are translated into safe, bounded tool
errors; no raw exception text, HTML, screenshots, cookies or headers ever
reach the output.
"""

import logging

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from application.responses.web_search_response import WebSearchResponse
from dependencies import get_web_search_use_case
from domain.errors.base import DomainError
from domain.errors.messages import ErrorMessage
from domain.logging.messages import LogMessage

logger = logging.getLogger(__name__)

mcp_web = FastMCP("Web")


@mcp_web.tool()
async def web_search(
    query: str,
    max_results: int = 5,
    explore_top_n: int = 3,
) -> WebSearchResponse:
    """Search the public web across ddgs backends and explore top results.

    Tries the configured ``WEB_SEARCH_BACKENDS`` backends in order (the first
    one returning entries wins) to ride out single-provider anti-bot
    challenges, validates every result URL against the public-destination
    policy, and explores up to ``explore_top_n`` URL-safe results
    concurrently with asynchronous crawling. Each explored page returns its
    Markdown — cut at ``WEB_SEARCH_PAGE_MAX_CHARS`` characters with the
    ``truncated`` flag set when the configured cap applies (default:
    complete, uncapped Markdown). Crawled pages are never indexed into
    storage and never trigger automatic memory writes.

    Args:
        query: Search query (trimmed; blank queries are rejected).
        max_results: Number of search results to request (1..10).
        explore_top_n: Number of URL-safe results to explore (0..5). Use 0
            for search-only responses.

    Returns:
        The :class:`WebSearchResponse` contract: results in provider order,
        each with its exploration outcome (``not_explored``, ``success`` with
        Markdown — possibly empty or ``truncated`` — or ``failed`` with a
        safe, concise reason).

    Raises:
        ToolError: Sanitized validation, provider, timeout or crawl errors.
    """
    try:
        report = await get_web_search_use_case().execute(
            query=query, max_results=max_results, explore_top_n=explore_top_n
        )
    except DomainError as error:
        raise ToolError(error.detail) from None
    except Exception:
        logger.exception(LogMessage.WEB_SEARCH_UNEXPECTED)
        raise ToolError(ErrorMessage.WEB_SEARCH_FAILED) from None
    return WebSearchResponse(**report.model_dump())
