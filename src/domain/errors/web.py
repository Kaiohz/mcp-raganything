"""Web-search domain errors (DuckDuckGo search + Crawl4AI exploration).

Error details are always safe, fixed, sanitized sentences: no provider
messages, stack traces, URLs, IPs or credentials may reach the tool output.
"""

from domain.errors.base import DomainError
from domain.errors.codes import ErrorCode


class WebSearchError(DomainError):
    """Base class for web-search domain failures."""


class WebSearchValidationError(WebSearchError):
    """Web-search arguments rejected before any external provider call."""

    status_code = ErrorCode.BAD_REQUEST


class WebSearchTimeoutError(WebSearchError):
    """The bounded search operation exceeded its outer timeout.

    The underlying synchronous worker may still be running; capacity is
    accounted for by the adapter until the thread actually finishes.
    """

    status_code = ErrorCode.SERVICE_UNAVAILABLE


class WebSearchProviderError(WebSearchError):
    """The search provider (DDGS) failed; the cause is never forwarded."""

    status_code = ErrorCode.SERVICE_UNAVAILABLE
