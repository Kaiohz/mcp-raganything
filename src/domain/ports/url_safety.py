"""Outbound port for URL safety validation (SSRF guard)."""

from abc import ABC, abstractmethod


class UrlSafetyPort(ABC):
    """Boundary that decides whether a destination may be fetched.

    The policy only allows absolute ``http``/``https`` URLs to global
    (public) destinations, rejecting local, private, link-local, multicast,
    reserved, unspecified and metadata destinations — including any resolved
    DNS answer. Implementations must perform bounded asynchronous resolution
    and return safe, sanitized reasons only (never hosts, IPs or URLs).
    """

    @abstractmethod
    async def validate(self, url: str) -> str | None:
        """Validate an absolute URL against the initial URL policy.

        Args:
            url: Candidate destination URL from search results.

        Returns:
            ``None`` when the URL is allowed to be fetched, otherwise a safe,
            concise rejection reason for the tool error surface.
        """
