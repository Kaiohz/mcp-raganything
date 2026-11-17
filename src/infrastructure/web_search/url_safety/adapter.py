"""URL-safety adapter: SSRF guard for public-web exploration.

Policy (initial URL check, applied before every crawl):
- Absolute URL with scheme exactly ``http`` or ``https``.
- Valid hostname and port, no embedded username/password.
- Literal hostnames are vetted directly; non-literal hostnames are resolved
  with bounded asynchronous DNS and EVERY resolved address must be global /
  public. Loopback, private, link-local, multicast, reserved, unspecified,
  IPv4-mapped forms of forbidden addresses, localhost-style names and
  metadata-service destinations (e.g. 169.254.169.254) are rejected.

Residual risk (documented): DNS-rebinding between resolution and fetch and
headless-browser sub-requests cannot be fully enforced with the pinned
Crawl4AI API; network-level egress restrictions (deny private/metadata
routes) remain the recommended defense in depth. Only sanitized reasons are
returned; hosts, IPs and URLs never reach the tool error surface.
"""

import asyncio
import ipaddress
import logging
from ipaddress import IPv4Address, IPv6Address
from urllib.parse import urlsplit

from domain.errors.messages import ErrorMessage
from domain.logging.messages import LogMessage
from domain.ports.url_safety import UrlSafetyPort
from infrastructure.web_search.url_safety import socket as socket_seam

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_LOCAL_HOSTNAMES = frozenset({"localhost", "ip6-localhost", "ip6-loopback"})


def _is_forbidden(address: IPv4Address | IPv6Address) -> bool:
    """Return whether the address may not be reached from a public crawl."""
    if isinstance(address, IPv6Address) and address.ipv4_mapped is not None:
        return _is_forbidden(address.ipv4_mapped)
    return (
        not address.is_global
        or address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _hostname_is_local(hostname: str) -> bool:
    """Return whether the hostname is a localhost-style name."""
    lowered = hostname.lower()
    return (
        lowered in _LOCAL_HOSTNAMES
        or lowered.endswith(".localhost")
        or lowered.endswith(".local")
    )


class UrlSafetyAdapter(UrlSafetyPort):
    """Bounded, DNS-backed URL safety validation."""

    def __init__(self, dns_timeout_seconds: float = 5.0) -> None:
        self._dns_timeout = dns_timeout_seconds

    async def validate(self, url: str) -> str | None:
        """Check the absolute URL against the initial URL policy.

        Args:
            url: Candidate destination URL from search results.

        Returns:
            ``None`` when safe to fetch; otherwise a safe, concise reason.
        """
        parsed = urlsplit(url)
        if parsed.scheme not in _ALLOWED_SCHEMES:
            return ErrorMessage.WEB_SEARCH_SCHEME_NOT_ALLOWED
        hostname = parsed.hostname or ""
        if not hostname:
            return ErrorMessage.WEB_SEARCH_HOST_MISSING
        if parsed.username or parsed.password:
            return ErrorMessage.WEB_SEARCH_URL_CREDENTIALS
        try:
            _ = parsed.port
        except ValueError:
            return ErrorMessage.WEB_SEARCH_PORT_INVALID
        if _hostname_is_local(hostname):
            reason = ErrorMessage.WEB_SEARCH_LOCAL_HOST
            logger.info(LogMessage.URL_SAFETY_REJECTED, "local hostname")
            return reason
        try:
            literal = ipaddress.ip_address(hostname)
        except ValueError:
            return await self._validate_resolved(hostname, parsed.port)
        if _is_forbidden(literal):
            reason = ErrorMessage.WEB_SEARCH_NON_PUBLIC_ADDRESS
            logger.info(LogMessage.URL_SAFETY_REJECTED, "non-public address")
            return reason
        return None

    async def _validate_resolved(self, hostname: str, port: int | None) -> str | None:
        """Resolve the hostname with bounded async DNS and vet every answer."""
        try:
            async with asyncio.timeout(self._dns_timeout):
                # asyncio.to_thread uses the bounded process-default executor:
                # DNS resolution is short-lived and this leaves the DDGS path's
                # dedicated executor solely for blocking provider calls.
                answers = await asyncio.to_thread(
                    socket_seam.getaddrinfo, hostname, port or 0
                )
        except TimeoutError:
            reason = ErrorMessage.WEB_SEARCH_DNS_TIMEOUT
            logger.info(LogMessage.URL_SAFETY_REJECTED, "dns timeout")
            return reason
        except OSError:
            reason = ErrorMessage.WEB_SEARCH_DNS_FAILED
            logger.info(LogMessage.URL_SAFETY_REJECTED, "dns failure")
            return reason
        except ValueError:
            # Malformed hostname (e.g. an IDNA label longer than 63 chars
            # raises UnicodeEncodeError, a ValueError, not an OSError) —
            # fail closed; the use case isolates this to the single result.
            reason = ErrorMessage.WEB_SEARCH_DNS_FAILED
            logger.info(LogMessage.URL_SAFETY_REJECTED, "malformed hostname")
            return reason
        addresses: list[IPv4Address | IPv6Address] = []
        for answer in answers:
            sockaddr = answer[4] if len(answer) > 4 else None
            if not sockaddr:
                return ErrorMessage.WEB_SEARCH_DNS_FAILED
            try:
                addresses.append(ipaddress.ip_address(str(sockaddr[0])))
            except ValueError:
                return ErrorMessage.WEB_SEARCH_DNS_FAILED
        if not addresses or any(_is_forbidden(address) for address in addresses):
            reason = ErrorMessage.WEB_SEARCH_NON_PUBLIC_ADDRESS
            logger.info(LogMessage.URL_SAFETY_REJECTED, "non-public dns answer")
            return reason
        return None
