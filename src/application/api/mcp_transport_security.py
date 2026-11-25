"""Pure-ASGI MCP transport-security guard (REQ-014).

Applies an explicit ``mcp``-SDK
:class:`~mcp.server.transport_security.TransportSecuritySettings` to each
FastMCP ``http_app`` — fastmcp 3.3.1 constructs its streamable transports
WITHOUT any security settings (the SDK then disables DNS-rebinding
protection by default), which let every non-localhost request Host through.
This wrapper keeps DNS-rebinding protection ENABLED and lets only the
configured allow-list through:

* a request Host in ``MCP_ALLOWED_HOSTS`` (exact, default-port ``:443`` /
  ``:80`` sibling, or ``host:*`` wildcard) passes;
* a foreign Host is rejected with the SDK's own ``421 Invalid Host header``;
* a POST without an ``application/json`` content type is rejected with the
  SDK's ``400 Invalid Content-Type header``;
* an Origin header outside the allowed origin list is rejected with the
  SDK's ``403``.

The host list is read from ``settings`` at REQUEST time (per-request
``TransportSecuritySettings`` construction is negligible and keeps the
configmap authoritative), and plain-hostname entries are expanded with their
default-port siblings so the SDK's exact Host matcher accepts
``host:443`` / ``host:80`` (QA BUG-002) while a foreign Host keeps failing
closed.
"""

import logging
import re
from typing import TYPE_CHECKING

from mcp.server.transport_security import (
    TransportSecurityMiddleware,
    TransportSecuritySettings,
)
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

from domain.logging.messages import LogMessage

if TYPE_CHECKING:  # pragma: no cover — typing only, avoids a runtime import
    from config import AppConfig

logger = logging.getLogger(__name__)

#: Plain-hostname allow-list entries (no wildcard, no port, no IPv6 brackets)
#: — the only shape the transport-security list may grow default-port
#: siblings for (QA BUG-002).
_PLAIN_HOSTNAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _rejection_reason(response: Response) -> str:
    """Return a short, non-sensitive reason snippet from the SDK's response."""
    body = bytes(response.body)
    return body.decode("latin-1", "replace").strip()[:120]


class McpTransportSecurityGuard:
    """ASGI wrapper applying the SDK transport security to one FastMCP app.

    Args:
        app: The inner ASGI app (the FastMCP ``http_app``).
        settings: The ``AppConfig`` instance; its ``MCP_ALLOWED_HOSTS``
            attribute is read per request.
    """

    def __init__(self, app: ASGIApp, settings: "AppConfig") -> None:
        self._app = app
        self._settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        rejection = await self._validate(scope)
        if rejection is None:
            await self._app(scope, receive, send)
            return
        logger.warning(LogMessage.MCP_TRANSPORT_REJECTED, _rejection_reason(rejection))
        await rejection(scope, receive, send)

    async def _validate(self, scope: Scope) -> Response | None:
        """Run the SDK's header validation; return the rejection or ``None``.

        A fresh ``TransportSecurityMiddleware`` per request keeps the
        ``MCP_ALLOWED_HOSTS`` configmap authoritative at all times.
        """
        middleware = TransportSecurityMiddleware(
            TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=transport_allowed_hosts(
                    list(self._settings.MCP_ALLOWED_HOSTS)
                ),
            )
        )
        request = Request(scope)
        is_post = scope.get("method", "").upper() == "POST"
        return await middleware.validate_request(request, is_post=is_post)


def transport_allowed_hosts(allowed_hosts: list[str]) -> list[str]:
    """Expand the allow-list for the ``mcp`` SDK's exact Host matcher.

    The SDK matches the request ``Host`` EXACTLY: a bare entry never covers
    the explicit port form (``Host: raganything.soludev.tech:443`` would be
    421'd although the host itself is allow-listed — QA BUG-002), and its
    ``host:*`` wildcard never covers the port-LESS form. Each plain-hostname
    entry therefore ships with its two default-port siblings — additive by
    construction, so a still-foreign Host keeps answering 421 with
    DNS-rebinding protection ENABLED. Wildcard entries (``host:*``),
    explicit-port entries and IPv6 literals pass through unchanged.
    """
    normalized = list(allowed_hosts)
    for entry in allowed_hosts:
        if _PLAIN_HOSTNAME.match(entry):
            for suffix in (":443", ":80"):
                port_form = entry + suffix
                if port_form not in normalized:
                    normalized.append(port_form)
    return normalized
