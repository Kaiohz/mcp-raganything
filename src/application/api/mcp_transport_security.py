"""Pure-ASGI MCP transport-security guard (REQ-014 / M-1).

Applies an explicit ``mcp``-SDK
:class:`~mcp.server.transport_security.TransportSecuritySettings` to each
FastMCP ``http_app`` — fastmcp 3.3.1 constructs its streamable transports
WITHOUT any security settings (the SDK then disables DNS-rebinding
protection by default), which let every non-localhost request Host through.
This wrapper keeps DNS-rebinding protection ENABLED and lets only the
configured allow-lists through:

* a request Host in ``MCP_ALLOWED_HOSTS`` (exact, default-port ``:443`` /
  ``:80`` sibling, or ``host:*`` wildcard) passes;
* a foreign Host is rejected with the SDK's own ``421 Invalid Host header``;
* a POST without an ``application/json`` content type is rejected with the
  SDK's ``400 Invalid Content-Type header``;
* an Origin header outside the Origin allow-list is rejected with the SDK's
  ``403 Invalid Origin header``.

Origin policy (M-1 — the SDK matches an Origin EXACTLY and has no global
wildcard, so an empty ``allowed_origins`` rejects EVERY browser Origin,
30/403-ing every browser-origin MCP client): the Origin allow-list comes
from ``MCP_ALLOWED_ORIGINS`` falling back to the app's CORS policy
(``ALLOWED_ORIGINS``). The CORS ``*`` wildcard cannot be expressed to the
SDK, so it is translated into "do not validate the Origin" — the same
semantic the CORS middleware applies.

Both allow-lists are read from ``settings`` at REQUEST time (per-request
``TransportSecuritySettings`` construction is negligible and keeps the
configmap authoritative), and plain-hostname entries are expanded with their
default-port siblings so the SDK's exact Host matcher accepts
``host:443`` / ``host:80`` (QA BUG-002) while a foreign Host keeps failing
closed.
"""

import logging
import re
from typing import TYPE_CHECKING

from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
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

#: The CORS-style wildcard: translated into "Origin not validated" (the SDK
#: has no wildcard — see the module docstring).
_ORIGIN_WILDCARD = "*"


def _rejection_reason(response: Response) -> str:
    """Return a short, non-sensitive reason snippet from the SDK's response."""
    body = bytes(response.body)
    return body.decode("latin-1", "replace").strip()[:120]


def _scope_without_origin(scope: Scope) -> Scope:
    """Return a copy of ``scope`` WITHOUT the ``Origin`` header.

    The SDK's matcher has no wildcard mode: an empty ``allowed_origins``
    rejects EVERY Origin, and a missing Origin is treated as same-origin and
    always passes. Mirrors the app's own CORS policy ("*" = any origin).
    """
    copied = dict(scope)
    copied["headers"] = [
        (key, value)
        for key, value in scope.get("headers", [])
        if key.decode("latin-1").lower() != "origin"
    ]
    return copied


class McpTransportSecurityGuard:
    """ASGI wrapper applying the SDK transport security to one FastMCP app.

    Args:
        app: The inner ASGI app (the FastMCP ``http_app``).
        settings: The ``AppConfig`` instance; its ``MCP_ALLOWED_HOSTS``,
            ``MCP_ALLOWED_ORIGINS`` and ``ALLOWED_ORIGINS`` attributes are
            read per request.
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

    def _origin_wildcard(self, origins: list[str]) -> bool:
        """Whether the configured Origin policy allows ANY browser origin.

        A ``*`` entry (either in ``MCP_ALLOWED_ORIGINS`` or in the inherited
        ``ALLOWED_ORIGINS`` CORS policy) mirrors the CORS wildcard semantics.
        """
        return _ORIGIN_WILDCARD in origins

    async def _validate(self, scope: Scope) -> Response | None:
        """Run the SDK's header validation; return the rejection or ``None``.

        A fresh ``TransportSecurityMiddleware`` per request keeps the
        ``MCP_ALLOWED_HOSTS`` configmap authoritative at all times. The
        Origin policy inherits the app CORS policy when the dedicated
        ``MCP_ALLOWED_ORIGINS`` setting is unset (M-1).
        """
        origins = list(
            self._settings.MCP_ALLOWED_ORIGINS or self._settings.ALLOWED_ORIGINS
        )
        wildcard = self._origin_wildcard(origins)
        middleware = TransportSecurityMiddleware(
            TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=transport_allowed_hosts(
                    list(self._settings.MCP_ALLOWED_HOSTS)
                ),
                allowed_origins=[origin for origin in origins if origin],
            )
        )
        # A wildcard Origin policy is translated by stripping the Origin
        # header before validation: the SDK treats a missing Origin as
        # same-origin (always allowed) — the "*" semantic.
        request = Request(_scope_without_origin(scope) if wildcard else scope)
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


# ---------------------------------------------------------------------------
# Streamable-HTTP request-body cap (413 on oversized MCP POSTs)
# ---------------------------------------------------------------------------

#: The mcp-SDK default body cap the SDK falls back to (4 MiB).
_SDK_DEFAULT_MAX_REQUEST_BODY_SIZE = 4 * 1024 * 1024


def mcp_max_request_body_size() -> int:
    """Body cap for MCP POSTs, derived from the FILES upload budget (REQ-092).

    A temporary upload carries its bytes base64-encoded inside the JSON-RPC
    envelope (x4/3) plus the envelope overhead, so the transport cap never
    sits BELOW fastmcp's 4 MiB SDK default. Read at CALL time — the manager
    is built inside the app lifespan, after settings are loaded.
    """
    from dependencies import minio_config

    return max(
        _SDK_DEFAULT_MAX_REQUEST_BODY_SIZE,
        int(minio_config.FILES_MAX_UPLOAD_SIZE * 4 / 3) + 2 * 1024 * 1024,
    )


class LargeBodyStreamableHTTPSessionManager(StreamableHTTPSessionManager):
    """Session manager with a REQUEST-BODY cap that fits the FILES budget.

    fastmcp 3.3.1 builds its ``StreamableHTTPSessionManager`` inside the app
    lifespan WITHOUT forwarding ``max_request_body_size`` — the mcp-SDK's
    4 MiB default answered ``413 Content Too Large`` BEFORE any tool ran,
    so every ``temporary_upload_file`` call with a payload above ~3 MB
    decoded (base64-inflated past 4 MiB) was rejected with a raw
    ``HTTPStatusError`` that surfaced on the caller as an opaque
    "unhandled errors in a TaskGroup (1 sub-exception)".
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs.setdefault("max_request_body_size", mcp_max_request_body_size())
        super().__init__(*args, **kwargs)


def apply_mcp_max_request_body_size() -> None:
    """Patch fastmcp's session-manager class so every mount lifts the cap.

    fastmcp 3.3.1 instantiates the manager inside the app lifespan through
    the ``StreamableHTTPSessionManager`` symbol of
    :mod:`fastmcp.server.http`; replacing that symbol makes all four mounts
    (files / classical / web / memories) accept MCP POSTs up to
    :func:`mcp_max_request_body_size` instead of the SDK's 4 MiB default.
    """
    import fastmcp.server.http as _fastmcp_http

    _fastmcp_http.StreamableHTTPSessionManager = LargeBodyStreamableHTTPSessionManager
