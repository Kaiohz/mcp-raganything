"""HTTP-level MCP auth guard + OAuth2 discovery derivation (REQ-001/002/017).

Mirrors composable-agents' ``McpAuthGuard`` pattern (READ-ONLY reference):
every request entering an MCP mount is authenticated at the HTTP level
BEFORE any MCP method runs — including ``initialize`` — and a rejected
request gets a plain HTTP 401 carrying the OAuth2 discovery challenge,
so MCP clients can start the authorization flow (401 → protected-resource
metadata → authorization server).

The discovery URL is never derived from a raw, spoofable request ``Host``
(M-2 hardening): :func:`public_discovery_origin` is the single derivation
point shared by the 401 challenge and the public ``/.well-known
/oauth-protected-resource`` route. The path-normalization rewrite keeps the
exact mount path (and its trailing-slash form) served DIRECTLY — no
Starlette Mount 307 and no ``http://`` scheme downgrade behind the
TLS-terminating proxy (REQ-015).
"""

import json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from starlette.types import ASGIApp, Receive, Scope, Send

from domain.entities.auth.auth_context import AuthContext
from domain.errors.messages import ErrorMessage
from domain.logging.messages import LogMessage

if TYPE_CHECKING:  # pragma: no cover — typing only, avoids a runtime import
    from config import AppConfig

logger = logging.getLogger(__name__)

#: Public OAuth2 protected-resource metadata document (MCP authorization
#: spec 2025-06-18): the discovery target advertised by the 401 challenge and
#: served by the public route wired in ``main.py`` (REQ-002-equivalent).
WELL_KNOWN_PROTECTED_RESOURCE_PATH = "/.well-known/oauth-protected-resource"

#: 401 response body — generic, non-enumerating (the centralized
#: ``ErrorMessage.AUTH_INVALID_CREDENTIALS`` text, no user data, no tools).
_UNAUTHORIZED_BODY = json.dumps(
    {"detail": ErrorMessage.AUTH_INVALID_CREDENTIALS.value}
).encode()

#: Default-port suffixes normalized on the request ``Host`` (QA BUG-002): an
#: ingress may forward the port the client dialed — ``host`` and ``host:443``
#: (https) / ``host:80`` (http) are the same endpoint behind the proxy.
_DEFAULT_PORT_SUFFIXES = (":443", ":80")

Authenticator = Callable[[str | None, str | None], Awaitable[AuthContext | None]]


def _without_default_port(host: str) -> str:
    """Return ``host`` without a trailing default port (``:443`` / ``:80``).

    QA BUG-002: behind the TLS-terminating proxy ``host`` and ``host:443``
    are the same endpoint, so the default-port suffix must neither turn an
    allow-listed Host into a foreign one nor leak into the advertised
    discovery origin. Any OTHER explicit port is preserved (only the
    ``host:*`` wildcard covers it).
    """
    for suffix in _DEFAULT_PORT_SUFFIXES:
        if host.endswith(suffix):
            return host[: -len(suffix)]
    return host


def _host_allowed(host: str, allowed_hosts: list[str]) -> bool:
    """Whether ``host`` matches the derivation allow-list shape.

    Exact entries plus ``<host>:*`` wildcard entries (any port). An exact
    entry also covers the same host presented with an explicit DEFAULT port
    (``host:443`` https / ``host:80`` http — QA BUG-002, the Host
    normalization strips the suffix before the match); every OTHER port form
    still requires the ``host:*`` wildcard, so a foreign host keeps failing
    closed.
    """
    bare_host = _without_default_port(host)
    for allowed in allowed_hosts:
        if allowed in (host, bare_host):
            return True
        if allowed.endswith(":*") and host.startswith(allowed[:-2] + ":"):
            return True
    return False


def public_discovery_origin(
    request_host: str,
    *,
    configured_base_url: str,
    allowed_hosts: list[str],
) -> str | None:
    """Return the canonical PUBLIC base origin used by the discovery documents.

    Single derivation point shared by the 401 ``WWW-Authenticate`` challenge
    and the well-known protected-resource route (M-2: the request ``Host``
    header is client-supplied and is never echoed unvalidated).

    Precedence:

    1. ``MCP_PUBLIC_BASE_URL`` when configured (always wins, trailing slash
       stripped);
    2. otherwise the request ``Host`` is used ONLY when it is explicitly
       allow-listed in ``MCP_ALLOWED_HOSTS`` (exact entry or ``host:*``
       wildcard) — TLS terminates at the ingress proxy, so the scheme is
       always ``https``; a default-port suffix (``host:443`` / ``host:80``,
       QA BUG-002) is matched against the bare entry and stripped from the
       advertised origin;
    3. otherwise ``None``: no trustworthy origin can be advertised, so the
       callers must serve a challenge/metadata WITHOUT a discovery URL
       instead of echoing a spoofable host.

    Args:
        request_host: The raw request ``Host`` header value (may be empty).
        configured_base_url: The ``MCP_PUBLIC_BASE_URL`` setting value.
        allowed_hosts: The ``MCP_ALLOWED_HOSTS`` allow-list.

    Returns:
        The public origin WITHOUT a trailing slash (``https://<host>``), or
        ``None`` when no trustworthy origin can be derived.
    """
    configured = (configured_base_url or "").strip().rstrip("/")
    if configured:
        return configured
    host = (request_host or "").strip()
    if not host or not _host_allowed(host, allowed_hosts):
        return None
    return f"https://{_without_default_port(host)}"


def www_authenticate_challenge(public_origin: str) -> str:
    """Build the MCP OAuth2 discovery challenge header value (REQ-001).

    Args:
        public_origin: The canonical public origin from
            :func:`public_discovery_origin` (scheme + host, no trailing
            slash) — never a raw, unvalidated request ``Host``.

    Returns:
        The ``WWW-Authenticate`` challenge pointing at the public
        protected-resource metadata document.
    """
    return (
        f'Bearer resource_metadata="{public_origin}'
        f'{WELL_KNOWN_PROTECTED_RESOURCE_PATH}"'
    )


class McpAuthGuard:
    """Pure-ASGI guard authenticating every request to one MCP mount.

    HTTP 401 semantics (REQ-001/REQ-017): the raw JSON-RPC probe (initialize
    / tools/call) observes a 401 status instead of a redirect or a JSON-RPC
    error frame that protocol clients could misread. On success the guard
    delegates to ONE MCP mount path and forwards the request to the wrapped
    ASGI app (the app's own ``McpApiKeyMiddleware`` stays behind as
    defense-in-depth).

    The 401's ``WWW-Authenticate`` discovery URL is derived through
    :func:`public_discovery_origin` reading ``MCP_PUBLIC_BASE_URL`` and
    ``MCP_ALLOWED_HOSTS`` from ``settings`` at REQUEST time (the configmap
    is authoritative per request, not per process boot).

    Args:
        app: The wrapped inner ASGI app (the FastMCP http_app behind a
            transport-security guard).
        authenticate: The dual-auth callable resolving ``(authorization,
            api_key)`` headers into an :class:`AuthContext` (or ``None``) —
            the wired :meth:`McpApiKeyMiddleware.authenticate_headers` seam.
        mount_path: This mount's public path (e.g. ``/files/mcp``). The
            guard rewrites the exact path and its trailing-slash form to
            ``/`` so the inner app serves the request DIRECTLY (REQ-015).
        settings: The ``AppConfig`` instance; its ``MCP_PUBLIC_BASE_URL``
            and ``MCP_ALLOWED_HOSTS`` attributes are read per request.
    """

    def __init__(
        self,
        app: ASGIApp,
        authenticate: Authenticator,
        *,
        mount_path: str,
        settings: "AppConfig",
    ) -> None:
        self._app = app
        self._authenticate = authenticate
        self._mount_path = mount_path.rstrip("/") or "/"
        self._settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        await self._handle_http(scope, receive, send)

    async def _handle_http(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Authenticate one HTTP request, then forward it to the MCP app."""
        # Path normalization (REQ-015): the exact mount path (and its
        # trailing-slash form) is rewritten to "/" so the inner Starlette
        # app's single route always matches — no redirect round-trips and no
        # ``http://`` downgrade behind the proxy; the auth guard still ran
        # FIRST on the original request.
        if scope.get("path", "").rstrip("/") == self._mount_path:
            scope = dict(scope)
            scope["path"] = "/"
        challenge = self._discovery_challenge(_asgi_host(scope))
        headers = _asgi_headers(scope)
        try:
            ctx = await self._authenticate(
                headers.get("authorization"), headers.get("x-api-key")
            )
        except Exception:  # noqa: BLE001 — never leak auth internals over the wire
            logger.exception(LogMessage.MCP_REQUEST_DENIED)
            await self._send_unauthorized(send, challenge)
            return
        if ctx is None:
            logger.warning(LogMessage.MCP_REQUEST_DENIED)
            await self._send_unauthorized(send, challenge)
            return
        logger.info(LogMessage.AUTH_CREDENTIALS_VALIDATED, ctx.user_id, ctx.method)
        await self._app(scope, receive, send)  # type: ignore[operator]

    def _discovery_challenge(self, request_host: str) -> str | None:
        """Build the ``WWW-Authenticate`` value for one request, or ``None``.

        ``None`` means no trustworthy public origin could be derived (the
        configured ``MCP_PUBLIC_BASE_URL`` is unset and the request ``Host``
        is not allow-listed) — the 401 is then served WITHOUT a discovery
        pointer instead of echoing a spoofable host (M-2 hardening).
        """
        origin = public_discovery_origin(
            request_host,
            configured_base_url=self._settings.MCP_PUBLIC_BASE_URL,
            allowed_hosts=list(self._settings.MCP_ALLOWED_HOSTS),
        )
        if origin is None:
            logger.warning(LogMessage.MCP_DISCOVERY_ORIGIN_UNAVAILABLE)
            return None
        return www_authenticate_challenge(origin)

    @staticmethod
    async def _send_unauthorized(send: Send, www_authenticate: str | None) -> None:
        """Emit a plain HTTP 401 carrying the OAuth2 discovery challenge.

        The body stays the generic, non-enumerating JSON; the
        ``WWW-Authenticate`` header points OAuth2-enabled MCP clients at the
        public protected-resource metadata document (REQ-001-equivalent).
        ``Cache-Control: no-store`` keeps a shared cache from ever replaying
        the (per-deployment) challenge to a different client. ``None`` means
        no trustworthy public origin exists — the 401 is sent WITHOUT the
        header.

        Args:
            send: The raw ASGI ``send`` callable.
            www_authenticate: The challenge header value, or ``None``.
        """
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(_UNAUTHORIZED_BODY)).encode()),
            (b"cache-control", b"no-store"),
        ]
        if www_authenticate is not None:
            headers.append((b"www-authenticate", www_authenticate.encode("latin-1")))
        await send({"type": "http.response.start", "status": 401, "headers": headers})
        await send({"type": "http.response.body", "body": _UNAUTHORIZED_BODY})


def _asgi_headers(scope: Scope) -> dict[str, str]:
    """Decode the raw ASGI header list into a lowercase-keyed dict."""
    return {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in scope.get("headers", [])
    }


def _asgi_host(scope: Scope) -> str:
    """Return the raw ``Host`` header value of one ASGI scope (may be empty)."""
    return _asgi_headers(scope).get("host", "")
