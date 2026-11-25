"""API key + JWT security for REST and MCP layers.

Provides two components:

- ``ComposableAgentsSecurity`` — FastAPI dependency that historically validated
  the ``X-API-Key`` header against a configured master key (``verify_api_key``,
  kept for backward-compat). The new dual-auth dependency ``verify_credentials``
  delegates to an injected :class:`~domain.services.auth.auth_service.AuthService`
  (JWT bearer token + per-user API key lookup), sets the RLS contextvars on
  success, and raises :class:`AuthenticationError` (401) on failure.

- ``McpApiKeyMiddleware`` — FastMCP middleware guarding ``tools/call`` and
  ``tools/list``. Historically checked only the ``X-API-Key`` master key; now
  extended to ALSO accept ``Authorization: Bearer`` (JWT) and per-user API keys
  via the same injected ``AuthService``. The RLS contextvars are set so that
  downstream asyncpg usage in the tool handler is RLS-scoped (contextvars
  propagate within the same asyncio task).

    feat-mcp-oauth2-dual-auth (REQ-017): the HTTP-level ASGI guard
    (:mod:`application.api.mcp_guard`) is the PRIMARY gate AND the RLS
    authority on the MCP mounts — it authenticates every request (including
    ``initialize``) at the HTTP level through the shared
    :meth:`McpApiKeyMiddleware.authenticate_headers` seam and caches the
    verified ``AuthContext`` for the request. The middleware stays mounted on
    the FastMCP apps as DEFENSE-IN-DEPTH behind the HTTP guard: when the same
    request's credential is already verified (the guard's cached context), it
    reuses that principal WITHOUT re-deriving it (one DB lookup / one JWT
    verification per HTTP request); on any other credential (or on a path
    not covered by the guard) it authenticates itself exactly as before.

When the master key is empty, the legacy ``verify_api_key`` / ``_check_api_key``
paths disable auth (useful for local dev/test). The dual-auth paths are
engaged only when ``set_auth_service`` has been called from the composition
root (``main.py`` lifespan).
"""

import contextvars
import logging
import secrets

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import Middleware, MiddlewareContext

from domain.entities.auth.auth_context import AuthContext
from domain.errors.messages import ErrorMessage
from domain.errors.security import AuthenticationError, InvalidApiKeyError
from infrastructure.database.rls_context import (
    current_auth_method,
    current_credential,
    current_user_id,
)

logger = logging.getLogger(__name__)

_BEARER_PREFIX = "Bearer "

#: Request-scoped handshake: the HTTP-level ASGI guard authenticates ONCE per
#: request through :meth:`McpApiKeyMiddleware.authenticate_headers`, which then
#: caches the verified :class:`AuthContext` here. The MCP tool task (stateless
#: transport ⇒ the same task context as the request) reads the cache instead of
#: re-deriving the principal — exactly ONE DB lookup / ONE JWT verification per
#: authenticated HTTP request (the "no extra round-trips on the MCP hot path"
#: NFR). A credential that does not match the cached one re-verifies, so the
#: defense-in-depth semantics are preserved on every path not covered by the
#: guard ("cached" is only ever a shortcut for the SAME request's credential).
_cached_auth_context: contextvars.ContextVar[AuthContext | None] = (
    contextvars.ContextVar("cached_mcp_auth_context", default=None)
)


def _presented_credential(authorization: str | None, api_key: str | None) -> str | None:
    """Return the credential a request actually presents (AuthService order).

    Mirrors :meth:`AuthService.authenticate` precedence: only ``Bearer`` is
    recognised as JWT (API keys travel exclusively in ``X-API-Key``) and the
    Bearer token wins when both are present.
    """
    if authorization and authorization.startswith(_BEARER_PREFIX):
        return authorization[len(_BEARER_PREFIX) :]
    return api_key


class ComposableAgentsSecurity:
    """Validates incoming credentials (legacy master key or dual JWT/API key).

    The class is instantiated once at the composition root with the master key
    (``AppConfig.API_KEY``). Its ``verify_api_key`` method (legacy, kept for
    backward-compat) validates the ``X-API-Key`` header against the master key.
    When the master key is empty the check is bypassed (dev/test mode).

    The dual-auth ``verify_credentials`` dependency delegates to an injected
    :class:`~domain.services.auth.auth_service.AuthService` (JWT bearer token +
    per-user API key) and sets the RLS contextvars on success. It is engaged
    only after ``set_auth_service`` has been called from the composition root.
    """

    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def __init__(self, master_key: str = "") -> None:
        self.master_key = master_key
        # Dual-auth service (JWT + per-user API key). Injected via
        # ``set_auth_service`` from the composition root; ``verify_credentials``
        # raises a clear RuntimeError if it is called before wiring.
        self._auth_service = None

    def set_auth_service(self, auth_service) -> None:
        """Inject the dual-auth :class:`AuthService`.

        Args:
            auth_service: The wired ``AuthService`` (JWT port + API key repo).
        """
        self._auth_service = auth_service

    def has_auth_service(self) -> bool:
        """Return whether a dual-auth :class:`AuthService` is wired."""
        return self._auth_service is not None

    # ------------------------------------------------------------------
    # Legacy: master-key X-API-Key check (backward-compat)
    # ------------------------------------------------------------------

    def verify_api_key(
        self,
        api_key_header: str | None = Depends(api_key_header),
    ) -> str:
        """Validate the ``X-API-Key`` header against ``master_key``.

        Kept for backward-compat with existing tests and the legacy wiring.
        New protected routers should use :meth:`verify_credentials` instead.

        Args:
            api_key_header: The value of the ``X-API-Key`` request header.

        Returns:
            The validated API key string, or an empty string when auth is
            disabled (no master key configured).

        Raises:
            InvalidApiKeyError: If the header is missing or does not match the
                master key.
        """
        if not self.master_key:
            logger.warning(ErrorMessage.API_KEY_DISABLED)
            return ""
        if not api_key_header:
            raise InvalidApiKeyError(ErrorMessage.API_KEY_EMPTY)
        if not secrets.compare_digest(api_key_header, self.master_key):
            raise InvalidApiKeyError(ErrorMessage.API_KEY_UNAUTHORIZED)
        return api_key_header

    # ------------------------------------------------------------------
    # Dual auth: JWT bearer token OR per-user API key
    # ------------------------------------------------------------------

    async def verify_credentials(self, request: Request) -> AuthContext | str:
        """Dual-auth FastAPI dependency: JWT bearer token OR per-user API key.

        Reads the ``Authorization`` and ``X-API-Key`` headers, delegates to the
        injected :class:`AuthService`, raises :class:`AuthenticationError` (401)
        when no credential validates, and otherwise sets the RLS contextvars
        (``current_user_id`` / ``current_credential`` / ``current_auth_method``)
        and returns the resolved :class:`AuthContext`.

        **Fallback when no AuthService is wired** (dev/test, or production
        before the lifespan runs): falls back to the legacy master-key
        ``verify_api_key`` check on the ``X-API-Key`` header. This keeps the
        REST routers working when dual auth is not configured (``LOGTO_URL``
        unset or asyncpg pool not available) and avoids a RuntimeError at
        request time.

        Args:
            request: The incoming FastAPI request (headers are read from it).

        Returns:
            The authenticated :class:`AuthContext` (dual-auth path) or the
            validated API key string / empty string (legacy fallback).

        Raises:
            AuthenticationError: If no credential could be validated (dual-auth
                path, 401).
            InvalidApiKeyError: If the legacy fallback rejects the key (401).
        """
        if self._auth_service is None:
            # Legacy fallback: no dual-auth wired. Use the master-key check.
            return self.verify_api_key(request.headers.get("x-api-key"))

        authorization = request.headers.get("authorization")
        api_key = request.headers.get("x-api-key")

        ctx = await self._auth_service.authenticate(
            authorization=authorization,
            api_key=api_key,
        )
        if ctx is None:
            raise AuthenticationError(ErrorMessage.AUTH_INVALID_CREDENTIALS)

        # Wire the RLS contextvars for downstream asyncpg usage (McpRegistryStore).
        current_user_id.set(ctx.user_id)
        current_credential.set(ctx.raw_credential)
        current_auth_method.set(ctx.method)
        return ctx


class McpApiKeyMiddleware(Middleware):
    """FastMCP middleware that enforces auth on MCP tools (master key or dual).

    Historically guarded ``tools/call`` and ``tools/list`` by checking the
    ``X-API-Key`` header against the configured master key. Now extended to
    ALSO accept ``Authorization: Bearer`` (JWT) and per-user API keys when an
    :class:`AuthService` is wired via :meth:`set_auth_service`.

    The RLS contextvars (``current_user_id`` / ``current_credential`` /
    ``current_auth_method``) are set on success so that downstream asyncpg
    usage in the tool handler is RLS-scoped. Contextvars propagate within
    the same asyncio task — the FastMCP request handler runs the middleware
    and the tool in the same task, so the tool sees the values set here.

    When the master key is empty AND no auth service is wired, authentication
    is bypassed (dev/test mode).
    """

    def __init__(self, master_key: str = "") -> None:
        self._master_key = master_key
        self._auth_service = None

    def set_auth_service(self, auth_service) -> None:
        """Inject the dual-auth :class:`AuthService`.

        Args:
            auth_service: The wired ``AuthService`` (JWT port + API key repo).
        """
        self._auth_service = auth_service

    def has_auth_service(self) -> bool:
        """Return whether a dual-auth :class:`AuthService` is wired."""
        return self._auth_service is not None

    async def _check_api_key(self) -> None:
        """Raise ``ToolError`` if the request lacks valid credentials.

        Resolution order:
        1. The HTTP-level guard already verified THIS request's credential →
           reuse its cached :class:`AuthContext` (no second DB lookup / JWT
           verification — the RLS contextvars were set by the guard).
        2. Dual-auth :class:`AuthService` wired → the shared
           :meth:`authenticate_headers` seam (Bearer JWT precedence, else
           X-API-Key lookup; RLS contextvars set on success).
        3. Dual-auth wired but credentials invalid → deny (no legacy
           fallback).
        4. Legacy master-key mode (no dual-auth wired) → master key empty
           bypasses, else the ``X-API-Key`` header must match the master key.
        """
        headers: dict[str, str] = get_http_headers(include_all=True) or {}
        authorization = headers.get("authorization")
        api_key = headers.get("x-api-key")

        # NOTE: ``get_http_headers()`` (default) strips ``authorization``; the
        # dual-auth middleware must see the raw ``Authorization: Bearer`` and
        # ``X-API-Key`` headers, hence ``include_all=True``.

        # 1. The guard verified this exact credential → skip the duplicate
        #    work (M-2: one authentication per HTTP request). The cache is
        #    request-scoped (set by this request's guard pass); a foreign
        #    credential (e.g. another user reusing a stateful session) does
        #    NOT match and keeps the full re-verification below.
        if self._is_guard_verified(authorization, api_key):
            return

        # 2+3. Dual-auth path (JWT + per-user API key), shared with the ASGI
        #       HTTP guard so both layers behave identically (REQ-017).
        ctx = await self.authenticate_headers(authorization, api_key)
        if ctx is not None:
            return
        if self._auth_service is not None:
            # Dual-auth wired: no credential validated → deny (an invalid
            # Bearer JWT never falls back to the API key).
            raise ToolError("Invalid or missing credentials")

        # 4. Legacy master-key path.
        self._legacy_master_key_check(api_key)

    def _is_guard_verified(
        self, authorization: str | None, api_key: str | None
    ) -> bool:
        """Whether the HTTP-level guard already verified THIS credential.

        The guard's pass through :meth:`authenticate_headers` caches the
        verified :class:`AuthContext` (request-scoped contextvar). The
        short-circuit is safe only for the credential the guard actually
        verified: the presented-credential extraction mirrors the
        :class:`AuthService` precedence (Bearer token wins, API key
        otherwise), so a foreign credential can never reuse another
        principal's context.
        """
        cached = _cached_auth_context.get()
        presented = _presented_credential(authorization, api_key)
        return (
            cached is not None
            and presented is not None
            and cached.raw_credential == presented
        )

    def _legacy_master_key_check(self, api_key: str | None) -> None:
        """Legacy master-key enforcement (no dual-auth wired)."""
        if not self._master_key:
            return
        if not api_key or not secrets.compare_digest(api_key, self._master_key):
            raise ToolError("Invalid or missing API key")

    async def authenticate_headers(
        self, authorization: str | None, api_key: str | None
    ) -> AuthContext | None:
        """Authenticate raw request headers at the HTTP level (REQ-017).

        The single seam shared by the FastMCP method-level middleware and the
        HTTP-level ASGI guard (``application.api.mcp_guard.McpAuthGuard``):
        both delegate to the wired :class:`AuthService` and resolve the
        RLS contextvars exactly alike, so MCP tool calls stay scoped to the
        authenticated user whichever layer authenticated the request. On
        success the verified context is cached (request-scoped contextvar)
        so the middleware's defense-in-depth pass on the SAME request reuses
        it instead of re-deriving it (one DB lookup / one JWT verification
        per HTTP request).

        Args:
            authorization: Raw ``Authorization`` header value (``None`` if
                absent); only ``Bearer`` (JWT) is recognised — API keys
                travel only in ``X-API-Key``.
            api_key: Raw ``X-API-Key`` header value (``None`` if absent).

        Returns:
            The authenticated :class:`AuthContext` with the RLS contextvars
            set, or ``None`` when no credential validates OR when no dual
            auth is wired (fail-closed for the HTTP-level callers).
        """
        if self._auth_service is None:
            # Fail-closed until the composition root wires dual auth; the
            # legacy master-key mode stays available only to the method-level
            # middleware (its own branch below).
            return None
        ctx = await self._auth_service.authenticate(
            authorization=authorization,
            api_key=api_key,
        )
        if ctx is None:
            return None
        # Set the RLS contextvars for downstream asyncpg usage and cache the
        # verified principal for the middleware's defense-in-depth pass on
        # the same request (single-auth per request, M-2).
        current_user_id.set(ctx.user_id)
        current_credential.set(ctx.raw_credential)
        current_auth_method.set(ctx.method)
        _cached_auth_context.set(ctx)
        return ctx

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        """Guard tool execution with the credential check."""
        await self._check_api_key()
        return await call_next(context)

    async def on_list_tools(self, context: MiddlewareContext, call_next):
        """Guard tool discovery with the credential check."""
        await self._check_api_key()
        return await call_next(context)
