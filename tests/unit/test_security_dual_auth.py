"""Tests for dual-auth (JWT + per-user API key) in ``security.py``.

Covers:
- ``ComposableAgentsSecurity.verify_credentials`` HTTP dependency: Bearer valid
  → 200, X-API-Key valid → 200, no credential → 401, wrong credential → 401,
  and the RLS contextvars are set on success.
- ``McpApiKeyMiddleware`` extended to ALSO accept ``Authorization: Bearer``
  (JWT) — set contextvars so downstream asyncpg usage is RLS-scoped.
- ``set_auth_service`` / ``has_auth_service`` lifecycle.
- ``AuthenticationError`` mapped to HTTP 401.

The in-memory FastMCP transport does not carry HTTP headers, so
``get_http_headers`` is patched where needed (same pattern as
``test_security.py``).
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastmcp import FastMCP
from fastmcp.client import Client
from fastmcp.client.transports.memory import FastMCPTransport
from fastmcp.exceptions import ToolError
from httpx import ASGITransport, AsyncClient

from domain.entities.auth.auth_context import AuthContext
from domain.errors.security import AuthenticationError, InvalidApiKeyError
from infrastructure.database.rls_context import (
    current_auth_method,
    current_credential,
    current_user_id,
)
from security import ComposableAgentsSecurity, McpApiKeyMiddleware, _cached_auth_context

_MASTER_KEY = "super-secret-key"


def _make_auth_service_mock(
    *,
    jwt_user_id: str | None = None,
    api_key_user_id: str | None = None,
) -> AsyncMock:
    """Return a mock AuthService that resolves JWT/api-key to AuthContext or None."""
    svc = AsyncMock()

    async def _authenticate(authorization: str | None, api_key: str | None):
        if authorization and authorization.startswith("Bearer "):
            if jwt_user_id is not None:
                return AuthContext(
                    user_id=jwt_user_id, method="jwt", raw_credential=authorization[7:]
                )
            return None
        if api_key is not None:
            if api_key_user_id is not None:
                return AuthContext(
                    user_id=api_key_user_id, method="api_key", raw_credential=api_key
                )
            return None
        return None

    svc.authenticate.side_effect = _authenticate
    return svc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_app_with_auth(auth_service: AsyncMock) -> FastAPI:
    """Build a minimal FastAPI app guarded by ``verify_credentials``."""
    security = ComposableAgentsSecurity(master_key=_MASTER_KEY)
    security.set_auth_service(auth_service)  # type: ignore[arg-type]
    app = FastAPI()

    protected = APIRouter()

    @protected.get("/protected")
    async def protected_route(_ctx: AuthContext = Depends(security.verify_credentials)):
        # Read the contextvars here, INSIDE the request task, so the test can
        # assert they were set (they don't propagate back to the caller context).
        return {
            "user_id": _ctx.user_id,
            "method": _ctx.method,
            "ctxvar_user_id": current_user_id.get(),
            "ctxvar_method": current_auth_method.get(),
            "ctxvar_credential": current_credential.get(),
        }

    app.include_router(protected)

    @app.exception_handler(AuthenticationError)
    async def _auth_err(_req: Request, exc: AuthenticationError) -> JSONResponse:
        return JSONResponse(
            status_code=int(exc.status_code), content={"detail": exc.detail}
        )

    @app.exception_handler(InvalidApiKeyError)
    async def _api_key_err(_req: Request, exc: InvalidApiKeyError) -> JSONResponse:
        return JSONResponse(
            status_code=int(exc.status_code), content={"detail": exc.detail}
        )

    return app


# ---------------------------------------------------------------------------
# 1. set_auth_service / has_auth_service lifecycle
# ---------------------------------------------------------------------------


class TestSetAuthService:
    """Tests for ``set_auth_service`` / ``has_auth_service``."""

    def test_has_auth_service_returns_false_before_set(self):
        # Arrange
        security = ComposableAgentsSecurity(master_key=_MASTER_KEY)

        # Act & Assert
        assert security.has_auth_service() is False

    def test_has_auth_service_returns_true_after_set(self):
        # Arrange
        security = ComposableAgentsSecurity(master_key=_MASTER_KEY)
        svc = _make_auth_service_mock(jwt_user_id="u1")

        # Act
        security.set_auth_service(svc)  # type: ignore[arg-type]

        # Assert
        assert security.has_auth_service() is True


# ---------------------------------------------------------------------------
# 2. verify_credentials (HTTP)
# ---------------------------------------------------------------------------


class TestVerifyCredentialsHttp:
    """Integration tests for ``verify_credentials`` over HTTP."""

    async def test_bearer_token_valid_returns_200_and_sets_contextvars(self):
        # Arrange
        svc = _make_auth_service_mock(jwt_user_id="user-jwt-123")
        app = _build_app_with_auth(svc)

        # Act
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/protected", headers={"Authorization": "Bearer my-jwt"}
            )

        # Assert
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == "user-jwt-123"
        assert body["method"] == "jwt"
        # Contextvars are read INSIDE the request task (see the route body) so
        # they don't need to propagate back to the test context.
        assert body["ctxvar_user_id"] == "user-jwt-123"
        assert body["ctxvar_credential"] == "my-jwt"
        assert body["ctxvar_method"] == "jwt"

    async def test_x_api_key_valid_returns_200_and_sets_contextvars(self):
        # Arrange
        svc = _make_auth_service_mock(api_key_user_id="user-api-456")
        app = _build_app_with_auth(svc)

        # Act
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/protected", headers={"X-API-Key": "cpk_xxx"})

        # Assert
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == "user-api-456"
        assert body["method"] == "api_key"
        assert body["ctxvar_user_id"] == "user-api-456"
        assert body["ctxvar_method"] == "api_key"

    async def test_no_credentials_returns_401(self):
        # Arrange — svc returns None for any credential
        svc = _make_auth_service_mock()
        app = _build_app_with_auth(svc)

        # Act
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/protected")

        # Assert
        assert resp.status_code == 401

    async def test_invalid_credentials_returns_401(self):
        # Arrange — svc resolves neither JWT nor API key
        svc = _make_auth_service_mock()
        app = _build_app_with_auth(svc)

        # Act
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/protected", headers={"Authorization": "Bearer bad-jwt"}
            )

        # Assert
        assert resp.status_code == 401

    async def test_authentication_error_status_code_is_401(self):
        # Act & Assert
        assert AuthenticationError().status_code == 401

    async def test_verify_credentials_falls_back_to_master_key_when_no_auth_service(
        self,
    ):
        """When no AuthService is wired, verify_credentials uses the master key.

        This covers the dev/test path and the startup window before the
        lifespan wires the AuthService: ``verify_credentials`` delegates to
        the legacy ``verify_api_key`` so the REST routers keep working.
        """
        # Arrange — no set_auth_service call
        security = ComposableAgentsSecurity(master_key=_MASTER_KEY)
        app = FastAPI()
        protected = APIRouter()

        @protected.get("/protected")
        async def protected_route(_=Depends(security.verify_credentials)):
            return {"ok": True}

        app.include_router(protected)

        @app.exception_handler(InvalidApiKeyError)
        async def _err(_req: Request, exc: InvalidApiKeyError) -> JSONResponse:
            return JSONResponse(
                status_code=int(exc.status_code), content={"detail": exc.detail}
            )

        # Act & Assert — valid master key → 200
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/protected", headers={"X-API-Key": _MASTER_KEY})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

        # Act & Assert — wrong key → 401
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/protected", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 3. McpApiKeyMiddleware dual auth (FastMCP in-memory transport)
# ---------------------------------------------------------------------------


def _build_mcp_server_with_auth(auth_service: AsyncMock) -> FastMCP:
    """Build a FastMCP server guarded by ``McpApiKeyMiddleware`` with dual auth.

    The echo tool returns the contextvar value seen INSIDE the tool body so the
    test can verify that contextvars set by the middleware propagate to the
    tool handler (same asyncio task, same context).
    """
    server = FastMCP("TestServer")
    middleware = McpApiKeyMiddleware(master_key=_MASTER_KEY)
    middleware.set_auth_service(auth_service)  # type: ignore[arg-type]
    server.add_middleware(middleware)

    @server.tool()
    async def echo(msg: str = "hello") -> str:
        """Echo its argument + the RLS contextvar user_id seen by the tool."""
        return (
            f"{msg}|user_id={current_user_id.get()}|method={current_auth_method.get()}"
        )

    return server


def _build_mcp_client(server: FastMCP) -> Client:
    return Client(transport=FastMCPTransport(mcp=server))


class TestMcpApiKeyMiddlewareDualAuth:
    """Tests for ``McpApiKeyMiddleware`` accepting Bearer + X-API-Key."""

    async def test_bearer_token_valid_succeeds_and_sets_contextvars(self):
        # Arrange
        svc = _make_auth_service_mock(jwt_user_id="user-jwt-1")
        server = _build_mcp_server_with_auth(svc)

        # Act
        with patch(
            "security.get_http_headers",
            return_value={"authorization": "Bearer my-jwt"},
        ):
            async with _build_mcp_client(server) as client:
                result = await client.call_tool("echo", {"msg": "test"})

        # Assert — the tool reads the contextvar INSIDE its own body (same
        # asyncio task as the middleware), so the value set by the middleware
        # is visible there. We assert it via the tool's return value, NOT by
        # reading the contextvar in the test (which runs in a different context).
        assert result.data == "test|user_id=user-jwt-1|method=jwt"

    async def test_x_api_key_valid_succeeds_and_sets_contextvars(self):
        # Arrange
        svc = _make_auth_service_mock(api_key_user_id="user-api-2")
        server = _build_mcp_server_with_auth(svc)

        # Act
        with patch(
            "security.get_http_headers",
            return_value={"x-api-key": "cpk_yyy"},
        ):
            async with _build_mcp_client(server) as client:
                result = await client.call_tool("echo", {"msg": "test"})

        # Assert
        assert result.data == "test|user_id=user-api-2|method=api_key"

    async def test_no_credentials_raises_tool_error(self):
        # Arrange
        svc = _make_auth_service_mock()
        server = _build_mcp_server_with_auth(svc)

        # Act & Assert
        with patch("security.get_http_headers", return_value={}):
            async with _build_mcp_client(server) as client:
                with pytest.raises(ToolError):
                    await client.call_tool("echo", {"msg": "test"})

    async def test_invalid_credentials_raises_tool_error(self):
        # Arrange
        svc = _make_auth_service_mock()
        server = _build_mcp_server_with_auth(svc)

        # Act & Assert
        with patch(
            "security.get_http_headers",
            return_value={"authorization": "Bearer bad-jwt"},
        ):
            async with _build_mcp_client(server) as client:
                with pytest.raises(ToolError):
                    await client.call_tool("echo", {"msg": "test"})


# ---------------------------------------------------------------------------
# 4. Backward-compat: verify_api_key still works without auth service
# ---------------------------------------------------------------------------


class TestVerifyApiKeyBackwardCompat:
    """``verify_api_key`` is preserved for backward-compat (no auth service wired)."""

    async def test_verify_api_key_still_works_without_auth_service(self):
        # Arrange
        security = ComposableAgentsSecurity(master_key=_MASTER_KEY)
        # no set_auth_service call — verify_api_key must still validate the master key

        # Act & Assert
        assert security.verify_api_key(api_key_header=_MASTER_KEY) == _MASTER_KEY
        with pytest.raises(InvalidApiKeyError):
            security.verify_api_key(api_key_header="wrong")


# ---------------------------------------------------------------------------
# 5. Defense-in-depth handshake (code-review M-2): the middleware reuses the
#    guard's verified principal for the SAME request's credential instead of
#    re-deriving it — one DB lookup / one JWT verification per HTTP request.
# ---------------------------------------------------------------------------


class TestCachedAuthHandshake:
    """The request-scoped cache written by ``authenticate_headers`` (the
    seam the HTTP-level guard calls) lets ``_check_api_key`` short-circuit
    when the presented credential is the one the guard already verified; a
    DIFFERENT credential never short-circuits (defense-in-depth preserved)."""

    def _middleware(self, auth_service: AsyncMock) -> "McpApiKeyMiddleware":
        from security import McpApiKeyMiddleware

        middleware = McpApiKeyMiddleware(master_key=_MASTER_KEY)
        middleware.set_auth_service(auth_service)  # type: ignore[arg-type]
        return middleware

    def _tool_context(self):
        from fastmcp.server.middleware import MiddlewareContext
        from mcp.types import CallToolRequest, CallToolRequestParams

        message = CallToolRequest(params=CallToolRequestParams(name="t", arguments={}))
        return MiddlewareContext(message=message, method="tools/call")

    async def _run_call_tool(self, middleware, call_next=None):
        from fastapi import Request as FastAPIRequest
        from fastmcp.server.dependencies import (
            _current_http_request,  # pyright: ignore[reportPrivateImportUsage]
        )

        async def _next(_ctx):
            return "ok"

        scope = {"type": "http", "method": "POST", "headers": [], "path": "/"}
        token = _current_http_request.set(FastAPIRequest(scope))
        try:
            return await middleware(self._tool_context(), call_next or _next)
        finally:
            _current_http_request.reset(token)

    async def test_skips_reverification_when_the_guard_cached_the_same_credential(self):
        svc = _make_auth_service_mock(api_key_user_id="user-alice")
        mw = self._middleware(svc)

        # The HTTP guard's pass: authenticates + caches the verified context.
        ctx = await mw.authenticate_headers(None, "key-alice")
        assert ctx is not None and ctx.user_id == "user-alice"
        svc.authenticate.reset_mock()

        # The middleware's defense-in-depth pass on the SAME request.
        with patch(
            "security.get_http_headers", return_value={"x-api-key": "key-alice"}
        ):
            result = await self._run_call_tool(mw)

        assert result == "ok"
        assert svc.authenticate.await_count == 0, (
            "the middleware must NOT re-verify a credential the guard already "
            "verified on this request (single authentication per request)"
        )

    async def test_foreign_credential_is_never_short_circuited(self):
        """A credential different from the cached one keeps the FULL
        re-verification: cross-request/session staleness can never be
        short-circuited into another user's principal."""
        svc = _make_auth_service_mock(
            jwt_user_id="user-bob", api_key_user_id="user-alice"
        )
        mw = self._middleware(svc)

        # The guard verified ALICE's API key on her request (cached).
        ctx = await mw.authenticate_headers(None, "key-alice")
        assert ctx is not None
        svc.authenticate.reset_mock()

        # BOB's tool call presents a DIFFERENT credential.
        with patch(
            "security.get_http_headers",
            return_value={"authorization": "Bearer jwt-bob"},
        ):
            result = await self._run_call_tool(mw)

        assert result == "ok"
        svc.authenticate.assert_awaited_once()

    async def test_no_cached_context_keeps_the_full_verification(self):
        """A path NOT covered by the HTTP guard (no cached context yet) keeps
        the middleware as the authenticator exactly as before."""
        svc = _make_auth_service_mock(api_key_user_id="user-api-2")
        mw = self._middleware(svc)
        svc.authenticate.reset_mock()

        with patch("security.get_http_headers", return_value={"x-api-key": "cpk_yyy"}):
            result = await self._run_call_tool(mw)

        assert result == "ok"
        svc.authenticate.assert_awaited_once()

    async def test_bearer_takes_precedence_for_the_cached_match(self):
        """The presented-credential extraction mirrors the AuthService
        precedence: a Bearer token wins over a colliding X-API-Key."""
        svc = _make_auth_service_mock(jwt_user_id="user-jwt")
        mw = self._middleware(svc)

        await mw.authenticate_headers("Bearer my-jwt", "key-alice")
        svc.authenticate.reset_mock()

        with patch(
            "security.get_http_headers",
            return_value={"authorization": "Bearer my-jwt", "x-api-key": "key-bob"},
        ):
            result = await self._run_call_tool(mw)

        assert result == "ok"
        assert svc.authenticate.await_count == 0, (
            "the Bearer token is the presented credential in the AuthService "
            "order — a matching cached context short-circuits"
        )


class TestCachedAuthContextReset:
    """Code-review MINOR-4: the handshake cache must be RESET at the start of
    every ``authenticate_headers`` pass — a stale entry can never outlive its
    request, so two SEQUENTIAL requests with different credentials on the SAME
    middleware instance never share the cached context (and a denied request
    leaves NO stale entry from the previous one)."""

    async def test_denied_request_leaves_no_stale_entry(self):
        """After Alice's request caches her context, a second request with a
        DIFFERENT (denied) credential must leave the cache EMPTY — before the
        reset the cache would still hold Alice's verified principal."""
        svc = AsyncMock()

        async def _authenticate(authorization: str | None, api_key: str | None):
            if api_key == "key-alice":
                return AuthContext(
                    user_id="user-alice", method="api_key", raw_credential="key-alice"
                )
            return None

        svc.authenticate.side_effect = _authenticate
        mw = McpApiKeyMiddleware(master_key=_MASTER_KEY)
        mw.set_auth_service(svc)

        # Request 1: alice authenticates (the guard's pass caches her context).
        ctx_first = await mw.authenticate_headers(None, "key-alice")
        assert ctx_first is not None and ctx_first.raw_credential == "key-alice"
        assert _cached_auth_context.get() is ctx_first

        # Request 2: a DIFFERENT credential is DENIED (e.g. a revoked key).
        ctx_second = await mw.authenticate_headers(None, "key-revoked")
        assert ctx_second is None
        assert _cached_auth_context.get() is None, (
            "a denied request must not leave a stale entry from the previous request"
        )

    async def test_two_sequential_requests_never_reuse_the_previous_principal(
        self,
    ):
        """Two sequential granted requests with different credentials: the
        cache holds ONLY the second request's own principal — a foreign
        credential (here: the previous request's one) can never be matched
        against it."""
        svc = AsyncMock()

        async def _authenticate(authorization: str | None, api_key: str | None):
            if api_key == "key-alice":
                return AuthContext(
                    user_id="user-alice", method="api_key", raw_credential="key-alice"
                )
            if api_key == "key-bob":
                return AuthContext(
                    user_id="user-bob", method="api_key", raw_credential="key-bob"
                )
            return None

        svc.authenticate.side_effect = _authenticate
        mw = McpApiKeyMiddleware(master_key=_MASTER_KEY)
        mw.set_auth_service(svc)

        # Request 1: alice authenticates (her context is cached).
        ctx_alice = await mw.authenticate_headers(None, "key-alice")
        assert ctx_alice is not None and ctx_alice.user_id == "user-alice"

        # Request 2: bob's own credential — SAME middleware instance.
        ctx_bob = await mw.authenticate_headers(None, "key-bob")
        assert ctx_bob is not None and ctx_bob.user_id == "user-bob"
        cached = _cached_auth_context.get()
        assert cached is not None
        assert cached is ctx_bob
        assert cached.raw_credential == "key-bob", (
            "alice's cached context from the previous request must be GONE — "
            "the reset runs before this request's own re-caching"
        )
