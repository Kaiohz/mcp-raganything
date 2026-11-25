"""Tests for MCP server registry FastAPI routes.

Mirrors the pattern from ``test_classical_query_routes.py``: dependencies are
wired through ``app.dependency_overrides`` (FastAPI stays in control of its own
providers), the API key check is bypassed (empty master key in tests), and
requests go through ``httpx.AsyncClient`` with ``ASGITransport``.

The use cases are mocked at the use-case boundary (AsyncMock). Each use case is
replaced with an AsyncMock so the route layer is exercised against its real
FastAPI wiring while the persistence and external MCP boundaries are stubbed.

CRUD endpoints under ``/api/v1/mcp/servers`` + ``/validate`` + ``/{name}/reveal``.
Tests cover masking of secrets in responses, 404/409 errors, the openapi create
flow, and validate dry-run.
"""

import logging
from datetime import UTC, datetime
from typing import Literal
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from application.api.mcp_registry_routes import mcp_registry_router
from application.use_cases.create_mcp_server import CreateMcpServerUseCase
from application.use_cases.delete_mcp_server import DeleteMcpServerUseCase
from application.use_cases.get_mcp_server import GetMcpServerUseCase
from application.use_cases.list_mcp_servers import ListMcpServersUseCase
from application.use_cases.update_mcp_server import UpdateMcpServerUseCase
from application.use_cases.validate_mcp_server import ValidateMcpServerUseCase
from dependencies import (
    get_create_mcp_server_use_case,
    get_delete_mcp_server_use_case,
    get_get_mcp_server_use_case,
    get_get_mcp_server_use_case_optional,
    get_list_mcp_servers_use_case,
    get_request_auth_context,
    get_update_mcp_server_use_case,
    get_validate_mcp_server_use_case,
)
from domain.entities.auth.auth_context import AuthContext
from domain.entities.mcp_server_config import McpAuthMode
from domain.entities.mcp_server_registry_entry import RegisteredMcpServer
from domain.errors.mcp import (
    McpServerAlreadyExistsError,
    McpServerInvalidAuthModeError,
    McpServerNotFoundError,
    McpServerUnreachableError,
)
from domain.errors.messages import ErrorMessage
from infrastructure.database.rls_context import current_auth_method, current_credential
from main import app

BASE_URL = "/api/v1/mcp/servers"


def _now() -> datetime:
    return datetime.now(UTC)


def _entry(
    name: str = "web-search",
    url: str = "http://localhost:3001/mcp",
    auth_token: str | None = "secret-token",
    tool_count: int = 3,
    source_type: str = "external",
    openapi_url: str | None = None,
    auth_mode: McpAuthMode = McpAuthMode.HEADERS,
) -> RegisteredMcpServer:
    return RegisteredMcpServer(
        name=name,
        url=url,
        headers={"Authorization": "Bearer tok"},
        env={"API_KEY": "abc"},
        auth_token=auth_token,
        auth_mode=auth_mode,
        tool_count=tool_count,
        source_type=source_type,
        openapi_url=openapi_url,
        created_at=_now(),
        updated_at=_now(),
    )


def _masked_entry(
    name: str = "web-search",
    url: str = "http://localhost:3001/mcp",
    tool_count: int = 3,
    source_type: str = "external",
    openapi_url: str | None = None,
    auth_mode: McpAuthMode = McpAuthMode.HEADERS,
) -> RegisteredMcpServer:
    """An entry as returned by masked endpoints (secrets stripped)."""
    return RegisteredMcpServer(
        name=name,
        url=url,
        headers={},
        env={},
        auth_token=None,
        auth_mode=auth_mode,
        tool_count=tool_count,
        source_type=source_type,
        openapi_url=openapi_url,
        created_at=_now(),
        updated_at=_now(),
    )


def _openapi_entry(
    name: str = "petstore",
    url: str = "http://raganything-api:8000/generated/petstore/mcp",
    openapi_url: str = "https://petstore.example.com/openapi.json",
    tool_count: int = 3,
) -> RegisteredMcpServer:
    return _entry(
        name=name,
        url=url,
        auth_token=None,
        tool_count=tool_count,
        source_type="openapi",
        openapi_url=openapi_url,
    )


def _masked_openapi_entry(
    name: str = "petstore",
    url: str = "http://raganything-api:8000/generated/petstore/mcp",
    openapi_url: str = "https://petstore.example.com/openapi.json",
    tool_count: int = 3,
) -> RegisteredMcpServer:
    return _masked_entry(
        name=name,
        url=url,
        tool_count=tool_count,
        source_type="openapi",
        openapi_url=openapi_url,
    )


# -- Fixtures ------------------------------------------------------------------


@pytest.fixture
def mock_create_use_case() -> AsyncMock:
    uc = AsyncMock(spec=CreateMcpServerUseCase)
    uc.execute = AsyncMock(return_value=_masked_entry(tool_count=2))
    return uc


@pytest.fixture
def mock_validate_use_case() -> AsyncMock:
    uc = AsyncMock(spec=ValidateMcpServerUseCase)
    uc.execute = AsyncMock(return_value=2)
    uc.execute_openapi = AsyncMock(return_value=2)
    return uc


@pytest.fixture
def mock_list_use_case() -> AsyncMock:
    uc = AsyncMock(spec=ListMcpServersUseCase)
    uc.execute = AsyncMock(
        return_value=[_masked_entry("server-a"), _masked_entry("server-b")]
    )
    return uc


@pytest.fixture
def mock_get_use_case() -> AsyncMock:
    uc = AsyncMock(spec=GetMcpServerUseCase)
    uc.execute = AsyncMock(return_value=_masked_entry())
    return uc


@pytest.fixture
def mock_update_use_case() -> AsyncMock:
    uc = AsyncMock(spec=UpdateMcpServerUseCase)
    uc.execute = AsyncMock(return_value=_masked_entry(tool_count=5))
    return uc


@pytest.fixture
def mock_delete_use_case() -> AsyncMock:
    uc = AsyncMock(spec=DeleteMcpServerUseCase)
    uc.execute = AsyncMock(return_value=None)
    return uc


@pytest.fixture(autouse=True)
def _wire_router_and_overrides(
    mock_create_use_case: AsyncMock,
    mock_validate_use_case: AsyncMock,
    mock_list_use_case: AsyncMock,
    mock_get_use_case: AsyncMock,
    mock_update_use_case: AsyncMock,
    mock_delete_use_case: AsyncMock,
):
    """Include the MCP registry router (idempotently) and wire mocked use cases."""
    # Include the router on the app if not already present
    existing_paths = {getattr(r, "path", "") for r in app.routes}
    if not any(p.startswith(BASE_URL) for p in existing_paths):
        app.include_router(mcp_registry_router)

    app.dependency_overrides[get_create_mcp_server_use_case] = lambda: (
        mock_create_use_case
    )
    app.dependency_overrides[get_validate_mcp_server_use_case] = lambda: (
        mock_validate_use_case
    )
    app.dependency_overrides[get_list_mcp_servers_use_case] = lambda: mock_list_use_case
    app.dependency_overrides[get_get_mcp_server_use_case] = lambda: mock_get_use_case
    # The validate route resolves the stored entry through the OPTIONAL
    # provider (lazy, BUG-001): wire it to the same mock.
    app.dependency_overrides[get_get_mcp_server_use_case_optional] = lambda: (
        mock_get_use_case
    )
    app.dependency_overrides[get_update_mcp_server_use_case] = lambda: (
        mock_update_use_case
    )
    app.dependency_overrides[get_delete_mcp_server_use_case] = lambda: (
        mock_delete_use_case
    )

    yield

    app.dependency_overrides.clear()


@pytest.fixture
def client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# -- POST /api/v1/mcp/servers (create, external) -------------------------------


class TestCreateMcpServerRoute:
    """Tests for POST /api/v1/mcp/servers (external source_type)."""

    async def test_returns_201_with_name(
        self, client: AsyncClient, mock_create_use_case: AsyncMock
    ):
        # Arrange
        mock_create_use_case.execute = AsyncMock(
            return_value=_masked_entry("web-search", tool_count=2)
        )
        body = {
            "name": "web-search",
            "url": "http://localhost:3001/mcp",
            "headers": {},
            "env": {},
            "auth_token": "tok",
        }

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 201
        assert resp.json()["name"] == "web-search"

    async def test_returns_201_with_tool_count(
        self, client: AsyncClient, mock_create_use_case: AsyncMock
    ):
        # Arrange
        mock_create_use_case.execute = AsyncMock(
            return_value=_masked_entry("web-search", tool_count=2)
        )
        body = {"name": "web-search", "url": "http://localhost:3001/mcp"}

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 201
        assert resp.json()["tool_count"] == 2

    async def test_masks_auth_token_in_response(
        self, client: AsyncClient, mock_create_use_case: AsyncMock
    ):
        # Arrange — the route must NOT return the plaintext auth_token
        mock_create_use_case.execute = AsyncMock(
            return_value=_masked_entry("web-search")
        )
        body = {
            "name": "web-search",
            "url": "http://localhost:3001/mcp",
            "auth_token": "secret",
        }

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 201
        assert resp.json()["auth_token"] is None

    async def test_masks_headers_in_response(
        self, client: AsyncClient, mock_create_use_case: AsyncMock
    ):
        # Arrange
        mock_create_use_case.execute = AsyncMock(
            return_value=_masked_entry("web-search")
        )
        body = {
            "name": "web-search",
            "url": "http://localhost:3001/mcp",
            "headers": {"Authorization": "Bearer secret"},
        }

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 201
        assert resp.json()["headers"] == {}

    async def test_masks_env_in_response(
        self, client: AsyncClient, mock_create_use_case: AsyncMock
    ):
        # Arrange
        mock_create_use_case.execute = AsyncMock(
            return_value=_masked_entry("web-search")
        )
        body = {
            "name": "web-search",
            "url": "http://localhost:3001/mcp",
            "env": {"API_KEY": "abc"},
        }

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 201
        assert resp.json()["env"] == {}

    async def test_returns_409_when_already_exists(
        self, client: AsyncClient, mock_create_use_case: AsyncMock
    ):
        # Arrange
        mock_create_use_case.execute = AsyncMock(
            side_effect=McpServerAlreadyExistsError("already exists")
        )
        body = {"name": "web-search", "url": "http://localhost:3001/mcp"}

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 409

    async def test_returns_422_when_unreachable(
        self, client: AsyncClient, mock_create_use_case: AsyncMock
    ):
        # Arrange
        mock_create_use_case.execute = AsyncMock(
            side_effect=McpServerUnreachableError("unreachable")
        )
        body = {"name": "web-search", "url": "http://localhost:3001/mcp"}

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 422

    async def test_rejects_missing_name(self, client: AsyncClient):
        # Act
        resp = await client.post(BASE_URL, json={"url": "http://x/mcp"})

        # Assert
        assert resp.status_code == 422

    async def test_rejects_missing_url_for_external(self, client: AsyncClient):
        # Act
        resp = await client.post(BASE_URL, json={"name": "web-search"})

        # Assert — external source_type requires url
        assert resp.status_code == 422


# -- POST /api/v1/mcp/servers (create, openapi) --------------------------------


class TestCreateMcpServerRouteOpenApi:
    """Tests for POST /api/v1/mcp/servers with source_type=openapi."""

    async def test_returns_201_with_source_type_openapi(
        self, client: AsyncClient, mock_create_use_case: AsyncMock
    ):
        # Arrange
        mock_create_use_case.execute = AsyncMock(
            return_value=_masked_openapi_entry("petstore", tool_count=2)
        )
        body = {
            "name": "petstore",
            "source_type": "openapi",
            "openapi_url": "https://petstore.example.com/openapi.json",
            "headers": {"Authorization": "Bearer upstream-token"},
        }

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 201
        assert resp.json()["source_type"] == "openapi"

    async def test_returns_201_with_absolute_mounted_url(
        self, client: AsyncClient, mock_create_use_case: AsyncMock
    ):
        # Arrange
        mock_create_use_case.execute = AsyncMock(
            return_value=_masked_openapi_entry(
                "petstore",
                url="http://raganything-api:8000/generated/petstore/mcp",
                tool_count=2,
            )
        )
        body = {
            "name": "petstore",
            "source_type": "openapi",
            "openapi_url": "https://x.example.com/openapi.json",
        }

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 201
        assert (
            resp.json()["url"] == "http://raganything-api:8000/generated/petstore/mcp"
        )

    async def test_rejects_openapi_without_openapi_url(self, client: AsyncClient):
        # Act
        resp = await client.post(
            BASE_URL, json={"name": "petstore", "source_type": "openapi"}
        )

        # Assert — openapi_url required when source_type=openapi
        assert resp.status_code == 422

    async def test_masks_headers_in_openapi_response(
        self, client: AsyncClient, mock_create_use_case: AsyncMock
    ):
        # Arrange — upstream headers must NOT be leaked
        mock_create_use_case.execute = AsyncMock(
            return_value=_masked_openapi_entry("petstore")
        )
        body = {
            "name": "petstore",
            "source_type": "openapi",
            "openapi_url": "https://x.example.com/openapi.json",
            "headers": {"Authorization": "Bearer secret"},
        }

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 201
        assert resp.json()["headers"] == {}


# -- POST /api/v1/mcp/servers/validate (create probe injection, REQ-004/BUG-006) -


class TestCreateAuthModeInjection:
    """``POST /`` injects the caller credential into the CREATE probe (REQ-004).

    The create-time reachability probe IS a load (it runs the same tool
    loading that records ``tool_count``), so in the ``user_token`` mode the
    caller's credential (``user_credential_headers`` domain rule + the
    ``get_request_auth_context`` provider, exactly like the validate route —
    BUG-006) must reach the target during create too. The injection is
    PROBE-ONLY: the route derives it from the request and the persisted entry
    keeps exactly the configured headers.
    """

    def _wire_caller(
        self, method: Literal["jwt", "api_key"] = "jwt", credential: str = "caller-jwt"
    ):
        app.dependency_overrides[get_request_auth_context] = lambda: AuthContext(
            user_id="user-alice",
            method=method,
            raw_credential=credential,
        )

    async def test_user_token_create_injects_caller_jwt_probe_only(
        self, client: AsyncClient, mock_create_use_case: AsyncMock
    ):
        # Arrange — the OAuth2 registration flow: user_token + JWT caller
        self._wire_caller(method="jwt", credential="caller-jwt")
        body = {
            "name": "oauth2-self",
            "url": "https://target.example.com/mcp",
            "headers": {"X-Custom": "value"},
            "auth_mode": "user_token",
        }

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert — the persisted entry keeps the configured headers; the
        # probe carries the INJECTED credential
        assert resp.status_code == 201, resp.text
        entry = mock_create_use_case.execute.await_args.args[0]
        probe_credentials = mock_create_use_case.execute.await_args.kwargs[
            "probe_credentials"
        ]
        assert entry.headers == {"X-Custom": "value"}
        assert probe_credentials == {"Authorization": "Bearer caller-jwt"}

    async def test_user_token_create_api_key_caller_injects_x_api_key(
        self, client: AsyncClient, mock_create_use_case: AsyncMock
    ):
        # Arrange — a caller authenticated with its per-user API key
        self._wire_caller(method="api_key", credential="key-alice")
        body = {
            "name": "oauth2-self-apikey",
            "url": "https://target.example.com/mcp",
            "auth_mode": "user_token",
        }

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 201, resp.text
        probe_credentials = mock_create_use_case.execute.await_args.kwargs[
            "probe_credentials"
        ]
        assert probe_credentials == {"X-API-Key": "key-alice"}

    async def test_headers_mode_create_sends_no_probe_credentials(
        self, client: AsyncClient, mock_create_use_case: AsyncMock
    ):
        # Arrange — a caller credential EXISTS but the server asks headers mode
        self._wire_caller(method="jwt")
        body = {
            "name": "headers-mode",
            "url": "https://target.example.com/mcp",
            "headers": {"X-API-Key": "configured-key"},
            "auth_mode": "headers",
        }

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert — no injection (behavior identical to the pre-feature one)
        assert resp.status_code == 201, resp.text
        probe_credentials = mock_create_use_case.execute.await_args.kwargs[
            "probe_credentials"
        ]
        assert probe_credentials == {}

    async def test_user_token_create_without_caller_credential_probes_unchanged(
        self, client: AsyncClient, mock_create_use_case: AsyncMock
    ):
        # Arrange — no dual-auth credential on the request (legacy fallback /
        # unattended run) → the provider returns None → drop-empty semantics
        app.dependency_overrides[get_request_auth_context] = lambda: None
        body = {
            "name": "unattended",
            "url": "https://target.example.com/mcp",
            "auth_mode": "user_token",
        }

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert — the credential is DROPPED, not guessed (REQ-004 error case)
        assert resp.status_code == 201, resp.text
        probe_credentials = mock_create_use_case.execute.await_args.kwargs[
            "probe_credentials"
        ]
        assert probe_credentials == {}


# -- POST /api/v1/mcp/servers/validate -----------------------------------------


class TestValidateMcpServerRoute:
    """Tests for POST /api/v1/mcp/servers/validate (dry-run)."""

    async def test_returns_200_with_tool_count(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock
    ):
        # Arrange
        mock_validate_use_case.execute = AsyncMock(return_value=5)
        body = {"url": "http://localhost:3001/mcp"}

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert
        assert resp.status_code == 200
        assert resp.json()["tool_count"] == 5

    async def test_returns_422_when_unreachable(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock
    ):
        # Arrange
        mock_validate_use_case.execute = AsyncMock(
            side_effect=McpServerUnreachableError("unreachable")
        )
        body = {"url": "http://localhost:3001/mcp"}

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert
        assert resp.status_code == 422

    async def test_openapi_validate_returns_tool_count(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock
    ):
        # Arrange
        mock_validate_use_case.execute_openapi = AsyncMock(return_value=10)
        body = {
            "source_type": "openapi",
            "openapi_url": "https://x.example.com/openapi.json",
        }

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert
        assert resp.status_code == 200
        assert resp.json()["tool_count"] == 10

    async def test_openapi_validate_calls_execute_openapi(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock
    ):
        # Arrange
        mock_validate_use_case.execute_openapi = AsyncMock(return_value=3)
        body = {
            "source_type": "openapi",
            "openapi_url": "https://x.example.com/openapi.json",
            "headers": {"Authorization": "Bearer tok"},
        }

        # Act
        await client.post(f"{BASE_URL}/validate", json=body)

        # Assert
        mock_validate_use_case.execute_openapi.assert_awaited_once()


# -- validate auth_mode credential injection (REQ-004, re-review MAJOR-1) ------


class TestValidateAuthModeInjection:
    """``POST /validate`` honors ``auth_mode`` at validate time (REQ-004).

    auth_mode used to be accepted and then DISCARDED by the validate route,
    so "Test connection" probed a ``user_token`` target with no credential
    (an authenticated target answered 401 → a false validation failure). The
    injected credential is derived by the domain rule
    ``user_credential_headers`` from the injected caller context.
    """

    def _wire_jwt_caller(
        self, method: Literal["jwt", "api_key"] = "jwt", credential: str = "jwt-token"
    ):
        app.dependency_overrides[get_request_auth_context] = lambda: AuthContext(
            user_id="user-alice",
            method=method,
            raw_credential=credential,
        )

    async def test_user_token_injects_caller_jwt_authorization(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock
    ):
        # Arrange — caller authenticated via JWT, request asks user_token
        self._wire_jwt_caller(method="jwt")
        body = {
            "url": "https://target.example.com/mcp",
            "auth_mode": "user_token",
            "headers": {},
        }

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert — the probe config carries the caller's credential
        assert resp.status_code == 200
        config = mock_validate_use_case.execute.await_args.args[0]
        assert config.headers == {"Authorization": "Bearer jwt-token"}

    async def test_user_token_injects_caller_api_key(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock
    ):
        # Arrange — caller authenticated via API key
        self._wire_jwt_caller(method="api_key", credential="key-alice")
        body = {"url": "https://target.example.com/mcp", "auth_mode": "user_token"}

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert — X-API-Key is the injected credential header
        assert resp.status_code == 200
        config = mock_validate_use_case.execute.await_args.args[0]
        assert config.headers == {"X-API-Key": "key-alice"}

    async def test_headers_mode_probes_exactly_the_entered_headers(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock
    ):
        # Arrange — a caller credential EXISTS but the server asks headers mode
        self._wire_jwt_caller(method="jwt")
        body = {
            "url": "https://target.example.com/mcp",
            "auth_mode": "headers",
            "headers": {"X-Custom": "value"},
        }

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert — no injection (behavior identical to the pre-feature one)
        assert resp.status_code == 200
        config = mock_validate_use_case.execute.await_args.args[0]
        assert config.headers == {"X-Custom": "value"}

    async def test_user_token_without_caller_credential_probes_unchanged(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock
    ):
        # Arrange — no dual-auth credential on the request (the provider
        # returns None: legacy fallback / unattended run) → drop-empty
        app.dependency_overrides[get_request_auth_context] = lambda: None
        body = {"url": "https://target.example.com/mcp", "auth_mode": "user_token"}

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert — the credential is DROPPED, not guessed (REQ-004 error case)
        assert resp.status_code == 200
        config = mock_validate_use_case.execute.await_args.args[0]
        assert config.headers == {}

    async def test_injected_credential_wins_over_colliding_header(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock
    ):
        # Arrange — an explicit Authorization header AND user_token (REQ-004:
        # "injected credential takes precedence over colliding header names")
        self._wire_jwt_caller(method="jwt", credential="injected-jwt")
        body = {
            "url": "https://target.example.com/mcp",
            "auth_mode": "user_token",
            "headers": {"Authorization": "Bearer explicit"},
        }

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert — one header only: the INJECTED credential
        assert resp.status_code == 200
        config = mock_validate_use_case.execute.await_args.args[0]
        assert config.headers == {"Authorization": "Bearer injected-jwt"}

    async def test_openapi_user_token_injects_credential_into_fetch_headers(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock
    ):
        # Arrange — an openapi server in user_token mode: the spec fetch is
        # the probe's target, so the credential reaches its headers too
        self._wire_jwt_caller(method="api_key", credential="key-alice")
        body = {
            "source_type": "openapi",
            "openapi_url": "https://specs.example.com/openapi.json",
            "auth_mode": "user_token",
        }

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert
        assert resp.status_code == 200
        headers = mock_validate_use_case.execute_openapi.await_args.kwargs["headers"]
        assert headers == {"X-API-Key": "key-alice"}


# -- validate-with-id: probe the STORED entry (UI BUG-001) ---------------------


class TestValidateStoredMerge:
    """``POST /validate`` with ``id`` of a stored server (UI BUG-001).

    The masked edit dialog cannot send the stored secrets back (REQ-016), so
    its "Test connection" payload reached the stateless validator with empty
    credential editors and a credential-REQUIRED target answered a FALSE 401.
    The route now merges the probe over the STORED (decrypted) entry: only a
    real payload value overrides a stored one, the response stays the standard
    validate payload (no stored secret is ever echoed), and an id that is not
    the caller's own entry answers 404 (per-user isolation).
    """

    def _stored_with_secrets(
        self,
        name: str = "web-search",
        auth_mode: McpAuthMode = McpAuthMode.HEADERS,
    ) -> RegisteredMcpServer:
        """A stored entry as the registry store returns it (secrets decrypted)."""
        return RegisteredMcpServer(
            name=name,
            url="https://stored.example.com/mcp",
            headers={"X-API-Key": "stored-key"},
            env={"STORED_VAR": "1"},
            auth_token="stored-bearer",
            auth_mode=auth_mode,
            tool_count=3,
            source_type="external",
            created_at=_now(),
            updated_at=_now(),
        )

    async def test_stored_credentials_supply_the_probe_config(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock, mock_get_use_case
    ):
        # Arrange — the edit-dialog shape: id + masked-empty credential fields
        mock_get_use_case.execute = AsyncMock(return_value=self._stored_with_secrets())
        mock_validate_use_case.execute = AsyncMock(return_value=5)
        body = {
            "id": "web-search",
            "name": "web-search",  # ignored (extra) — id is the reference
            "source_type": "external",
            "url": "https://stored.example.com/mcp",
            "headers": {},
            "env": {},
            "auth_token": None,
            "auth_mode": "headers",
        }

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert — the probe ran with the STORED (decrypted) values
        assert resp.status_code == 200, resp.text
        assert resp.json()["tool_count"] == 5
        assert set(resp.json().keys()) == {"tool_count"}  # no secret echo
        config = mock_validate_use_case.execute.await_args.args[0]
        assert config.name == "web-search"
        assert config.url == "https://stored.example.com/mcp"
        assert config.headers == {"X-API-Key": "stored-key"}
        assert config.auth_token == "stored-bearer"
        mock_get_use_case.execute.assert_awaited_once_with("web-search")

    async def test_masked_empty_values_do_not_wipe_the_stored_secrets(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock, mock_get_use_case
    ):
        # Arrange — the stored entry keeps its secrets; the payload is the
        # masked projection (empty editors) sent by the REQ-016 UI
        mock_get_use_case.execute = AsyncMock(return_value=self._stored_with_secrets())
        mock_validate_use_case.execute = AsyncMock(return_value=2)
        body = {
            "id": "web-search",
            "url": "https://stored.example.com/mcp",
            "headers": {},
            "env": {},
            "auth_token": "",
            "auth_mode": "headers",
        }

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert — the stored headers/env/auth_token reached the probe intact
        assert resp.status_code == 200, resp.text
        config = mock_validate_use_case.execute.await_args.args[0]
        assert config.headers == {"X-API-Key": "stored-key"}
        assert config.env == {"STORED_VAR": "1"}
        assert config.auth_token == "stored-bearer"

    async def test_explicit_payload_values_override_the_stored_ones(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock, mock_get_use_case
    ):
        # Arrange — the user edited url + key + token in the dialog
        mock_get_use_case.execute = AsyncMock(return_value=self._stored_with_secrets())
        mock_validate_use_case.execute = AsyncMock(return_value=7)
        body = {
            "id": "web-search",
            "url": "https://changed.example.com/mcp",
            "headers": {"X-API-Key": "changed-key"},
            "auth_token": "changed-bearer",
        }

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert — the dirty fields win; the untouched env stays stored
        assert resp.status_code == 200, resp.text
        config = mock_validate_use_case.execute.await_args.args[0]
        assert config.url == "https://changed.example.com/mcp"
        assert config.headers == {"X-API-Key": "changed-key"}
        assert config.auth_token == "changed-bearer"
        assert config.env == {"STORED_VAR": "1"}

    async def test_stored_auth_mode_is_used_and_explicit_mode_overrides(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock, mock_get_use_case
    ):
        # Arrange — a user_token-stored server, payload WITHHELS the mode
        mock_get_use_case.execute = AsyncMock(
            return_value=self._stored_with_secrets(auth_mode=McpAuthMode.USER_TOKEN)
        )
        mock_validate_use_case.execute = AsyncMock(return_value=2)
        app.dependency_overrides[get_request_auth_context] = lambda: AuthContext(
            user_id="user-alice", method="jwt", raw_credential="caller-jwt"
        )
        body = {"id": "web-search"}

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert — stored mode drives the injection; stored key + caller JWT
        # (the injected credential wins only over COLLIDING names)
        assert resp.status_code == 200, resp.text
        config = mock_validate_use_case.execute.await_args.args[0]
        assert config.headers == {
            "X-API-Key": "stored-key",
            "Authorization": "Bearer caller-jwt",
        }

        # Arrange 2 — an EXPLICIT mode wins over the stored one
        app.dependency_overrides[get_request_auth_context] = lambda: None
        body2 = {"id": "web-search", "auth_mode": "headers"}
        mock_validate_use_case.execute = AsyncMock(return_value=2)

        # Act 2
        resp2 = await client.post(f"{BASE_URL}/validate", json=body2)

        # Assert 2 — no injection: the headers-mode probe runs as-is
        assert resp2.status_code == 200, resp2.text
        config2 = mock_validate_use_case.execute.await_args.args[0]
        assert config2.headers == {"X-API-Key": "stored-key"}

    async def test_id_against_a_disabled_registry_answers_503(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock
    ):
        # Arrange — the registry store is NOT wired (lifespan never ran): the
        # id-driven probe cannot resolve, so it answers the "registry
        # disabled" 503 (the stateless no-id path is unaffected)
        app.dependency_overrides.pop(get_get_mcp_server_use_case_optional, None)
        body = {"id": "any", "url": "https://target.example.com/mcp"}

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert — 503, and no probe was fired
        assert resp.status_code == 503, resp.text
        mock_validate_use_case.execute.assert_not_awaited()

    async def test_stateless_validate_without_id_never_touches_the_store(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock, mock_get_use_case
    ):
        # Arrange — NO registry at all (the store is disabled): the legacy
        # stateless candidate probe must keep working untouched
        app.dependency_overrides.pop(get_get_mcp_server_use_case_optional, None)
        mock_validate_use_case.execute = AsyncMock(return_value=4)
        body = {"url": "https://target.example.com/mcp"}

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert
        assert resp.status_code == 200, resp.text
        assert resp.json()["tool_count"] == 4
        mock_get_use_case.execute.assert_not_awaited()

    async def test_unknown_or_foreign_id_answers_404(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock, mock_get_use_case
    ):
        # Arrange — the get use case applies the caller's RLS scope (a foreign
        # id and an unknown id raise the SAME not-found domain error)
        mock_get_use_case.execute = AsyncMock(
            side_effect=McpServerNotFoundError(
                ErrorMessage.MCP_SERVER_NOT_FOUND.format(name="ransom")
            )
        )
        mock_validate_use_case.execute = AsyncMock(return_value=2)
        body = {"id": "ransom"}

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert — 404, and NO probe was ever fired
        assert resp.status_code == 404
        mock_validate_use_case.execute.assert_not_awaited()

    async def test_openapi_id_merge_uses_the_stored_spec(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock, mock_get_use_case
    ):
        # Arrange — a stored openapi server validated WITHOUT payload fields
        mock_get_use_case.execute = AsyncMock(
            return_value=RegisteredMcpServer(
                name="petstore",
                url="http://api:8000/generated/petstore/mcp",
                headers={"X-API-Key": "stored-key"},
                env={},
                auth_token=None,
                auth_mode=McpAuthMode.HEADERS,
                tool_count=3,
                source_type="openapi",
                openapi_url="https://specs.example.com/openapi.json",
                created_at=_now(),
                updated_at=_now(),
            )
        )
        mock_validate_use_case.execute_openapi = AsyncMock(return_value=9)
        body = {"id": "petstore"}

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert — the stored spec url + stored headers drove the fetch
        assert resp.status_code == 200, resp.text
        assert resp.json()["tool_count"] == 9
        kwargs = mock_validate_use_case.execute_openapi.await_args.kwargs
        assert kwargs["openapi_url"] == "https://specs.example.com/openapi.json"
        assert kwargs["headers"] == {"X-API-Key": "stored-key"}

    # -- the ``server_id`` wire alias (BUG-005) --------------------------------

    async def test_server_id_alias_resolves_the_stored_entry(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock, mock_get_use_case
    ):
        """BUG-005: the composable-ui edit dialog sends the stored-entry
        reference as ``server_id`` — the alias must resolve the SAME merge as
        ``id`` (the two halves of the wire contract speak both names)."""
        # Arrange — the edit-dialog shape with the UI's field name
        mock_get_use_case.execute = AsyncMock(return_value=self._stored_with_secrets())
        mock_validate_use_case.execute = AsyncMock(return_value=5)
        body = {
            "server_id": "web-search",
            "name": "web-search",
            "source_type": "external",
            "url": "https://stored.example.com/mcp",
            "headers": {},
            "env": {},
            "auth_mode": "headers",
        }

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert — the merge ran over the referenced stored entry
        assert resp.status_code == 200, resp.text
        mock_get_use_case.execute.assert_awaited_once_with("web-search")
        config = mock_validate_use_case.execute.await_args.args[0]
        assert config.headers == {"X-API-Key": "stored-key"}
        assert config.auth_token == "stored-bearer"

    async def test_id_wins_when_both_reference_names_are_sent(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock, mock_get_use_case
    ):
        # Arrange — both forms in one payload: ``id`` is the primary name
        mock_get_use_case.execute = AsyncMock(return_value=self._stored_with_secrets())
        mock_validate_use_case.execute = AsyncMock(return_value=5)
        body = {"id": "web-search", "server_id": "server-ghost"}

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert — the PRIMARY name resolved the stored entry
        assert resp.status_code == 200, resp.text
        mock_get_use_case.execute.assert_awaited_once_with("web-search")

    async def test_server_id_alias_on_a_foreign_entry_answers_404(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock, mock_get_use_case
    ):
        # Arrange — the alias flows through the same RLS-scoped lookup
        mock_get_use_case.execute = AsyncMock(
            side_effect=McpServerNotFoundError(
                ErrorMessage.MCP_SERVER_NOT_FOUND.format(name="ransom")
            )
        )
        body = {"server_id": "ransom"}

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert — identical 404 semantics under the alias (no probe fired)
        assert resp.status_code == 404, resp.text
        mock_validate_use_case.execute.assert_not_awaited()

    async def test_real_provider_reads_the_request_contextvars(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock
    ):
        # Arrange — the REAL get_request_auth_context dependency (not
        # overridden): the RLS contextvars set by verify_credentials must be
        # projected into the domain rule's input
        token = current_credential.set("jwt-token")
        method_token = current_auth_method.set("jwt")
        body = {"url": "https://target.example.com/mcp", "auth_mode": "user_token"}

        # Act
        try:
            resp = await client.post(f"{BASE_URL}/validate", json=body)
        finally:
            current_auth_method.reset(method_token)
            current_credential.reset(token)

        # Assert
        assert resp.status_code == 200
        config = mock_validate_use_case.execute.await_args.args[0]
        assert config.headers == {"Authorization": "Bearer jwt-token"}

    async def test_real_provider_api_key_context_injects_x_api_key(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock
    ):
        # Arrange — the REAL provider with an api_key context (the dual-auth
        # X-API-Key caller shape)
        token = current_credential.set("key-alice")
        method_token = current_auth_method.set("api_key")
        body = {"url": "https://target.example.com/mcp", "auth_mode": "user_token"}

        # Act
        try:
            resp = await client.post(f"{BASE_URL}/validate", json=body)
        finally:
            current_auth_method.reset(method_token)
            current_credential.reset(token)

        # Assert
        assert resp.status_code == 200
        config = mock_validate_use_case.execute.await_args.args[0]
        assert config.headers == {"X-API-Key": "key-alice"}

    async def test_real_provider_drops_a_credential_without_a_dual_auth_method(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock
    ):
        # Arrange — a credential with NO method (legacy master-key path: no
        # contextvar was set by verify_api_key) → never guess an injection
        token = current_credential.set("master-key")
        body = {"url": "https://target.example.com/mcp", "auth_mode": "user_token"}

        # Act
        try:
            resp = await client.post(f"{BASE_URL}/validate", json=body)
        finally:
            current_credential.reset(token)

        # Assert — the header is DROPPED (drop-empty semantics)
        assert resp.status_code == 200
        config = mock_validate_use_case.execute.await_args.args[0]
        assert config.headers == {}


# -- GET /api/v1/mcp/servers (list) --------------------------------------------


class TestListMcpServersRoute:
    """Tests for GET /api/v1/mcp/servers."""

    async def test_returns_200_with_list(
        self, client: AsyncClient, mock_list_use_case: AsyncMock
    ):
        # Act
        resp = await client.get(BASE_URL)

        # Assert
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) == 2

    async def test_masks_secrets_in_list_response(
        self, client: AsyncClient, mock_list_use_case: AsyncMock
    ):
        # Act
        resp = await client.get(BASE_URL)

        # Assert
        for item in resp.json():
            assert item["auth_token"] is None
            assert item["headers"] == {}
            assert item["env"] == {}


# -- GET /api/v1/mcp/servers/{name} (get) --------------------------------------


class TestGetMcpServerRoute:
    """Tests for GET /api/v1/mcp/servers/{name}."""

    async def test_returns_200_with_entry(
        self, client: AsyncClient, mock_get_use_case: AsyncMock
    ):
        # Arrange
        mock_get_use_case.execute = AsyncMock(return_value=_masked_entry("web-search"))
        # Act
        resp = await client.get(f"{BASE_URL}/web-search")

        # Assert
        assert resp.status_code == 200
        assert resp.json()["name"] == "web-search"

    async def test_masks_secrets_in_get_response(
        self, client: AsyncClient, mock_get_use_case: AsyncMock
    ):
        # Arrange
        mock_get_use_case.execute = AsyncMock(return_value=_masked_entry("web-search"))

        # Act
        resp = await client.get(f"{BASE_URL}/web-search")

        # Assert
        assert resp.json()["auth_token"] is None
        assert resp.json()["headers"] == {}

    async def test_returns_404_when_not_found(
        self, client: AsyncClient, mock_get_use_case: AsyncMock
    ):
        # Arrange
        mock_get_use_case.execute = AsyncMock(
            side_effect=McpServerNotFoundError("not found")
        )

        # Act
        resp = await client.get(f"{BASE_URL}/missing")

        # Assert
        assert resp.status_code == 404


# -- GET /api/v1/mcp/servers/{name}/reveal -------------------------------------


class TestRevealMcpServerRoute:
    """Tests for GET /api/v1/mcp/servers/{name}/reveal (plaintext secrets)."""

    async def test_returns_200_with_plaintext_secrets(
        self, client: AsyncClient, mock_get_use_case: AsyncMock
    ):
        # Arrange — reveal returns the entry with secrets intact
        mock_get_use_case.execute = AsyncMock(
            return_value=_entry("web-search", auth_token="secret")
        )

        # Act
        resp = await client.get(f"{BASE_URL}/web-search/reveal")

        # Assert
        assert resp.status_code == 200
        assert resp.json()["auth_token"] == "secret"
        assert resp.json()["headers"] == {"Authorization": "Bearer tok"}

    async def test_returns_404_when_not_found(
        self, client: AsyncClient, mock_get_use_case: AsyncMock
    ):
        # Arrange
        mock_get_use_case.execute = AsyncMock(
            side_effect=McpServerNotFoundError("not found")
        )

        # Act
        resp = await client.get(f"{BASE_URL}/missing/reveal")

        # Assert
        assert resp.status_code == 404

    async def test_corrupted_auth_mode_maps_to_a_structured_500(
        self,
        client: AsyncClient,
        mock_get_use_case: AsyncMock,
        caplog: pytest.LogCaptureFixture,
    ):
        # Arrange — a corrupted stored auth_mode surfacing through the REAL
        # domain-error handler chain (502-class registry: McpError 500)
        mock_get_use_case.execute = AsyncMock(
            side_effect=McpServerInvalidAuthModeError(
                ErrorMessage.MCP_SERVER_AUTH_MODE_INVALID.format(
                    value="client_credentials"
                )
            )
        )

        # Act
        with caplog.at_level(logging.INFO, logger="application.error_handlers"):
            resp = await client.get(f"{BASE_URL}/corrupted/reveal")

        # Assert — a CONTROLLED 500 (not a raw ValueError stack), with the
        # CORRECT centralized log line (not "Unhandled domain error")
        assert resp.status_code == 500
        assert resp.json()["detail"] == (
            "MCP server registry contains an invalid stored auth_mode: "
            "client_credentials"
        )
        assert any(
            "MCP invalid stored configuration" in r.getMessage() for r in caplog.records
        )
        assert not any(
            "Unhandled domain error" in r.getMessage() for r in caplog.records
        )


# -- PUT /api/v1/mcp/servers/{name} (update) -----------------------------------


class TestUpdateMcpServerRoute:
    """Tests for PUT /api/v1/mcp/servers/{name}."""

    async def test_returns_200_with_updated_entry(
        self, client: AsyncClient, mock_update_use_case: AsyncMock
    ):
        # Arrange
        mock_update_use_case.execute = AsyncMock(
            return_value=_masked_entry("web-search", tool_count=5)
        )
        body = {"url": "http://new:3001/mcp"}

        # Act
        resp = await client.put(f"{BASE_URL}/web-search", json=body)

        # Assert
        assert resp.status_code == 200
        assert resp.json()["tool_count"] == 5

    async def test_masks_secrets_in_update_response(
        self, client: AsyncClient, mock_update_use_case: AsyncMock
    ):
        # Arrange
        mock_update_use_case.execute = AsyncMock(
            return_value=_masked_entry("web-search")
        )
        body = {"url": "http://new:3001/mcp", "auth_token": "new-secret"}

        # Act
        resp = await client.put(f"{BASE_URL}/web-search", json=body)

        # Assert
        assert resp.json()["auth_token"] is None

    async def test_returns_404_when_not_found(
        self, client: AsyncClient, mock_update_use_case: AsyncMock
    ):
        # Arrange
        mock_update_use_case.execute = AsyncMock(
            side_effect=McpServerNotFoundError("not found")
        )
        body = {"url": "http://new:3001/mcp"}

        # Act
        resp = await client.put(f"{BASE_URL}/missing", json=body)

        # Assert
        assert resp.status_code == 404


# -- DELETE /api/v1/mcp/servers/{name} -----------------------------------------


class TestDeleteMcpServerRoute:
    """Tests for DELETE /api/v1/mcp/servers/{name}."""

    async def test_returns_204_on_success(
        self, client: AsyncClient, mock_delete_use_case: AsyncMock
    ):
        # Act
        resp = await client.delete(f"{BASE_URL}/web-search")

        # Assert
        assert resp.status_code == 204

    async def test_returns_404_when_not_found(
        self, client: AsyncClient, mock_delete_use_case: AsyncMock
    ):
        # Arrange
        mock_delete_use_case.execute = AsyncMock(
            side_effect=McpServerNotFoundError("not found")
        )

        # Act
        resp = await client.delete(f"{BASE_URL}/missing")

        # Assert
        assert resp.status_code == 404


class TestPartialUpdateMergeContract:
    """BUG-001/BUG-002/BUG-003 regression tests: PUT /{name} is a PARTIAL update.

    Only the provided fields change; unprovided secrets/configuration are
    preserved from the decrypted existing entry (the registry owns the merge),
    and ``tool_count`` is inherited instead of being reset to 0.
    """

    async def test_partial_update_preserves_unspecified_secret_fields(
        self,
        client: AsyncClient,
        mock_get_use_case: AsyncMock,
        mock_update_use_case: AsyncMock,
    ):
        # Arrange — the existing entry carries secrets; the body sends ONLY url
        existing = _entry("web-search", auth_token="keep-secret-token")
        existing = existing.model_copy(
            update={
                "headers": {"X-K": "keep-v"},
                "env": {"E": "keep-v"},
                "tool_count": 4,
            }
        )
        mock_get_use_case.execute = AsyncMock(return_value=existing)
        body = {"url": "http://new:3001/mcp"}

        # Act
        resp = await client.put(f"{BASE_URL}/web-search", json=body)

        # Assert — the merged entity handed to the update use case preserves
        # every unprovided field, including the plaintext secrets.
        assert resp.status_code == 200
        merged = mock_update_use_case.execute.call_args.args[1]
        assert merged.url == "http://new:3001/mcp"
        assert merged.auth_token == "keep-secret-token"
        assert merged.headers == {"X-K": "keep-v"}
        assert merged.env == {"E": "keep-v"}
        assert merged.tool_count == 4

    async def test_tool_count_is_not_reset_to_zero(
        self, client, mock_get_use_case: AsyncMock, mock_update_use_case: AsyncMock
    ):
        # Arrange — url UNCHANGED (no re-validation): tool_count must be inherited
        existing = _entry("web-search", tool_count=6)
        mock_get_use_case.execute = AsyncMock(return_value=existing)
        body = {"headers": {"X-Updated": "v"}}

        # Act
        resp = await client.put(f"{BASE_URL}/web-search", json=body)

        # Assert
        assert resp.status_code == 200
        merged = mock_update_use_case.execute.call_args.args[1]
        assert merged.tool_count == 6, (
            "an unchanged-url update must not reset tool_count"
        )

    async def test_provided_map_fields_replace_the_stored_maps(
        self, client, mock_get_use_case: AsyncMock, mock_update_use_case: AsyncMock
    ):
        # Arrange — headers/env provided in the body REPLACE the stored maps
        existing = _entry("web-search")
        existing = existing.model_copy(
            update={"headers": {"X-Old": "old"}, "env": {"OLD": "old"}, "tool_count": 2}
        )
        mock_get_use_case.execute = AsyncMock(return_value=existing)
        body = {"headers": {"X-New": "new"}, "env": {"NEW": "new"}}

        # Act
        resp = await client.put(f"{BASE_URL}/web-search", json=body)

        # Assert
        assert resp.status_code == 200
        merged = mock_update_use_case.execute.call_args.args[1]
        assert merged.headers == {"X-New": "new"}
        assert merged.env == {"NEW": "new"}

    async def test_incoherent_merged_body_returns_422_with_actionable_message(
        self, client, mock_get_use_case: AsyncMock, mock_update_use_case: AsyncMock
    ):
        # Arrange — the merged result loses its url for an external server
        existing = _entry("web-search")
        mock_get_use_case.execute = AsyncMock(return_value=existing)
        mock_update_use_case.execute = AsyncMock(return_value=existing)

        # Act
        resp = await client.put(
            f"{BASE_URL}/web-search", json={"headers": {"X-Only": "v"}, "url": None}
        )

        # Assert — stable, actionable, non-sensitive message (BUG-002)
        assert resp.status_code == 422
        messages = {issue.get("msg", "") for issue in resp.json()["detail"]}
        assert any(
            "'url' is required when source_type='external'" in msg for msg in messages
        )
        mock_update_use_case.execute.assert_not_awaited()

    async def test_empty_body_is_rejected_422(
        self, client, mock_get_use_case: AsyncMock
    ):
        # Arrange
        body: dict = {}

        # Act
        resp = await client.put(f"{BASE_URL}/web-search", json=body)

        # Assert
        assert resp.status_code == 422


# -- auth_mode echo (feat-mcp-oauth2-dual-auth, registry M-1) -------------------


class TestAuthModeEcho:
    """``auth_mode`` (REQ-004) is persisted, echoed unmasked and typed.

    Contract with composable-ui (which posts ``auth_mode``) and
    composable-agents (whose registry client reads the echoed mode and warns
    when it is dropped — M-5):

    * create/validate accept ``auth_mode`` and the masked create/list/get
      responses ECHO it unmasked (non-secret metadata — unlike headers/auth_
      token/env);
    * an unknown value is rejected with 422 (never silently degraded);
    * PUT is a PARTIAL update: ABSENT ``auth_mode`` preserves the stored
      value, an explicit value updates it, an explicit ``null`` is rejected.
    """

    async def test_create_accepts_user_token_and_echoes_it_unmasked(
        self, client: AsyncClient, mock_create_use_case: AsyncMock
    ):
        # Arrange — the use case returns the persisted masked entry
        mock_create_use_case.execute = AsyncMock(
            return_value=_masked_entry("server-x", auth_mode=McpAuthMode.USER_TOKEN)
        )
        body = {
            "name": "server-x",
            "url": "http://localhost:3001/mcp",
            "auth_mode": "user_token",
        }

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert — the response echoes the mode while the secrets stay masked
        assert resp.status_code == 201
        assert resp.json()["auth_mode"] == "user_token"
        assert resp.json()["auth_token"] is None
        assert resp.json()["headers"] == {}

        # The entity handed to the use case carries the requested mode
        sent = mock_create_use_case.execute.call_args.args[0]
        assert sent.auth_mode is McpAuthMode.USER_TOKEN

    async def test_create_defaults_to_headers_when_absent(
        self, client: AsyncClient, mock_create_use_case: AsyncMock
    ):
        # Arrange
        mock_create_use_case.execute = AsyncMock(return_value=_masked_entry("server-x"))
        body = {"name": "server-x", "url": "http://localhost:3001/mcp"}

        # Act
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 201
        assert resp.json()["auth_mode"] == "headers"
        sent = mock_create_use_case.execute.call_args.args[0]
        assert sent.auth_mode is McpAuthMode.HEADERS

    async def test_create_rejects_unknown_auth_mode_422(
        self, client: AsyncClient, mock_create_use_case: AsyncMock
    ):
        # Arrange
        body = {
            "name": "server-x",
            "url": "http://localhost:3001/mcp",
            "auth_mode": "basic",
        }

        # Act — an unknown mode is REJECTED, never silently degrading
        resp = await client.post(BASE_URL, json=body)

        # Assert
        assert resp.status_code == 422
        mock_create_use_case.execute.assert_not_awaited()

    async def test_validate_accepts_user_token(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock
    ):
        # Arrange
        mock_validate_use_case.execute = AsyncMock(return_value=2)
        body = {
            "url": "http://localhost:3001/mcp",
            "auth_mode": "user_token",
        }

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert
        assert resp.status_code == 200
        assert resp.json()["tool_count"] == 2

    async def test_validate_rejects_unknown_auth_mode_422(
        self, client: AsyncClient, mock_validate_use_case: AsyncMock
    ):
        # Arrange
        body = {
            "url": "http://localhost:3001/mcp",
            "auth_mode": "bearer",
        }

        # Act
        resp = await client.post(f"{BASE_URL}/validate", json=body)

        # Assert
        assert resp.status_code == 422
        mock_validate_use_case.execute.assert_not_awaited()

    async def test_list_echoes_auth_mode_with_masked_secrets(
        self, client: AsyncClient, mock_list_use_case: AsyncMock
    ):
        # Arrange — one user_token entry, one default entry
        mock_list_use_case.execute = AsyncMock(
            return_value=[
                _masked_entry("server-a", auth_mode=McpAuthMode.USER_TOKEN),
                _masked_entry("server-b"),
            ]
        )

        # Act
        resp = await client.get(BASE_URL)

        # Assert
        assert resp.status_code == 200
        by_name = {item["name"]: item for item in resp.json()}
        assert by_name["server-a"]["auth_mode"] == "user_token"
        assert by_name["server-b"]["auth_mode"] == "headers"
        for item in by_name.values():
            assert item["auth_token"] is None
            assert item["headers"] == {}
            assert item["env"] == {}

    async def test_get_echoes_auth_mode(self, client, mock_get_use_case: AsyncMock):
        # Arrange
        mock_get_use_case.execute = AsyncMock(
            return_value=_masked_entry("web-search", auth_mode=McpAuthMode.USER_TOKEN)
        )

        # Act
        resp = await client.get(f"{BASE_URL}/web-search")

        # Assert
        assert resp.status_code == 200
        assert resp.json()["auth_mode"] == "user_token"

    async def test_reveal_echoes_auth_mode(self, client, mock_get_use_case: AsyncMock):
        # Arrange — reveal keeps plaintext secrets AND the mode
        mock_get_use_case.execute = AsyncMock(
            return_value=_entry("web-search", auth_mode=McpAuthMode.USER_TOKEN)
        )

        # Act
        resp = await client.get(f"{BASE_URL}/web-search/reveal")

        # Assert
        assert resp.status_code == 200
        assert resp.json()["auth_mode"] == "user_token"
        assert resp.json()["auth_token"] == "secret-token"

    async def test_partial_update_explicit_auth_mode_updates_stored_value(
        self, client, mock_get_use_case: AsyncMock, mock_update_use_case: AsyncMock
    ):
        # Arrange — existing entry in headers mode; the client switches to OAuth2
        existing = _entry("web-search")
        mock_get_use_case.execute = AsyncMock(return_value=existing)
        mock_update_use_case.execute = AsyncMock(
            return_value=_masked_entry("web-search", auth_mode=McpAuthMode.USER_TOKEN)
        )
        body = {"auth_mode": "user_token"}

        # Act
        resp = await client.put(f"{BASE_URL}/web-search", json=body)

        # Assert — the merged entity carries the NEW mode, other fields preserved
        assert resp.status_code == 200
        merged = mock_update_use_case.execute.call_args.args[1]
        assert merged.auth_mode is McpAuthMode.USER_TOKEN
        assert merged.auth_token == "secret-token"
        assert merged.headers == {"Authorization": "Bearer tok"}

    async def test_partial_update_absent_auth_mode_preserves_stored_value(
        self, client, mock_get_use_case: AsyncMock, mock_update_use_case: AsyncMock
    ):
        # Arrange — existing user_token entry; the body does NOT mention auth_mode
        existing = _entry("web-search", auth_mode=McpAuthMode.USER_TOKEN)
        mock_get_use_case.execute = AsyncMock(return_value=existing)
        body = {"headers": {"X-Only": "v"}}

        # Act
        resp = await client.put(f"{BASE_URL}/web-search", json=body)

        # Assert — the stored user_token mode is PRESERVED (exclude_unset)
        assert resp.status_code == 200
        merged = mock_update_use_case.execute.call_args.args[1]
        assert merged.auth_mode is McpAuthMode.USER_TOKEN
        assert merged.headers == {"X-Only": "v"}

    async def test_partial_update_rejects_null_auth_mode_422(
        self, client, mock_get_use_case: AsyncMock
    ):
        # Arrange — an explicit null is meaningless for the non-secret mode
        body = {"auth_mode": None}

        # Act
        resp = await client.put(f"{BASE_URL}/web-search", json=body)

        # Assert
        assert resp.status_code == 422
        mock_get_use_case.execute.assert_not_awaited()

    async def test_partial_update_rejects_unknown_auth_mode_422(
        self, client, mock_get_use_case: AsyncMock
    ):
        # Arrange
        body = {"auth_mode": "session"}

        # Act
        resp = await client.put(f"{BASE_URL}/web-search", json=body)

        # Assert
        assert resp.status_code == 422
        mock_get_use_case.execute.assert_not_awaited()
