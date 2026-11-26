import asyncio
import importlib.util
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
from fastmcp import Client
from fastmcp.client.client import CallToolResult
from fastmcp.client.transports import StreamableHttpTransport
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData, TextContent

from domain.entities.indexing_result import (
    FileIndexingResult,
    FolderIndexingResult,
    FolderIndexingStats,
    IndexingStatus,
)
from domain.services.auth.auth_service import AuthService

#: The production proxy Host (Host-header stand-in for the TLS-terminating
#: ingress; the test app itself runs in-process).
_PRODUCTION_PROXY_HOST = "raganything.soludev.tech"

#: MCP_ALLOWED_HOSTS seeded for the behavioral session: the public proxy
#: host, the ``mcp_client_factory`` base-URL authority (the REAL MCP client
#: below dials ``http://test<mount>``) and the dev/test loopback defaults
#: (REQ-014). The transport-security + discovery allow-list is enforced at
#: HTTP level, so the client host MUST stay in the list.
_SEEDED_MCP_ALLOWED_HOSTS = (
    _PRODUCTION_PROXY_HOST,
    "test",
    "127.0.0.1:*",
    "localhost:*",
    "[::1]:*",
)

# Load external fixtures from tests/fixtures/external.py without __init__.py
_fixtures_path = Path(__file__).parent / "fixtures" / "external.py"
_spec = importlib.util.spec_from_file_location("external_fixtures", _fixtures_path)
assert _spec is not None
_external = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_external)

# Re-export fixtures so pytest discovers them
mock_storage = _external.mock_storage
mock_document_reader = _external.mock_document_reader
mock_vector_store = _external.mock_vector_store
mock_llm = _external.mock_llm
auth_directory = _external.auth_directory
web_external = _external.web_external
memory_external = _external.memory_external


@pytest.fixture
def sample_file_indexing_result() -> FileIndexingResult:
    """Provide a sample successful file indexing result."""
    return FileIndexingResult(
        status=IndexingStatus.SUCCESS,
        message="Document indexed successfully",
        file_path="/tmp/test/document.pdf",
        file_name="document.pdf",
        processing_time_ms=150.0,
    )


@pytest.fixture
def sample_folder_indexing_result() -> FolderIndexingResult:
    """Provide a sample successful folder indexing result."""
    return FolderIndexingResult(
        status=IndexingStatus.SUCCESS,
        message="Folder indexed successfully",
        folder_path="/tmp/test/documents",
        recursive=True,
        stats=FolderIndexingStats(
            total_files=10,
            files_processed=8,
            files_failed=1,
            files_skipped=1,
        ),
        processing_time_ms=1200.0,
    )


def _credential_headers(credential: str | None) -> dict[str, str]:
    if credential is None:
        return {}
    if credential.startswith("jwt-"):
        return {"Authorization": f"Bearer {credential}"}
    return {"X-API-Key": credential}


async def _lifespan_owner(app, started: asyncio.Event, stop: asyncio.Event) -> None:
    try:
        async with app.router.lifespan_context(app):
            started.set()
            await stop.wait()
    finally:
        started.set()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def running_app(auth_directory):
    """Run the real application and its real middleware/use-case wiring."""
    environment = {
        "WEB_SEARCH_TIMEOUT_SECONDS": "0.15",
        "CRAWL_OVERALL_TIMEOUT_SECONDS": "0.25",
        "SECRET_ENCRYPTION_KEY": "",
        # feat-mcp-oauth2-dual-auth (REQ-010/REQ-014): the raganything config
        # gains MCP_ALLOWED_HOSTS (transport-security allow-list) with the
        # feature; the value below mirrors the production configmap shape
        # (public proxy host + dev/test loopback entries). The ``test`` host
        # is the base-URL authority of the ``mcp_client_factory`` REAL MCP
        # client below — since the HTTP-level guard + transport security are
        # enforced it must stay allow-listed, or every REAL-client probe
        # would get 421. Pydantic BaseSettings ignores these env keys until
        # the fields exist, so RED runs are unaffected. LOGTO_URL feeds the
        # JWKS base; MCP_AUTHORIZATION_SERVER is the advertised issuer of the
        # protected-resource metadata (REQ-002) — seeded EXPLICITLY because
        # the ``<LOGTO_URL>/oidc`` fallback is resolved ONCE at config load
        # (the AppConfig model validator): an import-time construction with
        # no env seeding must still advertise the same issuer.
        "MCP_ALLOWED_HOSTS": ", ".join(_SEEDED_MCP_ALLOWED_HOSTS),
        "LOGTO_URL": "https://logto.soludev.tech",
        "MCP_AUTHORIZATION_SERVER": "https://logto.soludev.tech/oidc",
    }
    with patch.dict(os.environ, environment):
        import main

        auth_service = AuthService(
            jwt_port=auth_directory.jwt_port,
            api_key_repo=auth_directory.api_key_repository,
        )
        previous_middleware_auth = main.mcp_middleware._auth_service
        main.mcp_middleware.set_auth_service(auth_service)
        # Pre-declared so the ``finally`` block type-checks even when fixture
        # setup itself fails before the lifespan was started.
        lifespan_task: asyncio.Task[None] | None = None
        stop: asyncio.Event | None = None
        try:
            with (
                patch("main._run_alembic_upgrade", return_value=None),
                patch.object(main.app_config, "SECRET_ENCRYPTION_KEY", ""),
                # The MCP-OAuth2 settings are seeded at the OBJECT level too:
                # in a full-suite run the unit-test modules import ``main``
                # at COLLECTION time, so ``app_config`` is built BEFORE this
                # fixture's env patch is applied and the env seeding alone
                # could not reach it. The behavioral values match the env
                # seeding above (REQ-002/REQ-014). MCP_AUTHORIZATION_SERVER
                # is the RESOLVED advertised issuer (the model-validator
                # fallback shape: <LOGTO_URL>/oidc).
                patch.object(
                    main.app_config,
                    "MCP_ALLOWED_HOSTS",
                    list(_SEEDED_MCP_ALLOWED_HOSTS),
                ),
                patch.object(
                    main.app_config, "LOGTO_URL", "https://logto.soludev.tech"
                ),
                patch.object(
                    main.app_config,
                    "MCP_AUTHORIZATION_SERVER",
                    "https://logto.soludev.tech/oidc",
                ),
            ):
                started = asyncio.Event()
                stop = asyncio.Event()
                lifespan_task = asyncio.create_task(
                    _lifespan_owner(main.app, started, stop)
                )
                await started.wait()
                if lifespan_task.done():
                    await lifespan_task
                yield main.app
                stop.set()
                await lifespan_task
        finally:
            # ``lifespan_task``/``stop`` stay None only if setup failed before
            # the lifespan was started; both None-checks mirror the historical
            # ``locals()`` lookups without the unbound-variable pattern.
            if stop is not None:
                stop.set()
            if lifespan_task is not None:
                await lifespan_task
            main.mcp_middleware.set_auth_service(previous_middleware_auth)


@pytest.fixture
def mcp_client_factory(running_app):
    """Create real streamable-HTTP MCP clients over the mounted ASGI app."""

    @asynccontextmanager
    async def open_client(
        mount: str,
        credential: str | None = "jwt-alice",
        headers: dict[str, str] | None = None,
    ):
        request_headers = (
            _credential_headers(credential) if headers is None else headers
        )

        def httpx_client_factory(
            headers: dict[str, str] | None = None,
            timeout: httpx.Timeout | None = None,
            auth: httpx.Auth | None = None,
            follow_redirects: bool = True,
            **kwargs,
        ):
            # Matches fastmcp's McpHttpClientFactory protocol (headers, timeout
            # and auth are the kwargs the transport passes); the extra session
            # kwargs (e.g. follow_redirects) flow through ``**kwargs``. The
            # client targets the in-process ASGI app.
            return httpx.AsyncClient(
                transport=httpx.ASGITransport(app=running_app),
                base_url="http://test",
                headers=headers,
                auth=auth,
                follow_redirects=follow_redirects,
                timeout=timeout or 5,
                event_hooks={"response": [_surface_guard_rejection]},
                **kwargs,
            )

        transport = StreamableHttpTransport(
            f"http://test{mount}",
            headers=request_headers,
            httpx_client_factory=httpx_client_factory,
        )
        async with Client(transport, timeout=5, init_timeout=5) as client:
            yield client

    return open_client


async def _surface_guard_rejection(response: httpx.Response) -> None:
    """Translate the HTTP-level MCP guard's 401 into the protocol error.

    feat-mcp-oauth2-dual-auth (REQ-017) harness seam: since the HTTP-level
    guard became the primary gate, credential rejections (missing, invalid
    JWT, unknown or revoked key) answer HTTP 401 + the OAuth2 discovery
    challenge BEFORE any MCP method runs — the mcp SDK transport surfaces
    that as a raw ``httpx.HTTPStatusError`` raised inside the session
    handshake. The behavioral contract (this suite + the other behavioral
    suites) pins the PROTOCOL-level credential error the
    ``McpApiKeyMiddleware`` raised when it was the gate:
    ``McpError("Invalid or missing credentials")``. This hook converts the
    guard's generic 401 detail into that SAME protocol error so the REAL
    client observes the identical rejection contract across the migration.
    """
    if response.status_code != 401:
        return
    try:
        content = await response.aread()
        detail = json.loads(content).get("detail") or "Invalid or missing credentials"
    except Exception:  # noqa: BLE001 — a non-JSON 401 keeps the generic text
        detail = "Invalid or missing credentials"
    raise McpError(error=ErrorData(code=401, message=str(detail)))


@pytest.fixture
def mcp_call(mcp_client_factory):
    async def call(
        mount: str,
        tool: str,
        arguments: dict,
        credential: str | None = "jwt-alice",
        headers: dict[str, str] | None = None,
    ):
        # Companion seam to _surface_guard_rejection: a guard-rejected
        # handshake answers the tool CALL with the call-level error result
        # (is_error=True, no tool execution → no upstream side effect) that
        # the method-level middleware produced when it was the gate.
        try:
            async with mcp_client_factory(
                mount, credential=credential, headers=headers
            ) as client:
                return await client.call_tool(tool, arguments, raise_on_error=False)
        except McpError as exc:
            return CallToolResult(
                content=[TextContent(type="text", text=str(exc.error.message))],
                structured_content=None,
                meta=None,
                is_error=True,
            )

    return call


@pytest.fixture
def mcp_list(mcp_client_factory):
    async def list_tools(
        mount: str,
        credential: str | None = "jwt-alice",
        headers: dict[str, str] | None = None,
    ):
        async with mcp_client_factory(
            mount, credential=credential, headers=headers
        ) as client:
            return await client.list_tools()

    return list_tools
