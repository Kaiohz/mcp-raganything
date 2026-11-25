"""Behavioral tests — OpenAPI registry create + a mount that STAYS healthy.

QA BUG-001 (iteration 4): ``POST /api/v1/mcp/servers`` with
``source_type="openapi"`` answered HTTP 500 AND left the generated
``/generated/{name}/mcp`` mount permanently broken (every later request 500):
the registry's ``count_tools()`` listed the mounted server's tools IN PROCESS
through its credential-guarded middleware chain (``list_tools()`` →
``McpApiKeyMiddleware.on_list_tools`` → ``ToolError`` escaping fastmcp's task
group). The count is an in-process metadata listing with NO credential by
design — the runner now bypasses the middleware chain
(``list_tools(run_middleware=False)``) and the create flow survives a count
failure (tool_count=0, mount stays live).

Entry point pinned here: the REAL app + REAL REST route + REAL create use
case + REAL ``GeneratedMcpRunner`` (+ real ``FastMCP.from_openapi`` on a real
spec — no upstream traffic: tool definitions come from the spec). The only
doubles are the external boundaries: the outbound registry store
(in-memory :class:`McpServerRegistryStore` double — the real adapter needs
the asyncpg pool + Fernet key) and the OpenAPI spec document fetch (the
upstream spec URL is unreachable from the in-process app).

Contracts:

* create-and-count happy path — HTTP 201, the persisted entry carries the
  discovered tool count and the mounted URL;
* the generated mount answers an UNAUTHENTICATED request with a clean
  HTTP 401 + its mount-specific discovery challenge (mount NOT corrupted);
* the generated mount serves AUTHENTICATED traffic (tools/list → the
  spec-derived tool) — the QA ``rag-34/35/36`` contract on the
  registry-created mount, never the QA ``rag-38`` 500.
"""

from contextlib import asynccontextmanager

import pytest

from domain.entities.mcp_server_registry_entry import RegisteredMcpServer
from domain.errors.mcp import McpServerNotFoundError
from domain.ports.mcp_server_registry_store import McpServerRegistryStore

pytestmark = pytest.mark.asyncio(loop_scope="session")

PROD_HOST = "raganything.soludev.tech"
WELL_KNOWN_PATH = "/.well-known/oauth-protected-resource"
SERVER_NAME = "qa-bug001-petstore"
GENERATED_MOUNT = f"/generated/{SERVER_NAME}/mcp"
#: Accepted credential from the root ``auth_directory`` external doubles
#: (the REST dependency accepts it in every wiring state: legacy master-key
#: fallback OR dual-auth hash lookup).
API_KEY_KEY_ALICE = "key-alice"
#: The upstream OpenAPI document URL (never contacted — the fetch boundary
#: is stubbed below; a tools/list never calls the spec-derived tool's
#: upstream either).
SPEC_URL = "https://specs.example.com/qa-bug001-openapi.json"

#: A REAL minimal OpenAPI 3.x document — ONE GET operation → ONE tool
#: (the QA stub's shape: `list_things`).
_OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {"title": "QA Bug001 Petstore", "version": "1.0.0"},
    "paths": {
        "/things": {
            "get": {
                "operationId": "list_things",
                "summary": "List things",
                "responses": {"200": {"description": "OK"}},
            }
        }
    },
    "servers": [{"url": "https://petstore.example.com"}],
}


class _InMemoryRegistryStore(McpServerRegistryStore):
    """Outbound registration boundary double: in-memory ``mcp_servers`` rows.

    Implements the WHOLE port (exists/get/save/list_all/delete) so the REAL
    create use case can persist; the real adapter needs the asyncpg pool +
    Fernet cipher the behavioral session deliberately does not stand up.
    """

    def __init__(self) -> None:
        self.entries: dict[str, RegisteredMcpServer] = {}

    async def save(self, entry: RegisteredMcpServer) -> None:
        self.entries[entry.name] = entry

    async def get(self, name: str) -> RegisteredMcpServer:
        entry = self.entries.get(name)
        if entry is None:
            raise McpServerNotFoundError(f"no registered server named {name!r}")
        return entry

    async def list_all(self) -> list[RegisteredMcpServer]:
        return list(self.entries.values())

    async def delete(self, name: str) -> None:
        self.entries.pop(name, None)

    async def exists(self, name: str) -> bool:
        return name in self.entries


@pytest.fixture
def registry_store(running_app, monkeypatch: pytest.MonkeyPatch):
    """Wire an in-memory registry store into the RUNNING app's provider seam.

    The REST route resolves ``CreateMcpServerUseCase`` through
    ``dependencies.get_create_mcp_server_use_case``, which reads the
    module-level ``mcp_registry_store`` singleton at REQUEST time — swapping
    the singleton (instead of patching the provider) exercises the REAL
    dependency chain end to end.
    """
    import dependencies as deps

    store = _InMemoryRegistryStore()
    monkeypatch.setattr(deps, "mcp_registry_store", store)
    return store


@pytest.fixture
def stub_spec_fetch(monkeypatch: pytest.MonkeyPatch):
    """Serve the in-process OpenAPI document from the spec-fetch boundary.

    The REAL factory validation/build path stays untouched; only the outbound
    HTTP fetch of the spec DOCUMENT is doubled (the spec URL is unreachable
    from the in-process app). The fetched document is the REAL spec the
    runner builds the mounted server from.
    """
    from infrastructure.openapi_mcp.adapter import FastMcpOpenApiFactory

    async def _fetch_spec(self, spec_url: str, headers: dict) -> dict:
        return dict(_OPENAPI_SPEC)

    monkeypatch.setattr(FastMcpOpenApiFactory, "fetch_spec", _fetch_spec)


@asynccontextmanager
async def created_openapi_server(http_client):
    """POST the openapi registration through the REAL REST route.

    Keeps the generated mount alive for the ``async with`` body (the
    follow-up probes target the mount), then ALWAYS unmounts — the
    session-scoped app never leaks the test's routes/lifespan.

    Yields:
        The create response (fully materialized — safe to assert after the
        unmount too; the tests assert inside the block for readability).
    """
    import main

    try:
        yield await http_client.post(
            "/api/v1/mcp/servers",
            json={
                "name": SERVER_NAME,
                "source_type": "openapi",
                "openapi_url": SPEC_URL,
            },
            headers={"X-API-Key": API_KEY_KEY_ALICE},
        )
    finally:
        await main.app.state.generated_mcp_runner.unmount(SERVER_NAME)


class TestOpenApiRegistryCreateAndCount:
    async def test_create_succeeds_with_discovered_tool_count(
        self, http_client, registry_store, stub_spec_fetch
    ):
        """QA BUG-001 regression pin: create + count answers 201 (never a
        500), persists the entry with the discovered tool count and the
        mounted URL."""
        async with created_openapi_server(http_client) as resp:
            body = resp.json()
            assert resp.status_code == 201, (
                f"the openapi registry create must succeed end to end, got "
                f"{resp.status_code}: {resp.text}"
            )
            assert body.get("tool_count") == 1, (
                "the count lists the ONE spec-derived tool in process WITHOUT "
                f"credentials (middleware bypassed), got: {body.get('tool_count')}"
            )
            assert body.get("source_type") == "openapi"
            assert body.get("url", "").endswith(GENERATED_MOUNT), (
                f"the mounted URL is returned to the caller, got: {body.get('url')!r}"
            )
        saved = registry_store.entries.get(SERVER_NAME)
        assert saved is not None and saved.tool_count == 1, (
            "the entry is persisted with the discovered tool count"
        )

    async def test_generated_mount_stays_unauthenticated_401_after_create(
        self, http_client, registry_store, stub_spec_fetch, mcp_post
    ):
        """The follow-up unauthenticated request on the generated mount is a
        clean 401 with the MOUNT-SPECIFIC challenge (never the corrupted 500
        of QA BUG-001, and never an MCP initialize success)."""
        async with created_openapi_server(http_client) as resp:
            assert resp.status_code == 201, resp.text
            probe = await mcp_post(http_client, GENERATED_MOUNT)
            assert probe.status_code == 401, (
                f"the generated mount must answer an unauthenticated request "
                f"with a clean 401 (mount not corrupted by the create), got "
                f"{probe.status_code}: {probe.text!r}"
            )
            assert probe.headers.get("www-authenticate") == (
                f'Bearer resource_metadata="https://{PROD_HOST}'
                f'{WELL_KNOWN_PATH}{GENERATED_MOUNT}"'
            ), (
                "the generated mount challenges with ITS OWN per-mount "
                f"document, got: {probe.headers.get('www-authenticate')!r}"
            )
            assert probe.json() == {"detail": "Invalid or missing credentials"}

    async def test_generated_mount_serves_authenticated_traffic_after_create(
        self, http_client, registry_store, stub_spec_fetch, mcp_list
    ):
        """The registry-created mount serves its spec-derived tool to an
        authenticated caller (the QA rag-35/36 contract on a created mount)."""
        async with created_openapi_server(http_client) as resp:
            assert resp.status_code == 201, resp.text
            tools = await mcp_list(GENERATED_MOUNT, credential=API_KEY_KEY_ALICE)
            tool_names = {t.name for t in tools}
            assert "list_things" in tool_names, (
                f"the registry-created mount must serve the spec-derived "
                f"tool, got: {sorted(tool_names)}"
            )
