"""Behavioral tests — registry ``auth_mode`` persistence + echo (REQ-004).

Review M-1 (composable-ui slice): the UI posts ``auth_mode`` to this
registry (POST/PUT ``/api/v1/mcp/servers``) and composable-agents' registry
client reads the echoed mode (``_to_execution_config``); a registry that
silently ignores the field makes the whole ``user_token`` path inert —
"OAuth2 — connected user's token" would register a server with NO credential
and NO stored mode, and nothing would say why.

Contracts pinned here (REAL app + REAL route + REAL create/update/get/list
use cases; the ONLY doubles are the outbound registry store boundary — the
in-memory :class:`McpServerRegistryStore` double mirroring the persisted
rows — and the outbound OpenAPI spec fetch, unreachable from the in-process
app):

* create with ``auth_mode: "user_token"`` → 201; the MASKED response echoes
  the mode unmasked (non-secret metadata, unlike headers/auth_token/env);
* GET (masked) + GET reveal + GET list echo the persisted mode;
* PUT is a PARTIAL update: ABSENT ``auth_mode`` preserves the stored value
  (the UI edit path), an explicit value updates it;
* unknown ``auth_mode`` → 422 (never silently degraded);
* persistence across a second app instantiation: a SECOND REST chain over
  the SAME store reads the persisted mode back (the mode lives in the row,
  not in the process). The real-adapter restart semantics (a fresh
  ``McpRegistryStore`` over the same backend rows) is pinned in
  ``tests/unit/test_mcp_registry_store.py::TestAuthModePersistence``.
"""

from contextlib import asynccontextmanager

import httpx
import pytest

from domain.entities.mcp_server_registry_entry import RegisteredMcpServer
from domain.errors.mcp import McpServerNotFoundError
from domain.ports.mcp_server_registry_store import McpServerRegistryStore

SERVER_NAME = "auth-mode-server"
SPEC_URL = "https://specs.example.com/auth-mode-openapi.json"
GENERATED_MOUNT = f"/generated/{SERVER_NAME}/mcp"

#: Accepted credential from the root ``auth_directory`` external doubles
#: (accepted in every REST wiring state of the behavioral session).
API_KEY_KEY_ALICE = "key-alice"

#: A REAL minimal OpenAPI 3.x document — one GET operation → one tool
#: (same stub shape as the QA BUG-001 behavioral file).
_OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {"title": "Auth Mode Registry", "version": "1.0.0"},
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

    The persisted rows the REAL store adapter writes (the real adapter needs
    the asyncpg pool + Fernet cipher the behavioral session deliberately does
    not stand up). Row-level persistence (save → read back through a fresh
    store over the same backend) is pinned by the real-adapter unit tests.
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

    The use-case providers resolve ``dependencies.mcp_registry_store`` at
    REQUEST time — swapping the singleton exercises the REAL dependency
    chain end to end.
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
    from the in-process app).
    """
    from infrastructure.openapi_mcp.adapter import FastMcpOpenApiFactory

    async def _fetch_spec(self, spec_url: str, headers: dict) -> dict:
        return dict(_OPENAPI_SPEC)

    monkeypatch.setattr(FastMcpOpenApiFactory, "fetch_spec", _fetch_spec)


@asynccontextmanager
async def registered_openapi_server(http_client: httpx.AsyncClient, payload: dict):
    """POST one registration through the REAL REST route, ALWAYS unmount.

    Keeps the generated mount alive for the ``async with`` body, then ALWAYS
    unmounts — the session-scoped app never leaks the test's mount.

    Yields:
        The create response (fully materialized — safe to assert after the
        unmount too).
    """
    import main

    try:
        yield await http_client.post(
            "/api/v1/mcp/servers",
            json=payload,
            headers={"X-API-Key": API_KEY_KEY_ALICE},
        )
    finally:
        await main.app.state.generated_mcp_runner.unmount(SERVER_NAME)


@pytest.mark.asyncio(loop_scope="session")
class TestRegistryAuthModeEcho:
    """POST/GET/PUT + list: ``auth_mode`` accepted, persisted, echoed (REQ-004)."""

    async def test_create_user_token_echoes_mode_and_masks_secrets(
        self, http_client, registry_store, stub_spec_fetch
    ):
        # Act — register in the OAuth2 mode (what the UI's auth select sends)
        async with registered_openapi_server(
            http_client,
            {
                "name": SERVER_NAME,
                "source_type": "openapi",
                "openapi_url": SPEC_URL,
                "auth_mode": "user_token",
            },
        ) as resp:
            # Assert — the masked create response echoes the mode unmasked
            assert resp.status_code == 201, resp.text
            body = resp.json()
            assert body["auth_mode"] == "user_token"
            assert body["headers"] == {}, "secrets stay masked in the echo"
            assert body["auth_token"] is None
            assert body["env"] == {}

        # The mode is persisted in the registry boundary (the whole point of M-1)
        assert registry_store.entries[SERVER_NAME].auth_mode.value == "user_token"

    async def test_get_and_list_echo_persisted_mode(
        self, http_client, registry_store, stub_spec_fetch
    ):
        # Arrange — register in the user_token mode first
        async with registered_openapi_server(
            http_client,
            {
                "name": SERVER_NAME,
                "source_type": "openapi",
                "openapi_url": SPEC_URL,
                "auth_mode": "user_token",
            },
        ) as create:
            assert create.status_code == 201, create.text

            # Act — a SECOND REST chain reads the persisted state back
            got = await http_client.get(
                f"/api/v1/mcp/servers/{SERVER_NAME}",
                headers={"X-API-Key": API_KEY_KEY_ALICE},
            )
            reveal = await http_client.get(
                f"/api/v1/mcp/servers/{SERVER_NAME}/reveal",
                headers={"X-API-Key": API_KEY_KEY_ALICE},
            )
            listed = await http_client.get(
                "/api/v1/mcp/servers",
                headers={"X-API-Key": API_KEY_KEY_ALICE},
            )

        # Assert — masked get/reveal/list all echo the persisted mode
        assert got.status_code == 200
        assert got.json()["auth_mode"] == "user_token"
        assert got.json()["auth_token"] is None, "secrets stay masked"
        assert reveal.status_code == 200
        assert reveal.json()["auth_mode"] == "user_token"
        entries = {e["name"] for e in listed.json()}
        modes = [e["auth_mode"] for e in listed.json() if e["name"] == SERVER_NAME]
        assert SERVER_NAME in entries
        assert modes == ["user_token"]

    async def test_partial_update_absent_auth_mode_preserves_stored_value(
        self, http_client, registry_store, stub_spec_fetch
    ):
        # Arrange — a user_token server the UI edits WITHOUT touching the mode
        async with registered_openapi_server(
            http_client,
            {
                "name": SERVER_NAME,
                "source_type": "openapi",
                "openapi_url": SPEC_URL,
                "auth_mode": "user_token",
            },
        ) as create:
            assert create.status_code == 201, create.text

            # Act — the UI edit payload omits auth_mode (REQ-016 flow: only
            # the touched fields travel; secrets/mode must be preserved)
            updated = await http_client.put(
                f"/api/v1/mcp/servers/{SERVER_NAME}",
                json={"openapi_url": SPEC_URL},
                headers={"X-API-Key": API_KEY_KEY_ALICE},
            )
            reread = await http_client.get(
                f"/api/v1/mcp/servers/{SERVER_NAME}",
                headers={"X-API-Key": API_KEY_KEY_ALICE},
            )

        # Assert — the user_token mode SURVIVED the partial update
        assert updated.status_code == 200, updated.text
        assert updated.json()["auth_mode"] == "user_token"
        assert reread.status_code == 200
        assert reread.json()["auth_mode"] == "user_token"

    async def test_partial_update_explicit_auth_mode_switches_mode(
        self, http_client, registry_store, stub_spec_fetch
    ):
        # Arrange — a headers-mode server the UI switches to OAuth2
        async with registered_openapi_server(
            http_client,
            {
                "name": SERVER_NAME,
                "source_type": "openapi",
                "openapi_url": SPEC_URL,
            },
        ) as create:
            assert create.status_code == 201, create.text
            assert create.json()["auth_mode"] == "headers"

            # Act — explicit mode change
            updated = await http_client.put(
                f"/api/v1/mcp/servers/{SERVER_NAME}",
                json={"auth_mode": "user_token"},
                headers={"X-API-Key": API_KEY_KEY_ALICE},
            )

        # Assert — the switch is persisted, not a silent no-op (M-1)
        assert updated.status_code == 200, updated.text
        assert updated.json()["auth_mode"] == "user_token"
        assert registry_store.entries[SERVER_NAME].auth_mode.value == "user_token"

    async def test_unknown_auth_mode_is_rejected_422_before_persistence(
        self, http_client, registry_store, stub_spec_fetch
    ):
        # Act — an unsupported mode must be REJECTED, never silently degraded
        resp = await http_client.post(
            "/api/v1/mcp/servers",
            json={
                "name": SERVER_NAME,
                "source_type": "openapi",
                "openapi_url": SPEC_URL,
                "auth_mode": "client_credentials",
            },
            headers={"X-API-Key": API_KEY_KEY_ALICE},
        )

        # Assert — 422 and NOTHING persisted (the generated mount never exists)
        assert resp.status_code == 422, resp.text
        assert await registry_store.exists(SERVER_NAME) is False

    async def test_mode_survives_a_second_app_instantiation(
        self, http_client, registry_store, stub_spec_fetch
    ):
        """Restart semantics: the persisted mode is read back by a new chain.

        The mode lives in the persisted row (the store boundary), not in the
        process: a SECOND REST chain over the SAME store — the app's
        startup rehydrate path (``list_all``) plus the REST read the agent's
        registry client performs — must observe the persisted mode.
        """
        # Arrange — register in the user_token mode
        async with registered_openapi_server(
            http_client,
            {
                "name": SERVER_NAME,
                "source_type": "openapi",
                "openapi_url": SPEC_URL,
                "auth_mode": "user_token",
            },
        ) as create:
            assert create.status_code == 201, create.text

            # Act — second instantiation: the startup rehydrate read
            # (fresh client, same persisted rows)
            entries = await registry_store.list_all()
            second_chain_get = await http_client.get(
                f"/api/v1/mcp/servers/{SERVER_NAME}/reveal",
                headers={"X-API-Key": API_KEY_KEY_ALICE},
            )

        # Assert — both rehydrate and REST observe the persisted mode
        modes = [e.auth_mode.value for e in entries if e.name == SERVER_NAME]
        assert modes == ["user_token"], (
            "the startup rehydrate path must see the persisted mode"
        )
        assert second_chain_get.status_code == 200
        assert second_chain_get.json()["auth_mode"] == "user_token"
