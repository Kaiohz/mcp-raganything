"""Behavioral tests — the CREATE probe injects the caller credential (BUG-006).

UI BUG-006 (registry CREATE in the ``user_token`` mode): the create-time
reachability probe IS a load — it runs the same tool loading that records
``tool_count`` — so REQ-004's "when ``user_token``, the backend SHALL inject,
at load/validate time, the connected user's credential" applies at CREATE too.
Before the fix the create route ran the probe statelessly (plain
``McpServerConfig`` over the entry headers), so registering a self-referential
OAuth2 server against a credential-REQUIRED target answered the target's 401
→ 422 "MCP server unreachable" (the spec's own Problem Statement: the
self-referential register flow was unusable from the UI), while the SAME
target through ``POST /validate`` succeeded (the validate route injects).

What is pinned here (the FULL real chain: REAL app → REAL route → REAL
``CreateMcpServerUseCase`` → REAL ``FastMcpToolLoader`` → a REAL streamable
MCP server on a local TCP socket, with a header-recording ASGI wrapper in
front of it):

* a ``user_token`` create requested by a JWT-authenticated caller probes the
  target with ``Authorization: Bearer <caller JWT>`` — the injected
  credential REACHES the target (observed on the socket, not mocked), the
  tool_count reflects the REAL tools, and the STORED entry does NOT carry
  the credential (probe-only injection — ``reveal`` returns the configured
  headers only);
* the same for an API-key caller → ``X-API-Key: <caller key>``;
* a ``headers``-mode create probes the target with NO injected credential —
  the pre-feature behavior, unchanged.

The probe target is a REAL FastMCP streamable-HTTP server (one ``ping`` tool)
served by uvicorn on ``127.0.0.1`` in a background thread; it accepts every
caller and only RECORDS the headers it receives (the socket boundary).
"""

import socket
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from fastmcp import FastMCP
from starlette.types import ASGIApp, Receive, Scope, Send

from domain.entities.mcp_server_registry_entry import RegisteredMcpServer
from domain.errors.mcp import McpServerNotFoundError
from domain.ports.mcp_server_registry_store import McpServerRegistryStore
from infrastructure.database.rls_context import current_user_id

#: Accepted per-user API key from the root ``auth_directory`` external doubles
#: (same users as the other behavioral suites).
API_KEY_ALICE = "key-alice"

#: The create request targets a credential-REQUIRED URL; the recorder observes
#: exactly what the registry transmitted.
SERVERS_PATH = "/api/v1/mcp/servers"

#: The target starts fast; failing after this long is a bug, not a wait.
_TARGET_START_TIMEOUT_S = 15.0


class _HeaderRecordingProbeTarget:
    """ASGI wrapper around the probe FastMCP app that records inbound headers.

    Records the lowercase header mapping of every HTTP scope so the tests can
    observe EXACTLY what the registry transmitted to the target during the
    CREATE probe (the socket boundary — not a mocked loader).
    """

    def __init__(self, inner: ASGIApp) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self.seen: list[dict[str, str]] = []

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = {
                k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]
            }
            with self._lock:
                self.seen.append(headers)
        assert self._inner is not None  # noqa: S101 — wrapped in the fixture
        await self._inner(scope, receive, send)

    def reset(self) -> None:
        """Forget everything recorded so far (one observation window per test)."""
        with self._lock:
            self.seen.clear()

    def authorization_values(self) -> list[str | None]:
        """The Authorization header seen per request (``None`` = absent)."""
        with self._lock:
            return [h.get("authorization") for h in self.seen]

    def api_key_values(self) -> list[str | None]:
        """The X-API-Key header seen per request (``None`` = absent)."""
        with self._lock:
            return [h.get("x-api-key") for h in self.seen]


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def create_probe_target():
    """Serve the REAL recording MCP target with uvicorn on a local TCP socket.

    Runs in its own daemon thread (an independent event loop) so the app's
    loader connects over a REAL network socket — no ASGI plumbing between the
    loader and the target.
    """
    import uvicorn

    mcp = FastMCP("create-probe-target")

    @mcp.tool
    def ping() -> str:
        return "pong"

    recorder = _HeaderRecordingProbeTarget(
        mcp.http_app(path="/mcp", stateless_http=True)
    )
    port = _free_tcp_port()
    config = uvicorn.Config(
        recorder, host="127.0.0.1", port=port, log_level="error", access_log=False
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run, name="create-probe-target", daemon=True
    )
    thread.start()
    deadline = time.monotonic() + _TARGET_START_TIMEOUT_S
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("the recording MCP probe target did not start in time")

    handle = SimpleNamespace(base_url=f"http://127.0.0.1:{port}", recorder=recorder)

    try:
        yield handle
    finally:
        server.should_exit = True
        thread.join(timeout=10)


class _RlsAwareRegistryStore(McpServerRegistryStore):
    """Outbound registration boundary double with per-user isolation.

    Mirrors the REAL ``McpRegistryStore`` contract: every row records the
    ``current_user_id`` RLS contextvar at save time, and reads are filtered by
    the CALLER's contextvar (same double as the validate-stored suite).
    """

    def __init__(self) -> None:
        self._rows: dict[str, tuple[str, RegisteredMcpServer]] = {}

    async def save(self, entry: RegisteredMcpServer) -> None:
        self._rows[entry.name] = (current_user_id.get() or "", entry)

    async def get(self, name: str) -> RegisteredMcpServer:
        row = self._rows.get(name)
        me = current_user_id.get()
        if row is None or (me is not None and row[0] != me):
            raise McpServerNotFoundError(f"no registered server named {name!r}")
        return row[1]

    async def list_all(self) -> list[RegisteredMcpServer]:
        me = current_user_id.get()
        return [e for owner, e in self._rows.values() if me is None or owner == me]

    async def delete(self, name: str) -> None:
        await self.get(name)
        self._rows.pop(name, None)

    async def exists(self, name: str) -> bool:
        row = self._rows.get(name)
        me = current_user_id.get()
        return row is not None and (me is None or row[0] == me)


@pytest.fixture
def registry_store(running_app, monkeypatch: pytest.MonkeyPatch):
    """Wire the RLS-aware store into the RUNNING app's provider seam."""
    import dependencies as deps

    store = _RlsAwareRegistryStore()
    monkeypatch.setattr(deps, "mcp_registry_store", store)
    return store


@pytest.mark.asyncio(loop_scope="session")
class TestCreateUserTokenInjection:
    """The CREATE probe carries the caller credential in the user_token mode."""

    async def test_user_token_create_probe_carries_the_callers_jwt(
        self,
        http_client: httpx.AsyncClient,
        create_probe_target,
        registry_store,
        real_jwt_signer,
    ):
        # Arrange — the OAuth2 registration flow: the UI sends auth_mode
        # "user_token" with EMPTY secret editors, the caller holds a JWT
        token = real_jwt_signer.issue_token(sub="user-alice")
        create_probe_target.recorder.reset()

        # Act — the REAL chain: route → use case → loader → REAL target socket
        resp = await http_client.post(
            SERVERS_PATH,
            json={
                "name": "qa-create-user-token",
                "source_type": "external",
                "url": f"{create_probe_target.base_url}/mcp",
                "headers": {},
                "env": {},
                "auth_mode": "user_token",
            },
            headers={"Authorization": f"Bearer {token}"},
        )

        # Assert — the create SUCCEEDED (the probe was authenticated with the
        # INJECTED caller credential) and the tool count is REAL
        assert resp.status_code == 201, resp.text
        assert resp.json()["tool_count"] == 1
        values = create_probe_target.recorder.authorization_values()
        assert values, "the create probe must have reached the real target"
        assert values == [f"Bearer {token}"] * len(values), (
            "every MCP request of the create probe must carry the INJECTED "
            "caller credential"
        )

        # Assert — the injected credential is PROBE-ONLY: the stored entry
        # keeps exactly the configured headers (reveal = plaintext view)
        reveal = await http_client.get(
            f"{SERVERS_PATH}/qa-create-user-token/reveal",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert reveal.status_code == 200, reveal.text
        assert reveal.json()["headers"] == {}, (
            "the caller's credential must never leak into the stored entry"
        )

    async def test_user_token_create_probe_carries_the_callers_api_key(
        self,
        http_client: httpx.AsyncClient,
        create_probe_target,
        registry_store,
        real_jwt_signer,
    ):
        # Arrange — a caller authenticated with its per-user API key
        create_probe_target.recorder.reset()

        # Act
        resp = await http_client.post(
            SERVERS_PATH,
            json={
                "name": "qa-create-user-token-key",
                "source_type": "external",
                "url": f"{create_probe_target.base_url}/mcp",
                "headers": {},
                "env": {},
                "auth_mode": "user_token",
            },
            headers={"X-API-Key": API_KEY_ALICE},
        )

        # Assert — the API-key credential is INJECTED as X-API-Key
        assert resp.status_code == 201, resp.text
        assert resp.json()["tool_count"] == 1
        values = create_probe_target.recorder.api_key_values()
        assert values, "the create probe must have reached the real target"
        assert values == [API_KEY_ALICE] * len(values)

    async def test_headers_mode_create_probes_without_a_credential(
        self,
        http_client: httpx.AsyncClient,
        create_probe_target,
        registry_store,
        real_jwt_signer,
    ):
        # Arrange — the caller is authenticated but the server asks the
        # default headers mode: no injection may happen (pre-feature behavior)
        token = real_jwt_signer.issue_token(sub="user-alice")
        create_probe_target.recorder.reset()

        # Act
        resp = await http_client.post(
            SERVERS_PATH,
            json={
                "name": "qa-create-headers",
                "source_type": "external",
                "url": f"{create_probe_target.base_url}/mcp",
                "headers": {},
                "env": {},
                "auth_mode": "headers",
            },
            headers={"Authorization": f"Bearer {token}"},
        )

        # Assert — the create probe ran WITHOUT any injected credential
        assert resp.status_code == 201, resp.text
        assert resp.json()["tool_count"] == 1
        assert all(
            v is None for v in create_probe_target.recorder.authorization_values()
        ), "headers mode must never inject the caller's credential at create"
        assert all(v is None for v in create_probe_target.recorder.api_key_values())
