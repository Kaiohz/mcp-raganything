"""Behavioral tests — ``validate`` honors ``auth_mode`` (REQ-004, RE-MAJOR-1).

The re-review MAJOR-1: ``POST /validate`` accepted ``auth_mode`` and then
DISCARDED it — "Test connection" probed a ``user_token`` target with no
credential, so an authenticated target answered 401 and the UI reported a
false validation failure for a server that would work at runtime (the
consumer injects the credential at load time).

What is pinned here (the FULL real chain: REAL app → REAL route → REAL
``ValidateMcpServerUseCase`` → REAL ``FastMcpToolLoader`` → a REAL streamable
MCP server open on a local TCP socket, with a header-recording ASGI wrapper in
front of it):

* a ``user_token`` validation requested by a JWT-authenticated caller probes
  the target with ``Authorization: Bearer <caller JWT>`` (the injected
  credential REACHES the target — observed on the socket, not mocked);
* the same for an API-key caller: ``X-API-Key: <caller key>``;
* a ``headers``-mode validation probes the target with NO injected
  credential — the pre-feature behavior, unchanged.

The probe target is a real FastMCP streamable-HTTP server (one ``ping``
tool) served by uvicorn on ``127.0.0.1`` in a background thread — it accepts
any caller (it enforces no auth of its own) and only RECORDS the headers it
receives; the assertions are about what the registry transmitted.
"""

import socket
import threading
import time
from collections.abc import Iterator

import httpx
import pytest
from fastmcp import FastMCP
from starlette.types import ASGIApp, Receive, Scope, Send

#: Accepted per-user API key from the root ``auth_directory`` external doubles
#: (same users as the other behavioral suites).
API_KEY_KEY_ALICE = "key-alice"

SERVER_NAME = "validate-user-token-probe"

#: The target starts fast; failing after this long is a bug, not a wait.
_TARGET_START_TIMEOUT_S = 15.0


class _HeaderRecordingProbeTarget:
    """ASGI wrapper around the probe FastMCP app that records inbound headers.

    Records the lowercase header mapping of every HTTP scope so the tests can
    observe EXACTLY what the registry transmitted to the target (the socket
    boundary — not a mocked loader).
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
        await self._inner(scope, receive, send)

    def reset(self) -> None:
        """Forget everything recorded so far (one observation window per test)."""
        with self._lock:
            self.seen.clear()

    def authorization_values(self) -> list[str | None]:
        """The Authorization header seen per request (``None`` = absent).

        ASGI header names are lowercase bytes, so the lookup is lowercase.
        """
        with self._lock:
            return [h.get("authorization") for h in self.seen]

    def api_key_values(self) -> list[str | None]:
        """The X-API-Key header seen per request (``None`` = absent)."""
        with self._lock:
            return [h.get("x-api-key") for h in self.seen]


class _ProbeTarget:
    """Handle for the running probe target (base URL + the recorder)."""

    def __init__(self, base_url: str, recorder: _HeaderRecordingProbeTarget) -> None:
        self.base_url = base_url
        self.recorder = recorder


def _free_tcp_port() -> int:
    """Reserve a free localhost TCP port for the target server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_probe_target() -> _ProbeTarget:
    """Build the FastMCP probe app wrapped with the header recorder."""
    mcp = FastMCP("validate-probe-target")

    @mcp.tool
    def ping() -> str:
        return "pong"

    recorder = _HeaderRecordingProbeTarget(
        mcp.http_app(path="/mcp", stateless_http=True)
    )
    return _ProbeTarget(f"http://127.0.0.1:{_free_tcp_port()}", recorder)


@pytest.fixture
def mcp_probe_target() -> Iterator[_ProbeTarget]:
    """Serve the REAL probe MCP server with uvicorn on a local TCP socket.

    Runs in its own daemon thread (an independent event loop) so the app's
    loader connects over a REAL network socket — no ASGI plumbing between the
    loader and the target. The recorder keeps thread-safe state; tests read it
    after the HTTP round-trip completed.
    """
    import uvicorn

    target = _build_probe_target()
    config = uvicorn.Config(
        target.recorder,
        host="127.0.0.1",
        port=int(target.base_url.rsplit(":", 1)[1]),
        log_level="error",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="mcp-probe-target", daemon=True)
    thread.start()
    deadline = time.monotonic() + _TARGET_START_TIMEOUT_S
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("the MCP probe target server did not start in time")

    try:
        yield target
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.mark.asyncio(loop_scope="session")
class TestValidateUserTokenInjection:
    """ "Test connection" probes a user_token target WITH the caller credential."""

    async def test_user_token_probe_carries_the_callers_jwt(
        self,
        http_client: httpx.AsyncClient,
        mcp_probe_target: _ProbeTarget,
        real_jwt_signer,
    ):
        # Arrange — the UI flow: caller is JWT-authenticated, the dialog sends
        # auth_mode "user_token" with EMPTY secret editors (buildInput shape)
        token = real_jwt_signer.issue_token(sub="user-alice")
        mcp_probe_target.recorder.reset()

        # Act — the real chain: route → use case → loader → REAL target socket
        resp = await http_client.post(
            "/api/v1/mcp/servers/validate",
            json={
                "name": SERVER_NAME,
                "source_type": "external",
                "url": f"{mcp_probe_target.base_url}/mcp",
                "auth_mode": "user_token",
                "headers": {},
            },
            headers={"Authorization": f"Bearer {token}"},
        )

        # Assert — validation SUCCEEDS and the credential reached the target
        assert resp.status_code == 200, resp.text
        assert resp.json()["tool_count"] == 1
        values = mcp_probe_target.recorder.authorization_values()
        assert values, "the probe must have reached the real target server"
        assert values == [f"Bearer {token}"] * len(values), (
            "every MCP request of the probe session must carry the INJECTED "
            "caller credential (Authorization wins over colliding names)"
        )

    async def test_user_token_probe_carries_the_callers_api_key(
        self,
        http_client: httpx.AsyncClient,
        mcp_probe_target: _ProbeTarget,
        real_jwt_signer,
    ):
        # Arrange — a caller authenticated with its per-user API key
        mcp_probe_target.recorder.reset()

        # Act
        resp = await http_client.post(
            "/api/v1/mcp/servers/validate",
            json={
                "name": SERVER_NAME,
                "source_type": "external",
                "url": f"{mcp_probe_target.base_url}/mcp",
                "auth_mode": "user_token",
                "headers": {},
            },
            headers={"X-API-Key": API_KEY_KEY_ALICE},
        )

        # Assert — the API-key credential is INJECTED as X-API-Key
        assert resp.status_code == 200, resp.text
        assert resp.json()["tool_count"] == 1
        values = mcp_probe_target.recorder.api_key_values()
        assert values, "the probe must have reached the real target server"
        assert values == [API_KEY_KEY_ALICE] * len(values)

    async def test_headers_mode_probes_the_target_without_a_credential(
        self,
        http_client: httpx.AsyncClient,
        mcp_probe_target: _ProbeTarget,
        real_jwt_signer,
    ):
        # Arrange — the caller is authenticated but the server asks the
        # default headers mode: no injection may happen (pre-feature behavior)
        token = real_jwt_signer.issue_token(sub="user-alice")
        mcp_probe_target.recorder.reset()

        # Act
        resp = await http_client.post(
            "/api/v1/mcp/servers/validate",
            json={
                "name": SERVER_NAME,
                "source_type": "external",
                "url": f"{mcp_probe_target.base_url}/mcp",
            },
            headers={"Authorization": f"Bearer {token}"},
        )

        # Assert — the probe runs WITHOUT any injected credential
        assert resp.status_code == 200, resp.text
        assert resp.json()["tool_count"] == 1
        assert all(
            v is None for v in mcp_probe_target.recorder.authorization_values()
        ), "headers mode never injects the caller's credential"
