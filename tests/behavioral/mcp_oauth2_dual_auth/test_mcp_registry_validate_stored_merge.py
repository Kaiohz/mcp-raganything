"""Behavioral tests — ``validate`` with ``id`` probes the STORED entry (BUG-001).

UI BUG-001 (composable-ui slice): the edit dialog's "Test connection" cannot
send the stored secrets (masked, REQ-016), so its stateless validate payload
carried EMPTY credential editors and a credential-REQUIRED target answered a
FALSE ``401 → 422 "MCP server unreachable"`` — the UI told the user a WORKING
server was broken (and the only way out was re-typing the hidden secret).

The fix: ``POST /api/v1/mcp/servers/validate`` accepts an optional ``id``
(the registered server's name); the stored (decrypted, RLS-scoped) entry
SUPPLIES the credential/config values and the payload fields override only
what the client explicitly changed — absent/masked-empty values (``{}``,
""``/``null``) never clear a stored secret, ``auth_mode`` keeps its current
validate semantics (explicit request value wins, stored otherwise).

What is pinned here (the FULL real chain: REAL app → REAL route → REAL
``GetMcpServerUseCase`` + ``ValidateMcpServerUseCase`` → REAL
``FastMcpToolLoader`` → a REAL streamable MCP server on a local TCP socket):

* registering a server WITH a valid key succeeds (create probes the target);
* the SAME server validated through the edit-dialog shape (``headers: {}``,
  no id) FAILS with a 401-driven 422 — the bug, reproduced;
* validating the same payload + ``id`` SUCCEEDS — the stored secret reached
  the target (observed on the socket, not mocked): the 401→200 proof;
* the UI's ``server_id`` wire name (BUG-005) runs the SAME stored merge —
  401 → 200 with the stored credential on the wire — while both-absent
  payloads keep the stateless candidate semantics;
* the response carries no stored secret (tool count only);
* a dirty override (url + auth_mode) is honored while the secrets still come
  from storage;
* masked-empty values alone never wipe the stored credentials;
* per-user isolation: another user's ``id`` (and an unknown ``id``) answers
  404 and never probes any target;
* the pre-fix stateless validate (no ``id``) still works unchanged.

The probe target is a REAL FastMCP streamable-HTTP server (one ``ping`` tool,
served by uvicorn on 127.0.0.1) behind an ENFORCING layer that answers a bare
401 to any request lacking the required credential — the loader then fails
exactly the way the bug report's target did.
"""

import socket
import threading
import time

import httpx
import pytest
from fastmcp import FastMCP
from starlette.types import ASGIApp, Receive, Scope, Send

from domain.entities.mcp_server_registry_entry import RegisteredMcpServer
from domain.errors.mcp import McpServerNotFoundError
from domain.ports.mcp_server_registry_store import McpServerRegistryStore
from infrastructure.database.rls_context import current_user_id

#: The credential the registry STORES for the server (never sent by the UI).
STORED_API_KEY = "cpk_stored_correct_key"

#: Accepted per-user API keys from the root ``auth_directory`` external
#: doubles (same users as the other behavioral suites).
API_KEY_ALICE = {"X-API-Key": "key-alice"}
API_KEY_BOB = {"X-API-Key": "key-bob"}

#: The target starts fast; failing after this long is a bug, not a wait.
_TARGET_START_TIMEOUT_S = 15.0

VALIDATE_PATH = "/api/v1/mcp/servers/validate"
SERVERS_PATH = "/api/v1/mcp/servers"


# ---------------------------------------------------------------------------
# Real probe target: enforcing FastMCP server on a REAL TCP socket
# ---------------------------------------------------------------------------


class _EnforcingProbe:
    """ASGI layer around the probe FastMCP app enforcing the stored credential.

    Requests WITHOUT the required ``X-API-Key`` value are answered with a bare
    401 BEFORE the MCP app runs (and are NOT recorded) — the MCP loader then
    fails the way the bug report's credential-REQUIRED target did. Accepted
    requests are recorded (lowercase header mapping) and served by the inner
    app.
    """

    def __init__(self, required_api_key: str) -> None:
        self._required_api_key = required_api_key
        self._inner: ASGIApp | None = None
        self._lock = threading.Lock()
        self.seen: list[dict[str, str]] = []

    def wrap(self, inner: ASGIApp) -> "_EnforcingProbe":
        self._inner = inner
        return self

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = {
                k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]
            }
            if headers.get("x-api-key") != self._required_api_key:
                await self._send_unauthorized(send)
                return
            with self._lock:
                self.seen.append(headers)
        assert self._inner is not None  # noqa: S101 — wrapped in the fixture
        await self._inner(scope, receive, send)

    async def _send_unauthorized(self, send: Send) -> None:
        body = b'{"detail":"Unauthorized"}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("latin-1")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    def reset(self) -> None:
        """Forget everything recorded so far (one observation window per test)."""
        with self._lock:
            self.seen.clear()

    def api_key_values(self) -> list[str | None]:
        """The X-API-Key header seen per ACCEPTED request (never on a 401)."""
        with self._lock:
            return [h.get("x-api-key") for h in self.seen]


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _ProbeHandle:
    """Handle for one running probe target (base URL + the enforcing layer)."""

    def __init__(self, base_url: str, probe: _EnforcingProbe) -> None:
        self.base_url = base_url
        self.probe = probe


@pytest.fixture
def probe_targets():
    """Serve REAL enforcing MCP probe targets on local TCP sockets.

    A factory: each ``start(required_api_key)`` spins one FastMCP server
    (one ``ping`` tool) behind :class:`_EnforcingProbe` in its own daemon
    thread (an independent event loop), so the app's loader connects over a
    REAL network socket — no ASGI plumbing between the loader and the target.
    All servers are stopped when the test ends.
    """
    import uvicorn

    running: list[tuple[uvicorn.Server, threading.Thread]] = []

    def start(required_api_key: str) -> _ProbeHandle:
        mcp = FastMCP("validate-stored-probe")

        @mcp.tool
        def ping() -> str:
            return "pong"

        probe = _EnforcingProbe(required_api_key=required_api_key)
        inner = probe.wrap(mcp.http_app(path="/mcp", stateless_http=True))
        port = _free_tcp_port()
        config = uvicorn.Config(
            inner, host="127.0.0.1", port=port, log_level="error", access_log=False
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, name="probe-target", daemon=True)
        thread.start()
        deadline = time.monotonic() + _TARGET_START_TIMEOUT_S
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not server.started:
            server.should_exit = True
            thread.join(timeout=5)
            raise RuntimeError("the enforcing MCP probe target did not start in time")
        running.append((server, thread))
        return _ProbeHandle(f"http://127.0.0.1:{port}", probe)

    try:
        yield start
    finally:
        for server, thread in running:
            server.should_exit = True
            thread.join(timeout=10)


# ---------------------------------------------------------------------------
# Registry store double: in-memory rows + the real adapter's RLS contract
# ---------------------------------------------------------------------------


class _RlsAwareRegistryStore(McpServerRegistryStore):
    """Outbound registration boundary double with per-user isolation.

    Mirrors the REAL ``McpRegistryStore`` contract: every row records the
    ``current_user_id`` RLS contextvar at save time, and reads are filtered by
    the CALLER's contextvar — another user's (or an unknown) name raises
    :class:`McpServerNotFoundError`, exactly like the asyncpg adapter with
    RLS engaged.
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
class TestValidateStoredEntry:
    """ "Test connection" against a stored server carries its credential."""

    async def test_edit_mode_probe_uses_the_stored_credential_401_to_200_proof(
        self,
        http_client: httpx.AsyncClient,
        probe_targets,
        registry_store,
        real_jwt_signer,
    ):
        # Arrange — the bug-report flow: a server registered with a VALID key,
        # its edit dialog sends the MASKED (empty) payload with no credential
        target = probe_targets(STORED_API_KEY)
        resp = await http_client.post(
            SERVERS_PATH,
            json={
                "name": "qa-validate-stored-apikey",
                "source_type": "external",
                "url": f"{target.base_url}/mcp",
                "headers": {"X-API-Key": STORED_API_KEY},
                "env": {},
                "auth_mode": "headers",
            },
            headers=API_KEY_ALICE,
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["tool_count"] == 1
        assert resp.json()["headers"] == {}  # the echo STAYS masked (REQ-018)
        target.probe.reset()  # drop the create session's records

        bug_body = {
            "name": "qa-validate-stored-apikey",
            "source_type": "external",
            "url": f"{target.base_url}/mcp",
            "headers": {},
            "env": {},
            "auth_mode": "headers",
        }

        # Act 1 — the BUG: the stateless probe (no id) runs UNAUTHENTICATED
        bug_resp = await http_client.post(
            VALIDATE_PATH, json=bug_body, headers=API_KEY_ALICE
        )

        # Assert 1 — a FALSE unreachable: the target answered 401
        assert bug_resp.status_code == 422, bug_resp.text
        assert "401" in bug_resp.json()["detail"]

        # Act 2 — the FIX: the same payload + id → stored credential travels
        fix_resp = await http_client.post(
            VALIDATE_PATH,
            json={**bug_body, "id": "qa-validate-stored-apikey"},
            headers=API_KEY_ALICE,
        )

        # Assert 2 — 200, the stored key reached the REAL target, no echo
        assert fix_resp.status_code == 200, fix_resp.text
        assert fix_resp.json()["tool_count"] == 1
        keys = target.probe.api_key_values()
        assert keys, "the probe must have reached the real target server"
        assert set(keys) == {STORED_API_KEY}
        assert set(fix_resp.json().keys()) == {"tool_count"}

    async def test_dirty_url_and_auth_mode_override_secrets_stay_stored(
        self,
        http_client: httpx.AsyncClient,
        probe_targets,
        registry_store,
        real_jwt_signer,
    ):
        # Arrange — stored url T1 requires the STORED key; the payload points
        # the probe at T2 (which requires the CALLER's key through user_token)
        stored_target = probe_targets(STORED_API_KEY)
        override_target = probe_targets("key-alice")
        resp = await http_client.post(
            SERVERS_PATH,
            json={
                "name": "qa-validate-override",
                "source_type": "external",
                "url": f"{stored_target.base_url}/mcp",
                "headers": {"X-API-Key": STORED_API_KEY},
                "env": {},
                "auth_mode": "headers",
            },
            headers=API_KEY_ALICE,
        )
        assert resp.status_code == 201, resp.text
        stored_target.probe.reset()  # drop the create session's records

        # Act — dirty override: new url + explicit user_token mode; the
        # payload carries NO credential field at all
        resp = await http_client.post(
            VALIDATE_PATH,
            json={
                "id": "qa-validate-override",
                "name": "qa-validate-override",
                "source_type": "external",
                "url": f"{override_target.base_url}/mcp",
                "auth_mode": "user_token",
            },
            headers=API_KEY_ALICE,
        )

        # Assert — the probe ran against the OVERRIDE url with the caller's
        # injected credential (API-key caller → X-API-Key), and the stored
        # target was never probed.
        assert resp.status_code == 200, resp.text
        assert resp.json()["tool_count"] == 1
        override_keys = override_target.probe.api_key_values()
        assert override_keys and set(override_keys) == {"key-alice"}
        assert stored_target.probe.api_key_values() == []

    async def test_masked_empty_credential_fields_never_clear_the_stored_ones(
        self,
        http_client: httpx.AsyncClient,
        probe_targets,
        registry_store,
        real_jwt_signer,
    ):
        # Arrange — the target requires the STORED key; the payload reproduces
        # the masked editors (empty headers/env) and a null/empty token
        target = probe_targets(STORED_API_KEY)
        resp = await http_client.post(
            SERVERS_PATH,
            json={
                "name": "qa-validate-masked",
                "source_type": "external",
                "url": f"{target.base_url}/mcp",
                "headers": {"X-API-Key": STORED_API_KEY},
                "env": {"BACKUP_KEY": "cpk_backup"},  # stored env, never sent
                "auth_mode": "headers",
            },
            headers=API_KEY_ALICE,
        )
        assert resp.status_code == 201, resp.text

        # Act — masked-empty credential fields must NOT wipe the stored values
        resp = await http_client.post(
            VALIDATE_PATH,
            json={
                "id": "qa-validate-masked",
                "url": f"{target.base_url}/mcp",
                "headers": {},
                "env": {},
                "auth_token": None,
                "auth_mode": "headers",
            },
            headers=API_KEY_ALICE,
        )

        # Assert — the stored key authenticated the probe (a wiped value
        # would 401 → 422)
        assert resp.status_code == 200, resp.text
        assert set(target.probe.api_key_values()) == {STORED_API_KEY}

    async def test_foreign_or_unknown_id_answers_404_without_probing(
        self,
        http_client: httpx.AsyncClient,
        probe_targets,
        registry_store,
        real_jwt_signer,
    ):
        # Arrange — ALICE owns the only registered server
        target = probe_targets(STORED_API_KEY)
        resp = await http_client.post(
            SERVERS_PATH,
            json={
                "name": "alice-server",
                "source_type": "external",
                "url": f"{target.base_url}/mcp",
                "headers": {"X-API-Key": STORED_API_KEY},
                "env": {},
                "auth_mode": "headers",
            },
            headers=API_KEY_ALICE,
        )
        assert resp.status_code == 201, resp.text

        # Act 1 — BOB validates ALICE's id (ransom/foreign reference)
        foreign = await http_client.post(
            VALIDATE_PATH,
            json={"id": "alice-server", "url": f"{target.base_url}/mcp"},
            headers=API_KEY_BOB,
        )
        target.probe.reset()

        # Act 2 — a made-up id
        unknown = await http_client.post(
            VALIDATE_PATH,
            json={"id": "no-such-server", "url": f"{target.base_url}/mcp"},
            headers=API_KEY_ALICE,
        )

        # Assert — both 404 (per-user RLS + unknown), and NO target probe
        assert foreign.status_code == 404, foreign.text
        assert unknown.status_code == 404, unknown.text
        assert target.probe.api_key_values() == []

    async def test_no_id_keeps_the_stateless_candidate_semantics(
        self,
        http_client: httpx.AsyncClient,
        probe_targets,
        registry_store,
        real_jwt_signer,
    ):
        # Arrange — the pre-fix payload WITH the credential in it must still
        # validate (the stateless dry-run of a candidate config is unchanged)
        target = probe_targets(STORED_API_KEY)

        # Act — no id, full credential in the payload
        resp = await http_client.post(
            VALIDATE_PATH,
            json={
                "name": "qa-validate-candidate",
                "source_type": "external",
                "url": f"{target.base_url}/mcp",
                "headers": {"X-API-Key": STORED_API_KEY},
                "env": {},
                "auth_mode": "headers",
            },
            headers=API_KEY_ALICE,
        )

        # Assert
        assert resp.status_code == 200, resp.text
        assert resp.json()["tool_count"] == 1
        assert set(target.probe.api_key_values()) == {STORED_API_KEY}

    async def test_server_id_wire_alias_and_both_absent_semantics(
        self,
        http_client: httpx.AsyncClient,
        probe_targets,
        registry_store,
        real_jwt_signer,
    ):
        """UI BUG-005: the edit dialog sends ``server_id`` — the alias must
        run the SAME stored merge as ``id`` (401 → 200, the stored credential
        observed on the wire), while both-absent payloads keep the stateless
        candidate semantics (the false 401 unchanged)."""
        # Arrange — a server registered with a VALID stored credential; the
        # edit dialog sends the masked payload referencing the entry under
        # the UI's field name
        target = probe_targets(STORED_API_KEY)
        resp = await http_client.post(
            SERVERS_PATH,
            json={
                "name": "qa-validate-server-id",
                "source_type": "external",
                "url": f"{target.base_url}/mcp",
                "headers": {"X-API-Key": STORED_API_KEY},
                "env": {},
                "auth_mode": "headers",
            },
            headers=API_KEY_ALICE,
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["tool_count"] == 1
        target.probe.reset()  # drop the create session's records

        masked_body = {
            "name": "qa-validate-server-id",
            "source_type": "external",
            "url": f"{target.base_url}/mcp",
            "headers": {},
            "env": {},
            "auth_mode": "headers",
        }

        # Act 1 — BOTH names absent: the stateless probe runs UNAUTHENTICATED
        # (the unchanged semantics — the false 401 must stay observable)
        stateless = await http_client.post(
            VALIDATE_PATH, json=masked_body, headers=API_KEY_ALICE
        )
        assert stateless.status_code == 422, stateless.text
        assert "401" in stateless.json()["detail"]

        # Act 2 — the UI's field name: the stored credential reaches the target
        by_alias = await http_client.post(
            VALIDATE_PATH,
            json={**masked_body, "server_id": "qa-validate-server-id"},
            headers=API_KEY_ALICE,
        )

        # Assert 2 — 200 with the REAL tool count, the stored key on the wire
        assert by_alias.status_code == 200, by_alias.text
        assert by_alias.json()["tool_count"] >= 1
        assert by_alias.json() == {"tool_count": 1}
        alias_keys = target.probe.api_key_values()
        assert alias_keys, "the alias probe must have reached the real target"
        assert set(alias_keys) == {STORED_API_KEY}
        target.probe.reset()

        # Act 3 — ``id`` keeps working (the primary name is unchanged)
        by_id = await http_client.post(
            VALIDATE_PATH,
            json={**masked_body, "id": "qa-validate-server-id"},
            headers=API_KEY_ALICE,
        )

        # Assert 3
        assert by_id.status_code == 200, by_id.text
        assert by_id.json()["tool_count"] >= 1
        assert set(target.probe.api_key_values()) == {STORED_API_KEY}
