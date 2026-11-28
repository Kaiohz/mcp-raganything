"""Behavioral tests — per-request RLS principals + single authentication.

feat-mcp-oauth2-dual-auth, mcp-raganything slice, review iteration 2.

Delegation contract pinned here:

* (code-review C-1) the RLS contextvars are set from the ACCEPTED request's
  principal — the MCP tool task inherits its OWN request's principal, not a
  session creator's stale one. The transport is built STATELESS
  (``FastMCP.http_app(stateless_http=True)``), so every HTTP request — and
  the tool task it dispatches — runs under the task context of the CURRENT
  request (in the fastmcp-default stateful mode the tool task inherits the
  SESSION CREATOR's context: an actual cross-user hazard when the
  method-level middleware is not re-deriving the principal);
* (code-review M-2) one verification per HTTP request: the ASGI guard
  authenticates once; the defense-in-depth ``McpApiKeyMiddleware`` reuses
  its cached :class:`AuthContext` instead of re-deriving it — exactly ONE
  ``api_keys`` lookup / ONE JWT verification per authenticated MCP HTTP
  request (before the fix every tool call authenticated TWICE).

Observation seam: the FILES mounts' temporary upload writes through the
outbound storage adapter (``file_storage_double``, the suite's RLS seam) —
the accepted request's principal composes the object path
(``{user_id}/{agent_slug}/...``), so each call's recorded ``put_object``
reveals which RLS context the tool task actually ran under. The API-key
repository / JWT adapter are counted through the existing ``auth_directory``
doubles.
"""

import asyncio
import base64
import json

import pytest

from domain.services.auth.api_key_hasher import ApiKeyHasher

pytestmark = pytest.mark.asyncio(loop_scope="session")

MOUNT = "/files/mcp"

_UPLOAD_PREFIX_ALICE = "chat_tmp/user-alice/session/"
_UPLOAD_PREFIX_BOB = "chat_tmp/user-bob/session/"
#: User × agent directive (chat_tmp Option 2): the temporary tools REQUIRE
#: the agent scope — the RLS-principal probes must carry it.
_UPLOAD_HEADERS = {"X-Composable-Agent": "agent-x"}


def _txt_payload(text: str) -> dict:
    """A valid ``temporary_upload_file`` argument set (text/plain probe)."""
    return {
        "file_name": "probe.txt",
        "content_base64": base64.b64encode(text.encode()).decode(),
        "content_type": "text/plain",
    }


def _parse_jsonrpc_body(text: str) -> dict:
    """Parse a JSON-RPC response written as raw JSON or as an SSE frame.

    The streamable-HTTP transport answers a tools/list POST with the media
    type it negotiated (``application/json`` or an ``event: message`` /
    ``data: {...}`` frame) — the assertion is about the JSON-RPC payload.
    """
    if text.lstrip().startswith("{"):
        return json.loads(text)
    for line in text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[len("data:") :].strip())
    raise AssertionError(f"no JSON-RPC payload in the MCP response: {text[:200]!r}")


#: Accepted credentials from the root ``auth_directory`` external doubles.
API_KEY_KEY_ALICE = "key-alice"
API_KEY_KEY_BOB = "key-bob"

#: A session id as a stale STATEFUL session would have presented one: the
#: value is never a live session of this process (stateless transports don't
#: assign one) — the pin is that it CANNOT influence the RLS principal.
STALE_SESSION_ID = "stale-session-id-created-by-alice"


class _RecordingStorage:
    """Minimal outbound storage double recording every ``put_object``."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.put_calls: list[dict] = []

    async def put_object(
        self, bucket: str, object_path: str, data: bytes, content_type: str
    ) -> None:
        self.put_calls.append({"object_path": object_path, "data": data})
        self.objects[object_path] = data


@pytest.fixture
def file_storage_double(monkeypatch):
    """Swap the outbound storage adapter singletons for the recording double.

    Both bindings are patched: the DI module's singleton (used by the file
    use case provider) and the module-level binding captured by the MCP
    file tools module.
    """
    double = _RecordingStorage()
    import application.api.mcp_file_tools as mcp_file_tools
    import dependencies

    monkeypatch.setattr(dependencies, "minio_adapter", double)
    monkeypatch.setattr(mcp_file_tools, "minio_adapter", double)
    return double


class _LookupCounter:
    """Count the ``api_keys`` repository lookups (one per HTTP request)."""

    def __init__(self, auth_directory) -> None:
        self.repo = auth_directory.api_key_repository
        self.seen_hashes: list[str] = []
        self._original = self.repo.find_active_by_hash.side_effect

    async def _counting(self, key_hash: str):
        self.seen_hashes.append(key_hash)
        result = self._original(key_hash)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    def __enter__(self) -> "_LookupCounter":
        self.repo.find_active_by_hash.side_effect = self._counting
        return self

    def __exit__(self, *_exc) -> None:
        self.repo.find_active_by_hash.side_effect = self._original


class TestRlsPrincipalIsTheAcceptedRequestPrincipal:
    """REQ-011/REQ-017: the RLS context comes from the ACCEPTED request."""

    async def test_alternating_users_each_tool_call_runs_under_its_own_principal(
        self, mcp_call, file_storage_double
    ):
        """Two users alternate temporary uploads on the SAME mount: each call
        must write under its OWN scope root (``{user_id}/...``) — never under
        the other user's (per-request RLS principal, stateless transport)."""
        results = []
        for credential, prefix, tag in (
            (API_KEY_KEY_ALICE, _UPLOAD_PREFIX_ALICE, "alice-1"),
            (API_KEY_KEY_BOB, _UPLOAD_PREFIX_BOB, "bob-1"),
            (API_KEY_KEY_ALICE, _UPLOAD_PREFIX_ALICE, "alice-2"),
            (API_KEY_KEY_BOB, _UPLOAD_PREFIX_BOB, "bob-2"),
        ):
            result = await mcp_call(
                MOUNT,
                "temporary_upload_file",
                {**_txt_payload(tag), "prefix": prefix},
                credential=credential,
                headers=_UPLOAD_HEADERS,
            )
            results.append(result)

        assert all(result.is_error is False for result in results), results
        paths = [call["object_path"] for call in file_storage_double.put_calls]
        assert len(paths) == 4, paths
        assert paths[0].startswith("user-alice/"), paths[0]
        assert paths[1].startswith("user-bob/"), paths[1]
        assert paths[2].startswith("user-alice/"), paths[2]
        assert paths[3].startswith("user-bob/"), paths[3]

    async def test_a_second_users_tools_list_runs_under_the_request_owner_despite_a_stale_session(
        self, http_client, auth_directory
    ):
        """BOB's tools/list carrying ALICE's stale session id must be
        authenticated with BOB's credential (its OWN principal): the API-key
        lookup derives from BOB's ``X-API-Key`` — never alice's hash."""
        with _LookupCounter(auth_directory) as counter:
            response = await http_client.post(
                MOUNT,
                headers={
                    "Accept": "application/json, text/event-stream",
                    "X-API-Key": API_KEY_KEY_BOB,
                    "Mcp-Session-Id": STALE_SESSION_ID,
                },
                json={"jsonrpc": "2.0", "id": 0, "method": "tools/list"},
            )

        assert response.status_code == 200, response.text
        payload = _parse_jsonrpc_body(response.text)
        assert payload.get("error") is None, payload
        assert payload.get("result", {}).get("tools"), payload
        # The request ran under BOB's principal: his key hash was looked up…
        assert counter.seen_hashes == [ApiKeyHasher.hash_key(API_KEY_KEY_BOB)]

    async def test_a_second_users_tool_call_runs_under_its_own_principal_despite_a_stale_session(
        self, mcp_call, file_storage_double
    ):
        """BOB's tools/call carrying ALICE's stale session id must execute
        under BOB's RLS principal: the upload lands under BOB's scope root
        (never alice's) even though the request reuses alice's session id
        (would be inherited stale-principal territory in stateful mode — the
        stateless transport keeps the per-request principal)."""
        result = await mcp_call(
            MOUNT,
            "temporary_upload_file",
            {**_txt_payload("bob-rewrite"), "prefix": _UPLOAD_PREFIX_BOB},
            headers={
                "X-API-Key": API_KEY_KEY_BOB,
                "Mcp-Session-Id": STALE_SESSION_ID,
                **_UPLOAD_HEADERS,
            },
        )

        assert result.is_error is False, result.content
        assert len(file_storage_double.put_calls) == 1
        assert file_storage_double.put_calls[0]["object_path"].startswith(
            "user-bob/"
        ), file_storage_double.put_calls[0]["object_path"]


class TestOneAuthenticationPerHttpRequest:
    """code-review M-2: the guard authenticates ONCE; the method-level
    middleware SHORT-CIRCUITS on the guard's cached principal (one
    ``api_keys`` lookup / one JWT verification per HTTP request)."""

    #: One raw JSON-RPC POST = exactly ONE HTTP request through the guard.
    _UNKNOWN_TOOL_BODY = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "tools/call",
        "params": {
            "name": "unknown-tool-for-auth-count-probe",
            "arguments": {},
        },
    }

    async def test_api_key_tool_call_authenticates_exactly_once(
        self, http_client, auth_directory
    ):
        with _LookupCounter(auth_directory) as counter:
            response = await http_client.post(
                MOUNT,
                headers={
                    "Accept": "application/json, text/event-stream",
                    "X-API-Key": API_KEY_KEY_ALICE,
                },
                json=self._UNKNOWN_TOOL_BODY,
            )

        assert response.status_code == 200, response.text
        assert counter.seen_hashes == [ApiKeyHasher.hash_key(API_KEY_KEY_ALICE)], (
            f"exactly ONE api_keys lookup is expected per HTTP request "
            f"(guard = 1, defense-in-depth middleware = 0), got "
            f"{len(counter.seen_hashes)}"
        )

    async def test_jwt_tool_call_verifies_exactly_once(
        self, http_client, real_jwt_signer
    ):
        """With a JWT the RSA verification runs ONCE per HTTP request (the
        JWKS document stays cache-reused; the double verification the review
        measured is gone)."""
        import infrastructure.security.jwt_adapter as jwt_adapter_module

        counter = {"verifications": 0}
        original = jwt_adapter_module.JwtAdapter.decode_token

        async def counting(adapter, token: str):
            counter["verifications"] += 1
            return await original(adapter, token)

        token = real_jwt_signer.issue_token(sub="user-single-auth-jwt")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(jwt_adapter_module.JwtAdapter, "decode_token", counting)
            response = await http_client.post(
                "/files/mcp",
                headers={
                    "Accept": "application/json, text/event-stream",
                    "Authorization": f"Bearer {token}",
                },
                json=self._UNKNOWN_TOOL_BODY,
            )

        assert response.status_code == 200, response.text
        assert counter["verifications"] == 1, (
            "exactly ONE JWT verification is expected per HTTP request "
            f"(guard = 1, defense-in-depth middleware = 0), got "
            f"{counter['verifications']}"
        )
