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

Observation seams: the memory upstream receives the credential of the
authenticated principal (``memory_external``, the suite's RLS seam), and the
API-key repository / JWT adapter are counted through the existing
``auth_directory`` doubles.
"""

import asyncio
import json

import pytest

from domain.services.auth.api_key_hasher import ApiKeyHasher

pytestmark = pytest.mark.asyncio(loop_scope="session")

MOUNT = "/memories/mcp"


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
        self, mcp_call, memory_external
    ):
        """Two users alternate tool calls on the SAME mount: each call must
        store under its OWN credential identity — never under the other
        user's (per-request RLS principal, stateless transport)."""
        results = []
        for credential, name, content in (
            (API_KEY_KEY_ALICE, "alpha-1", "alice-1"),
            (API_KEY_KEY_BOB, "beta-1", "bob-1"),
            (API_KEY_KEY_ALICE, "alpha-2", "alice-2"),
            (API_KEY_KEY_BOB, "beta-2", "bob-2"),
        ):
            result = await mcp_call(
                MOUNT,
                "save_memory",
                {"name": name, "content": content},
                credential=credential,
            )
            results.append(result)

        assert all(result.is_error is False for result in results), results
        assert memory_external.stored(API_KEY_KEY_ALICE, "alpha-1") == "alice-1"
        assert memory_external.stored(API_KEY_KEY_ALICE, "alpha-2") == "alice-2"
        assert memory_external.stored(API_KEY_KEY_BOB, "beta-1") == "bob-1"
        assert memory_external.stored(API_KEY_KEY_BOB, "beta-2") == "bob-2"
        # No cross-user leak: each name is only under its caller's identity.
        assert memory_external.stored(API_KEY_KEY_BOB, "alpha-1") is None
        assert memory_external.stored(API_KEY_KEY_BOB, "alpha-2") is None
        assert memory_external.stored(API_KEY_KEY_ALICE, "beta-1") is None
        assert memory_external.stored(API_KEY_KEY_ALICE, "beta-2") is None

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
        self, http_client, mcp_call, memory_external
    ):
        """BOB's tools/call carrying ALICE's stale session id must execute
        under BOB's RLS principal: the memory upstream receives BOB's
        credential (never alice's) even though the request reuses alice's
        session id (would be inherited stale-principal territory in stateful
        mode — the stateless transport keeps the per-request principal)."""
        # Seed alice's own namespace so the assertion observes a REWRITE to
        # bob's identity rather than an absolute creation.
        memory_external.seed(API_KEY_KEY_ALICE, "session-reuse", "alice-original")

        result = await mcp_call(
            MOUNT,
            "save_memory",
            {"name": "session-reuse", "content": "bob-rewrite"},
            headers={
                "X-API-Key": API_KEY_KEY_BOB,
                "Mcp-Session-Id": STALE_SESSION_ID,
            },
        )

        assert result.is_error is False, result.content
        assert memory_external.stored(API_KEY_KEY_BOB, "session-reuse") == "bob-rewrite"
        assert memory_external.stored(API_KEY_KEY_ALICE, "session-reuse") == (
            "alice-original"
        )


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
