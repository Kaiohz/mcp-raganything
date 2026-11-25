"""Behavioral tests — dual-auth precedence + API-key flow pins (REQ-011/012).

feat-mcp-oauth2-dual-auth, mcp-raganything slice. Entry points: raw JSON-RPC
POSTs (HTTP 401 contract) and the REAL MCP client from the root conftest
(``mcp_list``/``mcp_call`` over the mounts) for regression pins on the
existing ``McpApiKeyMiddleware`` flows that must NOT regress when the
HTTP-level guard (REQ-017) becomes the primary gate.

Delegation contract pinned here:

* an INVALID Bearer JWT NEVER falls back to a valid `X-API-Key` present on the
  same request (HTTP 401 with the discovery challenge) — `Authorization` is
  JWT-only, API keys travel only in `X-API-Key`;
* a revoked / no-longer-active API key is rejected with 401 (the port returns
  no active key for its hash);
* the plain valid `X-API-Key` flow still resolves its OWNER (hash lookup →
  RLS-scoped upstream call) exactly as today (REQ-012 regression pin);
* the RLS contextvars are preserved for every accepted credential — the
  memory-store upstream receives the credentials of the authenticated user
  only.

RED phase: the HTTP-level 401 tests fail today (Mount 307 instead of the
guard's 401). The pins that ride the existing middleware path already pass —
they are regression locks for the migration in front of them.
"""

import pytest
from mcp.shared.exceptions import McpError

pytestmark = pytest.mark.asyncio(loop_scope="session")

#: The expected 401 discovery challenge / generic 401 body (REQ-010/REQ-001).
EXPECTED_WWW_AUTHENTICATE = (
    'Bearer resource_metadata="https://raganything.soludev.tech'
    '/.well-known/oauth-protected-resource"'
)
GENERIC_401_BODY = {"detail": "Invalid or missing credentials"}
#: Accepted credentials from the root ``auth_directory`` external doubles.
API_KEY_KEY_ALICE = "key-alice"


def assert_unauthorized_with_challenge(resp) -> None:
    assert resp.status_code == 401, (
        f"expected HTTP 401, got {resp.status_code} with body {resp.text!r}"
    )
    assert resp.headers.get("www-authenticate") == EXPECTED_WWW_AUTHENTICATE, (
        f"401 must carry the discovery challenge, got: "
        f"{resp.headers.get('www-authenticate')!r}"
    )
    assert resp.json() == GENERIC_401_BODY, (
        f"401 body must stay the generic, non-enumerating JSON, got: {resp.text!r}"
    )


class TestInvalidJwtNeverFallsBackToApiKey:
    async def test_invalid_bearer_with_valid_api_key_is_rejected_with_challenge(
        self, http_client, mcp_post
    ):
        """A request with BOTH an invalid `Authorization: Bearer` and a valid
        `X-API-Key` must be 401 WITH the challenge — never a silent fallback
        to the API key."""
        resp = await mcp_post(
            http_client,
            "/files/mcp",
            headers={
                "Authorization": "Bearer invalid-not-a-jwt",
                "X-API-Key": API_KEY_KEY_ALICE,
            },
        )
        assert_unauthorized_with_challenge(resp)

    @pytest.mark.parametrize("mount", ["/classical/mcp", "/web/mcp", "/memories/mcp"])
    async def test_no_fallback_contract_covers_every_mount(
        self, http_client, mount, mcp_post
    ):
        resp = await mcp_post(
            http_client,
            mount,
            headers={
                "Authorization": "Bearer invalid-not-a-jwt",
                "X-API-Key": API_KEY_KEY_ALICE,
            },
        )
        assert_unauthorized_with_challenge(resp)


class TestApiKeyFlowRegressionPins:
    async def test_unknown_api_key_is_rejected_with_challenge(
        self, http_client, mcp_post
    ):
        """An unknown key has no active hash match → 401 + challenge."""
        resp = await mcp_post(
            http_client,
            "/files/mcp",
            headers={"X-API-Key": "key_unknown_never_issued"},
        )
        assert_unauthorized_with_challenge(resp)

    async def test_no_longer_active_api_key_is_rejected(self, mcp_list):
        """A key whose hash no longer resolves as ACTIVE (revoked) is
        rejected. The `api_keys` table is owned by composable-agents (this
        repo only reads it), so the revoked state is exercised through the
        read-port seam the root conftest wires — the observable contract
        (key not resolvable as active → rejected) is the pin.
        """
        import main

        auth_service = main.mcp_middleware._auth_service
        assert auth_service is not None
        repo = auth_service._api_key_repo
        original_side_effect = repo.find_active_by_hash.side_effect
        repo.find_active_by_hash.side_effect = lambda _hash: None
        try:
            with pytest.raises(McpError, match="Invalid or missing credentials"):
                await mcp_list("/files/mcp", credential=API_KEY_KEY_ALICE)
        finally:
            repo.find_active_by_hash.side_effect = original_side_effect

    async def test_valid_api_key_flow_is_unchanged(self, mcp_call, memory_external):
        """REQ-012 regression pin: the plain `X-API-Key` flow still
        authenticates via hash lookup and scopes the upstream call to the
        key's OWNER (RLS context)."""
        result = await mcp_call(
            "/memories/mcp",
            "save_memory",
            {"name": "precedence", "content": "api key pin"},
            credential=API_KEY_KEY_ALICE,
        )

        assert result.is_error is False, result.content
        assert memory_external.stored(API_KEY_KEY_ALICE, "precedence") == "api key pin"
        assert memory_external.stored("jwt-alice", "precedence") is None
        assert all(
            request.headers.get("x-api-key") == API_KEY_KEY_ALICE
            and request.headers.get("authorization") is None
            for request in memory_external.requests
        )

    async def test_valid_jwt_takes_precedence_over_the_api_key(
        self, mcp_call, memory_external
    ):
        """Edge pin: both valid credentials → the JWT wins and ONLY its owner
        context reaches the upstream (no cross-user leak)."""
        result = await mcp_call(
            "/memories/mcp",
            "save_memory",
            {"name": "precedence", "content": "jwt wins"},
            headers={"Authorization": "Bearer jwt-alice", "X-API-Key": "key-bob"},
        )

        assert result.is_error is False, result.content
        assert memory_external.stored("jwt-alice", "precedence") == "jwt wins"
        assert memory_external.stored("key-bob", "precedence") is None
