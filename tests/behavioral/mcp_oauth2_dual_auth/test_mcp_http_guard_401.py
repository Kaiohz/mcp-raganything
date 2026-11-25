"""Behavioral tests — HTTP-level MCP 401 guard with discovery (REQ-010/001/017).

feat-mcp-oauth2-dual-auth, mcp-raganything slice. Entry point: the four
streamable-HTTP MCP mounts on the REAL app, probed with raw JSON-RPC POSTs so
the HTTP status + headers are the behavior under test.

Delegation contract pinned here (REQ-010 replicating REQ-001/REQ-002/REQ-017):

* the four mounts (``/files/mcp``, ``/classical/mcp``, ``/web/mcp``,
  ``/memories/mcp``) adopt the SAME HTTP-level guard pattern as
  composable-agents' ``McpAuthGuard``: every request missing valid credentials
  is answered HTTP 401 BEFORE any MCP method runs — including an
  unauthenticated ``initialize`` (never a successful initialize followed by a
  JSON-RPC error at tools/list);
* every 401 carries
  ``WWW-Authenticate: Bearer resource_metadata="https://<host>/.well-known/oauth-protected-resource"``
  (the host comes from the MCP_ALLOWED_HOSTS allow-list, never an unvalidated
  request Host — M-2 hardening mirrored from composable-agents);
* the 401 body stays the existing generic, non-enumerating JSON and every 401
  is uncacheable (``Cache-Control: no-store``).

RED phase: today the mounts answer an unauthenticated ``initialize`` with a
307 (no trailing slash) or an MCP-level 200 success (trailing slash) and the
``McpApiKeyMiddleware`` only guards ``tools/call``/``tools/list`` — so the 401
assertions fail with "got 307" / "got 200" / missing header. That IS the
missing feature.
"""

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")

#: The production proxy Host (mirrors the ingress Host header).
PROD_HOST = "raganything.soludev.tech"
MCP_MOUNTS = ("/files/mcp", "/classical/mcp", "/web/mcp", "/memories/mcp")
WELL_KNOWN_PATH = "/.well-known/oauth-protected-resource"
#: The expected 401 discovery challenge for the production Host (REQ-010/REQ-001).
EXPECTED_WWW_AUTHENTICATE = (
    f'Bearer resource_metadata="https://{PROD_HOST}{WELL_KNOWN_PATH}"'
)
#: The existing generic, non-enumerating 401 body.
GENERIC_401_BODY = {"detail": "Invalid or missing credentials"}
#: A Host that is NOT in the test env's MCP_ALLOWED_HOSTS.
FOREIGN_HOST = "dns-rebind.evil.example"


def assert_unauthorized_with_discovery_challenge(resp) -> None:
    """One assertion block for REQ-010/REQ-001: 401 + challenge + generic body."""
    assert resp.status_code == 401, (
        f"expected HTTP 401 BEFORE any MCP method runs, got {resp.status_code} "
        f"with body {resp.text!r}"
    )
    assert resp.headers.get("www-authenticate") == EXPECTED_WWW_AUTHENTICATE, (
        "401 must carry the MCP protected-resource discovery challenge "
        f'("WWW-Authenticate: {EXPECTED_WWW_AUTHENTICATE}"), got: '
        f"{resp.headers.get('www-authenticate')!r}"
    )
    assert resp.json() == GENERIC_401_BODY, (
        f"401 body must stay the generic, non-enumerating JSON, got: {resp.text!r}"
    )


class TestHttp401BeforeAnyMcpMethod:
    async def test_missing_credentials_initialize_gets_http_401_with_challenge(
        self, http_client, mcp_post
    ):
        """No credentials on the exact path → HTTP 401 + discovery challenge."""
        resp = await mcp_post(http_client, "/files/mcp")
        assert_unauthorized_with_discovery_challenge(resp)

    @pytest.mark.parametrize("mount", MCP_MOUNTS)
    async def test_every_mount_challenges_unauthenticated_initialize(
        self, http_client, mount, mcp_post
    ):
        """The unified guard pattern covers ALL FOUR mounts."""
        resp = await mcp_post(http_client, mount)
        assert_unauthorized_with_discovery_challenge(resp)

    @pytest.mark.parametrize("mount", MCP_MOUNTS)
    async def test_invalid_bearer_jwt_gets_http_401_with_challenge(
        self, http_client, mount, mcp_post
    ):
        """An INVALID bearer JWT → HTTP 401 + discovery challenge."""
        resp = await mcp_post(
            http_client, mount, headers={"Authorization": "Bearer not-a-real-token"}
        )
        assert_unauthorized_with_discovery_challenge(resp)

    @pytest.mark.parametrize("mount", MCP_MOUNTS)
    async def test_unknown_api_key_gets_http_401_with_challenge(
        self, http_client, mount, mcp_post
    ):
        """An UNKNOWN `X-API-Key` → HTTP 401 + discovery challenge."""
        resp = await mcp_post(
            http_client, mount, headers={"X-API-Key": "key_unknown_never_issued"}
        )
        assert_unauthorized_with_discovery_challenge(resp)

    async def test_tools_call_is_also_rejected_at_http_level(
        self, http_client, mcp_post
    ):
        """A raw tools/call probe without credentials hits the same HTTP 401
        (the guard runs BEFORE the method layer, not only on tools/list)."""
        resp = await mcp_post(
            http_client, "/files/mcp", headers={}, method="tools/call"
        )
        assert_unauthorized_with_discovery_challenge(resp)


class TestInitializeNeverSucceedsUnauthenticated:
    """REQ-017: unauthenticated `initialize` must NOT succeed — not even the
    historical 307→trailing-slash→MCP-success path the old method-level
    middleware let through."""

    @pytest.mark.parametrize("mount", MCP_MOUNTS)
    async def test_trailing_slash_initialize_unauthenticated_is_also_401(
        self, http_client, mount, mcp_post
    ):
        """Today the redirect-following path reaches the FastMCP app and
        returns an MCP-level initialize SUCCESS; the HTTP-level contract is a
        401 on BOTH path forms."""
        resp = await mcp_post(http_client, f"{mount}/")
        assert_unauthorized_with_discovery_challenge(resp)
        assert "jsonrpc" not in resp.text, (
            f"the 401 must not carry any MCP/JSON-RPC response payload, got: {resp.text!r}"
        )


class TestChallengeOriginHardening:
    """M-2 hardening mirrored from composable-agents: the discovery URL is
    never derived from a raw, spoofable request Host."""

    async def test_foreign_host_on_401_never_receives_a_discovery_url(
        self, http_client, mcp_post
    ):
        """A spoofed foreign Host stays 401 but the response advertises NO
        resource_metadata URL for that Host."""
        resp = await mcp_post(
            http_client,
            f"http://{FOREIGN_HOST}/files/mcp",
            headers={"X-API-Key": "key_unknown_never_issued"},
        )
        assert resp.status_code == 401, (
            f"a foreign-Host probe stays 401, got {resp.status_code}"
        )
        challenge = resp.headers.get("www-authenticate")
        assert challenge is None, (
            f"a non-allow-listed request Host must NEVER be echoed into the "
            f"discovery challenge, got: {challenge!r}"
        )
        assert resp.json() == GENERIC_401_BODY, (
            f"the 401 body stays the generic JSON regardless of the Host, got: {resp.text!r}"
        )

    async def test_allowed_host_challenge_is_derived_from_the_allowlist(
        self, http_client, mcp_post
    ):
        """The allow-listed production Host IS the fallback discovery origin."""
        resp = await mcp_post(http_client, f"http://{PROD_HOST}/files/mcp", headers={})
        assert resp.status_code == 401
        assert resp.headers.get("www-authenticate") == EXPECTED_WWW_AUTHENTICATE

    @pytest.mark.parametrize("mount", MCP_MOUNTS)
    async def test_401_challenge_response_is_never_cacheable(
        self, http_client, mount, mcp_post
    ):
        """Every 401 carries `Cache-Control: no-store` (challenge responses
        must not be cached and replayed to a different client)."""
        resp = await mcp_post(http_client, mount)
        assert resp.status_code == 401
        assert resp.headers.get("cache-control") == "no-store", (
            f"401 responses must not be cacheable, got "
            f"cache-control={resp.headers.get('cache-control')!r}"
        )
