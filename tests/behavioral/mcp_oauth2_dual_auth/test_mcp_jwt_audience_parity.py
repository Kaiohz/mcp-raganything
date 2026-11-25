"""Behavioral tests — JWT audience validation parity (REQ-003-equivalent).

feat-mcp-oauth2-dual-auth, mcp-raganything slice. Entry points: the REAL MCP
client over ``/files/mcp`` (tools/list is the credential gate observable from
the outside).

Why THIS suite includes an audience pin: raganything validates JWTs ITSELF —
its ``JwtAdapter`` (``src/infrastructure/security/jwt_adapter.py``) passes the
single ``JWT_AUDIENCE`` setting to ``jwt.decode`` and rejects a token whose
``aud`` does not match. The feature therefore adds the same multi-audience
support (``JWT_AUDIENCES``) as the composable-agents slice; the parity
contract is pinned behaviorally:

* ``JWT_AUDIENCES`` unset (the test env never sets it) → the SINGLE
  ``JWT_AUDIENCE`` behavior stays byte-for-byte: a token issued for the
  configured audience is accepted, a token for any other audience is
  rejected — through the mounts, with REAL verification (real RS256 keys,
  real ``jwt.decode``; only the outbound Logto JWKS fetch is doubled, see the
  feature conftest).

These tests are also the guard for the RED→GREEN migration: once the HTTP
guard (REQ-017) fronts the mounts, audience handling must stay identical.
"""

import pytest
from mcp.shared.exceptions import McpError

pytestmark = pytest.mark.asyncio(loop_scope="session")

MOUNT = "/files/mcp"


class TestSingleAudienceParity:
    async def test_token_for_the_configured_audience_authenticates(
        self, mcp_list, real_jwt_signer
    ):
        """A real RS256 token whose `aud` matches the configured single
        `JWT_AUDIENCE` passes the mounts' authentication (real verification
        chain)."""
        token = real_jwt_signer.issue_token(sub="user-audience-probe")
        tools = await mcp_list(MOUNT, headers={"Authorization": f"Bearer {token}"})

        assert tools, "the token issued for the configured audience must authenticate"
        assert any(tool.name == "list_files" for tool in tools)

    async def test_token_for_a_foreign_audience_is_rejected(
        self, mcp_list, real_jwt_signer
    ):
        """A validly-SIGNED token for an audience NOT in the configured list
        is rejected: single-`JWT_AUDIENCE` behavior with `JWT_AUDIENCES`
        unset (parity contract)."""
        token = real_jwt_signer.issue_token(
            sub="user-audience-probe", aud="some-other-opencode-app-id"
        )
        with pytest.raises(McpError, match="Invalid or missing credentials"):
            await mcp_list(MOUNT, headers={"Authorization": f"Bearer {token}"})
