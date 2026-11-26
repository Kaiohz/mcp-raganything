"""Behavioral tests — exact-path MCP mounts without 307/scheme downgrade (REQ-015).

feat-mcp-oauth2-dual-auth, mcp-raganything slice. Entry point: the REAL app,
raw POST probes with ``follow_redirects=False`` so a redirect is OBSERVED.

Delegation contract pinned here:

* POST to each of the four mounts WITHOUT a trailing slash is SERVED directly
  (exact-path routes with path normalization, the composable-agents
  ``Route("/mcp")`` pattern) — never a Starlette Mount 307;
* any generated redirect/URL behind the proxy honours ``X-Forwarded-Proto``:
  no ``http://`` scheme downgrade is ever observable (today's production
  breakage: ``Location: http://raganything.soludev.tech/memories/mcp/``).

RED phase: today POST ``<mount>`` (no trailing slash) is a Mount 307 with an
``http://`` redirect target that ignores ``X-Forwarded-Proto`` — the exact
production bug.
"""

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")

REDIRECT_STATUSES = {301, 302, 307, 308}
MCP_MOUNTS = ("/files/mcp", "/classical/mcp", "/web/mcp", "/memories/mcp")
#: Accepted credentials from the root ``auth_directory`` external doubles.
API_KEY_KEY_ALICE = "key-alice"


class TestPostWithoutTrailingSlashIsServed:
    @pytest.mark.parametrize("mount", MCP_MOUNTS)
    async def test_post_no_trailing_slash_is_never_a_redirect(
        self, http_client, mount, mcp_post
    ):
        """POST `<mount>` with a valid API key is served directly (200), never
        a 307 redirect round-trip that loses MCP clients behind the proxy."""
        resp = await mcp_post(
            http_client,
            mount,
            headers={"X-API-Key": API_KEY_KEY_ALICE, "X-Forwarded-Proto": "https"},
        )
        assert resp.status_code not in REDIRECT_STATUSES, (
            f"{mount} must be served WITHOUT a trailing-slash redirect — got "
            f"{resp.status_code} Location={resp.headers.get('location')!r}"
        )
        assert resp.status_code == 200, (
            f"the authenticated POST to {mount} (no trailing slash) must reach "
            f"the MCP app, got {resp.status_code}: {resp.text!r}"
        )


class TestNoSchemeDowngradeBehindProxy:
    @pytest.mark.parametrize("mount", MCP_MOUNTS)
    async def test_proxy_presented_https_never_generates_an_http_location(
        self, http_client, mount, mcp_post
    ):
        """With ``X-Forwarded-Proto: https`` present, ANY generated redirect /
        URL must keep the https scheme (uvicorn proxy-header handling or
        explicit scheme handling). Primary contract: no redirect at all."""
        resp = await mcp_post(
            http_client,
            mount,
            headers={"X-API-Key": API_KEY_KEY_ALICE, "X-Forwarded-Proto": "https"},
        )
        location = resp.headers.get("location")
        assert resp.status_code not in REDIRECT_STATUSES, (
            f"{mount} must be served directly — got {resp.status_code} "
            f"Location={location!r}"
        )
        # Defensive companion: if a redirect ever exists it must never
        # downgrade to http:// (today's live production breakage).
        if location is not None:
            assert location.startswith("https://"), (
                f"a redirect target behind an https-forwarding proxy must keep "
                f"the https scheme, got: {location!r}"
            )
