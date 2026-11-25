"""Behavioral tests — public `/.well-known/oauth-protected-resource` (REQ-002).

feat-mcp-oauth2-dual-auth, mcp-raganything slice (REQ-010 replication).
Entry point: the REAL FastAPI app (httpx AsyncClient over ASGITransport), i.e.
the production router chain including the CORS middleware.

Delegation contract pinned here:

* ``GET /.well-known/oauth-protected-resource`` is PUBLIC (no credentials, no
  auth dependency at all) and returns a document whose
  ``authorization_servers`` is the Logto OIDC issuer
  (``https://logto.soludev.tech/oidc``) and whose ``resource`` is an ``https://``
  protected-MCP-endpoint URL (the scheme NEVER downgrades behind the proxy —
  same https-derivation rule as the composable-agents slice; the single-doc vs
  per-mount resource choice is the implementation's);
* ``OPTIONS`` preflights on the well-known path are answered by the CORS
  middleware WITHOUT any auth check;
* ``Host: <allowed-host>:443`` (QA BUG-002) is treated like the port-less
  production Host form;
* a spoofed foreign Host can never shape the document (M-2 hardening): the
  metadata is simply not served with an attacker-shaped resource URL.

RED phase: the route does not exist yet — expected to 404 until the feature
lands (the preflight pin already passes: CORS answers preflights app-wide).
"""

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")

#: The production proxy Host and discovery-chain constants (REQ-010
#: replicating REQ-002; the Logto OIDC issuer is pinned by the feature spec).
PROD_HOST = "raganything.soludev.tech"
LOGTO_ISSUER = "https://logto.soludev.tech/oidc"
WELL_KNOWN_PATH = "/.well-known/oauth-protected-resource"
#: A Host that is NOT in the test env's MCP_ALLOWED_HOSTS.
FOREIGN_HOST = "dns-rebind.evil.example"

#: A plain browser origin (the test env allows it via CORS).
BROWSER_ORIGIN = "http://localhost:5173"


class TestProtectedResourceMetadata:
    async def test_metadata_is_public_and_advertises_the_logto_issuer(
        self, http_client
    ):
        """Unauthenticated GET returns the static protected-resource metadata."""
        resp = await http_client.get(WELL_KNOWN_PATH)
        assert resp.status_code == 200, (
            f"the well-known route must be PUBLIC and return metadata, got "
            f"{resp.status_code}: {resp.text}"
        )
        body = resp.json()
        assert body.get("authorization_servers") == [LOGTO_ISSUER], (
            f"`authorization_servers` must be the Logto OIDC issuer ({LOGTO_ISSUER!r}), "
            f"got: {body.get('authorization_servers')}"
        )
        resource = body.get("resource")
        assert isinstance(resource, str) and resource.startswith("https://"), (
            f"`resource` must be an https:// protected-MCP-endpoint URL "
            f"(never an http:// downgrade behind the TLS proxy), got: {resource!r}"
        )

    async def test_metadata_is_public_even_with_garbage_credentials(self, http_client):
        """The route must be reachable with (invalid) credentials present too —
        it is a static public document, never an authenticated resource."""
        resp = await http_client.get(
            WELL_KNOWN_PATH,
            headers={"Authorization": "Bearer not-a-real-token"},
        )
        assert resp.status_code == 200, (
            f"the well-known route must stay public, got {resp.status_code}: {resp.text}"
        )

    async def test_options_preflight_answered_by_cors_without_auth(self, http_client):
        """A CORS preflight (OPTIONS) on the well-known path is answered by the
        CORS middleware without any auth check."""
        resp = await http_client.options(
            WELL_KNOWN_PATH,
            headers={
                "Origin": BROWSER_ORIGIN,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code in (200, 204), (
            f"preflight must be answered by CORS without auth, got "
            f"{resp.status_code}: {resp.text}"
        )
        assert resp.headers.get("access-control-allow-origin"), (
            "preflight must include an access-control-allow-origin header"
        )

    async def test_default_port_host_form_serves_the_same_document(self, http_client):
        """QA BUG-002: `Host: <allowed-host>:443` must be treated as the
        allow-listed production Host (the ingress may forward the dialed
        port) and derive the SAME canonical, port-less document."""
        resp = await http_client.get(
            WELL_KNOWN_PATH, headers={"Host": f"{PROD_HOST}:443"}
        )
        assert resp.status_code == 200, (
            f"the default-port Host form must be allow-listed, got "
            f"{resp.status_code}: {resp.text}"
        )
        body = resp.json()
        assert body.get("authorization_servers") == [LOGTO_ISSUER]

    async def test_metadata_refuses_a_spoofed_foreign_host(self, http_client):
        """M-2: without a trustworthy origin, a non-allow-listed request Host
        can NEVER shape the advertised document — no attacker-derived
        `resource` URL is ever echoed."""
        resp = await http_client.get(WELL_KNOWN_PATH, headers={"Host": FOREIGN_HOST})
        assert resp.status_code == 404, (
            "the metadata document must not be served with an attacker-shaped "
            f"resource URL, got {resp.status_code}: {resp.text}"
        )
        assert "resource" not in resp.text, (
            f"no derived resource URL may be echoed, got: {resp.text}"
        )
