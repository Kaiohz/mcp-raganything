"""Behavioral tests — public `/.well-known/oauth-protected-resource` (REQ-002).

feat-mcp-oauth2-dual-auth, mcp-raganything slice (REQ-010 replication).
Entry point: the REAL FastAPI app (httpx AsyncClient over ASGITransport), i.e.
the production router chain including the CORS middleware.

Delegation contract pinned here:

* ``GET /.well-known/oauth-protected-resource`` is PUBLIC (no credentials, no
  auth dependency at all) and returns a document whose
  ``authorization_servers`` is the Logto OIDC issuer
  (``https://logto.soludev.tech/oidc``) and whose ``resource`` is the
  canonical ``https://`` protected-MCP-endpoint URL (the scheme NEVER
  downgrades behind the proxy; the global document keeps advertising the
  canonical ``/files/mcp`` resource);
* ``GET /.well-known/oauth-protected-resource/<mount>`` (QA BUG-002) is
  PUBLIC and advertises THAT mount's own resource identifier
  (``https://<host><mount>``) — served for the four built-in mounts and the
  ``/generated/{name}/mcp`` pattern; unknown paths answer a plain 404;
* ``OPTIONS`` preflights on the well-known paths are answered by the CORS
  middleware WITHOUT any auth check;
* ``Host: <allowed-host>:443`` (QA BUG-002) is treated like the port-less
  production Host form;
* a spoofed foreign Host can never shape the documents (M-2 hardening): the
  metadata is simply not served with an attacker-shaped resource URL.
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

#: The four built-in MCP mounts + the generated pattern (the per-mount
#: documents QA BUG-002 requires; the generated path shape matches what the
#: registry mounts at runtime).
MCP_MOUNTS_FOR_METADATA = (
    "/files/mcp",
    "/classical/mcp",
    "/web/mcp",
    "/memories/mcp",
    "/generated/qa-gen-meta/mcp",
)


def _assert_static_document(resp) -> dict:
    """Shared block: 200, Logto issuer, https resource of the expected shape."""
    assert resp.status_code == 200, (
        f"the well-known route must be PUBLIC and return metadata, got "
        f"{resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert body.get("authorization_servers") == [LOGTO_ISSUER], (
        f"`authorization_servers` must be the Logto OIDC issuer ({LOGTO_ISSUER!r}), "
        f"got: {body.get('authorization_servers')}"
    )
    return body


def _assert_https_resource(resource) -> str:
    assert isinstance(resource, str) and resource.startswith("https://"), (
        f"`resource` must be an https:// protected-MCP-endpoint URL "
        f"(never an http:// downgrade behind the TLS proxy), got: {resource!r}"
    )
    return resource


class TestProtectedResourceMetadata:
    async def test_metadata_is_public_and_advertises_the_canonical_resource(
        self, http_client
    ):
        """Unauthenticated GET returns the static, CANONICAL document.

        QA BUG-002: one document cannot serve four distinct mounts — the
        global document keeps advertising the canonical ``/files/mcp``
        resource; every mount points at its OWN per-mount document (see
        ``TestPerMountProtectedResourceMetadata``).
        """
        resp = await http_client.get(WELL_KNOWN_PATH)
        body = _assert_static_document(resp)
        resource = _assert_https_resource(body.get("resource"))
        assert resource == f"https://{PROD_HOST}/files/mcp", (
            "the GLOBAL document advertises the canonical files resource "
            "(per mounts have their own documents — QA BUG-002), "
            f"got: {resource!r}"
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


class TestPerMountProtectedResourceMetadata:
    """QA BUG-002: ONE global document cannot serve four distinct mounts.

    Each mount's own document lives at
    ``/.well-known/oauth-protected-resource/<mount>`` and advertises THAT
    mount's resource identifier — the 401 challenge of a mount points at its
    own document, so a client connecting e.g. to ``/web/mcp`` is told to
    request a token for ``/web/mcp`` (never for the canonical ``/files/mcp``).
    """

    @pytest.mark.parametrize("mount", MCP_MOUNTS_FOR_METADATA)
    async def test_per_mount_document_advertises_that_mounts_resource(
        self, http_client, mount
    ):
        """A per-mount document is PUBLIC and advertises the mount's own
        resource (the four built-in mounts + the generated pattern)."""
        resp = await http_client.get(f"{WELL_KNOWN_PATH}{mount}")
        body = _assert_static_document(resp)
        resource = _assert_https_resource(body.get("resource"))
        assert resource == f"https://{PROD_HOST}{mount}", (
            f"the per-mount document must advertise the mount's OWN resource "
            f"({mount!r}), got: {resource!r}"
        )

    async def test_per_mount_document_is_public_with_garbage_credentials(
        self, http_client
    ):
        """The per-mount route is a static public document too."""
        resp = await http_client.get(
            f"{WELL_KNOWN_PATH}/web/mcp",
            headers={"Authorization": "Bearer not-a-real-token"},
        )
        body = _assert_static_document(resp)
        assert body.get("resource") == f"https://{PROD_HOST}/web/mcp"

    async def test_unknown_mount_path_answers_a_plain_404(self, http_client):
        """Only MCP mounts are advertised — everything else is a plain 404."""
        resp = await http_client.get(f"{WELL_KNOWN_PATH}/not-an-mcp-mount")
        assert resp.status_code == 404, (
            f"an unknown well-known sub-path is never a discovery document, "
            f"got {resp.status_code}: {resp.text}"
        )
        assert "resource" not in resp.text

    async def test_default_port_host_form_serves_the_per_mount_document(
        self, http_client
    ):
        """QA BUG-002 Host normalization applies to the per-mount documents
        too: `Host: <allowed-host>:443` derives the same canonical origin."""
        resp = await http_client.get(
            f"{WELL_KNOWN_PATH}/web/mcp", headers={"Host": f"{PROD_HOST}:443"}
        )
        body = _assert_static_document(resp)
        assert body.get("resource") == f"https://{PROD_HOST}/web/mcp"

    async def test_per_mount_document_refuses_a_spoofed_foreign_host(self, http_client):
        """M-2: a non-allow-listed Host never shapes a per-mount document."""
        resp = await http_client.get(
            f"{WELL_KNOWN_PATH}/web/mcp", headers={"Host": FOREIGN_HOST}
        )
        assert resp.status_code == 404, (
            "the per-mount metadata must not be served with an attacker-shaped "
            f"resource URL, got {resp.status_code}: {resp.text}"
        )
        assert "resource" not in resp.text

    async def test_preflight_on_per_mount_path_answered_by_cors_without_auth(
        self, http_client
    ):
        """CORS answers OPTIONS preflights on the per-mount path too."""
        resp = await http_client.options(
            f"{WELL_KNOWN_PATH}/web/mcp",
            headers={
                "Origin": BROWSER_ORIGIN,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code in (200, 204), (
            f"preflight must be answered by CORS without auth, got "
            f"{resp.status_code}: {resp.text}"
        )
        assert resp.headers.get("access-control-allow-origin")

    async def test_challenge_pointer_round_trip_is_the_mounts_own_document(
        self, http_client, mcp_post
    ):
        """The discovery chain for `/web/mcp` end-to-end: the mount's 401
        challenge advertises the per-mount URL, and GETting exactly that URL
        returns the document advertising `/web/mcp`."""
        resp = await mcp_post(http_client, "/web/mcp")
        challenge = resp.headers.get("www-authenticate")
        advertised_url = challenge.split('"')[1] if challenge is not None else ""
        assert advertised_url == f"https://{PROD_HOST}{WELL_KNOWN_PATH}/web/mcp", (
            f"the /web/mcp challenge must point at ITS OWN per-mount document, "
            f"got: {challenge!r}"
        )
        doc = await http_client.get(advertised_url)
        body = _assert_static_document(doc)
        assert body.get("resource") == f"https://{PROD_HOST}/web/mcp"
