"""Behavioral tests — explicit MCP TransportSecuritySettings (REQ-014).

feat-mcp-oauth2-dual-auth, mcp-raganything slice. Entry point: the four MCP
mounts on the REAL app, probed with a valid API key so the ONLY variable under
test is the transport-security validation of the Host header.

Delegation contract pinned here (mirrors the composable-agents REQ-013
pattern, mission-mandated for raganything as REQ-014):

* each of the four FastMCP apps receives an explicit
  ``TransportSecuritySettings(enable_dns_rebinding_protection=True,
  allowed_hosts=<settings.MCP_ALLOWED_HOSTS>)``;
* a ``Host: raganything.soludev.tech`` request does NOT get 421 (the probe
  presents a NON-localhost Host — the historical 127.0.0.1 probes masked the
  bug);
* a foreign Host is rejected with 421 — DNS-rebinding protection stays
  ENABLED;
* the default-port Host forms (``host:443`` / ``host:80``, QA BUG-002) are
  accepted for an allow-listed host and rejected for a foreign one.

The mount probes use the trailing-slash path deliberately: it reaches the
FastMCP transport layer TODAY without the (also broken, separately pinned in
``test_mcp_exact_path_no_redirect.py``) mount redirect in the way, so a
failure here is purely about transport security.

RED phase: fastmcp 3.3.1 constructs its streamable transports WITHOUT any
security_settings (the mcp SDK then disables DNS-rebinding protection by
default), so the foreign-Host 421 pins fail today with "got 200".
"""

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")

#: The production proxy Host / the foreign (never allowed) Host.
PROD_HOST = "raganything.soludev.tech"
FOREIGN_HOST = "dns-rebind.evil.example"
MCP_MOUNTS = ("/files/mcp", "/classical/mcp", "/web/mcp", "/memories/mcp")
#: Accepted credentials from the root ``auth_directory`` external doubles.
API_KEY_KEY_ALICE = "key-alice"


class TestHostAllowListEnforcement:
    @pytest.mark.parametrize("mount", MCP_MOUNTS)
    async def test_production_host_is_accepted_no_421(
        self, http_client, mount, mcp_post
    ):
        """Host `raganything.soludev.tech` (in MCP_ALLOWED_HOSTS) reaches the
        MCP handshake — never 421."""
        resp = await mcp_post(
            http_client,
            f"http://{PROD_HOST}{mount}/",
            headers={"X-API-Key": API_KEY_KEY_ALICE},
        )
        assert resp.status_code != 421, (
            f"Host {PROD_HOST} must be in MCP_ALLOWED_HOSTS — got 421: {resp.text!r}"
        )
        assert resp.status_code == 200, (
            f"the MCP handshake with the production Host must succeed, got "
            f"{resp.status_code}: {resp.text!r}"
        )

    @pytest.mark.parametrize("mount", MCP_MOUNTS)
    async def test_foreign_host_is_rejected_with_421(
        self, http_client, mount, mcp_post
    ):
        """DNS-rebinding protection stays ENABLED: an unknown Host → 421."""
        resp = await mcp_post(
            http_client,
            f"http://{FOREIGN_HOST}{mount}/",
            headers={"X-API-Key": API_KEY_KEY_ALICE},
        )
        assert resp.status_code == 421, (
            f"a non-allowed Host must still be rejected with 421 (DNS-rebinding "
            f"protection ENABLED), got {resp.status_code}: {resp.text!r}"
        )


class TestDefaultPortHostForms:
    """QA BUG-002 mirrored from composable-agents: the SDK's exact Host
    matcher must still accept the DEFAULT-port suffixes for an allow-listed
    host while a foreign Host stays foreign."""

    async def test_allowed_host_with_https_default_port_is_accepted(
        self, http_client, mcp_post
    ):
        resp = await mcp_post(
            http_client,
            f"http://{PROD_HOST}:443/memories/mcp/",
            headers={"X-API-Key": API_KEY_KEY_ALICE},
        )
        assert resp.status_code == 200, (
            f"Host {PROD_HOST}:443 is the allow-listed endpoint behind the TLS "
            f"proxy — got {resp.status_code}: {resp.text!r}"
        )

    async def test_allowed_host_with_http_default_port_is_accepted(
        self, http_client, mcp_post
    ):
        resp = await mcp_post(
            http_client,
            f"http://{PROD_HOST}:80/memories/mcp/",
            headers={"X-API-Key": API_KEY_KEY_ALICE},
        )
        assert resp.status_code == 200, (
            f"Host {PROD_HOST}:80 is the same allow-listed endpoint — got "
            f"{resp.status_code}: {resp.text!r}"
        )

    async def test_foreign_host_with_default_port_is_still_rejected(
        self, http_client, mcp_post
    ):
        resp = await mcp_post(
            http_client,
            f"http://{FOREIGN_HOST}:443/memories/mcp/",
            headers={"X-API-Key": API_KEY_KEY_ALICE},
        )
        assert resp.status_code == 421, (
            f"a foreign Host stays foreign with an explicit port — got "
            f"{resp.status_code}: {resp.text!r}"
        )


class TestOriginAllowList:
    """code-review M-1 (regression): the SDK's Origin matcher is EXACT and
    has no global wildcard — without the ``allowed_origins`` wiring, EVERY
    browser-origin MCP request was 403'd. The Origin policy is
    ``MCP_ALLOWED_ORIGINS`` falling back to the app CORS policy
    (``ALLOWED_ORIGINS``); the CORS ``*`` wildcard is mirrored as "Origin
    not validated" (the SDK treats a missing Origin as same-origin)."""

    async def test_browser_origin_request_is_not_403ed(self, http_client, mcp_post):
        """The deployed config shape (``ALLOWED_ORIGINS='["*"]'`` and no
        dedicated MCP list): an Origin-bearing request must be ACCEPTED —
        exactly what the app's own CORS policy allows. (Test env: both
        settings unset/unpatched → the wildcard default.)"""
        resp = await mcp_post(
            http_client,
            "/files/mcp/",
            headers={
                "X-API-Key": API_KEY_KEY_ALICE,
                "Origin": "http://localhost:5173",
            },
        )
        assert resp.status_code != 403, (
            f"a browser-origin MCP request was rejected by the transport "
            f"security: {resp.text!r}"
        )
        assert resp.status_code == 200, (
            f"the Origin-bearing request must reach the MCP handshake, got "
            f"{resp.status_code}: {resp.text!r}"
        )

    async def test_explicit_mcp_allowed_origins_accept_the_configured_origin(
        self, http_client, mcp_post, monkeypatch
    ):
        """An explicit (non-wildcard) ``MCP_ALLOWED_ORIGINS`` list: the
        configured Origin passes the SDK's exact matcher."""
        import main

        monkeypatch.setattr(
            main.app_config,
            "MCP_ALLOWED_ORIGINS",
            ["http://composable-ui.soludev.tech"],
        )
        resp = await mcp_post(
            http_client,
            "/files/mcp/",
            headers={
                "X-API-Key": API_KEY_KEY_ALICE,
                "Origin": "http://composable-ui.soludev.tech",
            },
        )
        assert resp.status_code == 200, (
            f"the configured Origin must be accepted (SDK exact match), got "
            f"{resp.status_code}: {resp.text!r}"
        )

    async def test_explicit_mcp_allowed_origins_reject_a_foreign_origin(
        self, http_client, mcp_post, monkeypatch
    ):
        """SDK semantics with an explicit list: a foreign Origin → 403
        (DNS-rebinding protection covers browser-side rebinding too)."""
        import main

        monkeypatch.setattr(
            main.app_config,
            "MCP_ALLOWED_ORIGINS",
            ["http://composable-ui.soludev.tech"],
        )
        resp = await mcp_post(
            http_client,
            "/files/mcp/",
            headers={
                "X-API-Key": API_KEY_KEY_ALICE,
                "Origin": "http://evil-rebinder.example",
            },
        )
        assert resp.status_code == 403, (
            f"a foreign Origin must be rejected per the SDK semantics with an "
            f"allow-listed Origin policy, got {resp.status_code}: {resp.text!r}"
        )
