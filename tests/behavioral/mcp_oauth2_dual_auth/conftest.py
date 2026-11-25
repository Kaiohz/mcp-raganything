"""Feature-local fixtures for the MCP OAuth2 + dual-auth behavioral suite.

feat-mcp-oauth2-dual-auth — mcp-raganything slice (step 1, RED phase).

Every probe goes through the REAL app from the root ``running_app`` fixture
(httpx ``ASGITransport``) with ``follow_redirects=False`` so a redirect is
OBSERVED instead of silently followed (REQ-015 pins exact-path serving). The
base URL carries the production proxy Host — the same non-localhost Host the
ingress presents, which the historical conftest probes masked (REQ-014).

Real implementations everywhere; the ONLY double is the outbound Logto JWKS
boundary in :func:`real_jwt_signer` (see its docstring).
"""

import base64
import time
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio

#: The production proxy Host (Host-header stand-in for the TLS-terminating
#: ingress; the test app itself runs in-process).
PROD_HOST = "raganything.soludev.tech"

#: The four streamable-HTTP MCP mounts (exact paths, no trailing slash).
MCP_MOUNTS = ("/files/mcp", "/classical/mcp", "/web/mcp", "/memories/mcp")

#: MCP authorization-spec discovery chain constants (REQ-010 replicating
#: REQ-001/REQ-002; the Logto OIDC issuer is pinned by the feature spec).
WELL_KNOWN_PATH = "/.well-known/oauth-protected-resource"
LOGTO_ISSUER = "https://logto.soludev.tech/oidc"


def expected_www_authenticate_for(mount_path: str) -> str:
    """The 401 discovery challenge pointing at ONE mount's own document.

    QA BUG-002: each mount advertises its OWN per-mount
    protected-resource document (``.well-known/oauth-protected-resource
    <mount_path>``), never the canonical ``/files/mcp`` one.

    Args:
        mount_path: The mount path (e.g. ``/files/mcp`` or
            ``/generated/{name}/mcp``).

    Returns:
        The expected ``WWW-Authenticate`` header value for the production Host.
    """
    return (
        f'Bearer resource_metadata="https://{PROD_HOST}{WELL_KNOWN_PATH}{mount_path}"'
    )


#: The existing generic, non-enumerating 401 body (AuthenticationError →
#: ErrorMessage.AUTH_INVALID_CREDENTIALS), unchanged by the feature.
GENERIC_401_BODY = {"detail": "Invalid or missing credentials"}

#: A Host that is NOT in the test env's MCP_ALLOWED_HOSTS.
FOREIGN_HOST = "dns-rebind.evil.example"

#: Accepted credentials from the root ``auth_directory`` external doubles
#: (same users as the other behavioral suites).
API_KEY_KEY_ALICE = "key-alice"


def jsonrpc_body(method: str) -> dict:
    """Build a raw JSON-RPC 2.0 probe payload for one MCP method."""
    body: dict = {"jsonrpc": "2.0", "id": 0, "method": method}
    if method == "initialize":
        body["params"] = {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "behavioral-probe", "version": "0.0.0"},
        }
    elif method == "tools/call":
        body["params"] = {"name": "list_files", "arguments": {}}
    return body


async def mcp_post_probe(
    client: httpx.AsyncClient,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    method: str = "initialize",
) -> httpx.Response:
    """Raw JSON-RPC POST to one MCP mount path — HTTP status is the behavior.

    ``path`` may be an absolute URL (used to override the Host via the URL's
    authority — ASGITransport performs no real network call).
    """
    return await client.post(
        path,
        headers={
            "Accept": "application/json, text/event-stream",
            **(headers or {}),
        },
        json=jsonrpc_body(method),
    )


@pytest.fixture
def mcp_post():
    """Raw JSON-RPC POST probe for one mount path.

    Usage: ``resp = await mcp_post(client, "/files/mcp", headers=..., method="initialize")``.
    ``path`` may be an absolute URL to override the request Host via the URL
    authority (ASGITransport performs no real network call).
    """

    async def probe(
        client: httpx.AsyncClient,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        method: str = "initialize",
    ) -> httpx.Response:
        return await mcp_post_probe(client, path, headers=headers, method=method)

    return probe


@pytest.fixture
def http_client(running_app):
    """httpx client over the REAL app; redirects are observable, not followed."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=running_app),
        base_url=f"http://{PROD_HOST}",
        follow_redirects=False,
    )


# ---------------------------------------------------------------------------
# Real-key JWT signing (Logto outbound boundary double)
# ---------------------------------------------------------------------------
_TEST_AUDIENCE = "mcp-raganything-behavioral-probe"
_TEST_KID = "behavioral-probe-kid"
_RSA_CACHE: tuple[bytes, dict] | None = None


def _b64url_uint(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _rsa_material() -> tuple[bytes, dict]:
    """Generate (once per session) a real RSA keypair + matching JWK."""
    global _RSA_CACHE
    if _RSA_CACHE is None:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        private_pem = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        numbers = key.public_key().public_numbers()
        jwk = {
            "kid": _TEST_KID,
            "kty": "RSA",
            "alg": "RS256",
            "use": "sig",
            "n": _b64url_uint(numbers.n),
            "e": _b64url_uint(numbers.e),
        }
        _RSA_CACHE = (private_pem, jwk)
    return _RSA_CACHE


@pytest_asyncio.fixture
async def real_jwt_signer(running_app, auth_directory):
    """Authenticate the mounts through the REAL JwtAdapter + real signatures.

    Logto's JWKS endpoint is the outbound external system: the adapter's JWKS
    HTTP fetch is stubbed with the behavioral RSA key set (the repo's own
    pattern from ``tests/unit/test_jwt_adapter.py``). Everything the JWT flow
    does stays REAL — TTLCache fetch/caching, kid resolution, ``jwt.decode``
    (signature + audience + expiry), ``AuthService`` orchestration and the RLS
    wiring — exactly mirroring the production wiring
    (``main._initialize_dual_auth``: adapter audience = ``app_config
    .JWT_AUDIENCE``). The adapter audience and ``app_config.JWT_AUDIENCE`` are
    set to the single ``JWT_AUDIENCE`` value, i.e. the ``JWT_AUDIENCES``-unset
    parity state (REQ-003-equivalent for raganything).
    """
    import jwt as pyjwt

    import dependencies as deps
    import main
    from domain.services.auth.auth_service import AuthService
    from infrastructure.security.jwt_adapter import JwtAdapter

    private_pem, jwk = _rsa_material()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(main.app_config, "JWT_AUDIENCE", _TEST_AUDIENCE)
    adapter = JwtAdapter(
        jwks_url=(
            f"{main.app_config.LOGTO_URL}/oidc/jwks"
            if main.app_config.LOGTO_URL
            else ""
        ),
        audience=main.app_config.JWT_AUDIENCE,
    )
    jwks_response = httpx.Response(
        200,
        json={"keys": [jwk]},
        # raise_for_status() requires the response to carry its request.
        request=httpx.Request("GET", "https://logto.soludev.tech/oidc/jwks"),
    )
    adapter._jwks_http_client = AsyncMock()
    adapter._jwks_http_client.get = AsyncMock(return_value=jwks_response)

    service = AuthService(
        jwt_port=adapter, api_key_repo=auth_directory.api_key_repository
    )
    previous_middleware_service = main.mcp_middleware._auth_service
    previous_rest_service = main.rest_security._auth_service
    previous_deps_auth = getattr(deps, "auth_service", None)
    main.mcp_middleware.set_auth_service(service)
    main.rest_security.set_auth_service(service)
    deps.auth_service = service
    try:
        yield _Signer(private_pem, pyjwt)
    finally:
        monkeypatch.undo()
        main.mcp_middleware.set_auth_service(previous_middleware_service)
        main.rest_security.set_auth_service(previous_rest_service)
        deps.auth_service = previous_deps_auth


class _Signer:
    """Issue real RS256 tokens with the behavioral key set."""

    def __init__(self, private_pem: bytes, pyjwt) -> None:
        self._private_pem = private_pem
        self._pyjwt = pyjwt

    def issue_token(self, *, sub: str, aud: str = _TEST_AUDIENCE) -> str:
        now = int(time.time())
        return self._pyjwt.encode(
            {"sub": sub, "aud": aud, "iat": now, "exp": now + 600},
            self._private_pem,
            algorithm="RS256",
            headers={"kid": _TEST_KID},
        )
