"""Behavioral tests — the generated ``/generated/{name}/mcp`` guard stack.

feat-mcp-oauth2-dual-auth, mcp-raganything slice, review iteration 2 (M-5):
the generated mounts ride the SAME HTTP-level guard stack as the four static
mounts (``main.build_mcp_guard_stack``: auth 401 + discovery challenge →
explicit transport security → STATELESS FastMCP ``http_app``). The only
previous test used a MagicMock app (the fallback branch), so the production
mount path was never exercised — the contract pinned here, on the REAL app:

* unauthenticated exact-path POST  → HTTP 401 + the OAuth2 discovery
  challenge + the generic body (never a 307 / never an initialize);
* authenticated exact-path POST    → 200, served DIRECTLY (no redirect);
* authenticated trailing-slash POST → 200 (the Mount keeps serving sub-paths
  and the guard normalizes the path);
* the exact-path ``Route`` is appended to the REAL app's ``router.routes``
  while mounted, and BOTH entries (Route + Mount) are removed on unmount;
* the RLS principal of every tool execution is the accepted request's
  principal (C-1 with the stateless transport) — the tool reads the RLS
  contextvars itself and reports them.

The generated server is mounted through the REAL ``GeneratedMcpRunner``
(the app's own ``app.state.generated_mcp_runner`` with its real guard
stack); ``FastMCP.from_openapi`` is patched to return a REAL FastMCP
instance with an RLS-observing probe tool (an upstream petstore is never
contacted), and the ``httpx.AsyncClient`` the runner builds is never used.
"""

from unittest.mock import patch

import pytest
from fastmcp import FastMCP
from starlette.routing import Mount, Route

from infrastructure.database.rls_context import (
    current_auth_method,
    current_credential,
    current_user_id,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

PROD_HOST = "raganything.soludev.tech"
GENERATED_NAME = "probe-petstore"
GENERATED_MOUNT = f"/generated/{GENERATED_NAME}/mcp"
EXPECTED_WWW_AUTHENTICATE = (
    f'Bearer resource_metadata="https://{PROD_HOST}'
    f'/.well-known/oauth-protected-resource"'
)
GENERIC_401_BODY = {"detail": "Invalid or missing credentials"}

#: Accepted credentials from the root ``auth_directory`` external doubles.
API_KEY_KEY_ALICE = "key-alice"
API_KEY_KEY_BOB = "key-bob"

#: The spec the runner would carry; ``from_openapi`` is patched so only the
#: fields the runner READS need to exist (``servers`` + ``info``).
_SPEC = {
    "openapi": "3.0.3",
    "info": {"title": "Probe Petstore", "version": "1.0.0"},
    "paths": {"/pets": {"get": {"operationId": "listPets"}}},
    "servers": [{"url": "https://petstore.example.com"}],
}


async def _mount_probe_server(running_app) -> str:
    """Mount a REAL FastMCP server whose tool reports its RLS principal.

    Returns:
        The absolute mounted URL (the runner's contract).
    """
    real_mcp = FastMCP("Probe Petstore")

    @real_mcp.tool()
    async def probe_rls_identity() -> str:
        """Report the RLS contextvars the tool task actually inherited."""
        return "|".join(
            (
                current_user_id.get() or "",
                current_auth_method.get() or "",
                current_credential.get() or "",
            )
        )

    runner = running_app.state.generated_mcp_runner
    with (
        patch("fastmcp.FastMCP.from_openapi", return_value=real_mcp),
        patch("httpx.AsyncClient"),
    ):
        return await runner.mount(GENERATED_NAME, _SPEC, headers={})


def _route_paths(app) -> list:
    return [getattr(route, "path", None) for route in app.router.routes]


class TestGeneratedMountGuardStack:
    async def test_unauthenticated_exact_path_post_is_401_with_challenge(
        self, running_app, http_client, mcp_post
    ):
        await _mount_probe_server(running_app)
        try:
            resp = await mcp_post(http_client, GENERATED_MOUNT)
        finally:
            await running_app.state.generated_mcp_runner.unmount(GENERATED_NAME)
        assert resp.status_code == 401, (
            f"the generated mount must answer 401 before any MCP method "
            f"(same contract as the static mounts), got {resp.status_code}: "
            f"{resp.text!r}"
        )
        assert resp.headers.get("www-authenticate") == EXPECTED_WWW_AUTHENTICATE
        assert resp.json() == GENERIC_401_BODY

    async def test_authenticated_exact_path_and_trailing_slash_are_served_directly(
        self, running_app, http_client, mcp_post
    ):
        await _mount_probe_server(running_app)
        responses = []
        try:
            for path in (GENERATED_MOUNT, f"{GENERATED_MOUNT}/"):
                responses.append(
                    await mcp_post(
                        http_client, path, headers={"X-API-Key": API_KEY_KEY_ALICE}
                    )
                )
        finally:
            await running_app.state.generated_mcp_runner.unmount(GENERATED_NAME)
        for path, resp in zip(
            (GENERATED_MOUNT, f"{GENERATED_MOUNT}/"), responses, strict=True
        ):
            assert resp.status_code == 200, (
                f"POST {path} must be served DIRECTLY (no 307, REQ-015), "
                f"got {resp.status_code}: {resp.text!r}"
            )

    async def test_exact_path_route_is_appended_and_removed_on_unmount(
        self, running_app
    ):
        runner = running_app.state.generated_mcp_runner
        assert GENERATED_MOUNT not in _route_paths(running_app)

        await _mount_probe_server(running_app)
        try:
            paths = _route_paths(running_app)
            assert paths.count(GENERATED_MOUNT) == 2, (
                "the generated mount must append the exact-path Route AND the "
                f"Mount on the REAL router.routes, got {paths}"
            )
            entries = [
                route
                for route in running_app.router.routes
                if getattr(route, "path", None) == GENERATED_MOUNT
            ]
            assert any(isinstance(entry, Route) for entry in entries), (
                "the exact-path entry must be a real Starlette Route"
            )
            assert any(isinstance(entry, Mount) for entry in entries)
        finally:
            await runner.unmount(GENERATED_NAME)

        assert GENERATED_MOUNT not in _route_paths(running_app), (
            "unmount must remove BOTH the exact-path Route and the Mount"
        )

    async def test_tool_executes_under_the_accepted_request_principal(
        self, running_app, mcp_client_factory
    ):
        await _mount_probe_server(running_app)
        try:
            async with mcp_client_factory(
                GENERATED_MOUNT, credential=API_KEY_KEY_BOB
            ) as client:
                result = await client.call_tool(
                    "probe_rls_identity", {}, raise_on_error=False
                )
        finally:
            await running_app.state.generated_mcp_runner.unmount(GENERATED_NAME)

        assert result.is_error is False, result.content
        text = result.content[0].text
        assert "user-bob" in text, (
            f"the generated-mount tool must run under BOB's RLS principal "
            f"(per-request, stateless transport), got: {text!r}"
        )
        assert "user-alice" not in text, f"cross-user leak: {text!r}"
        assert "api_key" in text, f"the auth method must reach the tool: {text!r}"
