"""Generated MCP runner: mount in-process FastMCP servers into the running app.

The runner builds a FastMCP server from an OpenAPI spec (via
``FastMCP.from_openapi`` + ``httpx.AsyncClient``), wraps it with the configured
middleware (e.g. :class:`McpApiKeyMiddleware`), mounts the resulting ASGI app
under ``/generated/{name}/mcp`` in the host FastAPI/Starlette app, and drives
its lifespan through an :class:`asyncio.AsyncExitStack`.

FastMCP's session manager (and the ``mcp.list_tools()`` API) only exists while
the lifespan is active, so the lifespan is started on ``mount`` and stopped on
``unmount``. Closing the lifespan also unmounts the route from the app.
"""

import logging
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any

import httpx
from fastmcp import FastMCP

from domain.errors.mcp import McpAlreadyMountedError, OpenApiInvalidSpecError
from domain.errors.messages import ErrorMessage
from domain.logging.messages import LogMessage
from domain.ports.generated_mcp_runner import GeneratedMcpRunnerPort

logger = logging.getLogger(__name__)


class GeneratedMcpRunner(GeneratedMcpRunnerPort):
    """Mount/unmount in-process generated MCP servers.

    Attributes:
        _app: The host FastAPI/Starlette app to mount servers into.
        _base_url: The absolute base URL (e.g. ``http://host:port``) used to
            build the mounted server URL.
        _middleware: Optional FastMCP middleware instances applied to each server.
        _guard_factory: Optional ``(inner_app, mount_path) -> asgi_app`` the
            runner applies to every built ``http_app`` before mounting — the
            HTTP-level auth + transport-security stack shared with the static
            mounts (feat-mcp-oauth2-dual-auth, REQ-017). ``None`` keeps the
            historic unguarded mounting.
        _mounts: Mapping of mounted name -> ``_Mount`` state (mcp, stack, route).
    """

    def __init__(
        self,
        app: Any,
        base_url: str,
        middleware: list[Any] | None = None,
        guard_factory: Callable[[Any, str], Any] | None = None,
    ) -> None:
        """Initialize the runner.

        Args:
            app: The host FastAPI/Starlette app servers are mounted into.
            base_url: Absolute base URL (e.g. ``http://host:port``) used to
                build the mounted server URL returned by ``mount``.
            middleware: Optional FastMCP middleware applied to each built server.
            guard_factory: Optional ASGI guard factory ``(inner_app,
                mount_path) -> guarded asgi app`` applied to every mounted
                server (the HTTP-level auth + transport-security stack). When
                ``None`` the built ``http_app`` is mounted directly.
        """
        self._app = app
        self._base_url = base_url.rstrip("/")
        self._middleware = list(middleware) if middleware else []
        self._guard_factory = guard_factory
        self._mounts: dict[str, _Mount] = {}

    async def mount(self, name: str, spec: dict, headers: dict[str, str]) -> str:
        """Build and mount a generated MCP server under ``/generated/{name}/mcp``.

        Args:
            name: The unique server name (used in the mount path).
            spec: The parsed OpenAPI 3.x document to build the server from.
            headers: Upstream auth headers forwarded to the httpx AsyncClient.

        Returns:
            The absolute URL the server is reachable at.

        Raises:
            McpAlreadyMountedError: If a server with this name is already mounted.
            OpenApiInvalidSpecError: If the spec has no ``servers`` entry.
        """
        if name in self._mounts:
            raise McpAlreadyMountedError(
                ErrorMessage.MCP_ALREADY_MOUNTED.format(name=name)
            )

        servers = spec.get("servers") or []
        if not servers:
            raise OpenApiInvalidSpecError(
                ErrorMessage.OPENAPI_INVALID_SPEC.format(
                    url="<mount>", reason="spec has no 'servers' entry"
                )
            )

        upstream_base_url = servers[0].get("url") or ""
        client = httpx.AsyncClient(base_url=upstream_base_url, headers=headers)

        info = spec.get("info") or {}
        mcp_name = info.get("title") or name
        mcp = FastMCP.from_openapi(openapi_spec=spec, client=client, name=mcp_name)

        for mw in self._middleware:
            mcp.add_middleware(mw)

        http_app = mcp.http_app(path="/", stateless_http=True)
        # Apply the shared HTTP-level guard stack (auth + transport security)
        # when one is wired — the generated mounts get the SAME treatment as
        # the static /files, /classical, /web and /memories servers
        # (feat-mcp-oauth2-dual-auth, REQ-017). The FastMCP middleware stay
        # mounted as defense-in-depth behind the HTTP guard. STATELESS
        # transport too (code-review C-1): the tool task inherits the
        # per-request context, so the guard's RLS principal — not a stale
        # session-creator principal — reaches the tool.
        mount_target = (
            self._guard_factory(http_app, f"/generated/{name}/mcp")
            if self._guard_factory is not None
            else http_app
        )
        mount_path = f"/generated/{name}/mcp"
        # Exact-path Route FIRST: POST <mount_path> (no trailing slash) is
        # served directly — no Starlette Mount 307 round-trip behind the
        # proxy (REQ-015, same pattern as the four static mounts) — while the
        # Mount keeps serving sub-paths. The HTTP guard rewrites both forms
        # to "/" for the inner app.
        self._mount_exact_path_route(mount_path, mount_target)
        self._app.mount(mount_path, mount_target)

        # Drive the mounted server's lifespan so FastMCP's session manager exists.
        stack = AsyncExitStack()
        await stack.enter_async_context(http_app.lifespan(self._app))

        self._mounts[name] = _Mount(mcp=mcp, stack=stack, mount_path=mount_path)
        url = f"{self._base_url}{mount_path}"
        logger.info(LogMessage.MCP_RUNNER_MOUNTED, name, url, -1)
        return url

    async def unmount(self, name: str) -> None:
        """Unmount a previously mounted generated MCP server (no-op if absent).

        Closes the lifespan (via the AsyncExitStack) and removes the mount path
        from the host app's routes.

        Args:
            name: The unique server name to unmount.
        """
        mount = self._mounts.pop(name, None)
        if mount is None:
            return
        try:
            await mount.stack.aclose()
        except Exception as e:  # noqa: BLE001 — lifespan close must not crash unmount
            logger.warning("Generated MCP lifespan close failed for %s: %s", name, e)
        _remove_mount_route(self._app, mount.mount_path)
        logger.info(LogMessage.MCP_RUNNER_UNMOUNTED, name)

    async def count_tools(self, name: str) -> int:
        """Count the tools exposed by a mounted generated MCP server.

        Args:
            name: The unique server name.

        Returns:
            The number of tools the server exposes.
        """
        mount = self._mounts[name]
        tools = await mount.mcp.list_tools()
        return len(tools)

    async def close(self) -> None:
        """Close all mounted servers and release their resources at shutdown."""
        # ``unmount`` pops from ``_mounts``, so iterate over a snapshot copy.
        for name in self._mounts.copy():
            await self.unmount(name)

    def _mount_exact_path_route(self, mount_path: str, mount_target: Any) -> None:
        """Append the exact-path ``Route`` serving ``mount_path`` directly.

        Best-effort: only a list-shaped ``app.router.routes`` (FastAPI or
        Starlette) is touched; anything else falls back to the Mount-only
        historic behavior (the redirect round-trip) instead of failing.
        """
        from starlette.routing import Route

        router = getattr(self._app, "router", None)
        routes = getattr(router, "routes", None) if router is not None else None
        if not isinstance(routes, list):
            logger.warning(LogMessage.MCP_EXACT_PATH_ROUTE_UNAVAILABLE, mount_path)
            return
        routes.append(
            Route(
                mount_path,
                mount_target,
                name=f"generated_mcp_{mount_path.strip('/').replace('/', '_')}_endpoint",
            )
        )


class _Mount:
    """Internal state for a mounted generated MCP server."""

    __slots__ = ("mcp", "stack", "mount_path")

    def __init__(self, mcp: Any, stack: AsyncExitStack, mount_path: str) -> None:
        self.mcp = mcp
        self.stack = stack
        self.mount_path = mount_path


def _remove_mount_route(app: Any, mount_path: str) -> None:
    """Remove a previously mounted route from a Starlette/FastAPI app.

    Starlette stores mounts as :class:`Mount` entries in ``app.routes``. We drop
    any route whose ``path`` matches the mount path so the server is no longer
    reachable. Failures are swallowed (best-effort cleanup).

    Args:
        app: The host FastAPI/Starlette app.
        mount_path: The mount path to remove (e.g. ``/generated/petstore/mcp``).
    """
    try:
        routes = getattr(app, "routes", None)
        if routes is None:
            return
        # Mutate the list in-place — Starlette exposes ``app.routes`` as a
        # read-only property backed by an internal list. Reassigning raises
        # AttributeError; slicing assignment (``routes[:] = ...``) mutates the
        # underlying list without triggering the property setter.
        routes[:] = [r for r in routes if getattr(r, "path", None) != mount_path]
    except Exception:  # noqa: BLE001 — route cleanup must not crash
        logger.warning("Failed to remove mount route %s", mount_path, exc_info=True)
