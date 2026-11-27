"""Generated MCP runner: mount in-process FastMCP servers into the running app.

The runner builds a FastMCP server from an OpenAPI spec (via
``FastMCP.from_openapi`` + ``httpx.AsyncClient``), wraps it with the configured
middleware (e.g. :class:`McpApiKeyMiddleware`), mounts the resulting ASGI app
under ``/generated/{name}/mcp`` in the host FastAPI/Starlette app, and drives
its lifespan through a dedicated runner-owned task
(:class:`_LifespanHolder`).

FastMCP's session manager (and the ``mcp.list_tools()`` API used by tool
counting, run with the middleware chain bypassed so the credential-guarded
server can be introspected without forging a credential) only exists while
the lifespan is active, so the lifespan is started on ``mount`` and stopped on
``unmount``. Closing the lifespan also unmounts the route from the app. The
lifespan MUST be owned by ONE task for the mount's whole lifetime: it enters
task-group machinery that only the entering task may exit — entering it
inside the create REQUEST's handler task corrupted the response machinery
once the handler returned (QA BUG-001), and closing it from a later request's
task only ever failed with "created in a different Context". The holder task
starts it on mount and closes it on unmount, whatever task calls ``mount``/
``unmount``.
"""

import asyncio
import logging
from collections.abc import Callable
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
        _mounts: Mapping of mounted name -> ``_Mount`` state (mcp, holder).
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

        # Drive the mounted server's lifespan from a DEDICATED runner-owned
        # task (QA BUG-001): the fastmcp lifespan enters task-group/cancel-
        # scope machinery that ONLY the entering task may exit — entering it
        # inside the create REQUEST's handler task and holding it open across
        # the request boundary corrupted the response machinery ("Attempted
        # to exit a cancel scope that isn't the current task's current cancel
        # scope" → every later request answers "No response returned"). The
        # task-owned lifecycle also lets ``unmount`` — called from ANY later
        # request (DELETE/update) — close the lifespan in the holder task's
        # own context instead of the previous cross-task ``stack.aclose()``.
        holder = _LifespanHolder(http_app, self._app)
        await holder.start()

        self._mounts[name] = _Mount(mcp=mcp, holder=holder, mount_path=mount_path)
        url = f"{self._base_url}{mount_path}"
        logger.info(LogMessage.MCP_RUNNER_MOUNTED, name, url, -1)
        return url

    async def unmount(self, name: str) -> None:
        """Unmount a previously mounted generated MCP server (no-op if absent).

        Closes the lifespan (in its holder task's own context) and removes the
        mount path from the host app's routes.

        Args:
            name: The unique server name to unmount.
        """
        mount = self._mounts.pop(name, None)
        if mount is None:
            return
        try:
            await mount.holder.stop()
        except Exception as e:  # noqa: BLE001 — lifespan close must not crash unmount
            logger.warning(LogMessage.MCP_RUNNER_LIFESPAN_CLOSE_FAILED, name, e)
        _remove_mount_route(self._app, mount.mount_path)
        logger.info(LogMessage.MCP_RUNNER_UNMOUNTED, name)

    async def count_tools(self, name: str) -> int:
        """Count the tools exposed by a mounted generated MCP server.

        The listing runs IN PROCESS through the FastMCP public API with the
        server's middleware chain BYPASSED (``list_tools(run_middleware=False)``
        — QA BUG-001): the mounted server carries ``McpApiKeyMiddleware`` as
        defense-in-depth, and the registry's own count call deliberately
        presents NO credential. Running the count through ``on_list_tools``
        raised ``ToolError("Invalid or missing credentials")`` inside fastmcp's
        task group — failing the registry create with a 500 AND corrupting the
        mounted server (every later request on the mount answered 500). Listing
        from the tool manager directly needs no credential and touches no
        session/middleware machinery; the count is a metadata annotation of an
        ALREADY-built server, not a caller.

        Args:
            name: The unique server name.

        Returns:
            The number of tools the server exposes.
        """
        mount = self._mounts[name]
        tools = await mount.mcp.list_tools(run_middleware=False)
        return len(tools)

    async def list_tool_names(self, name: str) -> list[str]:
        """List the tool NAMES of a mounted generated MCP server.

        Same in-process public listing as :meth:`count_tools` (middleware
        bypassed — QA BUG-001), returning the names for the registry's
        ``tool_names`` metadata.
        """
        if name not in self._mounts:
            return []
        try:
            tools = await self._mounts[name].mcp.list_tools(run_middleware=False)
        except Exception:  # noqa: BLE001 — annotating metadata must not raise
            return []
        return [str(tool.name) for tool in tools if getattr(tool, "name", None)]

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

    __slots__ = ("mcp", "holder", "mount_path")

    def __init__(self, mcp: Any, holder: "_LifespanHolder", mount_path: str) -> None:
        self.mcp = mcp
        self.holder = holder
        self.mount_path = mount_path


class _LifespanHolder:
    """Drive one mounted server's fastmcp lifespan from a dedicated task.

    The fastmcp streamable-HTTP lifespan enters anyio task-group/cancel-scope
    machinery that only the ENTERING task may exit and that must stay alive
    for the mount's whole lifetime — a request handler's task is neither
    (QA BUG-001). The holder starts the lifespan in its own ``asyncio`` task
    on :meth:`start`, keeps it open until :meth:`stop`, and closes it strictly
    within that task — ``unmount``/``close`` may be called from any other task
    and simply join the holder's exit.
    """

    def __init__(self, http_app: Any, host_app: Any) -> None:
        self._http_app = http_app
        self._host_app = host_app
        self._task: asyncio.Task[None] | None = None
        self._start_error: BaseException | None = None
        self._entered = asyncio.Event()
        self._stop = asyncio.Event()

    async def start(self) -> None:
        """Start the lifespan in the holder task and wait until it is entered.

        Args:
            http_app: (constructor) The FastMCP ASGI app whose lifespan is
                driven.
            host_app: (constructor) The host app the lifespan is attached to.

        Raises:
            Exception: Whatever the lifespan startup raised (surfaced to the
                caller after the task has reported entry failure).
        """
        self._task = asyncio.create_task(
            self._run(), name="generated-mcp-lifespan-holder"
        )
        await self._entered.wait()
        if self._start_error is not None:
            raise self._start_error

    async def stop(self) -> None:
        """Close the lifespan (inside the holder task) and join the task."""
        self._stop.set()
        if self._task is not None:
            # Joining re-raises the lifespan's own shutdown error, if any.
            await asyncio.gather(self._task)

    async def _run(self) -> None:
        try:
            async with self._http_app.lifespan(self._host_app):
                self._entered.set()
                await self._stop.wait()
        except BaseException as e:  # noqa: BLE001 — startup failures surface via start()
            if not self._entered.is_set():
                self._start_error = e
        finally:
            self._entered.set()


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
