"""Port for mounting in-process generated MCP servers into the running app.

The :class:`GeneratedMcpRunnerPort` abstracts the lifecycle of FastMCP servers
built from OpenAPI specs and mounted under ``/generated/{name}/mcp`` in the
running FastAPI/Starlette app. The runner owns the ASGI lifespan of each
mounted server (started on mount, stopped on unmount) so that FastMCP's
session manager — which only exists during lifespan — is available while the
server is reachable.
"""

from abc import ABC, abstractmethod


class GeneratedMcpRunnerPort(ABC):
    """Outbound port: mount/unmount in-process generated MCP servers."""

    @abstractmethod
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
        ...

    @abstractmethod
    async def unmount(self, name: str) -> None:
        """Unmount a previously mounted generated MCP server (no-op if absent).

        Args:
            name: The unique server name to unmount.
        """
        ...

    @abstractmethod
    async def count_tools(self, name: str) -> int:
        """Count the tools exposed by a mounted generated MCP server.

        Args:
            name: The unique server name.

        Returns:
            The number of tools the server exposes.
        """
        ...

    async def list_tool_names(self, name: str) -> list[str]:
        """List the tool NAMES exposed by a mounted generated MCP server.

        Default (non-abstract) implementation is empty so lightweight fakes
        of this port keep working; the real runner overrides it. Used to
        persist ``tool_names`` on registry entries (metadata only — never
        raises into the create/update flow that annotates them).

        Args:
            name: The unique server name.

        Returns:
            The tool names (empty when the mount is unknown or listing fails
            in the implementation).
        """
        return []

    @abstractmethod
    async def close(self) -> None:
        """Close all mounted servers and release their resources at shutdown."""
        ...
