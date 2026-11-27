"""Registered MCP server domain entity (registry entry).

A ``RegisteredMcpServer`` is the persisted representation of an MCP server
that has been registered in the registry. Unlike :class:`McpServerConfig`
(which is the connection configuration used to load tools at runtime), this
entity carries the registry metadata: name, url, encrypted-at-rest secrets
(headers/env/auth_token are decrypted by the store on read), the resolved
``tool_count``, the source type (``external`` vs ``openapi``), and audit
timestamps.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from domain.entities.mcp_server_config import McpAuthMode, McpTransportType


class RegisteredMcpServer(BaseModel):
    """A registered MCP server entry in the registry.

    Attributes:
        name: Unique server name (1-100 chars). Acts as the primary key.
        transport: Transport type (stdio or http). Defaults to http.
        url: Server URL. Required and non-empty for ``source_type="external"``
            (enforced by the request validator). ``None`` for an
            ``source_type="openapi"`` server whose mounted URL is not yet known
            — the mounted URL is set by the use case before persistence, so a
            persisted row always carries a non-empty ``url`` (the DB column is
            ``NOT NULL``).
        headers: HTTP headers to send to the server (decrypted on read).
        env: Environment variables for stdio transport (decrypted on read).
        auth_token: Bearer auth token (decrypted on read, None if unset).
        auth_mode: Credential-transmission mode (feat-mcp-oauth2 REQ-004) —
            ``"headers"`` (default: the stored headers are sent as-is) or
            ``"user_token"`` (the consuming service injects the connected
            user's credential at load/validate time). Non-secret metadata:
            echoed unmasked in every masked projection, unlike the secrets.
        tool_count: Number of tools discovered during the last validation.
        tool_names: Names of the tools discovered during the last
            validation/create/update probe (non-secret metadata, echoed
            unmasked like ``tool_count``; empty for unknown/legacy rows).
        source_type: Origin of the server — ``"external"`` (plain MCP server
            reached over the network) or ``"openapi"`` (server built in-process
            from an OpenAPI spec via FastMCP and mounted under
            ``/generated/{name}/mcp``). Defaults to ``"external"``.
        openapi_url: URL the OpenAPI spec was fetched from (only set when
            ``source_type == "openapi"``).
        created_at: Creation timestamp (UTC).
        updated_at: Last update timestamp (UTC).
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1, max_length=100)
    transport: McpTransportType = McpTransportType.HTTP
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    auth_token: str | None = None
    auth_mode: McpAuthMode = McpAuthMode.HEADERS
    tool_count: int = 0
    tool_names: list[str] = Field(default_factory=list)
    source_type: str = "external"
    openapi_url: str | None = None
    created_at: datetime
    updated_at: datetime
