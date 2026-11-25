"""FastAPI routes for the MCP server registry.

Endpoints (all under ``/api/v1/mcp/servers``, protected by the global API key
dependency injected when the router is included in ``main.py``):

    POST   /                 — create a registered MCP server (201, masked)
    POST   /validate         — dry-run validation of a server config (200)
    GET    /                 — list all registered servers (200, masked)
    GET    /{name}           — get a single server (200, masked)
    GET    /{name}/reveal    — get a single server with plaintext secrets (200)
    PUT    /{name}           — update a server (200, masked)
    DELETE /{name}           — delete a server (204)

The openapi source_type extends the request/response bodies with two optional
fields:

- ``source_type`` (``"external"`` default | ``"openapi"``)
- ``openapi_url`` (``str | None``, required when ``source_type="openapi"``)

When ``source_type="openapi"``, the backend fetches the OpenAPI spec, builds an
in-process FastMCP server via the ``GeneratedMcpRunnerPort``, and mounts it
under ``/generated/{name}/mcp``; the ``url`` field of the request body is then
omitted (the mounted URL is generated and returned in the response).
"""

import logging
from datetime import UTC, datetime
from typing import Annotated, Self

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from application.use_cases.create_mcp_server import CreateMcpServerUseCase
from application.use_cases.delete_mcp_server import DeleteMcpServerUseCase
from application.use_cases.get_mcp_server import GetMcpServerUseCase
from application.use_cases.list_mcp_servers import ListMcpServersUseCase
from application.use_cases.update_mcp_server import UpdateMcpServerUseCase
from application.use_cases.validate_mcp_server import ValidateMcpServerUseCase
from dependencies import (
    get_create_mcp_server_use_case,
    get_delete_mcp_server_use_case,
    get_get_mcp_server_use_case,
    get_list_mcp_servers_use_case,
    get_request_auth_context,
    get_update_mcp_server_use_case,
    get_validate_mcp_server_use_case,
)
from domain.entities.auth.auth_context import AuthContext
from domain.entities.mcp_server_config import (
    McpAuthMode,
    McpServerConfig,
    McpTransportType,
)
from domain.entities.mcp_server_registry_entry import RegisteredMcpServer
from domain.services.mcp_auth_mode import user_credential_headers

logger = logging.getLogger(__name__)

mcp_registry_router = APIRouter(prefix="/api/v1/mcp/servers", tags=["mcp-registry"])


class _McpServerBaseRequest(BaseModel):
    """Shared fields for create/update/validate request bodies.

    Per-source-type required fields are enforced by
    :meth:`validate_source_type_fields`: ``url`` is required for
    ``source_type="external"`` (a configured remote URL) and ``openapi_url`` is
    required for ``source_type="openapi"`` (the mounted URL is generated, so
    ``url`` is omitted on the request).

    ``auth_mode`` (feat-mcp-oauth2 REQ-004) selects the credential-transmission
    mode: ``"headers"`` (default — the entered headers are sent as-is) or
    ``"user_token"`` (the consuming service injects the connected user's
    credential at load/validate time). Unknown values are rejected (the field
    is typed, not an open string) so the registry never silently degrades an
    unsupported mode.
    """

    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    auth_token: str | None = None
    auth_mode: McpAuthMode = McpAuthMode.HEADERS
    source_type: str = "external"
    openapi_url: str | None = None

    @model_validator(mode="after")
    def validate_source_type_fields(self) -> Self:
        """Enforce per-source-type required fields.

        Returns:
            The validated request.

        Raises:
            ValueError: If ``source_type="openapi"`` without ``openapi_url``,
                or ``source_type="external"`` without ``url``.
        """
        if self.source_type == "openapi":
            if not self.openapi_url:
                raise ValueError("'openapi_url' is required when source_type='openapi'")
        else:  # external (default)
            if not self.url:
                raise ValueError("'url' is required when source_type='external'")
        return self


class McpServerCreateRequest(_McpServerBaseRequest):
    """Request body for creating a registered MCP server.

    Attributes:
        name: Unique server name (1-100 chars).
    """

    name: str = Field(..., min_length=1, max_length=100)


class McpServerValidateRequest(_McpServerBaseRequest):
    """Request body for dry-run validation of an MCP server config."""


class McpServerUpdateRequest(BaseModel):
    """Request body for updating a registered MCP server (PARTIAL update).

    Every field is optional; ONLY the explicitly provided fields are applied
    (BUG-001) — the unprovided secret and configuration fields are preserved
    by the route's server-side merge, since the registry owns the decrypted
    storage. An empty body is rejected. Coherence (per-source-type required
    fields) is validated on the MERGED entity server-side, after the merge —
    not on the partial body alone.

    ``auth_mode`` (feat-mcp-oauth2 REQ-004): ABSENT preserves the stored mode,
    an explicit value updates it, and an explicit ``null`` is rejected — the
    mode is non-secret metadata, so ``null`` carries no "keep stored secret"
    meaning the way ``headers``/``env``/``auth_token`` nulls might.

    The server ``name`` is taken from the URL path, not the body.
    """

    url: str | None = None
    headers: dict[str, str] | None = None
    env: dict[str, str] | None = None
    auth_token: str | None = None
    auth_mode: McpAuthMode | None = None
    source_type: str | None = None
    openapi_url: str | None = None

    @model_validator(mode="after")
    def _at_least_one_field(self) -> Self:
        """Reject a genuinely empty update body.

        Raises:
            ValueError: If no editable field was provided.
        """
        if not self.model_fields_set:
            raise ValueError("provide at least one editable field")
        return self

    @model_validator(mode="after")
    def _reject_null_auth_mode(self) -> Self:
        """Reject an explicit ``null`` ``auth_mode`` (REQ-004).

        Only ABSENT (preserve) and a VALUE (update) are meaningful for the
        non-secret mode; an explicit ``null`` would bypass the typed merge
        (``model_copy`` does not re-validate) and violate the entity's
        ``NOT NULL`` column downstream.

        Raises:
            ValueError: If ``auth_mode`` is explicitly ``null``.
        """
        if self.auth_mode is None and "auth_mode" in self.model_fields_set:
            raise ValueError("'auth_mode' cannot be null (omit it to preserve)")
        return self


class McpServerValidateResponse(BaseModel):
    """Response body for the validate endpoint."""

    tool_count: int


#: Controlled coherence messages (stable, non-sensitive — safe to surface to
#: REST and MCP clients; never embed request payloads or secrets).
_COHERENCE_EXTERNAL_URL_REQUIRED = "'url' is required when source_type='external'"
_COHERENCE_OPENAPI_URL_REQUIRED = "'openapi_url' is required when source_type='openapi'"


def _validate_merged_spec(
    *, url: str | None, openapi_url: str | None, source_type: str
) -> None:
    """Validate per-source-type coherence on the MERGED server spec.

    Raises:
        HTTPException: 422 with the controlled coherence message — the same
            wording as the create validator, always non-sensitive.
    """
    invalid = source_type == "openapi" and not openapi_url
    invalid = invalid or (source_type != "openapi" and not url)
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[
                {
                    "type": "value_error",
                    "loc": ["body"],
                    "msg": f"Value error, {_COHERENCE_OPENAPI_URL_REQUIRED if source_type == 'openapi' else _COHERENCE_EXTERNAL_URL_REQUIRED}",
                }
            ],
        )


def _mask(entry: RegisteredMcpServer) -> RegisteredMcpServer:
    """Return a copy of ``entry`` with secrets stripped (auth_token, headers, env).

    ``auth_token`` is set to ``None`` in masked responses. This is ambiguous
    with an entry that genuinely has no token; use the ``/reveal`` endpoint to
    fetch the plaintext value and distinguish "no token" from "masked token".

    Args:
        entry: The registered server entry with plaintext secrets.

    Returns:
        A frozen copy with ``auth_token=None``, ``headers={}``, ``env={}``.
        ``auth_mode`` is deliberately NOT masked: it is non-secret metadata
        (REQ-004) and the registry's consumers read it from the echo.
    """
    return entry.model_copy(update={"auth_token": None, "headers": {}, "env": {}})


@mcp_registry_router.post("", status_code=status.HTTP_201_CREATED)
async def create_mcp_server(
    body: McpServerCreateRequest,
    use_case: Annotated[
        CreateMcpServerUseCase, Depends(get_create_mcp_server_use_case)
    ],
) -> RegisteredMcpServer:
    """Create a new registered MCP server.

    Args:
        body: The request body with server configuration.
        use_case: Injected create use case.

    Returns:
        The created registered server entry (secrets masked).
    """
    now = datetime.now(UTC)
    entry = RegisteredMcpServer(
        name=body.name,
        transport=McpTransportType.HTTP,
        url=body.url,
        headers=body.headers,
        env=body.env,
        auth_token=body.auth_token,
        auth_mode=body.auth_mode,
        source_type=body.source_type,
        openapi_url=body.openapi_url,
        created_at=now,
        updated_at=now,
    )
    created = await use_case.execute(entry)
    return _mask(created)


@mcp_registry_router.post("/validate", status_code=status.HTTP_200_OK)
async def validate_mcp_server(
    body: McpServerValidateRequest,
    use_case: Annotated[
        ValidateMcpServerUseCase, Depends(get_validate_mcp_server_use_case)
    ],
    caller: Annotated[AuthContext | None, Depends(get_request_auth_context)],
) -> McpServerValidateResponse:
    """Dry-run validation of an MCP server connection.

    For ``source_type="external"`` (default), validates a remote MCP server URL.
    For ``source_type="openapi"``, fetches the OpenAPI spec, builds an in-process
    FastMCP server, and counts its tools (no mount, no persistence).

    ``auth_mode`` (feat-mcp-oauth2 REQ-004) is honored at validate time — "Test
    connection" probes the target EXACTLY as the runtime would: when the
    request carries ``auth_mode: "user_token"`` and the caller is
    authenticated, the caller's credential (``Authorization: Bearer <jwt>`` for
    a JWT caller, ``X-API-Key: <key>`` for an API-key caller) is injected into
    the probe headers (injected credentials win over colliding header names).
    A ``headers``-mode request or a request without a caller credential (the
    legacy path) is probed unchanged.

    Args:
        body: The request body with server configuration (no persistence).
        use_case: Injected validate use case.
        caller: The authenticated caller context (``None`` when the request
            carries no dual-auth credential).

    Returns:
        The number of tools discovered on the server.
    """
    if body.source_type == "openapi":
        fetch_headers = {
            **body.headers,
            **user_credential_headers(body.auth_mode, caller),
        }
        tool_count = await use_case.execute_openapi(
            openapi_url=body.openapi_url or "",
            headers=fetch_headers,
        )
    else:
        config = McpServerConfig(
            name="validate",
            transport=McpTransportType.HTTP,
            url=body.url or "",
            headers=body.headers,
            env=body.env,
            auth_token=body.auth_token,
        )
        injected = user_credential_headers(body.auth_mode, caller)
        if injected:  # user_token probe: the caller's credential reaches the target
            config = config.model_copy(
                update={"headers": {**config.headers, **injected}}
            )
        tool_count = await use_case.execute(config)
    return McpServerValidateResponse(tool_count=tool_count)


@mcp_registry_router.get("", status_code=status.HTTP_200_OK)
async def list_mcp_servers(
    use_case: Annotated[ListMcpServersUseCase, Depends(get_list_mcp_servers_use_case)],
) -> list[RegisteredMcpServer]:
    """List all registered MCP servers (secrets masked).

    Args:
        use_case: Injected list use case.

    Returns:
        A list of registered server entries with secrets stripped.
    """
    entries = await use_case.execute()
    return [_mask(e) for e in entries]


@mcp_registry_router.get("/{name}", status_code=status.HTTP_200_OK)
async def get_mcp_server(
    name: str,
    use_case: Annotated[GetMcpServerUseCase, Depends(get_get_mcp_server_use_case)],
) -> RegisteredMcpServer:
    """Get a single registered MCP server by name (secrets masked).

    Args:
        name: The unique server name.
        use_case: Injected get use case.

    Returns:
        The registered server entry with secrets stripped.
    """
    entry = await use_case.execute(name)
    return _mask(entry)


@mcp_registry_router.get("/{name}/reveal", status_code=status.HTTP_200_OK)
async def reveal_mcp_server(
    name: str,
    use_case: Annotated[GetMcpServerUseCase, Depends(get_get_mcp_server_use_case)],
) -> RegisteredMcpServer:
    """Get a single registered MCP server with plaintext secrets revealed.

    Args:
        name: The unique server name.
        use_case: Injected get use case.

    Returns:
        The registered server entry with plaintext secrets.
    """
    return await use_case.execute(name)


@mcp_registry_router.put("/{name}", status_code=status.HTTP_200_OK)
async def update_mcp_server(
    name: str,
    body: McpServerUpdateRequest,
    get_use_case: Annotated[GetMcpServerUseCase, Depends(get_get_mcp_server_use_case)],
    use_case: Annotated[
        UpdateMcpServerUseCase, Depends(get_update_mcp_server_use_case)
    ],
) -> RegisteredMcpServer:
    """Update an existing registered MCP server (PARTIAL, secrets masked in response).

    Only the explicitly provided fields change; every unprovided field
    (including the stored ``auth_token``, ``headers`` and ``env``) is
    preserved from the existing entry — the registry is the single owner of
    the decrypted storage and the only place where a secret-preserving merge
    is possible (BUG-001). ``tool_count`` is inherited from the existing entry
    and refreshed by the update use case when it re-validates (BUG-003).

    Args:
        name: The unique server name (from the URL path).
        body: The PARTIAL request body (provided fields only).
        get_use_case: Injected get use case (reads the decrypted entry).
        use_case: Injected update use case.

    Returns:
        The updated registered server entry with secrets stripped.

    Raises:
        HTTPException: 404 when the entry does not exist; 422 when the merged
            specification is incoherent (controlled, non-sensitive detail).
    """
    existing = await get_use_case.execute(name)
    merged = existing.model_copy(update=body.model_dump(exclude_unset=True))
    _validate_merged_spec(
        url=merged.url,
        openapi_url=merged.openapi_url,
        source_type=str(merged.source_type),
    )
    updated = await use_case.execute(name, merged)
    return _mask(updated)


@mcp_registry_router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_mcp_server(
    name: str,
    use_case: Annotated[
        DeleteMcpServerUseCase, Depends(get_delete_mcp_server_use_case)
    ],
) -> None:
    """Delete a registered MCP server by name.

    Args:
        name: The unique server name.
        use_case: Injected delete use case.
    """
    await use_case.execute(name)
