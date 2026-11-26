"""FastAPI routes for the MCP server registry.

Endpoints (all under ``/api/v1/mcp/servers``, protected by the global API key
dependency injected when the router is included in ``main.py``):

    POST   /                 — create a registered MCP server (201, masked);
                               the create-time probe carries the caller's
                               credential when ``auth_mode="user_token"``
                               (REQ-004 / BUG-006, probe-only injection)
    POST   /validate         — dry-run validation of a server config (200);
                               an optional ``id`` (or ``server_id`` —
                               BUG-005 alias) merges the probe over the
                               STORED entry (UI BUG-001: masked edit payload)
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
from pydantic import AliasChoices, BaseModel, Field, model_validator

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
    get_get_mcp_server_use_case_optional,
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
from domain.errors.config import DependencyNotInitializedError
from domain.services.mcp_auth_mode import user_credential_headers
from domain.services.mcp_validate_merge import (
    McpValidateOverrides,
    resolve_validate_candidate,
)

logger = logging.getLogger(__name__)

mcp_registry_router = APIRouter(prefix="/api/v1/mcp/servers", tags=["mcp-registry"])


#: Controlled coherence messages (stable, non-sensitive — safe to surface to
#: REST and MCP clients; never embed request payloads or secrets).
_COHERENCE_EXTERNAL_URL_REQUIRED = "'url' is required when source_type='external'"
_COHERENCE_OPENAPI_URL_REQUIRED = "'openapi_url' is required when source_type='openapi'"


def _check_source_type_fields(
    *, source_type: str, url: str | None, openapi_url: str | None
) -> None:
    """Enforce the per-source-type required fields (shared by create/validate).

    Args:
        source_type: The effective source type of the candidate.
        url: The effective remote URL (``None`` = unset).
        openapi_url: The effective spec URL (``None`` = unset).

    Raises:
        ValueError: If ``source_type="openapi"`` without ``openapi_url``,
            or ``source_type="external"`` without ``url``.
    """
    if source_type == "openapi":
        if not openapi_url:
            raise ValueError(_COHERENCE_OPENAPI_URL_REQUIRED)
    else:  # external (default)
        if not url:
            raise ValueError(_COHERENCE_EXTERNAL_URL_REQUIRED)


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
        _check_source_type_fields(
            source_type=self.source_type,
            url=self.url,
            openapi_url=self.openapi_url,
        )
        return self


class McpServerCreateRequest(_McpServerBaseRequest):
    """Request body for creating a registered MCP server.

    Attributes:
        name: Unique server name (1-100 chars).
    """

    name: str = Field(..., min_length=1, max_length=100)


class McpServerValidateRequest(_McpServerBaseRequest):
    """Request body for dry-run validation of an MCP server config.

    ``id`` (UI BUG-001, edit-mode "Test connection"): OPTIONAL primary key
    (the registered server's ``name``) of an EXISTING registered server. When
    present, the stored entry (secrets decrypted by the registry store,
    RLS-scoped to the caller; a foreign/unknown ``id`` answers 404) SUPPLIES
    the credential/config values the probe uses, and the payload fields
    override ONLY what the client explicitly changed — an absent or
    masked-empty value for a credential field (``headers={}``, ``env={}``,
    ``auth_token=""``) NEVER clears a stored secret (the masked UI of
    REQ-016 cannot send stored secrets back). ``auth_mode`` follows the
    current validate semantics: the explicitly provided request value wins,
    the stored mode otherwise. No source-type coherence is enforced on the
    payload alone when ``id`` is present — the check runs server-side on the
    MERGED specification (same contract as the partial update route).

    The stored-entry reference is accepted under BOTH wire names (BUG-005):
    ``id`` remains the primary (first) name and ``server_id`` is the
    composable-ui edit dialog's field — either populates the same ``id``
    attribute, and when BOTH are sent ``id`` wins. Accepting the UI's name
    keeps the two halves of the wire contract pinned together (the UI's
    behavioural suite sends ``server_id``; this backend's suite sends ``id``).
    """

    id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("id", "server_id"),
    )

    @model_validator(mode="after")
    def validate_source_type_fields(self) -> Self:
        """Enforce per-source-type coherence, deferring it when ``id`` is set.

        With an ``id`` the per-source-type required fields may come from the
        STORED entry (e.g. an ``openapi`` server validated without a payload
        ``url``), so the empty-payload check would misreject the merge; the
        coherence is validated on the merged values by the route instead.

        Returns:
            The validated request.

        Raises:
            ValueError: If no ``id`` is present and the per-source-type
                required fields are missing.
        """
        if self.id is not None:
            return self
        _check_source_type_fields(
            source_type=self.source_type,
            url=self.url,
            openapi_url=self.openapi_url,
        )
        return self


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
    caller: Annotated[AuthContext | None, Depends(get_request_auth_context)],
) -> RegisteredMcpServer:
    """Create a new registered MCP server.

    The create-time reachability probe (the tool load that records
    ``tool_count``; for ``source_type="openapi"`` the spec fetch) runs with
    ``auth_mode`` honored (REQ-004, BUG-006): when ``auth_mode="user_token"``
    and the caller is authenticated, the caller's credential
    (``Authorization: Bearer <jwt>`` / ``X-API-Key: <key>``, derived by the
    ``user_credential_headers`` domain rule) is merged into the PROBE headers
    ONLY — an injected credential wins over colliding header names. The
    injected value is never persisted: the stored entry keeps exactly the
    configured headers (the calling user's token is per-request state, not a
    stored secret). The default ``headers`` mode probes unchanged, and a
    ``user_token`` request without a caller credential (legacy/unattended
    path) drops the injection (drop-empty semantics — the probe then answers
    the target's own 401 exactly as before).

    Args:
        body: The request body with server configuration.
        use_case: Injected create use case.
        caller: The authenticated caller context (``None`` when the request
            carries no dual-auth credential).

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
    probe_credentials = user_credential_headers(body.auth_mode, caller)
    created = await use_case.execute(entry, probe_credentials=probe_credentials)
    return _mask(created)


def _payload_overrides(body: McpServerValidateRequest) -> McpValidateOverrides:
    """Project the candidate payload into the merge rule's override contract.

    Transport-boundary mapping (the resolution itself stays in the domain
    rule): a masked, empty or absent value becomes ``None`` — it can NEVER
    override a stored secret (BUG-001) — while a real value is an explicit
    dirty change. ``auth_mode`` and ``source_type`` are honored when the
    client EXPLICITLY provides them (``fields_set`` distinguishes a real
    value from the DTO default: current validate semantics).

    Args:
        body: The validated validate request body.

    Returns:
        The dirty-field overrides for :func:`resolve_validate_candidate`.
    """
    provided = body.model_fields_set
    return McpValidateOverrides(
        url=body.url or None,
        headers=body.headers or None,
        env=body.env or None,
        auth_token=body.auth_token or None,
        auth_mode=body.auth_mode if "auth_mode" in provided else None,
        source_type=body.source_type if "source_type" in provided else None,
        openapi_url=body.openapi_url or None,
    )


@mcp_registry_router.post("/validate", status_code=status.HTTP_200_OK)
async def validate_mcp_server(
    body: McpServerValidateRequest,
    use_case: Annotated[
        ValidateMcpServerUseCase, Depends(get_validate_mcp_server_use_case)
    ],
    caller: Annotated[AuthContext | None, Depends(get_request_auth_context)],
    get_use_case: Annotated[
        GetMcpServerUseCase | None,
        Depends(get_get_mcp_server_use_case_optional),
    ],
) -> McpServerValidateResponse:
    """Dry-run validation of an MCP server connection.

    For ``source_type="external"`` (default), validates a remote MCP server URL.
    For ``source_type="openapi"``, fetches the OpenAPI spec, builds an in-process
    FastMCP server, and counts its tools (no mount, no persistence).

    When the body carries ``id`` of an EXISTING registered server (UI BUG-001,
    edit-mode "Test connection"), the probe run against the STORED entry
    merged with the payload: the stored (decrypted, RLS-scoped) credentials
    are supplied by the registry and only the explicitly changed payload
    fields override them — a masked-empty value (``headers={}``, ``env={}``,
    ``auth_token=""``/``null``) never clears a stored secret. A foreign or
    unknown ``id`` answers 404 (per-user isolation). The response stays the
    standard validate payload — the stored secrets are never echoed. The
    reference is accepted as ``id`` (primary) or ``server_id`` (BUG-005: the
    composable-ui payload name — both populate the same field; ``id`` wins
    when both are sent).

    ``auth_mode`` (feat-mcp-oauth2 REQ-004) is honored at validate time — "Test
    connection" probes the target EXACTLY as the runtime would: when the
    effective ``auth_mode`` is ``"user_token"`` and the caller is
    authenticated, the caller's credential (``Authorization: Bearer <jwt>`` for
    a JWT caller, ``X-API-Key: <key>`` for an API-key caller) is injected into
    the probe headers (injected credentials win over colliding header names).
    A ``headers``-mode probe or a request without a caller credential (the
    legacy path) is probed unchanged.

    Args:
        body: The request body (candidate config, plus the optional ``id`` of
            a stored server to merge over).
        use_case: Injected validate use case.
        caller: The authenticated caller context (``None`` when the request
            carries no dual-auth credential).
        get_use_case: Injected get use case over the registry store (``None``
            when the registry is disabled — the stored entry is read ONLY
            when ``id`` is present).

    Returns:
        The number of tools discovered on the server.

    Raises:
        HTTPException: 404 when ``id`` names no server of this caller; 422
            when the probe cannot reach the target or the merged
            specification is incoherent (controlled, non-sensitive detail).
    """
    stored = None
    if body.id:
        if get_use_case is None:
            # id-driven validation against a DISABLED registry: the same
            # "registry disabled" contract as every store-dependent route
            # (503 — never a guessed 404).
            raise DependencyNotInitializedError(
                "MCP registry store is not initialized; id-based validation "
                "requires the application lifespan to wire the registry."
            )
        stored = await get_use_case.execute(body.id)
    spec = resolve_validate_candidate(stored, _payload_overrides(body))
    _validate_merged_spec(
        url=spec.url, openapi_url=spec.openapi_url, source_type=spec.source_type
    )
    if spec.source_type == "openapi":
        fetch_headers = {
            **spec.headers,
            **user_credential_headers(spec.auth_mode, caller),
        }
        tool_count = await use_case.execute_openapi(
            openapi_url=spec.openapi_url or "",
            headers=fetch_headers,
        )
    else:
        config = McpServerConfig(
            name=body.id or "validate",
            transport=McpTransportType.HTTP,
            url=spec.url or "",
            headers=spec.headers,
            env=spec.env,
            auth_token=spec.auth_token,
        )
        injected = user_credential_headers(spec.auth_mode, caller)
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
