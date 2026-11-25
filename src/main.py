"""Main entry point for the MCP service."""

import asyncio
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
import uvicorn
from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp

from application.api.classical_indexing_routes import classical_indexing_router
from application.api.classical_query_routes import classical_query_router
from application.api.file_routes import file_router
from application.api.health_routes import health_router
from application.api.mcp_classical_tools import mcp_classical
from application.api.mcp_file_tools import mcp_files
from application.api.mcp_guard import (
    WELL_KNOWN_PROTECTED_RESOURCE_PATH,
    McpAuthGuard,
    public_discovery_origin,
)
from application.api.mcp_memory_tools import mcp_memories
from application.api.mcp_registry_routes import mcp_registry_router
from application.api.mcp_transport_security import McpTransportSecurityGuard
from application.api.mcp_web_tools import mcp_web
from application.error_handlers import (
    authentication_error_handler,
    classical_error_handler,
    config_error_handler,
    dependency_not_initialized_handler,
    document_error_handler,
    domain_error_handler,
    file_error_handler,
    file_too_large_handler,
    file_validation_handler,
    indexing_error_handler,
    invalid_api_key_handler,
    llm_not_configured_handler,
    mcp_invalid_stored_config_handler,
    storage_error_handler,
    storage_not_found_handler,
    unsupported_format_handler,
    vector_bad_request_handler,
    vector_store_error_handler,
)
from dependencies import (
    app_config,
    classical_bm25_adapter,
    classical_vector_store,
    db_config,
)
from domain.errors.base import DomainError
from domain.errors.classical import ClassicalConfigError
from domain.errors.config import ConfigError, DependencyNotInitializedError
from domain.errors.document import DocumentReadError, UnsupportedFormatError
from domain.errors.file import FileError, FileTooLargeError, FileValidationError
from domain.errors.indexing import IndexingError
from domain.errors.llm import LlmNotConfiguredError
from domain.errors.mcp import McpServerInvalidAuthModeError
from domain.errors.security import AuthenticationError, InvalidApiKeyError
from domain.errors.storage import StorageError, StorageNotFoundError
from domain.errors.vector_store import VectorStoreConfigError, VectorStoreError
from domain.logging.messages import LogMessage
from domain.services.auth.auth_service import AuthService
from infrastructure.database.api_key_reader import AsyncpgApiKeyReader
from infrastructure.database.mcp_registry_store import McpRegistryStore
from infrastructure.database.user_llm_reader import AsyncpgUserLlmReader
from infrastructure.logging import RequestIdMiddleware, configure_logging
from infrastructure.openapi_mcp.rehydration import rehydrate_openapi_servers
from infrastructure.openapi_mcp.runner import GeneratedMcpRunner
from infrastructure.security.crypto import FernetSecretCipher
from infrastructure.security.jwt_adapter import JwtAdapter
from security import ComposableAgentsSecurity, McpApiKeyMiddleware

configure_logging(app_config)

logger = logging.getLogger(__name__)

#: The four built-in MCP mount paths (wired below). The per-mount
#: protected-resource documents (QA BUG-002) are served for these paths plus
#: the ``/generated/{name}/mcp`` pattern the registry mounts at runtime.
MCP_MOUNT_PATHS = ("/files/mcp", "/classical/mcp", "/web/mcp", "/memories/mcp")

#: The MCP resource advertised by the GLOBAL protected-resource metadata
#: document (REQ-002-equivalent). QA BUG-002: one document cannot serve four
#: distinct mounts, so the GLOBAL document keeps advertising the canonical
#: files server while each mount's OWN document
#: (``/.well-known/oauth-protected-resource/<mount>``) advertises
#: ``https://<origin><mount>``.
MCP_DISCOVERY_RESOURCE_PATH = "/files/mcp"


def _is_mcp_mount_path(path: str) -> bool:
    """Whether ``path`` is an MCP mount a per-mount discovery document serves.

    The four built-in mounts plus the ``/generated/{name}/mcp`` pattern the
    registry mounts at runtime (QA BUG-002) — everything else answers a plain
    404 (no discovery document is manufactured for non-mount paths).

    Args:
        path: The candidate mount path below the well-known prefix.

    Returns:
        True when a per-mount protected-resource document is served for it.
    """
    return (
        path in MCP_MOUNT_PATHS or re.fullmatch(r"/generated/.+/mcp", path) is not None
    )


def _run_alembic_upgrade() -> None:
    """Run Alembic migrations to head (owns the mcp_servers table schema).

    Runs in a worker thread because Alembic's online mode drives an async
    engine via ``asyncio.run`` internally. The migration state is tracked in
    the ``raganything_alembic_version`` table, separate from composable-agents'
    ``alembic_version``, so both services can share the same database without
    colliding on Alembic's bookkeeping.
    """
    from alembic.config import Config

    from alembic import command

    alembic_dir = Path(__file__).parent / "alembic"
    cfg = Config(str(Path(__file__).parent / "alembic.ini"))
    cfg.set_main_option("script_location", str(alembic_dir))
    command.upgrade(cfg, "head")


#: REST-layer security instance: master-key fallback when dual auth is not
#: wired, dual-auth once the lifespan injects the ``AuthService``. REST routes
#: authenticate through this instance only; the MCP tool mounts authenticate
#: through ``mcp_middleware``. Keeping the two apart means any wiring performed
#: on the MCP side never alters the REST credential path.
rest_security = ComposableAgentsSecurity(master_key=app_config.API_KEY)

#: asyncpg pool for the MCP registry store; created in the lifespan, closed on
#: shutdown. None until the lifespan runs.
_mcp_pool: asyncpg.Pool | None = None


@asynccontextmanager
async def db_lifespan(_app: FastAPI):
    """Close process-wide resources and connection pools on shutdown.

    Closes the shared web crawler (if it was lazily created), the dedicated
    DDGS search executor and any dangling memory-store HTTP client, in
    addition to the classical RAG adapters. Every close is idempotent so
    repeated shutdowns are harmless.
    """
    yield

    logger.info(LogMessage.APP_SHUTDOWN_INITIATED)
    if classical_bm25_adapter is not None:
        try:
            await classical_bm25_adapter.close()
        except Exception:
            logger.exception(LogMessage.APP_BM25_CLOSE_FAILED)
    if classical_vector_store is not None:
        try:
            await classical_vector_store.close()
        except Exception:
            logger.exception(LogMessage.APP_CLASSICAL_VS_CLOSE_FAILED)
    # The web/memory adapters are lazy singletons; access them through the
    # dependencies module at call time so shutdown sees the built instances.
    import dependencies as _deps

    if _deps.web_crawler_adapter is not None:
        try:
            await _deps.web_crawler_adapter.close()
        except Exception:
            logger.exception(LogMessage.CRAWLER_CLOSE_FAILED)
    if _deps.web_search_adapter is not None:
        try:
            _deps.web_search_adapter.close()
        except Exception:
            logger.exception(LogMessage.WEB_SEARCH_EXECUTOR_CLOSE_FAILED)
    if _deps.memory_store_adapter is not None:
        try:
            await _deps.memory_store_adapter.close()
        except Exception:
            logger.exception(LogMessage.MEMORY_STORE_CLOSE_FAILED)
    logger.info(LogMessage.APP_SHUTDOWN_COMPLETE)


mcp_middleware = McpApiKeyMiddleware(master_key=app_config.API_KEY)
mcp_files.add_middleware(mcp_middleware)
mcp_classical.add_middleware(mcp_middleware)
mcp_web.add_middleware(mcp_middleware)
mcp_memories.add_middleware(mcp_middleware)

# STATELESS transport (CRITICAL fix, code-review C-1): each HTTP request — and
# the MCP tool task it dispatches — runs under its OWN task context, so the
# RLS contextvars the HTTP guard sets on the ACCEPTED request's principal are
# the ones the tool sees. In the fastmcp-default STATEFUL mode the tool task
# inherits the SESSION CREATOR's context instead (a stale principal from an
# earlier request — an actual cross-user isolation hazard when the
# method-level middleware is not re-deriving it). Do NOT rely on the global
# FASTMCP_STATELESS_HTTP env var: the fastmcp default is stateful, and the
# prod configmap only sets it incidentally.
mcp_files_app = mcp_files.http_app(path="/", stateless_http=True)
mcp_classical_app = mcp_classical.http_app(path="/", stateless_http=True)
mcp_web_app = mcp_web.http_app(path="/", stateless_http=True)
mcp_memories_app = mcp_memories.http_app(path="/", stateless_http=True)


def build_mcp_guard_stack(inner_app: ASGIApp, mount_path: str) -> ASGIApp:
    """Compose one MCP mount's HTTP-level ASGI guard stack (REQ-014/015/017).

    The stack (outermost first) is: ``McpAuthGuard`` — dual-auth HTTP guard
    delegating to the wired dual-auth wiring (Bearer JWT + X-API-Key, RLS
    contextvars) and answering unauthenticated requests with HTTP 401 + the
    OAuth2 discovery challenge BEFORE any MCP method runs — then
    ``McpTransportSecurityGuard`` — the explicit ``TransportSecuritySettings``
    validation (Host allow-list — foreign Host → 421 — plus the Origin
    allow-list mirroring the app's CORS policy, M-1) — then the FastMCP
    ``http_app`` (STATELESS, so the guard's per-request RLS principal is the
    one the tool task inherits, code-review C-1).

    Host/origin lists are read from ``app_config`` at REQUEST time so the
    configmap stays authoritative without a rebuild. The inner
    ``McpApiKeyMiddleware`` (already mounted on every FastMCP server) stays
    as defense-in-depth behind the HTTP guard: it reuses the guard's
    verified principal for the same request's credential instead of
    re-deriving it (single authentication per HTTP request, M-2).

    Args:
        inner_app: The FastMCP ``http_app`` to guard.
        mount_path: The public mount path (e.g. ``/files/mcp`` or
            ``/generated/{name}/mcp``) for the guard's path normalization.

    Returns:
        The guarded ASGI callable to route/mount. ``ASGIApp`` = a callable
        ``(scope, receive, send) -> Awaitable[None]``.
    """
    return McpAuthGuard(
        McpTransportSecurityGuard(inner_app, app_config),
        mcp_middleware.authenticate_headers,
        mount_path=mount_path,
        settings=app_config,
    )


async def _initialize_mcp_registry(
    app: FastAPI,
) -> tuple[FernetSecretCipher | None, McpRegistryStore | None]:
    """Create the asyncpg pool + cipher and rehydrate generated MCP servers.

    Returns:
        The (cipher, store) pair (either may be ``None`` when the registry is
        disabled or initialization failed) and assigns the asyncpg pool to the
        module-level ``_mcp_pool``.

    Raises:
        Exception: Propagates nothing — every failure is logged and reported
            through the return values so startup can continue.
    """
    global _mcp_pool
    cipher = None
    store = None
    try:
        if app_config.SECRET_ENCRYPTION_KEY:
            cipher = FernetSecretCipher(app_config.SECRET_ENCRYPTION_KEY)
            _mcp_pool = await asyncpg.create_pool(
                dsn=db_config.asyncpg_url,
                statement_cache_size=db_config.POSTGRES_STATEMENT_CACHE_SIZE or 100,
            )
            store = McpRegistryStore(pool=_mcp_pool, cipher=cipher)
            _deps.mcp_registry_store = store
            runner = getattr(app.state, "generated_mcp_runner", None)
            if runner is not None:
                await rehydrate_openapi_servers(
                    store=store,
                    runner=runner,
                    factory=_deps.get_mcp_openapi_factory(),
                )
        else:
            logger.warning(
                "SECRET_ENCRYPTION_KEY not set; MCP server registry is disabled."
            )
    except Exception:
        logger.exception("MCP registry initialization failed; continuing startup.")
    return cipher, store


def _initialize_dual_auth() -> None:
    """Wire JWT + per-user API-key authentication into the wired security.

    The JwtAdapter fetches the JWKS document from LOGTO_URL/oidc/jwks and
    caches it in memory; the AsyncpgApiKeyReader reads the shared ``api_keys``
    table (owned by composable-agents) via the shared asyncpg pool. The
    :class:`AuthService` orchestrates JWT-first / API-key-fallback precedence
    and is injected into the REST security instance and the MCP middleware.
    When LOGTO_URL is empty, JWT verification is disabled and only per-user
    API keys are accepted.
    """
    try:
        if _mcp_pool is not None:
            jwt_adapter = JwtAdapter(
                jwks_url=(
                    f"{app_config.LOGTO_URL}/oidc/jwks" if app_config.LOGTO_URL else ""
                ),
                audience=app_config.JWT_AUDIENCE,
                audiences=app_config.JWT_AUDIENCES,
            )
            api_key_reader = AsyncpgApiKeyReader(pool=_mcp_pool)
            _deps.jwt_adapter = jwt_adapter
            _deps.api_key_reader = api_key_reader
            _deps.auth_service = AuthService(
                jwt_port=jwt_adapter,
                api_key_repo=api_key_reader,
            )
            rest_security.set_auth_service(_deps.auth_service)
            mcp_middleware.set_auth_service(_deps.auth_service)
            logger.info(
                "Dual auth wired: jwt=%s api_key_reader=asyncpg",
                "enabled" if app_config.LOGTO_URL else "disabled (no LOGTO_URL)",
            )
        else:
            logger.warning(
                "Dual auth not wired: asyncpg pool not available "
                "(SECRET_ENCRYPTION_KEY unset or pool creation failed)."
            )
    except Exception:
        logger.exception("Dual auth initialization failed; continuing startup.")


def _initialize_user_llm_reader(cipher: FernetSecretCipher | None) -> None:
    """Wire the per-user LLM credentials reader from the shared asyncpg pool.

    The reader resolves per-user ``(base_url, api_key)`` from the shared
    ``user_llm_settings`` table (owned by composable-agents, read-only here)
    so the per-user LLM/embeddings factories can build per-request clients.
    Reuses the same asyncpg pool + Fernet cipher as the MCP registry store.
    """
    try:
        if _mcp_pool is not None and cipher is not None:
            _deps.user_llm_reader = AsyncpgUserLlmReader(pool=_mcp_pool, cipher=cipher)
            logger.info("Per-user LLM credentials reader wired (asyncpg).")
        else:
            logger.warning(
                "Per-user LLM reader not wired: asyncpg pool or cipher "
                "unavailable. Per-user LLM features will fall back to env "
                "credentials (legacy) or raise LlmNotConfiguredError."
            )
    except Exception:
        logger.exception(
            "Per-user LLM reader initialization failed; continuing startup."
        )


async def _close_startup_resources() -> None:
    """Close generated MCP runner, JwtAdapter and the registry pool."""
    if _deps.generated_mcp_runner is not None:
        try:
            await _deps.generated_mcp_runner.close()
        except Exception:
            logger.exception("Failed to close generated MCP runner.")
    if _deps.jwt_adapter is not None:
        try:
            await _deps.jwt_adapter.close()
        except Exception:
            logger.exception("Failed to close JWT adapter HTTP client.")
    if _mcp_pool is not None:
        await _mcp_pool.close()


def _log_mcp_discovery_shape_warnings() -> None:
    """Log (non-fatal) the MCP OAuth2 discovery-configuration shape.

    Mirrors composable-agents' ``_log_mcp_discovery_shape_warnings`` (M-3):
    a silent misconfiguration becomes a visible startup warning. The
    metadata document only advertises an issuer/origin a PUBLIC MCP client
    can reach: NOTHING advertised (both ``MCP_AUTHORIZATION_SERVER`` and
    ``LOGTO_URL`` empty — the ``<LOGTO_URL>/oidc`` fallback is resolved ONCE
    at startup by the ``AppConfig`` model validator, so
    ``MCP_AUTHORIZATION_SERVER`` below carries the EFFECTIVE advertised
    issuer, explicit or fallback alike) and a non-https authorization
    server are warned for QA BUG-003; a non-https ``MCP_PUBLIC_BASE_URL``
    origin is warned for NEW-1 (a scheme-less value already REFUSES startup
    in ``AppConfig._validate_mcp_public_base_url``).
    """
    if not app_config.MCP_AUTHORIZATION_SERVER and not app_config.LOGTO_URL:
        logger.warning(LogMessage.MCP_AUTHORIZATION_SERVER_NOT_ADVERTISED)
    elif app_config.MCP_AUTHORIZATION_SERVER and not (
        app_config.MCP_AUTHORIZATION_SERVER.startswith("https://")
    ):
        logger.warning(
            LogMessage.MCP_AUTHORIZATION_SERVER_NOT_HTTPS,
            app_config.MCP_AUTHORIZATION_SERVER,
        )
    if app_config.MCP_PUBLIC_BASE_URL and not (
        app_config.MCP_PUBLIC_BASE_URL.startswith("https://")
    ):
        logger.warning(LogMessage.MCP_PUBLIC_BASE_URL_NOT_HTTPS)


@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    """Combine database lifecycle with MCP lifespans and registry rehydration."""
    # --- MCP OAuth2 discovery config shape warnings (M-3/QA BUG-003) ---
    _log_mcp_discovery_shape_warnings()

    # --- Alembic migrations: own the mcp_servers table schema ---
    # Runs unconditionally (the schema must exist even if the registry is
    # disabled). Runs in a worker thread because Alembic's online mode drives
    # an async engine via asyncio.run internally.
    try:
        await asyncio.to_thread(_run_alembic_upgrade)
    except Exception:
        logger.exception("Alembic migration failed; continuing startup.")

    # --- MCP registry: build pool, store, rehydrate servers ---
    cipher, _store_unused = await _initialize_mcp_registry(app)

    # --- Dual auth: REST security + MCP middleware (see helper docstring) ---
    _initialize_dual_auth()

    # --- Per-user LLM credentials (see helper docstring) ---
    _initialize_user_llm_reader(cipher)

    async with (
        db_lifespan(app),
        mcp_files_app.lifespan(app),
        mcp_classical_app.lifespan(app),
        mcp_web_app.lifespan(app),
        mcp_memories_app.lifespan(app),
    ):
        yield

    # --- Shutdown ---
    await _close_startup_resources()


app = FastAPI(
    title="RAG Anything API",
    lifespan=combined_lifespan,
)

# Wire the generated MCP runner now that the app exists. The runner holds the
# app reference so it can mount in-process FastMCP servers under
# /generated/{name}/mcp at runtime. Stored on app.state so the lifespan can
# access it, and on the dependencies module so the use-case providers resolve it.
generated_mcp_runner = GeneratedMcpRunner(
    app=app,
    base_url=app_config.GENERATED_MCP_BASE_URL,
    middleware=[mcp_middleware],
    # The generated /generated/{name}/mcp mounts get the SAME HTTP-level
    # guard stack (auth + transport security) as the four static mounts
    # (feat-mcp-oauth2-dual-auth, REQ-017); the FastMCP method-level
    # middleware stays as defense-in-depth inside each generated server.
    guard_factory=build_mcp_guard_stack,
)
app.state.generated_mcp_runner = generated_mcp_runner
import dependencies as _deps  # noqa: E402

_deps.generated_mcp_runner = generated_mcp_runner

app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REST_PATH = "/api/v1"
app.include_router(health_router, prefix=REST_PATH)

# Protected routers use the dual-auth dependency (JWT bearer token OR per-user
# API key). The dependency sets the RLS contextvars on success so downstream
# asyncpg usage (McpRegistryStore) is RLS-scoped. When no auth service is
# wired yet (LOGTO_URL unset / asyncpg pool not available, or before the
# lifespan runs), ``verify_credentials`` falls back to the legacy master-key
# ``verify_api_key`` check on the ``X-API-Key`` header, so the REST routers
# keep working in dev/test and during the startup window.
_auth_dep = [Depends(rest_security.verify_credentials)]
app.include_router(file_router, prefix=REST_PATH, dependencies=_auth_dep)
app.include_router(classical_indexing_router, prefix=REST_PATH, dependencies=_auth_dep)
app.include_router(classical_query_router, prefix=REST_PATH, dependencies=_auth_dep)
# mcp_registry_router already carries the full /api/v1/mcp/servers prefix, so
# it is included without an additional prefix (only the auth dependency).
app.include_router(mcp_registry_router, dependencies=_auth_dep)

# --- Public OAuth2 discovery (REQ-002-equivalent) -----------------------------
#
# A PUBLIC (unauthenticated) route returning the static RFC 9728-style
# protected-resource metadata document. MUST be registered BEFORE the MCP
# mounts so it stays reachable regardless of later mount prefixes (it shares
# no prefix with them — kept adjacent to the mount wiring for the same
# discovery-chain readability). OPTIONS preflights on it are answered by the
# app-wide CORS middleware without auth.


async def _protected_resource_document(
    request: Request, resource_path: str
) -> Response:
    """Derive the public origin and return one protected-resource document.

    Single derivation shared by the global and per-mount documents
    (QA BUG-002): one :func:`public_discovery_origin` derivation, one plain
    404 fallback, one ``authorization_servers`` shape (the RESOLVED
    ``MCP_AUTHORIZATION_SERVER`` — an explicit setting or the
    ``<LOGTO_URL>/oidc`` fallback resolved once at startup by the
    ``AppConfig`` model validator) — the list is EMPTY when neither is
    configured.

    Args:
        request: The incoming request (its ``Host`` header is used only when
            allow-listed by ``MCP_ALLOWED_HOSTS``).
        resource_path: The advertised resource path (the canonical
            ``MCP_DISCOVERY_RESOURCE_PATH`` for the global document, the
            mount path itself for a per-mount document).

    Returns:
        The static RFC 9728-style JSON document, or a plain 404 (no
        derivation detail) when no trustworthy origin can be derived.
    """
    origin = public_discovery_origin(
        request.headers.get("host", ""),
        configured_base_url=app_config.MCP_PUBLIC_BASE_URL,
        allowed_hosts=list(app_config.MCP_ALLOWED_HOSTS),
    )
    if origin is None:
        logger.warning(LogMessage.MCP_DISCOVERY_ORIGIN_UNAVAILABLE)
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND, content={"detail": "Not Found"}
        )
    return JSONResponse(
        content={
            "resource": f"{origin}{resource_path}",
            "authorization_servers": (
                [app_config.MCP_AUTHORIZATION_SERVER]
                if app_config.MCP_AUTHORIZATION_SERVER
                else []
            ),
        }
    )


@app.get(
    WELL_KNOWN_PROTECTED_RESOURCE_PATH,
    summary="OAuth2 protected-resource metadata (public discovery)",
    description=(
        "Unauthenticated RFC 9728-style protected-resource metadata for the "
        "MCP OAuth2 discovery chain: the canonical resource identifier plus "
        "the authorization servers accepting tokens for it. Each MCP mount "
        "also serves its OWN metadata document at "
        "'.well-known/oauth-protected-resource/<mount>' (QA BUG-002)."
    ),
    response_description="The static protected-resource metadata document.",
    include_in_schema=False,
)
async def oauth_protected_resource(request: Request) -> Response:
    """Return the canonical OAuth2 protected-resource metadata (public).

    Args:
        request: The incoming request (its ``Host`` header is used only when
            allow-listed by ``MCP_ALLOWED_HOSTS``).

    Returns:
        ``{"resource": "https://<public-origin>/files/mcp",
        "authorization_servers": [...]}`` — 404 (plain) when no trustworthy
        origin can be derived.
    """
    return await _protected_resource_document(request, MCP_DISCOVERY_RESOURCE_PATH)


@app.get(
    WELL_KNOWN_PROTECTED_RESOURCE_PATH + "/{mount_path:path}",
    summary="Per-mount OAuth2 protected-resource metadata (public discovery)",
    description=(
        "Unauthenticated RFC 9728-style protected-resource metadata for ONE "
        "MCP mount: the document advertises that mount's OWN resource "
        "identifier (QA BUG-002) plus the authorization servers accepting "
        "tokens for it. Served for the four built-in mounts and the "
        "/generated/{name}/mcp pattern."
    ),
    response_description="The mount-specific protected-resource metadata document.",
    include_in_schema=False,
)
async def oauth_protected_resource_for_mount(
    request: Request, mount_path: str
) -> Response:
    """Return the mount-specific OAuth2 protected-resource metadata (public).

    Args:
        request: The incoming request (its ``Host`` header is used only when
            allow-listed by ``MCP_ALLOWED_HOSTS``).
        mount_path: The mount path below the well-known prefix (its leading
            ``/`` is already part of the URL).

    Returns:
        ``{"resource": "https://<public-origin><mount_path>",
        "authorization_servers": [...]}``; a plain 404 when the path is not
        an MCP mount or when no trustworthy origin can be derived.
    """
    if not _is_mcp_mount_path(f"/{mount_path}"):
        # Not a mount (never a resource) → plain 404, like any unknown route.
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND, content={"detail": "Not Found"}
        )
    return await _protected_resource_document(request, f"/{mount_path}")


# --- Protected MCP mounts (feat-mcp-oauth2-dual-auth) --------------------------
#
# PRODUCTION PREREQUISITE (code-review M-4, deliberately DEFERRED to the flux
# wave-4 slice — no repo code change here): for this feature to be ALIVE in
# production, `flux/prd/composables/raganything/configmap.yaml` MUST carry
#   MCP_ALLOWED_HOSTS: "raganything.soludev.tech,127.0.0.1:*,localhost:*,[::1]:*"
#   MCP_PUBLIC_BASE_URL: "https://raganything.soludev.tech"
#   MCP_AUTHORIZATION_SERVER: "https://logto.soludev.tech/oidc"
#   JWT_AUDIENCES: "<logto-opencode-app-id>"
# and `flux/prd/composables/raganything/ingressroute.yaml` MUST add the
# `PathPrefix(/.well-known)` public route (no forwardAuth) and the
# `HeaderRegexp(Authorization, ^Bearer )` direct-backend route (priority 90),
# mirroring the composable-agents slice. Without them every MCP request
# answers 421 (localhost-only default list) and the 401 carries no
# WWW-Authenticate challenge — see the review addendum.
#
# The HTTP-level guard stack (auth 401 + discovery challenge, then explicit
# transport security) authenticates EVERY request — including ``initialize`` —
# BEFORE any MCP method runs (REQ-001/017). The exact-path Route serves
# POST <mount> directly — no Starlette Mount 307 and no ``http://`` scheme
# downgrade behind the TLS-terminating proxy (REQ-015) — while the Mount
# keeps serving "/<mount>/..." sub-paths; the guard rewrites both path forms
# to "/" for the inner app. The ``McpApiKeyMiddleware`` stays mounted INSIDE
# each FastMCP app (above) as defense-in-depth behind the HTTP guard.
for _mount_path, _mcp_app in zip(
    MCP_MOUNT_PATHS,
    (mcp_files_app, mcp_classical_app, mcp_web_app, mcp_memories_app),
    strict=True,
):
    _guarded = build_mcp_guard_stack(_mcp_app, _mount_path)
    app.router.routes.append(
        Route(
            _mount_path,
            _guarded,
            name=f"mcp_{_mount_path.strip('/').replace('/', '_')}_endpoint",
        )
    )
    app.mount(_mount_path, _guarded)

app.add_exception_handler(StorageNotFoundError, storage_not_found_handler)
app.add_exception_handler(StorageError, storage_error_handler)
app.add_exception_handler(UnsupportedFormatError, unsupported_format_handler)
app.add_exception_handler(DocumentReadError, document_error_handler)
app.add_exception_handler(VectorStoreConfigError, vector_bad_request_handler)
app.add_exception_handler(VectorStoreError, vector_store_error_handler)
app.add_exception_handler(ClassicalConfigError, classical_error_handler)
app.add_exception_handler(IndexingError, indexing_error_handler)
app.add_exception_handler(FileValidationError, file_validation_handler)
app.add_exception_handler(FileTooLargeError, file_too_large_handler)
app.add_exception_handler(FileError, file_error_handler)
app.add_exception_handler(
    DependencyNotInitializedError, dependency_not_initialized_handler
)
app.add_exception_handler(ConfigError, config_error_handler)
app.add_exception_handler(InvalidApiKeyError, invalid_api_key_handler)
app.add_exception_handler(AuthenticationError, authentication_error_handler)
app.add_exception_handler(LlmNotConfiguredError, llm_not_configured_handler)
app.add_exception_handler(
    McpServerInvalidAuthModeError, mcp_invalid_stored_config_handler
)
app.add_exception_handler(DomainError, domain_error_handler)


def run_fastapi():
    """Run the already-built FastAPI application with uvicorn.

    Serves the in-process ``app`` object (not the import string
    ``"main:app"``): the container entrypoint already executes this module as
    ``__main__``, and an import string would make uvicorn import and execute
    ``main`` a SECOND time in the same process. That second execution stacks
    another unwired ``McpApiKeyMiddleware`` on the shared FastMCP servers
    (``FastMCP.add_middleware`` only appends), so an unwired guard ends up in
    front of the wired one and — once the legacy master key is non-empty —
    rejects every legitimate credential (BUG-005). Passing the object keeps
    exactly one application import, one middleware per mount, and one lifespan
    wiring it.
    """
    uvicorn.run(
        app,
        host=app_config.HOST,
        port=app_config.PORT,
        ws="none",
    )


if __name__ == "__main__":
    run_fastapi()
