"""Dependency injection — module-level singletons following the pickpro pattern.

Per-user LLM credentials (chat + embeddings) are resolved at request time from
the ``current_user_id`` RLS contextvar (set by ``verify_credentials`` after a
successful dual-auth). When the contextvar is set, the factories
``get_embedding_for_user`` / ``get_chat_llm_for_user`` / ``get_vector_store_for_user``
resolve the user's ``(base_url, api_key)`` from the shared ``user_llm_settings``
table (read-only, owned by composable-agents) via the
:class:`UserLlmCredentialResolver`. When the user has no configured provider,
the factories raise :class:`LlmNotConfiguredError` (422). When the contextvar
is None (legacy/tests, no auth wired), the factories fall back to the
:class:`LLMConfig` env-based credentials — the original behaviour — so all
existing tests keep passing.
"""

import asyncio
import logging
import os
import time

from pydantic import SecretStr

from application.use_cases.classical_index_file_use_case import (
    ClassicalIndexFileUseCase,
)
from application.use_cases.classical_index_folder_use_case import (
    ClassicalIndexFolderUseCase,
)
from application.use_cases.classical_query_use_case import ClassicalQueryUseCase
from application.use_cases.create_folder_use_case import CreateFolderUseCase
from application.use_cases.create_mcp_server import CreateMcpServerUseCase
from application.use_cases.create_rag_space import CreateRagSpaceUseCase
from application.use_cases.delete_file_use_case import DeleteFileUseCase
from application.use_cases.delete_folder_use_case import DeleteFolderUseCase
from application.use_cases.delete_mcp_server import DeleteMcpServerUseCase
from application.use_cases.delete_rag_space import DeleteRagSpaceUseCase
from application.use_cases.ensure_rag_space import EnsureRagSpaceUseCase
from application.use_cases.get_mcp_server import GetMcpServerUseCase
from application.use_cases.list_files_use_case import ListFilesUseCase
from application.use_cases.list_folders_use_case import ListFoldersUseCase
from application.use_cases.list_mcp_servers import ListMcpServersUseCase
from application.use_cases.list_rag_spaces import ListRagSpacesUseCase
from application.use_cases.liveness_check_use_case import LivenessCheckUseCase
from application.use_cases.read_file_use_case import ReadFileUseCase
from application.use_cases.update_mcp_server import UpdateMcpServerUseCase
from application.use_cases.upload_file_use_case import UploadFileUseCase
from application.use_cases.validate_mcp_server import ValidateMcpServerUseCase
from application.use_cases.web_search_use_case import WebSearchUseCase
from config import (
    AppConfig,
    BM25Config,
    ClassicalRAGConfig,
    CrawlConfig,
    DatabaseConfig,
    LLMConfig,
    MinioConfig,
    WebSearchConfig,
)
from domain.constants import PERSONAL_AGENT_SLUG
from domain.entities.agent_scope import AgentScope
from domain.entities.auth.auth_context import AuthContext
from domain.errors.base import DomainError
from domain.errors.config import DependencyNotInitializedError
from domain.errors.llm import LlmNotConfiguredError
from domain.errors.messages import ErrorMessage
from domain.errors.scope import AgentScopeInvalidError
from domain.logging.messages import LogMessage
from domain.ports.auth.api_key_repository import ApiKeyRepositoryPort
from domain.ports.auth.jwt_service import JwtServicePort
from domain.ports.bm25_engine import BM25EnginePort
from domain.ports.generated_mcp_runner import GeneratedMcpRunnerPort
from domain.ports.llm_port import LLMPort
from domain.ports.mcp_server_registry_store import McpServerRegistryStore
from domain.ports.mcp_tool_loader import McpToolLoader
from domain.ports.openapi_mcp_factory import OpenApiMcpFactory
from domain.ports.rag_space_repository import RagSpaceRepository
from domain.ports.user_llm_settings_repository import UserLlmSettingsRepositoryPort
from domain.ports.vector_store_port import VectorStorePort
from domain.ports.vector_table_manager import VectorTableManagerPort
from domain.services.auth.auth_service import AuthService
from domain.services.llm.credential_resolver import UserLlmCredentialResolver
from domain.services.scope import resolve_agent_scope
from infrastructure.database.asyncpg_health_adapter import AsyncpgHealthAdapter
from infrastructure.database.rls_context import (
    current_agent_scope,
    current_auth_method,
    current_credential,
    current_user_id,
)
from infrastructure.document_reader.xberg_adapter import XbergAdapter
from infrastructure.mcp.tool_loader import FastMcpToolLoader
from infrastructure.openapi_mcp.adapter import FastMcpOpenApiFactory
from infrastructure.rag.classical_bm25_adapter import ClassicalBM25Adapter
from infrastructure.security.crypto import FernetSecretCipher
from infrastructure.storage.minio_adapter import MinioAdapter
from infrastructure.web_search.crawl4ai_adapter import Crawl4AIWebCrawlerAdapter
from infrastructure.web_search.ddgs_adapter import DDGSSearchAdapter
from infrastructure.web_search.url_safety.adapter import UrlSafetyAdapter

logger = logging.getLogger(__name__)

app_config = AppConfig()  # type: ignore
llm_config = LLMConfig()  # type: ignore
minio_config = MinioConfig()  # type: ignore
bm25_config = BM25Config()  # type: ignore
db_config = DatabaseConfig()  # type: ignore

os.makedirs(app_config.OUTPUT_DIR, exist_ok=True)

minio_adapter = MinioAdapter(
    host=minio_config.MINIO_HOST,
    access=minio_config.MINIO_ACCESS,
    secret=minio_config.MINIO_SECRET,
    secure=minio_config.MINIO_SECURE,
)

xberg_adapter = XbergAdapter()
postgres_health_adapter = AsyncpgHealthAdapter(db_config)

classical_rag_config = ClassicalRAGConfig()

# --- Web search tools (bounded process-local adapters) ------------------------
#
# These adapters are process-wide singletons, but they are built LAZILY on
# first use instead of at import time: their bounds (timeouts, concurrency,
# worker counts) are read from validated settings, and deferring construction
# — including the settings objects themselves — guarantees those settings were
# resolved in the environment the application actually starts in. The DDGS
# adapter owns its dedicated executor + semaphore (shut down at application
# shutdown), the crawl adapter lazily starts the shared crawler on first crawl
# (never at import/startup).
url_safety_adapter = UrlSafetyAdapter()

web_search_adapter: DDGSSearchAdapter | None = None
web_crawler_adapter: Crawl4AIWebCrawlerAdapter | None = None


def _build_web_search_adapter() -> DDGSSearchAdapter:
    """Create the web-search adapter from validated settings (lazy singleton)."""
    config = WebSearchConfig()  # type: ignore
    return DDGSSearchAdapter(
        timeout_seconds=config.WEB_SEARCH_TIMEOUT_SECONDS,
        max_concurrency=config.WEB_SEARCH_MAX_CONCURRENCY,
        executor_workers=config.WEB_SEARCH_EXECUTOR_WORKERS,
        backends=config.WEB_SEARCH_BACKENDS,
    )


def _build_web_crawler_adapter() -> Crawl4AIWebCrawlerAdapter:
    """Create the crawl adapter from validated settings (lazy singleton)."""
    config = CrawlConfig()  # type: ignore
    # The page cap is a search-output bound owned by WebSearchConfig; the
    # crawler adapter enforces it when building each successful outcome.
    search_config = WebSearchConfig()  # type: ignore
    return Crawl4AIWebCrawlerAdapter(
        max_concurrent_batches=config.CRAWL_MAX_CONCURRENT_BATCHES,
        max_sessions_per_batch=config.CRAWL_MAX_SESSIONS_PER_BATCH,
        memory_threshold_percent=config.CRAWL_MEMORY_THRESHOLD_PERCENT,
        page_timeout_ms=round(config.CRAWL_PAGE_TIMEOUT_SECONDS * 1000),
        overall_timeout_seconds=config.CRAWL_OVERALL_TIMEOUT_SECONDS,
        rate_limit_min_delay_seconds=config.CRAWL_RATE_LIMIT_MIN_DELAY_SECONDS,
        rate_limit_max_delay_seconds=config.CRAWL_RATE_LIMIT_MAX_DELAY_SECONDS,
        url_safety=url_safety_adapter,
        max_page_chars=search_config.WEB_SEARCH_PAGE_MAX_CHARS,
    )


def get_web_search_adapter() -> DDGSSearchAdapter:
    """Return the web-search adapter singleton (built on first use)."""
    global web_search_adapter
    if web_search_adapter is None:
        web_search_adapter = _build_web_search_adapter()
    return web_search_adapter


def get_web_crawler_adapter() -> Crawl4AIWebCrawlerAdapter:
    """Return the crawl adapter singleton (built on first use)."""
    global web_crawler_adapter
    if web_crawler_adapter is None:
        web_crawler_adapter = _build_web_crawler_adapter()
    return web_crawler_adapter


def get_web_search_use_case() -> WebSearchUseCase:
    """Return the web-search use case (ports wired to their adapters)."""
    return WebSearchUseCase(
        search=get_web_search_adapter(),
        crawler=get_web_crawler_adapter(),
        url_safety=url_safety_adapter,
    )


def get_request_auth_context() -> AuthContext | None:
    """Build the originating request's :class:`AuthContext` from the RLS
    contextvars (feat-mcp-oauth2 REQ-004).

    The dual-auth ``verify_credentials`` dependency sets
    ``current_credential`` / ``current_auth_method`` after a successful
    authentication; this provider re-projects them into a plain domain
    entity so the ``user_credential_headers`` rule (auth_mode
    ``"user_token"``) can inject the caller's credential at load/validate
    time.

    Returns:
        The authenticated context, or ``None`` when the request carries no
        dual-auth credential (unset contextvars — legacy master-key path,
        dev/test, or unattended runs): the injection rule then returns an
        empty mapping (drop-empty semantics).
    """
    credential = current_credential.get()
    method = current_auth_method.get()
    if not credential:
        return None
    if method == "jwt" or method == "api_key":
        return AuthContext(
            user_id=current_user_id.get() or "",
            method=method,
            raw_credential=credential,
        )
    return None


# --- Caller scope (user × agent) + RAG spaces (scope-rag-memories) ------------
#
# The RAW ``X-Composable-Agent`` header value travels on the
# ``current_agent_scope`` RLS contextvar (set by ``verify_credentials`` / the
# MCP guard); the scope resolution below validates it at the FIRST scoped
# storage operation (REQ-002) and composes every key/namespace/memory path
# (REQ-003/006/012). The space REPOSITORY is wired lazily by ``main.lifespan``
# (it needs the asyncpg pool).


rag_space_repository: RagSpaceRepository | None = None


def get_request_agent_scope() -> AgentScope:
    """Resolve the request's caller scope (REQ-001) and validate the slug.

    Reads ``current_user_id`` and the RAW ``current_agent_scope`` contextvar;
    the user id is NEVER derived from a client-supplied header or parameter
    (REQ-002) — the slug header only contributes ``agent_slug``.

    Returns:
        The :class:`AgentScope` (empty ``user_id`` in the legacy
        unauthenticated path; ``personal`` when no header is present).

    Raises:
        AgentScopeInvalidError: When the raw header value is invalid
            (mapped to 422 on REST / ``ToolError`` on MCP — before any
            storage operation).
    """
    return resolve_agent_scope(current_user_id.get(), current_agent_scope.get())


def get_required_agent_scope() -> AgentScope:
    """Resolve the caller scope REQUIRING an agent header (user × agent).

    Data-scoped surfaces (agent-facing files / classical tools and routes)
    must never fall through to the personal namespace: every operation is
    bound to an explicit ``X-Composable-Agent`` scope (user directive — the
    personal namespace stays a system-internal fallback only).

    Returns:
        The immutable :class:`AgentScope` for the request.

    Raises:
        AgentScopeInvalidError: When the header is absent/empty (or its value
            invalid) — mapped to 422 / ``ToolError`` before any storage op.
    """
    raw = current_agent_scope.get()
    if raw is None or not raw.strip():
        raise AgentScopeInvalidError(
            ErrorMessage.AGENT_SCOPE_REQUIRED.format(header="X-Composable-Agent")
        )
    return resolve_agent_scope(current_user_id.get(), raw)


# --- Background space-touch state (declared BEFORE its users, code-review
# --- 🟢 iteration-4: module-level names resolve at call time, but keeping the
# --- mutable tracking state next to the functions that mutate it keeps the
# --- coalescing design auditable in one place).
_space_touch_inflight: set[tuple[str, str]] = set()
_space_touch_tasks: set[asyncio.Task[None]] = set()
_seen_space_working_dirs: set[tuple[str, str, str]] = set()
#: Refresh/seen-touches are throttled to at most ONE scheduled write per space
#: per cooldown window (perf NFR: the ``last_used_at`` update is COALESCED).
_SPACE_TOUCH_COOLDOWN_SECONDS = 60.0
_last_space_touch: dict[tuple[str, str], float] = {}

_SPACE_TOUCH_DRAIN_TIMEOUT_SECONDS = 5.0


def _request_scope_for_tracking() -> "AgentScope | None":
    """The scope to book-keep, or ``None`` when tracking must be skipped.

    Tracking is bookkeeping and can never break a request: an unresolvable
    scope (already rejected 422/ToolError by the operation itself) and every
    non-bookkeeping scope (no authenticated user, ``personal``) yield ``None``.
    """
    try:
        scope = get_request_agent_scope()
    except DomainError:
        return None
    if not scope.user_id or scope.agent_slug == PERSONAL_AGENT_SLUG:
        return None
    return scope


def _coalesce_space_touch(key: tuple[str, str], working_dir: str | None) -> bool:
    """Whether THIS touch must schedule a record write (or is absorbed).

    A NEW (never-seen) ``working_dir`` is NEVER absorbed — REQ-008 tracking
    must be lossless, so its write is ALWAYS scheduled, EVEN while another
    write for the same space is in flight (the UPSERT merges the directory
    atomically and idempotently; an overlapping refresh write is harmless).
    This is the code-review C6 fix: the predicate is matched to its docstring.

    An already-seen ``working_dir`` (and every pure refresh) is coalesced:
    absorbed behind the in-flight write, otherwise throttled to at most ONE
    scheduled write per space per ``_SPACE_TOUCH_COOLDOWN_SECONDS`` window.
    """
    if working_dir is not None:
        marker = key + (working_dir,)
        is_new_working_dir = marker not in _seen_space_working_dirs
        _seen_space_working_dirs.add(marker)
        if is_new_working_dir:
            # REQ-008: the tracking write for a NEW namespace is never
            # coalesced away — the cascade delete must be able to drop it.
            _space_touch_inflight.add(key)
            return True

    if key in _space_touch_inflight:
        return False
    now = time.monotonic()
    if now - _last_space_touch.get(key, 0.0) < _SPACE_TOUCH_COOLDOWN_SECONDS:
        return False
    _last_space_touch[key] = now
    _space_touch_inflight.add(key)
    return True


def track_space_usage(working_dir: str | None = None) -> None:
    """Schedule a best-effort, COALESCED space write (REQ-011 + perf NFR).

    Synchronous by design (Sonar python:S7503): the function is a scheduler
    — it only resolves the in-memory scope state and hands a background task
    to the event loop, zero I/O on the request's critical path — so awaiting
    a no-op coroutine would add protocol overhead without any async benefit.
    The actual record write happens off-request in
    :func:`_execute_space_touch`, mirroring the "no additional query on
    file/RAG hot paths; the lazy space-record write is fire-and-forget with
    the ``last_used_at`` update coalesced or best-effort" non-functional
    requirement. A failed record write never blocks nor breaks the request.

    No repository wired (dev/test) → no-op. No authenticated user, or the
    ``personal`` scope → no-op (the personal tree carries no space record).
    """
    repository = rag_space_repository
    if repository is None:
        return
    scope = _request_scope_for_tracking()
    if scope is None:
        return
    key = (scope.user_id, scope.agent_slug)
    if not _coalesce_space_touch(key, working_dir):
        return
    task = asyncio.create_task(_execute_space_touch(repository, key, working_dir))
    _space_touch_tasks.add(task)
    task.add_done_callback(_space_touch_tasks.discard)


async def _execute_space_touch(
    repository: RagSpaceRepository,
    key: tuple[str, str],
    working_dir: str | None,
) -> None:
    """Run one background space write for ``(user_id, agent_slug)``.

    Best-effort by contract: the :class:`EnsureRagSpaceUseCase` already
    swallows record-write failures with a warning (REQ-011); the defensive
    catch keeps an unexpected scheduling failure from ever reaching the event
    loop's crash handler. The in-flight marker is always released so later
    uses are coalesced, never starved.

    When the record write could not be persisted (the ensure use case yielded
    no record), the seen-``working_dir`` markers of that space are dropped so
    a later scoped use retries the lazy provision (REQ-011's "eventually")
    instead of being coalesced away forever.
    """
    user_id, agent_slug = key
    try:
        space = await EnsureRagSpaceUseCase(repository).execute(
            user_id=user_id, agent_slug=agent_slug, working_dir=working_dir
        )
    except Exception as exc:  # noqa: BLE001 — fire-and-forget by contract
        logger.warning(LogMessage.SPACE_PROVISIONING_FAILED, agent_slug, str(exc))
        space = None
    finally:
        _space_touch_inflight.discard(key)
    if space is None:
        _forget_space_working_dirs(key)


def _forget_space_working_dirs(key: tuple[str, str]) -> None:
    """Drop the seen-``working_dir`` markers of ONE space (bounded size).

    Markers are in-memory only; they are dropped when a space is deleted
    (:func:`forget_space_tracking` — a recreated space re-records its
    namespaces from zero) or when a write failed (retry on the next scoped
    use), keeping the set proportional to the LIVE spaces' ``working_dirs``.
    """
    for tracked in [
        tracked for tracked in _seen_space_working_dirs if tracked[:2] == key
    ]:
        _seen_space_working_dirs.discard(tracked)


def forget_space_tracking(user_id: str, agent_slug: str) -> None:
    """Forget every in-memory tracking marker of a deleted RAG space.

    Called by the cascade-delete route (code-review iteration-4 🟡-4): the
    seen-markers set stays bounded to the LIVE spaces instead of growing
    forever with deleted-and-recreated ones.
    """
    _forget_space_working_dirs((user_id or "", agent_slug or ""))


async def drain_space_touch_tasks() -> None:
    """Await the pending background space writes (best-effort, bounded).

    Called from the application shutdown hook BEFORE the asyncpg pool is
    closed (code-review iteration-4 🟡-3): without the drain a pending
    provisioning write would hit the closed pool (silently swallowed) and
    asyncio would emit "Task was destroyed but it is pending!". Per Sonar
    python:S7483 an ASYNC function must not accept a timeout parameter —
    the caller owns the timeout policy through a timeout context manager —
    so the drain bounds itself with :data:`_SPACE_TOUCH_DRAIN_TIMEOUT_SECONDS`
    via :func:`asyncio.timeout`; tasks still pending after the timeout are
    cancelled so teardown always completes.
    """
    pending = [task for task in _space_touch_tasks if not task.done()]
    if not pending:
        return
    try:
        async with asyncio.timeout(_SPACE_TOUCH_DRAIN_TIMEOUT_SECONDS):
            await asyncio.wait(pending)
    except TimeoutError:
        pass  # bounded best-effort drain: stragglers are cancelled below
    for task in pending:
        if not task.done():
            task.cancel()


def get_create_rag_space_use_case() -> CreateRagSpaceUseCase:
    """Return the idempotent space-create use case (repository required)."""
    return CreateRagSpaceUseCase(repository=_required_rag_space_repository())


def get_list_rag_spaces_use_case() -> ListRagSpacesUseCase:
    """Return the space-list use case (repository required)."""
    return ListRagSpacesUseCase(repository=_required_rag_space_repository())


def get_delete_rag_space_use_case() -> DeleteRagSpaceUseCase:
    """Return the space-cascade-delete use case (MinIO + tables + record)."""
    return DeleteRagSpaceUseCase(
        repository=_required_rag_space_repository(),
        storage=minio_adapter,
        bucket=minio_config.MINIO_BUCKET,
        vector_tables=get_vector_table_manager(),
    )


def _required_rag_space_repository() -> RagSpaceRepository:
    """Return the space repository or fail with the 503 mapping error."""
    if rag_space_repository is None:
        raise DependencyNotInitializedError(
            "RAG space repository is not initialized; the application lifespan "
            "must create the asyncpg pool before serving requests."
        )
    return rag_space_repository


vector_table_manager: VectorTableManagerPort | None = None


def get_vector_table_manager() -> VectorTableManagerPort:
    """Return the drop-capable vector adapter singleton (code-review 🟡-3).

    A fresh adapter per delete request would lazily build (and never close) a
    new SQLAlchemy ``PGEngine`` pool — a leaked pool per cascade delete. The
    manager is therefore cached at module level and closed by the application
    shutdown hook (``main.lifespan`` → ``db_lifespan``). It reuses the legacy
    singleton's shared ``PGEngine`` when available (same pooling rationale as
    ``get_vector_store_for_user``); its own engine — when it has to build one
    — is lazily resolved inside the adapter on first drop and released at
    shutdown.
    """
    global vector_table_manager
    if vector_table_manager is not None:
        return vector_table_manager
    from infrastructure.vector_store.langchain_pgvector_adapter import (
        LangchainPgvectorAdapter,
    )

    shared_engine = (
        getattr(classical_vector_store, "_engine", None)
        if classical_vector_store is not None
        else None
    )
    vector_table_manager = LangchainPgvectorAdapter(
        connection_string=db_config.DATABASE_URL,
        table_prefix=classical_rag_config.CLASSICAL_TABLE_PREFIX,
        embedding_dimension=llm_config.EMBEDDING_DIM,
        embedding_service=None,
        engine=shared_engine,
    )
    return vector_table_manager


classical_bm25_adapter: BM25EnginePort | None = None
if bm25_config.BM25_ENABLED:
    try:
        classical_bm25_adapter = ClassicalBM25Adapter(
            db_url=db_config.asyncpg_url,
            table_prefix=classical_rag_config.CLASSICAL_TABLE_PREFIX,
            text_config=bm25_config.BM25_TEXT_CONFIG,
        )
    except Exception as e:
        logger.warning("Classical BM25 adapter initialization failed: %s", e)
        classical_bm25_adapter = None

# --- Classical RAG: legacy singletons (env-based, used when no auth wired) ---
#
# These are kept as fallbacks for the legacy/tests path (``current_user_id``
# is None). The per-request factories below (``get_embedding_for_user`` etc.)
# use them when the RLS contextvar is not set, so existing tests and the dev
# path keep working unchanged. When the contextvar IS set, the factories
# resolve per-user credentials and build fresh per-user instances.
classical_vector_store: VectorStorePort | None = None
_legacy_embedding = None
_legacy_llm: LLMPort | None = None
try:
    from langchain_openai import OpenAIEmbeddings

    _legacy_embedding = OpenAIEmbeddings(
        model=llm_config.EMBEDDING_MODEL,
        api_key=SecretStr(llm_config.api_key),
        base_url=llm_config.api_base_url,
    )
    from infrastructure.vector_store.langchain_pgvector_adapter import (
        LangchainPgvectorAdapter,
    )

    classical_vector_store = LangchainPgvectorAdapter(
        connection_string=db_config.DATABASE_URL,
        table_prefix=classical_rag_config.CLASSICAL_TABLE_PREFIX,
        embedding_dimension=llm_config.EMBEDDING_DIM,
        embedding_service=_legacy_embedding,
    )
    from infrastructure.llm.langchain_openai_adapter import LangchainOpenAIAdapter

    _legacy_llm = LangchainOpenAIAdapter(
        api_key=llm_config.api_key,
        base_url=llm_config.api_base_url,
        model=llm_config.CHAT_MODEL,
        temperature=classical_rag_config.CLASSICAL_LLM_TEMPERATURE,
    )
except Exception as e:
    logger.warning("Classical RAG adapter initialization failed: %s", e)


# --- Per-user LLM credential resolution ----------------------------------------
#
# The reader and resolver are wired lazily because they depend on the asyncpg
# pool (created in ``main.lifespan``). ``main.py`` assigns ``user_llm_reader``
# once the pool exists, then the resolver is built on first access.
user_llm_reader: UserLlmSettingsRepositoryPort | None = None


def get_user_llm_reader() -> UserLlmSettingsRepositoryPort:
    """Return the user LLM settings reader singleton (wired by ``main.lifespan``).

    Raises:
        DependencyNotInitializedError: If the reader has not been wired yet.
    """
    if user_llm_reader is None:
        raise DependencyNotInitializedError(
            "User LLM settings reader is not initialized; the application "
            "lifespan must create the asyncpg pool before serving requests."
        )
    return user_llm_reader


def get_user_llm_resolver() -> UserLlmCredentialResolver:
    """Return the per-user LLM credential resolver (built from the reader)."""
    return UserLlmCredentialResolver(repo=get_user_llm_reader())


# --- Per-user LLM/embeddings/vector-store factories ---------------------------
#
# These async factories replace the module-level singletons for the per-user
# path. They read ``current_user_id`` from the RLS contextvar:
# - set + user has settings → build per-user instance with the user's creds;
# - set + no settings → raise ``LlmNotConfiguredError`` (422);
# - None → fall back to the legacy env-based singletons (tests/dev).


async def get_embedding_for_user():
    """Build an ``OpenAIEmbeddings`` for the current user (or legacy fallback).

    Returns:
        An ``OpenAIEmbeddings`` instance bound to the user's credentials or
        the env-based config.

    Raises:
        LlmNotConfiguredError: If the user is authenticated but has no
            configured LLM provider.
        DependencyNotInitializedError: If the legacy fallback is needed but
            the env-based embedding singleton was not built at import time.
    """
    from langchain_openai import OpenAIEmbeddings

    user_id = current_user_id.get()
    if user_id is None:
        if _legacy_embedding is None:
            raise DependencyNotInitializedError(
                ErrorMessage.CLASSICAL_RAG_UNAVAILABLE_VECTOR
            )
        return _legacy_embedding

    resolver = get_user_llm_resolver()
    creds = await resolver.resolve(user_id)
    if creds is None:
        raise LlmNotConfiguredError(ErrorMessage.LLM_NOT_CONFIGURED)
    base_url, api_key = creds
    return OpenAIEmbeddings(
        model=llm_config.EMBEDDING_MODEL,
        api_key=SecretStr(api_key),
        base_url=base_url,
    )


async def get_chat_llm_for_user() -> LLMPort:
    """Build a ``LangchainOpenAIAdapter`` for the current user (or legacy fallback).

    Returns:
        A ``LangchainOpenAIAdapter`` wrapping ``ChatOpenAI`` bound to the
        user's credentials or the env-based config.

    Raises:
        LlmNotConfiguredError: If the user is authenticated but has no
            configured LLM provider.
        DependencyNotInitializedError: If the legacy fallback is needed but
            the env-based LLM singleton was not built at import time.
    """
    from infrastructure.llm.langchain_openai_adapter import LangchainOpenAIAdapter

    user_id = current_user_id.get()
    if user_id is None:
        if _legacy_llm is None:
            raise DependencyNotInitializedError(
                ErrorMessage.CLASSICAL_RAG_UNAVAILABLE_VECTOR_LLM
            )
        return _legacy_llm

    resolver = get_user_llm_resolver()
    creds = await resolver.resolve(user_id)
    if creds is None:
        raise LlmNotConfiguredError(ErrorMessage.LLM_NOT_CONFIGURED)
    base_url, api_key = creds
    return LangchainOpenAIAdapter(
        api_key=api_key,
        base_url=base_url,
        model=llm_config.CHAT_MODEL,
        temperature=classical_rag_config.CLASSICAL_LLM_TEMPERATURE,
    )


async def get_vector_store_for_user() -> VectorStorePort:
    """Build a ``LangchainPgvectorAdapter`` for the current user (or legacy fallback).

    The per-user adapter reuses the legacy singleton's ``PGEngine`` connection
    pool (so we don't create a pool per request) but binds a per-user
    ``OpenAIEmbeddings`` to the ``PGVectorStore`` created for each working dir.

    Returns:
        A ``LangchainPgvectorAdapter`` bound to the user's embeddings or the
        env-based config.

    Raises:
        LlmNotConfiguredError: If the user is authenticated but has no
            configured LLM provider.
        DependencyNotInitializedError: If the legacy fallback is needed but
            the env-based vector store singleton was not built at import time.
    """
    from infrastructure.vector_store.langchain_pgvector_adapter import (
        LangchainPgvectorAdapter,
    )

    user_id = current_user_id.get()
    if user_id is None:
        if classical_vector_store is None:
            raise DependencyNotInitializedError(
                ErrorMessage.CLASSICAL_RAG_UNAVAILABLE_VECTOR
            )
        return classical_vector_store

    # Per-user path: reuse the shared PGEngine from the legacy singleton so we
    # don't create a connection pool per request. The embeddings are per-user.
    embedding = await get_embedding_for_user()
    # ``_engine`` is the legacy adapter's internal shared PGEngine; the port
    # does not declare it, so access it defensively (None falls back to a new
    # engine built inside the adapter).
    shared_engine = (
        getattr(classical_vector_store, "_engine", None)
        if classical_vector_store is not None
        else None
    )
    return LangchainPgvectorAdapter(
        connection_string=db_config.DATABASE_URL,
        table_prefix=classical_rag_config.CLASSICAL_TABLE_PREFIX,
        embedding_dimension=llm_config.EMBEDDING_DIM,
        embedding_service=embedding,
        engine=shared_engine,
    )


async def get_classical_index_file_use_case() -> ClassicalIndexFileUseCase:
    vector_store = await get_vector_store_for_user()
    return ClassicalIndexFileUseCase(
        vector_store=vector_store,
        document_reader=xberg_adapter,
        storage=minio_adapter,
        bucket=minio_config.MINIO_BUCKET,
        output_dir=app_config.OUTPUT_DIR,
        space_recorder=track_space_usage,
    )


async def get_classical_index_folder_use_case() -> ClassicalIndexFolderUseCase:
    vector_store = await get_vector_store_for_user()
    return ClassicalIndexFolderUseCase(
        vector_store=vector_store,
        document_reader=xberg_adapter,
        storage=minio_adapter,
        bucket=minio_config.MINIO_BUCKET,
        output_dir=app_config.OUTPUT_DIR,
        space_recorder=track_space_usage,
    )


async def get_classical_query_use_case() -> ClassicalQueryUseCase:
    vector_store = await get_vector_store_for_user()
    llm = await get_chat_llm_for_user()
    return ClassicalQueryUseCase(
        vector_store=vector_store,
        llm=llm,
        config=classical_rag_config,
        bm25_engine=classical_bm25_adapter,
        rrf_k=classical_rag_config.CLASSICAL_RRF_K,
        space_recorder=track_space_usage,
    )


def get_list_files_use_case() -> ListFilesUseCase:
    return ListFilesUseCase(storage=minio_adapter, bucket=minio_config.MINIO_BUCKET)


def get_list_folders_use_case() -> ListFoldersUseCase:
    return ListFoldersUseCase(storage=minio_adapter, bucket=minio_config.MINIO_BUCKET)


def get_read_file_use_case() -> ReadFileUseCase:
    return ReadFileUseCase(
        storage=minio_adapter,
        document_reader=xberg_adapter,
        bucket=minio_config.MINIO_BUCKET,
        output_dir=app_config.OUTPUT_DIR,
    )


def get_upload_file_use_case() -> UploadFileUseCase:
    return UploadFileUseCase(storage=minio_adapter, bucket=minio_config.MINIO_BUCKET)


def get_create_folder_use_case() -> CreateFolderUseCase:
    return CreateFolderUseCase(storage=minio_adapter, bucket=minio_config.MINIO_BUCKET)


async def get_delete_file_use_case() -> DeleteFileUseCase:
    """Build the delete-file use case with an OPTIONAL per-user vector store.

    The vector store is only used to cascade-delete pgvector chunks for the
    file. When the authenticated user has no configured LLM provider
    (``user_llm_settings`` row absent), the per-user vector store cannot be
    built (it needs the user's embeddings). In that case we pass
    ``vector_store=None`` so the MinIO delete still proceeds — the pgvector
    cascade is skipped (the chunks, if any, are orphaned, but the user can
    still manage their files). This avoids a 422 on a simple MinIO delete for
    users without LLM credentials.
    """
    vector_store = None
    try:
        vector_store = await get_vector_store_for_user()
    except LlmNotConfiguredError:
        logger.warning(
            "Per-user vector store unavailable for delete-file (no LLM "
            "configured); pgvector cascade will be skipped."
        )
    return DeleteFileUseCase(
        storage=minio_adapter,
        bucket=minio_config.MINIO_BUCKET,
        vector_store=vector_store,
    )


async def get_delete_folder_use_case() -> DeleteFolderUseCase:
    """Build the delete-folder use case with an OPTIONAL per-user vector store.

    See ``get_delete_file_use_case`` for the rationale (cascade-delete is
    best-effort; MinIO delete must still work without LLM credentials).
    """
    vector_store = None
    try:
        vector_store = await get_vector_store_for_user()
    except LlmNotConfiguredError:
        logger.warning(
            "Per-user vector store unavailable for delete-folder (no LLM "
            "configured); pgvector cascade will be skipped."
        )
    return DeleteFolderUseCase(
        storage=minio_adapter,
        bucket=minio_config.MINIO_BUCKET,
        vector_store=vector_store,
    )


def get_liveness_check_use_case() -> LivenessCheckUseCase:
    return LivenessCheckUseCase(
        storage=minio_adapter,
        postgres_health=postgres_health_adapter,
        bucket=minio_config.MINIO_BUCKET,
    )


# --- MCP server registry -----------------------------------------------------
#
# The store and runner are wired lazily because they depend on resources that
# are only available after the application starts:
# - the store needs an asyncpg pool (created in ``main.lifespan``);
# - the runner needs the FastAPI ``app`` instance (created in ``main`` after
#   importing this module).
# ``main.py`` assigns these module-level singletons once the resources exist.
# The cipher is created lazily from ``SECRET_ENCRYPTION_KEY`` so importing this
# module never fails when the key is unset (tests/dev override the providers).

mcp_openapi_factory: OpenApiMcpFactory = FastMcpOpenApiFactory(
    max_spec_bytes=app_config.OPENAPI_MAX_SPEC_BYTES
)
mcp_secret_cipher: FernetSecretCipher | None = None
mcp_registry_store: McpServerRegistryStore | None = None
generated_mcp_runner: GeneratedMcpRunnerPort | None = None


def get_mcp_secret_cipher() -> FernetSecretCipher:
    """Return the Fernet secret cipher singleton.

    Created lazily from ``SECRET_ENCRYPTION_KEY``. Raises if the key is unset.

    Returns:
        The FernetSecretCipher instance.

    Raises:
        DependencyNotInitializedError: If ``SECRET_ENCRYPTION_KEY`` is empty.
    """
    global mcp_secret_cipher
    if mcp_secret_cipher is None:
        if not app_config.SECRET_ENCRYPTION_KEY:
            raise DependencyNotInitializedError(
                "SECRET_ENCRYPTION_KEY is not set; MCP registry secrets cannot "
                "be encrypted. Set SECRET_ENCRYPTION_KEY in the environment."
            )
        mcp_secret_cipher = FernetSecretCipher(app_config.SECRET_ENCRYPTION_KEY)
    return mcp_secret_cipher


def get_mcp_openapi_factory() -> OpenApiMcpFactory:
    """Return the OpenAPI MCP factory singleton."""
    return mcp_openapi_factory


def get_mcp_registry_store() -> McpServerRegistryStore:
    """Return the MCP registry store singleton.

    Returns:
        The McpRegistryStore instance (wired by ``main.lifespan``).

    Raises:
        DependencyNotInitializedError: If the store has not been wired yet.
    """
    if mcp_registry_store is None:
        raise DependencyNotInitializedError(
            "MCP registry store is not initialized; the application lifespan "
            "must create the asyncpg pool before serving requests."
        )
    return mcp_registry_store


def get_generated_mcp_runner() -> GeneratedMcpRunnerPort:
    """Return the generated MCP runner singleton.

    Returns:
        The GeneratedMcpRunner instance (wired by ``main`` after app creation).

    Raises:
        DependencyNotInitializedError: If the runner has not been wired yet.
    """
    if generated_mcp_runner is None:
        raise DependencyNotInitializedError(
            "Generated MCP runner is not initialized; the application must "
            "create it after the FastAPI app is built."
        )
    return generated_mcp_runner


def get_mcp_tool_loader() -> McpToolLoader:
    """Return the concrete fastmcp-based MCP tool loader.

    Returns:
        The FastMcpToolLoader instance configured with ``MCP_TOOL_TIMEOUT``.
    """
    return FastMcpToolLoader(tool_timeout=app_config.MCP_TOOL_TIMEOUT)


def get_create_mcp_server_use_case() -> CreateMcpServerUseCase:
    return CreateMcpServerUseCase(
        store=get_mcp_registry_store(),
        tool_loader=get_mcp_tool_loader(),
        openapi_factory=get_mcp_openapi_factory(),
        runner=get_generated_mcp_runner(),
    )


def get_list_mcp_servers_use_case() -> ListMcpServersUseCase:
    return ListMcpServersUseCase(store=get_mcp_registry_store())


def get_get_mcp_server_use_case() -> GetMcpServerUseCase:
    return GetMcpServerUseCase(store=get_mcp_registry_store())


def get_get_mcp_server_use_case_optional() -> GetMcpServerUseCase | None:
    """Return the get use case when the registry store is wired (None otherwise).

    The validate route (BUG-001 id-merge) reads the stored entry ONLY when the
    body carries ``id``; on the stateless candidate probe (no ``id``) the
    registry is not needed at all, so an UNWIRED store must not break the
    legacy validate callers that never had a store dependency. The ``{name}``
    routes keep the STRICT providers (a store-dependent call on a disabled
    registry answers 503, never a guess).

    Returns:
        The get use case, or ``None`` when the registry is disabled.
    """
    if mcp_registry_store is None:
        return None
    return GetMcpServerUseCase(store=mcp_registry_store)


def get_update_mcp_server_use_case() -> UpdateMcpServerUseCase:
    return UpdateMcpServerUseCase(
        store=get_mcp_registry_store(),
        tool_loader=get_mcp_tool_loader(),
        openapi_factory=get_mcp_openapi_factory(),
        runner=get_generated_mcp_runner(),
    )


def get_delete_mcp_server_use_case() -> DeleteMcpServerUseCase:
    return DeleteMcpServerUseCase(
        store=get_mcp_registry_store(),
        runner=get_generated_mcp_runner(),
    )


def get_validate_mcp_server_use_case() -> ValidateMcpServerUseCase:
    return ValidateMcpServerUseCase(
        tool_loader=get_mcp_tool_loader(),
        openapi_factory=get_mcp_openapi_factory(),
    )


# --- Dual auth (JWT + per-user API key) ---------------------------------------
#
# The JWT adapter (JWKS fetch over httpx) and the API-key reader (asyncpg read
# of the shared ``api_keys`` table owned by composable-agents) are wired
# lazily because they depend on resources that are only available after the
# application starts:
# - the JwtAdapter needs no external resource (it creates its own httpx client
#   and caches the JWKS), but its JWKS URL is derived from ``LOGTO_URL``;
# - the AsyncpgApiKeyReader needs the asyncpg pool (created in ``main.lifespan``).
# ``main.py`` assigns these module-level singletons once the resources exist,
# then calls ``security.set_auth_service(AuthService(jwt, reader))`` so the
# dual-auth dependency is engaged on protected routers and the MCP middleware.

jwt_adapter: JwtServicePort | None = None
api_key_reader: ApiKeyRepositoryPort | None = None
auth_service: AuthService | None = None


def get_jwt_adapter() -> JwtServicePort:
    """Return the JWT adapter singleton (wired by ``main.lifespan``).

    Raises:
        DependencyNotInitializedError: If the adapter has not been wired yet.
    """
    if jwt_adapter is None:
        raise DependencyNotInitializedError(
            "JWT adapter is not initialized; the application lifespan must "
            "create it before serving requests."
        )
    return jwt_adapter


def get_api_key_reader() -> ApiKeyRepositoryPort:
    """Return the API-key reader singleton (wired by ``main.lifespan``).

    Raises:
        DependencyNotInitializedError: If the reader has not been wired yet.
    """
    if api_key_reader is None:
        raise DependencyNotInitializedError(
            "API key reader is not initialized; the application lifespan must "
            "create the asyncpg pool before serving requests."
        )
    return api_key_reader


def get_auth_service() -> AuthService:
    """Return the dual-auth :class:`AuthService` singleton.

    Raises:
        DependencyNotInitializedError: If the service has not been wired yet.
    """
    if auth_service is None:
        raise DependencyNotInitializedError(
            "Auth service is not initialized; the application lifespan must "
            "wire the JwtAdapter and AsyncpgApiKeyReader before serving requests."
        )
    return auth_service
