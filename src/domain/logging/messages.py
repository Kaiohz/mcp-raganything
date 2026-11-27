"""Centralized log message templates.

All log message strings used across the application are declared here so they
are discoverable, non-duplicated, and easy to audit. Callers reference these
constants instead of inlining message text, e.g.::

    logger.info(LogMessage.APP_SHUTDOWN_INITIATED)

Each template may contain ``%s``/``%d`` style placeholders consumed by the
stdlib logging lazy interpolation.
"""

from enum import StrEnum


class LogMessage(StrEnum):
    """Catalog of centralized log message templates (stdlib ``%`` style)."""

    # --- Application lifecycle (main.py) ---
    APP_SHUTDOWN_INITIATED = "Application shutdown initiated"
    APP_SHUTDOWN_COMPLETE = "Application shutdown complete"
    APP_BM25_CLOSE_FAILED = "Failed to close BM25 adapter"
    APP_CLASSICAL_VS_CLOSE_FAILED = "Failed to close classical vector store"
    APP_MIGRATIONS_RUNNING = "Running database migrations..."
    APP_MIGRATIONS_DONE = "Database migrations completed"

    # --- Storage (MinIO) ---
    MINIO_OBJECT_NOT_FOUND = "Object not found: bucket=%s, path=%s"
    MINIO_RETRIEVE_ERROR = "MinIO error retrieving object: %s"
    MINIO_BUCKET_NOT_FOUND = "Bucket not found: %s"
    MINIO_UPLOAD_ERROR = "MinIO error uploading object: %s"
    MINIO_LIST_ERROR = "MinIO error listing objects: %s"
    MINIO_HEALTH_CHECK_FAILED = "MinIO health check failed"
    MINIO_REMOVE_OBJECT_ERROR = "MinIO error removing object: %s"
    MINIO_REMOVE_PREFIX_ERROR = "MinIO error removing prefix %s: %s"
    MINIO_REMOVE_OBJECT_PARTIAL_ERROR = "MinIO partial delete error for object %s: %s"

    # --- Delete cascade (MinIO → pgvector) ---
    DELETE_FILE_STARTED = "Delete file started: bucket=%s object=%s working_dir=%s"
    DELETE_FILE_MINIO_DONE = "MinIO object deleted: bucket=%s object=%s"
    DELETE_FILE_WORKING_DIR_FALLBACK = (
        "working_dir not provided for delete_file, falling back to %s"
    )
    DELETE_FILE_VECTORS_DONE = (
        "Vectors deleted for file: working_dir=%s file_path=%s count=%d"
    )
    DELETE_FILE_VECTORS_SKIPPED = (
        "Vector deletion skipped (no vector store configured): object=%s"
    )
    DELETE_FOLDER_STARTED = "Delete folder started: bucket=%s prefix=%s"
    DELETE_FOLDER_MINIO_DONE = "MinIO prefix deleted: bucket=%s prefix=%s"
    DELETE_FOLDER_VECTORS_DONE = (
        "Vectors deleted for folder: working_dir=%s prefix=%s count=%d"
    )
    DELETE_FOLDER_VECTORS_SKIPPED = (
        "Vector deletion skipped (no vector store configured): prefix=%s"
    )

    # --- Database health (asyncpg) ---
    POSTGRES_HEALTH_CHECK_FAILED = "PostgreSQL health check failed"
    POSTGRES_HEALTH_TLS_NO_VERIFY = (
        "SSL mode '%s' disables certificate and hostname verification for "
        "PostgreSQL health check; use 'verify-full' in production to prevent "
        "MITM attacks."
    )

    # --- Document reader (Kreuzberg) ---
    KREUZBERG_EXTRACTION_FAILED_FOR = "Kreuzberg extraction failed for %s: %s"
    KREUZBERG_EXTRACTION_DURATION = "Kreuzberg extraction done: file=%s elapsed_ms=%.2f"
    KREUZBERG_SERIALIZATION_DURATION = (
        "Kreuzberg serialization done: file=%s elapsed_ms=%.2f"
    )

    # --- Indexing use cases ---
    INDEXATION_FINISHED = "Indexation finished: %s"
    FOLDER_INDEXATION_FINISHED = "Folder indexation finished: %s"
    PAYLOAD_PREVIEW = "Payload : %s"
    CLASSICAL_FILE_INDEX_STARTED = (
        "Classical file indexing started: file=%s working_dir=%s"
    )
    CLASSICAL_FILE_INDEX_DONE = "Classical file indexing done: file=%s status=%s"
    CLASSICAL_FILE_INDEX_FAILED = "Classical file indexing failed: file=%s error=%s"
    CLASSICAL_FOLDER_INDEX_STARTED = (
        "Classical folder indexing started: working_dir=%s recursive=%s"
    )
    CLASSICAL_FOLDER_INDEX_LISTING = (
        "Classical folder indexing: found %d file(s) under prefix=%s"
    )
    CLASSICAL_FOLDER_INDEX_NO_FILES = (
        "Classical folder indexing: no files found under prefix=%s"
    )
    CLASSICAL_FOLDER_INDEX_FILE_START = "Indexing file %d/%d: %s"
    CLASSICAL_FOLDER_INDEX_FILE_DONE = "Indexed file %d/%d: %s (%d chunks)"
    CLASSICAL_FOLDER_INDEX_FILE_FAILED = "Failed to index file %d/%d: %s error=%s"
    CLASSICAL_FOLDER_INDEX_DONE = "Classical folder indexing done: working_dir=%s total=%d processed=%d failed=%d status=%s"

    # --- RAG engine (LightRAG) ---
    LIGHTRAG_INDEX_DOCUMENT_FAILED = "Failed to index document %s: %s"
    LIGHTRAG_INDEXED_FILE = "Indexed %s (%d/%d)"
    LIGHTRAG_INDEX_FILE_FAILED = "Failed to index %s: %s"

    # --- Classical query use case ---
    CLASSICAL_BM25_UNAVAILABLE_FALLBACK = (
        "BM25 unavailable, falling back to vector mode"
    )
    CLASSICAL_BM25_SEARCH_FAILED_HYBRID = "BM25 search failed in hybrid mode: %s"
    CLASSICAL_VECTOR_SEARCH_FAILED_HYBRID = "Vector search failed in hybrid mode: %s"

    # --- Query use case (hybrid+) ---
    QUERY_BM25_SEARCH_FAILED = "BM25 search failed in hybrid+ mode: %s"
    QUERY_VECTOR_SEARCH_FAILED = "Vector search failed in hybrid+ mode: %s"

    # --- Background task helpers (routes) ---
    BACKGROUND_TASK_FAILED = "Background %s failed"

    # --- pg_textsearch adapter ---
    PG_TEXTSEARCH_NOT_INSTALLED = (
        "pg_textsearch extension not installed. "
        "BM25 ranking <@> operator will not work. "
        "Run: CREATE EXTENSION pg_textsearch;"
    )
    PG_TEXTSEARCH_CHECK_FAILED = "Could not check pg_textsearch extension: %s"
    PG_TEXTSEARCH_INDEX_EXISTS = "BM25 index '%s' already exists for text_config='%s'"
    PG_TEXTSEARCH_INDEX_CREATED = "Created BM25 index '%s' with text_config='%s'"
    PG_TEXTSEARCH_ENSURE_INDEX_FAILED = "Failed to ensure BM25 index: %s"
    PG_TEXTSEARCH_UPDATING_TRIGGER = (
        "Updating trigger function from old text_config to '%s'"
    )
    PG_TEXTSEARCH_REBUILT_TSV = "Rebuilt content_tsv: %s with text_config='%s'"
    PG_TEXTSEARCH_TRIGGER_CHECK_FAILED = "Could not check/rebuild trigger function: %s"
    PG_TEXTSEARCH_SEARCH_FAILED = "BM25 search failed: %s"
    PG_TEXTSEARCH_INDEX_CREATION_FAILED = "BM25 index creation failed: %s"
    PG_TEXTSEARCH_INDEX_DROP_FAILED = "BM25 index drop failed: %s"

    # --- Classical BM25 adapter ---
    CLASSICAL_PG_TEXTSEARCH_NOT_INSTALLED = (
        "pg_textsearch extension not installed. "
        "BM25 ranking will not work. "
        "Run: CREATE EXTENSION pg_textsearch;"
    )
    CLASSICAL_BM25_TABLE_MISSING = "Table %s does not exist yet, skipping BM25 index"
    CLASSICAL_BM25_INDEX_CREATED = "Created BM25 index '%s' on %s with text_config='%s'"
    CLASSICAL_BM25_ENSURE_INDEX_FAILED = "Failed to ensure BM25 index on %s: %s"
    CLASSICAL_BM25_SEARCH_FAILED = "BM25 search failed on %s: %s"
    CLASSICAL_BM25_INDEX_DROP_FAILED = "BM25 index drop failed on %s: %s"

    # --- MCP file tools ---
    MCP_READ_FILE_UNEXPECTED_ERROR = "Unexpected error reading file: %s"

    # --- Temporary-file MCP tools (chat-pdf-attachments spec §5.3) ---
    TEMP_UPLOAD_REJECTED = "temporary_upload_file rejected (user=%s): %s"
    TEMP_UPLOAD_COMPLETED = "temporary_upload_file completed: object=%s size=%d"
    TEMP_DELETE_REJECTED = "temporary_delete_file rejected (user=%s): %s"
    TEMP_DELETE_COMPLETED = "temporary_delete_file completed: object=%s"
    TEMP_DELETE_FAILED = "temporary_delete_file failed (object=%s): %s"
    # NOTE: read at import time like the other singletons — FILES_MAX_UPLOAD_SIZE
    # must be set in the environment BEFORE the process boots.
    TEMP_DELETE_ABSENT = (
        "temporary_delete_file target already absent (idempotent no-op): object=%s (%s)"
    )

    # --- Exception handler log lines (main.py) ---
    LOG_STORAGE_ERROR = "Storage error: %s"
    LOG_RAG_ERROR = "RAG error: %s"
    LOG_RAG_UNAVAILABLE = "RAG unavailable: %s"
    LOG_VECTOR_STORE_ERROR = "Vector store error: %s"
    LOG_DOCUMENT_ERROR = "Document error: %s"
    LOG_CLASSICAL_ERROR = "Classical config error: %s"
    LOG_INDEXING_ERROR = "Indexing error: %s"
    LOG_CONFIG_ERROR = "Config error: %s"
    LOG_FILE_ERROR = "File error: %s"
    LOG_UNHANDLED_DOMAIN_ERROR = "Unhandled domain error: %s"
    LOG_INVALID_API_KEY = "Invalid API key: %s"
    LOG_MCP_ERROR = "MCP error: %s"
    LOG_MCP_INVALID_STORED_CONFIG = "MCP invalid stored configuration: %s"
    MCP_CONNECT_FAILED = "MCP connection to %s failed: %s"
    MCP_TOOLS_LOAD_FAILED = "MCP tool loading from %s failed: %s"

    # --- Auth (dual JWT / API key) ---
    # The enum value is a stable prefix kept verbatim in the formatted log line
    # (no ``%s`` inside it) so tests can assert ``LogMessage.X in r.getMessage()``.
    # No PII is interpolated here — only error type / message for diagnostics.
    AUTH_JWT_DECODE_FAILED = "JWT decode failed"
    AUTH_JWKS_FETCH_FAILED = "JWKS fetch failed"
    AUTH_CREDENTIALS_VALIDATED = "Credentials validated for user_id=%s method=%s"

    # --- MCP HTTP-level guard (feat-mcp-oauth2-dual-auth) ---
    # No credential material is ever interpolated (401s stay non-enumerating
    # and the logs stay PII-free).
    MCP_REQUEST_DENIED = "MCP request denied (missing or invalid credentials)"
    MCP_DISCOVERY_ORIGIN_UNAVAILABLE = (
        "MCP OAuth2 discovery origin unresolved: MCP_PUBLIC_BASE_URL is unset and "
        "the request Host is not in MCP_ALLOWED_HOSTS — no discovery URL is advertised"
    )
    MCP_TRANSPORT_REJECTED = "MCP request rejected by transport security: %s"
    # --- MCP OAuth2 discovery configuration shape (startup warnings, M-3 —
    # mirrors the composable-agents QA BUG-003 / code-review NEW-1 pattern).
    # The non-https messages deliberately do NOT interpolate the configured
    # value: an operator may embed credentials in the URL.
    MCP_AUTHORIZATION_SERVER_NOT_ADVERTISED = (
        "MCP protected-resource metadata advertises NO authorization server: both "
        "MCP_AUTHORIZATION_SERVER and LOGTO_URL are empty — MCP clients cannot complete "
        "the OAuth2 discovery chain (REQ-002)"
    )
    MCP_AUTHORIZATION_SERVER_NOT_HTTPS = (
        "MCP protected-resource metadata advertises the non-https authorization server '%s' "
        "— a PUBLIC MCP client may not reach it (a container-internal LOGTO_URL fallback or "
        "an explicit http value); set MCP_AUTHORIZATION_SERVER to the publicly reachable "
        "issuer, e.g. https://logto.soludev.tech/oidc (QA BUG-003)"
    )
    MCP_PUBLIC_BASE_URL_NOT_HTTPS = (
        "MCP discovery documents advertise a non-https public origin: MCP_PUBLIC_BASE_URL is an "
        "'http'-scheme value — public MCP clients reach the resource over TLS; set the 'https' "
        "public origin, e.g. https://raganything.soludev.tech (a SCHEME-LESS value already refuses "
        "startup). The configured value is never interpolated here on purpose: "
        "an operator may embed credentials in the URL."
    )

    # --- Generated MCP runner (openapi_mcp) ---
    MCP_EXACT_PATH_ROUTE_UNAVAILABLE = (
        "MCP exact-path route unavailable for %s: the host app has no "
        "router.routes list; the Mount-only 307 behavior applies"
    )

    # --- MCP server registry (use cases / store / factory / runner) ---
    MCP_SERVER_CREATED_UC = "MCP server created (use case): name=%s tool_count=%d"
    MCP_SERVER_UPDATED_UC = "MCP server updated (use case): name=%s tool_count=%d"
    MCP_SERVER_DELETED_UC = "MCP server deleted (use case): name=%s"
    MCP_SERVER_LISTED_UC = "MCP servers listed (use case): count=%d"
    MCP_SERVER_GET_UC = "MCP server retrieved (use case): name=%s"
    MCP_SERVER_VALIDATED_UC = "MCP server validated (use case): name=%s tool_count=%d"
    MCP_SERVER_SAVE_REPOSITORY = "MCP server saved to registry: name=%s"
    MCP_SERVER_DELETE_REPOSITORY = "MCP server deleted from registry: name=%s"
    MCP_OPENAPI_SPEC_FETCHED = "OpenAPI spec fetched: url=%s bytes=%d"
    MCP_OPENAPI_SPEC_FETCH_FAILED = "OpenAPI spec fetch failed: url=%s error=%s"
    MCP_OPENAPI_SPEC_INVALID = "OpenAPI spec invalid: url=%s reason=%s"
    MCP_OPENAPI_MCP_BUILT = "OpenAPI MCP server built: name=%s tool_count=%d"
    MCP_RUNNER_MOUNTED = "Generated MCP server mounted: name=%s url=%s tool_count=%d"
    MCP_RUNNER_UNMOUNTED = "Generated MCP server unmounted: name=%s"
    MCP_RUNNER_MOUNT_FAILED = "Generated MCP server mount failed: name=%s error=%s"
    MCP_RUNNER_LIFESPAN_CLOSE_FAILED = (
        "Generated MCP lifespan close failed: name=%s error=%s"
    )
    MCP_TOOL_COUNT_FAILED = (
        "Generated MCP tool count failed (the mounted server stays healthy): "
        "name=%s error=%s"
    )
    MCP_REHYDRATION_STARTED = "Rehydrating generated MCP servers from registry..."
    MCP_REHYDRATION_DONE = "Rehydration complete: mounted=%d skipped=%d"
    MCP_REHYDRATION_ENTRY_FAILED = (
        "Rehydration: failed to remount generated MCP server %s: %s"
    )
    MCP_REHYDRATION_SKIPPED_SUMMARY = (
        "Rehydration skipped %d generated MCP server(s) due to mount failures; "
        "they will be retried on the next startup."
    )
    # list_all skip reasons (feat-mcp-oauth2 REQ-004): the two failure modes are
    # labeled separately so a corrupted ``auth_mode`` value is never reported as
    # a "secret decryption failed" miss. No PII — the server name and the
    # exception detail (the unsupported non-secret mode value) only.
    MCP_SERVER_SKIPPED_DECRYPT_FAILED = "Skipping MCP server from the catalog: secret decryption failed: server=%s error=%s"
    MCP_REGISTRY_ENTRY_INVALID_AUTH_MODE = "Skipping MCP server from the catalog: invalid stored auth_mode: server=%s error=%s"

    # --- Web search tool (DDGS + Crawl4AI) ---
    # No query, URL or crawled content is interpolated (untrusted external data).
    WEB_SEARCH_EXECUTOR_CLOSED = "Web search executor closed"
    WEB_SEARCH_EXECUTOR_CLOSE_FAILED = "Web search executor shutdown failed"
    WEB_SEARCH_COMPLETED = (
        "Web search completed: results=%d explored=%d duration_ms=%.1f"
    )
    WEB_SEARCH_VALIDATION_REJECTED = "Web search validation rejected the arguments"
    WEB_SEARCH_PROVIDER_FAILED = "Web search provider call failed"
    WEB_SEARCH_BACKEND_NO_RESULTS = (
        "ddgs backend %s returned no results (max_results=%d): this may be "
        "either a genuinely empty result set or an anti-bot challenge / "
        "network block (ddgs collapses both into 'No results found.'); "
        "trying the next backend"
    )
    WEB_SEARCH_BACKEND_FAILED = (
        "ddgs backend %s failed; trying the next backend "
        "(details are never logged)"
    )
    WEB_SEARCH_UNEXPECTED = "Unexpected web search failure"
    WEB_SEARCH_TIMED_OUT = "Web search timed out after %.2fs"
    URL_SAFETY_REJECTED = "Unsafe URL rejected before crawling: reason=%s"
    CRAWLER_STARTED = "Shared web crawler started"
    CRAWLER_START_FAILED = "Shared web crawler failed to start"
    CRAWLER_CLOSED = "Shared web crawler closed"
    CRAWLER_CLOSE_FAILED = "Shared web crawler close failed"
    CRAWL_BATCH_STARTED = "Crawl batch started: pages=%d"
    CRAWL_BATCH_COMPLETED = (
        "Crawl batch completed: success=%d failed=%d duration_ms=%.1f"
    )
    CRAWL_BATCH_TIMEOUT = "Crawl batch timed out after %.2fs: unfinished=%d"
    CRAWL_CLOSE_FAILED = "Crawl stream close failed"
    MEMORY_SAVE_COMPLETED = "Memory saved: action=%s chars=%d duration_ms=%.1f"
    MEMORY_STORE_REQUEST_FAILED = "Memory store %s request failed: category=%s"
    MEMORY_STORE_CLOSED = "Memory store HTTP client closed"
    MEMORY_STORE_CLOSE_FAILED = "Memory store HTTP client close failed"
    MEMORY_IDENTITY_MISSING = "Memory save rejected: no user identity or credential"
    MEMORY_SAVE_UNEXPECTED = "Unexpected memory save failure"
    MEMORY_LIST_UNEXPECTED = "Unexpected memory list failure"
    MEMORY_READ_UNEXPECTED = "Unexpected memory read failure"
    MEMORY_DELETE_UNEXPECTED = "Unexpected memory delete failure"

    # --- Caller scope / RAG spaces (scope-rag-memories spec) ---
    SCOPE_RESOLVED = "Caller scope resolved: user_id=%s agent_slug=%s"
    SPACE_PROVISIONING_FAILED = (
        "RAG space record write failed (best-effort, request not blocked): "
        "agent_slug=%s error=%s"
    )
    SPACE_CREATED = "RAG space created: user_id=%s agent_slug=%s working_dir=%s"
    SPACE_DELETED = (
        "RAG space deleted (cascade): user_id=%s agent_slug=%s tracked_tables=%d"
    )
    SPACE_TABLE_DROPPED = "RAG space vector table dropped: working_dir=%s"
