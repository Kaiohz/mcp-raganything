"""Centralized error message templates.

The message text raised with every domain error is declared here so error
wording is discoverable, consistent and never duplicated. Callers format the
template with the runtime values::

    raise StorageNotFoundError(ErrorMessage.OBJECT_NOT_FOUND.format(bucket=b, path=p))
"""

from enum import StrEnum


class ErrorMessage(StrEnum):
    """Catalog of centralized error message templates (``str.format`` style)."""

    # --- Configuration / dependencies ---
    CLASSICAL_RAG_UNAVAILABLE_VECTOR = (
        "Classical RAG unavailable: vector store not initialized"
    )
    CLASSICAL_RAG_UNAVAILABLE_VECTOR_LLM = (
        "Classical RAG unavailable: vector store or LLM not initialized"
    )
    LLM_NOT_CONFIGURED = (
        "No LLM provider configured for the authenticated user. "
        "Configure your provider via the user LLM settings endpoint."
    )

    # --- Storage (MinIO) ---
    OBJECT_NOT_FOUND = "Object not found: bucket={bucket}, path={path}"
    BUCKET_NOT_FOUND = "Bucket not found: {bucket}"

    # --- Vector store (pgvector) ---
    VECTOR_STORE_NOT_INITIALIZED = (
        "Vector store not initialized for working_dir: {working_dir}"
    )

    # --- Document reader (Kreuzberg) ---
    UNSUPPORTED_FILE_FORMAT = "Unsupported file format: {error}"
    INVALID_FILE = "Invalid file: {error}"
    KREUZBERG_EXTRACTION_FAILED = "Kreuzberg extraction failed: {error}"
    KREUZBERG_EXTRACTION_TIMED_OUT = "Kreuzberg extraction timed out after {timeout}s"

    # --- Classical RAG helpers / BM25 ---
    FILE_NAME_ESCAPES_OUTPUT_DIR = "file_name escapes output directory: {file_name}"
    INVALID_TABLE_PREFIX = "Invalid table_prefix: {table_prefix!r}"
    INVALID_TEXT_CONFIG = "Invalid text_config: {text_config!r}"

    # --- Security ---
    API_KEY_UNAUTHORIZED = "The Api Key you provided is unauthorized"
    API_KEY_EMPTY = "The Api Key is empty"
    API_KEY_DISABLED = "Auth by Api key is disabled"
    AUTH_INVALID_CREDENTIALS = "Invalid or missing credentials"

    # --- File routes ---
    FILENAME_REQUIRED = "Filename is required"
    INVALID_FILENAME = "Invalid filename"
    FILE_TYPE_NOT_ALLOWED = "File type '{ext}' is not allowed"
    CONTENT_TYPE_NOT_ALLOWED = "Content type '{content_type}' is not allowed"
    PREFIX_MUST_BE_RELATIVE = "prefix must be a relative path within the bucket"
    PREFIX_MUST_BE_RELATIVE_SHORT = "prefix must be a relative path"
    FILE_TOO_LARGE = "File exceeds maximum allowed size of {max_mb} MB"
    FILE_PATH_EMPTY = "file_path must not be empty"
    MINIO_REMOVE_PARTIAL_FAILED = "Failed to delete {count} object(s) from storage"

    # --- Indexing / query ---
    VECTOR_SEARCH_FAILED = "Vector search failed in hybrid+ mode: {error}"

    # --- MCP server registry ---
    MCP_SERVER_NOT_FOUND = "MCP server not found: {name}"
    MCP_SERVER_ALREADY_EXISTS = "MCP server already exists: {name}"
    MCP_SERVER_UNREACHABLE = "MCP server unreachable: {error}"
    MCP_SERVER_STORAGE_FAILED = "MCP server storage failed for {name}: {error}"
    MCP_SERVER_AUTH_MODE_INVALID = (
        "MCP server registry contains an invalid stored auth_mode: {value}"
    )
    OPENAPI_FETCH_FAILED = "OpenAPI spec fetch failed for {url}: {error}"
    OPENAPI_INVALID_SPEC = "OpenAPI spec invalid for {url}: {reason}"
    MCP_ALREADY_MOUNTED = "MCP server already mounted: {name}"
    MCP_CONNECTION_ERROR = "MCP server connection failed: {error}"
    MCP_TOOL_LOAD_ERROR = "MCP tool loading failed: {error}"

    # --- Web search (DDGS + Crawl4AI) ---
    WEB_SEARCH_INVALID_QUERY = "The search query must contain visible characters."
    WEB_SEARCH_INVALID_MAX_RESULTS = "max_results must be an integer between 1 and 10."
    WEB_SEARCH_INVALID_EXPLORE_TOP_N = (
        "explore_top_n must be an integer between 0 and 5."
    )
    WEB_SEARCH_TIMED_OUT = "The web search did not complete in time. Try again later."
    WEB_SEARCH_PROVIDER_UNAVAILABLE = (
        "The web search provider is temporarily unavailable."
    )
    WEB_SEARCH_UNEXPECTED_ERROR = "The web search failed unexpectedly."
    WEB_SEARCH_FAILED = "The web search failed. Try again later."
    WEB_SEARCH_UNSAFE_URL = "The page was skipped: URL rejected ({reason})"
    WEB_SEARCH_SCHEME_NOT_ALLOWED = (
        "URL scheme is not allowed: only http and https URLs can be explored."
    )
    WEB_SEARCH_HOST_MISSING = "URL host is missing or invalid."
    WEB_SEARCH_URL_CREDENTIALS = "URLs containing embedded credentials are not allowed."
    WEB_SEARCH_PORT_INVALID = "URL port is invalid."
    WEB_SEARCH_LOCAL_HOST = "URL resolves to a local host and cannot be explored."
    WEB_SEARCH_NON_PUBLIC_ADDRESS = (
        "URL points to a non-public address and cannot be explored."
    )
    WEB_SEARCH_DNS_FAILED = "URL host could not be resolved safely."
    WEB_SEARCH_DNS_TIMEOUT = "URL host resolution timed out."
    WEB_SEARCH_CRAWL_FAILED = "Page crawl failed."
    WEB_SEARCH_CRAWL_TIMED_OUT = "Page crawl timed out."
    WEB_SEARCH_CRAWL_ROBOTS = "Page crawl was blocked by robots.txt."
    WEB_SEARCH_CRAWL_REDIRECTED = "Page was redirected to a disallowed destination."
    WEB_SEARCH_CRAWL_BATCH_FAILED = "The page exploration did not complete in time."

    # --- Memory store (composable-agents Store File API) ---
    MEMORY_IDENTITY_REQUIRED = (
        "Saving memories requires an authenticated user identity with a "
        "forwardable JWT or API key credential."
    )
    MEMORY_NAME_INVALID = (
        "Invalid memory name: use letters, digits, dots, hyphens or "
        "underscores without paths or separators."
    )
    MEMORY_CONTENT_REQUIRED = "Memory content must not be empty."
    MEMORY_EXISTS = "A memory with this name already exists (if_exists='fail')."
    MEMORY_MODE_INVALID = "if_exists must be one of: fail, append, replace."
    MEMORY_STORE_TIMEOUT = "The memory service did not respond in time."
    MEMORY_STORE_UNAVAILABLE = "The memory service is temporarily unavailable."
    MEMORY_STORE_REJECTED = "The memory service rejected the memory save request."
    MEMORY_STORE_UNEXPECTED = "The memory service request failed unexpectedly."
    MEMORY_STORE_FORBIDDEN = (
        "The memory request was rejected by the upstream service: authentication "
        "or permission failure."
    )
    MEMORY_STORE_INVALID_RESPONSE = "The memory service returned an invalid response."
    MEMORY_STORE_PATH_ESCAPE = "Memory storage path must stay inside /memories/."
    MEMORY_SAVE_FAILED = "Saving the memory failed. Try again later."
