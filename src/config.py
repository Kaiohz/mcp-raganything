import logging
import os
import tempfile
from typing import Annotated
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from dotenv import load_dotenv
from pydantic import Field, PrivateAttr, model_validator
from pydantic import field_validator as pydantic_field_validator
from pydantic_settings import BaseSettings, NoDecode

load_dotenv()

logger = logging.getLogger(__name__)

#: The single refusal message for an empty ``MCP_ALLOWED_HOSTS`` list, shared
#: by the empty env-string branch and the empty-Python-list branch (they are
#: the same silent-421 outage with DNS-rebinding protection enabled).
_EMPTY_MCP_ALLOWED_HOSTS_MESSAGE = (
    "MCP_ALLOWED_HOSTS must not be empty: with DNS-rebinding protection enabled "
    "an empty host allow-list rejects every MCP request with HTTP 421 — list the "
    "public proxy host plus the localhost defaults"
)


def _is_json_style_list_value(stripped: str) -> bool:
    """Whether a CSV env value is actually a JSON-style document.

    An author may copy the neighbouring ``ALLOWED_ORIGINS`` JSON form
    (``'["https://raganything.soludev.tech"]'``): it does not CSV-split and
    would become ONE entry that matches nothing — a silent total outage (421
    for every MCP request with DNS-rebinding protection enabled for
    ``MCP_ALLOWED_HOSTS``, dead audiences for ``JWT_AUDIENCES``). The
    characteristic JSON shapes are quote characters anywhere or a ``{``-wrap;
    a legit host[:port] CSV value never carries either (the ``[::1]`` IPv6
    literal is bracket-wrapped but never quoted).
    """
    return '"' in stripped or "'" in stripped or stripped.startswith("{")


def _json_punctuated_entry(part: str) -> bool:
    """Whether ONE split CSV entry still carries JSON punctuation.

    Review rule (C-2 hardening): a partially-JSON value like ``'[a, b]'``
    carries no quote/brace on the WHOLE string, so the pre-split check passes
    and the split produces entries that match nothing (silent outage). A
    post-split entry is refused when it still carries a quote character
    anywhere, a ``{`` at its start, or a ``[`` at its start WITHOUT a colon
    (the only bracket-wrapped values a host list may contain are the
    ``[...:...]`` IPv6 literals — a JSON array fragment never has the colon).
    """
    if '"' in part or "'" in part or part.startswith("{"):
        return True
    return part.startswith("[") and ":" not in part


def _json_style_list_error(field_name: str, value: str) -> str:
    """Build the refusal message for a JSON-style CSV env value."""
    return (
        f"{field_name} received a JSON-style value ({value!r}) — an "
        "ALLOWED_ORIGINS-style value; these env vars are parsed as "
        "comma-separated lists (e.g. MCP_ALLOWED_HOSTS=host-a,host-b); a "
        "JSON-written value stays ONE entry that matches nothing (silent auth "
        "failures; HTTP 421 for every MCP request)"
    )


def _coerce_csv_list_value(
    value: object, *, field_name: str, refuse_empty: bool
) -> object:
    """Coerce one comma-separated env-string setting into a list.

    Already-parsed lists pass through; absent values keep the field default
    (handled before the validators run — a ``None`` never reaches here).

    Args:
        value: The raw value (env string, Python list, or anything else that
            passes through unchanged).
        field_name: The setting name used in the refusal messages.
        refuse_empty: ``True`` for ``MCP_ALLOWED_HOSTS`` (an empty allow-list
            421s every MCP request with DNS-rebinding protection ENABLED —
            fail fast at startup instead); ``False`` for the list settings
            that tolerate empty (``JWT_AUDIENCES``, ``MCP_ALLOWED_ORIGINS``).

    Returns:
        The parsed list of stripped entries, or the value unchanged.
    """
    if isinstance(value, list):
        if not value and refuse_empty:
            raise ValueError(_EMPTY_MCP_ALLOWED_HOSTS_MESSAGE)
        return value
    if isinstance(value, str):
        return _coerce_csv_string_value(
            value, field_name=field_name, refuse_empty=refuse_empty
        )
    return value


def _coerce_csv_string_value(
    value: str, *, field_name: str, refuse_empty: bool
) -> list[str]:
    """Coerce one CSV string into stripped, non-blank entries.

    Refusals (fail fast at startup instead of a silent outage):
    * a JSON-style WHOLE value (see :func:`_is_json_style_list_value`);
    * an entry still carrying JSON punctuation after the split (a partially
      JSON value like ``'[a, b]'`` also matches nothing silently).
    """
    stripped = value.strip()
    if _is_json_style_list_value(stripped):
        raise ValueError(_json_style_list_error(field_name, stripped))
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts and refuse_empty:
        raise ValueError(_EMPTY_MCP_ALLOWED_HOSTS_MESSAGE)
    punctuated = [part for part in parts if _json_punctuated_entry(part)]
    if punctuated:
        raise ValueError(_json_style_list_error(field_name, value))
    return parts


class AppConfig(BaseSettings):
    """Application-level settings: server, CORS, logging, API key."""

    ALLOWED_ORIGINS: list[str] = Field(
        default=["*"], description="CORS allowed origins"
    )
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    UVICORN_LOG_LEVEL: str = Field(default="critical", description="Uvicorn log level")
    LOG_LEVEL: str = Field(default="info", description="Application log level")
    OUTPUT_DIR: str = Field(
        default=os.path.join(tempfile.gettempdir(), "output"),
        description="Directory for temporary output file storage",
    )
    API_KEY: str = Field(
        default="", description="API key for REST and MCP authentication"
    )
    SECRET_ENCRYPTION_KEY: str = Field(
        default="",
        description="Fernet key used to encrypt MCP registry secrets at rest. "
        'Generate with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`. '
        "Must be stable across restarts to decrypt previously encrypted secrets.",
    )
    GENERATED_MCP_BASE_URL: str = Field(
        default="http://localhost:8020",
        description="Absolute base URL of this service, used to build the URL "
        "returned for generated (openapi) MCP servers mounted under "
        "/generated/{name}/mcp. E.g. http://raganything-api:8000.",
    )
    OPENAPI_MAX_SPEC_BYTES: int = Field(
        default=5 * 1024 * 1024,
        description="Maximum accepted OpenAPI spec size in bytes when fetching "
        "a remote document (default 5 MB). Specs larger than this are rejected "
        "with OpenApiInvalidSpecError before parsing.",
    )
    MCP_TOOL_TIMEOUT: float = Field(
        default=60.0,
        description="Timeout (seconds) for MCP tool loading via the FastMcpToolLoader "
        "(connecting to external MCP servers and listing their tools).",
    )
    LOGTO_URL: str = Field(
        default="",
        description="Base URL of the Logto IdP instance. The JWKS endpoint is "
        "derived as {LOGTO_URL}/oidc/jwks for JWT verification. When empty, "
        "JWT verification is disabled (only per-user API keys are accepted).",
    )
    JWT_AUDIENCE: str = Field(
        default="",
        description="Expected JWT ``aud`` claim. Passed to ``jwt.decode`` for "
        "audience validation. When empty, no audience check is performed "
        "(PyJWT treats an empty string as 'no audience expected').",
    )
    # --- feat-mcp-oauth2-dual-auth (REQ-003-equivalent / REQ-014) ---
    # Additional accepted JWT audiences (comma-separated env var
    # ``JWT_AUDIENCES``, e.g. the Logto OpenCode app-id). The legacy single
    # ``JWT_AUDIENCE`` stays honored; the two lists are merged by the JWT
    # adapter (deduped, legacy value first). Empty/absent → legacy-only.
    JWT_AUDIENCES: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="Additional accepted JWT ``aud`` claims, comma-separated "
        "(e.g. the Logto OpenCode app-id). Merged with the legacy "
        "``JWT_AUDIENCE`` (deduplicated) by the JWT adapter.",
    )
    # MCP transport-security Host allow-list (comma-separated env var
    # ``MCP_ALLOWED_HOSTS``). DNS-rebinding protection stays ENABLED; only the
    # host list is configurable. The default mirrors the loopback lists used
    # for dev/QA; the production configmap adds the public proxy host
    # (raganything.soludev.tech). A present-but-empty value REFUSES startup
    # (an empty allow-list would 421 every MCP request — fail fast instead).
    MCP_ALLOWED_HOSTS: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["127.0.0.1:*", "localhost:*", "[::1]:*"],
        description="MCP transport-security Host allow-list, comma-separated "
        "(REQ-014). DNS-rebinding protection stays ENABLED. Must list the "
        "public proxy host (raganything.soludev.tech) in production plus the "
        "loopback defaults for tests/dev.",
    )
    # Public base origin advertised by the MCP OAuth2 discovery documents
    # (M-2 hardening), e.g. ``https://raganything.soludev.tech``. The request
    # ``Host`` header is client-supplied and MUST NOT be echoed unvalidated:
    # when this is configured it always wins; when unset the request Host is
    # trusted ONLY when it matches the ``MCP_ALLOWED_HOSTS`` allow-list.
    MCP_PUBLIC_BASE_URL: str = Field(
        default="",
        description="Configured public origin for the MCP OAuth2 discovery "
        "documents (401 challenge + /.well-known metadata). Always wins over "
        "the request Host when set.",
    )
    # Public OAuth2 authorization server advertised by the protected-resource
    # metadata document (REQ-002-equivalent): the issuer an MCP client can
    # actually REACH. Resolved ONCE at startup by ``resolve_mcp_authorization_server``
    # (the ``model_validator``): an explicit value wins verbatim; when unset it
    # falls back to ``<LOGTO_URL>/oidc`` — the SAME resolved value feeds the
    # ``/.well-known`` route AND the startup warnings; when both are empty
    # NOTHING is advertised (``authorization_servers: []``).
    MCP_AUTHORIZATION_SERVER: str = Field(
        default="",
        description="Explicit publicly-reachable OAuth2 authorization-server "
        "issuer advertised by the protected-resource metadata. Falls back to "
        "<LOGTO_URL>/oidc when unset (resolved once at startup by the "
        "AppConfig model validator).",
    )
    # MCP transport-security Origin allow-list (comma-separated env var
    # ``MCP_ALLOWED_ORIGINS``). The ``mcp`` SDK matches an Origin EXACTLY (no
    # global wildcard): an empty SDK list rejects EVERY browser Origin. The
    # MCP policy therefore defaults to the app CORS policy: an UNSET value
    # inherits ``ALLOWED_ORIGINS``, and the CORS ``*`` wildcard is mirrored by
    # not validating the Origin at all (M-1, browser-origin MCP clients).
    MCP_ALLOWED_ORIGINS: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="MCP transport-security Origin allow-list, comma-separated "
        "(M-1). Unset/empty → inherits ALLOWED_ORIGINS; a '*' entry mirrors the "
        "CORS wildcard (Origin not validated); other entries are matched "
        "exactly by the mcp SDK (an 'origin:*' port wildcard is also honoured).",
    )

    @pydantic_field_validator("MCP_ALLOWED_HOSTS", mode="before")
    @classmethod
    def _coerce_host_list(cls, value: object) -> object:
        """Coerce the ``MCP_ALLOWED_HOSTS`` CSV env value (REQ-014).

        A present-but-empty allow-list REFUSES startup — with DNS-rebinding
        protection ENABLED it would 421 every MCP request (silent outage).
        """
        return _coerce_csv_list_value(
            value, field_name="MCP_ALLOWED_HOSTS", refuse_empty=True
        )

    @pydantic_field_validator("JWT_AUDIENCES", mode="before")
    @classmethod
    def _coerce_audience_list(cls, value: object) -> object:
        """Coerce the ``JWT_AUDIENCES`` CSV env value (REQ-003-equivalent).

        Empty tolerates (legacy single-audience contract preserved).
        """
        return _coerce_csv_list_value(
            value, field_name="JWT_AUDIENCES", refuse_empty=False
        )

    @pydantic_field_validator("MCP_ALLOWED_ORIGINS", mode="before")
    @classmethod
    def _coerce_origin_list(cls, value: object) -> object:
        """Coerce the ``MCP_ALLOWED_ORIGINS`` CSV env value (M-1).

        Empty tolerates (the field then inherits ``ALLOWED_ORIGINS``).
        """
        return _coerce_csv_list_value(
            value, field_name="MCP_ALLOWED_ORIGINS", refuse_empty=False
        )

    @pydantic_field_validator("MCP_PUBLIC_BASE_URL", mode="before")
    @classmethod
    def _validate_mcp_public_base_url(cls, value: object) -> object:
        """Validate + normalize ``MCP_PUBLIC_BASE_URL`` (M-3, mirrors the
        composable-agents iter-4 validator).

        The advertised origin is the pointer the whole OAuth2 discovery chain
        starts from: a scheme-less value would publish a malformed
        ``resource_metadata`` URL no MCP client can fetch (the chain breaks
        SILENTLY), so a present value without an absolute ``http(s)://``
        scheme REFUSES startup — the same present-but-invalid ⇒ refuse rule
        as the sibling ``MCP_AUTHORIZATION_SERVER`` validator. Surrounding
        whitespace and a trailing slash are stripped here (single
        normalization source); a present ``http://`` value stays dev/QA-legal
        and is warned at startup (``MCP_PUBLIC_BASE_URL_NOT_HTTPS``).
        """
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and not stripped.startswith(("http://", "https://")):
                raise ValueError(
                    "MCP_PUBLIC_BASE_URL must be an absolute URL starting with "
                    "https:// (e.g. 'https://raganything.soludev.tech') — a "
                    "scheme-less value would advertise a malformed "
                    "resource_metadata URL no MCP client can fetch"
                )
            return stripped.rstrip("/")
        return value

    @pydantic_field_validator("MCP_AUTHORIZATION_SERVER", mode="before")
    @classmethod
    def _validate_mcp_authorization_server(cls, value: object) -> object:
        """Reject a malformed explicit ``MCP_AUTHORIZATION_SERVER`` (M-3).

        A blank value means "unset" (the logto-issuer fallback applies); a
        present-but-NON-absolute value would advertise the malformed
        ``["/oidc"]``-shaped issuer an MCP client cannot fetch — the same
        present-but-invalid ⇒ refuse-startup rule as the other configuration
        validators.
        """
        if isinstance(value, str):
            value = value.strip()
            if value and not value.startswith(("http://", "https://")):
                raise ValueError(
                    "MCP_AUTHORIZATION_SERVER must be an absolute URL starting "
                    "with https:// (e.g. 'https://logto.soludev.tech/oidc')"
                )
        return value

    @model_validator(mode="after")
    def resolve_mcp_authorization_server(self) -> "AppConfig":
        """Fall back to ``<LOGTO_URL>/oidc`` when the setting is unset.

        Mirrors composable-agents' ``resolve_mcp_authorization_server``
        (code-review MINOR-1): the advertised issuer used to be derived
        INLINE in the well-known route from the raw ``LOGTO_URL``, so an
        internal (container-only) address could reach the public discovery
        document with NO startup warning. Resolving the fallback at config
        load makes ``MCP_AUTHORIZATION_SERVER`` the ONE effective advertised
        issuer — the ``/.well-known`` route and the startup warnings both
        read the same resolved value, so a non-https fallback (e.g.
        ``http://logto:3001/oidc``) is warned exactly like an explicit
        non-https value.

        When neither value is configured NOTHING is advertised (the route
        serves ``authorization_servers: []`` and the
        ``MCP_AUTHORIZATION_SERVER_NOT_ADVERTISED`` warning fires).
        """
        if not self.MCP_AUTHORIZATION_SERVER and self.LOGTO_URL:
            self.MCP_AUTHORIZATION_SERVER = f"{self.LOGTO_URL.rstrip('/')}/oidc"
        return self


class DatabaseConfig(BaseSettings):
    """Database connection configuration."""

    DATABASE_URL: str = Field(
        default="postgresql://localhost:5432/raganything",
        description="Full PostgreSQL connection URL. Accepted schemes: "
        "postgresql://, postgres://, postgresql+asyncpg://. "
        "Normalized to postgresql+asyncpg:// for SQLAlchemy async. "
        "Credentials are intentionally NOT part of the source default — "
        "provide them through the environment (e.g. DATABASE_URL in .env "
        "or Docker secrets).",
    )
    POSTGRES_STATEMENT_CACHE_SIZE: int | None = Field(
        default=None,
        description="asyncpg statement cache size. Set to 0 for connection poolers "
        "(Neon, PgBouncer, etc.). None uses asyncpg default (100).",
    )
    _ssl_mode: str | None = PrivateAttr(default=None)

    @property
    def ssl_mode(self) -> str | None:
        return self._ssl_mode

    @model_validator(mode="after")
    def normalize_database_url(self) -> "DatabaseConfig":
        """Normalize DATABASE_URL for asyncpg (single-pass).

        - Validate scheme is one of postgresql, postgres, postgresql+asyncpg
        - Validate netloc is non-empty (catches malformed URLs at startup)
        - Normalize scheme to postgresql+asyncpg://
        - Extract sslmode and store it (passed via connect_args, not URL)
        - Strip sslmode and channel_binding (asyncpg doesn't accept them as query params)

        Mutates DATABASE_URL in place; expected to run once at config instantiation.
        """
        parsed = urlsplit(self.DATABASE_URL)

        if parsed.scheme not in {"postgresql", "postgres", "postgresql+asyncpg"}:
            raise ValueError(
                f"DATABASE_URL has unsupported scheme {parsed.scheme!r}; "
                "expected one of: postgresql, postgres, postgresql+asyncpg"
            )
        if not parsed.netloc:
            raise ValueError(
                f"DATABASE_URL is missing a host component; got {self.DATABASE_URL!r}"
            )

        params = parse_qs(parsed.query)
        ssl_values = params.get("sslmode")
        if ssl_values:
            self._ssl_mode = ssl_values[0]

        params.pop("sslmode", None)
        params.pop("channel_binding", None)

        scheme = "postgresql+asyncpg"
        new_query = urlencode(params, doseq=True)
        self.DATABASE_URL = urlunsplit(
            (scheme, parsed.netloc, parsed.path, new_query, parsed.fragment)
        )
        return self

    @property
    def asyncpg_url(self) -> str:
        """Return the URL without the +asyncpg driver suffix for raw asyncpg usage.

        Only the scheme's ``+asyncpg`` suffix is stripped, so credentials that
        legitimately contain ``+asyncpg`` (e.g. generated passwords) are preserved.
        """
        parsed = urlsplit(self.DATABASE_URL)
        scheme = parsed.scheme.replace("+asyncpg", "")
        return urlunsplit(
            (scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment)
        )


class LLMConfig(BaseSettings):
    """Large Language Model configuration."""

    OPEN_ROUTER_API_KEY: str | None = Field(default=None)
    OPENROUTER_API_KEY: str | None = Field(default=None)
    OPEN_ROUTER_API_URL: str = Field(default="https://openrouter.ai/api/v1")
    BASE_URL: str | None = Field(default=None)

    CHAT_MODEL: str = Field(
        default="openai/gpt-4o-mini", description="Model name for chat completions"
    )
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small", description="Model name for embeddings"
    )
    EMBEDDING_DIM: int = Field(
        default=1536, description="Dimension of the embedding vectors"
    )
    MAX_TOKEN_SIZE: int = Field(
        default=8192, description="Maximum token size for the embedding model"
    )
    VISION_MODEL: str = Field(
        default="openai/gpt-4o", description="Model name for vision tasks"
    )

    @property
    def api_key(self) -> str:
        """Get API key with fallback."""
        key = self.OPEN_ROUTER_API_KEY or self.OPENROUTER_API_KEY
        if not key:
            logger.warning("OPENROUTER_API_KEY not set. API calls will fail.")
        return key or ""

    @property
    def api_base_url(self) -> str:
        """Get API base URL with fallback."""
        return self.BASE_URL or self.OPEN_ROUTER_API_URL


class KreuzbergConfig(BaseSettings):
    """Kreuzberg extraction configuration (shared by Files, Classical)."""

    KREUZBERG_OCR_MODE: str = Field(
        default="vlm",
        description="OCR backend for Kreuzberg: 'vlm' or 'tesseract'",
    )
    KREUZBERG_EXTRACTION_TIMEOUT: int = Field(
        default=120,
        description="Timeout en secondes pour l'extraction Kreuzberg (via extraction_timeout_secs natif). Défaut: 120 (2 min)",
        gt=0,
    )
    KREUZBERG_EXTRACT_IMAGES: bool = Field(
        default=False,
        description="Extraire les images embarquées des PDF (rendu CPU coûteux). L'OCR des pages scannées reste actif via ocr_mode",
    )
    KREUZBERG_ENABLE_QUALITY_PROCESSING: bool = Field(
        default=False,
        description="Activer la retraite qualité locale Kreuzberg (CPU coûteux)",
    )
    KREUZBERG_VLM_TIMEOUT: int = Field(
        default=60,
        description="Timeout en secondes pour chaque appel OCR VLM (par image/page envoyée au modèle de vision)",
        gt=0,
    )


class BM25Config(BaseSettings):
    """BM25 search configuration."""

    BM25_ENABLED: bool = Field(default=True, description="Enable BM25 full-text search")
    BM25_TEXT_CONFIG: str = Field(
        default="english", description="PostgreSQL text search configuration"
    )
    BM25_RRF_K: int = Field(
        default=60, ge=1, description="RRF constant K for hybrid search"
    )


class ClassicalRAGConfig(BaseSettings):
    """Configuration for the classical RAG pathway."""

    CLASSICAL_CHUNK_SIZE: int = Field(
        default=1000, description="Max characters per chunk (Kreuzberg ChunkingConfig)"
    )
    CLASSICAL_CHUNK_OVERLAP: int = Field(
        default=200, description="Overlap characters between chunks"
    )
    CLASSICAL_NUM_QUERY_VARIATIONS: int = Field(
        default=3,
        description="Number of multi-query variations to generate",
        ge=1,
        le=10,
    )
    CLASSICAL_RELEVANCE_THRESHOLD: float = Field(
        default=5.0,
        description="Minimum LLM judge score (0-10) to include a chunk",
        ge=0.0,
        le=10.0,
    )
    CLASSICAL_TABLE_PREFIX: str = Field(
        default="classical_rag_", description="Prefix for PGVectorStore table names"
    )
    CLASSICAL_LLM_TEMPERATURE: float = Field(
        default=0.0,
        description="Temperature for LLM calls (multi-query + judge)",
        ge=0.0,
        le=2.0,
    )
    CLASSICAL_RRF_K: int = Field(
        default=60,
        ge=1,
        description="RRF constant K for hybrid BM25+vector search",
    )


class MinioConfig(BaseSettings):
    """MinIO object storage configuration."""

    MINIO_HOST: str = Field(default="localhost:9000")
    MINIO_ACCESS: str = Field(default="minioadmin")
    MINIO_SECRET: str = Field(default="minioadmin")
    MINIO_BUCKET: str = Field(default="raganything")
    MINIO_SECURE: bool = Field(default=False)
    # chat-pdf-attachments spec §5.3/REQ-092 (MCP side): maximum decoded size of
    # a temporary upload accepted by ``temporary_upload_file`` (50 MiB default).
    FILES_MAX_UPLOAD_SIZE: int = Field(
        default=52_428_800,
        ge=1,
        description="Maximum decoded size (bytes) accepted by the temporary-file "
        "MCP tool temporary_upload_file. Defaults to 52428800 (50 MiB).",
    )


class WebSearchConfig(BaseSettings):
    """Bounded search execution limits and provider backend selection.

    All values are per-Python-process; they do not coordinate across Uvicorn
    workers or replicas. ``WEB_SEARCH_BACKENDS`` is an ordered ddgs backend
    list (comma-separated env var): the first backend returning entries wins,
    so production can ride out one provider's anti-bot challenges. The
    ``WEB_SEARCH_PAGE_MAX_CHARS`` cap bounds the exploration output size
    (0 disables the cap and restores the complete-Markdown contract).
    """

    WEB_SEARCH_BACKENDS: Annotated[list[str], NoDecode] = Field(
        default=["duckduckgo"],
        description="Ordered ddgs backends tried per search, comma-separated "
        "(e.g. 'google,duckduckgo'). The first backend returning entries "
        "wins; present-but-empty falls back to the first default entry.",
    )
    WEB_SEARCH_PAGE_MAX_CHARS: int = Field(
        default=0,
        ge=0,
        description="Per-explored-page Markdown cap in characters. 0 "
        "disables the cap (complete Crawl4AI Markdown contract preserved).",
    )
    WEB_SEARCH_EXECUTOR_WORKERS: int = Field(
        default=4,
        ge=1,
        description="Dedicated ThreadPoolExecutor workers for synchronous "
        "DDGS calls (never the default executor).",
    )
    WEB_SEARCH_MAX_CONCURRENCY: int = Field(
        default=4,
        ge=1,
        description="Process-wide cap on active DDGS calls in flight; timed-out "
        "calls keep their capacity until their worker thread finishes.",
    )
    WEB_SEARCH_TIMEOUT_SECONDS: float = Field(
        default=10.0,
        gt=0,
        description="Outer asyncio timeout (seconds) around the complete "
        "search operation (capacity wait and the full backend fallback "
        "chain included).",
    )

    @pydantic_field_validator("WEB_SEARCH_BACKENDS", mode="before")
    @classmethod
    def _coerce_backend_list(cls, value: object) -> object:
        """Coerce the ``WEB_SEARCH_BACKENDS`` comma-separated env value.

        A present-but-empty value falls back to the first default entry: no
        backend would mean a permanently empty search, so startup with the
        documented default is safer than a silent outage.
        """
        parsed = _coerce_csv_list_value(
            value, field_name="WEB_SEARCH_BACKENDS", refuse_empty=False
        )
        if isinstance(parsed, list) and not parsed:
            return ["duckduckgo"]
        return parsed


class CrawlConfig(BaseSettings):
    """Bounded Crawl4AI browser exploration limits (process-local)."""

    CRAWL_MAX_CONCURRENT_BATCHES: int = Field(
        default=2,
        ge=1,
        description="Process-wide cap on concurrent crawl batches (semaphore "
        "held across batch startup and full stream consumption).",
    )
    CRAWL_MAX_SESSIONS_PER_BATCH: int = Field(
        default=2,
        ge=1,
        description="MemoryAdaptiveDispatcher max_session_permit per batch.",
    )
    CRAWL_MEMORY_THRESHOLD_PERCENT: int = Field(
        default=80,
        ge=1,
        le=90,
        description="MemoryAdaptiveDispatcher memory threshold percent (kept "
        "below the dispatcher's default 95% critical threshold).",
    )
    CRAWL_PAGE_TIMEOUT_SECONDS: float = Field(
        default=20.0,
        gt=0,
        description="Finite per-page timeout (seconds), converted to "
        "milliseconds for CrawlerRunConfig.page_timeout.",
    )
    CRAWL_OVERALL_TIMEOUT_SECONDS: float = Field(
        default=45.0,
        gt=0,
        description="Whole exploration deadline (seconds) covering batch "
        "startup and complete streaming consumption; kept below "
        "composable-agents' 60s MCP tool timeout. Completed pages survive it.",
    )
    CRAWL_RATE_LIMIT_MIN_DELAY_SECONDS: float = Field(
        default=1.0,
        gt=0,
        description="Shared RateLimiter base delay lower bound (seconds).",
    )
    CRAWL_RATE_LIMIT_MAX_DELAY_SECONDS: float = Field(
        default=2.0,
        gt=0,
        description="Shared RateLimiter base delay upper bound (seconds).",
    )

    @model_validator(mode="after")
    def validate_rate_limit_bounds(self) -> "CrawlConfig":
        """Ensure the minimum rate-limit delay does not exceed the maximum."""
        if (
            self.CRAWL_RATE_LIMIT_MIN_DELAY_SECONDS
            > self.CRAWL_RATE_LIMIT_MAX_DELAY_SECONDS
        ):
            raise ValueError(
                "CRAWL_RATE_LIMIT_MIN_DELAY_SECONDS must not exceed "
                "CRAWL_RATE_LIMIT_MAX_DELAY_SECONDS"
            )
        return self


class MemoryStoreConfig(BaseSettings):
    """composable-agents Store File API client configuration."""

    COMPOSABLE_AGENTS_BASE_URL: str = Field(
        default="http://localhost:8010",
        description="Base URL of the composable-agents Store File API used to "
        "persist per-user memories (e.g. http://composable-agents:8000 in "
        "Compose deployments).",
    )
    MEMORY_STORE_TIMEOUT_SECONDS: float = Field(
        default=15.0,
        gt=0,
        description="Upstream Store File API request timeout (seconds).",
    )

    @model_validator(mode="after")
    def validate_base_url(self) -> "MemoryStoreConfig":
        """Require a valid absolute HTTP/HTTPS base scheme+host URL."""
        parsed = urlsplit(self.COMPOSABLE_AGENTS_BASE_URL)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "COMPOSABLE_AGENTS_BASE_URL must be a valid absolute http(s) URL"
            )
        if parsed.path not in ("", "/"):
            # httpx concatenates a base URL path with relative request paths,
            # which would corrupt the Store File API route — reject it early.
            raise ValueError(
                "COMPOSABLE_AGENTS_BASE_URL must not include a path; only "
                "scheme and host are supported (e.g. http://composable-agents:8000)"
            )
        return self
