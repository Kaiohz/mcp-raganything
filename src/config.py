from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()


class AppConfig(BaseSettings):
    """
    Application configuration settings.
    """

    ALLOWED_ORIGINS: List[str] = Field(
        default=["*"], description="CORS allowed origins"
    )
    MCP_TRANSPORT: str = Field(
        default="stdio", description="MCP transport mode: stdio, sse, streamable"
    )
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    UVICORN_LOG_LEVEL: str = Field(
        default="critical", description="Uvicorn log level when running with MCP stdio"
    )


class DatabaseConfig(BaseSettings):
    """
    Database connection configuration.
    """

    POSTGRES_USER: str = Field(default="raganything")
    POSTGRES_PASSWORD: str = Field(default="raganything")
    POSTGRES_DATABASE: str = Field(default="raganything")
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: str = Field(default="5432")

    @property
    def DATABASE_URL(self) -> str:
        """Construct async PostgreSQL database URL."""
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DATABASE}"


class LLMConfig(BaseSettings):
    """
    Large Language Model configuration (Generic - supports multiple providers).
    """

    # Generic LLM Configuration
    LLM_BINDING: str = Field(
        default="openai",
        description="LLM provider binding: openai, ollama, azure, gemini, lollms"
    )
    LLM_API_KEY: Optional[str] = Field(default=None)
    LLM_BASE_URL: Optional[str] = Field(default="https://api.openai.com/v1")
    LLM_MODEL_NAME: str = Field(
        default="gpt-4o-mini", description="Model name for chat completions"
    )

    # Legacy support (deprecated - for backward compatibility)
    OPEN_ROUTER_API_KEY: Optional[str] = Field(default=None)
    OPENROUTER_API_KEY: Optional[str] = Field(default=None)
    OPEN_ROUTER_API_URL: str = Field(default="https://openrouter.ai/api/v1")
    BASE_URL: Optional[str] = Field(default=None)
    CHAT_MODEL: Optional[str] = Field(default=None)  # Deprecated, use LLM_MODEL_NAME

    # Embedding Configuration
    EMBEDDING_BINDING: str = Field(
        default="openai",
        description="Embedding provider binding: openai, ollama, azure"
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

    # Vision Model Configuration
    VISION_MODEL: str = Field(
        default="gpt-4o", description="Model name for vision tasks"
    )

    @property
    def api_key(self) -> str:
        """Get API key with fallback to legacy env vars."""
        key = self.LLM_API_KEY or self.OPEN_ROUTER_API_KEY or self.OPENROUTER_API_KEY
        if not key:
            print("WARNING: LLM_API_KEY not set. API calls will fail.")
        return key or ""

    @property
    def api_base_url(self) -> str:
        """Get API base URL with fallback to legacy env vars."""
        return self.LLM_BASE_URL or self.BASE_URL or self.OPEN_ROUTER_API_URL

    @property
    def model_name(self) -> str:
        """Get model name with fallback to legacy env vars."""
        return self.LLM_MODEL_NAME or self.CHAT_MODEL or "gpt-4o-mini"


class StorageConfig(BaseSettings):
    """
    Storage configuration for LightRAG (4 types: Vector, Graph, KV, DocStatus).
    """

    # Vector Storage Configuration
    VECTOR_STORAGE_TYPE: str = Field(
        default="pgvector",
        description="Vector storage type: pgvector, qdrant, milvus, local"
    )

    # Graph Storage Configuration (CRITICAL - for knowledge graph)
    GRAPH_STORAGE_TYPE: str = Field(
        default="postgres",
        description="Graph storage type: postgres, neo4j, networkx, memgraph"
    )

    # KV Storage Configuration (for LLM cache, chunks, documents)
    KV_STORAGE_TYPE: str = Field(
        default="postgres",
        description="KV storage type: postgres, redis, mongo, json"
    )

    # Doc Status Storage Configuration
    DOC_STATUS_STORAGE_TYPE: str = Field(
        default="postgres",
        description="Doc status storage type: postgres, mongo, json"
    )

    # Qdrant Configuration (if VECTOR_STORAGE_TYPE=qdrant)
    QDRANT_URL: Optional[str] = Field(default="http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = Field(default=None)

    # Neo4j Configuration (if GRAPH_STORAGE_TYPE=neo4j)
    NEO4J_URI: Optional[str] = Field(default="bolt://localhost:7687")
    NEO4J_USERNAME: Optional[str] = Field(default="neo4j")
    NEO4J_PASSWORD: Optional[str] = Field(default=None)

    # Redis Configuration (if KV_STORAGE_TYPE=redis)
    REDIS_URI: Optional[str] = Field(default="redis://localhost:6379")

    # Vector Index Configuration
    VECTOR_INDEX_TYPE: str = Field(default="HNSW")
    HNSW_M: int = Field(default=16)
    HNSW_EF: int = Field(default=64)


class RAGConfig(BaseSettings):
    """
    RAG-specific configuration for LightRAG.
    """

    COSINE_THRESHOLD: float = Field(
        default=0.2, description="Similarity threshold for vector search (0.0-1.0)"
    )
    MAX_CONCURRENT_FILES: int = Field(
        default=1, description="Number of files to process concurrently"
    )
    ENABLE_IMAGE_PROCESSING: bool = Field(
        default=True, description="Enable image processing during indexing"
    )
    ENABLE_TABLE_PROCESSING: bool = Field(
        default=True, description="Enable table processing during indexing"
    )
    ENABLE_EQUATION_PROCESSING: bool = Field(
        default=True, description="Enable equation processing during indexing"
    )
    MAX_WORKERS: int = Field(
        default=3, description="Number of workers for folder processing"
    )
    # Legacy (deprecated - use StorageConfig.VECTOR_STORAGE_TYPE)
    RAG_STORAGE_TYPE: str = Field(
        default="postgres", description="[DEPRECATED] Use VECTOR_STORAGE_TYPE instead"
    )


class ProxyConfig(BaseSettings):
    """
    Configuration for the LightRAG API proxy.
    """

    LIGHTRAG_API_URL: str = Field(
        default="http://localhost:9621", description="LightRAG API base URL"
    )
    LIGHTRAG_TIMEOUT: int = Field(
        default=60, description="Default timeout for proxy requests in seconds"
    )
    LIGHTRAG_STREAM_TIMEOUT: int = Field(
        default=300, description="Timeout for streaming proxy requests in seconds"
    )
