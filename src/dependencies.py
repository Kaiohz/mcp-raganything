"""
Dependency injection setup for the application.
Follows the pickpro_indexing_api pattern for wiring components.
"""

import os
import tempfile
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from config import DatabaseConfig, LLMConfig, RAGConfig, AppConfig, ProxyConfig, StorageConfig
from infrastructure.rag.lightrag_adapter import LightRAGAdapter
from infrastructure.proxy.lightrag_proxy_client import (
    LightRAGProxyClient,
    get_lightrag_client_instance,
    init_lightrag_client,
    close_lightrag_client,
)
from application.use_cases.index_file_use_case import IndexFileUseCase
from application.use_cases.index_folder_use_case import IndexFolderUseCase
from application.use_cases.index_batch_use_case import IndexBatchUseCase
from application.use_cases.lightrag_proxy_use_case import LightRAGProxyUseCase


# ============= CONFIG INITIALIZATION =============

app_config = AppConfig()  # type: ignore
db_config = DatabaseConfig()  # type: ignore
llm_config = LLMConfig()  # type: ignore
rag_config = RAGConfig()  # type: ignore
proxy_config = ProxyConfig()  # type: ignore
storage_config = StorageConfig()  # type: ignore

# ============= ENVIRONMENT SETUP =============

os.environ["POSTGRES_USER"] = db_config.POSTGRES_USER
os.environ["POSTGRES_PASSWORD"] = db_config.POSTGRES_PASSWORD
os.environ["POSTGRES_DATABASE"] = db_config.POSTGRES_DATABASE
os.environ["POSTGRES_HOST"] = db_config.POSTGRES_HOST
os.environ["POSTGRES_PORT"] = db_config.POSTGRES_PORT

# ============= DIRECTORIES =============

# Read WORKING_DIR from environment variable, fallback to temp dir
WORKING_DIR = os.getenv("WORKING_DIR", os.path.join(tempfile.gettempdir(), "rag_storage"))
os.makedirs(WORKING_DIR, exist_ok=True)
OUTPUT_DIR = os.path.join(WORKING_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============= DATABASE ENGINE =============


# ============= RAG SETUP =============


def get_storage_config() -> dict:
    """
    Factory function to get storage configuration based on StorageConfig.
    Returns dict with all 4 storage types: vector, graph, kv, doc_status.
    """
    storage_dict = {}

    # 1. VECTOR STORAGE
    if storage_config.VECTOR_STORAGE_TYPE == "pgvector":
        storage_dict["vector_storage"] = "PGVectorStorage"
    elif storage_config.VECTOR_STORAGE_TYPE == "qdrant":
        if storage_config.QDRANT_URL:
            os.environ["QDRANT_URL"] = storage_config.QDRANT_URL
        if storage_config.QDRANT_API_KEY:
            os.environ["QDRANT_API_KEY"] = storage_config.QDRANT_API_KEY
        storage_dict["vector_storage"] = "QdrantVectorDBStorage"
    elif storage_config.VECTOR_STORAGE_TYPE == "milvus":
        storage_dict["vector_storage"] = "MilvusVectorDBStorage"
    elif storage_config.VECTOR_STORAGE_TYPE == "local":
        storage_dict["vector_storage"] = "NanoVectorDBStorage"
    else:
        storage_dict["vector_storage"] = "PGVectorStorage"  # Default fallback

    # 2. GRAPH STORAGE (CRITICAL - for knowledge graph)
    if storage_config.GRAPH_STORAGE_TYPE == "postgres":
        storage_dict["graph_storage"] = "PGGraphStorage"
    elif storage_config.GRAPH_STORAGE_TYPE == "neo4j":
        if storage_config.NEO4J_URI:
            os.environ["NEO4J_URI"] = storage_config.NEO4J_URI
        if storage_config.NEO4J_USERNAME:
            os.environ["NEO4J_USERNAME"] = storage_config.NEO4J_USERNAME
        if storage_config.NEO4J_PASSWORD:
            os.environ["NEO4J_PASSWORD"] = storage_config.NEO4J_PASSWORD
        storage_dict["graph_storage"] = "Neo4JStorage"
    elif storage_config.GRAPH_STORAGE_TYPE == "networkx":
        storage_dict["graph_storage"] = "NetworkXStorage"
    elif storage_config.GRAPH_STORAGE_TYPE == "memgraph":
        if storage_config.NEO4J_URI:
            os.environ["NEO4J_URI"] = storage_config.NEO4J_URI
        if storage_config.NEO4J_USERNAME:
            os.environ["NEO4J_USERNAME"] = storage_config.NEO4J_USERNAME
        if storage_config.NEO4J_PASSWORD:
            os.environ["NEO4J_PASSWORD"] = storage_config.NEO4J_PASSWORD
        storage_dict["graph_storage"] = "MemgraphStorage"
    else:
        storage_dict["graph_storage"] = "PGGraphStorage"  # Default fallback

    # 3. KV STORAGE (for LLM cache, chunks, documents)
    if storage_config.KV_STORAGE_TYPE == "postgres":
        storage_dict["kv_storage"] = "PGKVStorage"
    elif storage_config.KV_STORAGE_TYPE == "redis":
        if storage_config.REDIS_URI:
            os.environ["REDIS_URI"] = storage_config.REDIS_URI
        storage_dict["kv_storage"] = "RedisKVStorage"
    elif storage_config.KV_STORAGE_TYPE == "mongo":
        storage_dict["kv_storage"] = "MongoKVStorage"
    elif storage_config.KV_STORAGE_TYPE == "json":
        storage_dict["kv_storage"] = "JsonKVStorage"
    else:
        storage_dict["kv_storage"] = "PGKVStorage"  # Default fallback

    # 4. DOC STATUS STORAGE
    if storage_config.DOC_STATUS_STORAGE_TYPE == "postgres":
        storage_dict["doc_status_storage"] = "PGDocStatusStorage"
    elif storage_config.DOC_STATUS_STORAGE_TYPE == "mongo":
        storage_dict["doc_status_storage"] = "MongoDocStatusStorage"
    elif storage_config.DOC_STATUS_STORAGE_TYPE == "json":
        storage_dict["doc_status_storage"] = "JsonDocStatusStorage"
    else:
        storage_dict["doc_status_storage"] = "PGDocStatusStorage"  # Default fallback

    # Add cosine threshold
    storage_dict["cosine_threshold"] = rag_config.COSINE_THRESHOLD

    return storage_dict


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    """LLM function for RAGAnything - supports multiple providers."""
    return await openai_complete_if_cache(
        llm_config.model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=llm_config.api_key,
        base_url=llm_config.api_base_url,
        **kwargs,
    )


async def vision_model_func(
    prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs
):
    """Vision function for RAGAnything."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)

    content = [{"type": "text", "text": prompt}]

    if image_data:
        images = image_data if isinstance(image_data, list) else [image_data]
        for img in images:
            # Simple heuristic: if it looks like a URL, use it; otherwise assume base64
            url = (
                img
                if isinstance(img, str) and img.startswith("http")
                else f"data:image/jpeg;base64,{img}"
            )
            content.append({"type": "image_url", "image_url": {"url": url}})

    messages.append({"role": "user", "content": content})

    return await openai_complete_if_cache(
        llm_config.VISION_MODEL,
        "Image Description Task",  # Dummy prompt to avoid lightrag appending None
        system_prompt=None,
        history_messages=messages,
        api_key=llm_config.api_key,
        base_url=llm_config.api_base_url,
        messages=messages,  # Explicitly pass constructed messages
        **kwargs,
    )


embedding_func = EmbeddingFunc(
    embedding_dim=llm_config.EMBEDDING_DIM,
    max_token_size=llm_config.MAX_TOKEN_SIZE,
    func=lambda texts: openai_embed(
        texts,
        model=llm_config.EMBEDDING_MODEL,
        api_key=llm_config.api_key,
        base_url=llm_config.api_base_url,
    ),
)

raganything_config = RAGAnythingConfig(
    working_dir=WORKING_DIR,
    parser="docling",
    parse_method="auto",
    enable_image_processing=rag_config.ENABLE_IMAGE_PROCESSING,
    enable_table_processing=rag_config.ENABLE_TABLE_PROCESSING,
    enable_equation_processing=rag_config.ENABLE_EQUATION_PROCESSING,
    max_concurrent_files=rag_config.MAX_CONCURRENT_FILES,
)

rag_instance = RAGAnything(
    config=raganything_config,
    llm_model_func=llm_model_func,
    vision_model_func=vision_model_func,
    embedding_func=embedding_func,
    lightrag_kwargs=get_storage_config(),
)

# ============= ADAPTERS =============

rag_adapter = LightRAGAdapter(rag_instance, rag_config.MAX_WORKERS)

# ============= DEPENDENCY INJECTION FUNCTIONS =============


async def get_index_file_use_case() -> IndexFileUseCase:
    """
    Dependency injection function for IndexFileUseCase.

    Returns:
        IndexFileUseCase: The configured use case.
    """
    return IndexFileUseCase(rag_adapter, OUTPUT_DIR)


async def get_index_folder_use_case() -> IndexFolderUseCase:
    """
    Dependency injection function for IndexFolderUseCase.

    Returns:
        IndexFolderUseCase: The configured use case.
    """
    return IndexFolderUseCase(rag_adapter, OUTPUT_DIR)


async def get_index_batch_use_case() -> IndexBatchUseCase:
    """
    Dependency injection function for IndexBatchUseCase.

    Returns:
        IndexBatchUseCase: The configured use case for batch file indexing.
    """
    return IndexBatchUseCase(rag_adapter, OUTPUT_DIR)


async def get_lightrag_client() -> LightRAGProxyClient:
    """
    Dependency injection function for LightRAG proxy client.

    Returns:
        LightRAGProxyClient: The configured proxy client.
    """
    return get_lightrag_client_instance()


async def get_lightrag_proxy_use_case() -> LightRAGProxyUseCase:
    """
    Dependency injection function for LightRAGProxyUseCase.

    Returns:
        LightRAGProxyUseCase: The configured use case for proxying requests.
    """
    proxy_client = get_lightrag_client_instance()
    return LightRAGProxyUseCase(proxy_client)
