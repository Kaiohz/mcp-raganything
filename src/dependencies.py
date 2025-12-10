"""
Dependency injection setup for the application.
Follows the pickpro_indexing_api pattern for wiring components.
"""
import os
import tempfile
import tempfile

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from config import DatabaseConfig, LLMConfig, RAGConfig, AppConfig
from infrastructure.rag.lightrag_adapter import LightRAGAdapter
from domain.services.indexing_service import IndexingService


from domain.services.query_service import QueryService
from application.use_cases.index_file_use_case import IndexFileUseCase
from application.use_cases.index_folder_use_case import IndexFolderUseCase
from application.use_cases.query_use_case import QueryUseCase


# ============= CONFIG INITIALIZATION =============

app_config = AppConfig()  # type: ignore
db_config = DatabaseConfig()  # type: ignore
llm_config = LLMConfig()  # type: ignore
rag_config = RAGConfig()  # type: ignore

# ============= ENVIRONMENT SETUP =============

os.environ["POSTGRES_USER"] = db_config.POSTGRES_USER
os.environ["POSTGRES_PASSWORD"] = db_config.POSTGRES_PASSWORD
os.environ["POSTGRES_DATABASE"] = db_config.POSTGRES_DATABASE
os.environ["POSTGRES_HOST"] = db_config.POSTGRES_HOST
os.environ["POSTGRES_PORT"] = db_config.POSTGRES_PORT

# ============= DIRECTORIES =============

WORKING_DIR = os.path.join(tempfile.gettempdir(), "rag_storage")
os.makedirs(WORKING_DIR, exist_ok=True)
OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============= DATABASE ENGINE =============


# ============= RAG SETUP =============


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    """LLM function for RAGAnything."""
    return await openai_complete_if_cache(
        llm_config.CHAT_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=llm_config.api_key,
        base_url=llm_config.api_base_url,
        **kwargs,
    )


async def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs):
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
            url = img if isinstance(img, str) and img.startswith("http") else f"data:image/jpeg;base64,{img}"
            content.append({
                "type": "image_url",
                "image_url": {"url": url}
            })

    messages.append({"role": "user", "content": content})

    return await openai_complete_if_cache(
        llm_config.VISION_MODEL,
        "Image Description Task", # Dummy prompt to avoid lightrag appending None
        system_prompt=None,
        history_messages=messages,
        api_key=llm_config.api_key,
        base_url=llm_config.api_base_url,
        messages=messages, # Explicitly pass constructed messages
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
    lightrag_kwargs={
        "kv_storage": "PGKVStorage",
        "vector_storage": "PGVectorStorage",
        "graph_storage": "PGGraphStorage",
        "doc_status_storage": "PGDocStatusStorage",
        "cosine_threshold": rag_config.COSINE_THRESHOLD,
    } if rag_config.RAG_STORAGE_TYPE == "postgres" else {
        "kv_storage": "JsonKVStorage",
        "vector_storage": "NanoVectorDBStorage",
        "graph_storage": "NetworkXStorage",
        "doc_status_storage": "JsonDocStatusStorage",
        "cosine_threshold": rag_config.COSINE_THRESHOLD,
    }
)

# ============= ADAPTERS =============

rag_adapter = LightRAGAdapter(rag_instance, rag_config.MAX_WORKERS)

# ============= SERVICES =============

indexing_service = IndexingService(
    rag_engine=rag_adapter
)

query_service = QueryService(
    rag_engine=rag_adapter
)

# ============= DEPENDENCY INJECTION FUNCTIONS =============


async def get_index_file_use_case() -> IndexFileUseCase:
    """
    Dependency injection function for IndexFileUseCase.

    Returns:
        IndexFileUseCase: The configured use case.
    """
    return IndexFileUseCase(indexing_service)


async def get_index_folder_use_case() -> IndexFolderUseCase:
    """
    Dependency injection function for IndexFolderUseCase.

    Returns:
        IndexFolderUseCase: The configured use case.
    """
    return IndexFolderUseCase(indexing_service)


async def get_query_use_case() -> QueryUseCase:
    """
    Dependency injection function for QueryUseCase.

    Returns:
        QueryUseCase: The configured use case.
    """
    return QueryUseCase(query_service)
