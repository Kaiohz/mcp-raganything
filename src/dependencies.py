"""
Dependency injection setup for the application.
Follows the pickpro_indexing_api pattern for wiring components.
"""
import os
import tempfile
from sqlalchemy.ext.asyncio import create_async_engine
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from config import DatabaseConfig, LLMConfig, RAGConfig, AppConfig
from infrastructure.rag.lightrag_adapter import LightRAGAdapter
from infrastructure.database.document_postgres import DocumentPostgres
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

postgres_engine = create_async_engine(db_config.DATABASE_URL, echo=False)

# ============= RAG SETUP =============


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    """LLM function for RAGAnything."""
    return await openai_complete_if_cache(
        "openai/gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=llm_config.api_key,
        base_url=llm_config.api_base_url,
        **kwargs,
    )


embedding_func = EmbeddingFunc(
    embedding_dim=1536,
    max_token_size=8192,
    func=lambda texts: openai_embed(
        texts,
        model="text-embedding-3-small",
        api_key=llm_config.api_key,
        base_url=llm_config.api_base_url,
    ),
)

raganything_config = RAGAnythingConfig(
    working_dir=WORKING_DIR,
    parser="docling",
    parse_method="auto",
    enable_image_processing=True,
    enable_table_processing=True,
    enable_equation_processing=True,
    max_concurrent_files=rag_config.MAX_CONCURRENT_FILES,
)

rag_instance = RAGAnything(
    config=raganything_config,
    llm_model_func=llm_model_func,
    embedding_func=embedding_func,
    lightrag_kwargs={
        "kv_storage": "PGKVStorage",
        "vector_storage": "PGVectorStorage",
        "graph_storage": "PGGraphStorage",
        "doc_status_storage": "PGDocStatusStorage",
        "cosine_threshold": rag_config.COSINE_THRESHOLD,
    }
)

# ============= ADAPTERS =============

rag_adapter = LightRAGAdapter(rag_instance)
document_repo = DocumentPostgres(postgres_engine)

# ============= SERVICES =============

indexing_service = IndexingService(
    rag_engine=rag_adapter,
    document_repo=document_repo
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
