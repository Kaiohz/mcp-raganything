"""
Main entry point for the RAGAnything API.
Simplified following hexagonal architecture pattern from pickpro_indexing_api.
"""
import contextlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from config import AppConfig
from dependencies import postgres_engine, rag_adapter
from application.api.indexing_routes import indexing_router
from application.api.query_routes import query_router
from application.api.health_routes import health_router
from application.api.mcp_tools import mcp


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifespan of the application."""
    async with contextlib.AsyncExitStack() as stack:
        # Initialize database tables
        from infrastructure.database.models.base import Base
        async with postgres_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Initialize RAG engine
        await rag_adapter.initialize()
        
        yield


app = FastAPI(title="RAG Anything API", lifespan=lifespan)

app_config = AppConfig()  # type: ignore

app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= REST API ROUTES =============

REST_PATH = "/api/v1"

app.include_router(indexing_router, prefix=REST_PATH)
app.include_router(query_router, prefix=REST_PATH)
app.include_router(health_router, prefix=REST_PATH)

# ============= MCP MOUNTING =============

MCP_PATH = "/mcp"

if app_config.MCP_TRANSPORT == "streamable":
    app.mount(MCP_PATH, mcp.streamable_http_app())
elif app_config.MCP_TRANSPORT == "sse":
    app.mount(MCP_PATH, mcp.sse_app())

# ============= MAIN =============

if __name__ == "__main__":
    # Standard usage: stdio for Claude Desktop
    mcp.run(transport="stdio")
