"""Main entry point for the RAGAnything API.
Simplified following hexagonal architecture pattern from pickpro_indexing_api.
"""
import contextlib
import sys
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import AppConfig
from dependencies import rag_adapter
from application.api.indexing_routes import indexing_router
from application.api.query_routes import query_router
from application.api.health_routes import health_router
from application.api.mcp_tools import mcp
import uvicorn



@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifespan of the application."""
    async with contextlib.AsyncExitStack() as stack:
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

    if app_config.MCP_TRANSPORT == "stdio":
        def run_fastapi():
            uvicorn.run(
                app, 
                host=app_config.HOST, 
                port=app_config.PORT,
                log_level="critical",
                access_log=False
            )
        
        api_thread = threading.Thread(target=run_fastapi, daemon=True)
        api_thread.start()
        
        mcp.run(transport="stdio")
    else:
        uvicorn.run(app, host=app_config.HOST, port=app_config.PORT, log_level=app_config.UVICORN_LOG_LEVEL)
