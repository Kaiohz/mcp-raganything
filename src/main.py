"""Main entry point for the RAGAnything API.
Simplified following hexagonal architecture pattern from pickpro_indexing_api.
"""

import contextlib
import threading
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from config import AppConfig
from dependencies import rag_adapter
from infrastructure.proxy.lightrag_proxy_client import (
    init_lightrag_client,
    close_lightrag_client,
)
from application.api.indexing_routes import indexing_router
from application.api.health_routes import health_router
from application.api.lightrag_proxy_routes import lightrag_proxy_router
from application.api.mcp_tools import mcp
import uvicorn

logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifespan of the application."""
    async with contextlib.AsyncExitStack() as stack:
        # Initialize RAG engine
        await rag_adapter.initialize()
        # Initialize LightRAG proxy client
        await init_lightrag_client()
        logger.info("LightRAG proxy client initialized")
        yield
        # Cleanup
        await close_lightrag_client()
        logger.info("LightRAG proxy client closed")


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
app.include_router(health_router, prefix=REST_PATH)
app.include_router(
    lightrag_proxy_router, prefix=f"{REST_PATH}/lightrag", tags=["LightRAG Proxy"]
)


# ============= CUSTOM OPENAPI WITH LIGHTRAG PROXY =============


def custom_openapi():
    """
    Generate custom OpenAPI schema that includes LightRAG API endpoints.
    Fetches LightRAG's OpenAPI spec and merges it with the local spec.
    """
    if app.openapi_schema:
        return app.openapi_schema

    # Generate base OpenAPI schema
    openapi_schema = get_openapi(
        title="RAG Anything API",
        version="1.0.0",
        description="""
## RAG Anything API

This API provides RAG (Retrieval-Augmented Generation) capabilities with LightRAG integration.

### Features:
- **Indexing**: Index files and folders for RAG
- **Query**: Query the RAG system
- **LightRAG Proxy**: Access all LightRAG API endpoints under `/api/v1/lightrag/*`

### LightRAG Proxy
All LightRAG endpoints are available under the `/api/v1/lightrag/` prefix.
Authorization tokens are automatically forwarded to LightRAG.
        """,
        routes=app.routes,
    )

    # Try to fetch and merge LightRAG OpenAPI spec
    try:
        import httpx
        from config import ProxyConfig

        proxy_config = ProxyConfig()  # type: ignore

        # Synchronous fetch for OpenAPI generation
        with httpx.Client(timeout=10) as client:
            response = client.get(f"{proxy_config.LIGHTRAG_API_URL}/openapi.json")
            if response.status_code == 200:
                lightrag_spec = response.json()

                # Add LightRAG paths with /api/v1/lightrag prefix
                lightrag_paths = lightrag_spec.get("paths", {})
                for path, path_item in lightrag_paths.items():
                    proxy_path = f"/api/v1/lightrag{path}"

                    # Add tag to identify LightRAG endpoints
                    for method_data in path_item.values():
                        if isinstance(method_data, dict):
                            existing_tags = method_data.get("tags", [])
                            method_data["tags"] = ["LightRAG Proxy"] + [
                                f"LightRAG - {tag}" for tag in existing_tags
                            ]

                    openapi_schema["paths"][proxy_path] = path_item

                # Merge components/schemas
                if "components" in lightrag_spec:
                    if "components" not in openapi_schema:
                        openapi_schema["components"] = {}

                    lightrag_schemas = lightrag_spec["components"].get("schemas", {})
                    if "schemas" not in openapi_schema["components"]:
                        openapi_schema["components"]["schemas"] = {}

                    # Add LightRAG schemas with prefix to avoid conflicts
                    for schema_name, schema_def in lightrag_schemas.items():
                        prefixed_name = f"LightRAG_{schema_name}"
                        openapi_schema["components"]["schemas"][
                            prefixed_name
                        ] = schema_def

                    # Update $ref references in LightRAG paths
                    def update_refs(obj):
                        if isinstance(obj, dict):
                            for key, value in obj.items():
                                if key == "$ref" and isinstance(value, str):
                                    if value.startswith("#/components/schemas/"):
                                        schema_name = value.split("/")[-1]
                                        obj[key] = (
                                            f"#/components/schemas/LightRAG_{schema_name}"
                                        )
                                else:
                                    update_refs(value)
                        elif isinstance(obj, list):
                            for item in obj:
                                update_refs(item)

                    # Update refs in lightrag paths
                    for path in openapi_schema["paths"]:
                        if path.startswith("/api/v1/lightrag"):
                            update_refs(openapi_schema["paths"][path])

                logger.info(
                    f"Merged {len(lightrag_paths)} LightRAG endpoints into OpenAPI spec"
                )

    except Exception as e:
        logger.warning(f"Failed to fetch LightRAG OpenAPI spec: {e}")

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# ============= MCP MOUNTING =============

MCP_PATH = "/mcp"

if app_config.MCP_TRANSPORT == "streamable":
    app.mount(MCP_PATH, mcp.streamable_http_app())
elif app_config.MCP_TRANSPORT == "sse":
    app.mount(MCP_PATH, mcp.sse_app())

# ============= MAIN =============

def run_fastapi():
    uvicorn.run(
        app,
        host=app_config.HOST,
        port=app_config.PORT,
        log_level="critical",
        access_log=False,
    )

if __name__ == "__main__":
    api_thread = threading.Thread(target=run_fastapi, daemon=True)
    api_thread.start()

    if app_config.MCP_TRANSPORT == "stdio":
        mcp.run(transport="stdio")
