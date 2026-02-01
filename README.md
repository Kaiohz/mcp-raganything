# MCP-RAGAnything

A FastAPI application that provides a REST API and MCP server for Retrieval Augmented Generation (RAG) using the [RAG-Anything](https://github.com/HKUDS/RAG-Anything) library. Built with **hexagonal architecture** for maintainability and testability.

## Features

- 🔍 **Multi-modal document processing** — PDF, DOCX, PPTX, images, tables, equations via [Docling](https://github.com/DS4SD/docling)
- 📁 **Batch folder indexing** — Recursive directory traversal with file extension filtering
- � **Batch file upload** — Upload and index multiple files simultaneously
- 🎨 **Multimodal queries** — Query with images, tables, and equations (MCP tool)
- 🔌 **LightRAG proxy** — Full pass-through to [LightRAG Server](https://github.com/HKUDS/LightRAG) for queries and knowledge graph operations
- 🤖 **MCP server** — Claude Desktop integration with multiple query tools
- 🗄️ **Flexible storage backends** — PostgreSQL, Qdrant, Neo4j, Redis, MongoDB, or local storage
- 🤖 **Multiple LLM providers** — OpenAI, Ollama, Azure, Gemini, OpenRouter
- 🏗️ **Hexagonal architecture** — Clean separation of domain, application, and infrastructure layers

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI App                              │
├─────────────────────────────────────────────────────────────────┤
│  Application Layer                                               │
│  ├── api/           → REST routes + MCP tools                   │
│  ├── use_cases/     → IndexFile, IndexFolder, LightRAGProxy     │
│  └── requests/      → Input DTOs (Pydantic)                     │
├─────────────────────────────────────────────────────────────────┤
│  Domain Layer (pure Python, no external imports)                │
│  ├── entities/      → IndexingResult, LightRAGProxyEntities     │
│  └── ports/         → RAGEnginePort, LightRAGProxyClientPort    │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                            │
│  ├── rag/           → LightRAGAdapter (implements RAGEnginePort)│
│  └── proxy/         → LightRAGProxyClient (HTTP client)         │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- **Python 3.13+**
- **Docker & Docker Compose** (recommended)
- API key from one of these providers:
  - [OpenRouter](https://openrouter.ai/)
  - [OpenAI](https://platform.openai.com/)
  - [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
  - Or use local Ollama

## Quick Start with Docker Compose

1. **Clone and configure:**

    ```bash
    git clone https://github.com/Kaiohz/mcp-raganything.git
    cd mcp-raganything

    # Configure the API service
    cp .env.example .env
    # Edit .env and set your LLM provider credentials

    # Configure LightRAG server
    cp .env.lightrag.server.example .env.lightrag.server
    # Edit .env.lightrag.server and set LLM provider credentials
    ```

2. **Choose your storage backend(s):**

    Default (PostgreSQL only - simplest):
    ```bash
    docker-compose up -d
    ```

    Or with optional backends (Qdrant, Neo4j, Redis):
    ```bash
    # With Qdrant vector storage
    docker-compose --profile qdrant up -d

    # With Neo4j graph storage
    docker-compose --profile neo4j up -d

    # With all optional backends
    docker-compose --profile qdrant --profile neo4j --profile redis up -d
    ```

    This starts three containers:
    | Container | Port | Description |
    |-----------|------|-------------|
    | `postgres` | 5432 | PostgreSQL 16 with pgvector + Apache AGE |
    | `api` | 8000 | RAG-Anything FastAPI service |
    | `lightrag-server` | 9621 | LightRAG Server with Web UI |

    **Optional services** (with profiles):
    | Container | Port | Profile | Description |
    |-----------|------|---------|-------------|
    | `qdrant` | 6333 | `qdrant` | High-performance vector storage |
    | `neo4j` | 7474, 7687 | `neo4j` | Advanced graph database with APOC/GDS |
    | `redis` | 6379 | `redis` | Fast KV storage and caching |

3. **Verify services:**

    ```bash
    curl http://localhost:8000/api/v1/health
    curl http://localhost:9621/health
    ```

4. **Access documentation:**

    - **API Docs (Swagger):** http://localhost:8000/docs
    - **LightRAG Web UI:** http://localhost:9621
    - **LightRAG API Docs:** http://localhost:9621/docs

## Configuration

Configuration is managed via environment files. See [IMPLEMENTATION.md](IMPLEMENTATION.md) for detailed configuration guide.

### LLM Provider Configuration

The system supports multiple LLM providers. Configure in `.env`:

**OpenAI:**
```env
LLM_BINDING=openai
LLM_API_KEY=sk-...
LLM_MODEL_NAME=gpt-4-turbo
```

**Ollama (Local):**
```env
LLM_BINDING=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL_NAME=llama2
```

**Azure OpenAI:**
```env
LLM_BINDING=azure
LLM_BASE_URL=https://your-resource.openai.azure.com
LLM_API_KEY=your_azure_key
LLM_MODEL_NAME=gpt-4
```

**OpenRouter:**
```env
LLM_BINDING=openai
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-v1-...
LLM_MODEL_NAME=anthropic/claude-3.5-sonnet
```

### Storage Backend Configuration

The system uses **4 independent storage types**:

| Storage Type | Purpose | Options |
|--------------|---------|---------|
| **VECTOR_STORAGE** | Embeddings | `pgvector`, `qdrant`, `local` |
| **GRAPH_STORAGE** | Knowledge graph | `postgres`, `neo4j`, `networkx` |
| **KV_STORAGE** | Cache & chunks | `postgres`, `redis`, `json` |
| **DOC_STATUS_STORAGE** | Processing status | `postgres`, `mongodb`, `json` |

**Example configurations:**

**Full PostgreSQL (simplest):**
```env
VECTOR_STORAGE_TYPE=pgvector
GRAPH_STORAGE_TYPE=postgres
KV_STORAGE_TYPE=postgres
DOC_STATUS_STORAGE_TYPE=postgres
```

**High Performance:**
```env
VECTOR_STORAGE_TYPE=qdrant
GRAPH_STORAGE_TYPE=neo4j
KV_STORAGE_TYPE=redis
DOC_STATUS_STORAGE_TYPE=postgres
```

**Local Development:**
```env
VECTOR_STORAGE_TYPE=local
GRAPH_STORAGE_TYPE=networkx
KV_STORAGE_TYPE=json
DOC_STATUS_STORAGE_TYPE=json
```

See [`.env.example`](.env.example) for all configuration options.

## Usage

Full API documentation is available at **http://localhost:8000/docs** (Swagger UI).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/file/index` | POST | Index a single file (background) |
| `/api/v1/folder/index` | POST | Index a folder (background) |
| `/api/v1/batch/index` | POST | Index multiple files at once |
| `/api/v1/lightrag/*` | ALL | Proxy to LightRAG API (query, documents, etc.) |

### Batch File Upload

Upload and index multiple files simultaneously:

```bash
curl -X POST "http://localhost:8000/api/v1/batch/index" \
  -F "files=@document1.pdf" \
  -F "files=@document2.docx" \
  -F "files=@slides.pptx"
```

**Response:**
```json
{
  "file_count": 3,
  "file_names": ["document1.pdf", "document2.docx", "slides.pptx"],
  "message": "Batch indexing started in background"
}
```

### Query Modes

When querying via `/api/v1/lightrag/query`:

| Mode | Description |
|------|-------------|
| `naive` | Vector search only (fast, recommended) |
| `local` | Entity-focused search |
| `global` | Relationship-focused search |
| `hybrid` | Combines local + global |
| `mix` | Knowledge graph + vector chunks |
| `bypass` | Direct LLM query without retrieval |

## MCP Server (Claude Desktop Integration)

The MCP server exposes **two tools** for searching the RAG knowledge base:

### Tool 1: `query_knowledge_base` (Text-only)

Basic text-based query tool.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | The search query |
| `mode` | string | `"naive"` | Search mode: `naive`, `local`, `global`, `hybrid` |
| `top_k` | integer | `10` | Number of chunks to retrieve |
| `only_need_context` | boolean | `true` | Return only context (no LLM answer) |

### Tool 2: `query_knowledge_base_multimodal` (With Images, Tables, Equations)

Advanced query tool supporting multimodal inputs.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | ✅ Yes | The search query |
| `mode` | string | No | Search mode (default: `naive`) |
| `top_k` | integer | No | Number of chunks (default: `10`) |
| `only_need_context` | boolean | No | Return only context (default: `true`) |
| `image_path` | string | No | Path to image file |
| `image_base64` | string | No | Base64-encoded image |
| `image_caption` | string | No | Description of the image |
| `table_data` | string | No | CSV-formatted table data |
| `table_caption` | string | No | Description of the table |
| `equation_latex` | string | No | LaTeX equation |
| `equation_caption` | string | No | Description of the equation |

**Example multimodal queries:**

Query with image:
```json
{
  "query": "What does this architecture diagram show?",
  "image_path": "/path/to/diagram.png",
  "image_caption": "System architecture diagram",
  "mode": "hybrid"
}
```

Query with table:
```json
{
  "query": "Compare these metrics with the document",
  "table_data": "Method,Precision,Recall\nRAG,0.95,0.92\nBaseline,0.87,0.85",
  "table_caption": "Performance comparison results"
}
```

Query with equation:
```json
{
  "query": "Explain this formula in the context of the paper",
  "equation_latex": "E = mc^2",
  "equation_caption": "Einstein's mass-energy equivalence"
}
```

### Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "raganything": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/mcp-raganything",
        "python",
        "-m",
        "src.main"
      ],
      "env": {
        "MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```

Restart Claude Desktop to activate.

## Development

```bash
uv sync                        # Install dependencies
uv run python src/main.py      # Run locally
uv run pytest --cov=src        # Run tests with coverage
uv run black src/              # Format code
uv run mypy src/               # Type checking
```

```bash
docker-compose build           # Build images
docker-compose up              # Start in foreground
docker-compose logs -f api     # View logs
docker-compose down -v         # Stop and remove volumes
```

## Troubleshooting

- **Empty results:** Lower `COSINE_THRESHOLD` (e.g., `0.1`) or increase `top_k`
- **Port conflicts:** `lsof -ti:8000 | xargs kill -9`
- **Config changes:** Restart server after changing `COSINE_THRESHOLD`, database config, or API keys
- **"Cannot connect to Qdrant"**: Ensure service is running: `docker-compose --profile qdrant up -d`
- **"No module named 'qdrant_client'"**: Install optional dependencies: `uv add qdrant-client neo4j redis`
- **Neo4j authentication failed**: Check `NEO4J_PASSWORD` in `.env` matches docker-compose
- **Multimodal query not working**: Verify LightRAG Server supports `vision_model_func` (requires recent version)

## Documentation

- **[IMPLEMENTATION.md](IMPLEMENTATION.md)** — Complete refactoring guide, configuration matrix, migration instructions
- **[API Docs](http://localhost:8000/docs)** — Interactive Swagger UI
- **[LightRAG Docs](http://localhost:9621/docs)** — LightRAG Server API reference

## License

MIT License - see LICENSE file for details
