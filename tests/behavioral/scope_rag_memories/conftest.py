"""Feature-local fixtures for the scope-RAG-&-memories behavioral suites (TDD RED).

Feature: Scope RAG & Memories per User × Agent (mcp-raganything side).

Boundary policy — mirrors the repo's ``tests/fixtures/external.py`` conventions:

* The application, its middleware, routers / MCP tools, use cases and DI wiring
  run REAL: every test goes through the real ``main.app`` (REST via httpx
  ``ASGITransport``; MCP via the mounted ``/files|mcp``, ``/classical|mcp``,
  ``/memories|mcp`` apps using the global ``mcp_call`` / ``mcp_list`` fixtures).
* Object storage (MinIO/RustFS) is an outbound object-storage system. The suite
  swaps ONLY the ``StoragePort`` adapter singletons (``dependencies.minio_adapter``
  and the ``mcp_file_tools`` binding used by the temporary chat tools) for a
  recording, prefix-aware ``StoragePort`` double so every request-scoped storage
  effect (the composed object keys, list prefixes and delete prefixes) stays
  observable through the real routes/tools without a live MinIO server.
* The per-user vector store / chat LLM DI seams
  (``dependencies.get_vector_store_for_user`` / ``get_chat_llm_for_user``) are
  swapped for a namespace-aware ``VectorStorePort`` double and the shared
  ``mock_llm`` external double, so the composed working_dir namespaces, the
  ``user_id`` chunk tag at index time and the query-time metadata filter stay
  observable through the real MCP tools and REST routes.
* The memories upstream (composable-agents Store File API) is the session
  ``memory_external`` HTTP-boundary double, extended by this conftest with
  directory-listing and DELETE handlers (instance-level, restored per test).

REST routes authenticate through the production composition-root hook:
``main.rest_security`` is wired with the REAL ``AuthService`` over the suite's
established JWT/API-key directory doubles (the same wiring ``running_app``
performs for the MCP middleware), so ``verify_credentials`` resolves the RLS
identity exactly like production.

TDD RED: no production scoping code exists yet; every new-behavior test in the
sibling modules must fail until step 2 implements the feature.
"""

from typing import Any

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from domain.errors.storage import StorageNotFoundError
from domain.ports.storage_port import FileInfo, StoragePort
from domain.ports.vector_store_port import SearchResult, VectorStorePort
from domain.services.auth.auth_service import AuthService

AGENT_HEADER = "X-Composable-Agent"

FILES_MOUNT = "/files/mcp"
CLASSICAL_MOUNT = "/classical/mcp"
MEMORIES_MOUNT = "/memories/mcp"

USER_ALICE = "user-alice"
USER_BOB = "user-bob"


def bearer(credential: str = "jwt-alice") -> dict[str, str]:
    """Credential headers as the real clients present them."""
    if credential.startswith("jwt-"):
        return {"Authorization": f"Bearer {credential}"}
    return {"X-API-Key": credential}


def error_text(result) -> str:
    """Extract the text content of an MCP error result."""
    return "\n".join(getattr(item, "text", "") for item in (result.content or []))


def tool_rows(result) -> list[dict]:
    """Serialise a FastMCP structured result into plain dict rows."""
    assert result.is_error is False, error_text(result)
    data = result.data
    if data is None:
        return []
    if not isinstance(data, list):
        data = [data]
    rows = []
    for item in data:
        if isinstance(item, dict):
            rows.append(item)
        elif isinstance(item, (list, tuple)):
            rows.append(dict(vars(item[0])) if hasattr(item, "__dict__") else item)
        elif hasattr(item, "__dataclass_fields__"):
            rows.append(dict(vars(item)))
        else:
            rows.append(dict(item))
    return rows


async def wait_for_record(predicate, timeout_seconds: float = 10.0) -> bool:
    """Poll until ``predicate`` turns True (background indexing is async)."""
    import asyncio

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return predicate()


@pytest.fixture(autouse=True)
def reset_feature_externals(memory_external):
    """Reset the shared Store File API double around every test.

    Mirrors the ``rag_web_memory`` feature-local reset convention: the global
    ``tests/conftest.py`` re-exports the session-double fixture but not its
    resetters, and this suite seeds/mutates the recorded state per test.
    """
    memory_external.reset()
    yield
    memory_external.reset()


@pytest.fixture(autouse=True)
def rest_dual_auth(running_app, auth_directory):
    """Wire the REST dual-auth exactly like the production composition root.

    ``running_app`` wires the REAL ``AuthService`` into the MCP middleware only
    (its REST counterpart is only wired in production when the asyncpg pool
    exists). The feature's REST scoping must resolve the authenticated
    identity from the RLS contextvars set by ``verify_credentials``, so this
    fixture engages the same production hook (``rest_security.set_auth_service``)
    with the suite's established credential directory, restoring the previous
    state afterwards.
    """
    import main

    previous_auth_service = main.rest_security._auth_service
    main.rest_security.set_auth_service(
        AuthService(
            jwt_port=auth_directory.jwt_port,
            api_key_repo=auth_directory.api_key_repository,
        )
    )
    yield
    main.rest_security.set_auth_service(previous_auth_service)


@pytest_asyncio.fixture
async def api(running_app):
    """httpx AsyncClient over the REAL application (REST surface)."""
    async with httpx.AsyncClient(
        transport=ASGITransport(app=running_app), base_url="http://test"
    ) as client:
        yield client


class RecordingStorage(StoragePort):
    """Prefix-aware, recording StoragePort double (outbound MinIO boundary).

    Behaves like the S3 prefix-scan semantics the adapter relies on: listings
    return only the seeded keys starting with the requested prefix, folder
    listings derive common prefixes from the seeded keys, and full-object keys
    are exactly what the use cases asked the boundary to store.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def seed(self, objects: dict[str, bytes]) -> None:
        self.objects.update(objects)

    def reset(self) -> None:
        self.objects.clear()
        self.calls.clear()

    def calls_for(self, operation: str) -> list[dict[str, Any]]:
        return [args for name, args in self.calls if name == operation]

    def record(self, operation: str, **kwargs) -> None:
        self.calls.append((operation, kwargs))

    async def get_object(self, bucket: str, object_path: str) -> bytes:
        self.record("get_object", bucket=bucket, object_path=object_path)
        if object_path not in self.objects:
            raise StorageNotFoundError(f"Object not found: {object_path}")
        return self.objects[object_path]

    async def put_object(
        self, bucket: str, object_path: str, data: bytes, content_type: str
    ) -> None:
        self.record(
            "put_object",
            bucket=bucket,
            object_path=object_path,
            content_type=content_type,
            size=len(data),
        )
        self.objects[object_path] = data

    async def remove_object(self, bucket: str, object_path: str) -> None:
        self.record("remove_object", bucket=bucket, object_path=object_path)
        self.objects.pop(object_path, None)

    async def remove_prefix(self, bucket: str, prefix: str) -> None:
        self.record("remove_prefix", bucket=bucket, prefix=prefix)
        for key in [key for key in self.objects if key.startswith(prefix)]:
            del self.objects[key]

    async def list_objects(
        self, bucket: str, prefix: str, recursive: bool = True
    ) -> list[str]:
        self.record("list_objects", bucket=bucket, prefix=prefix, recursive=recursive)
        keys = sorted(
            key for key in self.objects if key.startswith(prefix) and key != prefix
        )
        if not recursive:
            keys = [key for key in keys if "/" not in key[len(prefix) :]]
        return keys

    async def list_files_metadata(
        self, bucket: str, prefix: str, recursive: bool = True
    ) -> list[FileInfo]:
        self.record(
            "list_files_metadata", bucket=bucket, prefix=prefix, recursive=recursive
        )
        keys = sorted(
            key for key in self.objects if key.startswith(prefix) and key != prefix
        )
        if not recursive:
            keys = [key for key in keys if "/" not in key[len(prefix) :]]
        return [
            FileInfo(
                object_name=key,
                size=len(self.objects[key]),
                last_modified="2026-01-01T00:00:00+00:00",
            )
            for key in keys
        ]

    async def list_folders(self, bucket: str, prefix: str = "") -> list[str]:
        self.record("list_folders", bucket=bucket, prefix=prefix)
        folders: set[str] = set()
        for key in sorted(self.objects):
            if not key.startswith(prefix) or key == prefix:
                continue
            remainder = key[len(prefix) :]
            if "/" in remainder:
                folders.add(f"{prefix}{remainder.split('/', 1)[0]}/")
        return sorted(folders)

    async def ping(self, bucket: str) -> bool:
        self.record("ping", bucket=bucket)
        return True


@pytest.fixture
def storage_external(monkeypatch) -> RecordingStorage:
    """Swap the outbound storage adapter singletons for the recording double.

    Both bindings are patched: the DI module's singleton (used by every file
    use case provider) and the module-level binding captured by the MCP
    temporary chat tools.
    """
    double = RecordingStorage()
    import application.api.mcp_file_tools as mcp_file_tools
    import dependencies

    monkeypatch.setattr(dependencies, "minio_adapter", double)
    monkeypatch.setattr(mcp_file_tools, "minio_adapter", double)
    return double


class NamespaceVectorStore(VectorStorePort):
    """Namespace-aware VectorStorePort double (outbound pgvector boundary).

    Records every port call verbatim and keeps per-working_dir document
    namespaces so the isolation matrix (user × agent × project) is observable
    through the real MCP tools/REST routes: chunks indexed into one composed
    namespace are never reachable from another scope's composed namespace.
    """

    def __init__(self) -> None:
        self.namespaces: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._counter = 0

    def seed_namespace(self, working_dir: str, documents) -> None:
        self.namespaces.setdefault(working_dir, []).extend(documents)

    def reset(self) -> None:
        self.namespaces.clear()
        self.calls.clear()

    def calls_for(self, operation: str) -> list[tuple[Any, ...]]:
        return [payload for name, payload in self.calls if name == operation]

    async def ensure_table(self, working_dir: str) -> None:
        self.calls.append(("ensure_table", (working_dir,)))
        self.namespaces.setdefault(working_dir, [])

    async def add_documents(
        self, working_dir: str, documents: list[tuple[str, str, dict[str, Any]]]
    ) -> list[str]:
        self.calls.append(("add_documents", (working_dir, list(documents))))
        self.namespaces.setdefault(working_dir, []).extend(documents)
        ids = []
        for _ in documents:
            self._counter += 1
            ids.append(f"chunk-{self._counter}")
        return ids

    async def similarity_search(
        self,
        working_dir: str,
        query: str,
        top_k: int = 10,
        score_threshold: float | None = None,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        self.calls.append(
            (
                "similarity_search",
                (working_dir, query, top_k, score_threshold, metadata_filter),
            )
        )
        results: list[SearchResult] = []
        for content, file_path, metadata in self.namespaces.get(working_dir, []):
            if metadata_filter and any(
                metadata.get(key) != value for key, value in metadata_filter.items()
            ):
                continue
            self._counter += 1
            results.append(
                SearchResult(
                    chunk_id=f"chunk-{self._counter}",
                    content=content,
                    file_path=file_path,
                    score=0.1,
                    metadata=dict(metadata),
                )
            )
        return results[:top_k]

    async def delete_documents(self, working_dir: str, file_path: str) -> int:
        self.calls.append(("delete_documents", (working_dir, file_path)))
        documents = self.namespaces.get(working_dir, [])
        kept = [doc for doc in documents if doc[1] != file_path]
        self.namespaces[working_dir] = kept
        return len(documents) - len(kept)

    async def delete_by_prefix(self, working_dir: str, file_path_prefix: str) -> int:
        self.calls.append(("delete_by_prefix", (working_dir, file_path_prefix)))
        documents = self.namespaces.get(working_dir, [])
        kept = [doc for doc in documents if not doc[1].startswith(file_path_prefix)]
        self.namespaces[working_dir] = kept
        return len(documents) - len(kept)

    async def close(self) -> None:
        return None


@pytest.fixture
def rag_boundaries(monkeypatch, mock_llm) -> NamespaceVectorStore:
    """Swap the per-user vector/LLM DI seams for the namespace double.

    The REAL ``ClassicalQueryUseCase`` / ``ClassicalIndex*UseCase`` instances
    keep running; only the outbound pgvector boundary (with its embedded
    embeddings service) and the chat LLM become observable doubles.
    """
    double = NamespaceVectorStore()
    import dependencies

    async def vector_store_for_user():
        return double

    async def chat_llm_for_user():
        return mock_llm

    monkeypatch.setattr(
        dependencies, "get_vector_store_for_user", vector_store_for_user
    )
    monkeypatch.setattr(dependencies, "get_chat_llm_for_user", chat_llm_for_user)
    return double


@pytest.fixture
def memory_store_ext(memory_external, monkeypatch):
    """Extend the Store File API double with listing + DELETE handlers.

    The shared session double answers GET (file content) and PUT (write).
    The feature's ``list_memories``/``delete_memory`` tools need the
    directory listing (GET on a non-``.md`` path) and ``DELETE`` file
    routes of the Store File API; the extension is instance-level (monkeypatch
    restores the original handler afterwards) and delegates everything else to
    the existing handler so the established GET/PUT behaviour is untouched.
    """
    external = memory_external
    original_handle = external.handle
    path_prefix = "/api/v1/store/files/"

    def handle_with_listing_and_delete(request: httpx.Request) -> httpx.Response:
        # Mirrors the base handler's established observation contract: EVERY
        # request is recorded, and a scripted ``respond_next`` outcome wins
        # before any branch (the base ``handle`` does exactly this for the
        # GET/PUT paths — the extension must behave identically for the
        # directory-listing and DELETE paths).
        external.requests.append(request)
        method = request.method.upper()
        if external.outcomes and external.outcomes[0][0] == method:
            _method, outcome = external.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        path = request.url.path
        if method == "DELETE" and path.startswith(path_prefix):
            # QA BUG-004 evidence (steps I2/I3): the REAL store route answers
            # 204 No Content for an absent file exactly like for a successful
            # deletion — the double mirrors that contract faithfully; the
            # adapter's existence pre-check is what enforces not-found.
            file_path = "/" + path[len(path_prefix) :]
            identity = external.identity(request)
            external.files.pop((identity, file_path), None)
            return httpx.Response(204)
        if method == "GET" and path == "/api/v1/store/files":
            # QA BUG-002: the REAL listing route is
            # GET /api/v1/store/files?prefix=<scope> and answers a plain JSON
            # array of store paths (NOT a "files" document).
            scope_prefix = str(request.url.params.get("prefix", ""))
            entries = [
                stored_path
                for (owner, stored_path), _content in sorted(external.files.items())
                if owner == external.identity(request)
                and stored_path.startswith(scope_prefix)
                and stored_path.endswith(".md")
            ]
            return httpx.Response(200, json=entries)
        return original_handle(request)

    monkeypatch.setattr(external, "handle", handle_with_listing_and_delete)
    return external
