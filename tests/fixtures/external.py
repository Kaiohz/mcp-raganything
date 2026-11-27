"""Outbound-system fixtures used by the test suite.

The web/memory doubles below replace only third-party SDK and HTTP boundaries.
The real application, middleware, use cases, and internal adapters remain wired.
"""

import asyncio
import copy
import importlib
import json
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from domain.entities.user.user import User
from domain.ports.auth.api_key_repository import ApiKeyRepositoryPort
from domain.ports.auth.jwt_service import JwtServicePort
from domain.ports.document_reader_port import (
    DocumentContent,
    DocumentMetadata,
    DocumentReaderPort,
)
from domain.ports.llm_port import LLMPort
from domain.ports.storage_port import FileInfo, StoragePort
from domain.ports.vector_store_port import SearchResult, VectorStorePort
from domain.services.auth.api_key_hasher import ApiKeyHasher


@pytest.fixture
def mock_storage() -> AsyncMock:
    """Provide an AsyncMock of StoragePort for external adapter mocking."""
    mock = AsyncMock(spec=StoragePort)
    mock.get_object.return_value = b"fake file content"
    mock.list_objects.return_value = ["project/doc1.pdf", "project/doc2.pdf"]
    mock.list_files_metadata.return_value = [
        FileInfo(
            object_name="project/doc1.pdf",
            size=1024,
            last_modified="2026-01-01 00:00:00+00:00",
        ),
        FileInfo(
            object_name="project/doc2.pdf",
            size=2048,
            last_modified="2026-01-02 00:00:00+00:00",
        ),
    ]
    mock.list_folders.return_value = ["project/", "documents/"]
    return mock


@pytest.fixture
def mock_document_reader() -> AsyncMock:
    mock = AsyncMock(spec=DocumentReaderPort)
    mock.extract_content.return_value = DocumentContent(
        content=["Extracted text content from document."],
        metadata=DocumentMetadata(format_type="pdf", mime_type="application/pdf"),
        tables=[],
    )
    return mock


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    """Provide an AsyncMock of VectorStorePort for external adapter mocking."""
    mock = AsyncMock(spec=VectorStorePort)
    mock.ensure_table.return_value = None
    mock.add_documents.return_value = ["chunk-1", "chunk-2", "chunk-3"]
    mock.similarity_search.return_value = [
        SearchResult(
            chunk_id="chunk-abc123",
            content="Relevant text about the query topic.",
            file_path="/docs/report.pdf",
            score=0.92,
            metadata={"page": 1},
        ),
        SearchResult(
            chunk_id="chunk-def456",
            content="Another relevant chunk of text.",
            file_path="/docs/notes.txt",
            score=0.85,
            metadata={"page": 3},
        ),
    ]
    mock.delete_documents.return_value = 5
    mock.close.return_value = None
    return mock


@pytest.fixture
def mock_llm() -> AsyncMock:
    """Provide an AsyncMock of LLMPort for external adapter mocking."""
    mock = AsyncMock(spec=LLMPort)
    mock.generate.return_value = (
        '["alternative query 1", "alternative query 2", "alternative query 3"]'
    )
    mock.generate_chat.return_value = "LLM generated response text"
    return mock


@contextmanager
def _patch_when_feature_exists(target: str, replacement):
    """Keep the pre-implementation Red run free of fixture import errors.

    Before the feature exists its adapter module does not exist either, so the
    public request fails at its missing mount. Once the adapter exists, failure
    to import one of its declared dependencies remains a real test error.
    """
    module_name, _attribute = target.rsplit(".", 1)
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if module_name == exc.name or module_name.startswith(f"{exc.name}."):
            yield
            return
        raise
    with patch(target, replacement):
        yield


class AuthDirectory:
    """External JWT/API-key directory; the production AuthService stays real."""

    def __init__(self) -> None:
        self.jwt_users = {"jwt-alice": "user-alice", "jwt-bob": "user-bob"}
        self.api_key_users = {"key-alice": "user-alice", "key-bob": "user-bob"}
        self.jwt_port = AsyncMock(spec=JwtServicePort)
        self.api_key_repository = AsyncMock(spec=ApiKeyRepositoryPort)
        self.jwt_port.decode_token.side_effect = self._decode_token
        self.api_key_repository.find_active_by_hash.side_effect = self._find_key
        self._key_hashes = {
            ApiKeyHasher.hash_key(key): (user_id, f"key-{user_id}")
            for key, user_id in self.api_key_users.items()
        }

    async def _decode_token(self, token: str):
        user_id = self.jwt_users.get(token)
        return User(sub=user_id) if user_id else None

    async def _find_key(self, key_hash: str):
        return self._key_hashes.get(key_hash)


@pytest.fixture(scope="session")
def auth_directory() -> AuthDirectory:
    return AuthDirectory()


class FakeDDGS:
    """Synchronous DDGS double with observable thread/concurrency behavior."""

    def __init__(self, owner: "WebExternal") -> None:
        self.owner = owner

    def text(self, query: str, **kwargs):
        owner = self.owner
        with owner.lock:
            owner.active_searches += 1
            owner.max_active_searches = max(
                owner.max_active_searches, owner.active_searches
            )
            owner.search_calls.append(
                {
                    "query": query,
                    "kwargs": copy.deepcopy(kwargs),
                    "thread": threading.current_thread().name,
                }
            )
        try:
            if owner.search_gate is not None:
                owner.search_gate.wait()
            # Per-backend override (fallback tests): a list replaces the
            # shared results and a BaseException replaces the shared error
            # for THAT backend only — other backends behave as configured.
            try:
                backend_override = owner.backends_results[kwargs.get("backend")]
            except KeyError:
                backend_override = None
            if isinstance(backend_override, BaseException):
                raise backend_override
            if backend_override is not None:
                return copy.deepcopy(backend_override)
            if owner.search_error is not None:
                raise owner.search_error
            return copy.deepcopy(owner.search_results)
        finally:
            with owner.lock:
                owner.active_searches -= 1


class FakeCrawler:
    """Crawl4AI AsyncWebCrawler double supporting streaming arun_many."""

    def __init__(self, owner: "WebExternal", *args, **kwargs) -> None:
        self.owner = owner
        self.args = args
        self.kwargs = kwargs
        self.start_calls = 0
        self.close_calls = 0
        owner.crawler_instances.append(self)

    async def start(self):
        self.start_calls += 1
        return self

    async def close(self):
        self.close_calls += 1

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, *_args):
        await self.close()

    async def arun_many(self, urls, config=None, dispatcher=None, **_kwargs):
        owner = self.owner
        batch = SimpleNamespace(
            urls=list(urls), config=config, dispatcher=dispatcher, crawler=self
        )
        owner.crawl_batches.append(batch)
        owner.active_crawl_batches += 1
        owner.max_active_crawl_batches = max(
            owner.max_active_crawl_batches, owner.active_crawl_batches
        )

        async def stream():
            try:
                if owner.stream_handler is not None:
                    async for result in owner.stream_handler(list(urls)):
                        yield result
                    return
                for url in urls:
                    if url in owner.crawl_delays:
                        await asyncio.sleep(owner.crawl_delays[url])
                    yield owner.crawl_result(url)
            finally:
                owner.active_crawl_batches -= 1

        return stream()


class WebExternal:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.crawler_instances: list[FakeCrawler] = []
        self.ddgs_factory = lambda *_args, **_kwargs: FakeDDGS(self)
        self.crawler_factory = lambda *args, **kwargs: FakeCrawler(
            self, *args, **kwargs
        )
        # Declared state (reset() keeps the same shapes; annotations give the
        # dynamic doubles stable, checkable types across per-test resets).
        self.stream_handler: Callable[[list[str]], AsyncIterator] | None = None
        self.search_error: BaseException | None = None
        self.search_gate: threading.Event | None = None
        self.dns_answers: dict[str, list[str]] = {}
        self.backends_results: dict[str, list[dict] | BaseException] = {}
        self.reset()

    def reset(self) -> None:
        self.search_results = []
        self.search_calls = []
        self.search_error = None
        self.search_gate = None
        self.active_searches = 0
        self.max_active_searches = 0
        self.crawl_batches = []
        self.active_crawl_batches = 0
        self.max_active_crawl_batches = 0
        self.stream_handler = None
        self.markdown_by_url = {}
        self.final_urls = {}
        self.crawl_failures = {}
        self.crawl_delays = {}
        self.dns_answers = {}
        self.backends_results = {}

    @property
    def crawler_count(self) -> int:
        return len(self.crawler_instances)

    @property
    def crawler_start_count(self) -> int:
        return sum(crawler.start_calls for crawler in self.crawler_instances)

    @property
    def crawler_close_count(self) -> int:
        return sum(crawler.close_calls for crawler in self.crawler_instances)

    def crawl_result(self, url: str):
        final_url = self.final_urls.get(url, url)
        error = self.crawl_failures.get(url)
        markdown = self.markdown_by_url.get(url, f"# Content for {url}\n")
        markdown_result = SimpleNamespace(
            raw_markdown=markdown,
            fit_markdown=markdown,
            markdown_with_citations=markdown,
            references_markdown="",
        )
        return SimpleNamespace(
            # Real crawl4ai contract (async_webcrawler.py / CrawlResult):
            # ``url`` is the REQUESTED url, ``redirected_url`` is the browser's
            # FINAL url after redirects / JS navigation. The double must not
            # conflate the two, or the redirect SSRF guard is untestable.
            url=url,
            redirected_url=final_url,
            success=error is None,
            error_message=error,
            markdown=markdown_result,
            html="<html>not for public output</html>",
            cleaned_html="<p>not for public output</p>",
            screenshot=None,
            links={"internal": [{"href": "https://discovered.invalid/"}]},
        )

    def resolve(self, host, port, *args, **kwargs):
        del port, args, kwargs
        hostname = host.decode() if isinstance(host, bytes) else host
        answers = self.dns_answers.get(hostname, ["93.184.216.34"])
        return [
            (2, 1, 6, "", (answer, 0))
            if ":" not in answer
            else (10, 1, 6, "", (answer, 0, 0, 0))
            for answer in answers
        ]


@pytest.fixture(scope="session")
def web_external():
    external = WebExternal()
    with ExitStack() as stack:
        stack.enter_context(
            _patch_when_feature_exists(
                "infrastructure.web_search.ddgs_adapter.DDGS",
                external.ddgs_factory,
            )
        )
        stack.enter_context(
            _patch_when_feature_exists(
                "infrastructure.web_search.crawl4ai_adapter.AsyncWebCrawler",
                external.crawler_factory,
            )
        )
        stack.enter_context(
            _patch_when_feature_exists(
                "infrastructure.web_search.url_safety.socket.getaddrinfo",
                external.resolve,
            )
        )
        yield external


@pytest.fixture(autouse=True)
def reset_web_external(request):
    external = request.getfixturevalue("web_external")
    external.reset()
    yield
    if external.search_gate is not None:
        external.search_gate.set()


class FakeComposableAgentsStore:
    """HTTP-boundary double for composable-agents' canonical Store File API."""

    PATH_PREFIX = "/api/v1/store/files/"

    def __init__(self) -> None:
        self.client_kwargs_history = []
        self.clients = []
        self.reset()

    def reset(self) -> None:
        self.files = {}
        self.requests = []
        self.outcomes = []

    def identity(self, request: httpx.Request) -> str | None:
        authorization = request.headers.get("authorization")
        api_key = request.headers.get("x-api-key")
        if authorization:
            return authorization
        if api_key:
            return f"X-API-Key: {api_key}"
        return None

    def seed(self, credential: str, name: str, content: str) -> None:
        identity = (
            f"Bearer {credential}"
            if credential.startswith("jwt-")
            else f"X-API-Key: {credential}"
        )
        self.files[(identity, f"/memories/{name}.md")] = content

    def stored(self, credential: str, name: str) -> str | None:
        identity = (
            f"Bearer {credential}"
            if credential.startswith("jwt-")
            else f"X-API-Key: {credential}"
        )
        return self.files.get((identity, f"/memories/{name}.md"))

    def respond_next(self, method: str, status: int, body: str = "") -> None:
        self.outcomes.append((method.upper(), httpx.Response(status, text=body)))

    def fail_next(self, method: str, error: Exception) -> None:
        self.outcomes.append((method.upper(), error))

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        method = request.method.upper()
        if self.outcomes and self.outcomes[0][0] == method:
            _method, outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        path = request.url.path
        if not path.startswith(self.PATH_PREFIX):
            return httpx.Response(404, text="unknown upstream route")
        file_path = "/" + path[len(self.PATH_PREFIX) :]
        identity = self.identity(request)
        key = (identity, file_path)

        if method == "GET":
            if key not in self.files:
                return httpx.Response(404, json={"detail": "File not found"})
            return httpx.Response(
                200, json={"path": file_path, "content": self.files[key]}
            )
        if method == "PUT":
            payload = json.loads(request.content)
            content = payload["content"]
            self.files[key] = content
            return httpx.Response(200, json={"path": file_path, "content": content})
        return httpx.Response(405, text="method not allowed")

    def client_factory(self, *args, **kwargs):
        self.client_kwargs_history.append(copy.deepcopy(kwargs))
        kwargs["transport"] = httpx.MockTransport(self.handle)
        client = httpx.AsyncClient(*args, **kwargs)
        self.clients.append(client)
        return client


@pytest.fixture(scope="session")
def memory_external():
    external = FakeComposableAgentsStore()
    with _patch_when_feature_exists(
        "infrastructure.memory.composable_agents_store_adapter.httpx.AsyncClient",
        external.client_factory,
    ):
        yield external


@pytest.fixture(autouse=True)
def reset_memory_external(request):
    external = request.getfixturevalue("memory_external")
    external.reset()
    yield
