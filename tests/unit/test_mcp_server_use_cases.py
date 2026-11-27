"""Tests for the MCP server registry use cases.

Six use cases: create, list, get, update, delete, validate. Both the external
path (default, validates a remote MCP server via the tool loader) and the
openapi path (fetches an OpenAPI spec, mounts an in-process FastMCP server via
the ``GeneratedMcpRunner``, counts tools) are covered.

Mocked external boundaries (AsyncMock spec'd to the ports):
- McpServerRegistryStore -> AsyncMock
- McpToolLoader -> AsyncMock
- OpenApiMcpFactory -> AsyncMock
- GeneratedMcpRunnerPort -> AsyncMock

Internal components (use cases, domain entities, domain errors) are real.

The openapi create flow returns the absolute mounted URL
``http://raganything-api:8000/generated/{name}/mcp``.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from application.use_cases.create_mcp_server import CreateMcpServerUseCase
from application.use_cases.delete_mcp_server import DeleteMcpServerUseCase
from application.use_cases.get_mcp_server import GetMcpServerUseCase
from application.use_cases.list_mcp_servers import ListMcpServersUseCase
from application.use_cases.update_mcp_server import UpdateMcpServerUseCase
from application.use_cases.validate_mcp_server import ValidateMcpServerUseCase
from domain.entities.mcp_server_config import McpServerConfig, McpTransportType
from domain.entities.mcp_server_registry_entry import RegisteredMcpServer
from domain.errors.mcp import (
    McpConnectionError,
    McpServerAlreadyExistsError,
    McpServerNotFoundError,
    McpServerUnreachableError,
    McpToolLoadError,
    OpenApiFetchError,
    OpenApiInvalidSpecError,
)
from domain.ports.generated_mcp_runner import GeneratedMcpRunnerPort
from domain.ports.mcp_server_registry_store import McpServerRegistryStore
from domain.ports.mcp_tool_loader import McpToolLoader
from domain.ports.openapi_mcp_factory import OpenApiMcpFactory


def _now() -> datetime:
    return datetime.now(UTC)


def _external_entry(
    name: str = "web-search",
    url: str = "http://localhost:3001/mcp",
    auth_token: str | None = "secret-token",
    tool_count: int = 0,
) -> RegisteredMcpServer:
    return RegisteredMcpServer(
        name=name,
        transport=McpTransportType.HTTP,
        url=url,
        headers={"Authorization": "Bearer tok"},
        env={"API_KEY": "abc"},
        auth_token=auth_token,
        tool_count=tool_count,
        source_type="external",
        openapi_url=None,
        created_at=_now(),
        updated_at=_now(),
    )


def _openapi_entry(
    name: str = "petstore",
    openapi_url: str = "https://petstore.example.com/openapi.json",
    headers: dict | None = None,
    tool_count: int = 0,
    url: str | None = None,
) -> RegisteredMcpServer:
    return RegisteredMcpServer(
        name=name,
        transport=McpTransportType.HTTP,
        url=url,
        headers=headers or {"Authorization": "Bearer upstream-token"},
        env={},
        auth_token=None,
        source_type="openapi",
        openapi_url=openapi_url,
        tool_count=tool_count,
        created_at=_now(),
        updated_at=_now(),
    )


def _config(
    name: str = "web-search", url: str = "http://localhost:3001/mcp"
) -> McpServerConfig:
    return McpServerConfig(
        name=name,
        transport=McpTransportType.HTTP,
        url=url,
        headers={"Authorization": "Bearer tok"},
        auth_token="secret-token",
    )


# -- Fixtures ------------------------------------------------------------------


@pytest.fixture
def mock_store() -> AsyncMock:
    return AsyncMock(spec=McpServerRegistryStore)


@pytest.fixture
def mock_loader() -> AsyncMock:
    return AsyncMock(spec=McpToolLoader)


@pytest.fixture
def mock_factory() -> AsyncMock:
    return AsyncMock(spec=OpenApiMcpFactory)


@pytest.fixture
def mock_runner() -> AsyncMock:
    return AsyncMock(spec=GeneratedMcpRunnerPort)


# =============================================================================
# CreateMcpServerUseCase
# =============================================================================


class TestCreateMcpServerUseCase:
    """Tests for CreateMcpServerUseCase (external path)."""

    @pytest.fixture
    def use_case(self, mock_store, mock_loader) -> CreateMcpServerUseCase:
        return CreateMcpServerUseCase(
            store=mock_store,
            tool_loader=mock_loader,
        )

    async def test_returns_entry_with_tool_count_from_loader(
        self, use_case, mock_store, mock_loader
    ):
        # Arrange
        mock_store.exists.return_value = False
        mock_loader.load_tools.return_value = ["tool-a", "tool-b"]
        entry = _external_entry()

        # Act
        result = await use_case.execute(entry)

        # Assert
        assert result.tool_count == 2

    async def test_returns_entry_with_name(self, use_case, mock_store, mock_loader):
        # Arrange
        mock_store.exists.return_value = False
        mock_loader.load_tools.return_value = ["tool-a"]
        entry = _external_entry(name="my-server")

        # Act
        result = await use_case.execute(entry)

        # Assert
        assert result.name == "my-server"

    async def test_checks_existence_before_creating(
        self, use_case, mock_store, mock_loader
    ):
        # Arrange
        mock_store.exists.return_value = False
        mock_loader.load_tools.return_value = []

        # Act
        await use_case.execute(_external_entry("web-search"))

        # Assert
        mock_store.exists.assert_awaited_once_with("web-search")

    async def test_raises_already_exists_when_server_present(
        self, use_case, mock_store
    ):
        # Arrange
        mock_store.exists.return_value = True

        # Act & Assert
        with pytest.raises(McpServerAlreadyExistsError):
            await use_case.execute(_external_entry("web-search"))

    async def test_does_not_call_loader_when_already_exists(
        self, use_case, mock_store, mock_loader
    ):
        # Arrange
        mock_store.exists.return_value = True

        # Act
        with pytest.raises(McpServerAlreadyExistsError):
            await use_case.execute(_external_entry("web-search"))

        # Assert
        mock_loader.load_tools.assert_not_awaited()

    async def test_calls_loader_to_validate_connection(
        self, use_case, mock_store, mock_loader
    ):
        # Arrange
        mock_store.exists.return_value = False
        mock_loader.load_tools.return_value = ["tool-a"]

        # Act
        await use_case.execute(_external_entry("web-search"))

        # Assert
        mock_loader.load_tools.assert_awaited_once()

    async def test_saves_entry_in_store(self, use_case, mock_store, mock_loader):
        # Arrange
        mock_store.exists.return_value = False
        mock_loader.load_tools.return_value = ["tool-a", "tool-b"]

        # Act
        await use_case.execute(_external_entry("web-search"))

        # Assert
        mock_store.save.assert_awaited_once()
        saved_entry = mock_store.save.await_args.args[0]
        assert saved_entry.tool_count == 2

    async def test_raises_unreachable_on_connection_error(
        self, use_case, mock_store, mock_loader
    ):
        # Arrange
        mock_store.exists.return_value = False
        mock_loader.load_tools.side_effect = McpConnectionError("connection refused")

        # Act & Assert
        with pytest.raises(McpServerUnreachableError):
            await use_case.execute(_external_entry("web-search"))

    async def test_raises_unreachable_on_tool_load_error(
        self, use_case, mock_store, mock_loader
    ):
        # Arrange
        mock_store.exists.return_value = False
        mock_loader.load_tools.side_effect = McpToolLoadError("load failed")

        # Act & Assert
        with pytest.raises(McpServerUnreachableError):
            await use_case.execute(_external_entry("web-search"))

    async def test_does_not_save_when_validation_fails(
        self, use_case, mock_store, mock_loader
    ):
        # Arrange
        mock_store.exists.return_value = False
        mock_loader.load_tools.side_effect = McpConnectionError("connection refused")

        # Act
        with pytest.raises(McpServerUnreachableError):
            await use_case.execute(_external_entry("web-search"))

        # Assert
        mock_store.save.assert_not_awaited()

    # -- probe-only credential injection (REQ-004 / BUG-006) -------------------

    async def test_probe_credentials_reach_the_loader_but_not_the_entry(
        self, use_case, mock_store, mock_loader
    ):
        """BUG-006 (REQ-004): the create-time probe IS a load, so its
        ``user_token`` caller credential merges into the PROBE config (an
        injected credential wins over colliding names) and is NEVER
        persisted."""
        # Arrange
        mock_store.exists.return_value = False
        mock_loader.load_tools.return_value = ["tool-a"]
        entry = _external_entry()

        # Act
        await use_case.execute(entry, probe_credentials={"X-API-Key": "caller-key"})

        # Assert — the probe carried stored headers + the injected credential
        config = mock_loader.load_tools.await_args.args[0][0]
        assert config.headers == {
            "Authorization": "Bearer tok",
            "X-API-Key": "caller-key",
        }
        # ...while the STORED entry keeps exactly the configured headers
        saved = mock_store.save.await_args.args[0]
        assert saved.headers == {"Authorization": "Bearer tok"}
        assert mock_store.save.await_args.args[0].auth_token == "secret-token"

    async def test_injected_probe_credential_wins_over_colliding_header(
        self, use_case, mock_store, mock_loader
    ):
        # Arrange — an explicit header COLLIDING with the injected name
        mock_store.exists.return_value = False
        mock_loader.load_tools.return_value = ["tool-a"]
        entry = RegisteredMcpServer(
            name="oauth2-self",
            transport=McpTransportType.HTTP,
            url="http://localhost:3001/mcp",
            headers={"Authorization": "Bearer explicit"},
            auth_token=None,
            tool_count=0,
            source_type="external",
            created_at=_now(),
            updated_at=_now(),
        )

        # Act
        await use_case.execute(
            entry, probe_credentials={"Authorization": "Bearer caller-jwt"}
        )

        # Assert — one Authorization header only: the INJECTED credential
        config = mock_loader.load_tools.await_args.args[0][0]
        assert config.headers == {"Authorization": "Bearer caller-jwt"}

    async def test_no_probe_credentials_keeps_the_configured_probe(
        self, use_case, mock_store, mock_loader
    ):
        # Arrange — legacy callers pass no probe credentials at all
        mock_store.exists.return_value = False
        mock_loader.load_tools.return_value = ["tool-a"]
        entry = _external_entry()

        # Act
        await use_case.execute(entry)

        # Assert — the probe config carries exactly the configured headers
        config = mock_loader.load_tools.await_args.args[0][0]
        assert config.headers == {"Authorization": "Bearer tok"}


class TestCreateMcpServerUseCaseOpenApi:
    """Tests for the openapi source_type branch of CreateMcpServerUseCase."""

    @pytest.fixture
    def use_case(
        self, mock_store, mock_loader, mock_factory, mock_runner
    ) -> CreateMcpServerUseCase:
        return CreateMcpServerUseCase(
            store=mock_store,
            tool_loader=mock_loader,
            openapi_factory=mock_factory,
            runner=mock_runner,
        )

    async def test_fetches_spec_via_factory(
        self, use_case, mock_store, mock_factory, mock_runner
    ):
        # Arrange
        mock_store.exists.return_value = False
        mock_factory.fetch_spec.return_value = {"openapi": "3.0.3"}
        mock_runner.mount.return_value = (
            "http://raganything-api:8000/generated/petstore/mcp"
        )
        mock_runner.count_tools.return_value = 2
        entry = _openapi_entry()

        # Act
        await use_case.execute(entry)

        # Assert
        mock_factory.fetch_spec.assert_awaited_once_with(
            entry.openapi_url, headers=entry.headers
        )

    async def test_mounts_via_runner(
        self, use_case, mock_store, mock_factory, mock_runner
    ):
        # Arrange
        mock_store.exists.return_value = False
        spec = {
            "openapi": "3.0.3",
            "servers": [{"url": "https://petstore.example.com"}],
        }
        mock_factory.fetch_spec.return_value = spec
        mock_runner.mount.return_value = (
            "http://raganything-api:8000/generated/petstore/mcp"
        )
        mock_runner.count_tools.return_value = 2
        entry = _openapi_entry()

        # Act
        await use_case.execute(entry)

        # Assert
        mock_runner.mount.assert_awaited_once()
        call_args = mock_runner.mount.await_args
        assert call_args.args[0] == entry.name
        assert call_args.args[1] is spec

    async def test_returns_absolute_mounted_url(
        self, use_case, mock_store, mock_factory, mock_runner
    ):
        # Arrange
        mock_store.exists.return_value = False
        mock_factory.fetch_spec.return_value = {"openapi": "3.0.3"}
        mock_runner.mount.return_value = (
            "http://raganything-api:8000/generated/petstore/mcp"
        )
        mock_runner.count_tools.return_value = 5
        entry = _openapi_entry(name="petstore")

        # Act
        result = await use_case.execute(entry)

        # Assert
        assert result.url == "http://raganything-api:8000/generated/petstore/mcp"

    async def test_counts_tools_via_runner(
        self, use_case, mock_store, mock_factory, mock_runner
    ):
        # Arrange
        mock_store.exists.return_value = False
        mock_factory.fetch_spec.return_value = {"openapi": "3.0.3"}
        mock_runner.mount.return_value = (
            "http://raganything-api:8000/generated/petstore/mcp"
        )
        mock_runner.count_tools.return_value = 7
        entry = _openapi_entry()

        # Act
        result = await use_case.execute(entry)

        # Assert
        assert result.tool_count == 7
        mock_runner.count_tools.assert_awaited_once_with("petstore")

    async def test_count_failure_falls_back_to_zero_and_still_persists(
        self, use_case, mock_store, mock_factory, mock_runner
    ):
        """QA BUG-001: the count annotates an ALREADY-mounted, healthy
        server — a count failure must neither fail the create nor leave the
        mount unregistered (persisted with the 0 = unavailable convention)."""
        # Arrange
        mock_store.exists.return_value = False
        mock_factory.fetch_spec.return_value = {"openapi": "3.0.3"}
        mock_runner.mount.return_value = (
            "http://raganything-api:8000/generated/petstore/mcp"
        )
        mock_runner.count_tools.side_effect = RuntimeError(
            "Invalid or missing credentials"
        )

        # Act
        result = await use_case.execute(_openapi_entry())

        # Assert — create succeeds; the mounted server stays registered.
        assert result.tool_count == 0
        assert result.url == "http://raganything-api:8000/generated/petstore/mcp"
        mock_store.save.assert_awaited_once()

    async def test_saves_entry_with_mounted_url_and_tool_count(
        self, use_case, mock_store, mock_factory, mock_runner
    ):
        # Arrange
        mock_store.exists.return_value = False
        mock_factory.fetch_spec.return_value = {"openapi": "3.0.3"}
        mock_runner.mount.return_value = (
            "http://raganything-api:8000/generated/petstore/mcp"
        )
        mock_runner.count_tools.return_value = 3
        entry = _openapi_entry()

        # Act
        await use_case.execute(entry)

        # Assert
        mock_store.save.assert_awaited_once()
        saved = mock_store.save.await_args.args[0]
        assert saved.url == "http://raganything-api:8000/generated/petstore/mcp"
        assert saved.tool_count == 3
        assert saved.source_type == "openapi"

    async def test_raises_already_exists_for_openapi(self, use_case, mock_store):
        # Arrange
        mock_store.exists.return_value = True

        # Act & Assert
        with pytest.raises(McpServerAlreadyExistsError):
            await use_case.execute(_openapi_entry())

    async def test_raises_unreachable_on_fetch_error(
        self, use_case, mock_store, mock_factory, mock_runner
    ):
        # Arrange
        mock_store.exists.return_value = False
        mock_factory.fetch_spec.side_effect = OpenApiFetchError("fetch failed")

        # Act & Assert
        with pytest.raises(OpenApiFetchError):
            await use_case.execute(_openapi_entry())

    async def test_raises_invalid_spec_on_conversion_error(
        self, use_case, mock_store, mock_factory, mock_runner
    ):
        # Arrange
        mock_store.exists.return_value = False
        mock_factory.fetch_spec.side_effect = OpenApiInvalidSpecError("bad spec")

        # Act & Assert
        with pytest.raises(OpenApiInvalidSpecError):
            await use_case.execute(_openapi_entry())

    async def test_probe_credentials_reach_the_spec_fetch_not_the_mount(
        self, use_case, mock_store, mock_factory, mock_runner
    ):
        """BUG-006 (openapi): the spec fetch IS the create-time network probe,
        so its headers merge the ``user_token`` caller credential (injected
        wins over colliding names); the mounted server keeps the STORED
        runtime headers."""
        # Arrange
        mock_store.exists.return_value = False
        mock_factory.fetch_spec.return_value = {"openapi": "3.0.3"}
        mock_runner.mount.return_value = (
            "http://raganything-api:8000/generated/petstore/mcp"
        )
        mock_runner.count_tools.return_value = 2
        entry = _openapi_entry(headers={"Authorization": "Bearer upstream"})

        # Act
        await use_case.execute(entry, probe_credentials={"X-API-Key": "caller-key"})

        # Assert — the FETCH carried stored + injected headers...
        mock_factory.fetch_spec.assert_awaited_once_with(
            entry.openapi_url,
            headers={"Authorization": "Bearer upstream", "X-API-Key": "caller-key"},
        )
        # ...while the MOUNT (the runtime of the in-process server) keeps
        # exactly the configured headers.
        assert mock_runner.mount.await_args.kwargs["headers"] == {
            "Authorization": "Bearer upstream"
        }


# =============================================================================
# ListMcpServersUseCase
# =============================================================================


class TestListMcpServersUseCase:
    """Tests for ListMcpServersUseCase."""

    @pytest.fixture
    def use_case(self, mock_store) -> ListMcpServersUseCase:
        return ListMcpServersUseCase(store=mock_store)

    async def test_returns_list_from_store(self, use_case, mock_store):
        # Arrange
        mock_store.list_all.return_value = [_external_entry("a"), _external_entry("b")]

        # Act
        result = await use_case.execute()

        # Assert
        assert len(result) == 2

    async def test_delegates_to_store_list_all(self, use_case, mock_store):
        # Arrange
        mock_store.list_all.return_value = []

        # Act
        await use_case.execute()

        # Assert
        mock_store.list_all.assert_awaited_once()

    async def test_returns_empty_list_when_no_servers(self, use_case, mock_store):
        # Arrange
        mock_store.list_all.return_value = []

        # Act
        result = await use_case.execute()

        # Assert
        assert result == []


# =============================================================================
# GetMcpServerUseCase
# =============================================================================


class TestGetMcpServerUseCase:
    """Tests for GetMcpServerUseCase."""

    @pytest.fixture
    def use_case(self, mock_store) -> GetMcpServerUseCase:
        return GetMcpServerUseCase(store=mock_store)

    async def test_returns_entry_from_store(self, use_case, mock_store):
        # Arrange
        mock_store.get.return_value = _external_entry("web-search")

        # Act
        result = await use_case.execute("web-search")

        # Assert
        assert result.name == "web-search"

    async def test_delegates_to_store_get(self, use_case, mock_store):
        # Arrange
        mock_store.get.return_value = _external_entry("web-search")

        # Act
        await use_case.execute("web-search")

        # Assert
        mock_store.get.assert_awaited_once_with("web-search")

    async def test_propagates_not_found_error(self, use_case, mock_store):
        # Arrange
        mock_store.get.side_effect = McpServerNotFoundError("not found")

        # Act & Assert
        with pytest.raises(McpServerNotFoundError):
            await use_case.execute("missing")


# =============================================================================
# UpdateMcpServerUseCase
# =============================================================================


class TestUpdateMcpServerUseCase:
    """Tests for UpdateMcpServerUseCase (external path)."""

    @pytest.fixture
    def use_case(self, mock_store, mock_loader) -> UpdateMcpServerUseCase:
        return UpdateMcpServerUseCase(
            store=mock_store,
            tool_loader=mock_loader,
        )

    async def test_saves_updated_entry(self, use_case, mock_store, mock_loader):
        # Arrange
        existing = _external_entry(url="http://old:3001/mcp")
        mock_store.get.return_value = existing
        updated = _external_entry(url="http://new:3001/mcp")

        # Act
        await use_case.execute("web-search", updated)

        # Assert
        mock_store.save.assert_awaited_once()

    async def test_revalidates_when_url_changes(
        self, use_case, mock_store, mock_loader
    ):
        # Arrange
        existing = _external_entry(url="http://old:3001/mcp")
        mock_store.get.return_value = existing
        mock_loader.load_tools.return_value = ["t1", "t2"]
        updated = _external_entry(url="http://new:3001/mcp")

        # Act
        result = await use_case.execute("web-search", updated)

        # Assert
        mock_loader.load_tools.assert_awaited_once()
        assert result.tool_count == 2

    async def test_preserves_original_created_at_on_update(
        self, use_case, mock_store, mock_loader
    ):
        # Arrange — existing entry created in the past, update built with now().
        original_created_at = datetime(2025, 1, 1, tzinfo=UTC)
        existing = _external_entry(url="http://same:3001/mcp")
        existing = existing.model_copy(update={"created_at": original_created_at})
        mock_store.get.return_value = existing
        updated = _external_entry(url="http://same:3001/mcp")

        # Act
        result = await use_case.execute("web-search", updated)

        # Assert — created_at unchanged, updated_at moved forward.
        assert result.created_at == original_created_at
        assert result.updated_at > original_created_at
        saved = mock_store.save.await_args.args[0]
        assert saved.created_at == original_created_at

    async def test_does_not_reload_tools_when_url_unchanged(
        self, use_case, mock_store, mock_loader
    ):
        # Arrange
        existing = _external_entry(url="http://same:3001/mcp")
        mock_store.get.return_value = existing
        updated = _external_entry(url="http://same:3001/mcp")

        # Act
        await use_case.execute("web-search", updated)

        # Assert
        mock_loader.load_tools.assert_not_awaited()

    async def test_raises_not_found_when_server_missing(self, use_case, mock_store):
        # Arrange
        mock_store.get.side_effect = McpServerNotFoundError("not found")

        # Act & Assert
        with pytest.raises(McpServerNotFoundError):
            await use_case.execute("missing", _external_entry())

    async def test_raises_unreachable_on_connection_error(
        self, use_case, mock_store, mock_loader
    ):
        # Arrange
        existing = _external_entry(url="http://old:3001/mcp")
        mock_store.get.return_value = existing
        mock_loader.load_tools.side_effect = McpConnectionError("refused")
        updated = _external_entry(url="http://new:3001/mcp")

        # Act & Assert
        with pytest.raises(McpServerUnreachableError):
            await use_case.execute("web-search", updated)


class TestUpdateMcpServerUseCaseOpenApi:
    """Tests for the openapi source_type branch of UpdateMcpServerUseCase."""

    @pytest.fixture
    def use_case(
        self, mock_store, mock_loader, mock_factory, mock_runner
    ) -> UpdateMcpServerUseCase:
        return UpdateMcpServerUseCase(
            store=mock_store,
            tool_loader=mock_loader,
            openapi_factory=mock_factory,
            runner=mock_runner,
        )

    async def test_fetches_spec_via_factory(
        self, use_case, mock_store, mock_factory, mock_runner
    ):
        # Arrange
        mock_store.get.return_value = _openapi_entry()
        mock_factory.fetch_spec.return_value = {"openapi": "3.0.3"}
        mock_runner.mount.return_value = (
            "http://raganything-api:8000/generated/petstore/mcp"
        )
        mock_runner.count_tools.return_value = 3
        updated = _openapi_entry()

        # Act
        await use_case.execute("petstore", updated)

        # Assert
        mock_factory.fetch_spec.assert_awaited_once()

    async def test_unmounts_old_before_mounting_new(
        self, use_case, mock_store, mock_factory, mock_runner
    ):
        # Arrange
        mock_store.get.return_value = _openapi_entry()
        mock_factory.fetch_spec.return_value = {"openapi": "3.0.3"}
        mock_runner.mount.return_value = (
            "http://raganything-api:8000/generated/petstore/mcp"
        )
        mock_runner.count_tools.return_value = 3
        updated = _openapi_entry()

        # Act
        await use_case.execute("petstore", updated)

        # Assert — unmount called before mount
        mock_runner.unmount.assert_awaited_once_with("petstore")
        mock_runner.mount.assert_awaited_once()

    async def test_saves_entry_with_refreshed_tool_count_and_url(
        self, use_case, mock_store, mock_factory, mock_runner
    ):
        # Arrange
        mock_store.get.return_value = _openapi_entry()
        mock_factory.fetch_spec.return_value = {"openapi": "3.0.3"}
        mock_runner.mount.return_value = (
            "http://raganything-api:8000/generated/petstore/mcp"
        )
        mock_runner.count_tools.return_value = 5
        updated = _openapi_entry()

        # Act
        result = await use_case.execute("petstore", updated)

        # Assert
        assert result.tool_count == 5
        assert result.url == "http://raganything-api:8000/generated/petstore/mcp"
        mock_store.save.assert_awaited_once()

    async def test_preserves_original_created_at_on_openapi_update(
        self, use_case, mock_store, mock_factory, mock_runner
    ):
        # Arrange — existing openapi entry created in the past.
        original_created_at = datetime(2025, 1, 1, tzinfo=UTC)
        existing = _openapi_entry()
        existing = existing.model_copy(update={"created_at": original_created_at})
        mock_store.get.return_value = existing
        mock_factory.fetch_spec.return_value = {"openapi": "3.0.3"}
        mock_runner.mount.return_value = (
            "http://raganything-api:8000/generated/petstore/mcp"
        )
        mock_runner.count_tools.return_value = 5
        updated = _openapi_entry()

        # Act
        result = await use_case.execute("petstore", updated)

        # Assert
        assert result.created_at == original_created_at
        assert result.updated_at > original_created_at

    async def test_count_failure_falls_back_to_zero_and_still_persists(
        self, use_case, mock_store, mock_factory, mock_runner
    ):
        """QA BUG-001 (same guard as the create path): the count annotates an
        ALREADY-mounted, healthy server — a count failure must neither fail
        the update nor leave the mount unregistered."""
        # Arrange
        mock_store.get.return_value = _openapi_entry()
        mock_factory.fetch_spec.return_value = {"openapi": "3.0.3"}
        mock_runner.mount.return_value = (
            "http://raganything-api:8000/generated/petstore/mcp"
        )
        mock_runner.count_tools.side_effect = RuntimeError(
            "Invalid or missing credentials"
        )
        updated = _openapi_entry()

        # Act
        result = await use_case.execute("petstore", updated)

        # Assert — update succeeds; the remounted server stays registered.
        assert result.tool_count == 0
        assert result.url == "http://raganything-api:8000/generated/petstore/mcp"
        mock_store.save.assert_awaited_once()


# =============================================================================
# DeleteMcpServerUseCase
# =============================================================================


class TestDeleteMcpServerUseCase:
    """Tests for DeleteMcpServerUseCase."""

    @pytest.fixture
    def use_case(self, mock_store) -> DeleteMcpServerUseCase:
        return DeleteMcpServerUseCase(store=mock_store)

    async def test_delegates_to_store_delete(self, use_case, mock_store):
        # Act
        await use_case.execute("web-search")

        # Assert
        mock_store.delete.assert_awaited_once_with("web-search")

    async def test_propagates_not_found_error(self, use_case, mock_store):
        # Arrange
        mock_store.delete.side_effect = McpServerNotFoundError("not found")

        # Act & Assert
        with pytest.raises(McpServerNotFoundError):
            await use_case.execute("missing")


class TestDeleteMcpServerUseCaseOpenApi:
    """Tests for DeleteMcpServerUseCase when a runner is injected.

    ``runner.unmount`` is called unconditionally — it is a no-op for names
    that are not mounted (including all external servers), so the use case no
    longer reads the entry (avoiding a secret-decrypting round-trip).
    """

    @pytest.fixture
    def use_case(self, mock_store, mock_runner) -> DeleteMcpServerUseCase:
        return DeleteMcpServerUseCase(store=mock_store, runner=mock_runner)

    async def test_unmounts_before_delete(self, use_case, mock_store, mock_runner):
        # Act
        await use_case.execute("petstore")

        # Assert — unmount called unconditionally (no-op for absent names),
        # then the row is deleted.
        mock_runner.unmount.assert_awaited_once_with("petstore")
        mock_store.delete.assert_awaited_once_with("petstore")

    async def test_does_not_read_store_before_delete(self, use_case, mock_store):
        # Act
        await use_case.execute("petstore")

        # Assert — no get() round-trip (secrets stay encrypted in the DB).
        mock_store.get.assert_not_awaited()


# =============================================================================
# ValidateMcpServerUseCase
# =============================================================================


class TestValidateMcpServerUseCase:
    """Tests for ValidateMcpServerUseCase (external path)."""

    @pytest.fixture
    def use_case(self, mock_loader) -> ValidateMcpServerUseCase:
        return ValidateMcpServerUseCase(tool_loader=mock_loader)

    async def test_returns_probe_with_names_from_loader(self, use_case, mock_loader):
        # Arrange
        tools = [MagicMock(name=f"tool_{i}") for i in range(3)]
        for index, tool in enumerate(tools):
            tool.name = f"tool_{index}"
        mock_loader.load_tools.return_value = tools
        config = _config()

        # Act
        result = await use_case.execute(config)

        # Assert
        assert result.count == 3
        assert result.tool_names == ["tool_0", "tool_1", "tool_2"]

    async def test_raises_unreachable_on_connection_error(self, use_case, mock_loader):
        # Arrange
        mock_loader.load_tools.side_effect = McpConnectionError("refused")

        # Act & Assert
        with pytest.raises(McpServerUnreachableError):
            await use_case.execute(_config())


class TestValidateMcpServerUseCaseOpenApi:
    """Tests for the openapi dry-run branch of ValidateMcpServerUseCase."""

    @pytest.fixture
    def use_case(self, mock_loader, mock_factory) -> ValidateMcpServerUseCase:
        return ValidateMcpServerUseCase(
            tool_loader=mock_loader,
            openapi_factory=mock_factory,
        )

    async def test_fetches_spec_via_factory(self, use_case, mock_factory):
        # Arrange
        mock_factory.fetch_spec.return_value = {"openapi": "3.0.3"}
        mock_factory.build_mcp.return_value = MagicMock()
        mock_factory.count_tools.return_value = 4

        # Act
        await use_case.execute_openapi("https://x.example.com/openapi.json", headers={})

        # Assert
        mock_factory.fetch_spec.assert_awaited_once()

    async def test_builds_mcp_via_factory(self, use_case, mock_factory):
        # Arrange
        mock_factory.fetch_spec.return_value = {"openapi": "3.0.3"}
        mock_factory.build_mcp.return_value = MagicMock()
        mock_factory.count_tools.return_value = 4

        # Act
        await use_case.execute_openapi("https://x.example.com/openapi.json", headers={})

        # Assert
        mock_factory.build_mcp.assert_awaited_once()

    async def test_returns_probe_with_names_from_the_factory(
        self, use_case, mock_factory
    ):
        # Arrange
        mock_factory.fetch_spec.return_value = {"openapi": "3.0.3"}
        mock_factory.build_mcp.return_value = MagicMock()
        mock_factory.count_tools.return_value = 6
        mock_factory.list_tool_names.return_value = [f"op_{i}" for i in range(6)]

        # Act
        result = await use_case.execute_openapi(
            "https://x.example.com/openapi.json", headers={}
        )

        # Assert
        assert result.count == 6
        assert result.tool_names == [f"op_{i}" for i in range(6)]

    async def test_does_not_mount_or_persist(
        self, use_case, mock_factory, mock_store, mock_runner
    ):
        # Arrange
        mock_factory.fetch_spec.return_value = {"openapi": "3.0.3"}
        mock_factory.build_mcp.return_value = MagicMock()
        mock_factory.count_tools.return_value = 2

        # Act
        await use_case.execute_openapi("https://x.example.com/openapi.json", headers={})

        # Assert — store and runner are not involved in validate
        mock_store.save.assert_not_awaited()
        mock_runner.mount.assert_not_awaited()

    async def test_raises_on_fetch_error(self, use_case, mock_factory):
        # Arrange
        mock_factory.fetch_spec.side_effect = OpenApiFetchError("fetch failed")

        # Act & Assert
        with pytest.raises(OpenApiFetchError):
            await use_case.execute_openapi(
                "https://x.example.com/openapi.json", headers={}
            )
