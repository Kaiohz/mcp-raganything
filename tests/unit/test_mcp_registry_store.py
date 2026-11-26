"""Tests for the raw asyncpg MCP registry store.

The store (``infrastructure.database.mcp_registry_store``) persists registered
MCP servers in PostgreSQL using raw asyncpg (no SQLAlchemy, no alembic). It
encrypts ``headers`` at rest via a :class:`SecretCipher` (Fernet) and decrypts
on read. The asyncpg connection pool is mocked so no real database is needed.

Mocked external boundaries:
- asyncpg.Pool / connection / transaction (external I/O)
- SecretCipher (Fernet) — AsyncMock

Internal component (the store adapter) is exercised for real.
"""

import json
import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from domain.entities.mcp_server_config import McpAuthMode, McpTransportType
from domain.entities.mcp_server_registry_entry import RegisteredMcpServer
from domain.errors.mcp import McpServerInvalidAuthModeError, McpServerNotFoundError
from domain.ports.secret_cipher import SecretCipher
from infrastructure.database.mcp_registry_store import McpRegistryStore


def _now() -> datetime:
    return datetime.now(UTC)


def _entry(
    name: str = "web-search",
    url: str = "http://localhost:3001/mcp",
    headers: dict | None = None,
    auth_token: str | None = "secret-token",
    tool_count: int = 3,
) -> RegisteredMcpServer:
    return RegisteredMcpServer(
        name=name,
        transport=McpTransportType.HTTP,
        url=url,
        headers=headers or {"Authorization": "Bearer tok"},
        env={"API_KEY": "abc"},
        auth_token=auth_token,
        tool_count=tool_count,
        source_type="external",
        openapi_url=None,
        created_at=_now(),
        updated_at=_now(),
    )


def _row(
    name: str = "web-search",
    transport: str = "http",
    url: str = "http://localhost:3001/mcp",
    headers_encrypted: str = "ENC_HEADERS",
    env_encrypted: str = "ENC_ENV",
    auth_token_encrypted: str | None = "ENC_TOKEN",
    tool_count: int = 3,
    source_type: str = "external",
    openapi_url: str | None = None,
    auth_mode: str = "headers",
) -> dict:
    return {
        "name": name,
        "transport": transport,
        "url": url,
        "headers_encrypted": headers_encrypted,
        "env_encrypted": env_encrypted,
        "auth_token_encrypted": auth_token_encrypted,
        "tool_count": tool_count,
        "source_type": source_type,
        "openapi_url": openapi_url,
        "auth_mode": auth_mode,
        "created_at": _now(),
        "updated_at": _now(),
    }


# -- Fixtures ------------------------------------------------------------------


@pytest.fixture
def mock_cipher() -> AsyncMock:
    """Mock SecretCipher that encrypts by prefixing and decrypts by stripping."""
    cipher = AsyncMock(spec=SecretCipher)

    async def _encrypt(plaintext: str) -> str:
        return f"ENC({plaintext})"

    async def _decrypt(ciphertext: str) -> str:
        if ciphertext.startswith("ENC(") and ciphertext.endswith(")"):
            return ciphertext[4:-1]
        return ciphertext

    cipher.encrypt.side_effect = _encrypt
    cipher.decrypt.side_effect = _decrypt
    return cipher


@pytest.fixture
def mock_conn() -> AsyncMock:
    """A mock asyncpg.Connection."""
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=None)
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetch = AsyncMock(return_value=[])
    conn.execute = AsyncMock(return_value="OK")
    return conn


@pytest.fixture
def mock_pool(mock_conn: AsyncMock) -> MagicMock:
    """A mock asyncpg.Pool whose acquire() yields a mock connection."""
    pool = MagicMock()
    cm = AsyncMock()
    cm.__aenter__.return_value = mock_conn
    cm.__aexit__.return_value = None
    pool.acquire.return_value = cm
    return pool


@pytest.fixture
def store(mock_pool: MagicMock, mock_cipher: AsyncMock) -> McpRegistryStore:
    return McpRegistryStore(pool=mock_pool, cipher=mock_cipher)


# -- save ----------------------------------------------------------------------


class TestSave:
    """Tests for McpRegistryStore.save (encrypts secrets, stores row)."""

    async def test_encrypts_headers_before_storing(
        self, store: McpRegistryStore, mock_conn: AsyncMock, mock_cipher: AsyncMock
    ):
        # Arrange
        entry = _entry(headers={"Authorization": "Bearer tok"})

        # Act
        await store.save(entry)

        # Assert — cipher.encrypt was called for headers
        encrypt_calls = mock_cipher.encrypt.await_args_list
        encrypted_values = [c.args[0] for c in encrypt_calls]
        assert any("Authorization" in v for v in encrypted_values)

    async def test_encrypts_env_before_storing(
        self, store: McpRegistryStore, mock_conn: AsyncMock, mock_cipher: AsyncMock
    ):
        # Arrange
        entry = _entry()

        # Act
        await store.save(entry)

        # Assert — cipher.encrypt was called for env (JSON-encoded)
        encrypt_calls = mock_cipher.encrypt.await_args_list
        encrypted_values = [c.args[0] for c in encrypt_calls]
        assert any("API_KEY" in v for v in encrypted_values)

    async def test_encrypts_auth_token_when_present(
        self, store: McpRegistryStore, mock_conn: AsyncMock, mock_cipher: AsyncMock
    ):
        # Arrange
        entry = _entry(auth_token="my-secret-token")

        # Act
        await store.save(entry)

        # Assert
        encrypt_calls = mock_cipher.encrypt.await_args_list
        encrypted_values = [c.args[0] for c in encrypt_calls]
        assert "my-secret-token" in encrypted_values

    async def test_does_not_encrypt_auth_token_when_none(
        self, store: McpRegistryStore, mock_conn: AsyncMock, mock_cipher: AsyncMock
    ):
        # Arrange
        entry = _entry(auth_token=None)

        # Act
        await store.save(entry)

        # Assert — auth_token=None should not call encrypt for the token
        encrypt_calls = mock_cipher.encrypt.await_args_list
        encrypted_values = [c.args[0] for c in encrypt_calls]
        assert "my-secret-token" not in encrypted_values

    async def test_executes_insert_with_encrypted_values(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        # Arrange
        entry = _entry(name="my-server")

        # Act
        await store.save(entry)

        # Assert — execute was called with an INSERT/UPSERT statement
        sql_arg = mock_conn.execute.await_args.args[0]
        assert "INSERT" in sql_arg.upper() or "UPSERT" in sql_arg.upper()


# -- get -----------------------------------------------------------------------


class TestGet:
    """Tests for McpRegistryStore.get (decrypts secrets)."""

    async def test_returns_entry_with_decrypted_headers(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        # Arrange — fetchrow returns a row with encrypted headers
        mock_conn.fetchrow.return_value = _row(
            headers_encrypted="ENC("
            + json.dumps({"Authorization": "Bearer tok"})
            + ")",
            env_encrypted="ENC(" + json.dumps({"API_KEY": "abc"}) + ")",
            auth_token_encrypted="ENC(my-secret-token)",
        )

        # Act
        result = await store.get("web-search")

        # Assert
        assert result.headers == {"Authorization": "Bearer tok"}

    async def test_returns_entry_with_decrypted_env(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        # Arrange
        mock_conn.fetchrow.return_value = _row(
            headers_encrypted="ENC({})",
            env_encrypted="ENC(" + json.dumps({"API_KEY": "abc"}) + ")",
            auth_token_encrypted=None,
        )

        # Act
        result = await store.get("web-search")

        # Assert
        assert result.env == {"API_KEY": "abc"}

    async def test_returns_entry_with_decrypted_auth_token(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        # Arrange
        mock_conn.fetchrow.return_value = _row(
            headers_encrypted="ENC({})",
            env_encrypted="ENC({})",
            auth_token_encrypted="ENC(my-secret-token)",
        )

        # Act
        result = await store.get("web-search")

        # Assert
        assert result.auth_token == "my-secret-token"

    async def test_returns_none_auth_token_when_column_is_null(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        # Arrange
        mock_conn.fetchrow.return_value = _row(
            headers_encrypted="ENC({})",
            env_encrypted="ENC({})",
            auth_token_encrypted=None,
        )

        # Act
        result = await store.get("web-search")

        # Assert
        assert result.auth_token is None

    async def test_raises_not_found_when_row_missing(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        # Arrange
        mock_conn.fetchrow.return_value = None

        # Act & Assert
        with pytest.raises(McpServerNotFoundError):
            await store.get("missing-server")


# -- list_all ------------------------------------------------------------------


class TestListAll:
    """Tests for McpRegistryStore.list_all."""

    async def test_returns_list_of_entries(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        # Arrange
        mock_conn.fetch.return_value = [
            _row(
                name="server-a",
                headers_encrypted="ENC({})",
                env_encrypted="ENC({})",
                auth_token_encrypted=None,
            ),
            _row(
                name="server-b",
                headers_encrypted="ENC({})",
                env_encrypted="ENC({})",
                auth_token_encrypted=None,
            ),
        ]

        # Act
        result = await store.list_all()

        # Assert
        assert len(result) == 2
        names = [e.name for e in result]
        assert "server-a" in names
        assert "server-b" in names

    async def test_returns_empty_list_when_no_rows(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        # Arrange
        mock_conn.fetch.return_value = []

        # Act
        result = await store.list_all()

        # Assert
        assert result == []

    async def test_skips_rows_with_decryption_errors(
        self, store: McpRegistryStore, mock_conn: AsyncMock, mock_cipher: AsyncMock
    ):
        # Arrange — the first row fails to decrypt, the second succeeds
        mock_conn.fetch.return_value = [
            _row(
                name="broken",
                headers_encrypted="BAD-CIPHERTEXT",
                env_encrypted="ENC({})",
                auth_token_encrypted=None,
            ),
            _row(
                name="good",
                headers_encrypted="ENC({})",
                env_encrypted="ENC({})",
                auth_token_encrypted=None,
            ),
        ]

        async def _decrypt(ciphertext: str) -> str:
            if ciphertext == "BAD-CIPHERTEXT":
                raise ValueError("invalid token")
            if ciphertext.startswith("ENC(") and ciphertext.endswith(")"):
                return ciphertext[4:-1]
            return ciphertext

        mock_cipher.decrypt.side_effect = _decrypt

        # Act
        result = await store.list_all()

        # Assert — only the good row is returned
        assert len(result) == 1
        assert result[0].name == "good"


# -- delete --------------------------------------------------------------------


class TestDelete:
    """Tests for McpRegistryStore.delete."""

    async def test_executes_delete_statement(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        # Arrange — fetchval returns a row id so the row exists
        mock_conn.fetchval.return_value = "web-search"

        # Act
        await store.delete("web-search")

        # Assert — at least one execute with DELETE
        execute_sqls = [c.args[0] for c in mock_conn.execute.await_args_list]
        assert any("DELETE" in s.upper() for s in execute_sqls)

    async def test_raises_not_found_when_row_missing(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        # Arrange — fetchval returns None meaning no row matched
        mock_conn.fetchval.return_value = None

        # Act & Assert
        with pytest.raises(McpServerNotFoundError):
            await store.delete("missing-server")


# -- exists --------------------------------------------------------------------


class TestExists:
    """Tests for McpRegistryStore.exists."""

    async def test_returns_true_when_row_exists(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        # Arrange
        mock_conn.fetchval.return_value = 1

        # Act
        result = await store.exists("web-search")

        # Assert
        assert result is True

    async def test_returns_false_when_row_missing(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        # Arrange
        mock_conn.fetchval.return_value = None

        # Act
        result = await store.exists("missing-server")

        # Assert
        assert result is False


# -- auth_mode persistence (feat-mcp-oauth2-dual-auth, registry M-1) ------------


class TestAuthModePersistence:
    """``auth_mode`` (REQ-004, non-secret) round-trips through the store.

    The mode is persisted as plain text (no cipher round) and rehydrated into
    the entity on read — the mode must survive restarts, unlike secrets whose
    ciphertext is decrypted on read.
    """

    async def test_save_persists_auth_mode_as_plain_text(
        self, store: McpRegistryStore, mock_conn: AsyncMock, mock_cipher: AsyncMock
    ):
        # Arrange — a user_token entry
        entry = _entry(name="oauth-server")
        entry = entry.model_copy(update={"auth_mode": McpAuthMode.USER_TOKEN})

        # Act
        await store.save(entry)

        # Assert — the raw "user_token" value travels as the last SQL param
        # (plain text: the cipher is NEVER called with the mode value)
        params = mock_conn.execute.await_args.args
        assert params[-1] == "user_token"
        cipher_args = [c.args[0] for c in mock_cipher.encrypt.await_args_list]
        assert all("user_token" not in v for v in cipher_args)

    async def test_save_upsert_updates_auth_mode(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        # Arrange
        entry = _entry()
        entry = entry.model_copy(update={"auth_mode": McpAuthMode.HEADERS})

        # Act
        await store.save(entry)

        # Assert — the ON CONFLICT clause carries auth_mode
        sql_arg = mock_conn.execute.await_args.args[0]
        assert "auth_mode            = EXCLUDED.auth_mode" in sql_arg

    async def test_get_rehydrates_auth_mode_from_row(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        # Arrange — the row carries the persisted user_token mode
        mock_conn.fetchrow.return_value = _row(
            auth_mode="user_token",
            headers_encrypted="ENC({})",
            env_encrypted="ENC({})",
            auth_token_encrypted=None,
        )

        # Act
        result = await store.get("oauth-server")

        # Assert
        assert result.auth_mode is McpAuthMode.USER_TOKEN

    async def test_get_defaults_to_backfilled_headers_mode(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        # Arrange — a row backfilled by the 004 migration ('headers')
        mock_conn.fetchrow.return_value = _row(
            auth_mode="headers",
            headers_encrypted="ENC({})",
            env_encrypted="ENC({})",
            auth_token_encrypted=None,
        )

        # Act
        result = await store.get("web-search")

        # Assert
        assert result.auth_mode is McpAuthMode.HEADERS

    async def test_list_all_carries_auth_mode(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        # Arrange — one user_token row, one backfilled row
        mock_conn.fetch.return_value = [
            _row(
                name="oauth-server",
                auth_mode="user_token",
                headers_encrypted="ENC({})",
                env_encrypted="ENC({})",
                auth_token_encrypted=None,
            ),
            _row(
                name="legacy-server",
                auth_mode="headers",
                headers_encrypted="ENC({})",
                env_encrypted="ENC({})",
                auth_token_encrypted=None,
            ),
        ]

        # Act
        entries = {e.name: e for e in await store.list_all()}

        # Assert
        assert entries["oauth-server"].auth_mode is McpAuthMode.USER_TOKEN
        assert entries["legacy-server"].auth_mode is McpAuthMode.HEADERS


# -- corrupted persisted auth_mode (feat-mcp-oauth2 REQ-004, re-review MAJOR-2)--


class TestCorruptedAuthMode:
    """A corrupted stored ``auth_mode`` surfaces with its OWN domain error.

    Only manual DB surgery or a foreign writer can produce an unsupported
    mode value. Before this guard: ``get`` raised a raw ``ValueError`` (an
    uncentralized 500) and ``list_all`` DROPPED the row with the misleading
    "secret decryption failed" reason.
    """

    async def test_get_raises_the_domain_error_not_a_raw_value_error(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        # Arrange — a row forced (at the store level) to a corrupted mode
        mock_conn.fetchrow.return_value = _row(
            name="oauth-server",
            auth_mode="client_credentials",
            headers_encrypted="ENC({})",
            env_encrypted="ENC({})",
            auth_token_encrypted=None,
        )

        # Act & Assert — a domain error carrying the centralized message
        with pytest.raises(McpServerInvalidAuthModeError) as excinfo:
            await store.get("oauth-server")
        assert "client_credentials" in excinfo.value.detail

    async def test_list_all_skips_the_corrupted_row_with_the_labeled_log(
        self,
        store: McpRegistryStore,
        mock_conn: AsyncMock,
        caplog: pytest.LogCaptureFixture,
    ):
        # Arrange — one corrupted row (kept healthy secrets) + one good row
        mock_conn.fetch.return_value = [
            _row(
                name="corrupted",
                auth_mode="client_credentials",
                headers_encrypted="ENC({})",
                env_encrypted="ENC({})",
                auth_token_encrypted=None,
            ),
            _row(
                name="good",
                auth_mode="headers",
                headers_encrypted="ENC({})",
                env_encrypted="ENC({})",
                auth_token_encrypted=None,
            ),
        ]

        # Act
        with caplog.at_level(
            logging.WARNING, logger="infrastructure.database.mcp_registry_store"
        ):
            entries = await store.list_all()

        # Assert — the good row survives; the corrupted row is excluded
        assert [e.name for e in entries] == ["good"]
        labeled = [
            r
            for r in caplog.records
            if r.getMessage().count("invalid stored auth_mode")
        ]
        assert labeled, "the skip must be LABELED with its own reason"
        assert any("corrupted" in r.getMessage() for r in labeled), (
            "the log must name the skipped server"
        )

    async def test_list_all_never_reports_the_auth_mode_corruption_as_decryption_failure(
        self,
        store: McpRegistryStore,
        mock_conn: AsyncMock,
        caplog: pytest.LogCaptureFixture,
    ):
        # Arrange — ONLY a corrupted-auth_mode row (no decryption failure)
        mock_conn.fetch.return_value = [
            _row(
                name="corrupted",
                auth_mode="basic",
                headers_encrypted="ENC({})",
                env_encrypted="ENC({})",
                auth_token_encrypted=None,
            ),
        ]

        # Act
        with caplog.at_level(
            logging.WARNING, logger="infrastructure.database.mcp_registry_store"
        ):
            entries = await store.list_all()

        # Assert — the misleading "secret decryption failed" miss is gone
        assert entries == []
        messages = [r.getMessage() for r in caplog.records]
        assert any("invalid stored auth_mode" in m for m in messages)
        assert not any("secret decryption failed" in m for m in messages), (
            "the auth_mode corruption must not be mislabeled as a decryption failure"
        )

    async def test_list_all_null_auth_mode_is_coerced_to_backfilled_headers(
        self, store: McpRegistryStore, mock_conn: AsyncMock
    ):
        # Arrange — the defensive pre-backfill shape (the 004 column is
        # NOT NULL DEFAULT 'headers'; None is tolerated as today's default)
        mock_conn.fetch.return_value = [
            _row(
                name="pre-backfill",
                auth_mode=None,
                headers_encrypted="ENC({})",
                env_encrypted="ENC({})",
                auth_token_encrypted=None,
            ),
        ]

        # Act
        entries = await store.list_all()

        # Assert
        assert [e.auth_mode for e in entries] == [McpAuthMode.HEADERS]

    async def test_auth_mode_survives_a_fresh_store_instance(
        self, mock_pool: MagicMock, mock_conn: AsyncMock, mock_cipher: AsyncMock
    ):
        """Persistence across a second store instantiation (restart semantics).

        A NEW :class:`McpRegistryStore` built over the SAME backend must read
        back the mode persisted by the first instance — the mode lives in the
        row, not in the process.
        """
        # Arrange — first instance saves a user_token entry
        first = McpRegistryStore(pool=mock_pool, cipher=mock_cipher)
        entry = _entry(name="oauth-server")
        entry = entry.model_copy(update={"auth_mode": McpAuthMode.USER_TOKEN})
        await first.save(entry)

        # The second instantiation reads the row the FIRST one wrote
        persisted_params = mock_conn.execute.await_args.args
        mock_conn.fetchrow.return_value = _row(
            name="oauth-server",
            auth_mode=persisted_params[-1],
            headers_encrypted="ENC({})",
            env_encrypted="ENC({})",
            auth_token_encrypted=None,
        )

        # Act
        second = McpRegistryStore(pool=mock_pool, cipher=mock_cipher)
        result = await second.get("oauth-server")

        # Assert
        assert result.auth_mode is McpAuthMode.USER_TOKEN
