"""Unit tests — the ``MCP_PUBLIC_BASE_URL`` setting (M-3).

Adapted from the composable-agents sibling (code-review NEW-1 / NEW-3): the
advertised discovery origin is the pointer the whole OAuth2 discovery chain
starts from:

* a present-but-SCHEME-LESS value REFUSES startup (a malformed
  ``resource_metadata`` URL no MCP client can fetch — the chain would break
  SILENTLY);
* present values are NORMALIZED at the Settings level (surrounding whitespace
  and a trailing slash stripped — the single normalization source);
* a present ``http://`` value stays non-fatal (dev/QA deployments) but is
  WARNED at startup through the centralized ``MCP_PUBLIC_BASE_URL_NOT_HTTPS``
  log message (pinned through the REAL lifespan with caplog).
"""

import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

import main  # noqa: F401 — imported at collection time: configure_logging() at

# import rebuilds the root handlers, which would otherwise wipe pytest's
# per-test caplog handler AFTER the phase install (mid-test import breakage).
from config import AppConfig
from domain.logging.messages import LogMessage


class TestMcpPublicBaseUrlNormalization:
    def test_trailing_slash_is_stripped_at_settings_level(self):
        config = AppConfig(MCP_PUBLIC_BASE_URL="https://raganything.soludev.tech/")
        assert config.MCP_PUBLIC_BASE_URL == "https://raganything.soludev.tech"

    def test_surrounding_whitespace_is_stripped(self):
        config = AppConfig(MCP_PUBLIC_BASE_URL="  https://host.example.com/  ")
        assert config.MCP_PUBLIC_BASE_URL == "https://host.example.com"

    def test_http_value_is_accepted_for_dev(self):
        config = AppConfig(MCP_PUBLIC_BASE_URL="http://raganything.local:8000")
        assert config.MCP_PUBLIC_BASE_URL == "http://raganything.local:8000"

    def test_blank_value_is_treated_as_unset(self):
        config = AppConfig(MCP_PUBLIC_BASE_URL="   ")
        assert config.MCP_PUBLIC_BASE_URL == ""


class TestMcpPublicBaseUrlRefusal:
    """Present-but-invalid ⇒ refuse startup (the sibling validator's rule)."""

    def test_scheme_less_value_refuses_startup(self):
        with pytest.raises(ValidationError, match="MCP_PUBLIC_BASE_URL"):
            AppConfig(MCP_PUBLIC_BASE_URL="raganything.soludev.tech")

    def test_scheme_relative_value_refuses_startup(self):
        with pytest.raises(ValidationError, match="MCP_PUBLIC_BASE_URL"):
            AppConfig(MCP_PUBLIC_BASE_URL="//raganything.soludev.tech")

    def test_refusal_message_names_the_absolute_url_shape(self):
        with pytest.raises(ValidationError, match="absolute URL"):
            AppConfig(MCP_PUBLIC_BASE_URL="host.example.com/mcp")

    def test_refusal_message_explains_the_silently_broken_discovery_chain(self):
        with pytest.raises(ValidationError, match="malformed"):
            AppConfig(MCP_PUBLIC_BASE_URL="example.com")


class TestMcpPublicBaseUrlNotHttpsWarning:
    """The warn half (non-fatal): an ``http://`` origin is warned at startup
    through the centralized ``MCP_PUBLIC_BASE_URL_NOT_HTTPS`` message. A
    configured https:// origin (the production shape) and an UNSET origin
    (the Host-allow-list fallback) must NOT warn."""

    async def _run_lifespan(self) -> None:
        """Drive the real ``combined_lifespan`` with every external boundary
        mocked off (the ``test_lifespan`` pattern): Alembic migrations, the
        registry pool wiring and the four FastMCP session-manager lifespans
        (whose only session-scoped concern lives in the behavioral fixture).
        ``main`` is imported at module level (see the import note) so the
        lifespan runs against the already-configured app.
        """

        @asynccontextmanager
        async def _noop_lifespan(_app):
            yield

        with (
            patch("main._run_alembic_upgrade", return_value=None),
            patch.object(main.app_config, "SECRET_ENCRYPTION_KEY", ""),
            patch.object(main, "_mcp_pool", None),
            patch.object(main, "classical_bm25_adapter", None),
            patch.object(main, "classical_vector_store", None),
            patch.object(
                main, "mcp_files_app", SimpleNamespace(lifespan=_noop_lifespan)
            ),
            patch.object(
                main, "mcp_classical_app", SimpleNamespace(lifespan=_noop_lifespan)
            ),
            patch.object(main, "mcp_web_app", SimpleNamespace(lifespan=_noop_lifespan)),
            patch.object(
                main, "mcp_memories_app", SimpleNamespace(lifespan=_noop_lifespan)
            ),
        ):
            async with main.combined_lifespan(main.app):
                pass

    async def test_non_https_value_warns_at_startup(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        monkeypatch.setattr(
            main.app_config, "MCP_PUBLIC_BASE_URL", "http://raganything.local:8000"
        )
        with caplog.at_level(logging.WARNING, logger="main"):
            await self._run_lifespan()
        assert any(
            LogMessage.MCP_PUBLIC_BASE_URL_NOT_HTTPS in record.getMessage()
            for record in caplog.records
        )

    async def test_https_value_does_not_warn(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        monkeypatch.setattr(
            main.app_config,
            "MCP_PUBLIC_BASE_URL",
            "https://raganything.soludev.tech",
        )
        with caplog.at_level(logging.WARNING, logger="main"):
            await self._run_lifespan()
        assert not any(
            LogMessage.MCP_PUBLIC_BASE_URL_NOT_HTTPS in record.getMessage()
            for record in caplog.records
        )

    async def test_unset_value_does_not_warn(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        monkeypatch.setattr(main.app_config, "MCP_PUBLIC_BASE_URL", "")
        with caplog.at_level(logging.WARNING, logger="main"):
            await self._run_lifespan()
        assert not any(
            LogMessage.MCP_PUBLIC_BASE_URL_NOT_HTTPS in record.getMessage()
            for record in caplog.records
        )
