"""Unit tests — the ``MCP_AUTHORIZATION_SERVER`` setting (M-3 / MINOR-1).

Adapted from the composable-agents sibling (QA BUG-003): the public
protected-resource metadata advertised a container-internal issuer derived
inline from the raw ``LOGTO_URL``. The explicit setting keeps the PUBLIC
issuer decoupled:

* an explicit value wins verbatim (the configmap sets the public issuer);
* an unset/blank value falls back to ``<LOGTO_URL>/oidc`` — resolved ONCE at
  startup by the ``AppConfig`` model validator (code-review MINOR-1, mirroring
  the sibling) instead of inline at the well-known route, and empty when
  ``LOGTO_URL`` is empty too (nothing is advertised);
* a present-but-non-absolute value REFUSES startup (the same
  present-but-invalid ⇒ refuse rule as the other configuration validators);
* the two startup warnings (nothing advertised / non-https issuer) are part
  of the contract — pinned through the REAL lifespan with caplog.
"""

import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

import main  # noqa: F401 — imported at collection time: configure_logging() at

# import rebuilds the root handlers, which would otherwise wipe pytest's
# per-test caplog handler AFTER the phase install (mid-test import breakage).
from config import AppConfig
from domain.logging.messages import LogMessage


class TestMcpAuthorizationServerShape:
    def test_present_but_non_absolute_value_refuses_startup(self):
        with pytest.raises(ValidationError, match="MCP_AUTHORIZATION_SERVER"):
            AppConfig(MCP_AUTHORIZATION_SERVER="logto:3001/oidc")

    def test_relative_value_refuses_startup(self):
        with pytest.raises(ValidationError, match="MCP_AUTHORIZATION_SERVER"):
            AppConfig(MCP_AUTHORIZATION_SERVER="/oidc")

    def test_refusal_message_names_the_expected_shape(self):
        with pytest.raises(ValidationError, match="absolute URL"):
            AppConfig(MCP_AUTHORIZATION_SERVER="example.invalid/oidc")

    def test_explicit_value_is_kept_verbatim(self):
        config = AppConfig(MCP_AUTHORIZATION_SERVER="https://logto.soludev.tech/oidc")
        assert config.MCP_AUTHORIZATION_SERVER == "https://logto.soludev.tech/oidc"

    def test_blank_value_is_treated_as_unset(self):
        """A blank value means "unset" — the ``<LOGTO_URL>/oidc`` fallback
        resolution applies at config load (``AppConfig`` model validator);
        with ``LOGTO_URL`` configured the effective advertised issuer is the
        resolved fallback (no malformed ``["/oidc"]`` shape can slip
        through)."""
        config = AppConfig(
            MCP_AUTHORIZATION_SERVER="   ", LOGTO_URL="http://logto:3001"
        )
        assert config.MCP_AUTHORIZATION_SERVER == "http://logto:3001/oidc"


class TestMcpAuthorizationServerFallbackResolution:
    """Code-review MINOR-1: the ``<LOGTO_URL>/oidc`` fallback is resolved ONCE
    at config load by the ``AppConfig`` model validator (mirroring the
    composable-agents sibling ``resolve_mcp_authorization_server``) instead of
    inline at the well-known route — the effective advertised issuer is
    ALWAYS ``MCP_AUTHORIZATION_SERVER``, so the startup warnings cover the
    fallback (an internal ``http://`` issuer can no longer be published under
    the radar)."""

    def test_unset_value_falls_back_to_the_logto_issuer(self):
        config = AppConfig(MCP_AUTHORIZATION_SERVER="", LOGTO_URL="http://logto:3001")
        assert config.MCP_AUTHORIZATION_SERVER == "http://logto:3001/oidc"

    def test_unset_value_kept_empty_without_logto_url(self):
        """With neither value configured NOTHING is advertised (the route
        then serves ``authorization_servers: []``)."""
        config = AppConfig(MCP_AUTHORIZATION_SERVER="", LOGTO_URL="")
        assert config.MCP_AUTHORIZATION_SERVER == ""

    def test_trailing_slash_is_stripped_from_the_logto_url(self):
        config = AppConfig(
            MCP_AUTHORIZATION_SERVER="", LOGTO_URL="https://logto.soludev.tech/"
        )
        assert config.MCP_AUTHORIZATION_SERVER == "https://logto.soludev.tech/oidc"

    def test_explicit_value_wins_over_the_fallback(self):
        config = AppConfig(
            MCP_AUTHORIZATION_SERVER="https://logto.soludev.tech/oidc",
            LOGTO_URL="http://logto:3001",
        )
        assert config.MCP_AUTHORIZATION_SERVER == "https://logto.soludev.tech/oidc"

    async def test_route_advertises_the_single_resolved_issuer(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The well-known route reads ``MCP_AUTHORIZATION_SERVER`` ONLY (no
        inline ``<LOGTO_URL>/oidc`` derivation is left): the internal issuer
        a validator-resolved fallback produced is advertised verbatim —
        with the startup warning covering it (``MCP_AUTHORIZATION_SERVER_NOT_HTTPS``)."""
        monkeypatch.setattr(
            main.app_config,
            "MCP_PUBLIC_BASE_URL",
            "https://raganything.soludev.tech",
        )
        # The RESOLVED internal fallback shape (what the validator produced
        # for LOGTO_URL=http://logto:3001 with the setting unset).
        monkeypatch.setattr(
            main.app_config, "MCP_AUTHORIZATION_SERVER", "http://logto:3001/oidc"
        )
        async with AsyncClient(
            transport=ASGITransport(app=main.app), base_url="http://test"
        ) as client:
            resp = await client.get("/.well-known/oauth-protected-resource")
        assert resp.status_code == 200
        body = resp.json()
        assert body["authorization_servers"] == ["http://logto:3001/oidc"], (
            "the route must advertise the RESOLVED MCP_AUTHORIZATION_SERVER "
            "value (no second inline-derivation source)"
        )


class TestMcpAuthorizationServerStartupWarnings:
    """The two startup warnings are part of the contract (QA BUG-003 "loud"
    half) — pinned through the REAL ``combined_lifespan`` with caplog so a
    refactor that drops them fails a test."""

    async def _run_lifespan(self) -> None:
        """Drive the real ``combined_lifespan`` with every external boundary
        mocked off (the ``test_lifespan`` pattern); ``main`` is imported at
        module level (see the import note)."""

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
        ):
            async with main.combined_lifespan(main.app):
                pass

    async def test_empty_authorization_server_warns_at_startup(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        monkeypatch.setattr(main.app_config, "MCP_AUTHORIZATION_SERVER", "")
        # The "nothing advertised" condition needs BOTH sources empty; the
        # behavioral session fixture keeps LOGTO_URL injected while it is
        # alive, so pin the warning condition explicitly (full-suite parity).
        monkeypatch.setattr(main.app_config, "LOGTO_URL", "")
        with caplog.at_level(logging.WARNING, logger="main"):
            await self._run_lifespan()
        assert any(
            LogMessage.MCP_AUTHORIZATION_SERVER_NOT_ADVERTISED in record.getMessage()
            for record in caplog.records
        )

    async def test_non_https_authorization_server_warns_at_startup(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """The warning interpolates the configured (non-https) issuer — assert
        the FULL formatted line."""
        monkeypatch.setattr(
            main.app_config, "MCP_AUTHORIZATION_SERVER", "http://logto:3001/oidc"
        )
        expected = LogMessage.MCP_AUTHORIZATION_SERVER_NOT_HTTPS.replace(
            "%s", "http://logto:3001/oidc"
        )
        with caplog.at_level(logging.WARNING, logger="main"):
            await self._run_lifespan()
        assert any(expected in record.getMessage() for record in caplog.records)
