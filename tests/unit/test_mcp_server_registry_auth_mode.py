"""Unit tests for the ``auth_mode`` domain models (feat-mcp-oauth2 REQ-004).

Covers the ENTITY (``RegisteredMcpServer.auth_mode``) and the REQUEST models
(``McpServerCreateRequest`` / ``McpServerValidateRequest`` /
``McpServerUpdateRequest``) so the typed contract is pinned independent of
HTTP routing:

* the domain enum accepts exactly ``"headers"`` | ``"user_token"`` and rejects
  every unknown value (the registry must 422 unsupported modes, not degrade
  them silently — review M-1/M-5 contract with composable-agents);
* the entity defaults to ``"headers"`` and serializes the mode as plain text
  (the JSON contract composable-agents reads via ``entry.get("auth_mode")``);
* the update model treats ABSENT as "preserve" and rejects an explicit
  ``null`` (``model_copy`` merges bypass validation — a null would break the
  entity's ``NOT NULL`` column downstream).
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from application.api.mcp_registry_routes import (
    McpServerCreateRequest,
    McpServerUpdateRequest,
    McpServerValidateRequest,
)
from domain.entities.mcp_server_config import McpAuthMode
from domain.entities.mcp_server_registry_entry import RegisteredMcpServer


def _now() -> datetime:
    return datetime.now(UTC)


def _entry(auth_mode: McpAuthMode = McpAuthMode.HEADERS) -> RegisteredMcpServer:
    return RegisteredMcpServer(
        name="server-x",
        url="http://localhost:3001/mcp",
        headers={"Authorization": "Bearer tok"},
        env={"API_KEY": "abc"},
        auth_token="secret-token",
        auth_mode=auth_mode,
        tool_count=3,
        source_type="external",
        openapi_url=None,
        created_at=_now(),
        updated_at=_now(),
    )


# -- entity ---------------------------------------------------------------------


class TestRegisteredMcpServerAuthMode:
    """The persisted entity carries and types the ``auth_mode`` (REQ-004)."""

    def test_defaults_to_headers(self):
        # Arrange & Act
        entry = _entry()

        # Assert
        assert entry.auth_mode is McpAuthMode.HEADERS

    def test_accepts_user_token(self):
        # Act
        entry = _entry(McpAuthMode.USER_TOKEN)

        # Assert
        assert entry.auth_mode is McpAuthMode.USER_TOKEN

    def test_rejects_unknown_mode(self):
        # Act & Assert — an unknown mode is a domain invariant violation
        with pytest.raises(ValidationError, match="auth_mode"):
            RegisteredMcpServer(
                name="server-x",
                url="http://localhost:3001/mcp",
                auth_mode="basic",  # type: ignore[arg-type]
                created_at=_now(),
                updated_at=_now(),
            )

    def test_entity_equality_compares_modes(self):
        # Assert — mode is part of the persisted identity (two builds with
        # different modes are NOT equal)
        assert _entry() != _entry(McpAuthMode.USER_TOKEN)

    def test_serializes_mode_as_plain_text(self):
        # Act — the JSON contract composable-agents reads on the wire
        payload = _entry(McpAuthMode.USER_TOKEN).model_dump(mode="json")

        # Assert
        assert payload["auth_mode"] == "user_token"
        assert isinstance(payload["auth_mode"], str)

    def test_model_copy_switch_merges_typed_value(self):
        """The PUT merge shape (``model_copy(update=...)``) keeps the enum."""
        # Arrange & Act
        merged = _entry().model_copy(update={"auth_mode": McpAuthMode.USER_TOKEN})

        # Assert
        assert merged.auth_mode is McpAuthMode.USER_TOKEN


# -- create/validate request models ---------------------------------------------


class TestCreateRequestAuthMode:
    """create/validate accept ``auth_mode`` typed, defaulting to ``headers``."""

    def test_create_defaults_to_headers(self):
        # Act
        request = McpServerCreateRequest(
            name="server-x", url="http://localhost:3001/mcp"
        )

        # Assert
        assert request.auth_mode is McpAuthMode.HEADERS

    def test_create_accepts_user_token(self):
        # Act
        request = McpServerCreateRequest(
            name="server-x",
            url="http://localhost:3001/mcp",
            auth_mode="user_token",  # type: ignore[arg-type] — raw wire value
        )

        # Assert
        assert request.auth_mode is McpAuthMode.USER_TOKEN

    def test_create_rejects_unknown_mode(self):
        # Act & Assert — 422 at the request boundary (never silently degraded)
        with pytest.raises(ValidationError, match="auth_mode"):
            McpServerCreateRequest(
                name="server-x",
                url="http://localhost:3001/mcp",
                auth_mode="basic",  # type: ignore[arg-type] — raw wire value
            )

    def test_validate_accepts_user_token(self):
        # Act
        request = McpServerValidateRequest(
            url="http://localhost:3001/mcp",
            auth_mode="user_token",  # type: ignore[arg-type] — raw wire value
        )

        # Assert
        assert request.auth_mode is McpAuthMode.USER_TOKEN

    def test_validate_rejects_unknown_mode(self):
        # Act & Assert
        with pytest.raises(ValidationError, match="auth_mode"):
            McpServerValidateRequest(
                url="http://localhost:3001/mcp",
                auth_mode="none",  # type: ignore[arg-type] — raw wire value
            )


# -- update (partial) request model ----------------------------------------------


class TestUpdateRequestAuthMode:
    """PUT ``auth_mode``: ABSENT = preserve, value = update, null = rejected."""

    def test_absent_auth_mode_is_unset_not_null(self):
        # Act
        request = McpServerUpdateRequest(url="http://new:3001/mcp")

        # Assert — exclude_unset merge sees NO auth_mode (preserve)
        dumped = request.model_dump(exclude_unset=True)
        assert "auth_mode" not in dumped
        assert request.auth_mode is None

    def test_explicit_value_is_set_and_typed(self):
        # Act
        request = McpServerUpdateRequest(auth_mode="user_token")  # type: ignore[arg-type]

        # Assert — the merge payload carries the mode
        dumped = request.model_dump(exclude_unset=True)
        assert dumped["auth_mode"] is McpAuthMode.USER_TOKEN

    def test_explicit_headers_mode_is_set(self):
        # Act
        request = McpServerUpdateRequest(auth_mode="headers")  # type: ignore[arg-type]

        # Assert — a deliberate downgrade back to headers is an UPDATE
        dumped = request.model_dump(exclude_unset=True)
        assert dumped["auth_mode"] is McpAuthMode.HEADERS

    def test_rejects_unknown_mode(self):
        # Act & Assert
        with pytest.raises(ValidationError, match="auth_mode"):
            McpServerUpdateRequest(auth_mode="token")  # type: ignore[arg-type]

    def test_rejects_explicit_null_auth_mode(self):
        # Act & Assert — null would bypass the typed merge (model_copy does
        # not re-validate) and break the NOT NULL column downstream
        with pytest.raises(ValidationError, match="auth_mode"):
            McpServerUpdateRequest(auth_mode=None)

    def test_absent_auth_mode_does_not_trigger_the_null_guard(self):
        """Only an EXPLICIT null is rejected; the absent field stays valid."""
        # Act & Assert — no raise
        request = McpServerUpdateRequest(headers={"X-Only": "v"})
        assert "auth_mode" not in request.model_fields_set
