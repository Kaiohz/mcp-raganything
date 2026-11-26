"""Unit tests for the validate-with-id merge rule (``mcp_validate_merge``).

The rule (UI BUG-001, edit-mode "Test connection"): a validate probe carrying
the ``id`` of a stored server must run with the STORED (decrypted) credentials
unless the payload explicitly overrides them. A masked-hidden UI (REQ-016)
cannot send stored secrets back, so absent and MASKED-EMPTY values
(``headers={}``, ``env={}``, ``auth_token=""``/``None``) must NEVER clear a
stored secret, while a real value replaces the stored one. ``auth_mode``
(REQ-004, non-secret) follows the current validate semantics: the explicitly
provided request value wins, the stored one otherwise. Without a stored entry
the legacy stateless candidate defaults apply (pre-fix parity).
"""

from datetime import UTC, datetime

from domain.entities.mcp_server_config import McpAuthMode
from domain.entities.mcp_server_registry_entry import RegisteredMcpServer
from domain.services.mcp_validate_merge import (
    McpValidateOverrides,
    resolve_validate_candidate,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _overrides(
    *,  # keyword-only: reads like the route's payload mapping
    url: str | None = None,
    headers: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
    auth_token: str | None = None,
    auth_mode: McpAuthMode | None = None,
    source_type: str | None = None,
    openapi_url: str | None = None,
) -> McpValidateOverrides:
    """Build the dirty-field overrides (``None`` = masked/absent)."""
    return McpValidateOverrides(
        url=url,
        headers=headers,
        env=env,
        auth_token=auth_token,
        auth_mode=auth_mode,
        source_type=source_type,
        openapi_url=openapi_url,
    )


def _stored(
    *,
    name: str = "web-search",
    url: str = "https://stored.example.com/mcp",
    headers: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
    auth_token: str | None = "stored-bearer",
    auth_mode: McpAuthMode = McpAuthMode.HEADERS,
    source_type: str = "external",
    openapi_url: str | None = None,
) -> RegisteredMcpServer:
    return RegisteredMcpServer(
        name=name,
        url=url,
        headers=headers if headers is not None else {"X-API-Key": "stored-key"},
        env=env if env is not None else {"STORED_VAR": "1"},
        auth_token=auth_token,
        auth_mode=auth_mode,
        tool_count=3,
        source_type=source_type,
        openapi_url=openapi_url,
        created_at=_now(),
        updated_at=_now(),
    )


class TestStoredEntrySuppliesDefaults:
    """Withheld payload fields take the stored (decrypted) values."""

    def test_every_withheld_field_comes_from_storage(self):
        spec = resolve_validate_candidate(
            _stored(auth_mode=McpAuthMode.USER_TOKEN),
            _overrides(),
        )
        assert spec.url == "https://stored.example.com/mcp"
        assert spec.headers == {"X-API-Key": "stored-key"}
        assert spec.env == {"STORED_VAR": "1"}
        assert spec.auth_token == "stored-bearer"
        assert spec.auth_mode is McpAuthMode.USER_TOKEN
        assert spec.source_type == "external"

    def test_masked_empty_payload_values_never_clear_the_stored_secrets(self):
        """The exact edit-dialog shape: ``headers={}``, ``env={}``, ``auth_token=""``.

        The masked UI submits empty credential editors — the stored
        values must survive the merge (BUG-001's root cause).
        """
        spec = resolve_validate_candidate(
            _stored(),
            _overrides(
                url="https://stored.example.com/mcp",  # unchanged, prefilled
                headers={},
                env={},
                auth_token="",  # masked-empty string
                auth_mode=McpAuthMode.HEADERS,
                source_type="external",
            ),
        )
        assert spec.headers == {"X-API-Key": "stored-key"}
        assert spec.env == {"STORED_VAR": "1"}
        assert spec.auth_token == "stored-bearer"

    def test_explicit_null_auth_token_keeps_the_stored_token(self):
        spec = resolve_validate_candidate(
            _stored(),
            _overrides(),
        )
        assert spec.auth_token == "stored-bearer"

    def test_stored_entry_is_never_mutated(self):
        stored = _stored()
        spec = resolve_validate_candidate(stored, _overrides())
        spec.headers["X-API-Key"] = "tampered"
        spec.env["STORED_VAR"] = "tampered"
        assert stored.headers == {"X-API-Key": "stored-key"}
        assert stored.env == {"STORED_VAR": "1"}


class TestExplicitOverrides:
    """A real (non-empty) payload value replaces the stored one."""

    def test_dirty_headers_replace_the_stored_ones_wholesale(self):
        spec = resolve_validate_candidate(
            _stored(),
            _overrides(headers={"X-API-Key": "changed"}),
        )
        assert spec.headers == {"X-API-Key": "changed"}

    def test_dirty_env_replaces_the_stored_one(self):
        spec = resolve_validate_candidate(
            _stored(),
            _overrides(env={"API_KEY": "changed"}),
        )
        assert spec.env == {"API_KEY": "changed"}

    def test_dirty_url_and_auth_token_override(self):
        spec = resolve_validate_candidate(
            _stored(),
            _overrides(
                url="https://changed.example.com/mcp",
                auth_token="changed-bearer",
            ),
        )
        assert spec.url == "https://changed.example.com/mcp"
        assert spec.auth_token == "changed-bearer"

    def test_dirty_auth_mode_overrides_the_stored_mode(self):
        spec = resolve_validate_candidate(
            _stored(auth_mode=McpAuthMode.HEADERS),
            _overrides(auth_mode=McpAuthMode.USER_TOKEN),
        )
        assert spec.auth_mode is McpAuthMode.USER_TOKEN

    def test_stored_auth_mode_survives_when_withheld(self):
        spec = resolve_validate_candidate(
            _stored(auth_mode=McpAuthMode.USER_TOKEN),
            _overrides(),
        )
        assert spec.auth_mode is McpAuthMode.USER_TOKEN

    def test_empty_string_url_falls_back_to_the_stored_url(self):
        spec = resolve_validate_candidate(
            _stored(),
            _overrides(url=""),
        )
        assert spec.url == "https://stored.example.com/mcp"

    def test_stored_openapi_url_survives_without_a_payload_url(self):
        spec = resolve_validate_candidate(
            _stored(
                source_type="openapi",
                openapi_url="https://spec.example.com/openapi.json",
            ),
            _overrides(source_type="openapi"),
        )
        assert spec.openapi_url == "https://spec.example.com/openapi.json"

    def test_dirty_openapi_url_overrides(self):
        spec = resolve_validate_candidate(
            _stored(
                source_type="openapi",
                openapi_url="https://spec.example.com/openapi.json",
            ),
            _overrides(
                source_type="openapi",
                openapi_url="https://changed.example.com/openapi.json",
            ),
        )
        assert spec.openapi_url == "https://changed.example.com/openapi.json"


class TestStoredOpenapiEntry:
    """Withheld source-type fields keep the stored openapi shape coherent."""

    def test_openapi_entry_keeps_source_type_and_spec_url(self):
        spec = resolve_validate_candidate(
            _stored(
                url="http://api:8000/generated/petstore/mcp",
                headers={"X-API-Key": "stored-key"},
                source_type="openapi",
                openapi_url="https://spec.example.com/openapi.json",
            ),
            _overrides(),
        )
        assert spec.source_type == "openapi"
        assert spec.openapi_url == "https://spec.example.com/openapi.json"
        assert spec.headers == {"X-API-Key": "stored-key"}


class TestNoStoredEntryLegacyDefaults:
    """Without ``id`` the stateless candidate defaults apply (pre-fix parity)."""

    def test_all_masked_fields_take_the_legacy_defaults(self):
        spec = resolve_validate_candidate(
            None,
            _overrides(),
        )
        assert spec.url is None
        assert spec.headers == {}
        assert spec.env == {}
        assert spec.auth_token is None
        assert spec.auth_mode is McpAuthMode.HEADERS
        assert spec.source_type == "external"
        assert spec.openapi_url is None

    def test_explicit_values_pass_through_unchanged(self):
        spec = resolve_validate_candidate(
            None,
            _overrides(
                url="https://target.example.com/mcp",
                headers={"X-API-Key": "candidate-key"},
                env={"VAR": "candidate"},
                auth_token="candidate-token",
                auth_mode=McpAuthMode.USER_TOKEN,
                source_type="external",
            ),
        )
        assert spec.url == "https://target.example.com/mcp"
        assert spec.headers == {"X-API-Key": "candidate-key"}
        assert spec.env == {"VAR": "candidate"}
        assert spec.auth_token == "candidate-token"
        assert spec.auth_mode is McpAuthMode.USER_TOKEN

    def test_empty_string_auth_token_is_dropped_without_storage(self):
        spec = resolve_validate_candidate(
            None,
            _overrides(auth_token=""),
        )
        assert spec.auth_token is None

    def test_legacy_default_mappings_are_not_shared_mutable_state(self):
        first = resolve_validate_candidate(None, _overrides())
        first.headers["X-API-Key"] = "tampered"
        first.env["VAR"] = "tampered"
        second = resolve_validate_candidate(None, _overrides())
        assert second.headers == {}
        assert second.env == {}
