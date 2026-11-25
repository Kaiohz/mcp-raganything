"""Unit tests for the auth_mode credential-injection domain rule (REQ-004).

The rule (:func:`user_credential_headers`) lives in the DOMAIN so both the
validate probe and any future loader can share it; these tests pin it
independently of FastAPI: the injected header per authentication method and
the drop-empty semantics (no credential → no injection). An unknown
authentication method cannot reach the rule at all — the :class:`AuthContext`
entity validates ``method`` as a closed literal set.
"""

from typing import Any, cast

import pytest
from pydantic import ValidationError

from domain.entities.auth.auth_context import AuthContext
from domain.entities.mcp_server_config import McpAuthMode
from domain.services.mcp_auth_mode import user_credential_headers


def _context(method: str, credential: str) -> AuthContext:
    return AuthContext(
        user_id="user-alice",
        method=cast("Any", method),
        raw_credential=credential,
    )


class TestUserCredentialHeaders:
    """user_token mode injects the caller's credential header per method."""

    def test_user_token_jwt_injects_bearer_authorization(self):
        injected = user_credential_headers(
            McpAuthMode.USER_TOKEN, _context("jwt", "jwt-token")
        )
        assert injected == {"Authorization": "Bearer jwt-token"}

    def test_user_token_api_key_injects_x_api_key(self):
        injected = user_credential_headers(
            McpAuthMode.USER_TOKEN, _context("api_key", "key-alice")
        )
        assert injected == {"X-API-Key": "key-alice"}

    def test_headers_mode_never_injects(self):
        injected = user_credential_headers(
            McpAuthMode.HEADERS, _context("jwt", "jwt-token")
        )
        assert injected == {}

    def test_no_context_never_injects(self):
        injected = user_credential_headers(McpAuthMode.USER_TOKEN, None)
        assert injected == {}

    def test_empty_credential_never_injects(self):
        injected = user_credential_headers(McpAuthMode.USER_TOKEN, _context("jwt", ""))
        assert injected == {}

    def test_unknown_method_is_rejected_by_the_entity_before_the_rule(self):
        # The entity is a closed set (pydantic Literal): an unknown method can
        # never reach the rule, so no injection can be guessed from one.
        with pytest.raises(ValidationError):
            _context("legacy", "cred")
