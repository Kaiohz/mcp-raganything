"""Caller scope resolution and validation service.

The scope is ``(user_id, agent_slug)`` (scope-rag-memories spec REQ-001): the
user id always comes from the authenticated identity (the RLS contextvar —
never from a client-supplied header or parameter) while the agent slug comes
from the ``X-Composable-Agent`` header, defaulting to ``personal`` when the
header is absent. An invalid slug is a request-validation failure raised
BEFORE any storage operation (REQ-002, 422 on REST / ToolError on MCP).

This service takes the raw context values as PARAMETERS (no ambient state);
the composition rules themselves live on the :class:`AgentScope` entity
(:mod:`domain.entities.agent_scope`). Pure domain: no framework or
infrastructure knowledge.
"""

import logging
import re

from domain.constants import (
    AGENT_HEADER,
    AGENT_SLUG_PATTERN,
    PERSONAL_AGENT_SLUG,
)
from domain.entities.agent_scope import AgentScope
from domain.errors.messages import ErrorMessage
from domain.errors.scope import AgentScopeInvalidError
from domain.logging.messages import LogMessage

_logger = logging.getLogger(__name__)

__all__ = [
    "AGENT_HEADER",
    "AGENT_SLUG_PATTERN",
    "PERSONAL_AGENT_SLUG",
    "resolve_agent_scope",
    "validate_agent_slug",
]

_AGENT_SLUG_RE = re.compile(AGENT_SLUG_PATTERN)


def validate_agent_slug(value: str | None) -> str:
    """Validate a caller-supplied agent slug against the REQ-002 rule.

    Args:
        value: The raw ``X-Composable-Agent`` header value.

    Returns:
        The validated slug (lowercase ``[a-z0-9]`` body, 1..100 chars).

    Raises:
        AgentScopeInvalidError: When the value does not match
            ``^[a-z0-9][a-z0-9-]{0,99}$`` (no storage operation must happen).
    """
    if value is None or not _AGENT_SLUG_RE.fullmatch(value.strip()):
        raise AgentScopeInvalidError(
            ErrorMessage.AGENT_SCOPE_INVALID.format(header=AGENT_HEADER, value=value)
        )
    return value.strip()


def resolve_agent_scope(
    user_id: str | None, raw_agent_header: str | None
) -> AgentScope:
    """Resolve the caller scope from the credential identity and raw header.

    Args:
        user_id: The authenticated principal id from the RLS contextvar
            (``None`` in the legacy/unauthenticated path — the scope then
            carries an empty user id and composition is skipped by callers).
        raw_agent_header: The raw ``X-Composable-Agent`` header value (or
            ``None`` when absent → ``personal``).

    Returns:
        The immutable :class:`AgentScope` for the request.

    Raises:
        AgentScopeInvalidError: When the header value is present but invalid
            (REQ-002 — rejected before any storage operation).
    """
    if not raw_agent_header or not raw_agent_header.strip():
        agent_slug = PERSONAL_AGENT_SLUG
    else:
        agent_slug = validate_agent_slug(raw_agent_header)
    # DEBUG only (code-review 🟡-6): the resolution runs 2-3x per request;
    # INFO here would flood every request's logs.
    _logger.debug(LogMessage.SCOPE_RESOLVED, user_id, agent_slug)
    return AgentScope(user_id=user_id or "", agent_slug=agent_slug)
