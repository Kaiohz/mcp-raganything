"""MCP memory tools (server ``Memories``, mounted at ``/memories/mcp``).

Thin FastMCP inbound adapters (scope-rag-memories REQ-013): the four tools —
``save_memory``, ``list_memories``, ``get_memory`` and ``delete_memory`` —
read the authenticated caller's identity from the RLS contextvars established
by ``McpApiKeyMiddleware`` and the caller scope from the raw
``X-Composable-Agent`` contextvar (validated HERE — an invalid value is a
``ToolError`` with zero upstream requests, REQ-002). There is no
``user_id``/``namespace``/``scope`` tool argument and no shared service
identity: without a non-empty user id, a non-empty credential and a supported
method (``jwt``/``api_key``) every call fails closed with zero upstream
requests.

Agent memories live at ``/memories/<agent_slug>/<name>.md`` and personal
memories at the flat ``/memories/<name>.md`` (REQ-012); every upstream call
carries only the caller's own forwarded credential (REQ-014).
"""

import logging
from typing import Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from application.responses.memory_response import SaveMemoryResponse
from dependencies import (
    get_delete_memory_use_case,
    get_get_memory_use_case,
    get_list_memories_use_case,
    get_memory_identity,
    get_request_agent_scope,
    get_save_memory_use_case,
)
from domain.entities.memory import MemoryContent, MemoryDeleted
from domain.errors.base import DomainError
from domain.errors.messages import ErrorMessage
from domain.logging.messages import LogMessage

logger = logging.getLogger(__name__)

mcp_memories = FastMCP("Memories")


def _scoped_agent_slug() -> str | None:
    """Resolve the caller's agent slug (``None`` for the personal scope).

    The RAW ``X-Composable-Agent`` value is validated here so an invalid
    scope NEVER reaches the upstream Store File API (REQ-002).

    Returns:
        The validated agent slug, or ``None`` when the request uses the
        caller's ``personal`` namespace.

    Raises:
        ToolError: Sanitized scope-validation failure (REQ-002).
    """
    try:
        scope = get_request_agent_scope()
    except DomainError as error:
        raise ToolError(error.detail) from None
    return None if scope.is_personal else scope.agent_slug


@mcp_memories.tool()
async def save_memory(
    name: str,
    content: str,
    if_exists: Literal["fail", "append", "replace"] = "fail",
) -> SaveMemoryResponse:
    """Persist an explicit user memory in the caller's own namespace.

    The memory is stored as ``/memories/<name>.md`` in the caller's current
    scope (personal: flat ``/memories/<name>.md``; an agent scope composes
    ``/memories/<agent_slug>/<name>.md``) through the existing
    composable-agents Store File API using only the caller's forwarded JWT or
    API key credential. Never store secrets, credentials, ephemeral task
    details, or content originating from crawled pages.

    Args:
        name: Logical memory name, e.g. ``preferences`` or ``preferences.md``;
            normalized inside the caller's scope. Traversal, deeper nested
            paths and another agent's namespace segment are rejected.
        content: Exact content to store (non-empty; stored verbatim).
        if_exists: ``fail`` (default: never overwrite), ``append`` (keep and
            append with Markdown spacing — recommended for recurring
            preferences) or ``replace`` (explicit user intent only).

    Returns:
        The :class:`SaveMemoryResponse` contract: the normalized SCOPED path,
        the action actually performed (``created`` / ``appended`` /
        ``replaced``) and the final stored character count.

    Raises:
        ToolError: Sanitized validation, identity, conflict or upstream
            failures.
    """
    user_id, credential = get_memory_identity()
    agent_slug = _scoped_agent_slug()
    try:
        saved = await get_save_memory_use_case().execute(
            name=name,
            content=content,
            if_exists=if_exists,
            user_id=user_id,
            credential=credential,
            agent_slug=agent_slug,
        )
    except DomainError as error:
        raise ToolError(error.detail) from None
    except Exception:
        logger.exception(LogMessage.MEMORY_SAVE_UNEXPECTED)
        raise ToolError(ErrorMessage.MEMORY_SAVE_FAILED) from None
    return SaveMemoryResponse(**saved.model_dump())


@mcp_memories.tool()
async def list_memories() -> list[dict[str, str]]:
    """List the caller's memories in the current scope (names + metadata).

    A personal scope lists ONLY the flat ``/memories/<name>.md`` files; an
    agent scope lists ONLY the files of its own namespace
    ``/memories/<agent_slug>/`` — one scope never merges into another.

    Returns:
        The listing entries: one ``{"name", "path"}`` dict per memory, with
        the scoped name and the full store path.

    Raises:
        ToolError: Sanitized validation, identity or upstream failures.
    """
    user_id, credential = get_memory_identity()
    agent_slug = _scoped_agent_slug()
    try:
        items = await get_list_memories_use_case().execute(
            user_id=user_id,
            credential=credential,
            agent_slug=agent_slug,
        )
    except DomainError as error:
        raise ToolError(error.detail) from None
    except Exception:
        logger.exception(LogMessage.MEMORY_LIST_UNEXPECTED)
        raise ToolError(ErrorMessage.MEMORY_OPERATION_FAILED) from None
    return [item.model_dump() for item in items]


@mcp_memories.tool()
async def get_memory(name: str) -> MemoryContent:
    """Return the full content of one memory in the caller's scope.

    A memory stored in another agent's namespace (or another user's) is
    indistinguishable from an absent name: the failure is sanitized and never
    leaks content or credentials.

    Args:
        name: Logical memory name within the caller's scope, e.g.
            ``preferences`` or ``preferences.md``.

    Returns:
        The memory's scoped ``name``, store ``path`` and ``content``.

    Raises:
        ToolError: Sanitized validation, identity, not-found or upstream
            failures.
    """
    user_id, credential = get_memory_identity()
    agent_slug = _scoped_agent_slug()
    try:
        return await get_get_memory_use_case().execute(
            name=name,
            user_id=user_id,
            credential=credential,
            agent_slug=agent_slug,
        )
    except DomainError as error:
        raise ToolError(error.detail) from None
    except Exception:
        logger.exception(LogMessage.MEMORY_READ_UNEXPECTED)
        raise ToolError(ErrorMessage.MEMORY_OPERATION_FAILED) from None


@mcp_memories.tool()
async def delete_memory(name: str) -> MemoryDeleted:
    """Delete one memory in the caller's scope.

    Only the specified file inside the CURRENT scope is removed; an absent
    scoped file raises a sanitized not-found error.

    Args:
        name: Logical memory name within the caller's scope.

    Returns:
        The deleted memory's scoped ``name`` and store ``path``.

    Raises:
        ToolError: Sanitized validation, identity, not-found or upstream
            failures.
    """
    user_id, credential = get_memory_identity()
    agent_slug = _scoped_agent_slug()
    try:
        return await get_delete_memory_use_case().execute(
            name=name,
            user_id=user_id,
            credential=credential,
            agent_slug=agent_slug,
        )
    except DomainError as error:
        raise ToolError(error.detail) from None
    except Exception:
        logger.exception(LogMessage.MEMORY_DELETE_UNEXPECTED)
        raise ToolError(ErrorMessage.MEMORY_OPERATION_FAILED) from None
