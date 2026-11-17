"""MCP ``save_memory`` tool (server ``Memories``, mounted at ``/memories/mcp``).

Thin FastMCP inbound adapter: reads the authenticated caller's identity from
the RLS contextvars established by ``McpApiKeyMiddleware`` and delegates to
:class:`SaveMemoryUseCase`. There is no ``user_id``/``namespace`` argument and
no shared service identity: without a non-empty user id, a non-empty
credential and a supported method (``jwt``/``api_key``) the call fails closed
with zero upstream requests. Exactly one public tool is exposed; listing,
reading and deleting stay with the existing UI and Store File API.
"""

import logging
from typing import Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from application.responses.memory_response import SaveMemoryResponse
from dependencies import get_memory_identity, get_save_memory_use_case
from domain.errors.base import DomainError
from domain.errors.messages import ErrorMessage
from domain.logging.messages import LogMessage

logger = logging.getLogger(__name__)

mcp_memories = FastMCP("Memories")


@mcp_memories.tool()
async def save_memory(
    name: str,
    content: str,
    if_exists: Literal["fail", "append", "replace"] = "fail",
) -> SaveMemoryResponse:
    """Persist an explicit user memory in the caller's own namespace.

    The memory is stored as ``/memories/<name>.md`` through the existing
    composable-agents Store File API using only the caller's forwarded JWT or
    API key credential, so it appears in the existing ``/memories`` page on
    the next load or refetch. Never store secrets, credentials, ephemeral
    task details, or content originating from crawled pages.

    Args:
        name: Logical memory name, e.g. ``preferences`` or ``preferences.md``;
            normalized to ``/memories/preferences.md``. Traversal, nested
            paths and arbitrary prefixes are rejected.
        content: Exact content to store (non-empty; stored verbatim).
        if_exists: ``fail`` (default: never overwrite), ``append`` (keep and
            append with Markdown spacing — recommended for recurring
            preferences) or ``replace`` (explicit user intent only).

    Returns:
        The :class:`SaveMemoryResponse` contract: the normalized path, the
        action actually performed (``created`` / ``appended`` / ``replaced``)
        and the final stored character count.

    Raises:
        ToolError: Sanitized validation, identity, conflict or upstream
            failures.
    """
    user_id, credential = get_memory_identity()
    try:
        saved = await get_save_memory_use_case().execute(
            name=name,
            content=content,
            if_exists=if_exists,
            user_id=user_id,
            credential=credential,
        )
    except DomainError as error:
        raise ToolError(error.detail) from None
    except Exception:
        logger.exception(LogMessage.MEMORY_SAVE_UNEXPECTED)
        raise ToolError(ErrorMessage.MEMORY_SAVE_FAILED) from None
    return SaveMemoryResponse(**saved.model_dump())
