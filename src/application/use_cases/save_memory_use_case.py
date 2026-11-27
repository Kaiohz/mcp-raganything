"""Memory-save use case: validated per-user memory persistence.

Owns the invocation policy (authenticated identity, fail-closed), memory
name normalization, the ``fail``/``append``/``replace`` overwrite policy with
exact Markdown spacing, and final ``chars`` accounting. Upstream reads and
writes go through the :class:`MemoryStorePort`; the response is produced only
after the upstream write succeeds.
"""

import logging
import time
from typing import Literal

from domain.entities.memory import (
    MemoryAction,
    MemoryCredential,
    SavedMemory,
    memory_scope_path,
    normalize_memory_name,
)
from domain.errors.memory_store import (
    MemoryConflictError,
    MemoryIdentityError,
    MemoryValidationError,
)
from domain.errors.messages import ErrorMessage
from domain.logging.messages import LogMessage
from domain.ports.memory_store import MemoryStorePort

logger = logging.getLogger(__name__)

MemoryExistsMode = Literal["fail", "append", "replace"]
_MODES: tuple[str, ...] = ("fail", "append", "replace")


def _stitched(existing: str, supplied: str) -> str:
    """Join existing and supplied content with exact Markdown spacing.

    Rules: no separator for empty existing content and for content already
    ending in ``\\n\\n``; one newline when it ends with exactly one ``\\n``;
    otherwise ``\\n\\n``. The supplied content is appended verbatim.
    """
    if not existing:
        return supplied
    if existing.endswith("\n\n"):
        return existing + supplied
    if existing.endswith("\n"):
        return existing + "\n" + supplied
    return existing + "\n\n" + supplied


class SaveMemoryUseCase:
    """Persist an explicit user memory in the caller's own namespace."""

    def __init__(self, store: MemoryStorePort) -> None:
        self._store = store

    async def execute(
        self,
        *,
        name: str,
        content: str,
        if_exists: MemoryExistsMode,
        user_id: str,
        credential: MemoryCredential,
        agent_slug: str | None = None,
    ) -> SavedMemory:
        """Validate, resolve the overwrite policy and persist the memory.

        Args:
            name: Logical memory name (flat personal stem, or explicitly
                ``{agent_slug}/<stem>`` inside an agent scope — REQ-012).
            content: Exact content to store (must be non-empty).
            if_exists: ``fail`` / ``append`` / ``replace``.
            user_id: Authenticated caller id (gates the call; never stored
                nor forwarded).
            credential: The caller's own forwardable credential.
            agent_slug: The caller's agent slug; ``None`` keeps the flat
                personal namespace (REQ-012/REQ-014).

        Returns:
            The :class:`SavedMemory` for the successful upstream PUT (its
            ``path`` is the SCOPED store path).

        Raises:
            MemoryIdentityError: Fail-closed without identity/credential
                (zero upstream requests) or when upstream rejects the
                forwarded credential.
            MemoryValidationError: Invalid name, empty content or mode.
            MemoryConflictError: ``fail`` mode with an existing memory.
            MemoryUpstreamError: Upstream timeout/failure/malformed response.
        """
        if if_exists not in _MODES:
            raise MemoryValidationError(ErrorMessage.MEMORY_MODE_INVALID)
        if (
            not user_id
            or not credential.credential
            or credential.method
            not in (
                "jwt",
                "api_key",
            )
        ):
            logger.warning(LogMessage.MEMORY_IDENTITY_MISSING)
            raise MemoryIdentityError(ErrorMessage.MEMORY_IDENTITY_REQUIRED)
        if not isinstance(content, str) or not content:
            raise MemoryValidationError(ErrorMessage.MEMORY_CONTENT_REQUIRED)

        stem = normalize_memory_name(name, agent_slug=agent_slug)
        path = memory_scope_path(stem, agent_slug)
        started = time.monotonic()
        existing = await self._store.read(stem, credential, agent_slug=agent_slug)
        action: MemoryAction
        if existing is None:
            final_content = content
            action = "created"
        elif if_exists == "fail":
            raise MemoryConflictError(ErrorMessage.MEMORY_EXISTS)
        elif if_exists == "append":
            final_content = _stitched(existing, content)
            action = "appended"
        else:
            final_content = content
            action = "replaced"
        await self._store.write(stem, final_content, credential, agent_slug=agent_slug)
        logger.info(
            LogMessage.MEMORY_SAVE_COMPLETED,
            action,
            len(final_content),
            (time.monotonic() - started) * 1000,
        )
        return SavedMemory(
            path=path,
            action=action,
            chars=len(final_content),
        )
