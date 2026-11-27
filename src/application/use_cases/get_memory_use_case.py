"""Memory-read use case: scoped content retrieval (REQ-013).

Fails closed on a missing identity/credential and maps an upstream 404 to the
domain "not found" error — a memory of ANOTHER scope (or another user) is
indistinguishable from an absent name (no content ever leaks through the
error).
"""

from domain.entities.memory import (
    MemoryContent,
    MemoryCredential,
    memory_scope_path,
    normalize_memory_name,
)
from domain.errors.memory_store import (
    MemoryIdentityError,
    MemoryNotFoundError,
)
from domain.errors.messages import ErrorMessage
from domain.ports.memory_store import MemoryStorePort


class GetMemoryUseCase:
    """Return the content of one scoped memory."""

    def __init__(self, store: MemoryStorePort) -> None:
        self._store = store

    async def execute(
        self,
        *,
        name: str,
        user_id: str,
        credential: MemoryCredential,
        agent_slug: str | None = None,
    ) -> MemoryContent:
        """Read one memory inside the caller's scope.

        Args:
            name: Logical memory name (see
                :func:`domain.entities.memory.normalize_memory_name`).
            user_id: Authenticated caller id (gates the call, never stored).
            credential: The caller's own forwardable credential.
            agent_slug: The caller's agent slug, or ``None`` for the flat
                personal namespace.

        Returns:
            The :class:`MemoryContent` (scoped name, path and content).

        Raises:
            MemoryIdentityError: Fail-closed without identity/credential.
            MemoryValidationError: Invalid name.
            MemoryNotFoundError: The memory does not exist in this scope.
        """
        if (
            not user_id
            or not credential.credential
            or credential.method
            not in (
                "jwt",
                "api_key",
            )
        ):
            raise MemoryIdentityError(ErrorMessage.MEMORY_IDENTITY_REQUIRED)
        stem = normalize_memory_name(name, agent_slug=agent_slug)
        content = await self._store.read(stem, credential, agent_slug=agent_slug)
        if content is None:
            raise MemoryNotFoundError(ErrorMessage.MEMORY_NOT_FOUND.format(name=stem))
        return MemoryContent(
            name=stem,
            path=memory_scope_path(stem, agent_slug),
            content=content,
        )
