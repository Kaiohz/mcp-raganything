"""Memory-delete use case: scoped removal of one memory (REQ-013).

Fails closed on a missing identity/credential; an absent scoped file raises
the sanitized domain "not found" error (the adapter maps the upstream 404)
and the deletion only ever removes the file inside the caller's own scope.
"""

from domain.entities.memory import (
    MemoryCredential,
    MemoryDeleted,
    memory_scope_path,
    normalize_memory_name,
)
from domain.errors.memory_store import MemoryIdentityError
from domain.errors.messages import ErrorMessage
from domain.ports.memory_store import MemoryStorePort


class DeleteMemoryUseCase:
    """Delete one memory inside the caller's scope."""

    def __init__(self, store: MemoryStorePort) -> None:
        self._store = store

    async def execute(
        self,
        *,
        name: str,
        user_id: str,
        credential: MemoryCredential,
        agent_slug: str | None = None,
    ) -> MemoryDeleted:
        """Delete one memory file inside the caller's scope.

        Args:
            name: Logical memory name.
            user_id: Authenticated caller id (gates the call, never stored).
            credential: The caller's own forwardable credential.
            agent_slug: The caller's agent slug, or ``None`` for the flat
                personal namespace.

        Returns:
            The :class:`MemoryDeleted` (name + scoped store path deleted).

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
        path = memory_scope_path(stem, agent_slug)
        await self._store.delete(stem, credential, agent_slug=agent_slug)
        return MemoryDeleted(name=stem, path=path)
