"""Memory-list use case: scoped listing of the caller's memories (REQ-013).

Fails closed on a missing identity/credential (zero upstream requests) and
delegates the listing to the :class:`MemoryStorePort` with the caller's own
credential — names and store paths come back scoped (flat personal or
``/memories/<agent_slug>/``).
"""

from domain.entities.memory import MemoryCredential, MemoryListItem
from domain.errors.memory_store import MemoryIdentityError
from domain.errors.messages import ErrorMessage
from domain.ports.memory_store import MemoryStorePort


class ListMemoriesUseCase:
    """List the memory files visible in the caller's scope."""

    def __init__(self, store: MemoryStorePort) -> None:
        self._store = store

    async def execute(
        self,
        *,
        user_id: str,
        credential: MemoryCredential,
        agent_slug: str | None = None,
    ) -> list[MemoryListItem]:
        """List the caller's scoped memories (REQ-013).

        Args:
            user_id: Authenticated caller id (gates the call, never stored).
            credential: The caller's own forwardable credential.
            agent_slug: The caller's agent slug, or ``None`` for the flat
                personal namespace.

        Returns:
            The listing entries (names + store paths), scoped: a personal
            listing never merges agent memories into it and vice versa.

        Raises:
            MemoryIdentityError: Fail-closed without identity/credential.
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
        return await self._store.list(credential, agent_slug=agent_slug)
