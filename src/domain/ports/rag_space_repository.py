"""Outbound port for the RLS-protected ``rag_spaces`` persistence (REQ-009)."""

from abc import ABC, abstractmethod

from domain.entities.rag_space import RagSpace


class RagSpaceRepository(ABC):
    """Boundary for the per-user RAG space records.

    Implementations must set the RLS GUC (``app.user_id``) on every acquired
    connection AND scope their SQL by ``user_id`` (defence in depth) — the
    caller's identity is provided explicitly, never read from ambient state.
    """

    @abstractmethod
    async def ensure_user_space(
        self, user_id: str, agent_slug: str, working_dir: str | None = None
    ) -> RagSpace:
        """Create-or-touch (idempotent) one space record and return it.

        Concurrent first-use is safe: the unique ``(user_id, agent_slug)``
        constraint resolves the race (exactly one record afterwards). A
        ``working_dir`` is appended to the tracked ``working_dirs`` array
        when provided and not already present (REQ-008); every call refreshes
        ``last_used_at`` (REQ-011).

        Args:
            user_id: The authenticated owner id.
            agent_slug: The validated agent slug (never ``personal``).
            working_dir: The composed working_dir to track, or ``None``.

        Returns:
            The persisted record.
        """

    @abstractmethod
    async def get(self, user_id: str, agent_slug: str) -> RagSpace | None:
        """Return the caller's space record, or ``None`` when absent.

        Args:
            user_id: The authenticated owner id.
            agent_slug: The agent slug to look up.
        """

    @abstractmethod
    async def list_for_user(self, user_id: str) -> list[RagSpace]:
        """List ALL the caller's space records (RLS-scoped).

        Args:
            user_id: The authenticated owner id.

        Returns:
            The user's spaces ordered by creation time.
        """

    @abstractmethod
    async def delete(self, user_id: str, agent_slug: str) -> bool:
        """Delete the caller's space record.

        Args:
            user_id: The authenticated owner id.
            agent_slug: The agent slug to delete.

        Returns:
            ``True`` when a row was deleted, ``False`` when absent.
        """
