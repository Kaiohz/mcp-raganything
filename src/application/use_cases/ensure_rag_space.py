"""Space-ensure use case: best-effort lazy provisioning + last_used refresh.

Owns the REQ-011 semantics: a record is created ONLY for a non-``personal``
agent slug of an authenticated user, a record-write failure NEVER blocks the
request (the failure is swallowed and logged as a warning — the space is
auto-created lazily on a later use), and every call refreshes
``last_used_at`` and tracks a supplied ``working_dir`` (REQ-008).
"""

import logging

from domain.constants import PERSONAL_AGENT_SLUG
from domain.entities.rag_space import RagSpace
from domain.logging.messages import LogMessage
from domain.ports.rag_space_repository import RagSpaceRepository

logger = logging.getLogger(__name__)


class EnsureRagSpaceUseCase:
    """Best-effort lazy provision + ``last_used_at`` refresh of one space."""

    def __init__(self, repository: RagSpaceRepository) -> None:
        self._repository = repository

    async def execute(
        self,
        *,
        user_id: str,
        agent_slug: str,
        working_dir: str | None = None,
    ) -> RagSpace | None:
        """Ensure the caller's space record exists (and track its use).

        Args:
            user_id: The authenticated caller id (empty → no-op).
            agent_slug: The validated agent slug (``personal`` → no-op: the
                personal tree carries no space record, REQ-011/REQ-019).
            working_dir: The composed working_dir to track, or ``None``.

        Returns:
            The persisted record, or ``None`` when no record is required
            (personal scope / no identity) or when the write failed
            (best-effort, never raised).
        """
        if not user_id or agent_slug == PERSONAL_AGENT_SLUG:
            return None
        try:
            return await self._repository.ensure_user_space(
                user_id, agent_slug, working_dir
            )
        except Exception as exc:  # noqa: BLE001 — best-effort by contract (REQ-011)
            logger.warning(LogMessage.SPACE_PROVISIONING_FAILED, agent_slug, str(exc))
            return None
