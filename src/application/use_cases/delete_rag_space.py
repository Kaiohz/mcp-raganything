"""Space-delete use case: strict 404 + cascade (files, tables, record).

Cascade order (REQ-010): the record is checked FIRST — a nonexistent space
raises :class:`~domain.errors.scope.SpaceNotFoundError` (404) and performs NO
partial deletion. For an existing space the cascade deletes the MinIO
objects under ``{user_id}/{agent_slug}/`` (delete-by-prefix), drops every
tracked ``classical_rag_*`` table (the ``working_dirs`` tracked on the
record, REQ-008) and finally deletes the record itself.
"""

import logging

from domain.errors.messages import ErrorMessage
from domain.errors.scope import SpaceNotFoundError
from domain.logging.messages import LogMessage
from domain.ports.rag_space_repository import RagSpaceRepository
from domain.ports.storage_port import StoragePort
from domain.ports.vector_table_manager import VectorTableManagerPort
from domain.services.scope import validate_agent_slug

logger = logging.getLogger(__name__)


class DeleteRagSpaceUseCase:
    """Cascade-delete one space record and all of its storage artifacts."""

    def __init__(
        self,
        repository: RagSpaceRepository,
        storage: StoragePort,
        bucket: str,
        vector_tables: VectorTableManagerPort,
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._bucket = bucket
        self._vector_tables = vector_tables

    async def execute(self, *, user_id: str, agent_slug: str) -> str:
        """Delete the caller's space and its data (strict 404, no partials).

        Args:
            user_id: The authenticated caller id — the cascade NEVER touches
                another user's objects (the prefix is composed from the
                credential identity).
            agent_slug: The agent slug to cascade-delete.

        Returns:
            The deleted agent slug.

        Raises:
            AgentScopeInvalidError: When the slug violates the charset rule
                (via :func:`validate_agent_slug`, mapped to 422).
            SpaceNotFoundError: When the record does not exist (404 — checked
                BEFORE any deletion so nothing is partially removed).
        """
        validate_agent_slug(agent_slug)
        space = await self._repository.get(user_id, agent_slug)
        if space is None:
            raise SpaceNotFoundError(
                ErrorMessage.SPACE_NOT_FOUND.format(agent_slug=agent_slug)
            )
        await self._storage.remove_prefix(self._bucket, f"{user_id}/{agent_slug}/")
        for working_dir in space.working_dirs:
            await self._vector_tables.drop_table(working_dir)
        await self._repository.delete(user_id, agent_slug)
        logger.info(
            LogMessage.SPACE_DELETED,
            user_id,
            agent_slug,
            len(space.working_dirs),
        )
        return agent_slug
