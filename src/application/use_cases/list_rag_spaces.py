"""Space-list use case: the caller's RAG spaces (REQ-010)."""

from domain.entities.rag_space import RagSpace
from domain.ports.rag_space_repository import RagSpaceRepository


class ListRagSpacesUseCase:
    """List the caller's (own) space records."""

    def __init__(self, repository: RagSpaceRepository) -> None:
        self._repository = repository

    async def execute(self, *, user_id: str) -> list[RagSpace]:
        """Return the caller's spaces (RLS + explicit user_id filter).

        Args:
            user_id: The authenticated caller id.

        Returns:
            The caller's spaces ordered by creation time — never another
            user's records (REQ-010).
        """
        return await self._repository.list_for_user(user_id)
