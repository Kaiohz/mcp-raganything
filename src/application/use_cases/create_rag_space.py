"""Space-create use case: idempotent provisioning of one RAG space (REQ-010)."""

from domain.constants import MAX_SPACES_PER_USER, PERSONAL_AGENT_SLUG
from domain.entities.rag_space import RagSpace
from domain.errors.messages import ErrorMessage
from domain.errors.scope import ReservedAgentSlugError, SpaceLimitExceededError
from domain.ports.rag_space_repository import RagSpaceRepository
from domain.services.scope import validate_agent_slug


class CreateRagSpaceUseCase:
    """Create-or-return the caller's space record for one agent slug."""

    def __init__(self, repository: RagSpaceRepository) -> None:
        self._repository = repository

    async def execute(self, *, user_id: str, agent_slug: str) -> RagSpace:
        """Idempotently provision the space (the slug is validated upstream).

        A replay (existing record) refreshes ``last_used_at``; concurrent
        creations are resolved by the unique ``(user_id, agent_slug)``
        constraint. The repository resolves the upsert atomically in ONE
        round-trip (no useless pre-read — code-review 🟡-4).

        Guards (security review iteration-8): ``personal`` is NEVER stored —
        it names the caller's unprovisioned personal tree, and a personal
        record would enable the self-destructive
        ``DELETE /spaces/personal`` chain (FINDING-3); and the caller's LIVE
        spaces are capped by
        :data:`domain.constants.MAX_SPACES_PER_USER` (FINDING-4) — a replay
        of an existing space is exempt, so idempotent create keeps working
        at the cap. The lazy ``ensure_user_space`` path skips ``personal``
        by construction (verified + pinned behaviorally).

        Args:
            user_id: The authenticated caller id (never from the body).
            agent_slug: The validated agent slug (charset rule REQ-002).

        Returns:
            The persisted space record.

        Raises:
            AgentScopeInvalidError: When the slug violates the charset rule.
            ReservedAgentSlugError: When the slug is the reserved
                ``personal`` scope (never a stored space).
            SpaceLimitExceededError: When the caller already holds the
                per-user maximum of live spaces (new spaces only).
        """
        validate_agent_slug(agent_slug)
        if agent_slug == PERSONAL_AGENT_SLUG:
            raise ReservedAgentSlugError(
                ErrorMessage.SPACE_AGENT_SLUG_RESERVED.format(agent_slug=agent_slug)
            )
        live_spaces = await self._repository.list_for_user(user_id)
        other_live = [s for s in live_spaces if s.agent_slug != agent_slug]
        if len(other_live) >= MAX_SPACES_PER_USER:
            raise SpaceLimitExceededError(
                ErrorMessage.SPACE_LIMIT_EXCEEDED.format(limit=MAX_SPACES_PER_USER)
            )
        return await self._repository.ensure_user_space(user_id, agent_slug)
