"""Request models for the RAG spaces endpoints (REQ-010)."""

from pydantic import BaseModel, Field, field_validator

from domain.constants import PERSONAL_AGENT_SLUG
from domain.errors.messages import ErrorMessage
from domain.errors.scope import ReservedAgentSlugError
from domain.services.scope import validate_agent_slug


class CreateSpaceRequest(BaseModel):
    """Body of ``POST /api/v1/spaces`` (idempotent create, REQ-010).

    The body's ``agent_slug`` is an arbitrary valid slug — it does NOT have
    to match the caller's current request scope (composable-agents provisions
    each subagent's space with the USER's forwarded credential, REQ-018):
    validation is the REQ-002 charset rule PLUS the reserved-name rule —
    ``personal`` is never a stored space (security review FINDING-3: the
    migration-documented invariant, and the guard against the self-destructive
    ``DELETE /spaces/personal`` cascade). The user id ALWAYS comes from the
    authenticated credential, never from the body.
    """

    agent_slug: str = Field(
        ...,
        description="Agent slug of the space to provision (REQ-002 charset)",
        min_length=1,
        max_length=100,
    )

    @field_validator("agent_slug")
    @classmethod
    def validate_agent_slug_value(cls, value: str) -> str:
        """Reject values violating the REQ-002 slug rule (422)."""
        try:
            return validate_agent_slug(value)
        except Exception as error:
            raise ValueError(str(getattr(error, "detail", error))) from None

    @field_validator("agent_slug")
    @classmethod
    def reject_reserved_slugs(cls, value: str) -> str:
        """Reject ``personal`` — reserved, never a stored space (FINDING-3)."""
        if value == PERSONAL_AGENT_SLUG:
            raise ValueError(
                str(
                    ReservedAgentSlugError(
                        ErrorMessage.SPACE_AGENT_SLUG_RESERVED.format(agent_slug=value)
                    ).detail
                )
            )
        return value
