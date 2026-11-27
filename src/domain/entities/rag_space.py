"""RAG space domain entity (scope-rag-memories REQ-009).

One space per ``(user_id, agent_slug)``: the record tracks the classical-RAG
``working_dirs`` the space created (so agent deletion can drop the matching
``classical_rag_*`` tables) plus the lifecycle timestamps.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class RagSpace(BaseModel):
    """A provisioned RAG space for one user's agent scope.

    Attributes:
        id: Row identifier (``None`` until loaded from persistence).
        user_id: Owner (always the authenticated caller in this service).
        agent_slug: The validated agent slug (``personal`` is NOT stored —
            the personal tree needs no space record).
        working_dirs: The composed classical-RAG working_dirs created inside
            this space (tracked for the delete cascade, REQ-008).
        created_at: Record creation time.
        last_used_at: Last scoped-use refresh (REQ-011).
    """

    id: UUID | None = None
    user_id: str
    agent_slug: str
    working_dirs: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    last_used_at: datetime | None = None
