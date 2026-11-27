"""REST routes for the RAG spaces (REQ-010).

Thin HTTP layer over the space use cases: ``POST /api/v1/spaces`` (idempotent
create), ``GET /api/v1/spaces`` (the caller's list) and
``DELETE /api/v1/spaces/{agent_slug}`` (cascade, strict 404). The user id
ALWAYS comes from the authenticated identity (RLS contextvar) — never from a
client-supplied header or parameter.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, status

from application.requests.space_request import CreateSpaceRequest
from application.use_cases.create_rag_space import CreateRagSpaceUseCase
from application.use_cases.delete_rag_space import DeleteRagSpaceUseCase
from application.use_cases.list_rag_spaces import ListRagSpacesUseCase
from dependencies import (
    forget_space_tracking,
    get_create_rag_space_use_case,
    get_delete_rag_space_use_case,
    get_list_rag_spaces_use_case,
)
from domain.entities.rag_space import RagSpace
from infrastructure.database.rls_context import current_user_id

logger = logging.getLogger(__name__)

space_router = APIRouter(tags=["Spaces"])


def _caller_user_id() -> str:
    """Return the authenticated caller id (empty in the legacy dev path)."""
    return current_user_id.get() or ""


@space_router.post(
    "/spaces",
    status_code=status.HTTP_201_CREATED,
    responses={
        422: {"description": "Validation error — invalid agent_slug charset."},
    },
)
async def create_space(
    request: CreateSpaceRequest,
    use_case: Annotated[CreateRagSpaceUseCase, Depends(get_create_rag_space_use_case)],
) -> RagSpace:
    """Idempotently provision the caller's space for ``agent_slug`` (REQ-010).

    A replay returns the existing record (refreshing its ``last_used_at``);
    the user id always comes from the authenticated credential.
    """
    return await use_case.execute(
        user_id=_caller_user_id(), agent_slug=request.agent_slug
    )


@space_router.get("/spaces")
async def list_spaces(
    use_case: Annotated[ListRagSpacesUseCase, Depends(get_list_rag_spaces_use_case)],
) -> list[RagSpace]:
    """List the caller's spaces (never another user's, REQ-010)."""
    return await use_case.execute(user_id=_caller_user_id())


@space_router.delete(
    "/spaces/{agent_slug}",
    status_code=status.HTTP_200_OK,
    responses={
        404: {"description": "Space not found — no partial deletion occurs."},
        422: {
            "description": "Validation error — invalid agent_slug charset (REQ-002)."
        },
    },
)
async def delete_space(
    agent_slug: str,
    use_case: Annotated[DeleteRagSpaceUseCase, Depends(get_delete_rag_space_use_case)],
) -> dict:
    """Cascade-delete the caller's space (objects + tracked tables + record).

    A nonexistent space answers 404 and performs NO partial deletion
    (REQ-010, strict REST semantics).
    """
    await use_case.execute(user_id=_caller_user_id(), agent_slug=agent_slug)
    # Code-review iteration-4 🟡-4: forget the deleted space's in-memory
    # tracking markers so the seen-``working_dir`` set stays bounded to the
    # LIVE spaces (a recreated scope re-records from zero).
    forget_space_tracking(_caller_user_id(), agent_slug)
    return {"message": "Space deleted", "agent_slug": agent_slug}
