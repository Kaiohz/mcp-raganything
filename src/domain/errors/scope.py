"""Scope / RAG-space domain errors (scope-rag-memories spec).

Scope resolution failures are request-validation errors (422), space-record
lookups raise the strict 404 REST semantics (REQ-010, decided: no partial
deletion).
"""

from domain.errors.base import DomainError
from domain.errors.codes import ErrorCode


class AgentScopeInvalidError(DomainError):
    """The ``X-Composable-Agent`` value does not satisfy the slug charset rule.

    Mapped to HTTP 422 on REST and surfaced as a sanitized ``ToolError``
    before any storage operation on the MCP tools (REQ-002).
    """

    status_code = ErrorCode.UNPROCESSABLE_ENTITY


class SpaceNotFoundError(DomainError):
    """The referenced RAG space record does not exist for the caller.

    Mapped to HTTP 404; the cascade delete checks the record FIRST so a
    nonexistent space never performs a partial deletion (REQ-010).
    """

    status_code = ErrorCode.NOT_FOUND


class SpaceLimitExceededError(DomainError):
    """The caller has reached the per-user cap of LIVE RAG spaces.

    Mapped to HTTP 422 (security review FINDING-4): an authenticated caller
    must not be able to generate unbounded ``rag_spaces`` rows. Deleting a
    space frees capacity immediately (the cap counts live rows only).
    """

    status_code = ErrorCode.UNPROCESSABLE_ENTITY


class ReservedAgentSlugError(DomainError):
    """A space was requested for a slug reserved by the platform design.

    ``personal`` is intentionally NEVER a stored space (migration 005's
    invariant — REVIEW FINDING-3): it names the caller's unprovisioned
    personal tree, and a personal record would let ``DELETE /spaces/personal``
    cascade the caller's ENTIRE personal tree. Mapped to HTTP 422.
    """

    status_code = ErrorCode.UNPROCESSABLE_ENTITY
