"""Memory-store domain errors (composable-agents Store File API client).

Error details are always safe, fixed sentences: upstream bodies, upstream
statuses with credentials, stack traces and full memory content never reach
the tool output.
"""

from domain.errors.base import DomainError
from domain.errors.codes import ErrorCode


class MemoryStoreError(DomainError):
    """Base class for memory-store domain failures."""


class MemoryValidationError(MemoryStoreError):
    """Memory name/content/mode rejected before any upstream request."""

    status_code = ErrorCode.BAD_REQUEST


class MemoryIdentityError(MemoryStoreError):
    """Fail-closed: no authenticated user identity or forwardable credential."""

    status_code = ErrorCode.UNAUTHORIZED


class MemoryConflictError(MemoryStoreError):
    """``if_exists='fail'`` and the memory already exists (no PUT issued)."""

    status_code = ErrorCode.CONFLICT


class MemoryUpstreamError(MemoryStoreError):
    """Upstream Store File API failure, timeout or malformed response."""

    status_code = ErrorCode.BAD_GATEWAY
