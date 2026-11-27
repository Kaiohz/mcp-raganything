"""Memory domain entities and name normalization.

Pure Pydantic models plus the shared, defensive memory-name normalizer used
by BOTH the use case and the infrastructure adapter (independent confinement
to ``/memories/``: tool-level validation alone is insufficient).

No FastMCP, httpx or composable-agents knowledge is allowed here.
"""

import re
from typing import Literal

from pydantic import BaseModel

from domain.constants import AGENT_SLUG_PATTERN
from domain.errors.memory_store import MemoryValidationError
from domain.errors.messages import ErrorMessage

#: Canonical namespace root for user memory files.
MEMORY_ROOT = "/memories/"
#: Allowed logical name characters before the mandatory ``.md`` suffix.
_ALLOWED_NAME_PATTERN = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)
_ALLOWED_NAME_SET = frozenset(_ALLOWED_NAME_PATTERN)
_AGENT_SLUG_RE = re.compile(AGENT_SLUG_PATTERN)


def _split_memory_name(name: object) -> list[str]:
    """Split a caller-supplied memory name into its path segments.

    Args:
        name: The caller-supplied logical memory name.

    Returns:
        The ``/``-split segments (at most TWO: one optional agent segment +
        the stem; deeper nesting is rejected in every scope, REQ-012).

    Raises:
        MemoryValidationError: If the input is not a string or nests deeper
            than one agent segment.
    """
    if not isinstance(name, str):
        raise MemoryValidationError(ErrorMessage.MEMORY_NAME_INVALID)
    segments = name.split("/")
    if len(segments) > 2:
        raise MemoryValidationError(ErrorMessage.MEMORY_NAME_INVALID)
    return segments


def _validate_agent_segment(segment: str, agent_slug: str | None) -> None:
    """Reject a nesting segment that is not the caller's OWN agent slug.

    The namespace is composed from the caller's scope, never from the name,
    so the explicit ``{agent_slug}/{name}`` alias is accepted ONLY when it
    names the caller's own slug (REQ-012; an unscoped call — the flat
    personal namespace — rejects any nesting segment).

    Raises:
        MemoryValidationError: On a foreign (or unscoped-call) agent segment.
    """
    if agent_slug is None or segment != agent_slug:
        raise MemoryValidationError(ErrorMessage.MEMORY_NAME_INVALID)
    if not is_agent_slug_shape(segment):
        raise MemoryValidationError(ErrorMessage.MEMORY_NAME_INVALID)


def _normalize_stem(segment: str) -> str:
    """Strip EXACTLY one trailing ``.md`` and validate the filename stem.

    Raises:
        MemoryValidationError: If the stem is empty, ``.``/``..`` (a
            traversal or an empty name) or carries any character outside the
            allowed ``letters/digits/.-_`` set (this also rejects slashes,
            backslashes, encoded traversal and control characters).
    """
    stem = segment[: -len(".md")] if segment.endswith(".md") else segment
    if not stem or stem in (".", "..") or not set(stem) <= _ALLOWED_NAME_SET:
        raise MemoryValidationError(ErrorMessage.MEMORY_NAME_INVALID)
    return stem


def normalize_memory_name(name: str, *, agent_slug: str | None = None) -> str:
    """Normalize a logical memory name to its scoped-namespace stem.

    Args:
        name: Caller-supplied logical memory name (flat ``preferences`` /
            ``preferences.md``, or the caller's own agent alias
            ``agent-x/preferences.md``).
        agent_slug: The caller's validated agent slug, when the call runs in
            an agent scope (``None`` = the legacy flat personal namespace).

    Returns:
        The normalized filename stem (without ``.md``). The stored path is
        composed by :func:`memory_scope_path` from this stem and the scope.

    Raises:
        MemoryValidationError: If the name violates any rule (non-string
            input, deeper than one nesting segment, a foreign agent
            segment, an empty/traversal stem or invalid characters).
    """
    segments = _split_memory_name(name)
    if len(segments) == 2:
        _validate_agent_segment(segments[0], agent_slug)
    return _normalize_stem(segments[-1])


def is_agent_slug_shape(value: str) -> bool:
    """Whether a name segment carries the REQ-002 agent-slug charset shape."""
    return _AGENT_SLUG_RE.fullmatch(value) is not None


def memory_scope_path(stem: str, agent_slug: str | None = None) -> str:
    """Return the scope's store path.

    Personal memories keep the flat shape ``/memories/<stem>.md`` while agent
    memories nest one validated segment: ``/memories/<agent_slug>/<stem>.md``
    (REQ-012).

    Args:
        stem: Normalized filename stem (from :func:`normalize_memory_name`).
        agent_slug: The validated agent slug, or ``None`` for the personal
            namespace.

    Returns:
        The store path relative to the composable-agents store filesystem.
    """
    if agent_slug:
        return f"{MEMORY_ROOT}{agent_slug}/{stem}.md"
    return f"{MEMORY_ROOT}{stem}.md"


class MemoryCredential(BaseModel):
    """Forwardable per-user credential for the Store File API.

    Attributes:
        method: ``"jwt"`` or ``"api_key"``; anything else fails closed.
        credential: The raw caller credential (JWT token or API key).
    """

    method: str
    credential: str


#: Allowed ``SavedMemory.action`` values.
MemoryAction = Literal["created", "appended", "replaced"]


class SavedMemory(BaseModel):
    """Result of a successful memory save (returned only after the upstream
    PUT succeeds).

    Attributes:
        path: Normalized store path ``/memories/{agent_slug?}/<stem>.md``.
        action: The operation actually performed upstream.
        chars: ``len(final_stored_content)`` in Python code points. This is
            informational only and never an output-size guard.
    """

    path: str
    action: MemoryAction
    chars: int


class MemoryListItem(BaseModel):
    """One entry of a scoped memories listing (REQ-013).

    Attributes:
        name: Logical memory name within the caller's scope (stem, without
            ``.md`` and without the agent segment).
        path: Full store path of the memory file.
    """

    name: str
    path: str


class MemoryContent(BaseModel):
    """Content of one scoped memory (``get_memory`` result, REQ-013)."""

    name: str
    path: str
    content: str


class MemoryDeleted(BaseModel):
    """Result of a successful scoped memory deletion (``delete_memory``)."""

    name: str
    path: str
