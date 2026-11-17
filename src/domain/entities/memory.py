"""Memory domain entities and name normalization.

Pure Pydantic models plus the shared, defensive memory-name normalizer used
by BOTH the use case and the infrastructure adapter (independent confinement
to ``/memories/``: tool-level validation alone is insufficient).

No FastMCP, httpx or composable-agents knowledge is allowed here.
"""

from typing import Literal

from pydantic import BaseModel

from domain.errors.memory_store import MemoryValidationError
from domain.errors.messages import ErrorMessage

#: Canonical namespace root for user memory files.
MEMORY_ROOT = "/memories/"
#: Allowed logical name characters before the mandatory ``.md`` suffix.
_ALLOWED_NAME_PATTERN = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)
_ALLOWED_NAME_SET = frozenset(_ALLOWED_NAME_PATTERN)


def normalize_memory_name(name: str) -> str:
    """Normalize a logical memory name (e.g. ``preferences``/``x.md``).

    Rules:
    - Only letters, digits, dots, hyphens and underscores are allowed.
    - Exactly one trailing ``.md`` is stripped to form the filename stem.
    - The stem must be non-empty and may not be ``.`` or ``..`` (this rejects
      empty names, traversal, nested paths, slashes/backslashes, encoded
      traversal, control characters and absolute paths).

    Args:
        name: Caller-supplied logical memory name.

    Returns:
        The normalized filename stem (without ``.md``).

    Raises:
        MemoryValidationError: If the name does not satisfy the rules.
    """
    if not isinstance(name, str) or not set(name) <= _ALLOWED_NAME_SET:
        raise MemoryValidationError(ErrorMessage.MEMORY_NAME_INVALID)
    stem = name[: -len(".md")] if name.endswith(".md") else name
    if not stem or stem in (".", ".."):
        raise MemoryValidationError(ErrorMessage.MEMORY_NAME_INVALID)
    return stem


def memory_path(stem: str) -> str:
    """Return the normalized store path ``/memories/<stem>.md``."""
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
        path: Normalized store path ``/memories/<stem>.md``.
        action: The operation actually performed upstream.
        chars: ``len(final_stored_content)`` in Python code points. This is
            informational only and never an output-size guard.
    """

    path: str
    action: MemoryAction
    chars: int
