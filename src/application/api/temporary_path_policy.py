"""Shared temporary-file object-path policy for the MCP temporary tools
(chat-pdf-attachments spec §5.3, REQ-031/035/047 — amended by the user ×
agent directive, chat_tmp Option 2).

The temporary namespace is strictly scoped user × agent:

.. code-block:: text

    <user_id>/<agent_slug>/chat_tmp/<thread_id>/<attachment_id>/<filename>

The CLIENT-RELATIVE value the temporary tools accept carries NO user id
segment — the identity lives in the REQUIRED agent scope the backend
composes server-side (``X-Composable-Agent`` + the authenticated
principal), so the caller can never relocate another user's or another
agent's objects:

.. code-block:: text

    chat_tmp/<thread_id>/<attachment_id>/<filename>

Only relative paths matching ``chat_tmp/<thread_id>/<attachment_id>/...``
are accepted for upload prefixes / delete targets. Any traversal attempt,
absolute path, other root (``documents/``, ``uploads/``) or a foreign
``<user_id>`` segment (legacy shape) is refused.

Every predicate lives in its own tiny function (one rule, one decision) so the
validators cannot drift and each unit stays trivially testable — the detailed
contract notes are documented HERE, at module level, not per line.

Contract details:

* ``_normalize_relative`` — POSIX-normalises the value (backslashes become
  slashes, ``//``/``/./`` collapse, trailing slashes drop) and refuses empty
  values, absolute paths and residual ``..`` segments (``..`` survives
  ``normpath`` for shapes like ``a/../..``).
* ``_require_tmp_root`` — enforces ``chat_tmp/<thread_id>/<attachment_id>/…``
  with at least one object segment beneath it (delete targets): the bare
  root, root-with-slash, the legacy ``chat_tmp/<user_id>/…`` shape and any
  other root (``documents/``, ``uploads/``) are refused.
* ``validate_temporary_prefix`` — upload prefix: ``chat_tmp/<thread>/
  <attachment_id>/`` (two scoping segments + the trailing slash expected by
  :class:`UploadFileUseCase` is restored when the caller supplied one).
* ``validate_temporary_object_path`` — delete target: a full object key
  ``chat_tmp/<thread>/<attachment_id>/<file>`` (a file name segment is
  REQUIRED — no folder-form restoration, an object path always ends with a
  file name).
* ``sanitize_temporary_file_name`` — plain relative sub-paths
  (``some/dir/photo.png``) are REDUCED to the basename; traversal attempts
  (``..`` anywhere, a leading slash, an exploited backslash), control
  characters and names reducing to ``.``/``..`` are REFUSED — mirroring the
  request-side validation contract of composable-agents.
"""

import posixpath

TEMP_ROOT = "chat_tmp"

_ROOT_SCOPE_MESSAGE = "must stay inside {root}/<thread_id>/<attachment_id>/..."
_ISOLATION_MESSAGE = (
    "must start with {root}/<thread_id>/<attachment_id>/ "
    "(the caller identity is carried by the agent scope, never by the path)"
)


def _normalize_relative(value: str, label: str) -> str:
    """Return the POSIX-normalised relative form of ``value``."""
    if not value.strip():
        raise ValueError(f"{label} is required")
    normalized = posixpath.normpath(value.replace("\\", "/"))
    if normalized in ("", ".") or posixpath.isabs(normalized):
        raise ValueError(f"{label} {_ROOT_SCOPE_MESSAGE.format(root=TEMP_ROOT)}")
    if any(part == ".." for part in normalized.split("/")):
        raise ValueError(f"{label} {_ROOT_SCOPE_MESSAGE.format(root=TEMP_ROOT)}")
    return normalized


def _require_tmp_root(normalized: str, min_parts: int, label: str) -> None:
    """Refuse paths outside ``chat_tmp/<thread_id>/<attachment_id>/…``.

    The legacy ``chat_tmp/<user_id>/…`` shape is refused as well: the
    user identity is carried by the REQUIRED agent scope (chat_tmp
    Option 2), never by a client-controlled path segment.
    """
    parts = normalized.split("/")
    if len(parts) < min_parts or not parts[1] or not parts[2] or parts[0] != TEMP_ROOT:
        raise ValueError(f"{label} {_ISOLATION_MESSAGE.format(root=TEMP_ROOT)}")


def validate_temporary_prefix(prefix: str) -> str:
    """Validate a temporary upload prefix (REQ-031, scoped user × agent)."""
    normalized = _normalize_relative(prefix, "prefix")
    _require_tmp_root(normalized, 3, "prefix")
    if prefix.endswith("/") and not normalized.endswith("/"):
        normalized += "/"  # normpath strips the trailing slash — restore it
    return normalized


def validate_temporary_object_path(object_path: str) -> str:
    """Validate a temporary delete target (REQ-035/047, scoped user × agent)."""
    normalized = _normalize_relative(object_path, "object_path")
    _require_tmp_root(normalized, 4, "object_path")
    return normalized


def _has_traversal_token(value: str) -> bool:
    """Return True when the value carries a traversal/exploit sequence."""
    return ".." in value or "\\" in value or value.startswith("/")


def _refuse_unsafe_file_name(file_name: str) -> None:
    """Refuse traversal sequences and control characters in a file name."""
    if _has_traversal_token(file_name):
        raise ValueError(f"file_name must be a plain basename, got: {file_name!r}")
    if any(ord(char) < 32 for char in file_name):
        raise ValueError(
            f"file_name must not contain control characters, got: {file_name!r}"
        )


def sanitize_temporary_file_name(file_name: str) -> str:
    """Reduce ``file_name`` to a basename, refusing traversal attempts."""
    if not file_name or not file_name.strip():
        raise ValueError("file_name is required")
    _refuse_unsafe_file_name(file_name)
    basename = posixpath.basename(file_name)
    if not basename or basename in (".", ".."):
        raise ValueError(f"file_name must be a plain basename, got: {file_name!r}")
    return basename
