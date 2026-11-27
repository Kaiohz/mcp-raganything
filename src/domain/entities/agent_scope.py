"""The immutable caller scope ``(user_id, agent_slug)`` domain entity.

Pure domain composition (scope-rag-memories REQ-003/006/012): every scoped
storage key, classical-RAG working_dir and folder renaming of the feature is
composed by THIS entity, in one auditable place. Resolution/validation live
in :mod:`domain.services.scope` (the service that builds this entity from
the raw request values); nothing here reads ambient state.
"""

import posixpath

from pydantic import BaseModel, ConfigDict, Field

from domain.constants import PERSONAL_AGENT_SLUG


def _clean_relative(value: str) -> str:
    """Normalize a client-relative segment: strip slashes/edges, collapse ``.``."""
    normalized = posixpath.normpath(value.replace("\\", "/").strip("/"))
    if normalized in ("", "."):
        return ""
    return normalized


class AgentScope(BaseModel):
    """Immutable caller scope with the scoped-path composition helpers.

    Attributes:
        user_id: Authenticated principal id (empty in the legacy
            unauthenticated path — composition is skipped by callers then).
        agent_slug: Validated agent slug (``personal`` when no header).
    """

    model_config = ConfigDict(frozen=True)

    user_id: str = ""
    agent_slug: str = Field(default=PERSONAL_AGENT_SLUG)

    @property
    def root_prefix(self) -> str:
        """Return the scope's bucket root prefix ``<user_id>/<agent_slug>/``."""
        return f"{self.user_id}/{self.agent_slug}/"

    @property
    def is_personal(self) -> bool:
        """Whether this scope is the caller's ``personal`` space."""
        return self.agent_slug == PERSONAL_AGENT_SLUG

    def scoped_prefix(self, relative_prefix: str = "") -> str:
        """Compose a directory/listing prefix under the scope (trailing ``/``).

        The RELATIVE part must already be validated by the caller (the
        ``_validate_prefix`` traversal rules stay on the client-supplied
        path — the scope is applied after that check, REQ-003).

        Args:
            relative_prefix: Validated client-relative prefix (``""`` = the
                scope root).

        Returns:
            ``<user_id>/<agent_slug>/<relative>/`` (trailing slash always).
        """
        relative = _clean_relative(relative_prefix)
        if not relative:
            return self.root_prefix
        return f"{self.root_prefix}{relative}/"

    def scoped_object_path(self, relative_path: str) -> str:
        """Compose a full object key under the scope (no trailing slash).

        Pure prefix concatenation: the client path is treated as relative to
        the scope, so an absolute-looking key can never escape another user's
        or agent's namespace (REQ-003/REQ-005 — cross-user access impossible
        by construction).

        NOTE (intentional asymmetry with :meth:`scoped_prefix`): the object
        key is NOT ``..``-normalized — MinIO/RustFS keys are OPAQUE strings,
        so a ``..`` inside a client path composes to a harmless literal key
        that simply does not exist; stripping it instead would CHANGE which
        key is addressed and break the read-file contract. Do not "fix" this
        into a traversal strip.

        Args:
            relative_path: Client-supplied object path (empty → the scope
                root directory marker).

        Returns:
            The composed bucket key.
        """
        relative = relative_path.replace("\\", "/").lstrip("/")
        if not relative or relative == ".":
            return self.root_prefix
        return f"{self.root_prefix}{relative}"

    def scoped_working_dir(self, working_dir: str) -> str:
        """Compose the classical-RAG namespace (REQ-006).

        Preserves today's ``working_dir/`` suffix normalization: leading and
        trailing slashes are dropped from the client-supplied value and one
        trailing slash is appended; an empty value composes to the scope
        root namespace.

        Args:
            working_dir: Client-supplied project-level working_dir.

        Returns:
            ``<user_id>/<agent_slug>/<working_dir>/`` (or the scope root).
        """
        relative = _clean_relative(working_dir)
        if not relative:
            return self.root_prefix
        return f"{self.root_prefix}{relative}/"

    def relativize_folders(self, folders: list[str]) -> list[str]:
        """Rewrite listed folder prefixes relative to the scope root.

        A scoped folder listing must surface the caller's folders in the
        shape the caller requested them (``docs/`` — not the full bucket key
        ``<user_id>/<agent_slug>/docs/``), so any folder falling OUTSIDE the
        scope root (an impossible case) is passed through untouched.

        Args:
            folders: The raw folder prefixes returned by the storage port.

        Returns:
            The folder prefixes relative to this scope's root.
        """
        root = self.root_prefix
        return [
            value[len(root) :] if value.startswith(root) else value for value in folders
        ]
