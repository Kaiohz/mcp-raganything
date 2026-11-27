"""Outbound port for the composable-agents Store File API (memory storage)."""

from abc import ABC, abstractmethod

from domain.entities.memory import MemoryCredential, MemoryListItem


class MemoryStorePort(ABC):
    """Boundary for per-user memory persistence through Store File endpoints.

    The port is addressed purely by the normalized memory name (stem); the
    caller scope travels as the optional ``agent_slug`` argument (REQ-012/014)
    and the implementation independently confines every outbound path to
    ``/memories/`` (exactly one optional agent nesting segment) while
    forwarding only the caller's own credential (JWT bearer or API key
    header, never both).
    """

    @abstractmethod
    async def read(
        self,
        name: str,
        credential: MemoryCredential,
        *,
        agent_slug: str | None = None,
    ) -> str | None:
        """Read the current content of a memory file.

        Only a genuine upstream 404 means the file is absent; every other
        failure raises.

        Args:
            name: Normalized memory name (stem, without ``.md``).
            credential: The authenticated caller's own credential.
            agent_slug: The caller's agent slug, or ``None`` for the flat
                personal namespace (REQ-012).

        Returns:
            The stored content verbatim, or ``None`` when the file is absent.

        Raises:
            MemoryValidationError: If the name fails independent confinement.
            MemoryUpstreamError: For timeouts, malformed responses and
                upstream 5xx errors.
            MemoryIdentityError: For upstream 401/403 rejections and
                unsupported credential methods.
        """

    @abstractmethod
    async def write(
        self,
        name: str,
        content: str,
        credential: MemoryCredential,
        *,
        agent_slug: str | None = None,
    ) -> None:
        """Persist the exact final content of a memory file.

        Args:
            name: Normalized memory name (stem, without ``.md``).
            content: The exact final content to store (never truncated).
            credential: The authenticated caller's own credential.
            agent_slug: The caller's agent slug, or ``None`` for the flat
                personal namespace (REQ-012).

        Raises:
            Same sanitization contract as :meth:`read`.
        """

    @abstractmethod
    async def list(
        self, credential: MemoryCredential, *, agent_slug: str | None = None
    ) -> list[MemoryListItem]:
        """List the memory files visible in the caller's scope (REQ-013/014).

        Personal listings return ONLY flat names inside ``/memories/``; agent
        listings return ONLY names inside ``/memories/<agent_slug>/`` (names
        relative to that scope) — one scope never merges into another.

        Args:
            credential: The authenticated caller's own credential.
            agent_slug: The caller's agent slug, or ``None`` for the flat
                personal namespace.

        Returns:
            The listing entries (names + store paths).

        Raises:
            Same failure contract as :meth:`read`.
        """

    @abstractmethod
    async def delete(
        self,
        name: str,
        credential: MemoryCredential,
        *,
        agent_slug: str | None = None,
    ) -> None:
        """Delete one scoped memory file (REQ-013/014).

        An absent file raises :class:`~domain.errors.memory_store.
        MemoryNotFoundError`; every other upstream failure raises per the
        :meth:`read` contract.

        Args:
            name: Normalized memory name (stem, without ``.md``).
            credential: The authenticated caller's own credential.
            agent_slug: The caller's agent slug, or ``None`` for the flat
                personal namespace.
        """

    @abstractmethod
    async def close(self) -> None:
        """Close any long-lived HTTP client. Must be idempotent."""
