"""Outbound port for the composable-agents Store File API (memory storage)."""

from abc import ABC, abstractmethod

from domain.entities.memory import MemoryCredential


class MemoryStorePort(ABC):
    """Boundary for per-user memory persistence through Store File endpoints.

    The port is addressed purely by the normalized memory name; the
    implementation independently confines every outbound path to
    ``/memories/`` and forwards only the caller's own credential (JWT bearer
    or API key header, never both).
    """

    @abstractmethod
    async def read(self, name: str, credential: MemoryCredential) -> str | None:
        """Read the current content of a memory file.

        Only a genuine upstream 404 means the file is absent; every other
        failure raises.

        Args:
            name: Normalized memory name (stem, without ``.md``).
            credential: The authenticated caller's own credential.

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
        self, name: str, content: str, credential: MemoryCredential
    ) -> None:
        """Persist the exact final content of a memory file.

        Args:
            name: Normalized memory name (stem, without ``.md``).
            content: The exact final content to store (never truncated).
            credential: The authenticated caller's own credential.

        Raises:
            Same sanitization contract as :meth:`read`.
        """

    @abstractmethod
    async def close(self) -> None:
        """Close any long-lived HTTP client. Must be idempotent."""
