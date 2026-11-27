"""Outbound port for dropping one classical-RAG vector table by working_dir.

Agent deletion cascades the space's tracked ``classical_rag_*`` tables
(REQ-010). The port keeps the table-naming rule (``classical_rag_
<sha256(working_dir)[:16]>``) encapsulated in the vector-store adapter — the
caller only ever passes the tracked working_dir.
"""

from abc import ABC, abstractmethod


class VectorTableManagerPort(ABC):
    """Boundary for table-level lifecycle operations on the vector store."""

    @abstractmethod
    async def drop_table(self, working_dir: str) -> None:
        """Drop the vector table backing ``working_dir`` (idempotent).

        Args:
            working_dir: The tracked composed working_dir whose table must be
                dropped.
        """
