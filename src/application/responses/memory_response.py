"""Response models for the ``save_memory`` MCP tool (REQ-011 contract).

Mirrors the domain :class:`~domain.entities.memory.SavedMemory` entity so the
thin tool layer maps it with a direct ``**model_dump()`` transformation.
``chars`` is informational only — it is never an output-size guard.
"""

from typing import Literal

from pydantic import BaseModel, Field


class SaveMemoryResponse(BaseModel):
    """Result of a successful memory save."""

    path: str = Field(description="Normalized memory path (/memories/<name>.md)")
    action: Literal["created", "appended", "replaced"] = Field(
        description="Operation actually performed upstream"
    )
    chars: int = Field(
        description="Length of the final stored content in Python code points"
    )
