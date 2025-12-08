from pydantic import BaseModel, Field
from typing import Optional


class Document(BaseModel):
    """
    Represents a document entity for indexed files.
    """

    file_path: str = Field(..., description="Absolute path to the document file")
    filename: str = Field(..., description="Name of the document file")
    content_hash: Optional[str] = Field(
        default=None, description="Hash of the document content for change detection"
    )
    indexed_at: Optional[int] = Field(
        default=None, description="Timestamp when document was indexed"
    )
    status: Optional[str] = Field(
        default="pending", description="Indexing status (pending, indexed, failed)"
    )
