from pydantic import BaseModel, Field
from typing import Optional


class IndexFolderRequest(BaseModel):
    """
    Request model for indexing a folder of documents.
    """

    folder_path: str = Field(..., description="Absolute path to the folder")
    recursive: bool = Field(
        default=True, description="Process subdirectories recursively"
    )
    file_extensions: Optional[list[str]] = Field(
        default=None, description="List of file extensions to filter (e.g., ['.pdf', '.docx'])"
    )
    display_stats: bool = Field(
        default=True, description="Display processing statistics"
    )
    max_workers: Optional[int] = Field(
        default=None, description="Number of parallel workers for folder processing"
    )
