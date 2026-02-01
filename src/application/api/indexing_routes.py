from fastapi import APIRouter, UploadFile, File, Depends, BackgroundTasks, status
from typing import List
from application.use_cases.index_file_use_case import IndexFileUseCase
from application.use_cases.index_folder_use_case import IndexFolderUseCase
from application.use_cases.index_batch_use_case import IndexBatchUseCase
from application.requests.indexing_request import IndexFolderRequest
from dependencies import (
    get_index_file_use_case,
    get_index_folder_use_case,
    get_index_batch_use_case,
    OUTPUT_DIR,
)
import shutil
import os


indexing_router = APIRouter(tags=["Multimodal Indexing"])


@indexing_router.post(
    "/file/index", response_model=dict, status_code=status.HTTP_202_ACCEPTED
)
async def index_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    use_case: IndexFileUseCase = Depends(get_index_file_use_case),
):
    """
    Index a single file upload in the background.

    Args:
        background_tasks: FastAPI background tasks handler.
        file: The uploaded file to index.
        use_case: The indexing use case dependency.

    Returns:
        dict: Status message indicating indexing started.
    """
    file_name = file.filename or "upload"
    file_path = os.path.join(OUTPUT_DIR, file_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(
        use_case.execute,
        file_path=file_path,
        file_name=file_name,
    )

    return {"status": "accepted", "message": "File indexing started in background"}


@indexing_router.post(
    "/folder/index", response_model=dict, status_code=status.HTTP_202_ACCEPTED
)
async def index_folder(
    request: IndexFolderRequest,
    background_tasks: BackgroundTasks,
    use_case: IndexFolderUseCase = Depends(get_index_folder_use_case),
):
    """
    Index all documents in a folder in the background.

    Args:
        request: The indexing request containing folder path and parameters.
        background_tasks: FastAPI background tasks handler.
        use_case: The indexing use case dependency.

    Returns:
        dict: Status message indicating indexing started.
    """
    background_tasks.add_task(
        use_case.execute,
        request=request,
    )

    return {"status": "accepted", "message": "Folder indexing started in background"}


@indexing_router.post(
    "/batch/index", response_model=dict, status_code=status.HTTP_202_ACCEPTED
)
async def index_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    use_case: IndexBatchUseCase = Depends(get_index_batch_use_case),
):
    """
    Index multiple files in batch (background processing).

    This endpoint accepts multiple file uploads and processes them in parallel
    according to the MAX_WORKERS configuration. All files are saved to a temporary
    staging directory, indexed together, and the staging directory is cleaned up automatically.

    Args:
        background_tasks: FastAPI background tasks handler.
        files: List of files to index (supports any document format).
        use_case: The batch indexing use case dependency.

    Returns:
        dict: Status message indicating batch indexing started with file count.

    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/batch/index" \
          -F "files=@document1.pdf" \
          -F "files=@document2.docx" \
          -F "files=@presentation.pptx"
        ```
    """
    if not files:
        return {
            "status": "error",
            "message": "No files provided",
            "files_count": 0,
        }

    background_tasks.add_task(
        use_case.execute,
        files=files,
    )

    return {
        "status": "accepted",
        "message": f"Batch indexing started for {len(files)} file(s)",
        "files_count": len(files),
        "files": [f.filename for f in files if f.filename],
    }
