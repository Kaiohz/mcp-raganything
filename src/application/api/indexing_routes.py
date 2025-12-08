from fastapi import APIRouter, UploadFile, File, Depends
from application.use_cases.index_file_use_case import IndexFileUseCase
from application.use_cases.index_folder_use_case import IndexFolderUseCase
from application.requests.indexing_request import IndexFolderRequest
from dependencies import get_index_file_use_case, get_index_folder_use_case
from fastapi import status
import os
import shutil
import tempfile


indexing_router = APIRouter(tags=["Indexing"])


@indexing_router.post("/index", response_model=dict, status_code=status.HTTP_200_OK)
async def index_file(
    file: UploadFile = File(...),
    use_case: IndexFileUseCase = Depends(get_index_file_use_case),
):
    """
    Index a single file upload.

    Args:
        file: The uploaded file to index.
        use_case: The indexing use case dependency.

    Returns:
        dict: Status message indicating indexing result.
    """
    output_dir = os.path.join(tempfile.gettempdir(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, file.filename or "upload")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    result = await use_case.execute(
        file_path=file_path,
        filename=file.filename or "upload",
        output_dir=output_dir
    )
    return result


@indexing_router.post(
    "/index-folder", response_model=dict, status_code=status.HTTP_200_OK
)
async def index_folder(
    request: IndexFolderRequest,
    use_case: IndexFolderUseCase = Depends(get_index_folder_use_case),
):
    """
    Index all documents in a folder.

    Args:
        request: The indexing request containing folder path and parameters.
        use_case: The indexing use case dependency.

    Returns:
        dict: Indexing results and statistics.
    """
    output_dir = os.path.join(tempfile.gettempdir(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    result = await use_case.execute(request=request, output_dir=output_dir)
    return result

