"""MCP file tools for RAGAnything.

These tools are registered with FastMCP for Claude Desktop integration.

The Files server also exposes the two TECHNICAL tools consumed directly by
composable-agents (chat-pdf-attachments spec §5.3, REQ-029..035):

* ``temporary_upload_file`` — backend-driven temporary upload under the
  strictly user-isolated ``chat_tmp/<current_user_id>/...`` prefix;
* ``temporary_delete_file`` — backend-driven RustFS-only delete.

They are never exposed to the LLM: composable-agents' MCP tool loader filters
them out (REQ-036) while the normal ``list_files`` / ``list_folders`` /
``read_file`` tools keep flowing through (REQ-037). Deleting never triggers
the pgvector cascade and never touches the working_dir (REQ-035/REQ-047).
"""

import base64
import binascii
import logging
import posixpath
from dataclasses import asdict

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from application.api.file_routes import ALLOWED_EXTENSIONS, ALLOWED_MIME_PREFIXES
from application.api.temporary_path_policy import (
    sanitize_temporary_file_name,
    validate_temporary_object_path,
    validate_temporary_prefix,
)
from application.responses.file_response import FileContentResponse, FileInfoResponse
from dependencies import (
    get_list_files_use_case,
    get_list_folders_use_case,
    get_read_file_use_case,
    get_upload_file_use_case,
    minio_adapter,
    minio_config,
)
from domain.errors.storage import StorageNotFoundError
from domain.logging.messages import LogMessage
from infrastructure.database.rls_context import current_user_id

logger = logging.getLogger(__name__)

mcp_files = FastMCP("Files")

#: Max size of a temporary upload, from the env-configurable MinIO settings
#: (REQ-092 MCP side, default 50 MiB). Read at import time like every other
#: module-level settings singleton: FILES_MAX_UPLOAD_SIZE must be set in the
#: environment BEFORE the process boots (no per-call re-read).
TEMP_MAX_UPLOAD_SIZE = minio_config.FILES_MAX_UPLOAD_SIZE


def _require_user_id() -> str:
    """Return the authenticated user id or refuse (tool cannot scope safely)."""
    user_id = current_user_id.get()
    if not user_id:
        raise ToolError("temporary file tools require an authenticated user context")
    return user_id


def _decode_strict_base64(content_base64: str) -> bytes:
    """Decode a strict base64 payload (REQ-031).

    Raises:
        ToolError: When the payload is empty or not strictly decodable.
    """
    if not content_base64 or not content_base64.strip():
        raise ToolError("content_base64 is required")
    try:
        return base64.b64decode(content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ToolError("content_base64 must be strictly decodable base64") from exc


def _validate_temporary_file_type(filename: str, content_type: str) -> None:
    """Apply the EXISTING REST upload family rules to the temporary tool (REQ-031).

    Raises:
        ToolError: When the extension/MIME falls outside the allowed families.
    """
    ext = posixpath.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ToolError(
            f"file extension '{ext or filename}' is not allowed for temporary uploads"
        )
    if not any(content_type.startswith(prefix) for prefix in ALLOWED_MIME_PREFIXES):
        raise ToolError(
            f"content type '{content_type}' is not allowed for temporary uploads"
        )


@mcp_files.tool()
async def temporary_upload_file(
    file_name: str, content_base64: str, content_type: str, prefix: str
) -> dict:
    """Upload a TEMPORARY chat document (backend-driven, never LLM-exposed).

    Args:
        file_name: Attachment basename (plain single segment).
        content_base64: Strictly decodable base64 payload.
        content_type: Attachment MIME type.
        prefix: Destination prefix; MUST match chat_tmp/<current_user_id>/...

    Returns:
        The JSON contract: ``{"object_name", "size", "content_type"}`` (REQ-033).

    Raises:
        ToolError: On unauthenticated context, invalid base64, disallowed
            type, oversize payload, traversal or out-of-namespace prefix.
    """
    user_id = _require_user_id()
    try:
        safe_name = sanitize_temporary_file_name(file_name)
    except ValueError as exc:
        logger.warning(LogMessage.TEMP_UPLOAD_REJECTED, user_id, exc)
        raise ToolError(str(exc)) from None
    decoded = _decode_strict_base64(content_base64)
    _validate_temporary_file_type(safe_name, content_type)
    if len(decoded) > TEMP_MAX_UPLOAD_SIZE:
        max_mb = TEMP_MAX_UPLOAD_SIZE // (1024 * 1024)
        raise ToolError(
            f"temporary upload too large: {len(decoded)} bytes (maximum {max_mb} MB)"
        )
    try:
        normalized_prefix = validate_temporary_prefix(prefix, user_id)
    except ValueError as exc:
        logger.warning(LogMessage.TEMP_UPLOAD_REJECTED, user_id, exc)
        raise ToolError(str(exc)) from None

    # REQ-032: reuse the REAL UploadFileUseCase and the existing RustFS storage.
    use_case = get_upload_file_use_case()
    file_info = await use_case.execute(
        file_data=decoded,
        file_name=safe_name,
        prefix=normalized_prefix,
        content_type=content_type,
    )
    logger.info(LogMessage.TEMP_UPLOAD_COMPLETED, file_info.object_name, len(decoded))
    return {
        "object_name": file_info.object_name,
        "size": len(decoded),
        "content_type": content_type,
    }


@mcp_files.tool()
async def temporary_delete_file(object_path: str) -> dict:
    """Delete a TEMPORARY chat document from RustFS only (backend-driven).

    NEVER triggers the pgvector cascade and never touches the working_dir
    (REQ-035/REQ-047). Idempotent: an already-absent object is a logged no-op
    (EDG-015).

    Args:
        object_path: Path under ``chat_tmp/<current_user_id>/...``.

    Returns:
        ``{"deleted": object_path}`` on success or idempotent absence.

    Raises:
        ToolError: On unauthenticated context, traversal or any path outside
            the caller's own ``chat_tmp/<user_id>/`` namespace (REQ-047).
    """
    user_id = _require_user_id()
    try:
        normalized = validate_temporary_object_path(object_path, user_id)
    except ValueError as exc:
        logger.warning(LogMessage.TEMP_DELETE_REJECTED, user_id, exc)
        raise ToolError(str(exc)) from None

    # I2/C2 companion (REQ-035/REQ-047): plain StoragePort.remove_object — NOT
    # DeleteFileUseCase, NO pgvector cascade, NO working_dir involvement.
    # Absent (StorageNotFoundError) is an idempotent no-op (EDG-015); a REAL
    # storage failure MUST surface as a ToolError so the composable-agents
    # cleanup logs ERROR and retries exactly once (REQ-046).
    try:
        await minio_adapter.remove_object(minio_config.MINIO_BUCKET, normalized)
    except StorageNotFoundError:
        logger.info(LogMessage.TEMP_DELETE_ABSENT, normalized, "storage-not-found")
        return {"deleted": normalized, "present": False}
    except Exception as exc:  # noqa: BLE001 — a genuine failure reaches the caller
        logger.error(LogMessage.TEMP_DELETE_FAILED, normalized, exc)
        raise ToolError(
            f"temporary_delete_file failed for {normalized!r}: {exc}"
        ) from exc
    logger.info(LogMessage.TEMP_DELETE_COMPLETED, normalized)
    return {"deleted": normalized, "present": True}


def _validate_prefix(prefix: str) -> str:
    normalized = posixpath.normpath(prefix.replace("\\", "/"))
    if normalized == ".":
        normalized = ""
    if normalized.startswith("..") or posixpath.isabs(normalized):
        raise ToolError(
            f"prefix must be a relative path within the bucket, got: {prefix!r}"
        )
    if prefix.endswith("/") and not normalized.endswith("/"):
        normalized += "/"
    return normalized


@mcp_files.tool()
async def list_files(
    prefix: str = "", recursive: bool = True
) -> list[FileInfoResponse]:
    """List files in MinIO storage under a given prefix.

    Args:
        prefix: MinIO prefix/path to filter files by (e.g. 'documents/')
        recursive: Whether to list files in subdirectories (default True)

    Returns:
        List of file objects with object_name, size, and last_modified
    """
    prefix = _validate_prefix(prefix)
    use_case = get_list_files_use_case()
    files = await use_case.execute(prefix=prefix, recursive=recursive)
    return [FileInfoResponse(**asdict(f)) for f in files]


@mcp_files.tool()
async def list_folders(prefix: str = "") -> list[str]:
    """List folders in MinIO storage under a given prefix.

    Args:
        prefix: MinIO prefix/path to list folders under (e.g. 'documents/')

    Returns:
        List of folder prefix strings (e.g. ['docs/', 'photos/'])
    """
    prefix = _validate_prefix(prefix)
    use_case = get_list_folders_use_case()
    return await use_case.execute(prefix=prefix)


@mcp_files.tool()
async def read_file(file_path: str) -> FileContentResponse:
    """Read and extract text content from a file stored in MinIO.

    Supports 91 file formats including PDF, Office documents, images, HTML, etc.
    Uses Kreuzberg for document intelligence extraction.

    Args:
        file_path: Path to the file in MinIO bucket (e.g. 'documents/report.pdf')

    Returns:
        Extracted text content with metadata and any detected tables
    """
    use_case = get_read_file_use_case()
    try:
        result = await use_case.execute(file_path=file_path)
    except StorageNotFoundError:
        raise ToolError(f"File not found: {file_path}") from None
    except Exception:
        logger.exception(LogMessage.MCP_READ_FILE_UNEXPECTED_ERROR, file_path)
        raise ToolError(f"Failed to read file: {file_path}") from None
    return FileContentResponse(
        content=result.content,
        metadata=result.metadata,
        tables=result.tables,
    )
