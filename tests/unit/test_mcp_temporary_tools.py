"""TDD (Red) tests for the new temporary-file MCP tools — spec
``chat-pdf-attachments`` §5.3 (REQ-029..035, REQ-047, EDG-015) and §9.6.

Two tools are added to ``/files/mcp`` (module ``application.api.mcp_file_tools``):

* ``temporary_upload_file(file_name, content_base64, content_type, prefix)``
  — REQ-030/031/032/033: strict base64, basename + traversal guards, the
  EXISTING MIME/extension rules, the 50 MiB limit (``FILES_MAX_UPLOAD_SIZE``
  env overridable with the 52428800 default), user isolation
  ``prefix == chat_tmp/<current_user_id>/...``, reusing the REAL
  ``UploadFileUseCase`` + existing RustFS storage, and returning the
  ``{"object_name", "size", "content_type"}`` JSON contract.
* ``temporary_delete_file(object_path)`` — REQ-034/035/REQ-047: only paths
  under ``chat_tmp/<current_user_id>/`` are accepted, ONLY the RustFS object
  is removed and the pgvector cascade is NEVER triggered (the vector store /
  ``DeleteFileUseCase`` must remain untouched); the delete is idempotent
  (EDG-015).

These tests follow the established Files-MCP tool-test convention
(``tests/unit/test_mcp_file_tools.py``): the REAL FastMCP tool function is
invoked; ONLY the outbound external adapter (the RustFS object storage) is
mocked via ``AsyncMock(spec=StoragePort)`` — the real ``UploadFileUseCase``
runs, and the cascade use-case getters are only SURROUNDING SPIES asserted
never to be called.
"""

import base64
from unittest.mock import AsyncMock

import pytest
from fastmcp.exceptions import ToolError

from application.api.mcp_file_tools import mcp_files
from application.api.temporary_path_policy import (
    sanitize_temporary_file_name,
    validate_temporary_object_path,
    validate_temporary_prefix,
)
from domain.errors.storage import StorageNotFoundError
from domain.ports.storage_port import StoragePort
from infrastructure.database.rls_context import current_user_id

PDF_BYTES = b"%PDF-1.7 temporary document payload"
PDF_B64 = base64.b64encode(PDF_BYTES).decode()

AUTHORIZED_PREFIX = "chat_tmp/user-a/thread-1/att-1/"


def _patch_storage(monkeypatch, storage_mock: AsyncMock) -> None:
    """Route the tools at ONLY the external storage adapter (RustFS).

    The production resolution is ``dependencies.get_upload_file_use_case()``
    which reads the ``minio_adapter`` module global; we swap that single
    external adapter so the REAL use cases keep running.
    """
    import application.api.mcp_file_tools as mcp_file_tools
    import dependencies as deps

    monkeypatch.setattr(deps, "minio_adapter", storage_mock, raising=False)
    monkeypatch.setattr(mcp_file_tools, "minio_adapter", storage_mock, raising=False)


@pytest.fixture
def storage_mock() -> AsyncMock:
    mock = AsyncMock(spec=StoragePort)
    mock.put_object.return_value = None
    mock.remove_object.return_value = None
    return mock


@pytest.fixture
def as_user_a(monkeypatch):
    token = current_user_id.set("user-a")
    yield
    current_user_id.reset(token)


@pytest.fixture(autouse=True)
def _no_pgvector_cascade(monkeypatch):
    """REQ-047: any accidental cascade/pgvector usage fails the test."""
    import application.api.mcp_file_tools as mcp_file_tools
    import dependencies as deps

    def _forbidden(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError(
            "The temporary-file tools must NEVER trigger the pgvector cascade "
            "(DeleteFileUseCase / vector store) — REQ-047."
        )

    monkeypatch.setattr(deps, "get_delete_file_use_case", _forbidden, raising=False)
    monkeypatch.setattr(deps, "get_delete_folder_use_case", _forbidden, raising=False)
    monkeypatch.setattr(deps, "get_vector_store_for_user", _forbidden, raising=False)
    monkeypatch.setattr(
        mcp_file_tools, "get_delete_file_use_case", _forbidden, raising=False
    )


async def _upload(**overrides) -> dict:
    from application.api.mcp_file_tools import temporary_upload_file

    payload = {
        "file_name": "rapport.pdf",
        "content_base64": PDF_B64,
        "content_type": "application/pdf",
        "prefix": AUTHORIZED_PREFIX,
    }
    payload.update(overrides)
    return await temporary_upload_file(**payload)


class TestTemporaryUploadFile:
    async def test_valid_pdf_uploads_via_real_use_case_and_returns_json_contract(
        self, storage_mock, as_user_a, monkeypatch
    ):
        """REQ-030..033 happy path: decoded bytes reach the REAL
        UploadFileUseCase → StoragePort; the tool returns the JSON contract."""
        _patch_storage(monkeypatch, storage_mock)

        result = await _upload()

        # The tool RETURNS the client-relative path; the STORAGE key is
        # composed under the caller's scope (write side == read side of the
        # agent-facing tools, scope-rag-memories REQ-003).
        expected_object = f"{AUTHORIZED_PREFIX}rapport.pdf"
        assert result["object_name"] == expected_object
        assert result["size"] == len(PDF_BYTES)
        assert result["content_type"] == "application/pdf"
        bucket = storage_mock.put_object.call_args.args[
            0
        ] or storage_mock.put_object.call_args.kwargs.get("bucket")
        assert bucket == "raganything", storage_mock.put_object.call_args
        scoped_key = f"user-a/personal/{expected_object}"
        assert storage_mock.put_object.call_args.args[1:4] == (
            scoped_key,
            PDF_BYTES,
            "application/pdf",
        )


class TestEnvMaxUploadSize:
    def test_default_max_size_is_52428800(self, monkeypatch):
        """REQ-092/093 (MCP side): FILES_MAX_UPLOAD_SIZE default 50 MiB."""
        from config import MinioConfig

        monkeypatch.delenv("FILES_MAX_UPLOAD_SIZE", raising=False)
        config = MinioConfig(_env_file=None)  # pyright: ignore[reportCallIssue]  # pydantic-settings: `_env_file` accepted at runtime, not exposed in the generated __init__ signature
        assert getattr(config, "FILES_MAX_UPLOAD_SIZE", None) == 52428800

    def test_env_override_honored(self, monkeypatch):
        from config import MinioConfig

        monkeypatch.setenv("FILES_MAX_UPLOAD_SIZE", "1024")
        config = MinioConfig(_env_file=None)  # pyright: ignore[reportCallIssue]  # pydantic-settings: `_env_file` accepted at runtime, not exposed in the generated __init__ signature
        assert getattr(config, "FILES_MAX_UPLOAD_SIZE", None) == 1024

    async def test_rejects_invalid_strict_base64(
        self, storage_mock, as_user_a, monkeypatch
    ):
        _patch_storage(monkeypatch, storage_mock)
        with pytest.raises(ToolError):
            await _upload(content_base64="not%valid!!base64@@")
        storage_mock.put_object.assert_not_called()

    async def test_rejects_disallowed_mime_or_extension(
        self, storage_mock, as_user_a, monkeypatch
    ):
        _patch_storage(monkeypatch, storage_mock)
        with pytest.raises(ToolError):
            await _upload(
                file_name="payload.exe", content_type="application/x-msdownload"
            )
        storage_mock.put_object.assert_not_called()

    async def test_rejects_oversize_payload(self, storage_mock, as_user_a, monkeypatch):
        """REQ-031: max 50 MiB (default) — oversize rejected."""
        _patch_storage(monkeypatch, storage_mock)
        big = base64.b64encode(b"\x00" * (51 * 1024 * 1024)).decode()
        with pytest.raises(ToolError, match="(?i)large|too|size"):
            await _upload(content_base64=big)
        storage_mock.put_object.assert_not_called()

    async def test_rejects_path_traversal_in_file_name_and_prefix(
        self, storage_mock, as_user_a, monkeypatch
    ):
        _patch_storage(monkeypatch, storage_mock)
        with pytest.raises(ToolError):
            await _upload(file_name="../../etc/passwd.pdf")
        with pytest.raises(ToolError):
            await _upload(prefix="chat_tmp/user-a/../../etc")
        storage_mock.put_object.assert_not_called()

    async def test_rejects_prefix_outside_own_user_namespace(
        self, storage_mock, as_user_a, monkeypatch
    ):
        """REQ-031 isolation: prefix MUST start with chat_tmp/<current_user_id>/."""
        _patch_storage(monkeypatch, storage_mock)
        for foreign_prefix in (
            "chat_tmp/user-b/thread-1/att-1/",
            "documents/shared/",
            "chat_tmp/",  # missing user segment
            "uploads/tmp/",
        ):
            with pytest.raises(ToolError):
                await _upload(prefix=foreign_prefix)
        storage_mock.put_object.assert_not_called()

    async def test_requires_authenticated_user(self, storage_mock, monkeypatch):
        # No current_user_id set — the tool cannot establish the authorised
        # prefix and must refuse rather than upload unscoped.
        _patch_storage(monkeypatch, storage_mock)
        token = current_user_id.set(None)
        try:
            with pytest.raises(ToolError):
                await _upload()
        finally:
            current_user_id.reset(token)
        storage_mock.put_object.assert_not_called()


class TestTemporaryDeleteFile:
    async def test_deletes_only_the_rustfs_object_under_authorized_prefix(
        self, storage_mock, as_user_a, monkeypatch
    ):
        """REQ-034/035: remove_object on the authorized path; no cascade."""
        _patch_storage(monkeypatch, storage_mock)
        from application.api.mcp_file_tools import temporary_delete_file

        object_path = "chat_tmp/user-a/thread-1/att-1/rapport.pdf"
        result = await temporary_delete_file(object_path)

        storage_mock.remove_object.assert_called_once()
        args, kwargs = storage_mock.remove_object.call_args
        # The REMOVE composes the caller's scope (the write side is scoped).
        expected_scoped = f"user-a/personal/{object_path}"
        assert args[1] == expected_scoped or kwargs.get("object_path") == expected_scoped
        assert "deleted" in str(result).lower() or bool(result)

    async def test_refuses_paths_outside_the_authorized_prefix(
        self, storage_mock, as_user_a, monkeypatch
    ):
        _patch_storage(monkeypatch, storage_mock)
        from application.api.mcp_file_tools import temporary_delete_file

        for evil_path in (
            "documents/shared/keep.pdf",
            "chat_tmp/user-b/thread-2/att-2/foreign.pdf",
            "../../etc/passwd",
            "/absolute/path.pdf",
            "chat_tmp/",
        ):
            with pytest.raises(ToolError):
                await temporary_delete_file(evil_path)
        storage_mock.remove_object.assert_not_called()

    async def test_delete_is_idempotent_on_already_deleted_object(
        self, storage_mock, as_user_a, monkeypatch
    ):
        """EDG-015: deleting an already-deleted object stays idempotent."""
        _patch_storage(monkeypatch, storage_mock)
        from application.api.mcp_file_tools import temporary_delete_file

        object_path = "chat_tmp/user-a/thread-1/att-1/rapport.pdf"
        await temporary_delete_file(object_path)
        await temporary_delete_file(object_path)  # second call must not raise
        assert storage_mock.remove_object.call_count == 2

    async def test_delete_of_an_absent_object_is_a_silent_no_op(
        self, storage_mock, as_user_a, monkeypatch
    ):
        """EDG-015: S3 DELETE is idempotent — an absent object is not an error."""
        _patch_storage(monkeypatch, storage_mock)
        from application.api.mcp_file_tools import temporary_delete_file

        storage_mock.remove_object.side_effect = StorageNotFoundError(
            "no such bucket/object"
        )
        result = await temporary_delete_file(
            "chat_tmp/user-a/thread-1/att-1/rapport.pdf"
        )
        assert result["deleted"] == "chat_tmp/user-a/thread-1/att-1/rapport.pdf"
        assert result["present"] is False

    async def test_delete_surfaces_a_real_storage_failure_for_retry(
        self, storage_mock, as_user_a, monkeypatch
    ):
        """A genuine storage error MUST surface so the caller retries (REQ-046).

        C2: converting a real failure into a false "deleted" success disables
        the mandated single retry of the composable-agents cleanup.
        """
        _patch_storage(monkeypatch, storage_mock)
        from application.api.mcp_file_tools import temporary_delete_file

        storage_mock.remove_object.side_effect = RuntimeError("AccessDenied")
        with pytest.raises(ToolError):
            await temporary_delete_file("chat_tmp/user-a/thread-1/att-1/rapport.pdf")


class TestNoPgvectorCascadeEver:
    async def test_upload_and_delete_leave_the_vector_store_untouched(
        self, storage_mock, as_user_a, monkeypatch
    ):
        """REQ-047 / REQ-048: no RAG indexing, no working_dir involvement —
        covered transactionally by the autouse ``_no_pgvector_cascade`` guard:
        any internal call to DeleteFileUseCase / vector store would raise an
        AssertionError and fail this test."""
        _patch_storage(monkeypatch, storage_mock)
        from application.api.mcp_file_tools import temporary_delete_file

        await _upload()
        await temporary_delete_file("chat_tmp/user-a/thread-1/att-1/rapport.pdf")
        assert mcp_files.name == "Files"  # same server instance, no re-registration


class TestTemporaryPathPolicyBranches:
    """C3: every branch of the three policy functions (CRAP ≤ 30 gate).

    Parametrized over the exact error paths the review uncovered
    (empty value, absolute path, residual ``..`` segment, folder-form
    normalisation, foreign/other roots, basename reduction, control
    characters).
    """

    USER = "user-a"
    TMP_ROOT = "chat_tmp"

    # --- validate_temporary_prefix ------------------------------------------------
    @pytest.mark.parametrize(
        "bad_prefix",
        [
            "",  # empty → required
            "   ",  # whitespace-only → required
            "/absolute/prefix/",  # absolute
            "..",  # pure traversal (normpath keeps "..")
            "../etc",  # traversal above the root
            "chat_tmp/user-a/../../etc",  # traversal inside an authorized shape
            "chat_tmp",  # root alone → no user segment
            "chat_tmp/",  # root with slash → no user segment
            "chat_tmp/user-b/thread/att",  # foreign user
            "documents/shared/",  # outside the temporary root
            "uploads/tmp/",  # other root
        ],
    )
    def test_prefix_rejected(self, bad_prefix):
        with pytest.raises(ValueError):
            validate_temporary_prefix(bad_prefix, self.USER)

    @pytest.mark.parametrize(
        "good_prefix,expected",
        [
            ("chat_tmp/user-a/thread-1/att-1/", "chat_tmp/user-a/thread-1/att-1/"),
            # a missing trailing slash on an authorized shape stays accepted
            ("chat_tmp/user-a/thread-1/att-1", "chat_tmp/user-a/thread-1/att-1"),
            # normpath strips the trailing slash → the folder form is restored
            ("chat_tmp/user-a/thread-1//att-1///", "chat_tmp/user-a/thread-1/att-1/"),
            ("chat_tmp/user-a/./t/a", "chat_tmp/user-a/t/a"),
        ],
    )
    def test_prefix_accepted_and_normalized(self, good_prefix, expected):
        assert validate_temporary_prefix(good_prefix, self.USER) == expected

    # --- validate_temporary_object_path -------------------------------------------
    @pytest.mark.parametrize(
        "bad_path",
        [
            "",  # empty → required
            "/absolute/path.pdf",  # absolute
            "..",  # pure traversal
            "../../etc/passwd",  # traversal above the root
            "chat_tmp/user-a/../../etc/passwd",  # traversal inside the shape
            "chat_tmp",  # root alone
            "chat_tmp/user-b/thread-2/att-2/foreign.pdf",  # foreign user
            "documents/shared/keep.pdf",  # outside the temporary root
        ],
    )
    def test_object_path_rejected(self, bad_path):
        with pytest.raises(ValueError):
            validate_temporary_object_path(bad_path, self.USER)

    @pytest.mark.parametrize(
        "good_path,expected",
        [
            ("chat_tmp/user-a/t/a/rapport.pdf", "chat_tmp/user-a/t/a/rapport.pdf"),
            (
                "chat_tmp/user-a/t//a/x.pdf",
                "chat_tmp/user-a/t/a/x.pdf",
            ),  # double slash collapse
        ],
    )
    def test_object_path_accepted_and_normalized(self, good_path, expected):
        assert validate_temporary_object_path(good_path, self.USER) == expected

    # --- sanitize_temporary_file_name ----------------------------------------------
    @pytest.mark.parametrize(
        "bad_name",
        [
            "",  # empty → required
            "   ",  # whitespace-only → required
            "../../etc/passwd.pdf",  # traversal
            "..\\evil.png",  # backslash exploit
            "/abs/path.png",  # absolute
            "..",  # reduces to a traversal token
            ".",  # reduces to the current dir
            "name\nwith-newline.pdf",  # control characters
        ],
    )
    def test_file_name_rejected(self, bad_name):
        with pytest.raises(ValueError):
            sanitize_temporary_file_name(bad_name)

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("rapport.pdf", "rapport.pdf"),
            ("some/dir/photo.png", "photo.png"),  # plain relative sub-path → basename
            ("report final v2.docx", "report final v2.docx"),
        ],
    )
    def test_file_name_accepted_and_reduced(self, name, expected):
        assert sanitize_temporary_file_name(name) == expected
