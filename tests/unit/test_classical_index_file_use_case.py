from unittest.mock import AsyncMock

import pytest

from application.use_cases.classical_index_file_use_case import (
    ClassicalIndexFileUseCase,
)
from domain.entities.indexing_result import FileIndexingResult, IndexingStatus


class TestClassicalIndexFileUseCase:
    @pytest.fixture
    def mock_document_reader(self) -> AsyncMock:
        reader = AsyncMock()
        reader.extract_chunks.return_value = ["chunk text"]
        return reader

    @pytest.fixture
    def use_case(
        self,
        mock_vector_store: AsyncMock,
        mock_storage: AsyncMock,
        mock_document_reader: AsyncMock,
    ) -> ClassicalIndexFileUseCase:
        return ClassicalIndexFileUseCase(
            vector_store=mock_vector_store,
            document_reader=mock_document_reader,
            storage=mock_storage,
            bucket="test-bucket",
            output_dir="/tmp/output",
        )

    async def test_execute_downloads_file_from_storage(
        self,
        use_case: ClassicalIndexFileUseCase,
        mock_storage: AsyncMock,
    ) -> None:
        # Act
        await use_case.execute(
            file_name="reports/quarterly.pdf",
            working_dir="/tmp/rag/project_1",
        )

        # Assert
        mock_storage.get_object.assert_called_once_with(
            "test-bucket", "reports/quarterly.pdf"
        )

    async def test_execute_ensures_vector_store_table(
        self,
        use_case: ClassicalIndexFileUseCase,
        mock_vector_store: AsyncMock,
    ) -> None:
        # Act
        await use_case.execute(
            file_name="report.pdf",
            working_dir="/tmp/rag/project_1",
        )

        # Assert
        mock_vector_store.ensure_table.assert_called_once_with("/tmp/rag/project_1")

    async def test_execute_extracts_content_via_document_reader(
        self,
        use_case: ClassicalIndexFileUseCase,
        mock_document_reader: AsyncMock,
    ) -> None:
        # Act
        await use_case.execute(
            file_name="docs/report.pdf",
            working_dir="/tmp/rag/project_42",
        )
        await use_case.execute(
            file_name="report.pdf",
            working_dir="/tmp/rag/project_1",
            chunk_size=500,
            chunk_overlap=100,
        )

        # Assert — the reader is called once per file with the local temp path
        first, second = mock_document_reader.extract_chunks.call_args_list
        assert first.args[0].endswith("report.pdf")
        assert second.kwargs == {"chunk_size": 500, "chunk_overlap": 100}

    async def test_execute_adds_documents_to_vector_store(
        self,
        use_case: ClassicalIndexFileUseCase,
        mock_vector_store: AsyncMock,
    ) -> None:
        # Act
        await use_case.execute(
            file_name="report.pdf",
            working_dir="/tmp/rag/project_1",
        )

        # Assert
        mock_vector_store.add_documents.assert_called_once()
        call_kwargs = mock_vector_store.add_documents.call_args
        assert call_kwargs[1]["working_dir"] == "/tmp/rag/project_1"
        documents = call_kwargs[1]["documents"]
        assert isinstance(documents, list)
        assert documents == [("chunk text", "report.pdf", {"chunk_index": 0})]

    async def test_execute_uses_default_chunk_params(
        self,
        use_case: ClassicalIndexFileUseCase,
        mock_document_reader: AsyncMock,
    ) -> None:
        # Act
        await use_case.execute(
            file_name="report.pdf",
            working_dir="/tmp/rag/project_1",
        )

        # Assert
        mock_document_reader.extract_chunks.assert_called_once()
        call_args = mock_document_reader.extract_chunks.call_args
        assert call_args[1] == {"chunk_size": 1000, "chunk_overlap": 200}

    async def test_execute_returns_file_indexing_result_on_success(
        self,
        use_case: ClassicalIndexFileUseCase,
    ) -> None:
        # Act
        result = await use_case.execute(
            file_name="report.pdf",
            working_dir="/tmp/rag/project_1",
        )

        # Assert
        assert isinstance(result, FileIndexingResult)
        assert result.status == IndexingStatus.SUCCESS
        assert result.file_name == "report.pdf"

    async def test_execute_returns_failed_result_on_vector_store_error(
        self,
        use_case: ClassicalIndexFileUseCase,
        mock_vector_store: AsyncMock,
    ) -> None:
        # Arrange
        mock_vector_store.add_documents.side_effect = RuntimeError("Connection lost")

        # Act
        result = await use_case.execute(
            file_name="report.pdf",
            working_dir="/tmp/rag/project_1",
        )

        # Assert
        assert result.status == IndexingStatus.FAILED
        assert result.error is not None

    async def test_execute_returns_failed_result_on_extraction_error(
        self,
        use_case: ClassicalIndexFileUseCase,
        mock_document_reader: AsyncMock,
    ) -> None:
        # Arrange
        mock_document_reader.extract_chunks.side_effect = RuntimeError(
            "Unsupported format"
        )

        # Act
        result = await use_case.execute(
            file_name="corrupt.xyz",
            working_dir="/tmp/rag/project_1",
        )

        # Assert
        assert result.status == IndexingStatus.FAILED

    async def test_execute_returns_failed_result_on_storage_error(
        self,
        use_case: ClassicalIndexFileUseCase,
        mock_storage: AsyncMock,
    ) -> None:
        # Arrange
        mock_storage.get_object.side_effect = FileNotFoundError("File not found")

        # Act
        result = await use_case.execute(
            file_name="nonexistent.pdf",
            working_dir="/tmp/rag/project_1",
        )

        # Assert
        assert result.status == IndexingStatus.FAILED

    async def test_execute_adds_documents_with_file_path_metadata(
        self,
        use_case: ClassicalIndexFileUseCase,
        mock_vector_store: AsyncMock,
    ) -> None:
        # Act
        await use_case.execute(
            file_name="docs/report.pdf",
            working_dir="/tmp/rag/project_1",
        )

        # Assert
        call_kwargs = mock_vector_store.add_documents.call_args[1]
        documents = call_kwargs["documents"]
        for _content, file_path, _metadata in documents:
            assert file_path == "docs/report.pdf"

    async def test_execute_custom_bucket_and_output_dir(
        self,
        mock_vector_store: AsyncMock,
        mock_storage: AsyncMock,
        mock_document_reader: AsyncMock,
    ) -> None:
        # Arrange
        use_case = ClassicalIndexFileUseCase(
            vector_store=mock_vector_store,
            document_reader=mock_document_reader,
            storage=mock_storage,
            bucket="custom-bucket",
            output_dir="/custom/output",
        )

        mock_storage.get_object.side_effect = FileNotFoundError("skip extract")

        # Act
        await use_case.execute(
            file_name="test.pdf",
            working_dir="/tmp/rag/custom",
        )

        # Assert
        mock_storage.get_object.assert_called_once_with("custom-bucket", "test.pdf")

    async def test_execute_processes_empty_document(
        self,
        use_case: ClassicalIndexFileUseCase,
        mock_document_reader: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        # Arrange
        mock_document_reader.extract_chunks.return_value = []

        # Act
        result = await use_case.execute(
            file_name="empty.pdf",
            working_dir="/tmp/rag/project_1",
        )

        # Assert
        assert result.status == IndexingStatus.SUCCESS
        mock_vector_store.add_documents.assert_not_called()
