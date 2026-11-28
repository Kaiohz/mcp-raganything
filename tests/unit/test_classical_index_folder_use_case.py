from unittest.mock import AsyncMock

import pytest

from application.use_cases.classical_index_folder_use_case import (
    ClassicalIndexFolderUseCase,
)
from domain.entities.indexing_result import (
    FolderIndexingResult,
    IndexingStatus,
)


class TestClassicalIndexFolderUseCase:
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
    ) -> ClassicalIndexFolderUseCase:
        return ClassicalIndexFolderUseCase(
            vector_store=mock_vector_store,
            document_reader=mock_document_reader,
            storage=mock_storage,
            bucket="test-bucket",
            output_dir="/tmp/output",
        )

    async def test_execute_lists_objects_from_storage(
        self,
        use_case: ClassicalIndexFolderUseCase,
        mock_storage: AsyncMock,
    ) -> None:
        # Arrange
        mock_storage.list_objects.return_value = []

        # Act
        await use_case.execute(
            working_dir="project/docs",
            recursive=True,
        )

        # Assert
        mock_storage.list_objects.assert_called_once_with(
            "test-bucket", prefix="project/docs/", recursive=True
        )

    async def test_execute_ensures_vector_store_table(
        self,
        use_case: ClassicalIndexFolderUseCase,
        mock_vector_store: AsyncMock,
        mock_storage: AsyncMock,
    ) -> None:
        # Arrange
        mock_storage.list_objects.return_value = []

        # Act
        await use_case.execute(
            working_dir="project/docs",
            recursive=True,
        )

        # Assert
        mock_vector_store.ensure_table.assert_called_once_with("project/docs")

    async def test_execute_downloads_and_indexes_each_file(
        self,
        use_case: ClassicalIndexFolderUseCase,
        mock_storage: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        # Arrange
        mock_storage.list_objects.return_value = [
            "project/docs/report.pdf",
            "project/docs/notes.txt",
        ]

        # Act
        await use_case.execute(
            working_dir="project/docs",
            recursive=True,
        )

        # Assert
        assert mock_storage.get_object.call_count == 2
        mock_storage.get_object.assert_any_call(
            "test-bucket", "project/docs/report.pdf"
        )
        mock_storage.get_object.assert_any_call("test-bucket", "project/docs/notes.txt")

    async def test_execute_filters_by_file_extensions(
        self,
        use_case: ClassicalIndexFolderUseCase,
        mock_storage: AsyncMock,
    ) -> None:
        # Arrange
        mock_storage.list_objects.return_value = [
            "project/docs/report.pdf",
            "project/docs/notes.txt",
            "project/docs/data.xlsx",
        ]

        # Act
        await use_case.execute(
            working_dir="project/docs",
            file_extensions=[".pdf", ".txt"],
        )

        # Assert
        assert mock_storage.get_object.call_count == 2
        mock_storage.get_object.assert_any_call(
            "test-bucket", "project/docs/report.pdf"
        )
        mock_storage.get_object.assert_any_call("test-bucket", "project/docs/notes.txt")

    async def test_execute_returns_folder_indexing_result(
        self,
        use_case: ClassicalIndexFolderUseCase,
        mock_storage: AsyncMock,
    ) -> None:
        # Arrange
        mock_storage.list_objects.return_value = [
            "project/docs/report.pdf",
        ]

        # Act
        result = await use_case.execute(
            working_dir="project/docs",
            recursive=True,
        )

        # Assert
        assert isinstance(result, FolderIndexingResult)
        assert result.status == IndexingStatus.SUCCESS
        assert result.folder_path == "project/docs"

    async def test_execute_handles_empty_folder(
        self,
        use_case: ClassicalIndexFolderUseCase,
        mock_storage: AsyncMock,
    ) -> None:
        # Arrange
        mock_storage.list_objects.return_value = []

        # Act
        result = await use_case.execute(
            working_dir="project/empty",
            recursive=True,
        )

        # Assert
        assert isinstance(result, FolderIndexingResult)
        assert result.status == IndexingStatus.SUCCESS
        assert result.stats.total_files == 0

    async def test_execute_tracks_successful_and_failed_files(
        self,
        use_case: ClassicalIndexFolderUseCase,
        mock_storage: AsyncMock,
        mock_document_reader: AsyncMock,
    ) -> None:
        # Arrange
        mock_storage.list_objects.return_value = [
            "project/docs/good.pdf",
            "project/docs/bad.pdf",
        ]
        call_count = 0

        async def _extract_with_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Extraction failed")
            return ["Good content"]

        mock_document_reader.extract_chunks.side_effect = _extract_with_failure

        # Act
        result = await use_case.execute(
            working_dir="project/docs",
            recursive=True,
        )

        # Assert
        assert isinstance(result, FolderIndexingResult)
        assert result.stats.files_processed + result.stats.files_failed > 0

    async def test_execute_uses_custom_bucket(
        self,
        mock_vector_store: AsyncMock,
        mock_storage: AsyncMock,
        mock_document_reader: AsyncMock,
    ) -> None:
        # Arrange
        use_case = ClassicalIndexFolderUseCase(
            vector_store=mock_vector_store,
            document_reader=mock_document_reader,
            storage=mock_storage,
            bucket="custom-folder-bucket",
            output_dir="/custom/output",
        )

        mock_storage.list_objects.return_value = []

        # Act
        await use_case.execute(working_dir="project/docs")

        # Assert
        mock_storage.list_objects.assert_called_once_with(
            "custom-folder-bucket", prefix="project/docs/", recursive=True
        )

    async def test_execute_non_recursive_listing(
        self,
        use_case: ClassicalIndexFolderUseCase,
        mock_storage: AsyncMock,
    ) -> None:
        # Arrange
        mock_storage.list_objects.return_value = []

        # Act
        await use_case.execute(
            working_dir="project/docs",
            recursive=False,
        )

        # Assert
        mock_storage.list_objects.assert_called_once_with(
            "test-bucket", prefix="project/docs/", recursive=False
        )

    async def test_execute_passes_chunk_params(
        self,
        use_case: ClassicalIndexFolderUseCase,
        mock_storage: AsyncMock,
        mock_document_reader: AsyncMock,
    ) -> None:
        # Arrange
        mock_storage.list_objects.return_value = [
            "project/docs/report.pdf",
        ]

        # Act
        await use_case.execute(
            working_dir="project/docs",
            chunk_size=500,
            chunk_overlap=100,
        )

        # Assert
        mock_document_reader.extract_chunks.assert_called_once()
        call_kwargs = mock_document_reader.extract_chunks.call_args
        assert call_kwargs.kwargs["chunk_size"] == 500
        assert call_kwargs.kwargs["chunk_overlap"] == 100
