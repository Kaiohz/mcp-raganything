import os

import pytest

from domain.errors.classical import ClassicalConfigError
from domain.services.classical_helpers import (
    build_documents_from_chunks,
    validate_path,
)


class TestValidatePath:
    def test_simple_filename(self, tmp_path):
        # Arrange
        output_dir = str(tmp_path)

        # Act
        result = validate_path(output_dir, "report.pdf")

        # Assert
        assert result == os.path.join(output_dir, "report.pdf")

    def test_subdirectory_filename(self, tmp_path):
        # Arrange
        output_dir = str(tmp_path)

        # Act
        result = validate_path(output_dir, "subdir/report.pdf")

        # Assert
        assert result == os.path.join(output_dir, "subdir", "report.pdf")

    def test_path_traversal_raises(self, tmp_path):
        # Arrange
        output_dir = str(tmp_path)

        # Act & Assert
        with pytest.raises(ClassicalConfigError, match="escapes output directory"):
            validate_path(output_dir, "../../etc/passwd")

    def test_absolute_path_raises(self, tmp_path):
        # Arrange
        output_dir = str(tmp_path)

        # Act & Assert
        with pytest.raises(ClassicalConfigError, match="escapes output directory"):
            validate_path(output_dir, "/etc/passwd")

    def test_dotdot_in_middle_raises(self, tmp_path):
        # Arrange
        output_dir = str(tmp_path)

        # Act & Assert
        with pytest.raises(ClassicalConfigError, match="escapes output directory"):
            validate_path(output_dir, "docs/../../etc/shadow")


class TestBuildDocumentsFromChunks:
    def test_with_chunks(self):
        # Act
        documents = build_documents_from_chunks(
            ["chunk one text", "chunk two text"], "doc.pdf"
        )

        # Assert
        assert len(documents) == 2
        assert documents[0] == ("chunk one text", "doc.pdf", {"chunk_index": 0})
        assert documents[1] == ("chunk two text", "doc.pdf", {"chunk_index": 1})

    def test_with_no_chunks(self):
        # Act
        documents = build_documents_from_chunks([], "doc.pdf")

        # Assert
        assert documents == []
