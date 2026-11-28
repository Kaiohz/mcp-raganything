import os

from domain.errors.classical import ClassicalConfigError
from domain.errors.messages import ErrorMessage


def validate_path(output_dir: str, file_name: str) -> str:
    file_path = os.path.join(output_dir, file_name)
    real_output = os.path.realpath(output_dir)
    real_file = os.path.realpath(file_path)
    if not real_file.startswith(real_output + os.sep) and real_file != real_output:
        raise ClassicalConfigError(
            ErrorMessage.FILE_NAME_ESCAPES_OUTPUT_DIR.format(file_name=file_name)
        )
    return file_path


def build_documents_from_chunks(
    chunk_texts: list[str], file_name: str
) -> list[tuple[str, str, dict[str, int]]]:
    documents = []
    for i, chunk_text in enumerate(chunk_texts):
        documents.append((chunk_text, file_name, {"chunk_index": i}))
    return documents
