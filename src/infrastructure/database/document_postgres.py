from typing import Optional, List
from domain.ports.document_repo import DocumentRepoPort
from domain.entities.document import Document
from infrastructure.database.models.document import DocumentModel
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy import select, update
from fastapi.logger import logger


class DocumentPostgres(DocumentRepoPort):
    """
    PostgreSQL implementation of the DocumentRepoPort.
    Handles document metadata persistence using SQLAlchemy async ORM.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        """
        Initialize the DocumentPostgres repository.

        Args:
            engine: SQLAlchemy async engine.
        """
        self.engine = engine

    async def save_document(self, document: Document) -> int:
        """
        Save a document metadata to the repository.

        Args:
            document: The document entity to save.

        Returns:
            int: The ID of the saved document.
        """
        async with AsyncSession(self.engine) as session:
            try:
                # Check if document already exists
                result = await session.execute(
                    select(DocumentModel).where(
                        DocumentModel.file_path == document.file_path
                    )
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update existing document
                    existing.filename = document.filename
                    existing.content_hash = document.content_hash
                    existing.status = document.status
                    existing.indexed_at = document.indexed_at
                    await session.commit()
                    await session.refresh(existing)
                    return existing.id
                else:
                    # Create new document
                    doc_model = DocumentModel(
                        file_path=document.file_path,
                        filename=document.filename,
                        content_hash=document.content_hash,
                        indexed_at=document.indexed_at,
                        status=document.status,
                    )
                    session.add(doc_model)
                    await session.commit()
                    await session.refresh(doc_model)
                    return doc_model.id
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to save document: {e}", exc_info=True)
                raise
            finally:
                await session.close()

    async def get_document_by_path(self, file_path: str) -> Optional[Document]:
        """
        Retrieve a document by its file path.

        Args:
            file_path: The file path to search for.

        Returns:
            Optional[Document]: The document if found, None otherwise.
        """
        async with AsyncSession(self.engine) as session:
            try:
                result = await session.execute(
                    select(DocumentModel).where(DocumentModel.file_path == file_path)
                )
                doc_model = result.scalar_one_or_none()
                
                if doc_model:
                    return Document(
                        file_path=doc_model.file_path,
                        filename=doc_model.filename,
                        content_hash=doc_model.content_hash,
                        indexed_at=doc_model.indexed_at,
                        status=doc_model.status,
                    )
                return None
            except Exception as e:
                logger.error(f"Failed to get document: {e}", exc_info=True)
                return None
            finally:
                await session.close()

    async def list_documents(self, limit: int = 100) -> List[Document]:
        """
        List all documents in the repository.

        Args:
            limit: Maximum number of documents to return.

        Returns:
            List[Document]: List of document entities.
        """
        async with AsyncSession(self.engine) as session:
            try:
                result = await session.execute(
                    select(DocumentModel).limit(limit)
                )
                doc_models = result.scalars().all()
                
                return [
                    Document(
                        file_path=doc.file_path,
                        filename=doc.filename,
                        content_hash=doc.content_hash,
                        indexed_at=doc.indexed_at,
                        status=doc.status,
                    )
                    for doc in doc_models
                ]
            except Exception as e:
                logger.error(f"Failed to list documents: {e}", exc_info=True)
                return []
            finally:
                await session.close()

    async def update_document_status(
        self, file_path: str, status: str, indexed_at: int | None = None
    ) -> bool:
        """
        Update the status of a document.

        Args:
            file_path: The file path of the document to update.
            status: The new status value.
            indexed_at: Optional timestamp of when indexing completed.

        Returns:
            bool: True if update was successful, False otherwise.
        """
        async with AsyncSession(self.engine) as session:
            try:
                stmt = (
                    update(DocumentModel)
                    .where(DocumentModel.file_path == file_path)
                    .values(status=status)
                )
                
                if indexed_at is not None:
                    stmt = stmt.values(indexed_at=indexed_at)
                
                await session.execute(stmt)
                await session.commit()
                return True
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to update document status: {e}", exc_info=True)
                return False
            finally:
                await session.close()
