from typing import Optional, List
import time
from domain.ports.rag_engine import RAGEnginePort
from domain.entities.indexing_result import (
    FileIndexingResult,
    FolderIndexingResult,
    FolderIndexingStats,
    FileProcessingDetail,
    IndexingStatus,
)
from raganything import RAGAnything
from fastapi.logger import logger


class LightRAGAdapter(RAGEnginePort):
    """
    Adapter for RAGAnything/LightRAG implementing RAGEnginePort.
    Wraps the RAGAnything instance and provides a clean interface.
    
    This adapter uses RAGAnything's process_document_complete() for multimodal
    processing without doc_status registration (documents won't appear in Web UI).
    """

    def __init__(self, rag_instance: RAGAnything, max_workers: int) -> None:
        """
        Initialize the LightRAG adapter.

        Args:
            rag_instance: The configured RAGAnything instance.
            max_workers: Maximum number of concurrent workers.
        """
        self.rag = rag_instance
        self._initialized = False
        self.max_workers = max_workers

    async def initialize(self) -> bool:
        """
        Initialize the RAG engine.

        Returns:
            bool: True if initialization was successful.
        """
        try:
            if not self._initialized:
                await self.rag._ensure_lightrag_initialized()
                self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}", exc_info=True)
            return False

    async def index_document(
        self, file_path: str, file_name: str, output_dir: str
    ) -> FileIndexingResult:
        """
        Index a single document into the RAG system.

        Args:
            file_path: Absolute path to the document to index.
            file_name: Name of the file being indexed.
            output_dir: Directory for processing outputs.

        Returns:
            FileIndexingResult: Structured result of the indexing operation.
        """
        start_time = time.time()
        
        try:
            # Use RAGAnything's process_document_complete() for multimodal processing
            # Note: This does NOT register in doc_status table (no Web UI visibility)
            success = await self.rag.process_document_complete(
                file_path=file_path,
                output_dir=output_dir,
                parse_method="auto"
            )
            
            if success:
                processing_time_ms = (time.time() - start_time) * 1000
                logger.info(f"Successfully indexed document: {file_name}")
                return FileIndexingResult(
                    status=IndexingStatus.SUCCESS,
                    message=f"File '{file_name}' indexed successfully",
                    file_path=file_path,
                    file_name=file_name,
                    processing_time_ms=round(processing_time_ms, 2),
                )
            else:
                processing_time_ms = (time.time() - start_time) * 1000
                return FileIndexingResult(
                    status=IndexingStatus.FAILED,
                    message=f"Failed to index file '{file_name}'",
                    file_path=file_path,
                    file_name=file_name,
                    processing_time_ms=round(processing_time_ms, 2),
                    error="RAGAnything processing returned false",
                )
                
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            logger.error(f"Failed to index document {file_path}: {e}", exc_info=True)
            
            return FileIndexingResult(
                status=IndexingStatus.FAILED,
                message=f"Failed to index file '{file_name}'",
                file_path=file_path,
                file_name=file_name,
                processing_time_ms=round(processing_time_ms, 2),
                error=error_msg,
            )

    async def index_folder(
        self,
        folder_path: str,
        output_dir: str,
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None,
    ) -> FolderIndexingResult:
        """
        Index all documents in a folder.

        Args:
            folder_path: Absolute path to the folder containing documents.
            output_dir: Directory for processing outputs.
            recursive: Whether to process subdirectories recursively.
            file_extensions: Optional list of file extensions to filter.

        Returns:
            FolderIndexingResult: Structured result with statistics and file details.
        """
        import asyncio
        from pathlib import Path
        
        start_time = time.time()
        
        try:
            # Get all files in folder
            folder_path_obj = Path(folder_path)
            if not folder_path_obj.exists():
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            
            # Default extensions if not provided
            if file_extensions is None:
                file_extensions = [".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md"]
            
            # Collect files to process
            files_to_process = []
            for ext in file_extensions:
                pattern = f"**/*{ext}" if recursive else f"*{ext}"
                files_to_process.extend(folder_path_obj.glob(pattern))
            
            total_files = len(files_to_process)
            if total_files == 0:
                processing_time_ms = (time.time() - start_time) * 1000
                return FolderIndexingResult(
                    status=IndexingStatus.SUCCESS,
                    message=f"No files found to index in '{folder_path}'",
                    folder_path=folder_path,
                    recursive=recursive,
                    stats=FolderIndexingStats(total_files=0),
                    processing_time_ms=round(processing_time_ms, 2),
                )
            
            logger.info(f"Found {total_files} files to process in {folder_path}")
            
            # Process files with concurrency control
            semaphore = asyncio.Semaphore(self.max_workers)
            file_results: List[FileProcessingDetail] = []
            successful = 0
            failed = 0
            
            async def process_single_file(file_path: Path):
                nonlocal successful, failed
                async with semaphore:
                    file_name = file_path.name
                    try:
                        # Use process_document_complete() without doc_status registration
                        success = await self.rag.process_document_complete(
                            file_path=str(file_path),
                            output_dir=output_dir,
                            parse_method="auto"
                        )
                        
                        if success:
                            successful += 1
                            logger.info(f"Successfully indexed: {file_name}")
                            return FileProcessingDetail(
                                file_path=str(file_path),
                                file_name=file_name,
                                status=IndexingStatus.SUCCESS,
                            )
                        else:
                            failed += 1
                            return FileProcessingDetail(
                                file_path=str(file_path),
                                file_name=file_name,
                                status=IndexingStatus.FAILED,
                                error="RAGAnything processing returned false",
                            )
                    except Exception as e:
                        failed += 1
                        logger.error(f"Failed to index {file_name}: {e}")
                        return FileProcessingDetail(
                            file_path=str(file_path),
                            file_name=file_name,
                            status=IndexingStatus.FAILED,
                            error=str(e),
                        )
            
            # Create tasks for all files
            tasks = [process_single_file(file_path) for file_path in files_to_process]
            file_results = await asyncio.gather(*tasks)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Determine overall status
            stats = FolderIndexingStats(
                total_files=total_files,
                files_processed=successful,
                files_failed=failed,
                files_skipped=0,
            )
            
            if failed == 0 and successful > 0:
                status = IndexingStatus.SUCCESS
                message = f"Successfully indexed {successful} file(s) from '{folder_path}'"
            elif successful > 0 and failed > 0:
                status = IndexingStatus.PARTIAL
                message = f"Partially indexed folder '{folder_path}': {successful} succeeded, {failed} failed"
            elif successful == 0 and total_files > 0:
                status = IndexingStatus.FAILED
                message = f"Failed to index any files from '{folder_path}'"
            else:
                status = IndexingStatus.SUCCESS
                message = f"No files found to index in '{folder_path}'"
            
            return FolderIndexingResult(
                status=status,
                message=message,
                folder_path=folder_path,
                recursive=recursive,
                stats=stats,
                processing_time_ms=round(processing_time_ms, 2),
                file_results=list(file_results),
            )
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            logger.error(f"Failed to index folder {folder_path}: {e}", exc_info=True)
            return FolderIndexingResult(
                status=IndexingStatus.FAILED,
                message=f"Failed to index folder '{folder_path}'",
                folder_path=folder_path,
                recursive=recursive,
                stats=FolderIndexingStats(),
                processing_time_ms=round(processing_time_ms, 2),
                error=error_msg,
            )
