import hashlib
import logging
import uuid
from typing import cast

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGEngine, PGVectorStore
from sqlalchemy import text

from domain.errors.messages import ErrorMessage
from domain.errors.vector_store import VectorStoreConfigError
from domain.logging.messages import LogMessage
from domain.ports.vector_store_port import SearchResult, VectorStorePort
from domain.ports.vector_table_manager import VectorTableManagerPort

logger = logging.getLogger(__name__)


class LangchainPgvectorAdapter(VectorStorePort, VectorTableManagerPort):
    """PGVector-backed vector store adapter using langchain-postgres.

    Manages per-working-dir vector tables with SHA-256-hashed names. Lazily
    initializes the PGEngine and PGVectorStore on first access for each
    working directory. Also implements the table-level lifecycle port
    (:class:`VectorTableManagerPort`) so a space's tracked
    ``classical_rag_*`` tables can be dropped by working_dir (scope cascade,
    REQ-010) — the naming rule stays encapsulated in this adapter.
    """

    def __init__(
        self,
        connection_string: str,
        table_prefix: str,
        embedding_dimension: int,
        embedding_service: Embeddings | None = None,
        engine: PGEngine | None = None,
    ):
        self._connection_string = connection_string
        self._table_prefix = table_prefix
        self._embedding_dimension = embedding_dimension
        self._embedding_service = embedding_service
        self._engine = engine
        self._stores = {}

    def _get_table_name(self, working_dir: str) -> str:
        hashed = hashlib.sha256(working_dir.encode()).hexdigest()[:16]
        return f"{self._table_prefix}{hashed}"

    async def ensure_table(self, working_dir: str) -> None:
        if self._engine is None:
            self._engine = PGEngine.from_connection_string(self._connection_string)

        table_name = self._get_table_name(working_dir)

        if working_dir not in self._stores:
            engine = self._engine
            if engine is None:
                raise VectorStoreConfigError(
                    ErrorMessage.VECTOR_STORE_NOT_INITIALIZED.format(
                        working_dir=working_dir
                    )
                )
            try:
                await engine.ainit_vectorstore_table(
                    table_name=table_name,
                    vector_size=self._embedding_dimension,
                )
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise
            # langchain-postgres requires ``Embeddings`` at this call; the
            # adapter historically forwards exactly what was configured (the
            # production wiring always provides embeddings — see
            # ``dependencies.py``). The cast documents that contract without
            # changing the historically forwarded value.
            store = await PGVectorStore.create(
                engine=engine,
                table_name=table_name,
                embedding_service=cast("Embeddings", self._embedding_service),
            )
            self._stores[working_dir] = store

    async def add_documents(
        self,
        working_dir: str,
        documents: list[tuple[str, str, dict[str, str | int | None]]],
    ) -> list[str]:
        store = self._stores.get(working_dir)
        if store is None:
            raise VectorStoreConfigError(
                ErrorMessage.VECTOR_STORE_NOT_INITIALIZED.format(
                    working_dir=working_dir
                )
            )

        lc_docs = []
        ids = []
        for content, file_path, metadata in documents:
            doc_id = str(uuid.uuid4())
            meta = {"file_path": file_path, "chunk_id": doc_id, **metadata}
            lc_docs.append(Document(id=doc_id, page_content=content, metadata=meta))
            ids.append(doc_id)

        await store.aadd_documents(lc_docs, ids=ids)
        return ids

    async def similarity_search(
        self,
        working_dir: str,
        query: str,
        top_k: int = 10,
        score_threshold: float | None = None,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        if working_dir not in self._stores:
            await self.ensure_table(working_dir)

        store = self._stores.get(working_dir)
        if store is None:
            raise VectorStoreConfigError(
                ErrorMessage.VECTOR_STORE_NOT_INITIALIZED.format(
                    working_dir=working_dir
                )
            )

        search_kwargs: dict = {"k": top_k}
        if metadata_filter is not None:
            search_kwargs["filter"] = metadata_filter
        lc_results = await store.asimilarity_search_with_score(query, **search_kwargs)

        filtered = lc_results
        if score_threshold is not None:
            filtered = [
                (doc, score) for doc, score in lc_results if score <= score_threshold
            ]

        return [
            SearchResult(
                chunk_id=str(doc.metadata.get("chunk_id", doc.id)),
                content=doc.page_content,
                file_path=doc.metadata.get("file_path", ""),
                score=float(score),
                metadata={
                    k: v
                    for k, v in doc.metadata.items()
                    if k not in ("chunk_id", "file_path")
                },
            )
            for doc, score in filtered
        ]

    async def delete_documents(self, working_dir: str, file_path: str) -> int:
        if working_dir not in self._stores:
            await self.ensure_table(working_dir)

        table_name = self._get_table_name(working_dir)
        stmt = text(
            f'DELETE FROM "{table_name}" '
            f"WHERE langchain_metadata->>'file_path' = :file_path"
        )
        logger.info(
            "Deleting vectors by file_path: table=%s file_path=%s",
            table_name,
            file_path,
        )

        engine = self._engine
        if engine is None:
            raise VectorStoreConfigError(
                ErrorMessage.VECTOR_STORE_NOT_INITIALIZED.format(
                    working_dir=working_dir
                )
            )

        # langchain-postgres PGEngine._pool and _run_as_async are private but
        # the only way to execute raw SQL. Pin langchain-postgres in pyproject.toml.
        async def _delete() -> int:
            async with engine._pool.connect() as conn:
                result = await conn.execute(stmt, {"file_path": file_path})
                await conn.commit()
                return result.rowcount

        count = await engine._run_as_async(_delete())
        logger.info(
            "Vectors deleted by file_path: table=%s count=%d", table_name, count
        )
        return count

    async def delete_by_prefix(self, working_dir: str, file_path_prefix: str) -> int:
        if working_dir not in self._stores:
            await self.ensure_table(working_dir)

        table_name = self._get_table_name(working_dir)
        stmt = text(
            f'DELETE FROM "{table_name}" '
            f"WHERE starts_with(langchain_metadata->>'file_path', :prefix)"
        )
        logger.info(
            "Deleting vectors by prefix: table=%s prefix=%s",
            table_name,
            file_path_prefix,
        )

        engine = self._engine
        if engine is None:
            raise VectorStoreConfigError(
                ErrorMessage.VECTOR_STORE_NOT_INITIALIZED.format(
                    working_dir=working_dir
                )
            )

        async def _delete() -> int:
            async with engine._pool.connect() as conn:
                result = await conn.execute(stmt, {"prefix": file_path_prefix})
                await conn.commit()
                return result.rowcount

        count = await engine._run_as_async(_delete())
        logger.info("Vectors deleted by prefix: table=%s count=%d", table_name, count)
        return count

    async def drop_table(self, working_dir: str) -> None:
        """Drop the vector table backing ``working_dir`` (idempotent).

        Uses the SAME SHA-256 naming rule as every other operation so a
        tracked working_dir always resolves to the exact table created for
        it. The in-memory store cache entry is dropped with the table.

        Args:
            working_dir: The tracked composed working_dir.

        Raises:
            VectorStoreConfigError: If no PGEngine could be resolved.
        """
        engine = self._engine
        if engine is None:
            if self._connection_string:
                engine = PGEngine.from_connection_string(self._connection_string)
                self._engine = engine
            else:
                raise VectorStoreConfigError(
                    ErrorMessage.VECTOR_STORE_NOT_INITIALIZED.format(
                        working_dir=working_dir
                    )
                )
        table_name = self._get_table_name(working_dir)
        stmt = text(f'DROP TABLE IF EXISTS "{table_name}"')
        logger.info(LogMessage.SPACE_TABLE_DROPPED, working_dir)

        async def _drop() -> None:
            async with engine._pool.connect() as conn:
                await conn.execute(stmt)
                await conn.commit()

        await engine._run_as_async(_drop())
        self._stores.pop(working_dir, None)

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.close()
            self._engine = None
