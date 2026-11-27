import asyncio
import json
import logging
import re
from collections.abc import Callable
from typing import Literal

from application.responses.classical_query_response import (
    ClassicalChunkResponse,
    ClassicalQueryResponse,
)
from config import ClassicalRAGConfig
from domain.logging.messages import LogMessage
from domain.ports.bm25_engine import BM25EnginePort
from domain.ports.llm_port import LLMPort
from domain.ports.vector_store_port import SearchResult, VectorStorePort
from domain.services.scope import resolve_agent_scope
from infrastructure.database.rls_context import (
    current_agent_scope,
    current_user_id,
)
from infrastructure.rag.rrf_combiner import RRFCombiner

logger = logging.getLogger(__name__)


class ClassicalQueryUseCase:
    """Classical RAG query with multi-query expansion, LLM judge, and optional hybrid BM25+vector search.

    Supports two modes:
    - **vector**: multi-query expansion → parallel similarity search → LLM judge scoring.
    - **hybrid**: same, but combines BM25 and vector results via Reciprocal Rank Fusion (RRF)
      before LLM judge scoring. Falls back to vector-only if BM25 is unavailable.

    The effective namespace is the caller's SCOPE composed in front of the
    client-supplied ``working_dir`` (``{user_id}/{agent_slug}/{working_dir}``,
    scope-rag-memories REQ-006) with the ``user_id`` metadata filter kept as
    defense-in-depth (REQ-007).
    """

    def __init__(
        self,
        vector_store: VectorStorePort,
        llm: LLMPort,
        config: ClassicalRAGConfig,
        bm25_engine: BM25EnginePort | None = None,
        rrf_k: int = 60,
        space_recorder: Callable[[str], None] | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.llm = llm
        self.config = config
        self.bm25_engine = bm25_engine
        self.rrf_combiner = RRFCombiner(k=rrf_k)
        self._space_recorder = space_recorder

    def _compose_scoped_working_dir(self, working_dir: str) -> str | None:
        """Compose the caller-scoped working_dir, or ``None`` in legacy mode.

        Synchronous by design (Sonar python:S7503): the composition and the
        scope validation behind it contain no I/O — they read in-memory
        request state and record the use synchronously — so awaiting a
        no-op coroutine would add protocol overhead without any async
        benefit. The composition happens BEFORE any vector/boundary
        operation, so an invalid ``X-Composable-Agent`` value rejects the
        query with zero searches (REQ-002).

        Args:
            working_dir: The client-supplied working_dir.

        Returns:
            The composed working_dir with the ``working_dir/`` suffix
            normalization preserved, or ``None`` when no authenticated user
            is wired (legacy unscoped path).

        Raises:
            AgentScopeInvalidError: On an invalid agent-scope header value.
        """
        user_id = current_user_id.get()
        if user_id is None:
            return None
        scope = resolve_agent_scope(user_id, current_agent_scope.get())
        composed = scope.scoped_working_dir(working_dir)
        if self._space_recorder is not None:
            self._space_recorder(composed)
        return composed

    @staticmethod
    def _extract_json_array(text: str) -> list[str]:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return [v for v in parsed if isinstance(v, str)]
            except ValueError:
                pass
        return []

    async def _generate_variations(self, query: str, num_variations: int) -> list[str]:
        try:
            system_prompt = "You are a helpful assistant. Generate alternative phrasings of the given query for improved search retrieval."
            user_message = f"Generate {num_variations} alternative versions of this query: {query}. Return ONLY a JSON array of strings."
            response = await self.llm.generate(
                system_prompt=system_prompt, user_message=user_message
            )
            return self._extract_json_array(response)
        except Exception:
            return []

    async def _score_chunk(self, query: str, chunk: SearchResult) -> float:
        try:
            judge_system = "You are a relevance judge. Score how relevant the given chunk is to the query on a scale of 0 to 10. Return ONLY a number."
            judge_user = f"Query: {query}\nChunk: {chunk.content}\nScore 0-10:"
            score_response = await self.llm.generate(
                system_prompt=judge_system, user_message=judge_user
            )
            nums = re.findall(r"[\d.]+", score_response.strip())
            return float(nums[0]) if nums else 0.0
        except (ValueError, TypeError, IndexError):
            return 0.0

    async def execute(
        self,
        working_dir: str,
        query: str,
        top_k: int = 10,
        num_variations: int | None = None,
        relevance_threshold: float | None = None,
        vector_distance_threshold: float | None = None,
        enable_llm_judge: bool = True,
        mode: Literal["vector", "hybrid"] = "vector",
    ) -> ClassicalQueryResponse:
        """Execute a classical RAG query.

        Args:
            working_dir: Vector store namespace to search within.
            query: The user's natural-language query.
            top_k: Maximum number of chunks to return.
            num_variations: Number of LLM-generated query rephrasings
                (defaults to ``CLASSICAL_NUM_QUERY_VARIATIONS``).
            relevance_threshold: Minimum LLM judge score (0-10) to keep a chunk
                (defaults to ``CLASSICAL_RELEVANCE_THRESHOLD``).
            vector_distance_threshold: Optional cosine distance cutoff for vector search.
            enable_llm_judge: If False, skip LLM scoring and rank by raw similarity.
            mode: ``"vector"`` for pure vector search, ``"hybrid"`` for BM25+vector RRF fusion.

        Returns:
            ClassicalQueryResponse with ranked chunks and query provenance.
        """
        composed = self._compose_scoped_working_dir(working_dir)
        if composed is not None:
            working_dir = composed

        working_dir = working_dir if working_dir.endswith("/") else f"{working_dir}/"

        # Per-user RAG isolation: when the RLS contextvar is set, filter the
        # vector search by ``user_id`` metadata so a user only retrieves their
        # own chunks. When None (legacy/tests, no auth wired), no filter is
        # applied — the legacy behaviour is preserved.
        user_id = current_user_id.get()
        metadata_filter: dict[str, str] | None = (
            {"user_id": user_id} if user_id is not None else None
        )

        if num_variations is None:
            num_variations = self.config.CLASSICAL_NUM_QUERY_VARIATIONS
        if relevance_threshold is None:
            relevance_threshold = self.config.CLASSICAL_RELEVANCE_THRESHOLD

        if mode == "hybrid" and self.bm25_engine is not None:
            return await self._execute_hybrid(
                working_dir=working_dir,
                query=query,
                top_k=top_k,
                num_variations=num_variations,
                relevance_threshold=relevance_threshold,
                vector_distance_threshold=vector_distance_threshold,
                enable_llm_judge=enable_llm_judge,
                metadata_filter=metadata_filter,
            )

        if mode == "hybrid" and self.bm25_engine is None:
            logger.warning(LogMessage.CLASSICAL_BM25_UNAVAILABLE_FALLBACK)

        return await self._execute_vector(
            working_dir=working_dir,
            query=query,
            top_k=top_k,
            num_variations=num_variations,
            relevance_threshold=relevance_threshold,
            vector_distance_threshold=vector_distance_threshold,
            enable_llm_judge=enable_llm_judge,
            metadata_filter=metadata_filter,
        )

    async def _execute_vector(
        self,
        working_dir: str,
        query: str,
        top_k: int,
        num_variations: int,
        relevance_threshold: float,
        vector_distance_threshold: float | None,
        enable_llm_judge: bool,
        metadata_filter: dict[str, str] | None = None,
    ) -> ClassicalQueryResponse:
        variations = (
            await self._generate_variations(query, num_variations)
            if num_variations > 1
            else []
        )

        queries = [query] + variations

        search_tasks = [
            self.vector_store.similarity_search(
                working_dir=working_dir,
                query=q,
                top_k=top_k,
                score_threshold=vector_distance_threshold,
                metadata_filter=metadata_filter,
            )
            for q in queries
        ]
        search_results = await asyncio.gather(*search_tasks)

        all_results: dict[str, SearchResult] = self._dedupe_search_results(
            search_results
        )

        scored_chunks = await self._score_and_filter(
            query, all_results, relevance_threshold, enable_llm_judge
        )

        return ClassicalQueryResponse(
            status="success",
            message="",
            queries=queries,
            chunks=scored_chunks,
            mode="vector",
        )

    @staticmethod
    def _dedupe_search_results(
        search_results: list[list[SearchResult]],
    ) -> dict[str, SearchResult]:
        """Collapse the multi-query fan-out into one entry per chunk.

        The same underlying chunk surfaces once per query variation: in
        pgvector the chunk has ONE stable id (the ``chunk_id`` dedupe), and a
        result re-hit is additionally recognised by its
        ``(file_path, content)`` signature — a corpus re-indexed variation
        never duplicates a row in the response.
        """
        all_results: dict[str, SearchResult] = {}
        seen_chunks: set[str] = set()
        seen_equivalents: dict[tuple[str, str], str] = {}
        for results in search_results:
            for r in results:
                if not r.chunk_id or r.chunk_id in seen_chunks:
                    continue
                signature = (r.file_path, r.content)
                canonical = seen_equivalents.get(signature)
                if canonical is not None and canonical in all_results:
                    continue  # the same chunk surfaced by another variation
                seen_equivalents.setdefault(signature, r.chunk_id)
                seen_chunks.add(r.chunk_id)
                all_results[r.chunk_id] = r
        return all_results

    async def _execute_hybrid(
        self,
        working_dir: str,
        query: str,
        top_k: int,
        num_variations: int,
        relevance_threshold: float,
        vector_distance_threshold: float | None,
        enable_llm_judge: bool,
        metadata_filter: dict[str, str] | None = None,
    ) -> ClassicalQueryResponse:
        variations = (
            await self._generate_variations(query, num_variations)
            if num_variations > 1
            else []
        )
        queries = [query] + variations

        bm25_engine = self.bm25_engine
        if bm25_engine is None:
            # Defensive: hybrid mode is only entered with BM25 wired (see the
            # caller guard); fall back to vector mode to keep behavior.
            logger.warning(LogMessage.CLASSICAL_BM25_UNAVAILABLE_FALLBACK)
            return await self._execute_vector(
                working_dir=working_dir,
                query=query,
                top_k=top_k,
                num_variations=num_variations,
                relevance_threshold=relevance_threshold,
                vector_distance_threshold=vector_distance_threshold,
                enable_llm_judge=enable_llm_judge,
                metadata_filter=metadata_filter,
            )

        search_tasks = [
            self.vector_store.similarity_search(
                working_dir=working_dir,
                query=q,
                top_k=top_k,
                score_threshold=vector_distance_threshold,
                metadata_filter=metadata_filter,
            )
            for q in queries
        ]

        bm25_coro = bm25_engine.search(query, working_dir, top_k=top_k * 2)
        vector_coro = asyncio.gather(*search_tasks)

        gather_results = await asyncio.gather(
            bm25_coro, vector_coro, return_exceptions=True
        )

        bm25_result = gather_results[0]
        vector_result = gather_results[1]

        bm25_hits = self._extract_bm25_hits(bm25_result)
        vector_hits = self._dedupe_vector_hits(vector_result)

        combined = self.rrf_combiner.combine_classical(
            bm25_results=bm25_hits,
            vector_results=vector_hits,
            top_k=top_k,
        )

        combined_map: dict[str, SearchResult] = {}
        hybrid_meta: dict[str, dict] = {}
        for r in combined:
            sr = SearchResult(
                chunk_id=r.chunk_id,
                content=r.content,
                file_path=r.file_path,
                score=r.combined_score,
                metadata=r.metadata,
            )
            combined_map[r.chunk_id] = sr
            hybrid_meta[r.chunk_id] = {
                "bm25_score": r.bm25_score,
                "vector_score": r.vector_score,
                "combined_score": r.combined_score,
            }

        scored_chunks = await self._score_and_filter(
            query, combined_map, relevance_threshold, enable_llm_judge, hybrid_meta
        )

        return ClassicalQueryResponse(
            status="success",
            message="",
            queries=queries,
            chunks=scored_chunks,
            mode="hybrid",
        )

    def _extract_bm25_hits(self, bm25_result: object) -> list:
        if isinstance(bm25_result, Exception):
            logger.error(LogMessage.CLASSICAL_BM25_SEARCH_FAILED_HYBRID, bm25_result)
            return []
        if isinstance(bm25_result, list):
            return bm25_result
        return []

    @staticmethod
    def _dedupe_vector_hits(vector_result: object) -> list[SearchResult]:
        if isinstance(vector_result, Exception) or not isinstance(vector_result, list):
            logger.error(
                LogMessage.CLASSICAL_VECTOR_SEARCH_FAILED_HYBRID, vector_result
            )
            return []
        all_results = ClassicalQueryUseCase._dedupe_search_results(vector_result)
        return list(all_results.values())

    async def _score_and_filter(
        self,
        query: str,
        all_results: dict[str, SearchResult],
        relevance_threshold: float,
        enable_llm_judge: bool,
        hybrid_meta: dict[str, dict] | None = None,
    ) -> list[ClassicalChunkResponse]:
        if enable_llm_judge:
            sem = asyncio.Semaphore(5)

            async def _bounded_score(c: SearchResult) -> tuple[SearchResult, float]:
                async with sem:
                    score = await self._score_chunk(query, c)
                    return (c, score)

            scored = await asyncio.gather(
                *(_bounded_score(c) for c in all_results.values())
            )

            scored_chunks = sorted(
                [
                    self._build_chunk_response(
                        chunk=chunk,
                        relevance_score=score,
                        hybrid_meta=hybrid_meta,
                    )
                    for chunk, score in scored
                    if score >= relevance_threshold
                ],
                key=lambda c: c.relevance_score,
                reverse=True,
            )
        else:
            if hybrid_meta:
                relevance_base = [
                    self._build_chunk_response(
                        chunk=chunk,
                        relevance_score=round(chunk.score, 4),
                        hybrid_meta=hybrid_meta,
                    )
                    for chunk in all_results.values()
                ]
            else:
                relevance_base = [
                    self._build_chunk_response(
                        chunk=chunk,
                        relevance_score=round(1.0 - chunk.score, 4),
                        hybrid_meta=None,
                    )
                    for chunk in all_results.values()
                ]
            scored_chunks = sorted(
                relevance_base,
                key=lambda c: c.relevance_score,
                reverse=True,
            )

        return scored_chunks

    @staticmethod
    def _build_chunk_response(
        chunk: SearchResult,
        relevance_score: float,
        hybrid_meta: dict[str, dict] | None = None,
    ) -> ClassicalChunkResponse:
        meta = hybrid_meta.get(chunk.chunk_id, {}) if hybrid_meta else {}
        return ClassicalChunkResponse(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            file_path=chunk.file_path,
            relevance_score=relevance_score,
            metadata=chunk.metadata,
            bm25_score=meta.get("bm25_score"),
            vector_score=meta.get("vector_score"),
            combined_score=meta.get("combined_score"),
        )
