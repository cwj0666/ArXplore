import asyncio
import logging
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Self

from openai import OpenAIError

from arxplore.config import AppSettings, RetrievalMode
from arxplore.context import RetrievalObservation, record_retrieval
from arxplore.db.pool import AnyPool
from arxplore.domain.retrieval import HybridInputs, RetrievalContext, RetrievalHit, RetrievalResult
from arxplore.retrieval.diversity import apply_paper_diversity
from arxplore.retrieval.fusion import fuse_hybrid_hits
from arxplore.retrieval.lexical import fetch_lexical_candidates, filter_lexical_hits, rerank_lexical_hits
from arxplore.retrieval.query import candidate_fetch_limit, hybrid_branch_limit, normalize_search_query
from arxplore.retrieval.vector import fetch_vector_candidates, rerank_vector_hits
from arxplore.retrieval.windows import DEFAULT_ADJACENCY_WINDOW, attach_contexts, fetch_chunk_windows

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_LIMIT = 5


class QueryEmbedder(Protocol):
    def is_available(self) -> bool: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class EmbedderMissingError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    mode: RetrievalMode = "hybrid"
    embedding_model: str | None = "text-embedding-3-large"
    vector_min_similarity: float = 0.0

    @classmethod
    def from_settings(cls, settings: AppSettings) -> Self:
        return cls(
            mode=settings.retrieval_mode,
            embedding_model=settings.openai_embedding_model,
            vector_min_similarity=settings.vector_min_similarity,
        )


def resolve_retrieval_mode(mode: RetrievalMode, embedder: QueryEmbedder | None) -> RetrievalMode:
    if mode != "hybrid" or embedder is None or not embedder.is_available():
        return "lexical"
    return "hybrid"


async def run_hybrid_channels[L, V](
    lexical: Coroutine[Any, Any, L],
    vector: Coroutine[Any, Any, V],
) -> tuple[L, V]:
    vector_task = asyncio.ensure_future(vector)
    lexical_task = asyncio.ensure_future(lexical)

    def stop_vector_after_lexical_failure(task: asyncio.Future[L]) -> None:
        if not task.cancelled() and task.exception() is not None:
            vector_task.cancel()

    lexical_task.add_done_callback(stop_vector_after_lexical_failure)
    lexical_result, vector_result = await asyncio.gather(lexical_task, vector_task, return_exceptions=True)
    if isinstance(lexical_result, BaseException):
        raise lexical_result
    if isinstance(vector_result, BaseException):
        raise vector_result
    return lexical_result, vector_result


class Retriever:
    def __init__(
        self,
        pool: AnyPool,
        embedder: QueryEmbedder | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.pool = pool
        self.embedder = embedder
        self.config = config or RetrievalConfig()

    def resolve_mode(self) -> RetrievalMode:
        return resolve_retrieval_mode(self.config.mode, self.embedder)

    async def search_lexical(
        self, query: str, *, limit: int = DEFAULT_SEARCH_LIMIT, arxiv_id: str | None = None
    ) -> list[RetrievalHit]:
        normalized_query = normalize_search_query(query)
        if not normalized_query:
            return []
        normalized_limit = max(1, limit)
        fetch_limit = normalized_limit if arxiv_id else candidate_fetch_limit(normalized_limit)
        hits = await fetch_lexical_candidates(self.pool, normalized_query, limit=fetch_limit, arxiv_id=arxiv_id)
        hits = rerank_lexical_hits(normalized_query, hits)
        hits = filter_lexical_hits(normalized_query, hits)
        return apply_paper_diversity(hits, limit=normalized_limit, arxiv_id=arxiv_id)

    async def search_vector(
        self, query: str, *, limit: int = DEFAULT_SEARCH_LIMIT, arxiv_id: str | None = None
    ) -> list[RetrievalHit]:
        normalized_query = normalize_search_query(query)
        if not normalized_query:
            return []
        normalized_limit = max(1, limit)
        fetch_limit = normalized_limit if arxiv_id else candidate_fetch_limit(normalized_limit)
        embedding = (await self._require_embedder().embed([normalized_query]))[0]
        hits = await fetch_vector_candidates(
            self.pool,
            embedding,
            query=normalized_query,
            limit=fetch_limit,
            arxiv_id=arxiv_id,
            model_name=self.config.embedding_model,
            min_similarity=self.config.vector_min_similarity,
        )
        hits = rerank_vector_hits(normalized_query, hits)
        return apply_paper_diversity(hits, limit=normalized_limit, arxiv_id=arxiv_id)

    async def hybrid_inputs(
        self,
        query: str,
        *,
        limit: int = DEFAULT_SEARCH_LIMIT,
        arxiv_id: str | None = None,
        lexical_limit: int | None = None,
        vector_limit: int | None = None,
    ) -> HybridInputs:
        normalized_query = normalize_search_query(query)
        if not normalized_query:
            return HybridInputs(query=normalized_query, lexical=(), vector=())
        branch_limit = hybrid_branch_limit(max(1, limit))
        lexical, vector = await run_hybrid_channels(
            self.search_lexical(normalized_query, limit=lexical_limit or branch_limit, arxiv_id=arxiv_id),
            self.search_vector(normalized_query, limit=vector_limit or branch_limit, arxiv_id=arxiv_id),
        )
        return HybridInputs(query=normalized_query, lexical=tuple(lexical), vector=tuple(vector))

    async def search_hybrid(
        self,
        query: str,
        *,
        limit: int = DEFAULT_SEARCH_LIMIT,
        arxiv_id: str | None = None,
        lexical_limit: int | None = None,
        vector_limit: int | None = None,
    ) -> list[RetrievalHit]:
        inputs = await self.hybrid_inputs(
            query, limit=limit, arxiv_id=arxiv_id, lexical_limit=lexical_limit, vector_limit=vector_limit
        )
        if not inputs.query:
            return []
        fused = fuse_hybrid_hits(inputs.lexical, inputs.vector)
        return apply_paper_diversity(fused, limit=max(1, limit), arxiv_id=arxiv_id)

    async def build_contexts(
        self, hits: Sequence[RetrievalHit], *, adjacency_window: int = DEFAULT_ADJACENCY_WINDOW
    ) -> list[RetrievalContext]:
        if not hits:
            return []
        centers = [(hit.arxiv_id, hit.chunk_index) for hit in hits]
        windows = await fetch_chunk_windows(self.pool, centers, window=max(0, adjacency_window))
        return attach_contexts(hits, windows)

    async def search_lexical_contexts(
        self,
        query: str,
        *,
        limit: int = DEFAULT_SEARCH_LIMIT,
        arxiv_id: str | None = None,
        adjacency_window: int = DEFAULT_ADJACENCY_WINDOW,
    ) -> list[RetrievalContext]:
        hits = await self.search_lexical(query, limit=limit, arxiv_id=arxiv_id)
        return await self.build_contexts(hits, adjacency_window=adjacency_window)

    async def search_vector_contexts(
        self,
        query: str,
        *,
        limit: int = DEFAULT_SEARCH_LIMIT,
        arxiv_id: str | None = None,
        adjacency_window: int = DEFAULT_ADJACENCY_WINDOW,
    ) -> list[RetrievalContext]:
        hits = await self.search_vector(query, limit=limit, arxiv_id=arxiv_id)
        return await self.build_contexts(hits, adjacency_window=adjacency_window)

    async def search_hybrid_contexts(
        self,
        query: str,
        *,
        limit: int = DEFAULT_SEARCH_LIMIT,
        arxiv_id: str | None = None,
        adjacency_window: int = DEFAULT_ADJACENCY_WINDOW,
        lexical_limit: int | None = None,
        vector_limit: int | None = None,
    ) -> list[RetrievalContext]:
        hits = await self.search_hybrid(
            query, limit=limit, arxiv_id=arxiv_id, lexical_limit=lexical_limit, vector_limit=vector_limit
        )
        return await self.build_contexts(hits, adjacency_window=adjacency_window)

    async def retrieve_contexts(
        self, query: str, *, limit: int = DEFAULT_SEARCH_LIMIT, arxiv_id: str | None = None
    ) -> RetrievalResult:
        result = None
        if self.resolve_mode() == "hybrid":
            try:
                contexts = await self.search_hybrid_contexts(query, limit=limit, arxiv_id=arxiv_id)
                result = RetrievalResult(contexts=tuple(contexts), mode="hybrid")
            except OpenAIError as exc:
                logger.warning("질의 임베딩 실패로 lexical 검색으로 전환합니다: %s", type(exc).__name__)
        if result is None:
            contexts = await self.search_lexical_contexts(query, limit=limit, arxiv_id=arxiv_id)
            result = RetrievalResult(contexts=tuple(contexts), mode="lexical")
        record_retrieval(RetrievalObservation(query=query, arxiv_id=arxiv_id, limit=limit, result=result))
        return result

    def _require_embedder(self) -> QueryEmbedder:
        if self.embedder is None:
            raise EmbedderMissingError("vector search needs a query embedder")
        return self.embedder
