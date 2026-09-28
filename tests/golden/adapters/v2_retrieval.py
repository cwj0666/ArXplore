import asyncio
import os
from collections.abc import Awaitable
from dataclasses import asdict, replace
from typing import Any, TypeVar

from arxplore.api.services.store import PostgresPaperStore
from arxplore.config import AppSettings, load_settings
from arxplore.db.pool import Pool, create_pool_from_settings
from arxplore.domain.retrieval import ContextChunk, RetrievalContext, RetrievalHit
from arxplore.embedding import EmbeddingClient
from arxplore.llm.paper_chat import retrieve_paper_chat_sources
from arxplore.retrieval.service import RetrievalConfig, Retriever

T = TypeVar("T")


def _score_breakdown(hit: RetrievalHit) -> dict[str, Any]:
    if hit.fusion is not None:
        breakdown: dict[str, Any] = {}
        for channel, fused in (("lexical", hit.fusion.lexical), ("vector", hit.fusion.vector)):
            if fused is None:
                continue
            breakdown[f"{channel}_rank"] = fused.rank
            breakdown[f"{channel}_normalized_score"] = fused.normalized_score
            breakdown[f"{channel}_score"] = fused.score
            breakdown[f"{channel}_weight"] = fused.weight
        return breakdown
    if hit.lexical is not None:
        scores = hit.lexical
        return {
            "fts_score": scores.fts_score,
            "ilike_bonus": scores.ilike_bonus,
            "content_role_adjustment": scores.content_role_adjustment,
            "section_boost": scores.section_boost,
            "structural_adjustment": scores.structural_adjustment,
            "coverage": scores.coverage,
            "strict_match": scores.strict_match,
            "section_intent_bonus": scores.section_intent_bonus,
        }
    if hit.vector is not None:
        return {
            "raw_similarity_score": hit.vector.raw_similarity_score,
            "content_role_adjustment": hit.vector.content_role_adjustment,
            "section_boost": hit.vector.section_boost,
            "rerank_adjustment": hit.vector.rerank_adjustment,
        }
    return {}


def hit_record(hit: RetrievalHit) -> dict[str, Any]:
    record: dict[str, Any] = {
        "chunk_id": hit.chunk_id,
        "arxiv_id": hit.arxiv_id,
        "chunk_index": hit.chunk_index,
        "paper_title": hit.paper_title,
        "paper_abstract": hit.paper_abstract,
        "chunk_text": hit.chunk_text,
        "section_title": hit.section_title,
        "content_role": hit.content_role,
        "score": hit.score,
        "similarity_score": hit.score,
        "retrieval_method": hit.method,
        "score_source": hit.method,
        "snippet": hit.snippet,
        "score_breakdown": _score_breakdown(hit),
    }
    if hit.vector is not None:
        record["raw_similarity_score"] = hit.vector.raw_similarity_score
        record["rerank_adjustment"] = hit.vector.rerank_adjustment
    if hit.fusion is not None:
        record["matched_methods"] = list(hit.fusion.matched_methods)
    return record


def chunk_record(chunk: ContextChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "arxiv_id": chunk.arxiv_id,
        "chunk_index": chunk.chunk_index,
        "chunk_text": chunk.chunk_text,
        "section_title": chunk.section_title,
        "content_role": chunk.content_role,
    }


def context_record(context: RetrievalContext) -> dict[str, Any]:
    return {
        **hit_record(context.hit),
        "context_chunks": [chunk_record(chunk) for chunk in context.chunks],
        "context_text": context.text,
    }


class V2RetrievalAdapter:
    def __init__(self) -> None:
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        if not base_url:
            raise RuntimeError("the v2 retrieval golden needs OPENAI_BASE_URL pointing at the fake OpenAI server")
        settings: AppSettings = load_settings()
        self._runner = asyncio.Runner()
        self._pool: Pool = create_pool_from_settings(settings, role="golden")
        self._run(self._pool.open(wait=True, timeout=self._pool.timeout))
        self._store = PostgresPaperStore(self._pool)
        self._embedder = EmbeddingClient.from_settings(settings, base_url=base_url)
        self._config = RetrievalConfig.from_settings(settings)

    def describe(self) -> dict[str, str]:
        return {"implementation": "v2"}

    def _run(self, awaitable: Awaitable[T]) -> T:
        async def wrapper() -> T:
            return await awaitable

        return self._runner.run(wrapper())

    def _retriever(self, retrieval_mode: str | None = None) -> Retriever:
        config = self._config
        if retrieval_mode is not None:
            config = replace(config, mode="hybrid" if retrieval_mode == "hybrid" else "lexical")
        return Retriever(self._pool, self._embedder, config)

    def eval_search(self, method: str, query: str, *, k: int, adjacency_window: int) -> list[dict[str, Any]]:
        retriever = self._retriever()
        search = {
            "lexical": retriever.search_lexical_contexts,
            "vector": retriever.search_vector_contexts,
            "hybrid": retriever.search_hybrid_contexts,
        }[method]
        contexts = self._run(search(query, limit=k, adjacency_window=adjacency_window))
        return [context_record(context) for context in contexts]

    def retrieve_contexts(
        self, query: str, *, limit: int, arxiv_id: str | None, retrieval_mode: str
    ) -> tuple[list[dict[str, Any]], str]:
        result = self._run(self._retriever(retrieval_mode).retrieve_contexts(query, limit=limit, arxiv_id=arxiv_id))
        return [context_record(context) for context in result.contexts], result.mode

    def paper_chat_sources(
        self, arxiv_id: str, question: str, *, retrieval_mode: str
    ) -> tuple[list[dict[str, Any]], str] | None:
        paper = self._run(self._store.get_paper(arxiv_id))
        if paper is None:
            return None
        retrieval = self._run(
            retrieve_paper_chat_sources(paper, question, retriever=self._retriever(retrieval_mode), chunks=self._store)
        )
        return [asdict(source) for source in retrieval.sources], retrieval.retrieval_mode

    def close(self) -> None:
        try:
            self._run(self._embedder.aclose())
            self._run(self._pool.close())
        finally:
            self._runner.close()


def retrieval_adapter() -> V2RetrievalAdapter:
    return V2RetrievalAdapter()
