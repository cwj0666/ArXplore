from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from openai import OpenAIError

from arxplore.config import RetrievalMode
from arxplore.domain.retrieval import HybridInputs, RetrievalContext, RetrievalHit
from arxplore.retrieval.query import normalize_search_query
from arxplore.retrieval.service import RetrievalConfig, Retriever
from tests.unit.eval.conftest import build_context, build_hit


class StubEmbedder:
    def is_available(self) -> bool:
        return True

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


@dataclass
class ChannelData:
    lexical: list[RetrievalHit] = field(default_factory=list)
    vector: list[RetrievalHit] = field(default_factory=list)


def lexical_hit(chunk_id: int, score: float, arxiv_id: str, *, strict: bool = True) -> RetrievalHit:
    return build_hit(chunk_id, method="lexical", score=score, arxiv_id=arxiv_id, strict=strict, chunk_index=chunk_id)


def vector_hit(chunk_id: int, score: float, arxiv_id: str) -> RetrievalHit:
    return build_hit(chunk_id, method="vector", score=score, arxiv_id=arxiv_id, chunk_index=chunk_id)


def sample_channels() -> ChannelData:
    return ChannelData(
        lexical=[
            lexical_hit(1, 0.9, "2401.00001"),
            lexical_hit(2, 0.8, "2401.00001"),
            lexical_hit(3, 0.7, "2401.00002"),
            lexical_hit(4, 0.6, "2401.00003"),
            lexical_hit(5, 0.5, "2401.00004", strict=False),
            lexical_hit(6, 0.4, "2401.00005"),
        ],
        vector=[
            vector_hit(7, 0.95, "2401.00006"),
            vector_hit(3, 0.9, "2401.00002"),
            vector_hit(1, 0.85, "2401.00001"),
            vector_hit(8, 0.7, "2401.00007"),
            vector_hit(9, 0.5, "2401.00003"),
            vector_hit(10, 0.4, "2401.00008"),
        ],
    )


def sample_inputs(query: str = "policy loss") -> HybridInputs:
    data = sample_channels()
    return HybridInputs(query=query, lexical=tuple(data.lexical), vector=tuple(data.vector))


class FakeRetriever(Retriever):
    def __init__(
        self,
        data: dict[str, ChannelData] | None = None,
        *,
        mode: RetrievalMode = "hybrid",
        embed_error: bool = False,
    ) -> None:
        super().__init__(None, StubEmbedder(), RetrievalConfig(mode=mode))  # type: ignore[arg-type]
        self.data = data if data is not None else {}
        self.embed_error = embed_error
        self.hybrid_calls: list[tuple[str, int, str | None]] = []
        self.built: list[list[int]] = []

    def channels(self, query: str) -> ChannelData:
        normalized = normalize_search_query(query)
        return self.data.get(normalized, self.data.get(query, ChannelData()))

    async def search_lexical(self, query: str, *, limit: int = 5, arxiv_id: str | None = None) -> list[RetrievalHit]:
        return list(self.channels(query).lexical)

    async def search_vector(self, query: str, *, limit: int = 5, arxiv_id: str | None = None) -> list[RetrievalHit]:
        if self.embed_error:
            raise OpenAIError("embedding failed")
        return list(self.channels(query).vector)

    async def hybrid_inputs(self, query: str, **kwargs: Any) -> HybridInputs:
        self.hybrid_calls.append((query, kwargs.get("limit", 5), kwargs.get("arxiv_id")))
        return await super().hybrid_inputs(query, **kwargs)

    async def build_contexts(self, hits: Sequence[RetrievalHit], **kwargs: Any) -> list[RetrievalContext]:
        self.built.append([hit.chunk_id for hit in hits])
        return [build_context(replace(hit)) for hit in hits]
