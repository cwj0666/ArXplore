import logging
from dataclasses import asdict
from typing import Any

from openai import OpenAIError

from arxplore.context import RetrievalObservation, record_retrieval
from arxplore.domain.retrieval import (
    ChannelFusion,
    FusionScores,
    HybridInputs,
    LexicalScores,
    RetrievalHit,
    RetrievalResult,
    VectorScores,
)
from arxplore.retrieval.diversity import apply_paper_diversity
from arxplore.retrieval.fusion import LEXICAL_WEIGHT, fuse_hybrid_hits
from arxplore.retrieval.service import DEFAULT_SEARCH_LIMIT, Retriever
from arxplore_eval.research_fusion import LEGACY_RULES_FUSION, fuse_hits

logger = logging.getLogger(__name__)

FUSION_CONVEX = "convex"
FUSION_LEGACY_RULES = "legacy_rules"
FUSIONS = (FUSION_CONVEX, FUSION_LEGACY_RULES)
FUSION_DESCRIPTIONS = {
    FUSION_CONVEX: f"제품 fuse_hybrid_hits(min-max 정규화, lexical {LEXICAL_WEIGHT}·vector {1 - LEXICAL_WEIGHT:.2f})",
    FUSION_LEGACY_RULES: "research_fusion.LEGACY_RULES_FUSION(2026-09-29까지의 가중 RRF 규칙)",
}


def check_fusion(fusion: str) -> str:
    if fusion not in FUSIONS:
        raise ValueError(f"알 수 없는 fusion: {fusion!r}. 사용 가능: {', '.join(FUSIONS)}")
    return fusion


def fuse_inputs(
    inputs: HybridInputs, fusion: str, *, limit: int = DEFAULT_SEARCH_LIMIT, arxiv_id: str | None = None
) -> list[RetrievalHit]:
    check_fusion(fusion)
    if not inputs.query:
        return []
    if fusion == FUSION_CONVEX:
        fused = fuse_hybrid_hits(inputs.lexical, inputs.vector)
        return apply_paper_diversity(fused, limit=max(1, limit), arxiv_id=arxiv_id)
    if arxiv_id is not None:
        raise ValueError("legacy_rules 융합은 논문 범위(arxiv_id) 검색을 지원하지 않습니다.")
    return fuse_hits(inputs, LEGACY_RULES_FUSION, limit=limit)


class FusionRetriever:
    def __init__(self, retriever: Retriever, fusion: str) -> None:
        self.retriever = retriever
        self.fusion = check_fusion(fusion)

    async def retrieve_contexts(
        self, query: str, *, limit: int = DEFAULT_SEARCH_LIMIT, arxiv_id: str | None = None
    ) -> RetrievalResult:
        result = None
        if self.retriever.resolve_mode() == "hybrid":
            try:
                inputs = await self.retriever.hybrid_inputs(query, limit=limit, arxiv_id=arxiv_id)
                hits = fuse_inputs(inputs, self.fusion, limit=limit, arxiv_id=arxiv_id)
                contexts = await self.retriever.build_contexts(hits)
                result = RetrievalResult(contexts=tuple(contexts), mode="hybrid")
            except OpenAIError as exc:
                logger.warning("질의 임베딩 실패로 lexical 검색으로 전환합니다: %s", type(exc).__name__)
        if result is None:
            contexts = await self.retriever.search_lexical_contexts(query, limit=limit, arxiv_id=arxiv_id)
            result = RetrievalResult(contexts=tuple(contexts), mode="lexical")
        record_retrieval(RetrievalObservation(query=query, arxiv_id=arxiv_id, limit=limit, result=result))
        return result


def hit_payload(hit: RetrievalHit) -> dict[str, Any]:
    return asdict(hit)


def _optional(cls: type[Any], payload: Any) -> Any:
    return None if payload is None else cls(**payload)


def hit_from_payload(payload: dict[str, Any]) -> RetrievalHit:
    fusion = payload.get("fusion")
    return RetrievalHit(
        **{key: value for key, value in payload.items() if key not in ("lexical", "vector", "fusion")},
        lexical=_optional(LexicalScores, payload.get("lexical")),
        vector=_optional(VectorScores, payload.get("vector")),
        fusion=None
        if fusion is None
        else FusionScores(
            lexical=_optional(ChannelFusion, fusion.get("lexical")),
            vector=_optional(ChannelFusion, fusion.get("vector")),
        ),
    )


def inputs_payload(inputs: HybridInputs) -> dict[str, Any]:
    return {
        "query": inputs.query,
        "lexical": [hit_payload(hit) for hit in inputs.lexical],
        "vector": [hit_payload(hit) for hit in inputs.vector],
    }


def inputs_from_payload(payload: dict[str, Any]) -> HybridInputs:
    return HybridInputs(
        query=str(payload["query"]),
        lexical=tuple(hit_from_payload(item) for item in payload["lexical"]),
        vector=tuple(hit_from_payload(item) for item in payload["vector"]),
    )
