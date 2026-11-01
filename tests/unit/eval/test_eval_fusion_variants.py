import json
from dataclasses import replace

import pytest

from arxplore.context import observe_retrieval
from arxplore.domain.retrieval import (
    ChannelFusion,
    FusionScores,
    HybridInputs,
    LexicalScores,
    RetrievalHit,
    VectorScores,
)
from arxplore.retrieval.diversity import apply_paper_diversity
from arxplore.retrieval.fusion import fuse_hybrid_hits
from arxplore.retrieval.service import Retriever
from arxplore_eval.fusion_variants import (
    FUSION_CONVEX,
    FUSION_LEGACY_RULES,
    FUSIONS,
    FusionRetriever,
    check_fusion,
    fuse_inputs,
    hit_from_payload,
    hit_payload,
    inputs_from_payload,
    inputs_payload,
)
from arxplore_eval.research_fusion import LEGACY_RULES_FUSION, fuse_hits
from tests.unit.eval.conftest import build_hit
from tests.unit.eval.plan_support import FakeRetriever, sample_channels, sample_inputs


class TestFuseInputs:
    @pytest.mark.parametrize("limit", [1, 2, 3, 5, 10])
    def test_convex_equals_the_product_path(self, limit: int) -> None:
        inputs = sample_inputs()
        fused = fuse_hybrid_hits(inputs.lexical, inputs.vector)
        expected = apply_paper_diversity(fused, limit=limit, arxiv_id=None)
        assert fuse_inputs(inputs, FUSION_CONVEX, limit=limit) == expected
        assert expected

    def test_convex_with_arxiv_id_equals_the_product_path(self) -> None:
        inputs = sample_inputs()
        expected = apply_paper_diversity(
            fuse_hybrid_hits(inputs.lexical, inputs.vector), limit=4, arxiv_id="2401.00001"
        )
        assert fuse_inputs(inputs, FUSION_CONVEX, limit=4, arxiv_id="2401.00001") == expected

    def test_convex_clamps_the_limit_to_one(self) -> None:
        inputs = sample_inputs()
        assert len(fuse_inputs(inputs, FUSION_CONVEX, limit=0)) == 1

    @pytest.mark.parametrize("limit", [1, 3, 5])
    def test_legacy_rules_equals_the_ablation_path(self, limit: int) -> None:
        inputs = sample_inputs()
        expected = fuse_hits(inputs, LEGACY_RULES_FUSION, limit=limit)
        assert fuse_inputs(inputs, FUSION_LEGACY_RULES, limit=limit) == expected
        assert expected

    @pytest.mark.parametrize("fusion", FUSIONS)
    def test_empty_query_gives_nothing(self, fusion: str) -> None:
        inputs = replace(sample_inputs(), query="")
        assert fuse_inputs(inputs, fusion) == []

    def test_empty_query_wins_over_the_legacy_arxiv_id_check(self) -> None:
        inputs = HybridInputs(query="", lexical=(), vector=())
        assert fuse_inputs(inputs, FUSION_LEGACY_RULES, arxiv_id="2401.00001") == []

    @pytest.mark.parametrize("name", ["", "Convex", "rrf", "legacy"])
    def test_unknown_fusion_raises(self, name: str) -> None:
        with pytest.raises(ValueError, match="알 수 없는 fusion"):
            fuse_inputs(sample_inputs(), name)
        with pytest.raises(ValueError, match="알 수 없는 fusion"):
            check_fusion(name)

    def test_unknown_fusion_raises_even_for_an_empty_query(self) -> None:
        with pytest.raises(ValueError, match="알 수 없는 fusion"):
            fuse_inputs(HybridInputs(query="", lexical=(), vector=()), "nope")

    def test_legacy_rules_reject_a_paper_scope(self) -> None:
        with pytest.raises(ValueError, match="arxiv_id"):
            fuse_inputs(sample_inputs(), FUSION_LEGACY_RULES, arxiv_id="2401.00001")

    def test_check_fusion_returns_the_name(self) -> None:
        assert [check_fusion(name) for name in FUSIONS] == ["convex", "legacy_rules"]


async def _product(retriever: FakeRetriever, query: str, limit: int):
    return await Retriever.retrieve_contexts(retriever, query, limit=limit)


class TestFusionRetriever:
    def _retriever(self, **kwargs) -> FakeRetriever:
        return FakeRetriever({"policy loss": sample_channels()}, **kwargs)

    def test_unknown_fusion_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="알 수 없는 fusion"):
            FusionRetriever(self._retriever(), "nope")

    @pytest.mark.parametrize("limit", [1, 3, 5])
    async def test_convex_gives_the_same_contexts_as_the_product_retriever(self, limit: int) -> None:
        product = await _product(self._retriever(), "policy loss", limit)
        variant = await FusionRetriever(self._retriever(), FUSION_CONVEX).retrieve_contexts("policy loss", limit=limit)
        assert variant == product
        assert variant.mode == "hybrid"
        assert variant.contexts

    async def test_hybrid_flows_through_inputs_fuse_and_build_contexts(self) -> None:
        retriever = self._retriever()
        result = await FusionRetriever(retriever, FUSION_LEGACY_RULES).retrieve_contexts("policy loss", limit=3)
        expected = fuse_hits(sample_inputs(), LEGACY_RULES_FUSION, limit=3)
        assert retriever.hybrid_calls == [("policy loss", 3, None)]
        assert retriever.built == [[hit.chunk_id for hit in expected]]
        assert [context.hit.chunk_id for context in result.contexts] == [hit.chunk_id for hit in expected]
        assert result.mode == "hybrid"

    async def test_records_a_retrieval_observation(self) -> None:
        retriever = self._retriever()
        with observe_retrieval() as observations:
            result = await FusionRetriever(retriever, FUSION_CONVEX).retrieve_contexts("policy loss", limit=4)
        (observation,) = observations
        assert (observation.query, observation.arxiv_id, observation.limit) == ("policy loss", None, 4)
        assert observation.result == result

    async def test_embedding_failure_falls_back_to_lexical(self) -> None:
        retriever = self._retriever(embed_error=True)
        with observe_retrieval() as observations:
            result = await FusionRetriever(retriever, FUSION_CONVEX).retrieve_contexts("policy loss", limit=3)
        assert result.mode == "lexical"
        assert [context.hit.chunk_id for context in result.contexts] == [1, 2, 3, 4, 5, 6]
        assert [observation.result.mode for observation in observations] == ["lexical"]

    async def test_a_lexical_resolved_retriever_never_asks_for_hybrid_inputs(self) -> None:
        retriever = self._retriever(mode="lexical")
        result = await FusionRetriever(retriever, FUSION_LEGACY_RULES).retrieve_contexts("policy loss")
        assert result.mode == "lexical"
        assert retriever.hybrid_calls == []

    async def test_non_openai_errors_propagate(self) -> None:
        class Broken(FakeRetriever):
            async def search_lexical(self, query, *, limit=5, arxiv_id=None):
                raise RuntimeError("db down")

        with pytest.raises(RuntimeError, match="db down"):
            await FusionRetriever(Broken(), FUSION_CONVEX).retrieve_contexts("policy loss")


def _full_hit() -> RetrievalHit:
    return RetrievalHit(
        chunk_id=11,
        arxiv_id="2401.00011",
        chunk_index=3,
        paper_title="제목 Title",
        paper_abstract="abstract",
        chunk_text="chunk",
        section_title="3 Method",
        content_role="body",
        method="hybrid",
        score=0.123456789,
        snippet="snip",
        lexical=LexicalScores(
            fts_score=0.5,
            ilike_bonus=0.1,
            content_role_adjustment=-0.05,
            section_boost=0.2,
            structural_adjustment=0.0,
            coverage=0.75,
            strict_match=False,
            section_intent_bonus=0.3,
        ),
        vector=VectorScores(
            raw_similarity_score=0.8, content_role_adjustment=0.01, section_boost=0.02, rerank_adjustment=-0.1
        ),
        fusion=FusionScores(
            lexical=ChannelFusion(rank=2, score=0.5, normalized_score=0.25, weight=0.35),
            vector=ChannelFusion(rank=1, score=0.8, normalized_score=1.0, weight=0.65),
        ),
    )


def _variants() -> list[RetrievalHit]:
    full = _full_hit()
    return [
        full,
        replace(full, lexical=None),
        replace(full, vector=None),
        replace(full, fusion=None),
        replace(full, fusion=FusionScores(lexical=None, vector=full.fusion.vector if full.fusion else None)),
        replace(full, fusion=FusionScores(lexical=full.fusion.lexical if full.fusion else None, vector=None)),
        replace(full, fusion=FusionScores()),
        replace(full, lexical=None, vector=None, fusion=None),
        build_hit(5, method="vector", score=0.4),
        build_hit(6, method="lexical", score=0.3, strict=None),
    ]


class TestPayloads:
    @pytest.mark.parametrize("hit", _variants())
    def test_hit_round_trips_exactly(self, hit: RetrievalHit) -> None:
        assert hit_from_payload(hit_payload(hit)) == hit

    @pytest.mark.parametrize("hit", _variants())
    def test_hit_round_trips_through_json(self, hit: RetrievalHit) -> None:
        payload = json.loads(json.dumps(hit_payload(hit), ensure_ascii=False))
        restored = hit_from_payload(payload)
        assert restored == hit
        assert type(restored.lexical) is type(hit.lexical)
        assert type(restored.vector) is type(hit.vector)
        assert type(restored.fusion) is type(hit.fusion)

    def test_nested_types_are_rebuilt(self) -> None:
        restored = hit_from_payload(json.loads(json.dumps(hit_payload(_full_hit()))))
        assert isinstance(restored.lexical, LexicalScores)
        assert isinstance(restored.vector, VectorScores)
        assert isinstance(restored.fusion, FusionScores)
        assert isinstance(restored.fusion.lexical, ChannelFusion)
        assert isinstance(restored.fusion.vector, ChannelFusion)

    def test_hit_payload_is_plain_json_data(self) -> None:
        payload = hit_payload(_full_hit())
        assert payload["lexical"]["strict_match"] is False
        assert payload["fusion"]["vector"]["rank"] == 1
        assert json.loads(json.dumps(payload)) == payload

    def test_inputs_round_trip_through_json(self) -> None:
        fused = tuple(fuse_hybrid_hits(sample_inputs().lexical, sample_inputs().vector))
        inputs = HybridInputs(
            query="질의 query", lexical=(*sample_inputs().lexical, *fused), vector=sample_inputs().vector
        )
        payload = json.loads(json.dumps(inputs_payload(inputs), ensure_ascii=False))
        restored = inputs_from_payload(payload)
        assert restored == inputs
        assert isinstance(restored.lexical, tuple) and isinstance(restored.vector, tuple)

    def test_empty_inputs_round_trip(self) -> None:
        inputs = HybridInputs(query="", lexical=(), vector=())
        assert inputs_from_payload(json.loads(json.dumps(inputs_payload(inputs)))) == inputs

    def test_round_tripped_inputs_fuse_identically(self) -> None:
        inputs = sample_inputs()
        restored = inputs_from_payload(json.loads(json.dumps(inputs_payload(inputs))))
        for fusion in FUSIONS:
            assert fuse_inputs(restored, fusion, limit=5) == fuse_inputs(inputs, fusion, limit=5)
