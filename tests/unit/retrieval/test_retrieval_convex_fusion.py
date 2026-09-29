import pytest

from arxplore.domain.retrieval import RetrievalHit
from arxplore.retrieval.diversity import apply_paper_diversity
from arxplore.retrieval.fusion import (
    LEXICAL_WEIGHT,
    fuse_hybrid_hits,
    fusion_lexical_hits,
    normalize_channel_scores,
)


def _ids(hits: list[RetrievalHit]) -> list[int]:
    return [hit.chunk_id for hit in hits]


def _scores(hits: list[RetrievalHit]) -> dict[int, float]:
    return {hit.chunk_id: hit.score for hit in hits}


class TestNormalizeChannelScores:
    def test_minmax_spans_zero_to_one(self) -> None:
        assert normalize_channel_scores([3.0, 2.0, 1.0]) == [1.0, 0.5, 0.0]

    def test_single_candidate_is_degenerate_and_gets_one(self) -> None:
        assert normalize_channel_scores([0.42]) == [1.0]

    def test_equal_scores_get_one(self) -> None:
        assert normalize_channel_scores([0.7, 0.7, 0.7]) == [1.0, 1.0, 1.0]

    def test_empty_channel(self) -> None:
        assert normalize_channel_scores([]) == []


class TestPartialLexicalRows:
    def test_partial_rows_are_dropped_when_vector_has_results(self, make_hit) -> None:
        lexical = [make_hit(1, score=1.5), make_hit(2, score=0.8, strict=False)]
        vector = [make_hit(2, method="vector", score=0.8), make_hit(3, method="vector", score=0.7)]

        assert _ids(fusion_lexical_hits(lexical, vector)) == [1]

        fused = {hit.chunk_id: hit for hit in fuse_hybrid_hits(lexical, vector)}
        assert set(fused) == {1, 2, 3}
        assert fused[2].fusion is not None
        assert fused[2].fusion.matched_methods == ("vector",)
        assert fused[1].fusion is not None
        assert fused[1].fusion.matched_methods == ("lexical",)

    def test_partial_rows_are_kept_without_vector_results(self, make_hit) -> None:
        lexical = [make_hit(1, score=0.8, strict=False), make_hit(2, score=0.6, strict=False)]

        fused = fuse_hybrid_hits(lexical, [])

        assert _ids(fused) == [1, 2]
        assert all(hit.fusion is not None and hit.fusion.matched_methods == ("lexical",) for hit in fused)

    def test_single_lexical_row_left_after_the_drop_is_degenerate(self, make_hit) -> None:
        lexical = [make_hit(1, score=1.2), make_hit(2, score=0.9, strict=False)]
        vector = [make_hit(3, method="vector", score=0.9), make_hit(4, method="vector", score=0.5)]

        fused = fuse_hybrid_hits(lexical, vector, lexical_weight=0.5)

        assert _ids(fused) == [3, 1, 4]
        assert fused[1].fusion is not None
        assert fused[1].fusion.lexical is not None
        assert fused[1].fusion.lexical.normalized_score == 1.0


class TestConvexFusion:
    def test_product_weight_is_lexical_035(self) -> None:
        assert LEXICAL_WEIGHT == 0.35

    def test_missing_channel_counts_as_zero_and_ties_break_on_method_count(self, make_hit) -> None:
        lexical = [make_hit(1, score=2.0), make_hit(2, score=1.0)]
        vector = [make_hit(2, method="vector", score=0.9), make_hit(3, method="vector", score=0.5)]

        fused = fuse_hybrid_hits(lexical, vector, lexical_weight=0.5)

        assert [hit.score for hit in fused] == [0.5, 0.5, 0.0]
        assert _ids(fused) == [2, 1, 3]

    def test_each_channel_is_weighted_by_alpha(self, make_hit) -> None:
        lexical = [make_hit(1, score=3.0), make_hit(2, score=1.0)]
        vector = [
            make_hit(2, method="vector", score=0.9),
            make_hit(3, method="vector", score=0.6),
            make_hit(4, method="vector", score=0.3),
        ]

        fused = fuse_hybrid_hits(lexical, vector, lexical_weight=0.3)

        assert _scores(fused) == pytest.approx({1: 0.3, 2: 0.7, 3: 0.35, 4: 0.0})
        assert _ids(fused) == [2, 3, 1, 4]

    def test_default_weights_sum_scores_in_channel_order(self, make_hit) -> None:
        lexical = [make_hit(1, score=2.0), make_hit(2, score=1.0), make_hit(5, score=1.5)]
        vector = [make_hit(2, method="vector", score=0.8), make_hit(3, method="vector", score=0.4)]

        fused = {hit.chunk_id: hit for hit in fuse_hybrid_hits(lexical, vector)}

        assert fused[2].score == 0.0 + 0.35 * 0.0 + 0.65 * 1.0
        assert fused[5].score == 0.35 * 0.5
        assert fused[2].fusion is not None
        assert fused[2].fusion.lexical is not None
        assert fused[2].fusion.vector is not None
        assert (fused[2].fusion.lexical.rank, fused[2].fusion.vector.rank) == (2, 1)
        assert (fused[2].fusion.lexical.weight, fused[2].fusion.vector.weight) == (0.35, 0.65)

    def test_empty_lexical_channel_keeps_the_vector_order(self, make_hit) -> None:
        vector = [
            make_hit(5, method="vector", score=0.9),
            make_hit(3, method="vector", score=0.7),
            make_hit(9, method="vector", score=0.2),
        ]

        fused = fuse_hybrid_hits([], vector, lexical_weight=0.3)

        assert _ids(fused) == [5, 3, 9]
        assert [hit.score for hit in fused] == pytest.approx([0.7, 0.7 * 0.5 / 0.7, 0.0])

    def test_alpha_endpoints_follow_one_channel(self, make_hit) -> None:
        lexical = [make_hit(1, score=2.0), make_hit(2, score=1.5), make_hit(3, score=1.0)]
        vector = [
            make_hit(3, method="vector", score=0.9),
            make_hit(2, method="vector", score=0.6),
            make_hit(4, method="vector", score=0.3),
        ]

        assert _ids(fuse_hybrid_hits(lexical, vector, lexical_weight=0.0))[:3] == [3, 2, 4]
        assert _ids(fuse_hybrid_hits(lexical, vector, lexical_weight=1.0))[:3] == [1, 2, 3]

    def test_ties_break_on_chunk_id_after_method_count(self, make_hit) -> None:
        fused = fuse_hybrid_hits(
            [make_hit(5, score=0.9)], [make_hit(7, method="vector", score=0.9)], lexical_weight=0.5
        )

        assert fused[0].score == fused[1].score
        assert _ids(fused) == [7, 5]

    def test_lexical_snippet_wins_over_the_vector_snippet(self, make_hit) -> None:
        lexical = [make_hit(1, score=1.5, snippet="lexical snippet")]
        vector = [
            make_hit(1, method="vector", score=0.9, snippet="vector snippet"),
            make_hit(2, method="vector", score=0.8, snippet="vector only"),
        ]

        fused = {hit.chunk_id: hit for hit in fuse_hybrid_hits(lexical, vector)}

        assert fused[1].snippet == "lexical snippet"
        assert fused[2].snippet == "vector only"

    def test_merged_hits_are_marked_hybrid(self, make_hit) -> None:
        fused = fuse_hybrid_hits([make_hit(1, score=1.5)], [make_hit(1, method="vector", score=0.9)])

        assert fused[0].method == "hybrid"
        assert fused[0].fusion is not None
        assert fused[0].fusion.matched_methods == ("lexical", "vector")

    def test_result_goes_through_paper_diversity_in_the_product_path(self, make_hit) -> None:
        same_paper = [make_hit(chunk_id, score=1.5, arxiv_id="2401.00001") for chunk_id in range(1, 5)]
        vector = [
            *(make_hit(chunk_id, method="vector", score=0.9, arxiv_id="2401.00001") for chunk_id in range(1, 5)),
            make_hit(10, method="vector", score=0.9, arxiv_id="2401.00002"),
        ]

        fused = fuse_hybrid_hits(same_paper, vector)

        assert _ids(apply_paper_diversity(fused, limit=3)) == [4, 3, 10]


class TestMergedHitFields:
    def test_a_hit_found_by_both_channels_keeps_both_score_breakdowns(self, make_hit) -> None:
        lexical = [make_hit(1, score=1.5, abstract="")]
        vector = [make_hit(1, method="vector", score=0.9, abstract="We scale latent diffusion.")]

        merged = fuse_hybrid_hits(lexical, vector)[0]

        assert merged.lexical == lexical[0].lexical
        assert merged.vector == vector[0].vector
        assert merged.paper_abstract == "We scale latent diffusion."

    def test_the_lexical_abstract_is_kept_when_present(self, make_hit) -> None:
        lexical = [make_hit(1, score=1.5, abstract="Lexical row abstract.")]
        vector = [make_hit(1, method="vector", score=0.9, abstract="Vector row abstract.")]

        assert fuse_hybrid_hits(lexical, vector)[0].paper_abstract == "Lexical row abstract."

    def test_single_channel_hits_carry_only_their_own_scores(self, make_hit) -> None:
        lexical = [make_hit(1, score=1.5)]
        vector = [make_hit(2, method="vector", score=0.9)]

        fused = {hit.chunk_id: hit for hit in fuse_hybrid_hits(lexical, vector)}

        assert fused[1].lexical is not None
        assert fused[1].vector is None
        assert fused[2].lexical is None
        assert fused[2].vector is not None
