from __future__ import annotations

import itertools

import pytest

from src.integrations.paper_retriever import PaperRetriever

RRF_K = 60.0
BOTH_CHANNEL_BONUS = 0.015


def _retriever() -> PaperRetriever:
    return PaperRetriever(repository=object(), embedding_client=object(), vector_repository=object())


def _candidate(chunk_id: int | None, score: float, *, arxiv_id: str | None = None, snippet: str = "") -> dict:
    return {
        "chunk_id": chunk_id,
        "arxiv_id": arxiv_id or f"2401.{(chunk_id or 0):05d}",
        "chunk_index": 0,
        "chunk_text": "body",
        "section_title": "Method",
        "content_role": "body",
        "score": score,
        "snippet": snippet,
    }


def _ids(candidates: list[dict]) -> list[int]:
    return [candidate["chunk_id"] for candidate in candidates]


class TestMergeHybridCandidates:
    def test_rrf_scores_and_both_channel_bonus(self):
        lexical = [_candidate(1, 0.9), _candidate(2, 0.6)]
        vector = [_candidate(2, 0.8), _candidate(3, 0.7)]

        merged = _retriever()._merge_hybrid_candidates("policy loss", lexical, vector, arxiv_id=None, limit=10)

        by_id = {candidate["chunk_id"]: candidate for candidate in merged}
        assert by_id[1]["score"] == pytest.approx(1.0 / (RRF_K + 1))
        assert by_id[2]["score"] == pytest.approx(0.85 / (RRF_K + 2) + 1.0 / (RRF_K + 1) + BOTH_CHANNEL_BONUS)
        assert by_id[3]["score"] == pytest.approx(1.0 / (RRF_K + 2))
        assert _ids(merged) == [2, 1, 3]

    def test_score_breakdown_records_both_channels(self):
        lexical = [_candidate(1, 0.9), _candidate(2, 0.6)]
        vector = [_candidate(2, 0.8)]

        merged = _retriever()._merge_hybrid_candidates("policy loss", lexical, vector, arxiv_id=None, limit=10)
        entry = next(candidate for candidate in merged if candidate["chunk_id"] == 2)

        assert entry["matched_methods"] == ["lexical", "vector"]
        assert entry["retrieval_method"] == "hybrid"
        breakdown = entry["score_breakdown"]
        assert breakdown["lexical_rank"] == 2
        assert breakdown["vector_rank"] == 1
        assert breakdown["lexical_quality_weight"] == 0.85
        assert breakdown["vector_quality_weight"] == 1.0
        assert breakdown["lexical_rrf_score"] == pytest.approx(0.85 / 62)
        assert breakdown["vector_rrf_score"] == pytest.approx(1.0 / 61)
        assert breakdown["cross_method_overlap_bonus"] == BOTH_CHANNEL_BONUS
        assert entry["similarity_score"] == entry["score"]

    def test_single_channel_hits_get_no_bonus(self):
        merged = _retriever()._merge_hybrid_candidates(
            "policy loss", [_candidate(1, 0.9)], [_candidate(2, 0.9)], arxiv_id=None, limit=10
        )

        assert all("cross_method_overlap_bonus" not in candidate["score_breakdown"] for candidate in merged)

    def test_low_lexical_confidence_shifts_weight_to_vector(self):
        lexical = [_candidate(1, 0.25)]
        vector = [_candidate(2, 0.5)]

        merged = _retriever()._merge_hybrid_candidates("policy loss", lexical, vector, arxiv_id=None, limit=10)
        by_id = {candidate["chunk_id"]: candidate for candidate in merged}

        assert by_id[1]["score"] == pytest.approx(0.45 * 0.75 * 0.4 / 61)
        assert by_id[2]["score"] == pytest.approx(1.1 * 1.08 / 61)
        assert _ids(merged) == [2, 1]

    def test_lexical_snippet_wins_over_vector_snippet(self):
        lexical = [_candidate(1, 0.9, snippet="lexical snippet")]
        vector = [_candidate(1, 0.9, snippet="vector snippet"), _candidate(2, 0.9, snippet="vector only")]

        merged = _retriever()._merge_hybrid_candidates("policy loss", lexical, vector, arxiv_id=None, limit=10)
        by_id = {candidate["chunk_id"]: candidate for candidate in merged}

        assert by_id[1]["snippet"] == "lexical snippet"
        assert by_id[2]["snippet"] == "vector only"

    def test_candidates_without_chunk_id_are_never_merged(self):
        lexical = [_candidate(None, 0.9)]
        vector = [_candidate(None, 0.9)]

        merged = _retriever()._merge_hybrid_candidates("policy loss", lexical, vector, arxiv_id=None, limit=10)

        assert len(merged) == 2
        assert all(len(candidate["matched_methods"]) == 1 for candidate in merged)

    def test_ties_break_on_method_count_then_chunk_id(self):
        merged = _retriever()._merge_hybrid_candidates(
            "policy loss", [_candidate(5, 0.9)], [_candidate(7, 0.9)], arxiv_id=None, limit=10
        )

        assert merged[0]["score"] == pytest.approx(merged[1]["score"])
        assert _ids(merged) == [7, 5]

    def test_result_goes_through_paper_diversity(self):
        same_paper = [_candidate(i, 0.9, arxiv_id="2401.00001") for i in range(1, 5)]
        other_paper = _candidate(10, 0.9, arxiv_id="2401.00002")

        merged = _retriever()._merge_hybrid_candidates(
            "policy loss", same_paper, [*same_paper, other_paper], arxiv_id=None, limit=3
        )

        assert _ids(merged) == [1, 2, 10]


class TestResolveHybridMethodWeights:
    SHORT_QUERY = "policy loss"
    LONG_QUERY = "direct preference optimization policy loss reward"

    @pytest.mark.parametrize(
        ("query", "lexical_top", "overlap", "expected_lexical", "expected_vector"),
        [
            (SHORT_QUERY, 0.9, True, 1.0, 1.0),
            (LONG_QUERY, 0.9, True, 0.85, 1.05),
            (SHORT_QUERY, 0.29, True, 0.45, 1.1),
            (SHORT_QUERY, 0.3, True, 0.7, 1.05),
            (SHORT_QUERY, 0.49, True, 0.7, 1.05),
            (SHORT_QUERY, 0.5, True, 1.0, 1.0),
            (SHORT_QUERY, 0.35, False, 0.7 * 0.75, 1.05 * 1.08),
            (SHORT_QUERY, 0.4, False, 0.7, 1.05),
            (SHORT_QUERY, 0.9, False, 1.0, 1.0),
            (LONG_QUERY, 0.1, False, 0.85 * 0.45 * 0.75, 1.05 * 1.1 * 1.08),
        ],
    )
    def test_branches(self, query, lexical_top, overlap, expected_lexical, expected_vector):
        lexical = [_candidate(1, lexical_top)]
        vector = [_candidate(1 if overlap else 99, 0.9)]

        weights = _retriever()._resolve_hybrid_method_weights(query, lexical, vector)

        assert weights["lexical"] == pytest.approx(expected_lexical)
        assert weights["vector"] == pytest.approx(expected_vector)

    def test_token_count_threshold_is_five(self):
        retriever = _retriever()
        four = "direct preference policy reward"
        five = "direct preference policy reward optimization"
        assert len(retriever._query_tokens(four)) == 4
        assert len(retriever._query_tokens(five)) == 5

        lexical = [_candidate(1, 0.9)]
        vector = [_candidate(1, 0.9)]
        assert retriever._resolve_hybrid_method_weights(four, lexical, vector)["lexical"] == 1.0
        assert retriever._resolve_hybrid_method_weights(five, lexical, vector)["lexical"] == pytest.approx(0.85)

    def test_empty_lexical_channel_counts_as_zero_confidence(self):
        weights = _retriever()._resolve_hybrid_method_weights(self.SHORT_QUERY, [], [_candidate(1, 0.9)])

        assert weights["lexical"] == pytest.approx(0.45 * 0.75)
        assert weights["vector"] == pytest.approx(1.1 * 1.08)

    def test_overlap_only_counts_top_five(self):
        lexical = [_candidate(i, 0.35) for i in range(1, 7)]
        vector = [_candidate(i, 0.9) for i in (6, 10, 11, 12, 13)]

        weights = _retriever()._resolve_hybrid_method_weights(self.SHORT_QUERY, lexical, vector)

        assert weights["lexical"] == pytest.approx(0.7 * 0.75)

    def test_floors_are_never_reached_with_current_constants(self):
        retriever = _retriever()
        observed_lexical: list[float] = []
        observed_vector: list[float] = []

        for query, lexical_top, overlap in itertools.product(
            (self.SHORT_QUERY, self.LONG_QUERY),
            (0.0, 0.1, 0.35, 0.45, 0.9),
            (True, False),
        ):
            lexical = [_candidate(1, lexical_top)] if lexical_top else []
            vector = [_candidate(1 if overlap else 99, 0.9)]
            weights = retriever._resolve_hybrid_method_weights(query, lexical, vector)
            observed_lexical.append(weights["lexical"])
            observed_vector.append(weights["vector"])

        assert min(observed_lexical) == pytest.approx(0.85 * 0.45 * 0.75)
        assert min(observed_lexical) > 0.2
        assert min(observed_vector) == 1.0
        assert min(observed_vector) > 0.5


class TestCandidateQualityWeight:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (0.0, 0.2),
            (0.19, 0.2),
            (0.2, 0.4),
            (0.29, 0.4),
            (0.3, 0.65),
            (0.49, 0.65),
            (0.5, 0.85),
            (0.79, 0.85),
            (0.8, 1.0),
            (3.0, 1.0),
        ],
    )
    def test_lexical_score_bands(self, score, expected):
        assert _retriever()._candidate_hybrid_quality_weight("lexical", {"score": score}) == expected

    @pytest.mark.parametrize("score", [0.0, 0.1, 0.9])
    def test_vector_is_always_full_weight(self, score):
        assert _retriever()._candidate_hybrid_quality_weight("vector", {"score": score}) == 1.0

    def test_unparseable_score_is_treated_as_zero(self):
        assert _retriever()._candidate_hybrid_quality_weight("lexical", {"score": "n/a"}) == 0.2
