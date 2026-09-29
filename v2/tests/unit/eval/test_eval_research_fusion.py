import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from arxplore.domain.retrieval import HybridInputs
from arxplore.retrieval.diversity import apply_paper_diversity as product_diversity
from arxplore.retrieval.fusion import LEXICAL_WEIGHT, fuse_hybrid_hits
from arxplore.retrieval.fusion import normalize_channel_scores as product_normalize
from arxplore.retrieval.lexical import STRICT_MATCH_BONUS
from arxplore.retrieval.query import query_tokens
from arxplore_eval.candidate_cache import cached_hit
from arxplore_eval.research_fusion import (
    LEGACY_RULES_FUSION,
    PRODUCT_FUSION,
    STANDARD_RRF_FUSION,
    STRICT_LEXICAL_SCORE_FLOOR,
    HybridFusionConfig,
    apply_paper_diversity,
    candidate_from_hit,
    fuse_hits,
    fuse_hybrid_candidates,
    hybrid_quality_weight,
    lexical_confidence,
    normalize_channel_scores,
    resolve_hybrid_method_weights,
)

RRF_K = 60.0
BOTH_CHANNEL_BONUS = 0.015
GOLDEN_PATH = Path(__file__).parent / "fixtures" / "hybrid_fusion_golden.json"
GOLDEN_CASES = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]


def _ids(candidates: list[dict[str, Any]]) -> list[Any]:
    return [candidate["chunk_id"] for candidate in candidates]


def _merge(
    query: str, lexical: list[dict[str, Any]], vector: list[dict[str, Any]], *, limit: int, config: HybridFusionConfig
) -> list[dict[str, Any]]:
    return apply_paper_diversity(fuse_hybrid_candidates(query, lexical, vector, config), limit=limit)


def _legacy(query: str, lexical: list[dict[str, Any]], vector: list[dict[str, Any]], *, limit: int = 10):
    return _merge(query, lexical, vector, limit=limit, config=LEGACY_RULES_FUSION)


def fusion_fingerprint(candidates: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(candidates, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _plain_rrf_reference(ranked_lists: list[list[dict[str, Any]]], rank_constant: float = 60.0) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for candidates in ranked_lists:
        for rank, item in enumerate(candidates, start=1):
            chunk_id = int(item.get("chunk_id") or 0)
            entry = merged.setdefault(chunk_id, {**item, "retrieval_method": "hybrid_plain_rrf", "score": 0.0})
            entry["score"] += 1.0 / (rank_constant + rank)
    return sorted(merged.values(), key=lambda item: (item["score"], int(item.get("chunk_id") or 0)), reverse=True)


def _convex(alpha: float, normalization: str = "minmax", **overrides: Any) -> HybridFusionConfig:
    return HybridFusionConfig(weighting="convex", convex_alpha=alpha, score_normalization=normalization, **overrides)


class TestLegacyRules:
    def test_rrf_scores_and_both_channel_bonus(self, make_candidate):
        lexical = [make_candidate(1, 0.9), make_candidate(2, 0.6)]
        vector = [make_candidate(2, 0.8), make_candidate(3, 0.7)]

        merged = _legacy("policy loss", lexical, vector)

        by_id = {item["chunk_id"]: item for item in merged}
        assert by_id[1]["score"] == pytest.approx(1.0 / (RRF_K + 1))
        assert by_id[2]["score"] == pytest.approx(0.85 / (RRF_K + 2) + 1.0 / (RRF_K + 1) + BOTH_CHANNEL_BONUS)
        assert by_id[3]["score"] == pytest.approx(1.0 / (RRF_K + 2))
        assert _ids(merged) == [2, 1, 3]

    def test_score_breakdown_records_both_channels(self, make_candidate):
        merged = _legacy("policy loss", [make_candidate(1, 0.9), make_candidate(2, 0.6)], [make_candidate(2, 0.8)])
        entry = next(item for item in merged if item["chunk_id"] == 2)

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

    def test_single_channel_hits_get_no_bonus(self, make_candidate):
        merged = _legacy("policy loss", [make_candidate(1, 0.9)], [make_candidate(2, 0.9)])
        assert all("cross_method_overlap_bonus" not in item["score_breakdown"] for item in merged)

    def test_low_lexical_confidence_shifts_weight_to_vector(self, make_candidate):
        merged = _legacy("policy loss", [make_candidate(1, 0.25)], [make_candidate(2, 0.5)])
        by_id = {item["chunk_id"]: item for item in merged}

        assert by_id[1]["score"] == pytest.approx(0.45 * 0.75 * 0.4 / 61)
        assert by_id[2]["score"] == pytest.approx(1.1 * 1.08 / 61)
        assert _ids(merged) == [2, 1]

    def test_lexical_snippet_wins_over_vector_snippet(self, make_candidate):
        lexical = [make_candidate(1, 0.9, snippet="lexical snippet")]
        vector = [make_candidate(1, 0.9, snippet="vector snippet"), make_candidate(2, 0.9, snippet="vector only")]

        by_id = {item["chunk_id"]: item for item in _legacy("policy loss", lexical, vector)}

        assert by_id[1]["snippet"] == "lexical snippet"
        assert by_id[2]["snippet"] == "vector only"

    def test_candidates_without_chunk_id_are_never_merged(self, make_candidate):
        merged = _legacy("policy loss", [make_candidate(None, 0.9)], [make_candidate(None, 0.9)])

        assert len(merged) == 2
        assert all(len(item["matched_methods"]) == 1 for item in merged)

    def test_ties_break_on_method_count_then_chunk_id(self, make_candidate):
        merged = _legacy("policy loss", [make_candidate(5, 0.9)], [make_candidate(7, 0.9)])

        assert merged[0]["score"] == pytest.approx(merged[1]["score"])
        assert _ids(merged) == [7, 5]

    def test_result_goes_through_paper_diversity(self, make_candidate):
        same_paper = [make_candidate(index, 0.9, arxiv_id="2401.00001") for index in range(1, 5)]
        other_paper = make_candidate(10, 0.9, arxiv_id="2401.00002")

        merged = _legacy("policy loss", same_paper, [*same_paper, other_paper], limit=3)

        assert _ids(merged) == [1, 2, 10]


class TestLegacyWeights:
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
    def test_branches(self, make_candidate, query, lexical_top, overlap, expected_lexical, expected_vector):
        lexical = [make_candidate(1, lexical_top)]
        vector = [make_candidate(1 if overlap else 99, 0.9)]

        weights = resolve_hybrid_method_weights(query, lexical, vector, LEGACY_RULES_FUSION)

        assert weights["lexical"] == pytest.approx(expected_lexical)
        assert weights["vector"] == pytest.approx(expected_vector)

    def test_token_count_threshold_is_five_with_the_product_tokenizer(self, make_candidate):
        four = "direct preference policy reward"
        five = "direct preference policy reward optimization"
        assert len(query_tokens(four)) == 4
        assert len(query_tokens(five)) == 5

        lexical = [make_candidate(1, 0.9)]
        vector = [make_candidate(1, 0.9)]
        assert resolve_hybrid_method_weights(four, lexical, vector, LEGACY_RULES_FUSION)["lexical"] == 1.0
        assert resolve_hybrid_method_weights(five, lexical, vector, LEGACY_RULES_FUSION)["lexical"] == pytest.approx(
            0.85
        )

    def test_empty_lexical_channel_counts_as_zero_confidence(self, make_candidate):
        weights = resolve_hybrid_method_weights(self.SHORT_QUERY, [], [make_candidate(1, 0.9)], LEGACY_RULES_FUSION)

        assert weights["lexical"] == pytest.approx(0.45 * 0.75)
        assert weights["vector"] == pytest.approx(1.1 * 1.08)

    def test_overlap_only_counts_top_five(self, make_candidate):
        lexical = [make_candidate(index, 0.35) for index in range(1, 7)]
        vector = [make_candidate(index, 0.9) for index in (6, 10, 11, 12, 13)]

        weights = resolve_hybrid_method_weights(self.SHORT_QUERY, lexical, vector, LEGACY_RULES_FUSION)

        assert weights["lexical"] == pytest.approx(0.7 * 0.75)

    def test_floors_are_never_reached_with_the_legacy_constants(self, make_candidate):
        observed_lexical: list[float] = []
        observed_vector: list[float] = []
        for query, lexical_top, overlap in itertools.product(
            (self.SHORT_QUERY, self.LONG_QUERY), (0.0, 0.1, 0.35, 0.45, 0.9), (True, False)
        ):
            lexical = [make_candidate(1, lexical_top)] if lexical_top else []
            vector = [make_candidate(1 if overlap else 99, 0.9)]
            weights = resolve_hybrid_method_weights(query, lexical, vector, LEGACY_RULES_FUSION)
            observed_lexical.append(weights["lexical"])
            observed_vector.append(weights["vector"])

        assert min(observed_lexical) == pytest.approx(0.85 * 0.45 * 0.75)
        assert min(observed_lexical) > 0.2
        assert min(observed_vector) == 1.0


class TestQualityWeight:
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
        assert hybrid_quality_weight("lexical", {"score": score}, LEGACY_RULES_FUSION) == expected

    @pytest.mark.parametrize("score", [0.0, 0.1, 0.9])
    def test_vector_is_always_full_weight(self, score):
        assert hybrid_quality_weight("vector", {"score": score}, LEGACY_RULES_FUSION) == 1.0

    def test_unparseable_score_is_treated_as_zero(self):
        assert hybrid_quality_weight("lexical", {"score": "n/a"}, LEGACY_RULES_FUSION) == 0.2


class TestTwoTierLexicalCandidates:
    def test_strict_rows_are_weighted_without_the_tier_bonus(self, make_lexical_candidate):
        strict = make_lexical_candidate(1, STRICT_MATCH_BONUS + 0.25, strict=True)
        assert lexical_confidence(strict) == pytest.approx(0.25)
        assert hybrid_quality_weight("lexical", strict, LEGACY_RULES_FUSION) == 0.4
        strong = make_lexical_candidate(1, STRICT_MATCH_BONUS + 0.9, strict=True)
        assert hybrid_quality_weight("lexical", strong, LEGACY_RULES_FUSION) == 1.0

    def test_partial_rows_have_zero_confidence(self, make_lexical_candidate):
        partial = make_lexical_candidate(1, 0.87, strict=False, coverage=0.8)
        assert lexical_confidence(partial) == 0.0
        assert hybrid_quality_weight("lexical", partial, LEGACY_RULES_FUSION) == 0.2

    def test_rows_without_tier_information_keep_their_score(self, make_candidate):
        assert lexical_confidence(make_candidate(1, 0.6)) == 0.6

    def test_partial_rows_are_dropped_when_vector_has_results(self, make_candidate, make_lexical_candidate):
        lexical = [
            make_lexical_candidate(1, STRICT_MATCH_BONUS + 0.5, strict=True),
            make_lexical_candidate(2, 0.8, strict=False, coverage=0.7),
        ]
        vector = [make_candidate(2, 0.8), make_candidate(3, 0.7)]

        by_id = {item["chunk_id"]: item for item in _legacy("policy loss", lexical, vector)}

        assert set(by_id) == {1, 2, 3}
        assert by_id[2]["matched_methods"] == ["vector"]
        assert "cross_method_overlap_bonus" not in by_id[2]["score_breakdown"]
        assert by_id[1]["matched_methods"] == ["lexical"]

    def test_partial_rows_are_kept_without_vector_results(self, make_lexical_candidate):
        lexical = [
            make_lexical_candidate(1, 0.8, strict=False, coverage=0.7),
            make_lexical_candidate(2, 0.6, strict=False, coverage=0.5),
        ]

        merged = _legacy("policy loss", lexical, [])

        assert _ids(merged) == [1, 2]
        assert all(item["matched_methods"] == ["lexical"] for item in merged)


class TestLegacyGolden:
    @pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda case: f"seed{case['seed']}")
    def test_legacy_rules_reproduce_the_recorded_output(self, synthetic_case, case):
        inputs = synthetic_case(case["seed"])
        assert (len(inputs["lexical"]), len(inputs["vector"]), inputs["limit"]) == (
            case["n_lexical"],
            case["n_vector"],
            case["limit"],
        )

        fused = fuse_hybrid_candidates(inputs["query"], inputs["lexical"], inputs["vector"], LEGACY_RULES_FUSION)
        assert _ids(fused) == case["fused_chunk_ids"]
        assert fusion_fingerprint(fused) == case["fused_sha256"]
        assert _ids(apply_paper_diversity(fused, limit=inputs["limit"])) == case["diversified_chunk_ids"]

    def test_golden_cases_cover_the_rule_branches(self, synthetic_case):
        weights = set()
        for case in GOLDEN_CASES:
            inputs = synthetic_case(case["seed"])
            lexical = [
                item
                for item in inputs["lexical"]
                if (item.get("score_breakdown") or {}).get("strict_match") is not False
            ]
            resolved = resolve_hybrid_method_weights(inputs["query"], lexical, inputs["vector"], LEGACY_RULES_FUSION)
            weights.add((round(resolved["lexical"], 6), round(resolved["vector"], 6)))
        assert len(weights) >= 5
        assert len(GOLDEN_CASES) == 60


class TestFusionConfig:
    def test_static_weights_scale_each_channel(self, make_candidate):
        config = HybridFusionConfig(
            weighting="static", static_weights=(0.3, 1.0), quality_weight=False, overlap_bonus=0
        )
        fused = fuse_hybrid_candidates("policy loss", [make_candidate(1, 0.9)], [make_candidate(2, 0.9)], config)

        by_id = {item["chunk_id"]: item for item in fused}
        assert by_id[1]["score"] == pytest.approx(0.3 / 61)
        assert by_id[2]["score"] == pytest.approx(1.0 / 61)
        assert _ids(fused) == [2, 1]

    @pytest.mark.parametrize(
        ("confidence", "expected_lexical"), [(0.0, 0.2), (0.15, 0.2 + 0.8 * 0.5), (0.3, 1.0), (0.9, 1.0)]
    )
    def test_confidence_linear_weight(self, make_candidate, make_lexical_candidate, confidence, expected_lexical):
        config = HybridFusionConfig(weighting="confidence_linear", confidence_min_weight=0.2, confidence_tau=0.3)
        lexical = [make_lexical_candidate(1, STRICT_MATCH_BONUS + confidence, strict=True)]

        weights = resolve_hybrid_method_weights("policy loss", lexical, [make_candidate(2, 0.9)], config)

        assert weights == {"lexical": pytest.approx(expected_lexical), "vector": 1.0}

    def test_confidence_linear_without_lexical_results_uses_the_floor(self, make_candidate):
        config = HybridFusionConfig(weighting="confidence_linear", confidence_min_weight=0.4, confidence_tau=0.5)
        assert resolve_hybrid_method_weights("policy loss", [], [make_candidate(2, 0.9)], config)["lexical"] == 0.4

    def test_quality_weight_switch(self, make_lexical_candidate):
        lexical = [make_lexical_candidate(1, STRICT_MATCH_BONUS + 0.1, strict=True)]
        on = fuse_hybrid_candidates("policy loss", lexical, [], HybridFusionConfig(weighting="static"))
        off = fuse_hybrid_candidates(
            "policy loss", lexical, [], HybridFusionConfig(weighting="static", quality_weight=False)
        )

        assert on[0]["score"] == pytest.approx(0.2 / 61)
        assert off[0]["score"] == pytest.approx(1.0 / 61)

    def test_rank_constant_and_overlap_bonus_are_configurable(self, make_candidate):
        config = HybridFusionConfig(weighting="static", quality_weight=False, rank_constant=10.0, overlap_bonus=0.5)
        fused = fuse_hybrid_candidates("policy loss", [make_candidate(1, 0.9)], [make_candidate(1, 0.9)], config)

        assert fused[0]["score"] == pytest.approx(2.0 / 11 + 0.5)
        assert fused[0]["score_breakdown"]["cross_method_overlap_bonus"] == 0.5

    def test_partial_lexical_rows_kept_when_filter_is_off(self, make_candidate, make_lexical_candidate):
        lexical = [make_lexical_candidate(1, 0.8, strict=False)]
        fused = fuse_hybrid_candidates(
            "policy loss", lexical, [make_candidate(2, 0.9)], HybridFusionConfig(drop_partial_lexical=False)
        )
        assert set(_ids(fused)) == {1, 2}

    @pytest.mark.parametrize(
        "overrides",
        [
            {"weighting": "rrf"},
            {"rank_constant": 0.0},
            {"weighting": "confidence_linear", "confidence_tau": 0.0},
            {"convex_alpha": 1.5},
            {"convex_alpha": -0.1},
            {"score_normalization": "zscore"},
        ],
    )
    def test_invalid_config_is_rejected(self, overrides):
        with pytest.raises(ValueError):
            HybridFusionConfig(**overrides)


class TestStandardRrfPreset:
    @pytest.mark.parametrize("seed", range(60))
    def test_matches_the_plain_rrf_reference(self, synthetic_case, seed):
        inputs = synthetic_case(seed)
        lexical = [item for item in inputs["lexical"] if item["chunk_id"] is not None]

        fused = fuse_hybrid_candidates(inputs["query"], lexical, inputs["vector"], STANDARD_RRF_FUSION)
        reference = _plain_rrf_reference([lexical, inputs["vector"]])

        assert _ids(fused) == _ids(reference)
        assert [item["score"] for item in fused] == [item["score"] for item in reference]

    def test_ties_break_by_chunk_id_descending(self, make_candidate):
        fused = fuse_hybrid_candidates(
            "policy loss", [make_candidate(1, 0.9)], [make_candidate(5, 0.9)], STANDARD_RRF_FUSION
        )
        assert _ids(fused) == [5, 1]

    def test_partial_lexical_rows_are_kept(self, make_candidate, make_lexical_candidate):
        lexical = [
            make_lexical_candidate(1, 0.8, strict=False),
            make_lexical_candidate(2, STRICT_MATCH_BONUS + 0.1, strict=True),
        ]
        vector = [make_candidate(2, 0.9)]

        fused = fuse_hybrid_candidates("policy loss", lexical, vector, STANDARD_RRF_FUSION)

        assert _ids(fused) == _ids(_plain_rrf_reference([lexical, vector])) == [2, 1]


class TestNormalizeChannelScores:
    def test_minmax_spans_zero_to_one(self):
        assert normalize_channel_scores([3.0, 2.0, 1.0], "minmax") == pytest.approx([1.0, 0.5, 0.0])

    def test_theoretical_uses_the_floor_instead_of_the_minimum(self):
        assert normalize_channel_scores([0.8, 0.2], "theoretical", -1.0) == pytest.approx([1.0, 1.2 / 1.8])
        assert normalize_channel_scores([2.0, 1.0], "theoretical", 0.0) == pytest.approx([1.0, 0.5])

    def test_scores_below_the_floor_stay_monotone(self):
        normalized = normalize_channel_scores([1.0, 0.5], "theoretical", STRICT_LEXICAL_SCORE_FLOOR)
        assert normalized[0] == 1.0 and normalized[1] < 0

    @pytest.mark.parametrize("normalization", ["minmax", "theoretical"])
    def test_single_candidate_is_degenerate_and_gets_one(self, normalization):
        assert normalize_channel_scores([0.42], normalization, 0.0) == [1.0]

    def test_theoretical_with_the_maximum_at_or_below_the_floor_is_degenerate(self):
        assert normalize_channel_scores([0.3, 0.2], "theoretical", STRICT_LEXICAL_SCORE_FLOOR) == [1.0, 1.0]

    def test_empty_channel_and_unknown_normalization(self):
        assert normalize_channel_scores([], "minmax") == []
        with pytest.raises(ValueError):
            normalize_channel_scores([1.0], "zscore")

    def test_strict_floor_is_the_bonus_minus_the_lowest_sql_adjustments(self):
        assert STRICT_LEXICAL_SCORE_FLOOR == pytest.approx(STRICT_MATCH_BONUS - 0.24 - 0.08 - 0.12) == 0.56

    @pytest.mark.parametrize("seed", range(20))
    def test_minmax_matches_the_product_normalization(self, synthetic_case, seed):
        scores = [item["score"] for item in synthetic_case(seed)["vector"]]
        assert normalize_channel_scores(scores, "minmax") == product_normalize(scores)


class TestConvexFusion:
    def test_missing_channel_scores_count_as_zero_and_ties_break_on_method_count(self, make_candidate):
        lexical = [make_candidate(1, 2.0), make_candidate(2, 1.0)]
        vector = [make_candidate(2, 0.9), make_candidate(3, 0.5)]

        fused = fuse_hybrid_candidates("policy loss", lexical, vector, _convex(0.5))

        assert [item["score"] for item in fused] == pytest.approx([0.5, 0.5, 0.0])
        assert _ids(fused) == [2, 1, 3]

    def test_combination_weights_each_channel_by_alpha(self, make_candidate):
        lexical = [make_candidate(1, 3.0), make_candidate(2, 1.0)]
        vector = [make_candidate(2, 0.9), make_candidate(3, 0.6), make_candidate(4, 0.3)]

        fused = fuse_hybrid_candidates("policy loss", lexical, vector, _convex(0.3))

        assert {item["chunk_id"]: item["score"] for item in fused} == pytest.approx({1: 0.3, 2: 0.7, 3: 0.35, 4: 0.0})
        assert _ids(fused) == [2, 3, 1, 4]

    def test_theoretical_normalization_uses_the_configured_floors(self, make_candidate):
        lexical = [make_candidate(1, 2.0), make_candidate(2, 1.0)]
        vector = [make_candidate(2, 0.5), make_candidate(3, 0.2)]
        config = _convex(0.4, "theoretical", score_floors=(0.0, -1.0))

        fused = fuse_hybrid_candidates("policy loss", lexical, vector, config)

        assert {item["chunk_id"]: item["score"] for item in fused} == pytest.approx(
            {1: 0.4, 2: 0.4 * 0.5 + 0.6, 3: 0.6 * 1.2 / 1.5}
        )

    def test_empty_lexical_channel_keeps_the_vector_order(self, make_candidate):
        vector = [make_candidate(5, 0.9), make_candidate(3, 0.7), make_candidate(9, 0.2)]

        fused = fuse_hybrid_candidates("policy loss", [], vector, _convex(0.3))

        assert _ids(fused) == [5, 3, 9]
        assert [item["score"] for item in fused] == pytest.approx([0.7, 0.7 * 0.5 / 0.7, 0.0])

    def test_alpha_endpoints_follow_one_channel(self, make_candidate):
        lexical = [make_candidate(1, 2.0), make_candidate(2, 1.5), make_candidate(3, 1.0)]
        vector = [make_candidate(3, 0.9), make_candidate(2, 0.6), make_candidate(4, 0.3)]

        assert _ids(fuse_hybrid_candidates("q", lexical, vector, _convex(0.0)))[:3] == [3, 2, 4]
        assert _ids(fuse_hybrid_candidates("q", lexical, vector, _convex(1.0)))[:3] == [1, 2, 3]

    def test_single_lexical_candidate_after_dropping_partial_rows_is_degenerate(
        self, make_candidate, make_lexical_candidate
    ):
        lexical = [
            make_lexical_candidate(1, STRICT_MATCH_BONUS + 0.2, strict=True),
            make_lexical_candidate(2, 0.9, strict=False),
        ]
        vector = [make_candidate(3, 0.9), make_candidate(4, 0.5)]

        fused = fuse_hybrid_candidates("policy loss", lexical, vector, _convex(0.5))

        assert _ids(fused) == [3, 1, 4]
        assert fused[1]["score_breakdown"]["lexical_normalized_score"] == 1.0

    def test_partial_rows_join_the_normalization_when_the_filter_is_off(self, make_candidate, make_lexical_candidate):
        lexical = [
            make_lexical_candidate(1, STRICT_MATCH_BONUS + 0.2, strict=True),
            make_lexical_candidate(2, 0.2, strict=False),
        ]

        fused = fuse_hybrid_candidates(
            "policy loss", lexical, [make_candidate(3, 0.9)], _convex(0.5, drop_partial_lexical=False)
        )

        by_id = {item["chunk_id"]: item for item in fused}
        assert by_id[2]["score"] == 0.0
        assert by_id[1]["score"] == pytest.approx(0.5)

    def test_rrf_only_terms_are_not_applied(self, make_candidate):
        fused = fuse_hybrid_candidates(
            "q", [make_candidate(1, 2.0)], [make_candidate(1, 0.9)], _convex(0.5, overlap_bonus=0.5)
        )

        assert fused[0]["score"] == pytest.approx(1.0)
        breakdown = fused[0]["score_breakdown"]
        assert "cross_method_overlap_bonus" not in breakdown
        assert "lexical_rrf_score" not in breakdown and "lexical_quality_weight" not in breakdown
        assert breakdown["lexical_normalized_score"] == breakdown["vector_normalized_score"] == 1.0
        assert breakdown["lexical_weight"] == breakdown["vector_weight"] == 0.5

    @pytest.mark.parametrize("seed", range(20))
    def test_convex_fields_do_not_change_rank_fusion(self, synthetic_case, seed):
        inputs = synthetic_case(seed)
        tweaked = HybridFusionConfig(
            weighting="rules", convex_alpha=0.9, score_normalization="theoretical", score_floors=(0.3, 0.0)
        )
        assert fuse_hybrid_candidates(inputs["query"], inputs["lexical"], inputs["vector"], tweaked) == (
            fuse_hybrid_candidates(inputs["query"], inputs["lexical"], inputs["vector"], LEGACY_RULES_FUSION)
        )


def _product_hits(inputs: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    lexical = [cached_hit(item, channel="lexical") for item in inputs["lexical"] if item["chunk_id"] is not None]
    vector = [cached_hit(item, channel="vector") for item in inputs["vector"]]
    return lexical, vector


class TestProductFusion:
    def test_product_fusion_is_the_minmax_convex_pick(self):
        assert HybridFusionConfig() == PRODUCT_FUSION
        assert PRODUCT_FUSION.weighting == "convex"
        assert PRODUCT_FUSION.convex_alpha == LEXICAL_WEIGHT == 0.35
        assert PRODUCT_FUSION.score_normalization == "minmax"
        assert PRODUCT_FUSION.drop_partial_lexical

    def test_legacy_rules_keep_every_old_constant(self):
        assert (
            HybridFusionConfig(
                weighting="rules",
                rank_constant=60.0,
                overlap_bonus=0.015,
                drop_partial_lexical=True,
                long_query_min_tokens=5,
                long_query_multipliers=(0.85, 1.05),
                confidence_bands=((0.3, 0.45, 1.1), (0.5, 0.7, 1.05)),
                no_overlap_top_n=5,
                no_overlap_max_confidence=0.4,
                no_overlap_multipliers=(0.75, 1.08),
                weight_floors=(0.2, 0.5),
                quality_weight=True,
                quality_tiers=((0.2, 0.2), (0.3, 0.4), (0.5, 0.65), (0.8, 0.85)),
            )
            == LEGACY_RULES_FUSION
        )

    @pytest.mark.parametrize("seed", range(60))
    def test_research_convex_matches_the_product_fusion(self, synthetic_case, seed):
        inputs = synthetic_case(seed)
        lexical_hits, vector_hits = _product_hits(inputs)
        lexical = [item for item in inputs["lexical"] if item["chunk_id"] is not None]

        research = fuse_hybrid_candidates(inputs["query"], lexical, inputs["vector"], PRODUCT_FUSION)
        product = fuse_hybrid_hits(lexical_hits, vector_hits)

        assert _ids(research) == [hit.chunk_id for hit in product]
        assert [item["score"] for item in research] == [hit.score for hit in product]
        limit = inputs["limit"]
        assert _ids(apply_paper_diversity(research, limit=limit)) == [
            hit.chunk_id for hit in product_diversity(product, limit=limit)
        ]

    def test_channel_tops_beat_a_chunk_that_is_only_middling_in_both(self, make_candidate):
        lexical = [
            make_candidate(1, 11.0, arxiv_id="2608.30597"),
            make_candidate(2, 10.0, arxiv_id="2608.30597"),
            make_candidate(9, 3.8, arxiv_id="2609.16034"),
            make_candidate(4, 1.2, arxiv_id="2609.00004"),
        ]
        vector = [
            make_candidate(5, 0.49, arxiv_id="2608.30597"),
            make_candidate(6, 0.48, arxiv_id="2608.30597"),
            make_candidate(9, 0.40, arxiv_id="2609.16034"),
            make_candidate(7, 0.30, arxiv_id="2609.00007"),
        ]

        assert _ids(_merge("DPO", lexical, vector, limit=3, config=PRODUCT_FUSION))[0] == 5
        assert _ids(_legacy("DPO", lexical, vector, limit=3))[0] == 9


class TestHitConversion:
    def test_diversity_keeps_the_product_rule_for_dict_candidates(self, make_candidate):
        candidates = [make_candidate(index, 0.9, arxiv_id="A") for index in range(1, 5)]
        candidates += [make_candidate(10, 0.8, arxiv_id="B"), make_candidate(None, 0.7, arxiv_id="")]

        assert _ids(apply_paper_diversity(candidates, limit=4)) == [1, 2, 10, None]
        assert _ids(apply_paper_diversity(candidates, limit=10)) == _ids(candidates)
        assert _ids(apply_paper_diversity(candidates, limit=3, arxiv_id="A")) == [1, 2, 3]
        assert apply_paper_diversity([], limit=3) == []

    def test_candidate_from_hit_carries_the_tier_and_scores(self, make_hit):
        lexical = candidate_from_hit(make_hit(3, score=1.4, strict=False))
        vector = candidate_from_hit(make_hit(4, method="vector", score=0.6))

        assert lexical["score_breakdown"]["strict_match"] is False
        assert lexical["score"] == 1.4 and lexical["retrieval_method"] == "lexical"
        assert vector["score_breakdown"]["raw_similarity_score"] == 0.6
        assert "strict_match" not in vector["score_breakdown"]

    def test_fuse_hits_maps_the_research_order_back_to_hits(self, make_hit):
        inputs = HybridInputs(
            query="policy loss",
            lexical=(make_hit(1, score=1.9), make_hit(2, score=1.1, strict=False)),
            vector=(make_hit(2, method="vector", score=0.8), make_hit(3, method="vector", score=0.7)),
        )

        hits = fuse_hits(inputs, LEGACY_RULES_FUSION, limit=5)

        assert [hit.chunk_id for hit in hits] == [2, 1, 3]
        assert [hit.score for hit in hits] == pytest.approx([1 / 61, 1 / 61, 1 / 62])
        assert all(hit.method == "hybrid" for hit in hits)
        assert hits[0].vector == inputs.vector[0].vector
        assert hits[1].lexical == inputs.lexical[0].lexical
        assert fuse_hits(HybridInputs(query="", lexical=(), vector=()), LEGACY_RULES_FUSION, limit=5) == []
