from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path

import pytest

from src.integrations.hybrid_fusion import (
    DEFAULT_HYBRID_FUSION,
    STANDARD_RRF_FUSION,
    STRICT_LEXICAL_SCORE_FLOOR,
    HybridFusionConfig,
    apply_paper_diversity,
    fuse_hybrid_candidates,
    normalize_channel_scores,
    resolve_hybrid_method_weights,
)
from src.integrations.paper_repository import STRICT_MATCH_BONUS
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


def _lexical(chunk_id: int, score: float, *, strict: bool, coverage: float = 1.0) -> dict:
    return {
        **_candidate(chunk_id, score),
        "score_breakdown": {"strict_match": strict, "coverage": coverage},
    }


class TestTwoTierLexicalCandidates:
    def test_strict_rows_are_weighted_without_the_tier_bonus(self):
        retriever = _retriever()

        assert retriever._lexical_confidence(_lexical(1, STRICT_MATCH_BONUS + 0.25, strict=True)) == pytest.approx(0.25)
        assert (
            retriever._candidate_hybrid_quality_weight("lexical", _lexical(1, STRICT_MATCH_BONUS + 0.25, strict=True))
            == 0.4
        )
        assert (
            retriever._candidate_hybrid_quality_weight("lexical", _lexical(1, STRICT_MATCH_BONUS + 0.9, strict=True))
            == 1.0
        )

    def test_partial_rows_have_zero_confidence(self):
        retriever = _retriever()
        partial = _lexical(1, 0.87, strict=False, coverage=0.8)

        assert retriever._lexical_confidence(partial) == 0.0
        assert retriever._candidate_hybrid_quality_weight("lexical", partial) == 0.2

    def test_rows_without_tier_information_keep_their_score(self):
        assert _retriever()._lexical_confidence(_candidate(1, 0.6)) == 0.6

    def test_partial_lexical_rows_are_dropped_when_vector_has_results(self):
        lexical = [_lexical(1, STRICT_MATCH_BONUS + 0.5, strict=True), _lexical(2, 0.8, strict=False, coverage=0.7)]
        vector = [_candidate(2, 0.8), _candidate(3, 0.7)]

        merged = _retriever()._merge_hybrid_candidates("policy loss", lexical, vector, arxiv_id=None, limit=10)

        by_id = {candidate["chunk_id"]: candidate for candidate in merged}
        assert set(by_id) == {1, 2, 3}
        assert by_id[2]["matched_methods"] == ["vector"]
        assert "cross_method_overlap_bonus" not in by_id[2]["score_breakdown"]
        assert by_id[1]["matched_methods"] == ["lexical"]

    def test_partial_lexical_rows_are_kept_without_vector_results(self):
        lexical = [_lexical(1, 0.8, strict=False, coverage=0.7), _lexical(2, 0.6, strict=False, coverage=0.5)]

        merged = _retriever()._merge_hybrid_candidates("policy loss", lexical, [], arxiv_id=None, limit=10)

        assert _ids(merged) == [1, 2]
        assert all(candidate["matched_methods"] == ["lexical"] for candidate in merged)


GOLDEN_PATH = Path(__file__).parent / "fixtures" / "hybrid_fusion_golden.json"
GOLDEN_QUERIES = (
    "policy loss",
    "direct preference optimization policy loss reward",
    "강화학습 보상 모델의 한계는 무엇인가",
    "LoRA",
    "What limitations does the paper discuss about sparse attention kernels",
)
GOLDEN_SCORE_POINTS = (0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 0.95)


def synthetic_fusion_case(seed: int) -> dict:
    """융합 회귀 고정용 합성 입력. 신뢰도 구간 경계값, strict/부분 일치/등급 없음 행, 채널 겹침, chunk_id 없는 행을 섞는다."""
    rng = random.Random(seed)
    pool = list(range(1, 81))
    rng.shuffle(pool)
    papers = [f"2409.{number:05d}" for number in range(rng.randint(3, 12))]

    def row(chunk_id: int, score: float, breakdown: dict, snippet: str) -> dict:
        return {
            **_candidate(chunk_id, score, arxiv_id=papers[chunk_id % len(papers)], snippet=snippet),
            "score_breakdown": breakdown,
        }

    def lexical_row(chunk_id: int) -> dict:
        confidence = rng.choice(GOLDEN_SCORE_POINTS) if rng.random() < 0.5 else round(rng.random(), 4)
        tier = rng.random()
        if tier < 0.6:
            return row(chunk_id, STRICT_MATCH_BONUS + confidence, {"strict_match": True}, f"lexical {chunk_id}")
        if tier < 0.9:
            return row(chunk_id, confidence, {"strict_match": False, "coverage": 0.5}, f"lexical {chunk_id}")
        return row(chunk_id, confidence, {}, f"lexical {chunk_id}")

    def vector_row(chunk_id: int) -> dict:
        snippet = f"vector {chunk_id}" if rng.random() < 0.8 else ""
        return row(chunk_id, round(rng.uniform(0.2, 0.9), 4), {"rerank_adjustment": 0.0}, snippet)

    lexical_ids = pool[: rng.randint(0, 30)]
    shared = rng.randint(0, len(lexical_ids))
    vector_ids = rng.sample(lexical_ids, shared) + pool[30 : 30 + rng.randint(0, 30 - shared)]
    rng.shuffle(vector_ids)
    lexical = sorted((lexical_row(chunk_id) for chunk_id in lexical_ids), key=lambda item: item["score"], reverse=True)
    vector = sorted((vector_row(chunk_id) for chunk_id in vector_ids), key=lambda item: item["score"], reverse=True)
    if lexical and rng.random() < 0.1:
        lexical[-1]["chunk_id"] = None
    return {
        "query": rng.choice(GOLDEN_QUERIES),
        "lexical": lexical,
        "vector": vector,
        "limit": rng.choice((1, 3, 5, 10)),
    }


def fusion_fingerprint(candidates: list[dict]) -> str:
    """후보 목록의 모든 필드(점수 float 포함)를 정렬된 JSON으로 직렬화한 sha256."""
    return hashlib.sha256(json.dumps(candidates, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _golden_cases() -> list[dict]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]


def _legacy_plain_rrf(ranked_lists: list[list[dict]], rank_constant: float = 60.0) -> list[dict]:
    """삭제 전 `eval.runner.plain_rrf` 구현 그대로(비교 기준)."""
    merged: dict[int, dict] = {}
    for candidates in ranked_lists:
        for rank, candidate in enumerate(candidates, start=1):
            chunk_id = int(candidate.get("chunk_id") or 0)
            entry = merged.setdefault(chunk_id, {**candidate, "retrieval_method": "hybrid_plain_rrf", "score": 0.0})
            entry["score"] += 1.0 / (rank_constant + rank)
    return sorted(merged.values(), key=lambda item: (item["score"], int(item.get("chunk_id") or 0)), reverse=True)


class TestFusionRefactorGolden:
    """리팩터링 전(89b100f) `_merge_hybrid_candidates` 출력을 fixtures에 고정하고 기본 설정이 그대로 재현하는지 본다."""

    @pytest.mark.parametrize("case", _golden_cases(), ids=lambda case: f"seed{case['seed']}")
    def test_default_config_reproduces_pre_refactor_output(self, case):
        inputs = synthetic_fusion_case(case["seed"])
        assert (len(inputs["lexical"]), len(inputs["vector"]), inputs["limit"]) == (
            case["n_lexical"],
            case["n_vector"],
            case["limit"],
        )

        fused = fuse_hybrid_candidates(inputs["query"], inputs["lexical"], inputs["vector"], DEFAULT_HYBRID_FUSION)
        assert [candidate["chunk_id"] for candidate in fused] == case["fused_chunk_ids"]
        assert fusion_fingerprint(fused) == case["fused_sha256"]

        merged = _retriever()._merge_hybrid_candidates(
            inputs["query"], inputs["lexical"], inputs["vector"], arxiv_id=None, limit=inputs["limit"]
        )
        assert [candidate["chunk_id"] for candidate in merged] == case["diversified_chunk_ids"]
        assert apply_paper_diversity(fused, limit=inputs["limit"]) == merged

    def test_golden_cases_cover_the_rule_branches(self):
        weights = set()
        for case in _golden_cases():
            inputs = synthetic_fusion_case(case["seed"])
            lexical = [
                candidate
                for candidate in inputs["lexical"]
                if (candidate.get("score_breakdown") or {}).get("strict_match") is not False
            ]
            resolved = resolve_hybrid_method_weights(inputs["query"], lexical, inputs["vector"])
            weights.add((round(resolved["lexical"], 6), round(resolved["vector"], 6)))
        assert len(weights) >= 5


class TestFusionConfig:
    def test_hybrid_search_uses_the_same_inputs_as_hybrid_fusion_inputs(self):
        class Repository:
            def list_chunk_candidates_by_query(self, query, *, limit, arxiv_id=None):
                return [_lexical(1, STRICT_MATCH_BONUS + 0.9, strict=True), _lexical(2, 0.4, strict=False)]

        class Embeddings:
            def embed_texts(self, texts):
                return [[0.0] for _ in texts]

        class Vectors:
            def search_paper_chunks(self, embedding, *, limit, arxiv_id=None):
                return [_candidate(2, 0.8), _candidate(3, 0.7)]

        retriever = PaperRetriever(repository=Repository(), embedding_client=Embeddings(), vector_repository=Vectors())
        query, lexical, vector = retriever.hybrid_fusion_inputs("  policy\x00 loss ", limit=2)

        assert query == "policy loss"
        assert _ids(lexical) == [1, 2]
        assert _ids(vector) == [2, 3]
        expected = apply_paper_diversity(fuse_hybrid_candidates(query, lexical, vector), limit=2)
        assert retriever.search_paper_chunks_by_hybrid("  policy\x00 loss ", limit=2) == expected
        assert retriever.hybrid_fusion_inputs("\x00 ", limit=2) == ("", [], [])

    def test_static_weights_scale_each_channel(self):
        config = HybridFusionConfig(
            weighting="static", static_weights=(0.3, 1.0), quality_weight=False, overlap_bonus=0
        )
        fused = fuse_hybrid_candidates("policy loss", [_candidate(1, 0.9)], [_candidate(2, 0.9)], config)

        by_id = {candidate["chunk_id"]: candidate for candidate in fused}
        assert by_id[1]["score"] == pytest.approx(0.3 / 61)
        assert by_id[2]["score"] == pytest.approx(1.0 / 61)
        assert _ids(fused) == [2, 1]

    @pytest.mark.parametrize(
        ("confidence", "expected_lexical"),
        [(0.0, 0.2), (0.15, 0.2 + 0.8 * 0.5), (0.3, 1.0), (0.9, 1.0)],
    )
    def test_confidence_linear_weight(self, confidence, expected_lexical):
        config = HybridFusionConfig(weighting="confidence_linear", confidence_min_weight=0.2, confidence_tau=0.3)
        lexical = [_lexical(1, STRICT_MATCH_BONUS + confidence, strict=True)]

        weights = resolve_hybrid_method_weights("policy loss", lexical, [_candidate(2, 0.9)], config)

        assert weights == {"lexical": pytest.approx(expected_lexical), "vector": 1.0}

    def test_confidence_linear_without_lexical_results_uses_the_floor(self):
        config = HybridFusionConfig(weighting="confidence_linear", confidence_min_weight=0.4, confidence_tau=0.5)
        assert resolve_hybrid_method_weights("policy loss", [], [_candidate(2, 0.9)], config)["lexical"] == 0.4

    def test_quality_weight_switch(self):
        lexical = [_lexical(1, STRICT_MATCH_BONUS + 0.1, strict=True)]
        on = fuse_hybrid_candidates("policy loss", lexical, [], HybridFusionConfig(weighting="static"))
        off = fuse_hybrid_candidates(
            "policy loss", lexical, [], HybridFusionConfig(weighting="static", quality_weight=False)
        )

        assert on[0]["score"] == pytest.approx(0.2 / 61)
        assert off[0]["score"] == pytest.approx(1.0 / 61)

    def test_rank_constant_and_overlap_bonus_are_configurable(self):
        config = HybridFusionConfig(weighting="static", quality_weight=False, rank_constant=10.0, overlap_bonus=0.5)
        fused = fuse_hybrid_candidates("policy loss", [_candidate(1, 0.9)], [_candidate(1, 0.9)], config)

        assert fused[0]["score"] == pytest.approx(2.0 / 11 + 0.5)
        assert fused[0]["score_breakdown"]["cross_method_overlap_bonus"] == 0.5

    def test_partial_lexical_rows_kept_when_filter_is_off(self):
        lexical = [_lexical(1, 0.8, strict=False)]
        config = HybridFusionConfig(drop_partial_lexical=False)

        fused = fuse_hybrid_candidates("policy loss", lexical, [_candidate(2, 0.9)], config)

        assert set(_ids(fused)) == {1, 2}

    @pytest.mark.parametrize(
        "overrides",
        [{"weighting": "rrf"}, {"rank_constant": 0.0}, {"weighting": "confidence_linear", "confidence_tau": 0.0}],
    )
    def test_invalid_config_is_rejected(self, overrides):
        with pytest.raises(ValueError):
            HybridFusionConfig(**overrides)


class TestStandardRrfPreset:
    """`STANDARD_RRF_FUSION`이 삭제한 `eval.runner.plain_rrf`와 같은 순서·점수를 내는지(동점 규칙 포함) 본다."""

    @pytest.mark.parametrize("seed", range(60))
    def test_matches_legacy_plain_rrf(self, seed):
        inputs = synthetic_fusion_case(seed)
        lexical = [candidate for candidate in inputs["lexical"] if candidate["chunk_id"] is not None]

        fused = fuse_hybrid_candidates(inputs["query"], lexical, inputs["vector"], STANDARD_RRF_FUSION)
        legacy = _legacy_plain_rrf([lexical, inputs["vector"]])

        assert _ids(fused) == _ids(legacy)
        assert [candidate["score"] for candidate in fused] == [candidate["score"] for candidate in legacy]

    def test_ties_break_by_chunk_id_descending(self):
        fused = fuse_hybrid_candidates("policy loss", [_candidate(1, 0.9)], [_candidate(5, 0.9)], STANDARD_RRF_FUSION)
        assert _ids(fused) == [5, 1] == _ids(_legacy_plain_rrf([[_candidate(1, 0.9)], [_candidate(5, 0.9)]]))

    def test_partial_lexical_rows_are_kept_like_the_legacy_ablation(self):
        lexical = [_lexical(1, 0.8, strict=False), _lexical(2, STRICT_MATCH_BONUS + 0.1, strict=True)]
        vector = [_candidate(2, 0.9)]

        fused = fuse_hybrid_candidates("policy loss", lexical, vector, STANDARD_RRF_FUSION)

        assert _ids(fused) == _ids(_legacy_plain_rrf([lexical, vector])) == [2, 1]


def _convex(alpha: float, normalization: str = "minmax", **overrides) -> HybridFusionConfig:
    return HybridFusionConfig(weighting="convex", convex_alpha=alpha, score_normalization=normalization, **overrides)


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

    def test_all_equal_scores_get_one_under_minmax(self):
        assert normalize_channel_scores([0.7, 0.7, 0.7], "minmax") == [1.0, 1.0, 1.0]

    def test_theoretical_with_the_maximum_at_or_below_the_floor_is_degenerate(self):
        assert normalize_channel_scores([0.3, 0.2], "theoretical", STRICT_LEXICAL_SCORE_FLOOR) == [1.0, 1.0]

    def test_empty_channel(self):
        assert normalize_channel_scores([], "minmax") == []

    def test_unknown_normalization_is_rejected(self):
        with pytest.raises(ValueError):
            normalize_channel_scores([1.0], "zscore")

    def test_strict_floor_is_the_bonus_minus_the_lowest_sql_adjustments(self):
        assert STRICT_LEXICAL_SCORE_FLOOR == pytest.approx(STRICT_MATCH_BONUS - 0.24 - 0.08 - 0.12) == 0.56


class TestConvexFusion:
    def test_missing_channel_scores_count_as_zero_and_ties_break_on_method_count(self):
        lexical = [_candidate(1, 2.0), _candidate(2, 1.0)]
        vector = [_candidate(2, 0.9), _candidate(3, 0.5)]

        fused = fuse_hybrid_candidates("policy loss", lexical, vector, _convex(0.5))

        # 1: 0.5 × 1 + 0.5 × 0(결측), 2: 0.5 × 0 + 0.5 × 1, 3: 0.5 × 0(결측) + 0.5 × 0
        assert [candidate["score"] for candidate in fused] == pytest.approx([0.5, 0.5, 0.0])
        assert _ids(fused) == [2, 1, 3]

    def test_combination_weights_each_channel_by_alpha(self):
        lexical = [_candidate(1, 3.0), _candidate(2, 1.0)]
        vector = [_candidate(2, 0.9), _candidate(3, 0.6), _candidate(4, 0.3)]

        fused = fuse_hybrid_candidates("policy loss", lexical, vector, _convex(0.3))

        by_id = {candidate["chunk_id"]: candidate["score"] for candidate in fused}
        assert by_id == pytest.approx({1: 0.3, 2: 0.7, 3: 0.35, 4: 0.0})
        assert _ids(fused) == [2, 3, 1, 4]

    def test_theoretical_normalization_uses_the_configured_floors(self):
        lexical = [_candidate(1, 2.0), _candidate(2, 1.0)]
        vector = [_candidate(2, 0.5), _candidate(3, 0.2)]
        config = _convex(0.4, "theoretical", score_floors=(0.0, -1.0))

        fused = fuse_hybrid_candidates("policy loss", lexical, vector, config)

        by_id = {candidate["chunk_id"]: candidate["score"] for candidate in fused}
        assert by_id == pytest.approx({1: 0.4, 2: 0.4 * 0.5 + 0.6, 3: 0.6 * 1.2 / 1.5})

    def test_empty_lexical_channel_keeps_the_vector_order(self):
        vector = [_candidate(5, 0.9), _candidate(3, 0.7), _candidate(9, 0.2)]

        fused = fuse_hybrid_candidates("policy loss", [], vector, _convex(0.3))

        assert _ids(fused) == [5, 3, 9]
        assert [candidate["score"] for candidate in fused] == pytest.approx([0.7, 0.7 * 0.5 / 0.7, 0.0])

    def test_alpha_endpoints_follow_one_channel(self):
        lexical = [_candidate(1, 2.0), _candidate(2, 1.5), _candidate(3, 1.0)]
        vector = [_candidate(3, 0.9), _candidate(2, 0.6), _candidate(4, 0.3)]

        assert _ids(fuse_hybrid_candidates("q", lexical, vector, _convex(0.0)))[:3] == [3, 2, 4]
        assert _ids(fuse_hybrid_candidates("q", lexical, vector, _convex(1.0)))[:3] == [1, 2, 3]

    def test_single_lexical_candidate_after_dropping_partial_rows_is_degenerate(self):
        lexical = [_lexical(1, STRICT_MATCH_BONUS + 0.2, strict=True), _lexical(2, 0.9, strict=False)]
        vector = [_candidate(3, 0.9), _candidate(4, 0.5)]

        fused = fuse_hybrid_candidates("policy loss", lexical, vector, _convex(0.5))

        assert _ids(fused) == [3, 1, 4]
        assert fused[1]["score_breakdown"]["lexical_normalized_score"] == 1.0
        assert 2 not in _ids(fused)

    def test_partial_rows_join_the_normalization_when_the_filter_is_off(self):
        lexical = [_lexical(1, STRICT_MATCH_BONUS + 0.2, strict=True), _lexical(2, 0.2, strict=False)]
        vector = [_candidate(3, 0.9)]

        fused = fuse_hybrid_candidates("policy loss", lexical, vector, _convex(0.5, drop_partial_lexical=False))

        by_id = {candidate["chunk_id"]: candidate for candidate in fused}
        assert by_id[2]["score"] == 0.0
        assert by_id[1]["score"] == pytest.approx(0.5)

    def test_rrf_only_terms_are_not_applied(self):
        fused = fuse_hybrid_candidates("q", [_candidate(1, 2.0)], [_candidate(1, 0.9)], _convex(0.5, overlap_bonus=0.5))

        assert fused[0]["score"] == pytest.approx(1.0)
        breakdown = fused[0]["score_breakdown"]
        assert "cross_method_overlap_bonus" not in breakdown
        assert "lexical_rrf_score" not in breakdown and "lexical_quality_weight" not in breakdown
        assert breakdown["lexical_normalized_score"] == breakdown["vector_normalized_score"] == 1.0
        assert breakdown["lexical_weight"] == breakdown["vector_weight"] == 0.5

    @pytest.mark.parametrize("seed", range(20))
    def test_convex_fields_do_not_change_rank_fusion(self, seed):
        inputs = synthetic_fusion_case(seed)
        tweaked = HybridFusionConfig(convex_alpha=0.9, score_normalization="theoretical", score_floors=(0.3, 0.0))

        assert fuse_hybrid_candidates(inputs["query"], inputs["lexical"], inputs["vector"], tweaked) == (
            fuse_hybrid_candidates(inputs["query"], inputs["lexical"], inputs["vector"])
        )

    def test_product_default_is_still_the_rule_based_rrf(self):
        assert DEFAULT_HYBRID_FUSION.weighting == "rules"
        assert DEFAULT_HYBRID_FUSION == HybridFusionConfig()

    @pytest.mark.parametrize(
        "overrides",
        [{"convex_alpha": 1.5}, {"convex_alpha": -0.1}, {"score_normalization": "zscore"}],
    )
    def test_invalid_convex_config_is_rejected(self, overrides):
        with pytest.raises(ValueError):
            HybridFusionConfig(weighting="convex", **overrides)
