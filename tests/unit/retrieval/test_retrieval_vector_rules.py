import pytest

from arxplore.domain.retrieval import RetrievalHit
from arxplore.retrieval.vector import (
    build_vector_search_query,
    hnsw_ef_search_statement,
    lexical_overlap_bonus,
    rerank_vector_hits,
    resolve_candidate_limit,
    resolve_hnsw_ef_search,
)

NEUTRAL_QUERY = "zzqx"


def _adjustment(make_hit, query: str, **hit_options: object) -> float:
    reranked = rerank_vector_hits(query, [make_hit(1, method="vector", score=0.5, **hit_options)])
    assert reranked[0].vector is not None
    return reranked[0].vector.rerank_adjustment


class TestRerankVectorHits:
    @pytest.mark.parametrize(
        ("section_title", "content_role", "expected"),
        [
            ("Method", "body", 0.0),
            ("Appendix A", "body", -0.08),
            ("Implementation Details", "body", -0.08),
            ("Conclusion", "body", -0.03),
            ("Limitations", "body", -0.03),
            ("References", "body", -0.18),
            ("7 References", "body", -0.18),
            ("Acknowledgments", "body", -0.18),
            ("3.2 Reference Model", "body", 0.0),
            ("Method", "front_matter", -0.02),
            ("Method", "table_like", -0.02),
            ("Appendix B: Discussion", "table_like", -0.08 - 0.03 - 0.02),
        ],
    )
    def test_section_and_role_penalties(self, make_hit, section_title: str, content_role: str, expected: float) -> None:
        adjustment = _adjustment(make_hit, NEUTRAL_QUERY, section_title=section_title, content_role=content_role)

        assert adjustment == pytest.approx(expected)

    def test_reference_like_text_penalty_stacks_with_the_title_penalty(self, make_hit) -> None:
        adjustment = _adjustment(make_hit, NEUTRAL_QUERY, section_title="References", text="[1] a\n[2] b\n[3] c")

        assert adjustment == pytest.approx(-0.18 - 0.14)

    def test_requested_sections_are_not_penalized(self, make_hit) -> None:
        assert _adjustment(make_hit, "appendix zzqx", section_title="Appendix A") == pytest.approx(0.03)
        assert _adjustment(make_hit, "list citations", section_title="References") == pytest.approx(0.0)
        assert _adjustment(make_hit, "zzqx discussion", section_title="Discussion") == pytest.approx(0.03 + 0.1)

    def test_section_intent_bonus(self, make_hit) -> None:
        assert _adjustment(make_hit, "what are the limitations", section_title="Limitations") == pytest.approx(0.17)
        assert _adjustment(make_hit, "future work", section_title="Future Directions") == pytest.approx(0.12 + 0.03)

    def test_overlap_bonus_is_capped(self, make_hit) -> None:
        text = "alpha bravo charlie delta echo foxtrot"

        assert _adjustment(make_hit, text, text=text) == pytest.approx(0.12)

    def test_korean_query_gets_no_overlap_bonus(self, make_hit) -> None:
        assert _adjustment(make_hit, "강화학습 보상 모델", text="강화학습 보상 모델", section_title="보상 모델") == 0.0

    def test_scores_and_order(self, make_hit) -> None:
        hits = [
            make_hit(1, method="vector", score=0.6, section_title="References"),
            make_hit(2, method="vector", score=0.5, section_title="Method"),
            make_hit(3, method="vector", score=0.5, section_title="Results"),
        ]

        reranked = rerank_vector_hits(NEUTRAL_QUERY, hits)

        assert [hit.chunk_id for hit in reranked] == [3, 2, 1]
        assert reranked[2].score == pytest.approx(0.6 - 0.18)
        assert reranked[2].vector is not None
        assert reranked[2].vector.rerank_adjustment == pytest.approx(-0.18)
        assert reranked[2].vector.raw_similarity_score == 0.6

    def test_vector_rerank_only_penalizes_and_never_drops(self, make_hit) -> None:
        hits: list[RetrievalHit] = [
            make_hit(1, method="vector", content_role="references", section_title="References"),
            make_hit(2, method="vector", content_role="front_matter", section_title="Front Matter"),
        ]

        assert len(rerank_vector_hits(NEUTRAL_QUERY, hits)) == 2


class TestLexicalOverlapBonus:
    def test_three_hundredths_per_shared_token(self) -> None:
        assert lexical_overlap_bonus(frozenset({"policy", "loss"}), "Method: the policy loss") == pytest.approx(0.06)

    def test_no_tokens_or_no_overlap(self) -> None:
        assert lexical_overlap_bonus(frozenset(), "policy") == 0.0
        assert lexical_overlap_bonus(frozenset({"policy"}), "") == 0.0
        assert lexical_overlap_bonus(frozenset({"policy"}), "reward") == 0.0


class TestVectorQueryBuilder:
    @pytest.mark.parametrize(
        ("limit", "scoped", "expected"),
        [(75, False, 600), (5, False, 80), (150, False, 1200), (5, True, 40), (15, True, 60), (0, False, 80)],
    )
    def test_candidate_limits(self, limit: int, scoped: bool, expected: int) -> None:
        assert resolve_candidate_limit(limit, scoped=scoped) == expected

    @pytest.mark.parametrize(("candidates", "expected"), [(40, None), (41, 41), (600, 600), (1200, 1000)])
    def test_ef_search_grows_with_the_candidate_count(self, candidates: int, expected: int | None) -> None:
        assert resolve_hnsw_ef_search(candidates) == expected

    def test_ef_search_statement_is_transaction_local_and_skipped_for_paper_scoped_searches(self) -> None:
        assert hnsw_ef_search_statement(600) == ("SELECT set_config('hnsw.ef_search', %s, true)", ("600",))
        assert hnsw_ef_search_statement(1200) == ("SELECT set_config('hnsw.ef_search', %s, true)", ("1000",))
        assert hnsw_ef_search_statement(40) is None
        assert hnsw_ef_search_statement(600, arxiv_id="2401.00001") is None

    def test_query_vector_is_sent_as_a_twelve_digit_literal(self) -> None:
        _, params = build_vector_search_query([0.5, -1.0, 1e-13, 0.123456789012345], limit=5)

        assert params["query"] == "[0.500000000000,-1.000000000000,0.000000000000,0.123456789012]"

    def test_global_search_parameters(self) -> None:
        sql, params = build_vector_search_query([0.1, 0.2], limit=75, model_name="text-embedding-3-large")

        assert params == {
            "query": "[0.100000000000,0.200000000000]",
            "candidate_limit": 600,
            "limit": 75,
            "per_paper_cap": 3,
            "model_name": "text-embedding-3-large",
        }
        assert "WHERE e.model_name = %(model_name)s" in sql
        assert "raw_similarity_score >= %(min_similarity)s" not in sql

    def test_scoped_search_sorts_exactly_without_a_cap(self) -> None:
        sql, params = build_vector_search_query([0.1], limit=15, arxiv_id="2401.00001", min_similarity=0.3)

        assert params["candidate_limit"] == 60
        assert params["min_similarity"] == 0.3
        assert "per_paper_cap" not in params
        assert "ORDER BY distance ASC, chunk_id DESC" in sql
        assert "AND e.model_name" not in sql
