import pytest

from arxplore.domain.retrieval import RetrievalHit
from arxplore.retrieval.lexical import (
    build_lexical_candidates_query,
    escape_like,
    filter_lexical_hits,
    rerank_lexical_hits,
)


def _ids(hits: list[RetrievalHit]) -> list[int]:
    return [hit.chunk_id for hit in hits]


class TestFilterLexicalHits:
    QUERY = "how does the policy loss work"

    def test_drops_non_body_roles_titles_and_noisy_text(self, make_hit) -> None:
        hits = [
            make_hit(1),
            make_hit(2, content_role="references"),
            make_hit(3, content_role="front_matter"),
            make_hit(4, section_title="References"),
            make_hit(5, section_title="Front Matter"),
            make_hit(6, text="[1] A. Smith. 2019.\n[2] B. Jones. 2020.\n[3] C. Lee. 2021."),
            make_hit(7, text="The rest of this paper is organized as follows. Section 2 presents the method."),
            make_hit(8, content_role="table_like"),
            make_hit(9, section_title="3.2 Reference Model"),
            make_hit(10, text="See [1], [2], [3] for details on the policy loss."),
            make_hit(11, section_title="7 References"),
            make_hit(12, section_title="Preference Optimization"),
        ]

        assert _ids(filter_lexical_hits(self.QUERY, hits)) == [1, 8, 9, 10, 12]

    @pytest.mark.parametrize(
        "query",
        ["list the references", "show the bibliography", "which works cited this", "Reference list please"],
    )
    def test_reference_intent_keeps_everything(self, make_hit, query: str) -> None:
        hits = [
            make_hit(1),
            make_hit(2, content_role="references"),
            make_hit(3, content_role="front_matter"),
            make_hit(4, text="The rest of this paper is organized as follows."),
            make_hit(5, section_title="References"),
        ]

        assert filter_lexical_hits(query, hits) == hits

    @pytest.mark.parametrize("query", ["referenced baseline", "cite count", "list citations"])
    def test_near_miss_queries_keep_the_filter_on(self, make_hit, query: str) -> None:
        hits = [make_hit(1), make_hit(2, content_role="references")]

        assert _ids(filter_lexical_hits(query, hits)) == [1]

    def test_the_word_reference_alone_turns_the_filter_off(self, make_hit) -> None:
        hits = [make_hit(1), make_hit(2, content_role="references")]

        assert _ids(filter_lexical_hits("reference model loss", hits)) == [1, 2]

    def test_filter_does_not_refill_below_the_limit(self, make_hit) -> None:
        hits = [make_hit(1, content_role="references"), make_hit(2, content_role="front_matter")]

        assert filter_lexical_hits("policy loss", hits) == []


class TestRerankLexicalHits:
    def test_section_intent_bonus_is_added_and_recorded(self, make_hit) -> None:
        hits = [make_hit(1, score=1.2, section_title="Method"), make_hit(2, score=1.1, section_title="6 Limitations")]

        reranked = rerank_lexical_hits("what are the limitations", hits)

        assert _ids(reranked) == [2, 1]
        assert reranked[0].score == 1.1 + 0.14
        assert reranked[0].lexical is not None
        assert reranked[0].lexical.section_intent_bonus == 0.14
        assert reranked[1].score == 1.2
        assert reranked[1].lexical is not None
        assert reranked[1].lexical.section_intent_bonus == 0.0

    def test_ties_break_on_chunk_id_descending(self, make_hit) -> None:
        hits = [make_hit(3, score=1.0), make_hit(9, score=1.0), make_hit(5, score=1.0)]

        assert _ids(rerank_lexical_hits("policy loss", hits)) == [9, 5, 3]


class TestLexicalQueryBuilder:
    def test_empty_query_builds_nothing(self) -> None:
        assert build_lexical_candidates_query("  \n ", limit=5) is None

    def test_parameters_for_a_global_search(self) -> None:
        built = build_lexical_candidates_query("  multi-head/self   attention ", limit=75)

        assert built is not None
        sql, params = built
        assert params == {
            "query": "multi-head/self attention",
            "fts_query": "multi head self attention",
            "like_pattern": "%multi-head/self attention%",
            "limit": 75,
            "partial_pool": 1500,
            "per_paper_cap": 3,
        }
        assert "WHERE paper_rank <= %(per_paper_cap)s" in sql
        assert "%(arxiv_id)s" not in sql

    def test_scoped_search_has_no_per_paper_cap(self) -> None:
        built = build_lexical_candidates_query("attention", limit=5, arxiv_id="2401.00001")

        assert built is not None
        sql, params = built
        assert params["arxiv_id"] == "2401.00001"
        assert params["partial_pool"] == 500
        assert "per_paper_cap" not in params
        assert "paper_rank <=" not in sql

    def test_cap_can_be_turned_off(self) -> None:
        built = build_lexical_candidates_query("attention", limit=5, per_paper_cap=None)

        assert built is not None
        assert "per_paper_cap" not in built[1]

    @pytest.mark.parametrize(
        ("raw", "escaped"),
        [("50%_off", "50\\%\\_off"), ("a\\b", "a\\\\b"), ("plain", "plain")],
    )
    def test_like_patterns_escape_wildcards(self, raw: str, escaped: str) -> None:
        assert escape_like(raw) == escaped
