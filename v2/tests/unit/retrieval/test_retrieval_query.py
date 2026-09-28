import pytest

from arxplore.retrieval.query import (
    candidate_fetch_limit,
    citation_intent_requested,
    hybrid_branch_limit,
    normalize_search_query,
    query_tokens,
    reference_intent_requested,
    requested_section_intents,
    section_intent_bonus,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("RAG\x00에서 retriever\x07의 역할은?", "RAG에서 retriever의 역할은?"),
        ("  policy\n\tloss  ", "policy loss"),
        ("a\x1fb\x7fc\x0bd", "abcd"),
        ("\x00\x01 \x02", ""),
        ("", ""),
    ],
)
def test_normalization_strips_control_characters_and_collapses_whitespace(raw: str, expected: str) -> None:
    assert normalize_search_query(raw) == expected


@pytest.mark.parametrize(("limit", "expected"), [(1, 30), (5, 30), (6, 30), (7, 35), (10, 50), (0, 30), (-3, 30)])
def test_channel_fetch_limit_is_five_per_result_with_a_floor_of_thirty(limit: int, expected: int) -> None:
    assert candidate_fetch_limit(limit) == expected


@pytest.mark.parametrize(("limit", "expected"), [(1, 10), (3, 10), (4, 12), (5, 15), (10, 30), (0, 10)])
def test_hybrid_branch_limit_is_three_per_result_with_a_floor_of_ten(limit: int, expected: int) -> None:
    assert hybrid_branch_limit(limit) == expected


class TestQueryTokens:
    def test_lowercases_and_drops_short_tokens_and_stopwords(self) -> None:
        assert query_tokens("The RL policy and the DPO loss for LLMs with 7B") == {"policy", "dpo", "loss", "llms"}

    def test_korean_query_yields_no_tokens(self) -> None:
        assert query_tokens("강화학습에서 보상 모델은 어떻게 학습되나요") == set()

    def test_mixed_query_keeps_only_ascii_tokens(self) -> None:
        assert query_tokens("RLHF 보상 모델 DPO 비교") == {"rlhf", "dpo"}


class TestReferenceIntent:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("show me the references", True),
            ("which bibliography entries", True),
            ("which works cited this", True),
            ("Reference list please", True),
            ("list the cited works", True),
            ("direct preference optimization", False),
            ("preferences of annotators", False),
            ("referenced baseline", False),
            ("cite count", False),
            ("reference model loss", True),
        ],
    )
    def test_uses_word_boundaries(self, query: str, expected: bool) -> None:
        assert reference_intent_requested(query) is expected

    @pytest.mark.parametrize(
        ("query", "expected"),
        [("list citations", True), ("citation graph", True), ("the references", True), ("citational", False)],
    )
    def test_citation_words_also_count_for_the_vector_channel(self, query: str, expected: bool) -> None:
        assert citation_intent_requested(query) is expected
        assert reference_intent_requested(query) is (query == "the references")


class TestSectionIntent:
    def test_only_rules_named_in_the_query_are_active(self) -> None:
        intents = requested_section_intents("What limitations and future work are discussed?")

        assert [intent.bonus for intent in intents] == [0.14, 0.12]
        assert section_intent_bonus(intents, "6 Limitations") == 0.14
        assert section_intent_bonus(intents, "Future Directions") == 0.12
        assert section_intent_bonus(intents, "Discussion") == 0.0

    def test_first_matching_rule_wins(self) -> None:
        intents = requested_section_intents("conclusion and limitations")

        assert section_intent_bonus(intents, "Conclusion and Limitations") == 0.14

    def test_plain_query_gives_no_bonus(self) -> None:
        assert requested_section_intents("policy loss") == ()
        assert section_intent_bonus((), "Limitations") == 0.0

    def test_section_titles_are_matched_case_insensitively(self) -> None:
        intents = requested_section_intents("DISCUSSION")

        assert section_intent_bonus(intents, "5 DISCUSSION") == 0.1
