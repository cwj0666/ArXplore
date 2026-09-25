from __future__ import annotations

import pytest

from src.integrations.paper_retriever import PaperRetriever

BODY = (
    "The policy is trained to prefer chosen responses over rejected ones using a simple "
    "classification loss on pairwise comparisons. This removes the need for a separate reward model."
)


def _retriever() -> PaperRetriever:
    return PaperRetriever(repository=object(), embedding_client=object(), vector_repository=object())


def _candidate(
    chunk_id: int,
    *,
    text: str = BODY,
    section_title: str = "Method",
    content_role: str = "body",
    score: float = 0.5,
    arxiv_id: str | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "arxiv_id": arxiv_id if arxiv_id is not None else f"2401.{chunk_id:05d}",
        "chunk_index": 0,
        "chunk_text": text,
        "section_title": section_title,
        "content_role": content_role,
        "score": score,
    }


def _ids(candidates: list[dict]) -> list[int]:
    return [candidate["chunk_id"] for candidate in candidates]


class TestReferenceLikeText:
    @pytest.mark.parametrize(
        "text",
        [
            "as shown in [1], [2] and [3] the method scales",
            "prior work [4] in 2019 and [5] in 2021 studied this",
            "Proceedings of the Conference on Vision 2020, and ICLR 2021",
            "Smith, J., Doe, A. Scaling laws. CVPR 2020.",
        ],
        ids=["three-bracket-cites", "two-cites-two-years", "two-venues-two-years", "author-list-year-venue"],
    )
    def test_detected(self, text):
        assert PaperRetriever._looks_reference_like_text(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            BODY,
            "",
            "We follow the setup of [1] published in 2020.",
            "Results at NeurIPS 2022 show gains.",
        ],
        ids=["plain-body", "empty", "one-cite-one-year", "one-venue-one-year"],
    )
    def test_not_detected(self, text):
        assert PaperRetriever._looks_reference_like_text(text) is False

    def test_only_first_1200_chars_are_inspected(self):
        tail = " [1] [2] [3]"
        assert PaperRetriever._looks_reference_like_text("x" * 1100 + tail) is True
        assert PaperRetriever._looks_reference_like_text("x" * 1200 + tail) is False


class TestFilterLexicalCandidates:
    QUERY = "how does the policy loss work"

    def test_drops_non_body_roles_titles_and_noisy_text(self):
        candidates = [
            _candidate(1),
            _candidate(2, content_role="references"),
            _candidate(3, content_role="front_matter"),
            _candidate(4, section_title="References"),
            _candidate(5, section_title="Front Matter"),
            _candidate(6, text="See [1], [2], [3] for details."),
            _candidate(7, text="The rest of this paper is organized as follows. Section 2 presents the method."),
            _candidate(8, content_role="table_like"),
            _candidate(9, section_title="3.2 Reference Model"),
        ]

        filtered = _retriever()._filter_lexical_candidates(self.QUERY, candidates)

        assert _ids(filtered) == [1, 8, 9]

    @pytest.mark.parametrize(
        "text",
        [
            "The remainder of this paper is structured as below.",
            "This paper is organized as follows.",
            "We conclude in Section 6.",
            "Section 4 describes the experiments.",
        ],
    )
    def test_outline_like_text(self, text):
        assert PaperRetriever._looks_outline_like_text(text) is True

    def test_outline_detection_is_whitespace_and_case_insensitive(self):
        assert PaperRetriever._looks_outline_like_text("THE REST\nOF THIS   PAPER is ...") is True
        assert PaperRetriever._looks_outline_like_text("Section 6 presents results.") is False

    @pytest.mark.parametrize(
        "query",
        ["list the references", "show the bibliography", "which works cited this", "Reference list please"],
    )
    def test_reference_intent_returns_everything_unfiltered(self, query):
        candidates = [
            _candidate(1),
            _candidate(2, content_role="references"),
            _candidate(3, content_role="front_matter"),
            _candidate(4, text="The rest of this paper is organized as follows."),
        ]

        assert _retriever()._filter_lexical_candidates(query, candidates) == candidates

    @pytest.mark.parametrize("query", ["referenced baseline", "cite count"])
    def test_near_miss_queries_keep_filter_on(self, query):
        candidates = [_candidate(1), _candidate(2, content_role="references")]

        assert _ids(_retriever()._filter_lexical_candidates(query, candidates)) == [1]

    def test_standalone_reference_word_disables_filter_even_for_reference_model(self):
        candidates = [_candidate(1), _candidate(2, content_role="references")]

        assert _ids(_retriever()._filter_lexical_candidates("reference model loss", candidates)) == [1, 2]


class TestApplyPaperDiversity:
    def test_caps_two_chunks_per_paper_before_overflow(self):
        candidates = [
            _candidate(1, arxiv_id="A"),
            _candidate(2, arxiv_id="A"),
            _candidate(3, arxiv_id="A"),
            _candidate(4, arxiv_id="B"),
            _candidate(5, arxiv_id="C"),
        ]

        selected = _retriever()._apply_paper_diversity(candidates, limit=4, arxiv_id=None)

        assert _ids(selected) == [1, 2, 4, 5]

    def test_refills_from_overflow_in_original_order(self):
        candidates = [
            _candidate(1, arxiv_id="A"),
            _candidate(2, arxiv_id="A"),
            _candidate(3, arxiv_id="A"),
            _candidate(4, arxiv_id="A"),
            _candidate(5, arxiv_id="B"),
        ]

        selected = _retriever()._apply_paper_diversity(candidates, limit=4, arxiv_id=None)

        assert _ids(selected) == [1, 2, 5, 3]

    def test_single_paper_scope_skips_diversity(self):
        candidates = [_candidate(i, arxiv_id="A") for i in range(1, 6)]

        selected = _retriever()._apply_paper_diversity(candidates, limit=4, arxiv_id="A")

        assert _ids(selected) == [1, 2, 3, 4]

    def test_short_lists_are_returned_as_is(self):
        candidates = [_candidate(i, arxiv_id="A") for i in range(1, 4)]

        assert _ids(_retriever()._apply_paper_diversity(candidates, limit=3, arxiv_id=None)) == [1, 2, 3]

    def test_missing_arxiv_id_is_not_capped(self):
        candidates = [_candidate(i, arxiv_id="") for i in range(1, 5)] + [_candidate(9, arxiv_id="B")]

        assert _ids(_retriever()._apply_paper_diversity(candidates, limit=4, arxiv_id=None)) == [1, 2, 3, 4]

    def test_limit_below_one_is_clamped(self):
        candidates = [_candidate(1, arxiv_id="A"), _candidate(2, arxiv_id="B")]

        assert _ids(_retriever()._apply_paper_diversity(candidates, limit=0, arxiv_id=None)) == [1]


class TestRerankVectorCandidates:
    NEUTRAL_QUERY = "zzqx"

    def _adjustment(self, query: str, **candidate_kwargs) -> float:
        reranked = _retriever()._rerank_vector_candidates(query, [_candidate(1, score=0.5, **candidate_kwargs)])
        return reranked[0]["rerank_adjustment"]

    @pytest.mark.parametrize(
        ("section_title", "content_role", "expected"),
        [
            ("Method", "body", 0.0),
            ("Appendix A", "body", -0.08),
            ("Implementation Details", "body", -0.08),
            ("Conclusion", "body", -0.03),
            ("Limitations", "body", -0.03),
            ("References", "body", -0.18),
            ("Acknowledgments", "body", -0.18),
            ("Method", "front_matter", -0.02),
            ("Method", "table_like", -0.02),
            ("Appendix B: Discussion", "table_like", -0.08 - 0.03 - 0.02),
        ],
    )
    def test_section_penalties(self, section_title, content_role, expected):
        adjustment = self._adjustment(self.NEUTRAL_QUERY, section_title=section_title, content_role=content_role)

        assert adjustment == pytest.approx(expected)

    def test_reference_like_text_penalty_stacks_with_title_penalty(self):
        adjustment = self._adjustment(self.NEUTRAL_QUERY, section_title="References", text="[1] a [2] b [3] c")

        assert adjustment == pytest.approx(-0.18 - 0.14)

    def test_requested_sections_are_not_penalized(self):
        assert self._adjustment("appendix zzqx", section_title="Appendix A") == pytest.approx(0.03)
        assert self._adjustment("list citations", section_title="References") == pytest.approx(0.0)
        assert self._adjustment("zzqx discussion", section_title="Discussion") == pytest.approx(0.03 + 0.1)

    def test_section_intent_bonus(self):
        assert self._adjustment("what are the limitations", section_title="Limitations") == pytest.approx(0.17)
        assert self._adjustment("future work", section_title="Future Directions") == pytest.approx(0.12 + 0.03)

    def test_overlap_bonus_is_capped(self):
        text = "alpha bravo charlie delta echo foxtrot"
        adjustment = self._adjustment("alpha bravo charlie delta echo foxtrot", text=text)

        assert adjustment == pytest.approx(0.12)

    def test_scores_and_order(self):
        candidates = [
            _candidate(1, score=0.6, section_title="References"),
            _candidate(2, score=0.5, section_title="Method"),
            _candidate(3, score=0.5, section_title="Results"),
        ]

        reranked = _retriever()._rerank_vector_candidates(self.NEUTRAL_QUERY, candidates)

        assert _ids(reranked) == [3, 2, 1]
        assert reranked[2]["score"] == pytest.approx(0.6 - 0.18)
        assert reranked[2]["similarity_score"] == reranked[2]["score"]
        assert reranked[2]["score_breakdown"]["rerank_adjustment"] == pytest.approx(-0.18)


class TestSearchSnippet:
    def test_centers_on_first_matching_term(self):
        text = "a" * 200 + " policy " + "b" * 400

        snippet = PaperRetriever._build_search_snippet("policy", text, "", "")

        index = text.find("policy")
        start = index - 280 // 3
        assert snippet == "..." + text[start : start + 280].strip() + "..."

    def test_no_ellipsis_when_whole_text_fits(self):
        assert PaperRetriever._build_search_snippet("policy", "the policy loss", "", "") == "the policy loss"

    def test_falls_back_to_abstract_then_title(self):
        assert PaperRetriever._build_search_snippet("reward", "no match", "reward model", "T") == "reward model"
        assert PaperRetriever._build_search_snippet("scaling", "no match", "none", "Scaling Laws") == "Scaling Laws"

    def test_short_terms_are_ignored_and_first_nonempty_text_is_compacted(self):
        text = "line one\n\n  line   two " + "x" * 400

        snippet = PaperRetriever._build_search_snippet("rl is ok", text, "abstract", "title")

        compact = " ".join(text.split())
        assert snippet == compact[:280] + "..."

    def test_all_empty_returns_empty_string(self):
        assert PaperRetriever._build_search_snippet("policy", "", "", "") == ""


class TestQueryTokens:
    def test_lowercases_and_drops_short_tokens_and_stopwords(self):
        tokens = PaperRetriever._query_tokens("The RL policy and the DPO loss for LLMs with 7B")

        assert tokens == {"policy", "dpo", "loss", "llms"}

    def test_korean_query_yields_no_tokens(self):
        assert PaperRetriever._query_tokens("강화학습에서 보상 모델은 어떻게 학습되나요") == set()

    def test_mixed_query_keeps_only_ascii_tokens(self):
        assert PaperRetriever._query_tokens("RLHF 보상 모델 DPO 비교") == {"rlhf", "dpo"}

    def test_korean_query_gets_no_overlap_bonus(self):
        candidate = _candidate(1, text="강화학습 보상 모델", section_title="보상 모델")

        reranked = _retriever()._rerank_vector_candidates("강화학습 보상 모델", [candidate])

        assert reranked[0]["rerank_adjustment"] == 0.0
