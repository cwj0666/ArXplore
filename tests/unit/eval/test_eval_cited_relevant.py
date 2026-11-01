from typing import Any

import pytest

from arxplore_eval.behavior import (
    ANSWER_CHECK_DESCRIPTIONS,
    ANSWER_CHECK_METRICS,
    OUTCOME_ANSWERED,
    OUTCOME_ERROR,
    OUTCOME_REFUSED,
    OUTCOME_REJECTED,
    answer_checks,
    cited_arxiv_ids,
)

KO_ANSWER = "이 논문은 선호 최적화를 다룹니다."


def _record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "mode": "agent",
        "lang": "ko",
        "answer": KO_ANSWER,
        "relevant_arxiv_ids": ["2401.00001"],
        "citations": [],
    }
    record.update(overrides)
    return record


def _citation(arxiv_id: str, *, in_answer: bool = True) -> dict[str, Any]:
    return {"arxiv_id": arxiv_id, "title": "t", "url": f"https://arxiv.org/abs/{arxiv_id}", "in_answer": in_answer}


def _cited_relevant(record: dict[str, Any], outcome: str = OUTCOME_ANSWERED) -> float | None:
    return answer_checks(record, outcome)[1].get("cited_relevant")


class TestCitedRelevant:
    def test_registered_as_a_reported_metric(self) -> None:
        assert "cited_relevant" in ANSWER_CHECK_METRICS
        assert "cited_relevant" in ANSWER_CHECK_DESCRIPTIONS

    def test_in_answer_citation_of_a_relevant_paper_scores_one(self) -> None:
        assert _cited_relevant(_record(citations=[_citation("2401.00001")])) == 1.0

    def test_citation_not_in_answer_does_not_count(self) -> None:
        assert _cited_relevant(_record(citations=[_citation("2401.00001", in_answer=False)])) == 0.0

    def test_irrelevant_citation_scores_zero(self) -> None:
        assert _cited_relevant(_record(citations=[_citation("2401.99999")])) == 0.0

    def test_any_relevant_id_among_several_is_enough(self) -> None:
        record = _record(
            relevant_arxiv_ids=["2401.00001", "2401.00002"],
            citations=[_citation("2401.99999"), _citation("2401.00002")],
        )
        assert _cited_relevant(record) == 1.0

    def test_version_suffix_is_stripped_on_citation_and_relevant_ids(self) -> None:
        assert _cited_relevant(_record(citations=[_citation("2401.00001v3")])) == 1.0
        record = _record(relevant_arxiv_ids=["2401.00001v2"], citations=[_citation("2401.00001")])
        assert _cited_relevant(record) == 1.0

    def test_arxiv_link_in_the_answer_text_counts(self) -> None:
        answer = f"{KO_ANSWER} [논문](https://arxiv.org/abs/2401.00001)"
        assert _cited_relevant(_record(answer=answer)) == 1.0

    def test_pdf_and_versioned_links_count(self) -> None:
        for link in ("https://arxiv.org/pdf/2401.00001v2", "https://arxiv.org/abs/2401.00001v2"):
            assert _cited_relevant(_record(answer=f"{KO_ANSWER} {link}")) == 1.0

    def test_link_to_an_irrelevant_paper_scores_zero(self) -> None:
        answer = f"{KO_ANSWER} [논문](https://arxiv.org/abs/2401.99999)"
        assert _cited_relevant(_record(answer=answer)) == 0.0

    def test_bare_id_without_a_link_does_not_count(self) -> None:
        assert _cited_relevant(_record(answer=f"{KO_ANSWER} 2401.00001")) == 0.0

    def test_no_citations_scores_zero(self) -> None:
        assert _cited_relevant(_record()) == 0.0

    def test_refused_outcome_is_still_checked(self) -> None:
        assert _cited_relevant(_record(answer="죄송하지만 찾을 수 없습니다."), OUTCOME_REFUSED) == 0.0

    def test_absent_for_paper_chat(self) -> None:
        record = _record(mode="paper_chat", citations=[_citation("2401.00001")])
        assert "cited_relevant" not in answer_checks(record, OUTCOME_ANSWERED)[1]

    @pytest.mark.parametrize("relevant", [[], None, [""], ["", None]])
    def test_absent_without_relevant_ids(self, relevant: Any) -> None:
        record = _record(relevant_arxiv_ids=relevant, citations=[_citation("2401.00001")])
        assert "cited_relevant" not in answer_checks(record, OUTCOME_ANSWERED)[1]

    def test_absent_for_an_empty_answer(self) -> None:
        record = _record(answer="  ", citations=[_citation("2401.00001")])
        assert answer_checks(record, OUTCOME_ANSWERED) == ("", {})

    @pytest.mark.parametrize("outcome", [OUTCOME_REJECTED, OUTCOME_ERROR])
    def test_absent_for_rejected_and_error(self, outcome: str) -> None:
        assert answer_checks(_record(citations=[_citation("2401.00001")]), outcome) == ("", {})

    def test_other_checks_are_unchanged(self) -> None:
        language, scores = answer_checks(_record(citations=[_citation("2401.00001")]), OUTCOME_ANSWERED)
        assert language == "ko"
        assert scores == {"lang_match": 1.0, "cited_relevant": 1.0}


class TestCitedArxivIds:
    def test_ignores_citations_that_are_not_in_the_answer(self) -> None:
        record = {"citations": [_citation("2401.00001", in_answer=False), _citation("2401.00002")], "answer": ""}
        assert cited_arxiv_ids(record) == {"2401.00002"}

    def test_normalizes_versions_and_merges_answer_links(self) -> None:
        record = {
            "citations": [_citation("2401.00002v2")],
            "answer": "see https://arxiv.org/abs/2401.00003v1 and https://arxiv.org/pdf/2401.00004",
        }
        assert cited_arxiv_ids(record) == {"2401.00002", "2401.00003", "2401.00004"}

    def test_skips_blank_ids_and_non_mapping_entries(self) -> None:
        record = {"citations": [{"arxiv_id": "", "in_answer": True}, "x", None, {"in_answer": True}], "answer": None}
        assert cited_arxiv_ids(record) == set()

    def test_missing_fields(self) -> None:
        assert cited_arxiv_ids({}) == set()
