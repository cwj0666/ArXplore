import pytest

from arxplore_eval.behavior import (
    DEFAULT_REFUSAL_PHRASES,
    OUTCOME_ANSWERED,
    OUTCOME_REFUSED,
    behavior_scores,
    classify_outcome,
    contains_refusal,
)

WITHHELD_ANSWERS = (
    "제공된 발췌문으로는 AlexNet 논문에서 ImageNet에 대한 top-5 오류에 대한 정보를 찾기 어렵습니다.",
    "제공된 발췌문에서는 정확한 숫자를 포함하고 있지 않습니다. 따라서 이 질문에 대한 답변을 제공하기 어렵습니다.",
    "The search did not return specific information about the reported accuracy.",
)


def _record(answer: str, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": "pc-other-paper-en",
        "mode": "paper_chat",
        "answer": answer,
        "error": None,
        "citations": [],
        "hit_arxiv_ids": [],
        "expected_behavior": "answer",
        "expect_rejected": False,
        "must_not_contain": [],
        "must_mention_arxiv_ids": [],
    }
    record.update(overrides)
    return record


def test_the_phrases_missed_in_the_manual_labels_are_in_the_default_list() -> None:
    assert {"찾기 어렵", "제공하기 어렵", "did not return"} <= set(DEFAULT_REFUSAL_PHRASES)
    assert len(set(DEFAULT_REFUSAL_PHRASES)) == len(DEFAULT_REFUSAL_PHRASES)


@pytest.mark.parametrize("answer", WITHHELD_ANSWERS)
def test_withheld_answers_count_as_refusals(answer: str) -> None:
    assert contains_refusal(answer)
    assert classify_outcome(_record(answer)) == OUTCOME_REFUSED


@pytest.mark.parametrize(
    "answer",
    [
        "이 논문은 긴 문맥에서 근거를 찾는 방법을 제안하고, 표 2에서 정확도를 4.1점 높였다고 보고합니다.",
        "The retriever returns five chunks and the model cites them as [1] and [2].",
        "저자들은 데이터를 공개하고 재현 코드를 제공합니다.",
    ],
)
def test_ordinary_answers_stay_answered(answer: str) -> None:
    assert not contains_refusal(answer)
    assert classify_outcome(_record(answer)) == OUTCOME_ANSWERED


def test_a_paper_chat_refusal_written_with_the_new_phrase_is_scored_correct() -> None:
    outcome, scores = behavior_scores(_record(WITHHELD_ANSWERS[0], expected_behavior="refuse"))

    assert outcome == OUTCOME_REFUSED
    assert scores["refusal_correct"] == 1.0
