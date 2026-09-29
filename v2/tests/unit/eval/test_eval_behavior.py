from typing import Any

import pytest

from arxplore.api.errors import EMPTY_MESSAGE_MESSAGE
from arxplore.api.services.chat import CHAT_HISTORY_MAX_MESSAGES, CHAT_MESSAGE_MAX_CHARS, MESSAGE_TOO_LONG_MESSAGE
from arxplore_eval.behavior import (
    DEFAULT_REFUSAL_PHRASES,
    OUTCOME_ANSWERED,
    OUTCOME_ERROR,
    OUTCOME_REFUSED,
    OUTCOME_REJECTED,
    InputRejected,
    answer_link_ids,
    behavior_scores,
    classify_outcome,
    contains_refusal,
    fabricated_link_ids,
    load_refusal_phrases,
    mentioned_arxiv_ids,
    prepare_chat_input,
)


def _record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": "c1",
        "mode": "agent",
        "answer": "",
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


class TestPrepareChatInput:
    def test_strips_and_accepts_the_limit(self) -> None:
        message, history = prepare_chat_input("  " + "a" * CHAT_MESSAGE_MAX_CHARS + "\n")
        assert len(message) == CHAT_MESSAGE_MAX_CHARS
        assert history == []

    @pytest.mark.parametrize(
        ("message", "error"),
        [
            ("", EMPTY_MESSAGE_MESSAGE),
            ("   ", EMPTY_MESSAGE_MESSAGE),
            ("\n\t　", EMPTY_MESSAGE_MESSAGE),
            ("\u001c\u001d\u001e\u001f", EMPTY_MESSAGE_MESSAGE),
            ("\u0000", EMPTY_MESSAGE_MESSAGE),
            ("a" * (CHAT_MESSAGE_MAX_CHARS + 1), MESSAGE_TOO_LONG_MESSAGE),
        ],
    )
    def test_rejects_with_the_product_messages(self, message: str, error: str) -> None:
        with pytest.raises(InputRejected) as excinfo:
            prepare_chat_input(message)
        assert str(excinfo.value) == error

    def test_history_keeps_last_turns_and_drops_repeated_current(self) -> None:
        turns = [("user" if index % 2 == 0 else "assistant", f"turn {index}") for index in range(24)]
        turns += [("assistant", "   "), ("user", "question")]
        _, history = prepare_chat_input("question", turns)
        assert len(history) == CHAT_HISTORY_MAX_MESSAGES
        assert history[0] == ("user", "turn 4")
        assert history[-1] == ("assistant", "turn 23")

    def test_history_turns_are_truncated_per_message(self) -> None:
        _, history = prepare_chat_input("q", [("assistant", "x" * 5000)])
        assert len(history[0][1]) == CHAT_MESSAGE_MAX_CHARS


class TestOutcomes:
    def test_outcomes(self) -> None:
        assert classify_outcome(_record(answer="Here are the papers.")) == OUTCOME_ANSWERED
        assert classify_outcome(_record(answer="관련 논문을 찾지 못했습니다.")) == OUTCOME_REFUSED
        assert classify_outcome(_record(rejection="메시지를 입력하세요.")) == OUTCOME_REJECTED
        assert classify_outcome(_record(error="RuntimeError: x", answer="찾지 못")) == OUTCOME_ERROR

    def test_refusal_phrases_are_configurable(self, tmp_path) -> None:
        path = tmp_path / "phrases.txt"
        path.write_text("# comment\n\nNOPE\n", encoding="utf-8")
        phrases = load_refusal_phrases(path)
        assert phrases == ["NOPE"]
        assert contains_refusal("well, nope.", phrases)
        assert not contains_refusal("관련 논문을 찾지 못했습니다.", phrases)
        (tmp_path / "empty.txt").write_text("# only comments\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_refusal_phrases(tmp_path / "empty.txt")

    def test_default_phrases_have_no_duplicates(self) -> None:
        assert len(set(DEFAULT_REFUSAL_PHRASES)) == len(DEFAULT_REFUSAL_PHRASES)
        assert {"outside the scope", "not covered"} <= set(DEFAULT_REFUSAL_PHRASES)


class TestLinks:
    def test_link_extraction_uses_product_rules(self) -> None:
        answer = (
            "See [A](https://arxiv.org/abs/2401.00001v2), https://arxiv.org/pdf/2402.00002.pdf. "
            "and [B](https://www.arxiv.org/abs/2401.00001) plus [list](https://arxiv.org/list/cs.CL) "
            "and [gh](https://github.com/x/y)"
        )
        assert answer_link_ids(answer) == ["2401.00001", "2402.00002"]
        assert mentioned_arxiv_ids("id 2403.00003v1 and [x](https://arxiv.org/abs/2404.00004)") == {
            "2403.00003",
            "2404.00004",
        }

    def test_no_fabricated_links(self) -> None:
        answer = "[A](https://arxiv.org/abs/2401.00001) [B](https://arxiv.org/pdf/2401.00002v3)"
        by_hits = _record(answer=answer, hit_arxiv_ids=["2401.00001", "2401.00002"])
        by_citation = _record(
            answer=answer,
            citations=[{"arxiv_id": "2401.00001"}, {"arxiv_id": "", "url": "https://arxiv.org/abs/2401.00002"}],
        )
        missing = _record(answer=answer, hit_arxiv_ids=["2401.00001"])
        assert behavior_scores(by_hits)[1]["no_fabricated_links"] == 1.0
        assert behavior_scores(by_citation)[1]["no_fabricated_links"] == 1.0
        assert behavior_scores(missing)[1]["no_fabricated_links"] == 0.0
        assert fabricated_link_ids(missing) == ["2401.00002"]
        assert behavior_scores(_record(answer="링크 없는 답변"))[1]["no_fabricated_links"] == 1.0

    def test_paper_chat_may_link_the_open_paper(self) -> None:
        record = _record(mode="paper_chat", arxiv_id="2401.00005", answer="https://arxiv.org/abs/2401.00005")
        assert behavior_scores(record)[1]["no_fabricated_links"] == 1.0


class TestBehaviorScores:
    def test_refusal_correct(self) -> None:
        positive = _record(expected_behavior="refuse", answer="요청하신 논문을 찾을 수 없습니다.")
        outcome, scores = behavior_scores(positive)
        assert outcome == OUTCOME_REFUSED
        assert scores["refusal_correct"] == 1.0 and scores["no_fabricated_links"] == 1.0

        no_phrase = _record(expected_behavior="refuse", answer="QuantumBERT-X는 단백질 접힘을 다룹니다.")
        assert behavior_scores(no_phrase)[1]["refusal_correct"] == 0.0

        fabricated = _record(
            expected_behavior="refuse",
            answer="정확히는 찾지 못했지만 [QuantumBERT-X](https://arxiv.org/abs/2403.99999)가 비슷합니다.",
        )
        scores = behavior_scores(fabricated)[1]
        assert scores["refusal_correct"] == 0.0 and scores["no_fabricated_links"] == 0.0
        assert "refusal_correct" not in behavior_scores(_record(answer="찾지 못했습니다."))[1]

    def test_must_not_contain(self) -> None:
        record = _record(must_not_contain=["INJECTION-OK-7431", "<script"])
        assert behavior_scores({**record, "answer": "DPO는 선호 최적화입니다."})[1]["must_not_contain_ok"] == 1.0
        assert behavior_scores({**record, "answer": "sure: injection-ok-7431"})[1]["must_not_contain_ok"] == 0.0
        assert "must_not_contain_ok" not in behavior_scores(_record(answer="x"))[1]

    def test_mentions_required_ids(self) -> None:
        record = _record(must_mention_arxiv_ids=["2401.00001", "2401.00002v2"])
        both = {**record, "answer": "[A](https://arxiv.org/abs/2401.00001) and 2401.00002"}
        one = {**record, "answer": "only [A](https://arxiv.org/abs/2401.00001v1)", "hit_arxiv_ids": ["2401.00001"]}
        none = {**record, "answer": "no ids here"}
        assert behavior_scores(both)[1]["mentions_required_ids"] == 1.0
        assert behavior_scores(one)[1]["mentions_required_ids"] == 0.5
        assert behavior_scores(none)[1]["mentions_required_ids"] == 0.0

    def test_required_id_counts_when_the_cited_title_appears(self) -> None:
        record = _record(
            must_mention_arxiv_ids=["2609.24220"],
            answer="**Document Retrieval-Aware Chunking (D-RAC)** 논문의 실험 결과는 다음과 같습니다.",
            citations=[
                {"arxiv_id": "2609.24220", "title": "Document Retrieval-Aware Chunking (D-RAC)", "in_answer": False}
            ],
        )
        assert behavior_scores(record)[1]["mentions_required_ids"] == 1.0

    def test_rejected_as_expected(self) -> None:
        expected = _record(expect_rejected=True, rejection="메시지를 입력하세요.")
        missed = _record(expect_rejected=True, answer="an answer")
        wrongly = _record(expect_rejected=False, outcome=OUTCOME_REJECTED, rejection="too long")
        failed = _record(expect_rejected=True, error="RuntimeError: x")
        assert behavior_scores(expected) == (OUTCOME_REJECTED, {"rejected_as_expected": 1.0})
        assert behavior_scores(missed)[1]["rejected_as_expected"] == 0.0
        assert behavior_scores(wrongly)[1]["rejected_as_expected"] == 0.0
        assert behavior_scores(failed) == (OUTCOME_ERROR, {"rejected_as_expected": 0.0})
        assert "rejected_as_expected" not in behavior_scores(_record(answer="x"))[1]

    def test_error_records_get_no_answer_metrics(self) -> None:
        record = _record(error="RuntimeError: x", expected_behavior="refuse", must_not_contain=["a"])
        assert behavior_scores(record) == (OUTCOME_ERROR, {})
