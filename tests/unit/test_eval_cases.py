from __future__ import annotations

import csv
import json
import re
import sys
import types
from pathlib import Path

import pytest

from eval.answers import GenerationResult, build_answer_record, collect_answers, prepare_query
from eval.behavior import (
    CHAT_HISTORY_MAX_MESSAGES,
    CHAT_MESSAGE_MAX_CHARS,
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
from eval.dataset import (
    VALID_BEHAVIORS,
    DatasetError,
    EvalQuery,
    dataset_digest,
    dump_queries,
    fill_placeholders,
    find_placeholders,
    load_queries,
    parse_lines,
)
from eval.generation import (
    SKIP_NON_ANSWER_CASE,
    SKIP_REJECTED,
    aggregate_scores,
    plan_record,
    render_markdown,
    score_records,
    write_sample_csv,
    write_summary_csv,
)
from eval.runner import aggregate, run_evaluation, split_retrieval_queries

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_DIR = REPO_ROOT / "eval" / "cases"
REGISTRY_PATH = CASES_DIR / "placeholders.json"
README_PATH = REPO_ROOT / "eval" / "README.md"
MIN_PERSPECTIVES = 5
MIN_CASES = 40


def _line(**overrides) -> str:
    payload = {
        "id": "q1",
        "query": "sparse attention",
        "lang": "en",
        "relevant_arxiv_ids": ["2401.00001"],
        "source": "manual",
    }
    payload.update(overrides)
    return json.dumps({key: value for key, value in payload.items() if value is not None}, ensure_ascii=False)


def _parse(**overrides) -> EvalQuery:
    (query,) = parse_lines([_line(**overrides)], origin="cases.jsonl")
    return query


def _load_script(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"_eval_cases_script_{name}", REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestSchemaDefaults:
    def test_old_records_get_defaults(self):
        query = _parse()
        assert query.category == ""
        assert query.expected_behavior == "answer"
        assert query.mode == "both"
        assert query.history == ()
        assert query.must_not_contain == ()
        assert query.must_mention_arxiv_ids == ()
        assert query.expect_rejected is False
        assert query.open_arxiv_id == ""
        assert query.paper_chat_arxiv_id == "2401.00001"
        assert query.retrieval_eligible

    def test_old_records_dump_without_new_keys(self):
        assert set(_parse().to_dict()) == {"id", "query", "lang", "relevant_arxiv_ids", "source", "notes"}

    def test_new_fields_roundtrip(self):
        line = _line(
            category="conversation",
            expected_behavior="refuse",
            mode="agent",
            relevant_arxiv_ids=[],
            history=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "[x](https://arxiv.org/abs/1)"},
            ],
            must_not_contain=["[Fake", "[Fake"],
            must_mention_arxiv_ids=["2401.00002"],
            open_arxiv_id="2401.00003",
        )
        (query,) = parse_lines([line])
        assert query.history == (("user", "hi"), ("assistant", "[x](https://arxiv.org/abs/1)"))
        assert query.must_not_contain == ("[Fake",)
        assert query.paper_chat_arxiv_id == "2401.00003"
        assert parse_lines(dump_queries([query]).splitlines()) == [query]

    def test_query_whitespace_is_preserved(self):
        assert _parse(query="  padded question \n").query == "  padded question \n"

    def test_expect_rejected_allows_empty_query_without_ids(self):
        query = _parse(query="", relevant_arxiv_ids=None, expect_rejected=True, expected_behavior="clarify")
        assert query.query == ""
        assert not query.retrieval_eligible

    def test_answer_case_accepts_must_mention_instead_of_relevant(self):
        query = _parse(relevant_arxiv_ids=[], must_mention_arxiv_ids=["2401.00009"], mode="agent")
        assert query.relevant_arxiv_ids == ()
        assert not query.retrieval_eligible

    def test_refuse_case_needs_no_ids(self):
        assert _parse(relevant_arxiv_ids=[], expected_behavior="refuse", mode="agent").relevant_arxiv_ids == ()

    @pytest.mark.parametrize(
        "overrides, message",
        [
            ({"expected_behavior": "maybe"}, "'expected_behavior'"),
            ({"mode": "web"}, "'mode'"),
            ({"category": 3}, "'category'"),
            ({"history": "hi"}, "'history'"),
            ({"history": ["hi"]}, r"'history\[0\]'"),
            ({"history": [{"role": "system", "content": "x"}]}, r"'history\[0\]\.role'"),
            ({"history": [{"role": "user", "content": 1}]}, r"'history\[0\]\.content'"),
            ({"must_not_contain": ["ok", " "]}, "'must_not_contain'"),
            ({"must_not_contain": "x"}, "'must_not_contain'"),
            ({"must_mention_arxiv_ids": [1]}, "'must_mention_arxiv_ids'"),
            ({"expect_rejected": "yes"}, "'expect_rejected'"),
            ({"open_arxiv_id": 5}, "'open_arxiv_id'"),
            ({"query": ""}, "'query'"),
            ({"relevant_arxiv_ids": "2401.00001"}, "'relevant_arxiv_ids'"),
            ({"relevant_arxiv_ids": [], "mode": "agent"}, "answer cases"),
            ({"relevant_arxiv_ids": [], "expected_behavior": "refuse"}, "open_arxiv_id"),
            ({"relevant_arxiv_ids": [], "expected_behavior": "refuse", "mode": "paper_chat"}, "open_arxiv_id"),
        ],
    )
    def test_validation_errors(self, overrides, message):
        with pytest.raises(DatasetError, match=message) as excinfo:
            parse_lines([_line(**overrides)], origin="cases.jsonl")
        assert "cases.jsonl:1" in str(excinfo.value)

    def test_supports_mode(self):
        assert _parse().supports_mode("agent") and _parse().supports_mode("paper_chat")
        assert _parse(mode="agent").supports_mode("agent")
        assert not _parse(mode="agent").supports_mode("paper_chat")

    @pytest.mark.parametrize(
        "overrides",
        [
            {"expected_behavior": "refuse", "open_arxiv_id": "2401.00001"},
            {"expect_rejected": True},
            {"mode": "paper_chat"},
            {"history": [{"role": "user", "content": "earlier"}]},
        ],
    )
    def test_retrieval_ineligible(self, overrides):
        assert not _parse(**overrides).retrieval_eligible

    def test_clarify_case_with_ids_is_retrieval_eligible(self):
        assert _parse(expected_behavior="clarify").retrieval_eligible


class TestDirectoryLoading:
    def test_loads_all_jsonl_files_in_name_order(self, tmp_path):
        (tmp_path / "b.jsonl").write_text(_line(id="b1") + "\n", encoding="utf-8")
        (tmp_path / "a.jsonl").write_text(_line(id="a1") + "\n" + _line(id="a2") + "\n", encoding="utf-8")
        (tmp_path / "notes.json").write_text("{}", encoding="utf-8")
        assert [query.id for query in load_queries(tmp_path)] == ["a1", "a2", "b1"]

    def test_duplicate_ids_across_files(self, tmp_path):
        (tmp_path / "a.jsonl").write_text(_line(id="same") + "\n", encoding="utf-8")
        (tmp_path / "b.jsonl").write_text(_line(id="same") + "\n", encoding="utf-8")
        with pytest.raises(DatasetError, match="duplicate id 'same'"):
            load_queries(tmp_path)

    def test_digest_changes_with_content(self, tmp_path):
        (tmp_path / "a.jsonl").write_text(_line(id="a1") + "\n", encoding="utf-8")
        first = dataset_digest(tmp_path)
        (tmp_path / "b.jsonl").write_text(_line(id="b1") + "\n", encoding="utf-8")
        assert dataset_digest(tmp_path) != first
        assert len(first) == 12

    def test_missing_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_queries(tmp_path / "missing.jsonl")


def _catalogue() -> list[EvalQuery]:
    return load_queries(CASES_DIR)


def _registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


class TestCatalogue:
    def test_loads_with_enough_cases_and_perspectives(self):
        cases = _catalogue()
        categories = {case.category for case in cases}
        assert len(cases) >= MIN_CASES
        assert len(categories) >= MIN_PERSPECTIVES
        for category in categories:
            assert sum(1 for case in cases if case.category == category) >= 3

    def test_one_file_per_perspective(self):
        for path in sorted(CASES_DIR.glob("*.jsonl")):
            assert {case.category for case in load_queries(path)} == {path.stem}

    def test_every_case_has_category_and_behavior(self):
        for case in _catalogue():
            assert case.category, case.id
            assert case.expected_behavior in VALID_BEHAVIORS, case.id
            assert case.source == "manual", case.id

    def test_answer_cases_have_expected_papers(self):
        for case in _catalogue():
            if case.expected_behavior == "answer" and not case.expect_rejected:
                assert case.relevant_arxiv_ids or case.must_mention_arxiv_ids, case.id

    def test_refuse_cases_check_something_beyond_the_phrase(self):
        refuse = [case for case in _catalogue() if case.expected_behavior == "refuse"]
        assert refuse
        assert all(case.mode == "agent" or case.paper_chat_arxiv_id for case in refuse)

    def test_boundary_lengths_are_exact(self):
        by_id = {case.id: case for case in _catalogue()}
        exact = by_id["sf-exactly-4000-chars"].query
        assert len(exact) == CHAT_MESSAGE_MAX_CHARS and exact == exact.strip()
        over = by_id["sf-4001-chars"].query
        assert len(over) == CHAT_MESSAGE_MAX_CHARS + 1 and over == over.strip()
        padded = by_id["sf-4000-chars-padded"].query
        assert len(padded) > CHAT_MESSAGE_MAX_CHARS and len(padded.strip()) == CHAT_MESSAGE_MAX_CHARS
        assert by_id["sf-4001-chars"].expect_rejected
        assert not by_id["sf-exactly-4000-chars"].expect_rejected

    def test_expect_rejected_matches_the_api_mirror(self):
        for case in _catalogue():
            try:
                prepare_query(case)
            except InputRejected:
                rejected = True
            else:
                rejected = False
            assert rejected == case.expect_rejected, case.id

    def test_long_paragraph_cases_exceed_800_chars(self):
        long_cases = [case for case in _catalogue() if case.id.startswith("qf-long-paragraph")]
        assert len(long_cases) == 2
        assert all(len(case.query) > 800 for case in long_cases)

    def test_history_cases_cover_truncation(self):
        by_id = {case.id: case for case in _catalogue()}
        long_history = by_id["cv-history-over-20-turns"]
        assert len(long_history.history) > CHAT_HISTORY_MAX_MESSAGES
        prepared = prepare_query(long_history)
        assert len(prepared.history) == CHAT_HISTORY_MAX_MESSAGES
        assert all("ZEBRA-19" not in content for _, content in prepared.history)
        assert "ZEBRA-19" in long_history.must_not_contain

    def test_placeholders_are_registered(self):
        registry = _registry()
        cases = _catalogue()
        used = find_placeholders([case.to_dict() for case in cases])
        assert used
        assert used <= set(registry["placeholders"])
        assert set(registry["absent"]) <= {case.id for case in cases}

    def test_registry_loads_through_the_attach_helper(self):
        script = _load_script("eval_build_queries")
        registry = script.load_placeholder_registry(REGISTRY_PATH)
        assert set(registry["placeholders"]) == set(_registry()["placeholders"])

    def test_readme_lists_the_actual_counts(self):
        counts: dict[str, int] = {}
        for case in _catalogue():
            counts[case.category] = counts.get(case.category, 0) + 1
        text = README_PATH.read_text(encoding="utf-8")
        for category, count in counts.items():
            assert re.search(rf"\| `{category}` \| {count} \|", text), category
        assert f"| 합계 | {sum(counts.values())} |" in text


class TestPrepareChatInput:
    def test_strips_and_accepts_the_limit(self):
        message, history = prepare_chat_input("  " + "a" * CHAT_MESSAGE_MAX_CHARS + "\n")
        assert len(message) == CHAT_MESSAGE_MAX_CHARS
        assert history == []

    @pytest.mark.parametrize("message", ["", "   ", "\n\t　", "\u001c\u001d\u001e\u001f", "a" * 4001])
    def test_rejects(self, message):
        with pytest.raises(InputRejected):
            prepare_chat_input(message)

    def test_nul_is_not_whitespace(self):
        with pytest.raises(InputRejected):
            prepare_chat_input("\u0000")

    def test_history_keeps_last_20_non_empty_turns_and_drops_repeated_current(self):
        turns = [("user" if index % 2 == 0 else "assistant", f"turn {index}") for index in range(24)]
        turns += [("assistant", "   "), ("user", "question")]
        _, history = prepare_chat_input("question", turns)
        assert len(history) == CHAT_HISTORY_MAX_MESSAGES
        assert history[0] == ("user", "turn 4")
        assert history[-1] == ("assistant", "turn 23")

    def test_history_turns_are_truncated_per_message(self):
        _, history = prepare_chat_input("q", [("assistant", "x" * 5000)])
        assert len(history[0][1]) == CHAT_MESSAGE_MAX_CHARS


def _behavior_record(**overrides) -> dict:
    record = {
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


class TestBehaviorMetrics:
    def test_outcomes(self):
        assert classify_outcome(_behavior_record(answer="Here are the papers.")) == OUTCOME_ANSWERED
        assert classify_outcome(_behavior_record(answer="관련 논문을 찾지 못했습니다.")) == OUTCOME_REFUSED
        assert classify_outcome(_behavior_record(rejection="메시지를 입력하세요.")) == OUTCOME_REJECTED
        assert classify_outcome(_behavior_record(error="RuntimeError: x", answer="찾지 못")) == OUTCOME_ERROR

    def test_refusal_phrases_are_configurable(self, tmp_path):
        path = tmp_path / "phrases.txt"
        path.write_text("# comment\n\nNOPE\n", encoding="utf-8")
        phrases = load_refusal_phrases(path)
        assert phrases == ["NOPE"]
        assert contains_refusal("well, nope.", phrases)
        assert not contains_refusal("관련 논문을 찾지 못했습니다.", phrases)
        (tmp_path / "empty.txt").write_text("# only comments\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_refusal_phrases(tmp_path / "empty.txt")

    def test_link_extraction_uses_product_rules(self):
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

    def test_refusal_correct_positive(self):
        record = _behavior_record(expected_behavior="refuse", answer="요청하신 논문을 찾을 수 없습니다.")
        outcome, scores = behavior_scores(record)
        assert outcome == OUTCOME_REFUSED
        assert scores["refusal_correct"] == 1.0
        assert scores["no_fabricated_links"] == 1.0

    def test_refusal_correct_negative_without_phrase(self):
        record = _behavior_record(expected_behavior="refuse", answer="QuantumBERT-X는 단백질 접힘을 다룹니다.")
        assert behavior_scores(record)[1]["refusal_correct"] == 0.0

    def test_refusal_correct_negative_with_unverified_link(self):
        record = _behavior_record(
            expected_behavior="refuse",
            answer="정확히는 찾지 못했지만 [QuantumBERT-X](https://arxiv.org/abs/2403.99999)가 비슷합니다.",
        )
        _, scores = behavior_scores(record)
        assert scores["refusal_correct"] == 0.0
        assert scores["no_fabricated_links"] == 0.0

    def test_refusal_correct_only_for_refuse_cases(self):
        assert "refusal_correct" not in behavior_scores(_behavior_record(answer="찾지 못했습니다."))[1]

    def test_no_fabricated_links(self):
        answer = "[A](https://arxiv.org/abs/2401.00001) [B](https://arxiv.org/pdf/2401.00002v3)"
        by_hits = _behavior_record(answer=answer, hit_arxiv_ids=["2401.00001", "2401.00002"])
        by_citation = _behavior_record(
            answer=answer,
            citations=[{"arxiv_id": "2401.00001"}, {"arxiv_id": "", "url": "https://arxiv.org/abs/2401.00002"}],
        )
        missing = _behavior_record(answer=answer, hit_arxiv_ids=["2401.00001"])
        assert behavior_scores(by_hits)[1]["no_fabricated_links"] == 1.0
        assert behavior_scores(by_citation)[1]["no_fabricated_links"] == 1.0
        assert behavior_scores(missing)[1]["no_fabricated_links"] == 0.0
        assert fabricated_link_ids(missing) == ["2401.00002"]
        assert behavior_scores(_behavior_record(answer="링크 없는 답변"))[1]["no_fabricated_links"] == 1.0

    def test_paper_chat_may_link_the_open_paper(self):
        record = _behavior_record(mode="paper_chat", arxiv_id="2401.00005", answer="https://arxiv.org/abs/2401.00005")
        assert behavior_scores(record)[1]["no_fabricated_links"] == 1.0

    def test_must_not_contain(self):
        record = _behavior_record(must_not_contain=["INJECTION-OK-7431", "<script"])
        assert behavior_scores({**record, "answer": "DPO는 선호 최적화입니다."})[1]["must_not_contain_ok"] == 1.0
        assert behavior_scores({**record, "answer": "sure: injection-ok-7431"})[1]["must_not_contain_ok"] == 0.0
        assert "must_not_contain_ok" not in behavior_scores(_behavior_record(answer="x"))[1]

    def test_mentions_required_ids(self):
        record = _behavior_record(must_mention_arxiv_ids=["2401.00001", "2401.00002v2"])
        both = {**record, "answer": "[A](https://arxiv.org/abs/2401.00001) and 2401.00002"}
        one = {**record, "answer": "only [A](https://arxiv.org/abs/2401.00001v1)", "hit_arxiv_ids": ["2401.00001"]}
        none = {**record, "answer": "no ids here"}
        assert behavior_scores(both)[1]["mentions_required_ids"] == 1.0
        assert behavior_scores(one)[1]["mentions_required_ids"] == 0.5
        assert behavior_scores(none)[1]["mentions_required_ids"] == 0.0

    def test_rejected_as_expected(self):
        expected = _behavior_record(expect_rejected=True, rejection="메시지를 입력하세요.")
        missed = _behavior_record(expect_rejected=True, answer="an answer")
        wrongly = _behavior_record(expect_rejected=False, outcome=OUTCOME_REJECTED, rejection="too long")
        failed = _behavior_record(expect_rejected=True, error="RuntimeError: x")
        assert behavior_scores(expected) == (OUTCOME_REJECTED, {"rejected_as_expected": 1.0})
        assert behavior_scores(missed)[1]["rejected_as_expected"] == 0.0
        assert behavior_scores(wrongly)[1]["rejected_as_expected"] == 0.0
        assert behavior_scores(failed) == (OUTCOME_ERROR, {"rejected_as_expected": 0.0})
        assert "rejected_as_expected" not in behavior_scores(_behavior_record(answer="x"))[1]

    def test_error_records_get_no_answer_metrics(self):
        record = _behavior_record(error="RuntimeError: x", expected_behavior="refuse", must_not_contain=["a"])
        assert behavior_scores(record) == (OUTCOME_ERROR, {})


def _q(query_id: str, **kwargs) -> EvalQuery:
    base = {"query": "question", "lang": "en", "relevant_arxiv_ids": ("2401.00001",), "source": "manual"}
    base.update(kwargs)
    return EvalQuery(id=query_id, **base)


class TestCollection:
    def test_rejected_inputs_skip_generation_and_mode_hints_filter(self):
        calls: list[tuple[str, str, str]] = []

        def generator(mode):
            def generate(query):
                calls.append((mode, query.id, query.query))
                return GenerationResult(answer="찾을 수 없습니다.", contexts=[], hit_arxiv_ids=["2401.00001"])

            return generate

        queries = [
            _q("ok", query="  padded  ", category="safety", mode="agent"),
            _q("empty", query=" ", expect_rejected=True, expected_behavior="clarify", category="safety"),
            _q("chat-only", mode="paper_chat", expected_behavior="refuse"),
        ]
        records = collect_answers(queries, ["agent", "paper_chat"], {m: generator(m) for m in ("agent", "paper_chat")})

        assert [(record["id"], record["mode"]) for record in records] == [
            ("ok", "agent"),
            ("empty", "agent"),
            ("empty", "paper_chat"),
            ("chat-only", "paper_chat"),
        ]
        assert calls == [("agent", "ok", "padded"), ("paper_chat", "chat-only", "question")]
        empty = records[1]
        assert empty["outcome"] == OUTCOME_REJECTED
        assert empty["rejection"] == "메시지를 입력하세요."
        assert empty["error"] is None and empty["answer"] == "" and empty["latency_ms"] is None
        assert empty["expect_rejected"] is True
        assert records[0]["outcome"] == OUTCOME_REFUSED
        assert records[0]["category"] == "safety"
        assert records[0]["hit_arxiv_ids"] == ["2401.00001"]
        assert records[0]["history_turns"] == 0

    def test_generators_receive_truncated_history(self):
        seen: list[int] = []
        history = tuple(("user" if index % 2 == 0 else "assistant", f"t{index}") for index in range(26))

        def generate(query):
            seen.append(len(query.history))
            return GenerationResult(answer="ok", contexts=["c"])

        records = collect_answers([_q("h", history=history, mode="agent")], ["agent"], {"agent": generate})
        assert seen == [CHAT_HISTORY_MAX_MESSAGES]
        assert records[0]["history_turns"] == CHAT_HISTORY_MAX_MESSAGES

    def test_custom_refusal_phrases(self):
        records = collect_answers(
            [_q("x", mode="agent")],
            ["agent"],
            {"agent": lambda query: GenerationResult(answer="NOPE", contexts=[])},
            refusal_phrases=["nope"],
        )
        assert records[0]["outcome"] == OUTCOME_REFUSED

    def test_paper_chat_generator_opens_the_open_paper_and_passes_history(self, monkeypatch):
        from unittest.mock import MagicMock

        from eval.answers import make_paper_chat_generator
        from src.core.agent import paper_chat

        captured: dict = {}

        def fake_retrieve(paper, question, *, retriever):
            return paper_chat.PaperChatRetrieval(
                sources=[{"text": "Abs.", "arxiv_id": paper["arxiv_id"]}], retrieval_mode="lexical"
            )

        def fake_stream(question, **kwargs):
            captured.update(kwargs)
            yield {"chunk": "이 논문에는 없습니다 [1]."}
            yield {"citations": []}

        monkeypatch.setattr(paper_chat, "retrieve_paper_chat_sources", fake_retrieve)
        monkeypatch.setattr(paper_chat, "stream_paper_chat_answer", fake_stream)
        repository = MagicMock()
        repository.get_paper.return_value = {"arxiv_id": "2401.00009", "title": "Open", "abstract": "Abs."}
        generate = make_paper_chat_generator(repository=repository, retriever_factory=lambda repo: MagicMock())

        query = _q("pc", open_arxiv_id="2401.00009", mode="paper_chat", history=(("user", "hi"),))
        result = generate(query)

        repository.get_paper.assert_called_once_with("2401.00009")
        assert captured["chat_history"] == [("user", "hi")]
        assert captured["paper"]["arxiv_id"] == "2401.00009"
        assert result.answer == "이 논문에는 없습니다 [1]."
        assert result.hit_arxiv_ids == ["2401.00009"]

    def test_paper_chat_generator_without_target_raises(self):
        from unittest.mock import MagicMock

        from eval.answers import make_paper_chat_generator

        generate = make_paper_chat_generator(repository=MagicMock(), retriever_factory=lambda repo: MagicMock())
        with pytest.raises(LookupError):
            generate(_q("none", relevant_arxiv_ids=(), mode="agent"))

    def test_agent_generator_passes_history(self, monkeypatch):
        from eval.answers import generate_agent_answer
        from src.core.agent import chatbot, tools

        captured: dict = {}

        def fake_agent_search(question, **kwargs):
            captured.update(kwargs)
            tools.retrieve_contexts("q", retriever=None, limit=5)
            return {"answer": "a", "citations": []}

        monkeypatch.setattr(
            tools,
            "retrieve_contexts",
            lambda query, **kwargs: ([{"arxiv_id": "2401.00007", "chunk_text": "t"}], "lexical"),
        )
        monkeypatch.setattr(chatbot, "agent_search", fake_agent_search)
        result = generate_agent_answer(_q("a", history=(("user", "earlier"),)))
        assert captured["chat_history"] == [("user", "earlier")]
        assert result.hit_arxiv_ids == ["2401.00007"]


class TestRetrievalByCategory:
    class Retriever:
        def __init__(self):
            self.queries: list[str] = []

        def _search(self, query, *, limit, adjacency_window):
            self.queries.append(query)
            return [{"arxiv_id": "A", "chunk_id": 1, "content_role": "body"}]

        search_paper_contexts = _search
        search_paper_contexts_by_vector = _search
        search_paper_contexts_by_hybrid = _search

    QUERIES = [
        _q("lang-1", query="l1", relevant_arxiv_ids=("A",), category="language"),
        _q("lang-2", query="l2", relevant_arxiv_ids=("B",), category="language"),
        _q("qf-1", query="q1", relevant_arxiv_ids=("A",), category="query_form", expected_behavior="clarify"),
        _q(
            "oc-1",
            query="o1",
            relevant_arxiv_ids=(),
            category="out_of_corpus",
            expected_behavior="refuse",
            mode="agent",
        ),
        _q("sf-1", query="", relevant_arxiv_ids=(), category="safety", expect_rejected=True),
        _q("pc-1", query="p1", relevant_arxiv_ids=("A",), category="paper_chat", mode="paper_chat"),
        _q("cv-1", query="c1", relevant_arxiv_ids=("A",), category="conversation", history=(("user", "x"),)),
    ]

    def test_split_and_run_skip_ineligible_cases(self):
        kept, skipped = split_retrieval_queries(self.QUERIES)
        assert [query.id for query in kept] == ["lang-1", "lang-2", "qf-1"]
        assert [query.id for query in skipped] == ["oc-1", "sf-1", "pc-1", "cv-1"]
        retriever = self.Retriever()
        results = run_evaluation(retriever, self.QUERIES, ["lexical"], k=5)
        assert retriever.queries == ["l1", "l2", "q1"]
        assert [(row.query_id, row.category, row.expected_behavior) for row in results] == [
            ("lang-1", "language", "answer"),
            ("lang-2", "language", "answer"),
            ("qf-1", "query_form", "clarify"),
        ]

    def test_aggregate_by_category_and_behavior(self):
        results = run_evaluation(self.Retriever(), self.QUERIES, ["lexical"], k=5)
        rows = {row.subset: row for row in aggregate(results)}
        assert rows["category:language"].n_queries == 2
        assert rows["category:language"].paper["hit@1"] == pytest.approx(0.5)
        assert rows["category:query_form"].paper["hit@1"] == 1.0
        assert rows["behavior:answer"].n_queries == 2
        assert rows["behavior:clarify"].n_queries == 1
        assert "category:out_of_corpus" not in rows

    def test_single_behavior_adds_no_behavior_subsets(self):
        results = run_evaluation(self.Retriever(), self.QUERIES[:2], ["lexical"], k=5)
        subsets = {row.subset for row in aggregate(results)}
        assert "category:language" in subsets
        assert not any(subset.startswith("behavior:") for subset in subsets)

    def test_query_csv_has_category_columns(self, tmp_path):
        from eval.report import write_query_csv

        results = run_evaluation(self.Retriever(), self.QUERIES, ["lexical"], k=5)
        write_query_csv(results, tmp_path / "q.csv")
        with (tmp_path / "q.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["category"] for row in rows] == ["language", "language", "query_form"]
        assert rows[2]["expected_behavior"] == "clarify"


def _generation_records() -> list[dict]:
    answer_ok = build_answer_record(
        _q("lang-1", category="language"),
        "agent",
        result=GenerationResult(
            answer="[P](https://arxiv.org/abs/2401.00001)", contexts=["ctx"], hit_arxiv_ids=["2401.00001"]
        ),
    )
    answer_fabricated = build_answer_record(
        _q("lang-2", category="language"),
        "agent",
        result=GenerationResult(answer="[Fake](https://arxiv.org/abs/2099.00001)", contexts=["ctx"]),
    )
    refuse_ok = build_answer_record(
        _q("oc-1", relevant_arxiv_ids=(), category="out_of_corpus", expected_behavior="refuse", mode="agent"),
        "agent",
        result=GenerationResult(answer="관련 논문을 찾지 못했습니다.", contexts=[]),
    )
    rejected = build_answer_record(
        _q("sf-1", query="", relevant_arxiv_ids=(), category="safety", expect_rejected=True),
        "agent",
        rejection="메시지를 입력하세요.",
    )
    return [answer_ok, answer_fabricated, refuse_ok, rejected]


class FixedScorer:
    def __init__(self):
        self.requests = []

    def score(self, requests):
        self.requests.extend(requests)
        return [0.5 for _ in requests]


class TestGenerationByCategory:
    def test_plan_skips_rejected_and_non_answer_cases(self):
        records = _generation_records()
        requests, skipped, _ = plan_record(records[2], ["faithfulness", "answer_relevancy"])
        assert requests == [] and skipped == {
            "faithfulness": SKIP_NON_ANSWER_CASE,
            "answer_relevancy": SKIP_NON_ANSWER_CASE,
        }
        requests, skipped, _ = plan_record(records[3], ["answer_relevancy"])
        assert requests == [] and skipped == {"answer_relevancy": SKIP_REJECTED}

    def test_behavior_is_scored_per_record_and_aggregated_by_category(self):
        scorer = FixedScorer()
        samples = score_records(_generation_records(), ["answer_relevancy"], scorer)
        assert [request.user_input for request in scorer.requests] == ["question", "question"]
        assert [sample.outcome for sample in samples] == ["answered", "answered", "refused", "rejected"]
        assert samples[1].behavior == {"no_fabricated_links": 0.0}
        assert samples[2].behavior == {"no_fabricated_links": 1.0, "refusal_correct": 1.0}
        assert samples[3].behavior == {"rejected_as_expected": 1.0}

        rows = {(row.mode, row.subset): row for row in aggregate_scores(samples, ["answer_relevancy"])}
        language = rows[("agent", "category:language")]
        assert language.n_records == 2
        assert language.means["answer_relevancy"] == 0.5
        assert language.behavior_means["no_fabricated_links"] == 0.5
        assert language.behavior_counts["no_fabricated_links"] == 2
        assert rows[("agent", "category:out_of_corpus")].behavior_means["refusal_correct"] == 1.0
        assert rows[("agent", "behavior:refuse")].n_records == 1
        assert rows[("agent", "behavior:answer")].n_records == 3
        overall = rows[("agent", "all")]
        assert overall.outcomes == {"answered": 2, "refused": 1, "rejected": 1, "error": 0}
        assert overall.behavior_counts["rejected_as_expected"] == 1
        assert overall.behavior_means["refusal_correct"] == 1.0

    def test_markdown_behavior_table_and_csv(self, tmp_path):
        samples = score_records(_generation_records(), [], FixedScorer())
        aggregates = aggregate_scores(samples, [])
        markdown = render_markdown(aggregates, samples, [], title="t", meta={})
        section = markdown.split("## 행동 지표 (LLM 판정 없음)")[1].split("## 건너뛴 지표")[0]
        assert "| agent | all | 4 | 2 | 1 | 1 | 0 | 1.000 (1) | 0.667 (3) | - | - | 1.000 (1) |" in section
        assert "| agent | category:safety | 1 | 0 | 0 | 1 | 0 | - | - | - | - | 1.000 (1) |" in section
        assert "`refusal_correct`" in markdown

        write_sample_csv(samples, [], tmp_path / "s.csv")
        write_summary_csv(aggregates, [], tmp_path / "sum.csv")
        with (tmp_path / "s.csv").open(encoding="utf-8") as handle:
            sample_rows = list(csv.DictReader(handle))
        assert [row["outcome"] for row in sample_rows] == ["answered", "answered", "refused", "rejected"]
        assert sample_rows[1]["no_fabricated_links"] == "0.0000"
        assert sample_rows[0]["category"] == "language"
        with (tmp_path / "sum.csv").open(encoding="utf-8") as handle:
            summary = {row["subset"]: row for row in csv.DictReader(handle)}
        assert summary["all"]["n_rejected"] == "1"
        assert summary["category:language"]["no_fabricated_links"] == "0.5000"
        assert summary["category:language"]["no_fabricated_links_n"] == "2"


class TestAttachIds:
    PAPERS = {"0000.0000a": ("2405.01234", "Direct Preference Optimization Revisited For Everyone")}

    def test_fill_and_find_placeholders(self):
        payload = {
            "query": "{{title:0000.0000a}} vs {{title_head:0000.0000a:2}} and {{title:0000.0000b}}",
            "relevant_arxiv_ids": ["0000.0000a", "0000.0000b"],
            "history": [{"role": "assistant", "content": "[x](https://arxiv.org/abs/0000.0000a)"}],
        }
        filled = fill_placeholders(payload, self.PAPERS)
        assert filled["query"] == (
            "Direct Preference Optimization Revisited For Everyone vs Direct Preference and {{title:0000.0000b}}"
        )
        assert filled["relevant_arxiv_ids"] == ["2405.01234", "0000.0000b"]
        assert filled["history"][0]["content"] == "[x](https://arxiv.org/abs/2405.01234)"
        assert find_placeholders(filled) == {"0000.0000b"}
        assert payload["relevant_arxiv_ids"][0] == "0000.0000a"

    def test_choose_distinct_avoids_reusing_a_paper(self):
        script = _load_script("eval_build_queries")
        chosen = script.choose_distinct(
            {
                "0000.0000a": [("1", "One"), ("2", "Two")],
                "0000.0000b": [("1", "One"), ("3", "Three")],
                "0000.0000c": [],
            }
        )
        assert chosen == {"0000.0000a": ("1", "One"), "0000.0000b": ("3", "Three")}

    def test_attach_catalogue_drops_unresolved_and_present_absent_cases(self):
        script = _load_script("eval_build_queries")
        payloads = [
            {"id": "keep", "query": "{{title:0000.0000a}}", "relevant_arxiv_ids": ["0000.0000a"]},
            {"id": "unresolved", "query": "x", "relevant_arxiv_ids": ["0000.0000b"]},
            {"id": "absent", "query": "AlexNet top-5?", "relevant_arxiv_ids": []},
        ]
        attached, dropped = script.attach_catalogue(payloads, self.PAPERS, absent_present={"absent": "1 AlexNet"})
        assert [payload["id"] for payload in attached] == ["keep"]
        assert attached[0]["relevant_arxiv_ids"] == ["2405.01234"]
        assert len(dropped) == 2 and dropped[0].startswith("unresolved") and dropped[1].startswith("absent")

    def test_candidate_sql(self):
        script = _load_script("eval_build_queries")
        sql, params = script.candidate_sql(
            {"title_keyword": "Expert", "min_chunks": 25, "section_keyword": "Limitation", "content_role": "appendix"},
            5,
        )
        assert sql.count("%s") == len(params)
        assert params == ["%Expert%", 25, "%Limitation%", "appendix", 5]
        sql, params = script.candidate_sql({"title_keyword": "", "chunk_pattern": "Figure 1"}, 3)
        assert params == ["%%", 1, "%Figure 1%", 3]

    def test_attached_catalogue_is_valid(self, tmp_path):
        script = _load_script("eval_build_queries")
        registry = script.load_placeholder_registry(REGISTRY_PATH)
        papers = {
            placeholder: (f"2501.{index:05d}", f"Paper number {index} about things")
            for index, placeholder in enumerate(sorted(registry["placeholders"]), start=1)
        }
        attached, dropped = script.attach_catalogue(script.read_case_payloads(CASES_DIR), papers, absent_present={})
        assert dropped == []
        out = tmp_path / "queries.cases.jsonl"
        out.write_text(
            "".join(json.dumps(payload, ensure_ascii=False) + "\n" for payload in attached), encoding="utf-8"
        )
        loaded = load_queries(out)
        assert len(loaded) == len(_catalogue())
        assert not find_placeholders([query.to_dict() for query in loaded])
        by_id = {query.id: query for query in loaded}
        assert by_id["qf-exact-title"].query == papers["0000.0000g"][1]
        assert by_id["qf-partial-title"].query.startswith("Paper number 9 about ")

    @pytest.mark.parametrize(
        "argv",
        [["--write"], ["--attach-ids", "--candidates", "0"]],
    )
    def test_parse_args_rejects(self, argv):
        script = _load_script("eval_build_queries")
        with pytest.raises(SystemExit):
            script.parse_args(argv)

    def test_bad_override_and_unreachable_db(self, monkeypatch, capsys):
        script = _load_script("eval_build_queries")
        assert script.main(["--attach-ids", "--set", "a=b"]) == script.EXIT_PRECONDITION
        assert "--set" in capsys.readouterr().err

        def refuse():
            raise OSError("connection refused")

        monkeypatch.setattr(script, "_connect", refuse)
        assert script.main(["--attach-ids"]) == script.EXIT_PRECONDITION
        assert "PostgreSQL에 연결할 수 없습니다" in capsys.readouterr().err


class TestScripts:
    @pytest.fixture(autouse=True)
    def _settings(self, monkeypatch):
        import src.shared

        self.settings = types.SimpleNamespace(
            openai_api_key=None,
            openai_model="gpt-4o",
            openai_embedding_model="text-embedding-3-large",
            openai_embedding_dimensions=1536,
        )
        monkeypatch.setattr(src.shared, "get_settings", lambda: self.settings)

    def test_behavior_only_scores_without_key_or_db(self, tmp_path, monkeypatch, capsys):
        from unittest.mock import MagicMock

        from eval.answers import append_answer_record

        script = _load_script("eval_generation")
        monkeypatch.setattr(script, "_connect", MagicMock(side_effect=AssertionError("no db")))
        monkeypatch.setattr(script, "build_scorer", MagicMock(side_effect=AssertionError("no judge")))
        answers = tmp_path / "answers.jsonl"
        for record in _generation_records():
            append_answer_record(answers, record)
        out_dir = tmp_path / "results"

        code = script.main(
            ["--answers", str(answers), "--behavior-only", "--category", "language,safety", "--out-dir", str(out_dir)]
        )

        assert code == 0
        markdown = next(out_dir.glob("generation_*.md")).read_text(encoding="utf-8")
        assert "RAGAS 생략(--behavior-only)" in markdown
        assert "케이스 카테고리: language 2, safety 1" in markdown
        assert "| agent | category:safety | 1 | 0 | 0 | 1 | 0 |" in markdown
        assert "결과 종류: answered 2, refused 0, rejected 1, error 0" in capsys.readouterr().err

    def test_collect_still_needs_a_key(self, tmp_path, capsys):
        script = _load_script("eval_generation")
        code = script.main(["--collect", "--behavior-only", "--queries", str(CASES_DIR), "--out-dir", str(tmp_path)])
        assert code == script.EXIT_PRECONDITION
        assert "OPENAI_API_KEY" in capsys.readouterr().err

    def test_collect_refuses_placeholder_catalogue(self, tmp_path, capsys):
        script = _load_script("eval_generation")
        self.settings.openai_api_key = "sk-test"
        code = script.main(["--collect", "--queries", str(CASES_DIR), "--out-dir", str(tmp_path / "r")])
        assert code == script.EXIT_PRECONDITION
        assert "--attach-ids" in capsys.readouterr().err
        assert not (tmp_path / "r").exists()

    def test_preflight_targets_only_paper_chat_cases(self):
        script = _load_script("eval_generation")
        queries = [
            _q("a", mode="agent", relevant_arxiv_ids=("1",)),
            _q("b", open_arxiv_id="2", relevant_arxiv_ids=("3",)),
            _q("c", mode="paper_chat", relevant_arxiv_ids=("4",)),
            _q("d", query="", expect_rejected=True, open_arxiv_id="5"),
        ]
        assert script.paper_chat_targets(queries, ["agent", "paper_chat"]) == ["2", "4"]
        assert script.paper_chat_targets(queries, ["agent"]) == []

    def test_retrieval_refuses_placeholder_catalogue(self, tmp_path, capsys):
        script = _load_script("eval_retrieval")
        code = script.main(["--queries", str(CASES_DIR), "--methods", "lexical", "--out-dir", str(tmp_path / "r")])
        assert code == script.EXIT_PRECONDITION
        err = capsys.readouterr().err
        assert "검색 평가 대상이 아닌 케이스" in err and "--attach-ids" in err
        assert not (tmp_path / "r").exists()

    def test_retrieval_meta_reports_skipped_cases(self, tmp_path, monkeypatch):
        import src.integrations.paper_retriever as retriever_module

        script = _load_script("eval_retrieval")
        corpus = {"papers": 1, "papers_with_chunks": 1, "chunks": 1, "embeddings": 1, "fulltext_sources": {}}
        monkeypatch.setattr(script, "preflight", lambda settings, queries, needs_embeddings: corpus)
        monkeypatch.setattr(retriever_module, "PaperRetriever", lambda: TestRetrievalByCategory.Retriever())
        queries = tmp_path / "cases"
        queries.mkdir()
        (queries / "cases.jsonl").write_text(dump_queries(TestRetrievalByCategory.QUERIES), encoding="utf-8")
        out_dir = tmp_path / "results"

        code = script.main(["--queries", str(queries), "--methods", "lexical", "--out-dir", str(out_dir)])

        assert code == 0
        markdown = next(out_dir.glob("*.md")).read_text(encoding="utf-8")
        assert "`cases` sha256:" in markdown
        assert "검색 평가 제외: 4개" in markdown
        assert "케이스 카테고리: language 2, query_form 1" in markdown
        assert "| lexical | category:language | 2 |" in markdown


def test_required_id_counts_when_cited_paper_title_appears_in_answer():
    from eval.behavior import behavior_scores

    record = {
        "expected_behavior": "answer",
        "must_mention_arxiv_ids": ["2609.24220"],
        "answer": "**Document Retrieval-Aware Chunking (D-RAC)** 논문의 실험 결과는 다음과 같습니다.",
        "outcome": "answered",
        "citations": [
            {"arxiv_id": "2609.24220", "title": "Document Retrieval-Aware Chunking (D-RAC)", "in_answer": False}
        ],
    }
    _, scores = behavior_scores(record)
    assert scores["mentions_required_ids"] == 1.0
