import json
from typing import Any

import pytest

from arxplore.api.services.chat import CHAT_HISTORY_MAX_MESSAGES, CHAT_MESSAGE_MAX_CHARS
from arxplore_eval.answers import prepare_query
from arxplore_eval.behavior import InputRejected
from arxplore_eval.dataset import (
    VALID_BEHAVIORS,
    DatasetError,
    EvalQuery,
    chunk_synth_query,
    dataset_digest,
    dump_queries,
    fill_placeholders,
    find_placeholders,
    known_item_queries,
    load_queries,
    longest_shared_word_run,
    parse_categories,
    parse_lines,
    placeholder_problems,
    select_queries,
)

MIN_PERSPECTIVES = 5
MIN_CASES = 40


def _line(**overrides: Any) -> str:
    payload: dict[str, Any] = {
        "id": "q1",
        "query": "sparse attention",
        "lang": "en",
        "relevant_arxiv_ids": ["2401.00001"],
        "source": "manual",
    }
    payload.update(overrides)
    return json.dumps({key: value for key, value in payload.items() if value is not None}, ensure_ascii=False)


def _parse(**overrides: Any) -> EvalQuery:
    (query,) = parse_lines([_line(**overrides)], origin="cases.jsonl")
    return query


class TestParsing:
    def test_sample_file_loads_and_roundtrips(self, eval_dir) -> None:
        queries = load_queries(eval_dir / "queries.sample.jsonl")
        assert len(queries) == 6
        assert {query.lang for query in queries} == {"ko", "en"}
        assert {query.source for query in queries} == {"known_item", "llm_synth", "manual"}
        assert any(query.relevant_chunk_ids for query in queries)
        assert parse_lines(dump_queries(queries).splitlines()) == queries

    def test_blank_lines_are_ignored(self) -> None:
        assert len(parse_lines(["", _line(), "   "])) == 1

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"lang": "ja"}, "'lang'"),
            ({"source": "crawl"}, "'source'"),
            ({"relevant_arxiv_ids": []}, "relevant_arxiv_ids"),
            ({"relevant_chunk_ids": ["12"]}, "relevant_chunk_ids"),
            ({"relevant_chunk_ids": [True]}, "relevant_chunk_ids"),
            ({"query": "  "}, "'query'"),
            ({"notes": 3}, "'notes'"),
            ({"expected_behavior": "maybe"}, "'expected_behavior'"),
            ({"mode": "web"}, "'mode'"),
            ({"category": 3}, "'category'"),
            ({"history": "hi"}, "'history'"),
            ({"history": ["hi"]}, r"'history\[0\]'"),
            ({"history": [{"role": "system", "content": "x"}]}, r"'history\[0\]\.role'"),
            ({"history": [{"role": "user", "content": 1}]}, r"'history\[0\]\.content'"),
            ({"must_not_contain": ["ok", " "]}, "'must_not_contain'"),
            ({"must_mention_arxiv_ids": [1]}, "'must_mention_arxiv_ids'"),
            ({"expect_rejected": "yes"}, "'expect_rejected'"),
            ({"open_arxiv_id": 5}, "'open_arxiv_id'"),
            ({"relevant_arxiv_ids": "2401.00001"}, "'relevant_arxiv_ids'"),
            ({"relevant_arxiv_ids": [], "mode": "agent"}, "answer cases"),
            ({"relevant_arxiv_ids": [], "expected_behavior": "refuse"}, "open_arxiv_id"),
        ],
    )
    def test_validation_errors_name_the_field_and_line(self, overrides: dict[str, Any], message: str) -> None:
        with pytest.raises(DatasetError, match=message) as excinfo:
            parse_lines([_line(), _line(id="q2", **overrides)], origin="queries.jsonl")
        assert "queries.jsonl:2" in str(excinfo.value)

    def test_duplicates_and_invalid_json_are_rejected(self) -> None:
        with pytest.raises(DatasetError, match="duplicate id"):
            parse_lines([_line(), _line()])
        with pytest.raises(DatasetError, match="invalid JSON"):
            parse_lines(["{not json"])

    def test_duplicate_relevant_ids_collapse(self) -> None:
        (query,) = parse_lines([_line(relevant_arxiv_ids=["a", "a", "b"], relevant_chunk_ids=[3, 3])])
        assert query.relevant_arxiv_ids == ("a", "b")
        assert query.relevant_chunk_ids == (3,)


class TestSchemaDefaults:
    def test_old_records_get_defaults_and_dump_without_new_keys(self) -> None:
        query = _parse()
        assert (query.category, query.expected_behavior, query.mode) == ("", "answer", "both")
        assert query.history == () and query.must_not_contain == () and query.must_mention_arxiv_ids == ()
        assert query.expect_rejected is False and query.open_arxiv_id == ""
        assert query.paper_chat_arxiv_id == "2401.00001"
        assert query.retrieval_eligible
        assert set(_parse(notes="").to_dict()) == {"id", "query", "lang", "relevant_arxiv_ids", "source", "notes"}

    def test_new_fields_roundtrip(self) -> None:
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

    def test_query_whitespace_is_preserved(self) -> None:
        assert _parse(query="  padded question \n").query == "  padded question \n"

    def test_expect_rejected_allows_empty_query_without_ids(self) -> None:
        query = _parse(query="", relevant_arxiv_ids=None, expect_rejected=True, expected_behavior="clarify")
        assert query.query == ""
        assert not query.retrieval_eligible

    def test_answer_case_accepts_must_mention_instead_of_relevant(self) -> None:
        query = _parse(relevant_arxiv_ids=[], must_mention_arxiv_ids=["2401.00009"], mode="agent")
        assert query.relevant_arxiv_ids == ()
        assert not query.retrieval_eligible

    def test_supports_mode(self) -> None:
        assert _parse().supports_mode("agent") and _parse().supports_mode("paper_chat")
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
    def test_retrieval_ineligible(self, overrides: dict[str, Any]) -> None:
        assert not _parse(**overrides).retrieval_eligible

    def test_clarify_case_with_ids_is_retrieval_eligible(self) -> None:
        assert _parse(expected_behavior="clarify").retrieval_eligible


class TestDirectoryLoading:
    def test_loads_all_jsonl_files_in_name_order(self, tmp_path) -> None:
        (tmp_path / "b.jsonl").write_text(_line(id="b1") + "\n", encoding="utf-8")
        (tmp_path / "a.jsonl").write_text(_line(id="a1") + "\n" + _line(id="a2") + "\n", encoding="utf-8")
        (tmp_path / "notes.json").write_text("{}", encoding="utf-8")
        assert [query.id for query in load_queries(tmp_path)] == ["a1", "a2", "b1"]

    def test_duplicate_ids_across_files(self, tmp_path) -> None:
        (tmp_path / "a.jsonl").write_text(_line(id="same") + "\n", encoding="utf-8")
        (tmp_path / "b.jsonl").write_text(_line(id="same") + "\n", encoding="utf-8")
        with pytest.raises(DatasetError, match="duplicate id 'same'"):
            load_queries(tmp_path)

    def test_digest_changes_with_content(self, tmp_path) -> None:
        (tmp_path / "a.jsonl").write_text(_line(id="a1") + "\n", encoding="utf-8")
        first = dataset_digest(tmp_path)
        (tmp_path / "b.jsonl").write_text(_line(id="b1") + "\n", encoding="utf-8")
        assert dataset_digest(tmp_path) != first
        assert len(first) == 12

    def test_missing_path(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_queries(tmp_path / "missing.jsonl")


class TestSelection:
    def test_select_by_lang_category_and_limit(self) -> None:
        queries = [
            _parse(id="a", lang="ko", category="language"),
            _parse(id="b", lang="en", category="language"),
            _parse(id="c", lang="en", category="safety"),
        ]
        assert [query.id for query in select_queries(queries, lang="en")] == ["b", "c"]
        assert [query.id for query in select_queries(queries, categories={"language"}, limit=1)] == ["a"]
        assert parse_categories(" language , ,safety") == {"language", "safety"}
        assert parse_categories(" , ") is None
        assert parse_categories(None) is None

    def test_placeholder_problems_look_at_every_id_field(self) -> None:
        queries = [
            _parse(id="clean"),
            _parse(id="query", query="about 0000.0000a"),
            _parse(id="open", open_arxiv_id="0000.0000b"),
            _parse(id="history", history=[{"role": "assistant", "content": "see 0000.0000c"}]),
        ]
        assert placeholder_problems(queries) == ["query", "open", "history"]


class TestBuilders:
    def test_longest_shared_word_run(self) -> None:
        source = "We propose a sparse attention mechanism that scales linearly with sequence length."
        assert longest_shared_word_run("a sparse attention mechanism that scales", source) == 6
        assert longest_shared_word_run("Linear-time attention for long inputs", source) == 1
        assert longest_shared_word_run("", source) == 0

    def test_known_item_builder_skips_empty_language(self) -> None:
        records = known_item_queries(
            arxiv_id="2401.00001", title="T", queries_by_lang={"ko": "질의", "en": " "}, basis="abstract"
        )
        assert [record.id for record in records] == ["ki-2401.00001-ko"]
        assert records[0].source == "known_item"

    def test_chunk_synth_builder(self) -> None:
        record = chunk_synth_query(
            chunk_id=7, arxiv_id="2401.00001", chunk_index=3, section_title="Method", question="Q?", lang="en"
        )
        assert record is not None and record.relevant_chunk_ids == (7,) and record.source == "llm_synth"
        empty = chunk_synth_query(chunk_id=7, arxiv_id="a", chunk_index=3, section_title="", question=" ", lang="en")
        assert empty is None


PLACEHOLDER_PAPERS = {"0000.0000a": ("2405.01234", "Direct Preference Optimization Revisited For Everyone")}


class TestPlaceholders:
    def test_fill_and_find_placeholders(self) -> None:
        payload = {
            "query": "{{title:0000.0000a}} vs {{title_head:0000.0000a:2}} and {{title:0000.0000b}}",
            "relevant_arxiv_ids": ["0000.0000a", "0000.0000b"],
            "history": [{"role": "assistant", "content": "[x](https://arxiv.org/abs/0000.0000a)"}],
        }
        filled = fill_placeholders(payload, PLACEHOLDER_PAPERS)
        assert filled["query"] == (
            "Direct Preference Optimization Revisited For Everyone vs Direct Preference and {{title:0000.0000b}}"
        )
        assert filled["relevant_arxiv_ids"] == ["2405.01234", "0000.0000b"]
        assert filled["history"][0]["content"] == "[x](https://arxiv.org/abs/2405.01234)"
        assert find_placeholders(filled) == {"0000.0000b"}
        assert payload["relevant_arxiv_ids"][0] == "0000.0000a"

    def test_set_placeholder_expands_in_lists_and_uses_first_paper_in_text(self) -> None:
        papers: dict[str, Any] = {
            "0000.0000a": ("2405.01234", "Single Paper"),
            "0000.0000r": [("2501.00001", "Diffusion One"), ("2501.00002", "Diffusion Two"), ("2405.01234", "Single")],
        }
        payload = {
            "query": "{{title_head:0000.0000r:2}} and 0000.0000r",
            "relevant_arxiv_ids": ["0000.0000r", "0000.0000a"],
            "must_mention_arxiv_ids": ["0000.0000a", "0000.0000r"],
        }
        filled = fill_placeholders(payload, papers)
        assert filled["query"] == "Diffusion One and 2501.00001"
        assert filled["relevant_arxiv_ids"] == ["2501.00001", "2501.00002", "2405.01234"]
        assert filled["must_mention_arxiv_ids"] == ["2405.01234", "2501.00001", "2501.00002"]
        assert fill_placeholders(["0000.0000z"], {"0000.0000z": []}) == ["0000.0000z"]
        assert find_placeholders(filled) == set()


class TestCatalogue:
    def test_loads_with_enough_cases_and_perspectives(self, eval_dir) -> None:
        cases = load_queries(eval_dir / "cases")
        categories = {case.category for case in cases}
        assert len(cases) >= MIN_CASES
        assert len(categories) >= MIN_PERSPECTIVES
        assert all(sum(1 for case in cases if case.category == category) >= 3 for category in categories)

    def test_one_file_per_perspective(self, eval_dir) -> None:
        for path in sorted((eval_dir / "cases").glob("*.jsonl")):
            assert {case.category for case in load_queries(path)} == {path.stem}

    def test_every_case_has_category_behavior_and_expected_papers(self, eval_dir) -> None:
        for case in load_queries(eval_dir / "cases"):
            assert case.category and case.expected_behavior in VALID_BEHAVIORS and case.source == "manual", case.id
            if case.expected_behavior == "answer" and not case.expect_rejected:
                assert case.relevant_arxiv_ids or case.must_mention_arxiv_ids, case.id

    def test_refuse_cases_check_something_beyond_the_phrase(self, eval_dir) -> None:
        refuse = [case for case in load_queries(eval_dir / "cases") if case.expected_behavior == "refuse"]
        assert refuse
        assert all(case.mode == "agent" or case.paper_chat_arxiv_id for case in refuse)

    def test_boundary_lengths_are_exact(self, eval_dir) -> None:
        by_id = {case.id: case for case in load_queries(eval_dir / "cases")}
        exact = by_id["sf-exactly-4000-chars"].query
        assert len(exact) == CHAT_MESSAGE_MAX_CHARS and exact == exact.strip()
        over = by_id["sf-4001-chars"].query
        assert len(over) == CHAT_MESSAGE_MAX_CHARS + 1 and over == over.strip()
        padded = by_id["sf-4000-chars-padded"].query
        assert len(padded) > CHAT_MESSAGE_MAX_CHARS and len(padded.strip()) == CHAT_MESSAGE_MAX_CHARS
        assert by_id["sf-4001-chars"].expect_rejected
        assert not by_id["sf-exactly-4000-chars"].expect_rejected

    def test_expect_rejected_matches_the_product_validation(self, eval_dir) -> None:
        for case in load_queries(eval_dir / "cases"):
            try:
                prepare_query(case)
            except InputRejected:
                rejected = True
            else:
                rejected = False
            assert rejected == case.expect_rejected, case.id

    def test_history_cases_cover_truncation(self, eval_dir) -> None:
        by_id = {case.id: case for case in load_queries(eval_dir / "cases")}
        long_history = by_id["cv-history-over-20-turns"]
        assert len(long_history.history) > CHAT_HISTORY_MAX_MESSAGES
        prepared = prepare_query(long_history)
        assert len(prepared.history) == CHAT_HISTORY_MAX_MESSAGES
        assert all("ZEBRA-19" not in content for _, content in prepared.history)
        assert "ZEBRA-19" in long_history.must_not_contain

    def test_placeholders_are_registered(self, eval_dir) -> None:
        registry = json.loads((eval_dir / "cases" / "placeholders.json").read_text(encoding="utf-8"))
        cases = load_queries(eval_dir / "cases")
        used = find_placeholders([case.to_dict() for case in cases])
        assert used
        assert used <= set(registry["placeholders"])
        assert set(registry["absent"]) <= {case.id for case in cases}

    def test_topic_cases_use_set_placeholders(self, eval_dir) -> None:
        registry = json.loads((eval_dir / "cases" / "placeholders.json").read_text(encoding="utf-8"))["placeholders"]
        topic_cases = {case.id: case for case in load_queries(eval_dir / "cases") if case.category == "query_form"}
        for case_id in ("qf-single-word-en", "qf-statement-en", "qf-numbers-units-ko", "qf-typo-key-term-en"):
            (placeholder,) = topic_cases[case_id].relevant_arxiv_ids
            assert registry[placeholder].get("all_matches") is True, case_id
        for case_id in ("qf-exact-title", "qf-partial-title"):
            (placeholder,) = topic_cases[case_id].relevant_arxiv_ids
            assert not registry[placeholder].get("all_matches"), case_id
