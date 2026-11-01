import csv
import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from arxplore.llm.paper_chat import NO_EXCERPTS_TEXT, format_paper_chat_context
from arxplore.llm.prompts.paper_chat import PAPER_CHAT_PROMPT
from arxplore_eval.answers import (
    SOURCES_FROM_REPLAY,
    SOURCES_FROM_RETRIEVAL,
    AnswersError,
    GenerationResult,
    PaperChatOptions,
    RecordedSource,
    RecordedSources,
    append_answer_record,
    build_answer_record,
    collect_answers,
    generate_paper_chat_answer,
    load_recorded_sources,
    replayed_retrieval,
)
from arxplore_eval.behavior import (
    answer_checks,
    answer_language,
    citation_numbers_valid,
    hangul_ratio,
    source_count,
)
from arxplore_eval.cli import app
from arxplore_eval.commands import generation as generation_commands
from arxplore_eval.compare_generation import (
    CompareGenerationError,
    RunValues,
    compare_generation,
    pair_generation,
    read_run,
    write_variance_csv,
)
from arxplore_eval.corpus import PreconditionError
from arxplore_eval.dataset import EvalQuery, dump_queries
from arxplore_eval.generation import (
    NullScorer,
    aggregate_scores,
    score_records,
    write_sample_csv,
    write_summary_csv,
)
from arxplore_eval.generation import (
    render_markdown as render_generation_markdown,
)
from arxplore_eval.prompt_variants import (
    PAPER_CHAT_PROMPT_BEFORE_LANGUAGE_RULE,
    PAPER_CHAT_PROMPT_VARIANTS,
    PROMPT_BEFORE,
    PROMPT_CURRENT,
    PROMPT_REFUSAL_RULE,
    paper_chat_prompt,
)
from tests.unit.eval.test_eval_paper_chat_ab import (
    ARXIV_ID,
    CountingRetriever,
    _contexts,
    _paper,
    _q,
    _service,
    _system_and_user,
)

runner = CliRunner()
BEFORE_SHA256 = "6fdef3d49780d525410677aa88dceb7337665e03d150055c4c4efd57dfc6e4ce"
HONORIFIC_RULE = "모든 답변은 정중한 존댓말"
LANGUAGE_RULE = "사용자의 마지막 질문과 같은 언어로"
ENGLISH_REFUSAL = "I cannot answer this from the provided excerpts."


def _system(template: Any) -> str:
    return str(template.messages[0].prompt.template)


class TestPromptVariantText:
    def test_before_system_hash_is_pinned(self) -> None:
        system = _system(PAPER_CHAT_PROMPT_BEFORE_LANGUAGE_RULE)
        assert hashlib.sha256(system.encode("utf-8")).hexdigest() == BEFORE_SHA256

    def test_before_has_the_honorific_rule_without_language_rule(self) -> None:
        system = _system(paper_chat_prompt(PROMPT_BEFORE))
        assert HONORIFIC_RULE in system
        assert LANGUAGE_RULE not in system
        assert ENGLISH_REFUSAL not in system
        assert "영어 질문" not in system

    def test_current_has_the_language_rule_without_blanket_honorific(self) -> None:
        system = _system(paper_chat_prompt(PROMPT_CURRENT))
        assert HONORIFIC_RULE not in system
        assert LANGUAGE_RULE in system
        assert ENGLISH_REFUSAL in system

    def test_current_is_the_product_constant(self) -> None:
        assert paper_chat_prompt(PROMPT_CURRENT) is PAPER_CHAT_PROMPT
        assert paper_chat_prompt("current") == PAPER_CHAT_PROMPT

    def test_variant_names(self) -> None:
        assert PAPER_CHAT_PROMPT_VARIANTS == (PROMPT_CURRENT, PROMPT_BEFORE, PROMPT_REFUSAL_RULE)
        assert PAPER_CHAT_PROMPT_VARIANTS == ("current", "before", "refusal_rule")

    @pytest.mark.parametrize("name", ["", "Current", "after", " before", "BEFORE"])
    def test_unknown_names_raise(self, name: str) -> None:
        with pytest.raises(ValueError, match="알 수 없는 상세 챗 프롬프트"):
            paper_chat_prompt(name)

    def test_options_validate_the_variant(self) -> None:
        with pytest.raises(ValueError, match="상세 챗 프롬프트"):
            PaperChatOptions(prompt_variant="after")


def _recorded(
    sources: list[RecordedSource], *, digest: str | None = None, mode: str = "hybrid", arxiv_id: str | None = ARXIV_ID
) -> RecordedSources:
    return RecordedSources(
        query_id="q", arxiv_id=arxiv_id, retrieval_mode=mode, sources=tuple(sources), context_sha256=digest
    )


def _digest_of(sources: list[RecordedSource]) -> str:
    rebuilt = replayed_retrieval(_paper(), _recorded(sources))
    return hashlib.sha256(format_paper_chat_context(rebuilt.sources).encode("utf-8")).hexdigest()


class TestReplayRetrieval:
    SOURCES: ClassVar[list[RecordedSource]] = [
        RecordedSource(None, "Abstract", "We study X.\n\nTwo lines."),
        RecordedSource(30, "4.2 Setup", "lr is 3e-4.\n| a | b |"),
        RecordedSource(12, None, "  다국어 발췌문 [1] 그대로  "),
        RecordedSource(7, "Empty", ""),
    ]

    def test_order_ids_titles_and_text_are_preserved(self) -> None:
        retrieval = replayed_retrieval(_paper(), _recorded(self.SOURCES, mode="lexical"))
        assert retrieval.retrieval_mode == "lexical"
        assert [(s.chunk_id, s.section_title, s.text) for s in retrieval.sources] == [
            (item.chunk_id, item.section_title, item.text) for item in self.SOURCES
        ]
        assert {s.arxiv_id for s in retrieval.sources} == {ARXIV_ID}
        assert {s.title for s in retrieval.sources} == {"Paper Title: 한글 subtitle"}

    def test_formatted_context_matches_the_recorded_digest(self) -> None:
        digest = _digest_of(self.SOURCES)
        retrieval = replayed_retrieval(_paper(), _recorded(self.SOURCES, digest=digest))
        formatted = format_paper_chat_context(retrieval.sources)
        assert hashlib.sha256(formatted.encode("utf-8")).hexdigest() == digest
        assert formatted.startswith("[1] 섹션: Abstract")
        assert "[4] 섹션: Empty | chunk_id: 7\n" in formatted

    def test_digest_mismatch_raises_and_missing_digest_is_not_checked(self) -> None:
        with pytest.raises(ValueError, match="context_sha256"):
            replayed_retrieval(_paper(), _recorded(self.SOURCES, digest="0" * 64))
        assert replayed_retrieval(_paper(), _recorded(self.SOURCES, digest=None)).sources

    def test_reordering_changes_the_digest(self) -> None:
        digest = _digest_of(self.SOURCES)
        with pytest.raises(ValueError, match="context_sha256"):
            replayed_retrieval(_paper(), _recorded(list(reversed(self.SOURCES)), digest=digest))

    def test_empty_sources_use_the_no_excerpts_text(self) -> None:
        digest = hashlib.sha256(NO_EXCERPTS_TEXT.encode("utf-8")).hexdigest()
        retrieval = replayed_retrieval(_paper(), _recorded([], digest=digest, mode="first_chunks"))
        assert retrieval.sources == ()
        assert retrieval.retrieval_mode == "first_chunks"
        assert format_paper_chat_context(retrieval.sources) == NO_EXCERPTS_TEXT
        with pytest.raises(ValueError, match="context_sha256"):
            replayed_retrieval(_paper(), _recorded([], digest="1" * 64))

    async def test_empty_and_first_chunks_records_replay_into_the_prompt(
        self, fake_openai, make_hit, make_context
    ) -> None:
        service = _service(fake_openai, CountingRetriever(_contexts(make_hit, make_context)))
        digest = hashlib.sha256(NO_EXCERPTS_TEXT.encode("utf-8")).hexdigest()
        replay = {"pc-1-en": _recorded([], digest=digest, mode="first_chunks")}

        result = await generate_paper_chat_answer(service, _q(), PaperChatOptions(replay=replay))

        _, user = _system_and_user(fake_openai.requests_for("/v1/chat/completions")[0])
        assert NO_EXCERPTS_TEXT in user
        assert result.contexts == []
        assert result.retrieval_mode == "first_chunks"
        assert result.paper_chat is not None
        assert result.paper_chat.sources == []
        assert result.paper_chat.sources_origin == SOURCES_FROM_REPLAY
        record = build_answer_record(_q(), "paper_chat", result=result)
        assert record["sources"] == [] and record["retrieval_mode"] == "first_chunks"

    async def test_history_is_sent_identically_under_replay(
        self, tmp_path, fake_openai, make_hit, make_context
    ) -> None:
        retriever = CountingRetriever(_contexts(make_hit, make_context))
        service = _service(fake_openai, retriever)
        query = _q(history=(("user", "earlier question"), ("assistant", "earlier reply")))
        direct = await generate_paper_chat_answer(service, query)
        direct_messages = fake_openai.requests_for("/v1/chat/completions")[0].body["messages"]
        path = tmp_path / "sources.jsonl"
        only = await generate_paper_chat_answer(service, query, PaperChatOptions(sources_only=True))
        append_answer_record(path, build_answer_record(query, "paper_chat", result=only))
        fake_openai.reset()

        await generate_paper_chat_answer(
            service, query, PaperChatOptions(prompt_variant=PROMPT_CURRENT, replay=load_recorded_sources(path))
        )

        replay_messages = fake_openai.requests_for("/v1/chat/completions")[0].body["messages"]
        assert replay_messages == direct_messages
        assert [m["content"] for m in replay_messages[1:3]] == ["earlier question", "earlier reply"]
        assert direct.paper_chat is not None

    async def test_replay_for_another_paper_is_refused(self, fake_openai, make_hit, make_context) -> None:
        service = _service(fake_openai, CountingRetriever(_contexts(make_hit, make_context)))
        replay = {"pc-1-en": _recorded([], arxiv_id="2401.99999")}
        with pytest.raises(LookupError, match="기록된 논문"):
            await generate_paper_chat_answer(service, _q(), PaperChatOptions(replay=replay))

    def test_loading_rejects_records_without_a_retrieval_mode(self, tmp_path) -> None:
        path = tmp_path / "s.jsonl"
        record = build_answer_record(_q(), "paper_chat", result=GenerationResult(answer="", contexts=[]))
        append_answer_record(path, {**record, "sources": [], "retrieval_mode": None})
        with pytest.raises(AnswersError, match="retrieval_mode"):
            load_recorded_sources(path)


class TestSourcesOnly:
    async def test_collect_makes_no_chat_call_and_records_empty_answers(
        self, fake_openai, make_hit, make_context
    ) -> None:
        retriever = CountingRetriever(_contexts(make_hit, make_context))
        service = _service(fake_openai, retriever)
        options = PaperChatOptions(prompt_variant=PROMPT_BEFORE, sources_only=True)
        generators = {"paper_chat": lambda query: generate_paper_chat_answer(service, query, options)}

        records = await collect_answers([_q("a-en"), _q("a-ko", lang="ko")], ["paper_chat"], generators, model=None)

        assert fake_openai.requests_for("/v1/chat/completions") == []
        assert len(retriever.calls) == 2
        for record in records:
            assert record["answer"] == ""
            assert record["sources_only"] is True
            assert record["prompt_variant"] is None
            assert record["model"] is None
            assert record["sources_origin"] == SOURCES_FROM_RETRIEVAL
            assert len(record["sources"]) == 3
            assert record["context_sha256"]

    def test_scoring_command_names_every_sources_only_record(self, clean_env) -> None:
        path = clean_env / "sources.jsonl"
        base = build_answer_record(_q(), "paper_chat", result=GenerationResult(answer="", contexts=[]))
        append_answer_record(path, {**base, "id": "x-en", "sources_only": True})
        append_answer_record(path, {**base, "id": "y-en", "sources_only": False, "answer": "ok"})
        append_answer_record(path, {**base, "id": "z-en", "sources_only": True})
        result = runner.invoke(app, ["generation", "--answers", str(path), "--behavior-only"])
        assert result.exit_code == 2
        assert "검색 전용" in result.stderr
        assert "'x-en', 'z-en'" in result.stderr and "y-en" not in result.stderr

    def test_compare_generation_command_refuses_sources_only_files(self, tmp_path) -> None:
        queries = tmp_path / "queries.jsonl"
        queries.write_text(dump_queries([_q()]), encoding="utf-8")
        path = tmp_path / "sources.jsonl"
        record = build_answer_record(_q(), "paper_chat", result=GenerationResult(answer="", contexts=[]))
        append_answer_record(path, {**record, "sources_only": True})
        result = runner.invoke(
            app,
            [
                "compare-generation",
                "--queries",
                str(queries),
                "--a",
                str(path),
                "--b",
                str(path),
                "--out-dir",
                str(tmp_path / "o"),
            ],
        )
        assert result.exit_code == 2
        assert "검색 전용" in result.stderr


class TestHangulRatio:
    @pytest.mark.parametrize(
        ("text", "ratio"),
        [
            ("", 0.0),
            ("12345 !!! [1] ...", 0.0),
            ("가나다", 1.0),
            ("ㅋㅋㅋ", 1.0),
            ("abc", 0.0),
            ("가나다" + "a" * 7, 0.3),
            ("가나 abc 123", 0.4),
        ],
    )
    def test_ratio_counts_letters_only(self, text: str, ratio: float) -> None:
        assert hangul_ratio(text) == pytest.approx(ratio)

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("", ""),
            ("\n\t ", ""),
            ("1234567890", "en"),
            ("[1][2] ... !!!", "en"),
            ("가나다" + "a" * 7, "en"),
            ("가나다" + "a" * 6, "ko"),
            ("학습률은 LoRA 방법입니다 3e-4", "ko"),
            ("The LoRA 방법은 efficient 합니다", "en"),
        ],
    )
    def test_language_boundaries(self, answer: str, expected: str) -> None:
        assert answer_language(answer) == expected

    @pytest.mark.parametrize(
        ("lang", "answer", "expected"),
        [
            ("en", "The answer is 3e-4.", 1.0),
            ("en", "학습률은 3e-4입니다.", 0.0),
            ("ko", "학습률은 3e-4입니다.", 1.0),
            ("ko", "The answer is 3e-4.", 0.0),
        ],
    )
    def test_lang_match_compares_with_the_query_language(self, lang: str, answer: str, expected: float) -> None:
        record = {"mode": "agent", "lang": lang, "answer": answer}
        assert answer_checks(record, "answered")[1] == {"lang_match": expected}

    def test_lang_match_is_absent_without_a_known_query_lang(self) -> None:
        for lang in ("", None, "fr"):
            language, scores = answer_checks({"mode": "agent", "lang": lang, "answer": "The answer"}, "answered")
            assert (language, scores) == ("en", {})

    @pytest.mark.parametrize("outcome", ["rejected", "error"])
    def test_no_checks_for_rejected_and_error(self, outcome: str) -> None:
        assert answer_checks({"mode": "paper_chat", "lang": "en", "answer": "x"}, outcome) == ("", {})

    def test_refused_answers_are_checked_too(self) -> None:
        record = {"mode": "paper_chat", "lang": "en", "answer": ENGLISH_REFUSAL, "contexts": ["a"]}
        assert answer_checks(record, "refused") == ("en", {"lang_match": 1.0, "citation_numbers_valid": 1.0})


class TestCitationNumbers:
    @pytest.mark.parametrize(
        ("answer", "sources", "expected"),
        [
            ("[0]", 3, False),
            ("[1][0]", 3, False),
            ("[4]", 3, False),
            ("[3]", 3, True),
            ("[1] and [1] again [1]", 1, True),
            ("[1][1][4]", 3, False),
            ("[1][2]", 2, True),
            ("[1][2][3]", 2, False),
            ("no refs at all", 3, True),
            ("no refs at all", 0, True),
            ("", 0, True),
            ("see [1]", 0, False),
            ("see [2]", 1, False),
            ("[1, 2, 3]", 3, True),
            ("[1, 5]", 3, False),
        ],
    )
    def test_numbers_must_fall_in_range(self, answer: str, sources: int, expected: bool) -> None:
        assert citation_numbers_valid(answer, sources) is expected

    def test_source_count_prefers_recorded_sources_over_contexts(self) -> None:
        assert source_count({"sources": [{}, {}, {}], "contexts": ["a"]}) == 3
        assert source_count({"sources": [], "contexts": ["a", "b"]}) == 0
        assert source_count({"contexts": ["a", "  ", "b", ""]}) == 2
        assert source_count({}) == 0

    def test_zero_sources_with_a_number_scores_zero_in_the_record(self) -> None:
        record = {"mode": "paper_chat", "lang": "en", "answer": "see [1]", "sources": []}
        assert answer_checks(record, "answered")[1]["citation_numbers_valid"] == 0.0


def _eq(query_id: str, lang: str = "en", source: str = "known_item", **kwargs: Any) -> EvalQuery:
    return EvalQuery(id=query_id, query="q", lang=lang, relevant_arxiv_ids=("A",), source=source, **kwargs)


QUERIES = [
    _eq("p1-ko", "ko"),
    _eq("p1-en", "en"),
    _eq("p2-ko", "ko", "manual", category="paper_chat"),
    _eq("p2-en", "en", "manual", category="paper_chat"),
    _eq("r1-ko", "ko", "manual", expected_behavior="refuse", open_arxiv_id="A"),
    _eq("r1-en", "en", "manual", expected_behavior="refuse", open_arxiv_id="A"),
    _eq("s1-en", "en", "llm_synth"),
]


def _csv(path: Path, rows: dict[str, dict[str, str]], *, columns: tuple[str, ...] = ("faithfulness",)) -> Path:
    fields = ["id", "mode", "lang", "outcome", *columns, "lang_match", "citation_numbers_valid"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        for query_id, values in rows.items():
            writer.writerow({"id": query_id, "mode": "paper_chat", "outcome": "answered", **values})
    return path


def _runs(paths: list[Path]) -> list[RunValues]:
    return [read_run(path) for path in paths]


def _compare(a: list[Path], b: list[Path], queries: list[EvalQuery] = QUERIES, **kwargs: Any) -> Any:
    paired = pair_generation(_runs(a), _runs(b), queries, label_a="A", label_b="B")
    return compare_generation(paired, resamples=kwargs.pop("resamples", 300), seed=kwargs.pop("seed", 7), **kwargs)


def _all_rows(ids: list[str], **values: str) -> dict[str, dict[str, str]]:
    return {query_id: dict(values) for query_id in ids}


IDS = [query.id for query in QUERIES]


class TestCompareGenerationValues:
    def test_per_query_mean_over_repetitions_and_delta_sign(self, tmp_path) -> None:
        a = [
            _csv(tmp_path / "a1.csv", {"p1-en": {"faithfulness": "0.2"}, "p1-ko": {"faithfulness": "0.4"}}),
            _csv(tmp_path / "a2.csv", {"p1-en": {"faithfulness": "0.4"}, "p1-ko": {"faithfulness": "0.6"}}),
            _csv(tmp_path / "a3.csv", {"p1-en": {"faithfulness": "0.6"}, "p1-ko": {"faithfulness": "0.8"}}),
        ]
        b = [_csv(tmp_path / "b1.csv", {"p1-en": {"faithfulness": "0.9"}, "p1-ko": {"faithfulness": "0.7"}})]

        comparison = _compare(a, b)

        overall = {row.metric: row.result for row in comparison.deltas if row.subset == "all"}["faithfulness"]
        assert (overall.mean_a, overall.mean_b) == (pytest.approx(0.5), pytest.approx(0.8))
        assert overall.delta == pytest.approx(0.3)
        assert overall.low <= overall.delta <= overall.high
        pairs = {pair.query.id: pair for pair in comparison.paired.pairs}
        assert pairs["p1-en"].a.mean("faithfulness") == pytest.approx(0.4)
        assert pairs["p1-en"].a.count("faithfulness") == 3
        assert pairs["p1-ko"].b.stdev("faithfulness") is None
        assert pairs["p1-ko"].a.stdev("faithfulness") == pytest.approx(0.2)

    def test_negative_delta_when_b_is_lower(self, tmp_path) -> None:
        a = [_csv(tmp_path / "a.csv", _all_rows(IDS, faithfulness="0.9"))]
        b = [_csv(tmp_path / "b.csv", _all_rows(IDS, faithfulness="0.5"))]
        row = next(row for row in _compare(a, b).deltas if row.subset == "all" and row.metric == "faithfulness")
        assert row.result.delta == pytest.approx(-0.4)
        assert row.result.excludes_zero is True

    def test_same_seed_reproduces_the_interval(self, tmp_path) -> None:
        values_a = {query_id: {"faithfulness": f"{0.3 + 0.07 * index:.2f}"} for index, query_id in enumerate(IDS)}
        values_b = {query_id: {"faithfulness": f"{0.6 - 0.05 * index:.2f}"} for index, query_id in enumerate(IDS)}
        a = [_csv(tmp_path / "a.csv", values_a)]
        b = [_csv(tmp_path / "b.csv", values_b)]

        first = _compare(a, b, seed=11, resamples=500)
        second = _compare(a, b, seed=11, resamples=500)
        other = _compare(a, b, seed=12, resamples=500)

        def interval(comparison: Any) -> tuple[float, float]:
            row = next(row for row in comparison.deltas if row.subset == "all" and row.metric == "faithfulness")
            return row.result.low, row.result.high

        assert interval(first) == interval(second)
        assert interval(first) != interval(other)

    def test_translation_pairs_are_one_cluster(self, tmp_path) -> None:
        a = [_csv(tmp_path / "a.csv", _all_rows(IDS, faithfulness="0.2"))]
        b = [_csv(tmp_path / "b.csv", _all_rows(IDS, faithfulness="0.4"))]
        overall = {row.subset: row.result for row in _compare(a, b).deltas if row.metric == "faithfulness"}
        assert (overall["all"].n, overall["all"].n_clusters) == (7, 4)
        assert (overall["ko"].n, overall["ko"].n_clusters) == (3, 3)
        assert (overall["en"].n, overall["en"].n_clusters) == (4, 4)

    def test_subsets_cover_lang_source_category_and_behavior(self, tmp_path) -> None:
        a = [_csv(tmp_path / "a.csv", _all_rows(IDS, faithfulness="0.2"))]
        b = [_csv(tmp_path / "b.csv", _all_rows(IDS, faithfulness="0.4"))]
        comparison = _compare(a, b)
        sizes = {row.subset: row.result.n for row in comparison.deltas if row.metric == "faithfulness"}
        assert sizes == {
            "all": 7,
            "ko": 3,
            "en": 4,
            "known_item": 2,
            "llm_synth": 1,
            "manual": 4,
            "category:paper_chat": 2,
            "behavior:answer": 5,
            "behavior:refuse": 2,
        }
        assert comparison.paired.subsets() == [
            "all",
            "ko",
            "en",
            "known_item",
            "llm_synth",
            "manual",
            "category:paper_chat",
            "behavior:answer",
            "behavior:refuse",
        ]

    def test_refusal_two_by_two(self, tmp_path) -> None:
        refused = {"outcome": "refused"}
        a_rows = {
            "r1-ko": refused,
            "r1-en": refused,
            "p1-ko": refused,
            "p1-en": {},
            "p2-ko": refused,
            "p2-en": {},
        }
        b_rows = {
            "r1-ko": refused,
            "r1-en": {},
            "p1-ko": {},
            "p1-en": {},
            "p2-ko": refused,
            "p2-en": refused,
        }
        a = [_csv(tmp_path / "a.csv", a_rows)]
        b = [_csv(tmp_path / "b.csv", b_rows)]
        paired = pair_generation(_runs(a), _runs(b), QUERIES, label_a="A", label_b="B")
        refusal = {row.behavior: row for row in compare_generation(paired, resamples=50).refusal}

        keep, over = refusal["refuse"], refusal["answer"]
        assert (keep.n, keep.rate_a, keep.rate_b) == (2, 1.0, 0.5)
        assert keep.transitions == {"both_refused": 1, "a_only": 1, "b_only": 0, "neither": 0}
        assert keep.query_ids["a_only"] == ("r1-en",)
        assert (over.n, over.rate_a, over.rate_b) == (4, 0.5, 0.5)
        assert over.transitions == {"both_refused": 1, "a_only": 1, "b_only": 1, "neither": 1}
        assert over.query_ids["both_refused"] == ("p2-ko",)
        assert over.query_ids["a_only"] == ("p1-ko",)
        assert over.query_ids["b_only"] == ("p2-en",)

    def test_variance_with_different_repetition_counts(self, tmp_path) -> None:
        ids = ["p1-ko", "p1-en"]
        a = [
            _csv(tmp_path / f"a{index}.csv", _all_rows(ids, faithfulness=value))
            for index, value in enumerate(["0.2", "0.4", "0.6"], start=1)
        ]
        b = [_csv(tmp_path / "b.csv", _all_rows(ids, faithfulness="0.5"))]
        queries = [query for query in QUERIES if query.id in ids]
        comparison = _compare(a, b, queries)
        variance = {(row.label, row.metric): row for row in comparison.variance}

        a_row, b_row = variance[("A", "faithfulness")], variance[("B", "faithfulness")]
        assert (a_row.reps, a_row.n_queries) == (3, 2)
        assert a_row.mean_stdev == pytest.approx(0.2) and a_row.max_stdev == pytest.approx(0.2)
        assert a_row.rep_means == (pytest.approx(0.2), pytest.approx(0.4), pytest.approx(0.6))
        assert (b_row.reps, b_row.n_queries, b_row.mean_stdev, b_row.max_stdev) == (1, 0, None, None)
        path = tmp_path / "variance.csv"
        write_variance_csv(comparison.variance, path)
        with path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            assert reader.fieldnames is not None and reader.fieldnames[-3:] == ["rep1_mean", "rep2_mean", "rep3_mean"]
        by_label = {(row["label"], row["metric"]): row for row in rows}
        assert by_label[("B", "faithfulness")]["rep2_mean"] == ""
        assert by_label[("A", "faithfulness")]["rep3_mean"] == "0.6000"

    def test_queries_on_one_side_only_are_dropped_with_a_warning(self, tmp_path) -> None:
        a = [_csv(tmp_path / "a.csv", _all_rows(["p1-ko", "p1-en", "p2-en"], faithfulness="0.2"))]
        b = [_csv(tmp_path / "b.csv", _all_rows(["p1-ko", "p1-en", "s1-en"], faithfulness="0.4"))]
        paired = pair_generation(_runs(a), _runs(b), QUERIES, label_a="A", label_b="B")
        assert [pair.query.id for pair in paired.pairs] == ["p1-ko", "p1-en"]
        assert any(
            "한쪽 조건에만 있는 질의 2개" in line and "p2-en" in line and "s1-en" in line for line in paired.warnings
        )

    def test_no_common_queries_is_an_error(self, tmp_path) -> None:
        a = [_csv(tmp_path / "a.csv", _all_rows(["p1-ko"], faithfulness="0.2"))]
        b = [_csv(tmp_path / "b.csv", _all_rows(["p1-en"], faithfulness="0.4"))]
        with pytest.raises(CompareGenerationError, match="공통 질의가 없습니다"):
            pair_generation(_runs(a), _runs(b), QUERIES, label_a="A", label_b="B")
        with pytest.raises(CompareGenerationError, match="하나 이상씩"):
            pair_generation([], _runs(b), QUERIES, label_a="A", label_b="B")

    def test_missing_scores_are_left_out_per_metric(self, tmp_path) -> None:
        a = [_csv(tmp_path / "a.csv", {"p1-ko": {"faithfulness": "0.2"}, "p1-en": {"faithfulness": "0.4"}})]
        b = [_csv(tmp_path / "b.csv", {"p1-ko": {"faithfulness": ""}, "p1-en": {"faithfulness": "0.8"}})]
        comparison = _compare(a, b)
        row = next(row for row in comparison.deltas if row.subset == "all" and row.metric == "faithfulness")
        assert row.result.n == 1
        assert (row.result.mean_a, row.result.mean_b) == (pytest.approx(0.4), pytest.approx(0.8))
        assert len(comparison.paired.pairs) == 2

    def test_a_metric_missing_on_one_side_is_not_compared(self, tmp_path) -> None:
        a = [_csv(tmp_path / "a.csv", _all_rows(["p1-ko", "p1-en"], faithfulness="0.2"))]
        b = [_csv(tmp_path / "b.csv", _all_rows(["p1-ko", "p1-en"]), columns=())]
        paired = pair_generation(_runs(a), _runs(b), QUERIES, label_a="A", label_b="B")
        assert "faithfulness" not in paired.metrics
        assert "refused" in paired.metrics

    def test_error_outcomes_carry_no_refused_value(self, tmp_path) -> None:
        a = [_csv(tmp_path / "a.csv", {"p1-ko": {"outcome": "error"}, "p1-en": {"outcome": "answered"}})]
        b = [_csv(tmp_path / "b.csv", {"p1-ko": {"outcome": "refused"}, "p1-en": {"outcome": "refused"}})]
        comparison = _compare(a, b)
        refused = next(row for row in comparison.deltas if row.subset == "all" and row.metric == "refused")
        assert refused.result.n == 1
        assert (refused.result.mean_a, refused.result.mean_b) == (0.0, 1.0)
        assert any("rejected·error" in line for line in comparison.paired.warnings)

    async def test_csv_and_jsonl_inputs_give_the_same_comparison(self, tmp_path) -> None:
        answers = {
            "p1-ko": "학습률은 3e-4입니다 [1].",
            "p1-en": "The rate is 3e-4 [1].",
            "p2-ko": "I cannot answer this from the provided excerpts.",
            "p2-en": "죄송하지만 답하기 어렵습니다 [3].",
        }
        queries = [query for query in QUERIES if query.id in answers]
        records_a = [
            build_answer_record(query, "paper_chat", result=GenerationResult(answer=text, contexts=["c1", "c2"]))
            for query, text in ((q, answers[q.id]) for q in queries)
        ]
        flipped = {key: value for key, value in zip(answers, reversed(list(answers.values())), strict=True)}
        records_b = [
            build_answer_record(query, "paper_chat", result=GenerationResult(answer=flipped[query.id], contexts=["c1"]))
            for query in queries
        ]
        paths: dict[str, tuple[Path, Path]] = {}
        for label, records in (("a", records_a), ("b", records_b)):
            jsonl = tmp_path / f"{label}.jsonl"
            for record in records:
                append_answer_record(jsonl, record)
            samples = await score_records(records, [], NullScorer())
            sample_csv = tmp_path / f"{label}.csv"
            write_sample_csv(samples, [], sample_csv)
            paths[label] = (jsonl, sample_csv)

        from_jsonl = _compare([paths["a"][0]], [paths["b"][0]], queries)
        from_csv = _compare([paths["a"][1]], [paths["b"][1]], queries)

        def summary(comparison: Any) -> list[tuple[Any, ...]]:
            return [
                (
                    row.subset,
                    row.metric,
                    row.result.n,
                    row.result.mean_a,
                    row.result.mean_b,
                    row.result.low,
                    row.result.high,
                )
                for row in comparison.deltas
            ]

        assert summary(from_jsonl) == summary(from_csv)
        assert {row.metric for row in from_jsonl.deltas} >= {"lang_match", "refused", "citation_numbers_valid"}
        assert [row.refusal for row in (from_jsonl,)] and from_jsonl.refusal == from_csv.refusal
        assert from_jsonl.paired.runs_a[0].kind == "answers" and from_csv.paired.runs_a[0].kind == "samples"


class TestCompareGenerationCli:
    def _files(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        queries = tmp_path / "queries.jsonl"
        queries.write_text(dump_queries(QUERIES), encoding="utf-8")
        a = _csv(tmp_path / "a.csv", _all_rows(IDS, faithfulness="0.3"))
        b = _csv(tmp_path / "b.csv", _all_rows(IDS, faithfulness="0.6"))
        return queries, a, b

    def _argv(self, tmp_path: Path, *extra: str) -> list[str]:
        queries, a, b = self._files(tmp_path)
        return ["compare-generation", "--queries", str(queries), "--a", str(a), "--b", str(b), *extra]

    def test_same_seed_gives_identical_files(self, tmp_path) -> None:
        for name in ("one", "two"):
            result = runner.invoke(
                app, self._argv(tmp_path, "--seed", "5", "--resamples", "80", "--out-dir", str(tmp_path / name))
            )
            assert result.exit_code == 0, result.output
        first = (tmp_path / "one" / "compare_generation_metrics.csv").read_text(encoding="utf-8")
        second = (tmp_path / "two" / "compare_generation_metrics.csv").read_text(encoding="utf-8")
        assert first == second
        assert "faithfulness" in first

    def test_labels_and_confidence_reach_the_report(self, tmp_path) -> None:
        argv = self._argv(
            tmp_path,
            "--label-a",
            "before",
            "--label-b",
            "after",
            "--confidence",
            "0.9",
            "--resamples",
            "60",
            "--out-dir",
            str(tmp_path / "o"),
        )
        result = runner.invoke(app, argv)
        assert result.exit_code == 0, result.output
        markdown = (tmp_path / "o" / "compare_generation.md").read_text(encoding="utf-8")
        assert "# 생성 비교: after vs before" in markdown
        assert "90% 구간" in markdown

    @pytest.mark.parametrize(
        "extra",
        [
            ["--resamples", "0"],
            ["--confidence", "0.4"],
            ["--confidence", "1.0"],
            ["--seed", "abc"],
            ["--mode", "bogus"],
            ["--refusal-phrases", "/nonexistent/phrases.txt"],
        ],
    )
    def test_invalid_options_exit_2(self, tmp_path, extra: list[str]) -> None:
        result = runner.invoke(app, self._argv(tmp_path, *extra, "--out-dir", str(tmp_path / "o")))
        assert result.exit_code == 2

    def test_missing_options_and_files_exit_2(self, tmp_path) -> None:
        queries, a, b = self._files(tmp_path)
        assert runner.invoke(app, ["compare-generation", "--queries", str(queries), "--a", str(a)]).exit_code == 2
        assert runner.invoke(app, ["compare-generation", "--queries", str(queries), "--b", str(b)]).exit_code == 2
        missing = ["compare-generation", "--queries", str(queries), "--a", str(tmp_path / "no.csv"), "--b", str(b)]
        assert runner.invoke(app, missing).exit_code == 2
        no_queries = ["compare-generation", "--queries", str(tmp_path / "no.jsonl"), "--a", str(a), "--b", str(b)]
        assert runner.invoke(app, no_queries).exit_code == 2

    def test_disjoint_runs_exit_2(self, tmp_path) -> None:
        queries, _, _ = self._files(tmp_path)
        left = _csv(tmp_path / "l.csv", _all_rows(["p1-ko"], faithfulness="0.1"))
        right = _csv(tmp_path / "r.csv", _all_rows(["p1-en"], faithfulness="0.1"))
        result = runner.invoke(
            app, ["compare-generation", "--queries", str(queries), "--a", str(left), "--b", str(right)]
        )
        assert result.exit_code == 2
        assert "짝지을 수 없습니다" in result.stderr


def _options(tmp_path: Path, **kwargs: Any) -> generation_commands.CollectionOptions:
    base: dict[str, Any] = {
        "queries": tmp_path / "q.jsonl",
        "modes": ["paper_chat"],
        "lang": None,
        "limit": None,
        "category": None,
        "out": tmp_path / "out.jsonl",
        "answer_model": "m",
        "keep_going": False,
        "base_url": None,
        "trace": False,
        "refusal_phrases": [],
    }
    base.update(kwargs)
    return generation_commands.CollectionOptions(**base)


def _source_record(query: EvalQuery, **extra: Any) -> dict[str, Any]:
    record = build_answer_record(
        query, "paper_chat", result=GenerationResult(answer="", contexts=["c"], retrieval_mode="hybrid")
    )
    record.update({"sources": [{"chunk_id": 1, "section_title": "A", "text": "c"}], "context_sha256": None, **extra})
    return record


class TestCollectionPreconditions:
    def test_setup_tuple_follows_the_flags(self, tmp_path) -> None:
        assert _options(tmp_path).paper_chat_setup == (PROMPT_CURRENT, SOURCES_FROM_RETRIEVAL)
        assert _options(tmp_path, paper_chat_prompt=PROMPT_BEFORE).paper_chat_setup == (
            PROMPT_BEFORE,
            SOURCES_FROM_RETRIEVAL,
        )
        assert _options(tmp_path, sources_only=True).paper_chat_setup == ("sources_only", SOURCES_FROM_RETRIEVAL)
        replay = _options(tmp_path, paper_chat_prompt=PROMPT_BEFORE, replay_sources=tmp_path / "s.jsonl")
        assert replay.paper_chat_setup == (PROMPT_BEFORE, SOURCES_FROM_REPLAY)

    def test_resume_checks_sources_only_files_against_the_wanted_setup(self, tmp_path) -> None:
        path = tmp_path / "out.jsonl"
        append_answer_record(path, _source_record(_q(), sources_only=True, sources_origin=SOURCES_FROM_RETRIEVAL))
        done = generation_commands._done_keys(path, None, ("sources_only", SOURCES_FROM_RETRIEVAL))
        assert done == {("pc-1-en", "paper_chat")}
        with pytest.raises(PreconditionError, match="다른 --out"):
            generation_commands._done_keys(path, None, (PROMPT_CURRENT, SOURCES_FROM_RETRIEVAL))

    def test_resume_ignores_failed_records_and_other_modes(self, tmp_path) -> None:
        path = tmp_path / "out.jsonl"
        failed = build_answer_record(_q("bad-en"), "paper_chat", error="boom")
        append_answer_record(path, {**failed, "prompt_variant": PROMPT_BEFORE, "sources_origin": SOURCES_FROM_REPLAY})
        agent = build_answer_record(_q("ag-en"), "agent", result=GenerationResult(answer="a", contexts=[]))
        append_answer_record(path, {**agent, "prompt_variant": PROMPT_BEFORE})
        generation_commands._done_keys(path, None, (PROMPT_CURRENT, SOURCES_FROM_RETRIEVAL))

    def test_resume_without_a_paper_chat_setup_skips_the_check(self, tmp_path) -> None:
        path = tmp_path / "out.jsonl"
        append_answer_record(
            path, _source_record(_q(), prompt_variant=PROMPT_BEFORE, sources_origin=SOURCES_FROM_REPLAY)
        )
        assert generation_commands._done_keys(path, None, None) == {("pc-1-en", "paper_chat")}

    def test_replay_file_missing_queries_blocks_the_start(self, tmp_path) -> None:
        sources = tmp_path / "sources.jsonl"
        append_answer_record(sources, _source_record(_q("have-en")))
        queries = [_q("have-en"), _q("lost-en"), _q("lost-ko", lang="ko")]
        options = _options(tmp_path, replay_sources=sources)
        with pytest.raises(PreconditionError, match=r"질의 2개: \['lost-en', 'lost-ko'\]"):
            generation_commands.load_replay(options, queries)

    def test_keep_going_lets_missing_queries_through(self, tmp_path) -> None:
        sources = tmp_path / "sources.jsonl"
        append_answer_record(sources, _source_record(_q("have-en")))
        options = _options(tmp_path, replay_sources=sources, keep_going=True)
        replay = generation_commands.load_replay(options, [_q("have-en"), _q("lost-en")])
        assert replay is not None and list(replay) == ["have-en"]

    def test_rejected_input_and_other_mode_queries_are_not_required(self, tmp_path) -> None:
        sources = tmp_path / "sources.jsonl"
        append_answer_record(sources, _source_record(_q("have-en")))
        rejected = _q("empty-en", query="   ")
        agent_only = _q("agent-en", mode="agent")
        options = _options(tmp_path, replay_sources=sources)
        replay = generation_commands.load_replay(options, [_q("have-en"), rejected, agent_only])
        assert replay is not None and set(replay) == {"have-en"}

    def test_replay_file_errors_become_preconditions(self, tmp_path) -> None:
        with pytest.raises(PreconditionError, match="재생할 소스 파일이 없습니다"):
            generation_commands.load_replay(_options(tmp_path, replay_sources=tmp_path / "no.jsonl"), [_q()])
        broken = tmp_path / "broken.jsonl"
        record = _source_record(_q())
        record.pop("sources")
        append_answer_record(broken, record)
        with pytest.raises(PreconditionError, match="재생할 소스 파일 형식 오류"):
            generation_commands.load_replay(_options(tmp_path, replay_sources=broken), [_q()])

    def test_no_replay_option_means_no_replay(self, tmp_path) -> None:
        assert generation_commands.load_replay(_options(tmp_path), [_q()]) is None

    def test_failed_records_in_the_sources_file_are_not_replayed(self, tmp_path) -> None:
        sources = tmp_path / "sources.jsonl"
        append_answer_record(sources, build_answer_record(_q("bad-en"), "paper_chat", error="boom"))
        options = _options(tmp_path, replay_sources=sources)
        with pytest.raises(PreconditionError, match="bad-en"):
            generation_commands.load_replay(options, [_q("bad-en")])

    def test_replay_and_sources_only_together_exit_2(self, clean_env) -> None:
        argv = ["generation", "--collect", "--mode", "paper_chat", "--sources-only", "--replay-sources", "s.jsonl"]
        result = runner.invoke(app, argv)
        assert result.exit_code == 2 and "함께 쓸 수 없습니다" in result.stderr

    def test_replay_needs_paper_chat_only(self, clean_env) -> None:
        for mode in ("agent", "both"):
            result = runner.invoke(app, ["generation", "--collect", "--mode", mode, "--replay-sources", "s.jsonl"])
            assert result.exit_code == 2
            assert "--mode paper_chat에서만" in result.stderr

    def test_paper_chat_prompt_with_agent_only_exits_2(self, clean_env) -> None:
        result = runner.invoke(app, ["generation", "--collect", "--mode", "agent", "--paper-chat-prompt", "before"])
        assert result.exit_code == 2
        assert "paper_chat 모드를 수집할 때만" in result.stderr

    def test_paper_chat_prompt_with_both_modes_passes_the_flag_checks(self, clean_env) -> None:
        result = runner.invoke(app, ["generation", "--collect", "--paper-chat-prompt", "before"])
        assert result.exit_code == 2
        assert "paper_chat 모드를 수집할 때만" not in result.stderr
        assert "OPENAI_API_KEY" in result.stderr

    def test_unknown_prompt_name_exits_2(self, clean_env) -> None:
        result = runner.invoke(app, ["generation", "--collect", "--paper-chat-prompt", "after"])
        assert result.exit_code == 2


class TestReports:
    async def _samples(self) -> list[Any]:
        records = [
            build_answer_record(
                _q("a-en"), "paper_chat", result=GenerationResult(answer="The rate [1].", contexts=["c"])
            ),
            build_answer_record(
                _q("b-en"), "paper_chat", result=GenerationResult(answer="학습률은 3e-4입니다 [2].", contexts=["c"])
            ),
            build_answer_record(
                _q("c-ko", lang="ko"), "paper_chat", result=GenerationResult(answer="학습률입니다.", contexts=["c"])
            ),
            build_answer_record(_q("d-ko", lang="ko"), "paper_chat", error="boom"),
        ]
        return await score_records(records, [], NullScorer())

    async def test_sample_csv_appends_the_new_columns_last(self, tmp_path) -> None:
        samples = await self._samples()
        write_sample_csv(samples, ["faithfulness"], tmp_path / "s.csv")
        with (tmp_path / "s.csv").open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            header = list(reader.fieldnames or [])
        assert header[:10] == [
            "id",
            "mode",
            "lang",
            "source",
            "category",
            "expected_behavior",
            "outcome",
            "n_contexts",
            "reference_kind",
            "generation_error",
        ]
        assert header[10:13] == ["faithfulness", "faithfulness_skipped", "faithfulness_error"]
        assert header[13:18] == [
            "refusal_correct",
            "no_fabricated_links",
            "must_not_contain_ok",
            "mentions_required_ids",
            "rejected_as_expected",
        ]
        assert header[18:] == ["answer_lang", "lang_match", "citation_numbers_valid", "cited_relevant"]
        assert [(row["id"], row["answer_lang"], row["lang_match"], row["citation_numbers_valid"]) for row in rows] == [
            ("a-en", "en", "1.0000", "1.0000"),
            ("b-en", "ko", "0.0000", "0.0000"),
            ("c-ko", "ko", "1.0000", "1.0000"),
            ("d-ko", "", "", ""),
        ]

    async def test_summary_csv_appends_the_new_columns_last(self, tmp_path) -> None:
        samples = await self._samples()
        write_summary_csv(aggregate_scores(samples, []), [], tmp_path / "sum.csv")
        with (tmp_path / "sum.csv").open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = {row["subset"]: row for row in reader}
            header = list(reader.fieldnames or [])
        assert header[:5] == ["mode", "subset", "n_records", "n_generation_errors", "n_metric_errors"]
        assert header[5:9] == ["n_answered", "n_refused", "n_rejected", "n_error"]
        assert header[-6:] == [
            "lang_match",
            "lang_match_n",
            "citation_numbers_valid",
            "citation_numbers_valid_n",
            "cited_relevant",
            "cited_relevant_n",
        ]
        assert header[9:-6] == [
            "refusal_correct",
            "refusal_correct_n",
            "no_fabricated_links",
            "no_fabricated_links_n",
            "must_not_contain_ok",
            "must_not_contain_ok_n",
            "mentions_required_ids",
            "mentions_required_ids_n",
            "rejected_as_expected",
            "rejected_as_expected_n",
        ]
        assert rows["all"]["lang_match"] == "0.6667" and rows["all"]["lang_match_n"] == "3"
        assert rows["en"]["citation_numbers_valid"] == "0.5000"

    async def test_markdown_has_the_answer_language_table(self) -> None:
        samples = await self._samples()
        markdown = render_generation_markdown(aggregate_scores(samples, []), samples, [], title="t", meta={"k": "v"})
        assert "## 답변 언어 (질의 lang x 답변 언어)" in markdown
        assert "| mode | 질의 lang | 답변 수 | 답변 ko | 답변 en |" in markdown
        assert "| paper_chat | ko | 1 | 1 | 0 |" in markdown
        assert "| paper_chat | en | 2 | 1 | 1 |" in markdown
        assert "`lang_match`" in markdown and "`citation_numbers_valid`" in markdown

    async def test_markdown_without_answers_says_so(self) -> None:
        records = [build_answer_record(_q(), "paper_chat", error="boom")]
        samples = await score_records(records, [], NullScorer())
        markdown = render_generation_markdown(aggregate_scores(samples, []), samples, [], title="t", meta={})
        assert "답변이 있는 레코드가 없습니다." in markdown


def test_answers_jsonl_round_trips_replay_fields(tmp_path) -> None:
    path = tmp_path / "a.jsonl"
    record = _source_record(_q(), prompt_variant=PROMPT_BEFORE, sources_only=False, sources_origin=SOURCES_FROM_REPLAY)
    append_answer_record(path, record)
    stored = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert stored["sources"] == [{"chunk_id": 1, "section_title": "A", "text": "c"}]
