import csv
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from arxplore.api.services.chat import build_chat_service
from arxplore.config import AppSettings
from arxplore.domain.chunks import StoredChunk
from arxplore.domain.papers import Paper
from arxplore.domain.retrieval import RetrievalContext, RetrievalResult
from arxplore.llm.models import ChatModelFactory
from arxplore.llm.paper_chat import PaperChatSource, format_paper_chat_context
from arxplore.llm.prompts.paper_chat import PAPER_CHAT_PROMPT
from arxplore.llm.tracing import Tracer
from arxplore_eval.answers import (
    SOURCES_FROM_REPLAY,
    SOURCES_FROM_RETRIEVAL,
    AnswersError,
    GenerationResult,
    PaperChatOptions,
    append_answer_record,
    build_answer_record,
    collect_answers,
    context_digest,
    generate_paper_chat_answer,
    load_answer_records,
    load_recorded_sources,
    recorded_sources,
    replayed_retrieval,
)
from arxplore_eval.behavior import answer_checks, answer_language, citation_numbers_valid
from arxplore_eval.cli import app
from arxplore_eval.commands import generation as generation_commands
from arxplore_eval.compare_generation import (
    CompareGenerationError,
    compare_generation,
    pair_generation,
    read_run,
    render_markdown,
)
from arxplore_eval.corpus import PreconditionError
from arxplore_eval.dataset import EvalQuery, dump_queries
from arxplore_eval.generation import NullScorer, aggregate_scores, score_records, write_sample_csv
from arxplore_eval.prompt_variants import (
    PAPER_CHAT_PROMPT_BEFORE_LANGUAGE_RULE,
    PAPER_CHAT_PROMPT_VARIANTS,
    PROMPT_BEFORE,
    PROMPT_CURRENT,
    paper_chat_prompt,
)

runner = CliRunner()
ARXIV_ID = "2401.00002"
BEFORE_SYSTEM_SHA256 = "6fdef3d49780d525410677aa88dceb7337665e03d150055c4c4efd57dfc6e4ce"


def _paper(abstract: str = "We study X.\n\nTwo lines.") -> Paper:
    return Paper(
        arxiv_id=ARXIV_ID,
        title="Paper Title: 한글 subtitle",
        authors=[],
        abstract=abstract,
        primary_category=None,
        categories=[],
        pdf_url=None,
        published_at=None,
        updated_at=None,
        upvotes=3,
        github_url=None,
        github_stars=None,
        citation_count=None,
    )


@dataclass
class FakeStore:
    papers: dict[str, Paper] = field(default_factory=dict)

    async def get_paper(self, arxiv_id: str) -> Paper | None:
        return self.papers.get(arxiv_id)

    async def list_recent_papers(self, *, limit: int) -> Sequence[Paper]:
        return list(self.papers.values())[:limit]

    async def list_chunks(self, arxiv_id: str, *, limit: int) -> Sequence[StoredChunk]:
        return []


@dataclass
class CountingRetriever:
    contexts: Sequence[RetrievalContext]
    mode: str = "hybrid"
    calls: list[str] = field(default_factory=list)

    async def retrieve_contexts(self, query: str, *, limit: int = 5, arxiv_id: str | None = None) -> RetrievalResult:
        self.calls.append(query)
        return RetrievalResult(contexts=tuple(self.contexts), mode=self.mode)


def _service(fake_openai: Any, retriever: CountingRetriever) -> Any:
    models = ChatModelFactory(
        server_api_key=fake_openai.expected_api_key, default_model="gpt-4o", base_url=fake_openai.base_url
    )
    return build_chat_service(
        AppSettings.model_validate({}),
        store=FakeStore({ARXIV_ID: _paper()}),
        retriever=retriever,
        models=models,
        tracer=Tracer.disabled(),
    )


def _q(query_id: str = "pc-1-en", **kwargs: Any) -> EvalQuery:
    base: dict[str, Any] = {
        "query": "What learning rate was used?",
        "lang": "en",
        "relevant_arxiv_ids": (ARXIV_ID,),
        "source": "manual",
    }
    base.update(kwargs)
    return EvalQuery(id=query_id, **base)


def _contexts(make_hit: Any, make_context: Any) -> list[RetrievalContext]:
    return [
        make_context(make_hit(11, arxiv_id=ARXIV_ID, section_title="4.2 Setup"), text="lr is 3e-4.\n| a | b |"),
        make_context(make_hit(12, arxiv_id=ARXIV_ID, section_title=""), text="  다국어 발췌문 [1] 그대로  "),
        make_context(make_hit(11, arxiv_id=ARXIV_ID), text="duplicate chunk is dropped"),
    ]


def _system_and_user(request: Any) -> tuple[str, str]:
    messages = request.body["messages"]
    return messages[0]["content"], messages[-1]["content"]


class TestPromptVariants:
    def test_before_prompt_is_the_head_text_and_current_is_the_product_prompt(self) -> None:
        system = PAPER_CHAT_PROMPT_BEFORE_LANGUAGE_RULE.messages[0].prompt.template
        assert hashlib.sha256(system.encode("utf-8")).hexdigest() == BEFORE_SYSTEM_SHA256
        assert paper_chat_prompt(PROMPT_CURRENT) is PAPER_CHAT_PROMPT
        assert paper_chat_prompt(PROMPT_BEFORE) is PAPER_CHAT_PROMPT_BEFORE_LANGUAGE_RULE
        assert PAPER_CHAT_PROMPT_BEFORE_LANGUAGE_RULE.messages[2] == PAPER_CHAT_PROMPT.messages[2]
        assert PAPER_CHAT_PROMPT_BEFORE_LANGUAGE_RULE.input_variables == PAPER_CHAT_PROMPT.input_variables
        with pytest.raises(ValueError, match="상세 챗 프롬프트"):
            paper_chat_prompt("after")

    def test_cli_choices_follow_the_variants(self) -> None:
        assert tuple(choice.value for choice in generation_commands.PaperChatPrompt) == PAPER_CHAT_PROMPT_VARIANTS


class TestCollection:
    async def test_prompt_variant_reaches_the_model_and_the_record(self, fake_openai, make_hit, make_context) -> None:
        fake_openai.chat_handler = lambda request: "The rate is 3e-4 [1]."
        service = _service(fake_openai, CountingRetriever(_contexts(make_hit, make_context)))
        generators = {
            "paper_chat": lambda query: generate_paper_chat_answer(
                service, query, PaperChatOptions(prompt_variant=PROMPT_BEFORE)
            )
        }

        records = await collect_answers([_q()], ["paper_chat"], generators, model="gpt-4o")

        system, _ = _system_and_user(fake_openai.requests_for("/v1/chat/completions")[0])
        assert system == PAPER_CHAT_PROMPT_BEFORE_LANGUAGE_RULE.messages[0].prompt.template
        record = records[0]
        assert record["prompt_variant"] == PROMPT_BEFORE
        assert record["sources_origin"] == SOURCES_FROM_RETRIEVAL
        assert record["sources_only"] is False
        assert [source["chunk_id"] for source in record["sources"]] == [None, 11, 12]
        assert [source["section_title"] for source in record["sources"]] == ["Abstract", "4.2 Setup", None]
        assert record["contexts"] == [source["text"] for source in record["sources"]]

    async def test_default_prompt_is_the_product_prompt(self, fake_openai, make_hit, make_context) -> None:
        service = _service(fake_openai, CountingRetriever(_contexts(make_hit, make_context)))

        result = await generate_paper_chat_answer(service, _q())

        system, _ = _system_and_user(fake_openai.requests_for("/v1/chat/completions")[0])
        assert system == PAPER_CHAT_PROMPT.messages[0].prompt.template
        assert result.paper_chat is not None
        assert result.paper_chat.prompt_variant == PROMPT_CURRENT

    async def test_sources_only_never_calls_the_model(self, fake_openai, make_hit, make_context) -> None:
        retriever = CountingRetriever(_contexts(make_hit, make_context))
        service = _service(fake_openai, retriever)

        result = await generate_paper_chat_answer(service, _q(), PaperChatOptions(sources_only=True))
        record = build_answer_record(_q(), "paper_chat", result=result)

        assert fake_openai.requests_for("/v1/chat/completions") == []
        assert retriever.calls == ["What learning rate was used?"]
        assert (record["answer"], record["prompt_variant"], record["sources_only"]) == ("", None, True)
        assert record["retrieval_mode"] == "hybrid"
        assert len(record["sources"]) == 3

    async def test_replay_sends_a_byte_identical_context_without_searching(
        self, tmp_path, fake_openai, make_hit, make_context
    ) -> None:
        retriever = CountingRetriever(_contexts(make_hit, make_context), mode="lexical")
        service = _service(fake_openai, retriever)
        query = _q(history=(("user", "earlier"), ("assistant", "reply")))
        direct = await generate_paper_chat_answer(service, query)
        _, direct_user = _system_and_user(fake_openai.requests_for("/v1/chat/completions")[0])
        sources_file = tmp_path / "sources.jsonl"
        sources_only = await generate_paper_chat_answer(service, query, PaperChatOptions(sources_only=True))
        append_answer_record(sources_file, build_answer_record(query, "paper_chat", result=sources_only))
        fake_openai.reset()

        replay = load_recorded_sources(sources_file)
        replayed = await generate_paper_chat_answer(
            service, query, PaperChatOptions(prompt_variant=PROMPT_BEFORE, replay=replay)
        )

        _, replay_user = _system_and_user(fake_openai.requests_for("/v1/chat/completions")[0])
        assert len(retriever.calls) == 2
        assert replay_user == direct_user
        assert replayed.paper_chat is not None and direct.paper_chat is not None
        assert replayed.paper_chat.context_sha256 == direct.paper_chat.context_sha256
        assert replayed.paper_chat.sources_origin == SOURCES_FROM_REPLAY
        assert replayed.retrieval_mode == "lexical"
        assert replayed.contexts == direct.contexts
        rebuilt = replayed_retrieval(_paper(), replay[query.id])
        original = [
            PaperChatSource(
                arxiv_id=ARXIV_ID,
                title="t",
                url="u",
                section_title=item["section_title"],
                chunk_id=item["chunk_id"],
                text=item["text"],
            )
            for item in direct.paper_chat.sources
        ]
        assert format_paper_chat_context(rebuilt.sources).encode() == format_paper_chat_context(original).encode()

    async def test_replay_refuses_missing_or_altered_sources(
        self, tmp_path, fake_openai, make_hit, make_context
    ) -> None:
        service = _service(fake_openai, CountingRetriever(_contexts(make_hit, make_context)))
        result = await generate_paper_chat_answer(service, _q(), PaperChatOptions(sources_only=True))
        record = build_answer_record(_q(), "paper_chat", result=result)
        altered = {**record, "sources": [{**record["sources"][0], "text": "changed"}, *record["sources"][1:]]}

        with pytest.raises(ValueError, match="context_sha256"):
            replayed_retrieval(_paper(), recorded_sources(altered))
        with pytest.raises(LookupError, match="소스 기록이 없습니다"):
            await generate_paper_chat_answer(service, _q("other"), PaperChatOptions(replay={}))
        with pytest.raises(AnswersError, match="sources"):
            recorded_sources({**record, "sources": None})
        with pytest.raises(AnswersError, match="chunk_id"):
            recorded_sources({**record, "sources": [{"chunk_id": True, "section_title": None, "text": "x"}]})
        with pytest.raises(ValueError):
            PaperChatOptions(sources_only=True, replay={})
        assert context_digest(()) == hashlib.sha256("제공된 발췌문이 없습니다.".encode()).hexdigest()

    def test_agent_records_keep_their_keys(self) -> None:
        record = build_answer_record(_q(), "agent", result=GenerationResult(answer="a", contexts=[]))
        assert "prompt_variant" not in record and "sources" not in record


class TestAnswerChecks:
    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("", ""),
            ("   ", ""),
            ("12345 [1].", "en"),
            ("The rate is 3e-4.", "en"),
            ("학습률은 3e-4입니다.", "ko"),
            ("가나다abcdef", "ko"),
            ("가나다 abc 123 !!! def", "ko"),
            ("가나다abcdefgh", "en"),
        ],
    )
    def test_answer_language_uses_the_hangul_letter_ratio(self, answer: str, expected: str) -> None:
        assert answer_language(answer) == expected

    def test_exactly_thirty_percent_is_english(self) -> None:
        assert answer_language("가나다" + "a" * 7) == "en"

    @pytest.mark.parametrize(
        ("answer", "sources", "expected"),
        [
            ("no numbers", 2, True),
            ("ok [1], [2] and [1, 2]", 2, True),
            ("zero [0]", 2, False),
            ("too far [3]", 2, False),
            ("[link](https://arxiv.org/abs/2401.00001) is not a number", 0, True),
        ],
    )
    def test_citation_numbers(self, answer: str, sources: int, expected: bool) -> None:
        assert citation_numbers_valid(answer, sources) is expected

    def test_checks_follow_mode_and_outcome(self) -> None:
        chat = {"mode": "paper_chat", "lang": "en", "answer": "답변입니다 [2].", "contexts": ["a"]}
        assert answer_checks(chat, "answered") == ("ko", {"lang_match": 0.0, "citation_numbers_valid": 0.0})
        assert answer_checks({**chat, "sources": [{}, {}]}, "refused")[1]["citation_numbers_valid"] == 1.0
        assert answer_checks({**chat, "mode": "agent"}, "answered") == ("ko", {"lang_match": 0.0})
        assert answer_checks({**chat, "answer": ""}, "answered") == ("", {})
        assert answer_checks(chat, "error") == ("", {})

    async def test_sample_csv_and_markdown_have_the_new_columns(self, tmp_path) -> None:
        records = [
            build_answer_record(
                _q("a-en"), "paper_chat", result=GenerationResult(answer="한국어 답변 [1]", contexts=["c"])
            ),
            build_answer_record(
                _q("b-ko", lang="ko"), "paper_chat", result=GenerationResult(answer="한국어", contexts=[])
            ),
        ]
        samples = await score_records(records, [], NullScorer())
        write_sample_csv(samples, [], tmp_path / "s.csv")
        with (tmp_path / "s.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert [(row["answer_lang"], row["lang_match"], row["citation_numbers_valid"]) for row in rows] == [
            ("ko", "0.0000", "1.0000"),
            ("ko", "1.0000", "1.0000"),
        ]
        rows_by_subset = {row.subset: row for row in aggregate_scores(samples, [])}
        assert rows_by_subset["all"].behavior_means["lang_match"] == 0.5
        assert rows_by_subset["all"].behavior_counts["citation_numbers_valid"] == 2


QUERIES = [
    EvalQuery(id="p1-ko", query="q", lang="ko", relevant_arxiv_ids=("A",), source="known_item"),
    EvalQuery(id="p1-en", query="q", lang="en", relevant_arxiv_ids=("A",), source="known_item"),
    EvalQuery(id="p2-en", query="q", lang="en", relevant_arxiv_ids=("B",), source="manual", category="paper_chat"),
    EvalQuery(
        id="oc-1-en",
        query="q",
        lang="en",
        relevant_arxiv_ids=(),
        source="manual",
        expected_behavior="refuse",
        open_arxiv_id="A",
    ),
]


def _sample_csv(path: Path, rows: dict[str, dict[str, str]]) -> Path:
    fields = ["id", "mode", "lang", "outcome", "faithfulness", "lang_match", "citation_numbers_valid"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for query_id, values in rows.items():
            writer.writerow({"id": query_id, "mode": "paper_chat", **values})
    return path


def _row(outcome: str = "answered", faithfulness: str = "", lang_match: str = "1.0000") -> dict[str, str]:
    return {"outcome": outcome, "faithfulness": faithfulness, "lang_match": lang_match, "citation_numbers_valid": "1"}


class TestCompareGeneration:
    def _runs(self, tmp_path: Path) -> tuple[list[Path], list[Path]]:
        a1 = _sample_csv(
            tmp_path / "a1.csv",
            {
                "p1-ko": _row(faithfulness="0.5"),
                "p1-en": _row(faithfulness="0.4", lang_match="0.0000"),
                "p2-en": _row(outcome="refused", lang_match="0.0000"),
                "oc-1-en": _row(outcome="refused"),
            },
        )
        a2 = _sample_csv(
            tmp_path / "a2.csv",
            {
                "p1-ko": _row(faithfulness="0.7"),
                "p1-en": _row(faithfulness="0.6", lang_match="0.0000"),
                "p2-en": _row(outcome="error", lang_match=""),
            },
        )
        b1 = _sample_csv(
            tmp_path / "b1.csv",
            {
                "p1-ko": _row(faithfulness="0.6"),
                "p1-en": _row(faithfulness="0.9"),
                "p2-en": _row(),
                "oc-1-en": _row(),
            },
        )
        return [a1, a2], [b1]

    def test_repetitions_are_averaged_before_pairing(self, tmp_path) -> None:
        runs_a, runs_b = self._runs(tmp_path)
        paired = pair_generation(
            [read_run(path) for path in runs_a], [read_run(path) for path in runs_b], QUERIES, label_a="A", label_b="B"
        )
        comparison = compare_generation(paired, resamples=200, seed=1)

        assert [pair.query.id for pair in paired.pairs] == ["p1-ko", "p1-en", "p2-en", "oc-1-en"]
        assert paired.metrics == ("faithfulness", "lang_match", "refused", "citation_numbers_valid")
        overall = {row.metric: row.result for row in comparison.deltas if row.subset == "all"}
        assert overall["faithfulness"].mean_a == pytest.approx(0.55)
        assert overall["faithfulness"].mean_b == pytest.approx(0.75)
        assert overall["faithfulness"].n_clusters == 1
        assert overall["lang_match"].delta == pytest.approx(0.5)
        assert overall["refused"].mean_a == pytest.approx(0.5)
        refusal = {row.behavior: row for row in comparison.refusal}
        assert (refusal["refuse"].rate_a, refusal["refuse"].rate_b) == (1.0, 0.0)
        assert refusal["refuse"].transitions["a_only"] == 1
        assert refusal["answer"].transitions == {"both_refused": 0, "a_only": 1, "b_only": 0, "neither": 2}
        variance = {(row.label, row.metric): row for row in comparison.variance}
        assert variance[("A", "faithfulness")].mean_stdev == pytest.approx(0.1414, abs=1e-4)
        assert variance[("A", "faithfulness")].rep_means == (pytest.approx(0.45), pytest.approx(0.65))
        assert variance[("B", "faithfulness")].n_queries == 0
        assert any("일부 반복에만" in line for line in paired.warnings)
        assert any("rejected·error" in line for line in paired.warnings)
        markdown = render_markdown(comparison, title="t", meta={"k": "v"})
        assert "| 거절 기대(거절 유지율) | 1 | 1.000 | 0.000 | 0 | 1 | 0 | 0 |" in markdown
        assert "category:paper_chat" in markdown and "behavior:refuse" in markdown

    def test_unknown_queries_and_wrong_files_are_refused(self, tmp_path) -> None:
        stray = _sample_csv(tmp_path / "s.csv", {"zz-1": _row()})
        with pytest.raises(CompareGenerationError, match="질의셋에 없는 id"):
            pair_generation([read_run(stray)], [read_run(stray)], QUERIES, label_a="a", label_b="b")
        with pytest.raises(CompareGenerationError, match="샘플 CSV"):
            read_run(tmp_path / "x.txt")
        empty = _sample_csv(tmp_path / "e.csv", {})
        with pytest.raises(CompareGenerationError, match="레코드가 없습니다"):
            read_run(empty)

    def test_answer_files_are_scored_on_the_fly(self, tmp_path) -> None:
        path = tmp_path / "answers.jsonl"
        result = GenerationResult(answer="I cannot answer this from the provided excerpts.", contexts=["c"])
        record = build_answer_record(QUERIES[1], "paper_chat", result=result)
        append_answer_record(path, {**record, "prompt_variant": PROMPT_BEFORE})
        run = read_run(path)
        assert run.values["p1-en"]["refused"] == 1.0
        assert run.values["p1-en"]["lang_match"] == 1.0
        assert run.values["p1-en"]["faithfulness"] is None
        assert run.prompt_variants == (PROMPT_BEFORE,)
        append_answer_record(path, {**record, "id": "p1-ko", "sources_only": True})
        with pytest.raises(CompareGenerationError, match="검색 전용"):
            read_run(path)

    def test_command_writes_markdown_and_csv(self, tmp_path) -> None:
        runs_a, runs_b = self._runs(tmp_path)
        queries = tmp_path / "queries.jsonl"
        queries.write_text(dump_queries(QUERIES), encoding="utf-8")
        out_dir = tmp_path / "out"
        argv = ["compare-generation", "--queries", str(queries), "--label-a", "before", "--label-b", "after"]
        argv += [item for path in runs_a for item in ("--a", str(path))]
        argv += [item for path in runs_b for item in ("--b", str(path))]
        argv += ["--resamples", "50", "--out-dir", str(out_dir)]

        result = runner.invoke(app, argv)

        assert result.exit_code == 0, result.output
        assert sorted(path.name for path in out_dir.iterdir()) == [
            "compare_generation.md",
            "compare_generation_metrics.csv",
            "compare_generation_queries.csv",
            "compare_generation_refusal.csv",
            "compare_generation_variance.csv",
        ]
        markdown = (out_dir / "compare_generation.md").read_text(encoding="utf-8")
        assert "a = before 반복 2" in markdown
        assert "# 생성 비교: after vs before" in markdown


class TestGenerationOptions:
    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            (["generation", "--answers", "a.jsonl", "--sources-only"], "--collect와 함께만"),
            (["generation", "--answers", "a.jsonl", "--paper-chat-prompt", "before"], "--collect와 함께만"),
            (["generation", "--collect", "--sources-only", "--replay-sources", "s.jsonl"], "함께 쓸 수 없습니다"),
            (["generation", "--collect", "--sources-only", "--paper-chat-prompt", "before"], "--paper-chat-prompt"),
            (["generation", "--collect", "--sources-only"], "--mode paper_chat에서만"),
            (["generation", "--collect", "--mode", "agent", "--paper-chat-prompt", "before"], "paper_chat 모드"),
        ],
    )
    def test_preconditions(self, clean_env, argv: list[str], message: str) -> None:
        result = runner.invoke(app, argv)
        assert result.exit_code == 2
        assert message in result.stderr

    def test_scoring_refuses_sources_only_files(self, clean_env) -> None:
        path = clean_env / "sources.jsonl"
        record = build_answer_record(_q(), "paper_chat", result=GenerationResult(answer="", contexts=[]))
        append_answer_record(path, {**record, "sources_only": True})
        result = runner.invoke(app, ["generation", "--answers", str(path), "--behavior-only"])
        assert result.exit_code == 2
        assert "검색 전용" in result.stderr

    def test_resuming_with_another_prompt_or_source_is_refused(self, tmp_path) -> None:
        path = tmp_path / "answers.jsonl"
        record = build_answer_record(_q(), "paper_chat", result=GenerationResult(answer="a", contexts=[]))
        append_answer_record(path, {**record, "prompt_variant": PROMPT_BEFORE, "sources_origin": SOURCES_FROM_REPLAY})
        assert generation_commands._done_keys(path, None, (PROMPT_BEFORE, SOURCES_FROM_REPLAY)) == {
            ("pc-1-en", "paper_chat")
        }
        with pytest.raises(PreconditionError, match="상세 챗 레코드 설정"):
            generation_commands._done_keys(path, None, (PROMPT_CURRENT, SOURCES_FROM_REPLAY))
        with pytest.raises(PreconditionError, match="상세 챗 레코드 설정"):
            generation_commands._done_keys(path, None, (PROMPT_BEFORE, SOURCES_FROM_RETRIEVAL))
        legacy = tmp_path / "legacy.jsonl"
        append_answer_record(legacy, record)
        assert generation_commands._done_keys(legacy, None, (PROMPT_BEFORE, SOURCES_FROM_REPLAY))

    def test_meta_lists_the_prompt_setup(self, tmp_path) -> None:
        path = tmp_path / "answers.jsonl"
        record = build_answer_record(_q(), "paper_chat", result=GenerationResult(answer="a", contexts=[]))
        append_answer_record(path, {**record, "prompt_variant": PROMPT_BEFORE, "sources_origin": SOURCES_FROM_REPLAY})
        records = load_answer_records(path)
        meta = generation_commands.build_meta(
            started_at="t",
            answers_path=path,
            records=records,
            metrics=[],
            behavior_only=True,
            judge=generation_commands.JudgeOptions("j", "e", None, 1, True, None),
            refusal_phrases_path=None,
            corpus=None,
        )
        assert meta["상세 챗 프롬프트·소스"] == "프롬프트 before / 소스 replay"


def test_replay_file_keeps_only_successful_paper_chat_records(tmp_path) -> None:
    path = tmp_path / "sources.jsonl"
    sources = [{"chunk_id": None, "section_title": "Abstract", "text": "abs"}]
    result = GenerationResult(answer="", contexts=["abs"], retrieval_mode="hybrid")
    ok = build_answer_record(_q("ok"), "paper_chat", result=result)
    ok.update({"sources": sources, "context_sha256": None})
    failed = build_answer_record(_q("bad"), "paper_chat", error="LookupError: x")
    agent = build_answer_record(_q("ag"), "agent", result=GenerationResult(answer="", contexts=[]))
    for record in (ok, failed, agent):
        append_answer_record(path, record)
    loaded = load_recorded_sources(path)
    assert list(loaded) == ["ok"]
    assert loaded["ok"].arxiv_id is None
    assert replace(loaded["ok"], context_sha256=None).sources[0].text == "abs"
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["id"] == "ok"
