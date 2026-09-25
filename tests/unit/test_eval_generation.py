from __future__ import annotations

import csv
import json
import math
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import FakeListChatModel

from eval.answers import (
    AnswersError,
    GenerationResult,
    append_answer_record,
    build_answer_record,
    capture_agent_contexts,
    collect_answers,
    completed_keys,
    generate_agent_answer,
    load_answer_records,
    load_reference_answers,
    make_paper_chat_generator,
    resolve_modes,
)
from eval.dataset import EvalQuery, dump_queries
from eval.generation import (
    REFERENCE_ANSWER,
    REFERENCE_GOLD_CHUNKS,
    SKIP_EMPTY_ANSWER,
    SKIP_GENERATION_ERROR,
    SKIP_NO_CONTEXTS,
    SKIP_NO_REFERENCE,
    SKIP_NO_REFERENCE_ANSWER,
    SKIP_UNDEFINED,
    MetricRequest,
    aggregate_scores,
    is_reasoning_model,
    judge_model_kwargs,
    plan_record,
    render_markdown,
    render_readme_table,
    resolve_metrics,
    score_records,
    write_sample_csv,
    write_summary_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ALL_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

QUERIES = [
    EvalQuery(
        id="ki-2401.00001-ko",
        query="선호 최적화를 다시 본 논문",
        lang="ko",
        relevant_arxiv_ids=("2401.00001",),
        source="known_item",
    ),
    EvalQuery(
        id="cs-11",
        query="What learning rate was used?",
        lang="en",
        relevant_arxiv_ids=("2401.00002",),
        source="llm_synth",
        relevant_chunk_ids=(11,),
    ),
]


class FakeClock:
    def __init__(self, step: float = 0.25):
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def _result(answer: str = "answer", contexts: list[str] | None = None) -> GenerationResult:
    return GenerationResult(
        answer=answer,
        contexts=contexts if contexts is not None else ["ctx a", "ctx b"],
        citations=[{"arxiv_id": "2401.00001", "in_answer": True}],
        retrieval_mode="hybrid",
    )


def _record(**overrides) -> dict:
    record = build_answer_record(QUERIES[1], "agent", result=_result(), latency_ms=10.0)
    record.update(overrides)
    return record


class FakeScorer:
    """지표별 고정 점수를 돌려주고 받은 요청을 기록한다. 값이 예외면 그 예외 객체를 결과로 넘긴다."""

    def __init__(self, values: dict[str, object] | None = None):
        self.values = values or {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6,
        }
        self.requests: list[MetricRequest] = []

    def score(self, requests):
        self.requests.extend(requests)
        return [self.values[request.metric] for request in requests]


class TestModesAndMetrics:
    def test_both_expands_and_dedupes(self):
        assert resolve_modes(["both", "agent"]) == ["agent", "paper_chat"]
        assert resolve_modes([" paper_chat "]) == ["paper_chat"]

    @pytest.mark.parametrize("values", [["chat"], [""]])
    def test_invalid_modes(self, values):
        with pytest.raises(ValueError):
            resolve_modes(values)

    def test_metrics(self):
        assert resolve_metrics(["all"]) == ALL_METRICS
        assert resolve_metrics(["faithfulness", "faithfulness"]) == ["faithfulness"]
        with pytest.raises(ValueError):
            resolve_metrics(["bleu"])


class TestCollectAnswers:
    def test_builds_records_from_fake_generators(self):
        sunk: list[dict] = []
        generators = {
            "agent": lambda query: _result(answer=f"agent:{query.id}"),
            "paper_chat": lambda query: GenerationResult(
                answer="chat",
                contexts=["abstract"],
                retrieval_mode="first_chunks",
                arxiv_id=query.relevant_arxiv_ids[0],
            ),
        }

        records = collect_answers(
            QUERIES,
            ["agent", "paper_chat"],
            generators,
            reference_answers={"cs-11": "3e-4"},
            model="gpt-4o",
            sink=sunk.append,
            clock=FakeClock(0.5),
        )

        assert records == sunk
        assert [(record["id"], record["mode"]) for record in records] == [
            ("ki-2401.00001-ko", "agent"),
            ("ki-2401.00001-ko", "paper_chat"),
            ("cs-11", "agent"),
            ("cs-11", "paper_chat"),
        ]
        first = records[0]
        assert first["answer"] == "agent:ki-2401.00001-ko"
        assert first["contexts"] == ["ctx a", "ctx b"]
        assert first["citations"] == [{"arxiv_id": "2401.00001", "in_answer": True}]
        assert first["latency_ms"] == 500.0
        assert first["model"] == "gpt-4o"
        assert first["reference_answer"] is None
        assert first["error"] is None
        assert records[3]["reference_answer"] == "3e-4"
        assert records[3]["relevant_chunk_ids"] == [11]
        assert records[3]["arxiv_id"] == "2401.00002"
        assert records[3]["retrieval_mode"] == "first_chunks"
        assert {"id", "query", "answer", "contexts", "citations", "mode"} <= set(first)

    def test_done_keys_are_skipped(self):
        calls: list[str] = []

        def generate(query):
            calls.append(query.id)
            return _result()

        records = collect_answers(QUERIES, ["agent"], {"agent": generate}, done={("ki-2401.00001-ko", "agent")})
        assert calls == ["cs-11"]
        assert len(records) == 1

    def test_keep_going_records_error(self):
        def generate(query):
            if query.id == "cs-11":
                raise RuntimeError("boom")
            return _result()

        records = collect_answers(QUERIES, ["agent"], {"agent": generate}, keep_going=True)
        assert records[0]["error"] is None
        assert records[1]["error"] == "RuntimeError: boom"
        assert records[1]["answer"] == ""
        assert records[1]["contexts"] == []
        assert records[1]["latency_ms"] is None

    def test_without_keep_going_raises_after_sinking_earlier_records(self):
        sunk: list[dict] = []

        def generate(query):
            if query.id == "cs-11":
                raise RuntimeError("boom")
            return _result()

        with pytest.raises(RuntimeError):
            collect_answers(QUERIES, ["agent"], {"agent": generate}, sink=sunk.append)
        assert [record["id"] for record in sunk] == ["ki-2401.00001-ko"]


class TestAnswerFiles:
    def test_append_load_last_wins_and_completed_keys(self, tmp_path):
        path = tmp_path / "nested" / "answers.jsonl"
        failed = build_answer_record(QUERIES[0], "agent", error="RuntimeError: x")
        ok = build_answer_record(QUERIES[0], "agent", result=_result())
        other = build_answer_record(QUERIES[1], "paper_chat", error="LookupError: y")
        for record in (failed, other, ok):
            append_answer_record(path, record)

        records = load_answer_records(path)

        assert [(record["id"], record["mode"], record["error"]) for record in records] == [
            ("cs-11", "paper_chat", "LookupError: y"),
            ("ki-2401.00001-ko", "agent", None),
        ]
        assert completed_keys(records) == {("ki-2401.00001-ko", "agent")}

    @pytest.mark.parametrize(
        "line",
        [
            "not json",
            json.dumps(["list"]),
            json.dumps({"id": "a", "query": "q", "mode": "agent", "answer": ""}),
            json.dumps({"id": "a", "query": "q", "mode": "web", "answer": "", "contexts": []}),
            json.dumps({"id": "a", "query": "q", "mode": "agent", "answer": "", "contexts": [1]}),
        ],
    )
    def test_malformed_lines(self, tmp_path, line):
        path = tmp_path / "answers.jsonl"
        path.write_text(line + "\n", encoding="utf-8")
        with pytest.raises(AnswersError):
            load_answer_records(path)

    def test_reference_answers_are_optional(self, tmp_path):
        path = tmp_path / "queries.jsonl"
        lines = [
            {**QUERIES[0].to_dict(), "reference_answer": "  DPO를 다시 검토한다. "},
            {**QUERIES[1].to_dict(), "reference_answer": ""},
        ]
        path.write_text("\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n", encoding="utf-8")
        assert load_reference_answers(path) == {"ki-2401.00001-ko": "DPO를 다시 검토한다."}


def _chunk_context(chunk_id: int, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "arxiv_id": "2401.00001",
        "paper_title": "Paper",
        "chunk_text": f"raw {chunk_id}",
        "context_text": text,
        "section_title": "3 Method",
    }


class TestAgentContexts:
    def test_capture_records_texts_the_tool_passes_to_the_llm_and_restores(self, monkeypatch):
        from src.core.agent import tools

        contexts = [_chunk_context(1, "window one"), _chunk_context(2, "window two")]
        fake_retrieve = MagicMock(return_value=(contexts, "hybrid"))
        monkeypatch.setattr(tools, "retrieve_contexts", fake_retrieve)
        monkeypatch.setattr(tools, "PaperRetriever", lambda: object())

        with capture_agent_contexts() as captured:
            output = tools.search_paper_chunks_tool.invoke({"query": "loss"})

        assert captured.texts == ["window one", "window two"]
        assert captured.retrieval_modes == ["hybrid"]
        assert all(text in output for text in captured.texts)
        assert tools.retrieve_contexts is fake_retrieve

    def test_generate_agent_answer_dedupes_contexts_across_tool_calls(self, monkeypatch):
        from src.core.agent import chatbot, tools

        monkeypatch.setattr(
            tools,
            "retrieve_contexts",
            lambda query, **kwargs: ([_chunk_context(1, "same"), _chunk_context(2, query)], "lexical"),
        )

        def fake_agent_search(question, **kwargs):
            tools.retrieve_contexts("first", retriever=None, limit=5)
            tools.retrieve_contexts("second", retriever=None, limit=5)
            return {"answer": "final", "citations": [{"arxiv_id": "2401.00001"}]}

        monkeypatch.setattr(chatbot, "agent_search", fake_agent_search)

        result = generate_agent_answer(QUERIES[0])

        assert result.answer == "final"
        assert result.contexts == ["same", "first", "second"]
        assert result.retrieval_mode == "lexical"
        assert result.citations == [{"arxiv_id": "2401.00001"}]


class TestPaperChatGenerator:
    PAPER = {"arxiv_id": "2401.00002", "title": "Paper", "abstract": "We study X.", "pdf_url": None}

    def _retriever(self):
        retriever = MagicMock()
        retriever.embedding_client.is_available.return_value = True
        retriever.search_paper_contexts_by_hybrid.return_value = [_chunk_context(11, "lr is 3e-4")]
        return retriever

    def test_contexts_are_the_numbered_sources(self):
        from src.core.agent import retrieval

        repository = MagicMock()
        repository.get_paper.return_value = dict(self.PAPER)
        retriever = self._retriever()
        generate = make_paper_chat_generator(
            repository=repository,
            retriever_factory=lambda repo: retriever,
            llm=FakeListChatModel(responses=["The rate is 3e-4 [2]."]),
        )

        with patch.object(retrieval, "get_settings", return_value=types.SimpleNamespace(retrieval_mode="hybrid")):
            result = generate(QUERIES[1])

        repository.get_paper.assert_called_once_with("2401.00002")
        assert result.answer == "The rate is 3e-4 [2]."
        assert result.contexts == ["We study X.", "lr is 3e-4"]
        assert result.retrieval_mode == "hybrid"
        assert result.arxiv_id == "2401.00002"
        assert [citation["chunk_id"] for citation in result.citations] == [11]

    def test_missing_paper_raises(self):
        repository = MagicMock()
        repository.get_paper.return_value = None
        generate = make_paper_chat_generator(repository=repository, retriever_factory=lambda repo: MagicMock())
        with pytest.raises(LookupError):
            generate(QUERIES[1])


class TestPlanRecord:
    def test_all_metrics_with_reference_answer(self):
        record = _record(reference_answer="3e-4")
        requests, skipped, kind = plan_record(record, ALL_METRICS, chunk_texts={11: "gold"})
        assert [request.metric for request in requests] == ALL_METRICS
        assert skipped == {}
        assert kind == REFERENCE_ANSWER
        by_metric = {request.metric: request for request in requests}
        assert by_metric["context_precision"].reference == "3e-4"
        assert by_metric["context_recall"].reference == "3e-4"
        assert by_metric["faithfulness"].retrieved_contexts == ("ctx a", "ctx b")

    def test_gold_chunks_feed_context_precision_but_not_recall(self):
        requests, skipped, kind = plan_record(_record(), ALL_METRICS, chunk_texts={11: "gold passage"})
        by_metric = {request.metric: request for request in requests}
        assert by_metric["context_precision"].reference == "gold passage"
        assert kind == REFERENCE_GOLD_CHUNKS
        assert skipped == {"context_recall": SKIP_NO_REFERENCE_ANSWER}

    def test_no_reference_skips_precision_and_recall(self):
        record = _record(relevant_chunk_ids=[])
        requests, skipped, kind = plan_record(record, ALL_METRICS)
        assert [request.metric for request in requests] == ["faithfulness", "answer_relevancy"]
        assert skipped == {"context_precision": SKIP_NO_REFERENCE, "context_recall": SKIP_NO_REFERENCE_ANSWER}
        assert kind == ""

    def test_chunk_text_missing_from_db_counts_as_no_reference(self):
        _requests, skipped, _kind = plan_record(_record(), ["context_precision"], chunk_texts={})
        assert skipped == {"context_precision": SKIP_NO_REFERENCE}

    def test_generation_error_skips_everything(self):
        record = build_answer_record(QUERIES[1], "agent", error="RuntimeError: x", reference_answer="r")
        requests, skipped, _kind = plan_record(record, ALL_METRICS, chunk_texts={11: "gold"})
        assert requests == []
        assert skipped == dict.fromkeys(ALL_METRICS, SKIP_GENERATION_ERROR)

    def test_empty_answer_and_no_contexts(self):
        record = _record(answer="  ", contexts=[" "], reference_answer="r")
        requests, skipped, _kind = plan_record(record, ALL_METRICS)
        assert requests == []
        assert skipped == {
            "faithfulness": SKIP_EMPTY_ANSWER,
            "answer_relevancy": SKIP_EMPTY_ANSWER,
            "context_precision": SKIP_NO_CONTEXTS,
            "context_recall": SKIP_NO_CONTEXTS,
        }

    def test_answer_relevancy_does_not_need_contexts(self):
        requests, skipped, _kind = plan_record(_record(contexts=[]), ["faithfulness", "answer_relevancy"])
        assert [request.metric for request in requests] == ["answer_relevancy"]
        assert skipped == {"faithfulness": SKIP_NO_CONTEXTS}


def _records() -> list[dict]:
    return [
        build_answer_record(QUERIES[0], "agent", result=_result(), reference_answer="ref"),
        build_answer_record(QUERIES[1], "agent", result=_result()),
        build_answer_record(QUERIES[0], "paper_chat", result=_result(), reference_answer="ref"),
        build_answer_record(QUERIES[1], "paper_chat", error="LookupError: missing"),
    ]


class TestScoring:
    def test_fake_scorer_scores_map_back_to_records(self):
        scorer = FakeScorer()
        samples = score_records(_records(), ALL_METRICS, scorer, chunk_texts={11: "gold"})

        assert len(scorer.requests) == 4 + 3 + 4
        assert samples[0].scores == {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6,
        }
        assert samples[1].scores.keys() == {"faithfulness", "answer_relevancy", "context_precision"}
        assert samples[1].reference_kind == REFERENCE_GOLD_CHUNKS
        assert samples[3].scores == {}
        assert samples[3].generation_error == "LookupError: missing"
        assert {request.lang for request in scorer.requests} == {"ko", "en"}

    def test_exceptions_and_nan(self):
        scorer = FakeScorer(
            {
                "faithfulness": float("nan"),
                "answer_relevancy": ValueError("bad json"),
                "context_precision": None,
                "context_recall": 1.0,
            }
        )
        samples = score_records(_records()[:1], ALL_METRICS, scorer)
        sample = samples[0]
        assert sample.scores == {"context_recall": 1.0}
        assert sample.errors == {"answer_relevancy": "ValueError: bad json"}
        assert sample.skipped == {"faithfulness": SKIP_UNDEFINED, "context_precision": SKIP_UNDEFINED}

    def test_result_count_mismatch_raises(self):
        class Short:
            def score(self, requests):
                return []

        with pytest.raises(RuntimeError):
            score_records(_records()[:1], ["faithfulness"], Short())

    def test_no_requests_skips_scorer(self):
        class Exploding:
            def score(self, requests):
                raise AssertionError("must not be called")

        samples = score_records([_records()[3]], ALL_METRICS, Exploding())
        assert samples[0].skipped["faithfulness"] == SKIP_GENERATION_ERROR


class TestAggregationAndReport:
    def _samples(self):
        scorer = FakeScorer()
        samples = score_records(_records(), ALL_METRICS, scorer, chunk_texts={11: "gold"})
        samples[1].scores["faithfulness"] = 0.5
        return samples

    def test_aggregate_means_counts_and_subsets(self):
        aggregates = aggregate_scores(self._samples(), ALL_METRICS)
        index = {(row.mode, row.subset): row for row in aggregates}

        agent_all = index[("agent", "all")]
        assert agent_all.n_records == 2
        assert agent_all.means["faithfulness"] == pytest.approx(0.7)
        assert agent_all.counts == {
            "faithfulness": 2,
            "answer_relevancy": 2,
            "context_precision": 2,
            "context_recall": 1,
        }
        chat_all = index[("paper_chat", "all")]
        assert chat_all.n_generation_errors == 1
        assert chat_all.counts["faithfulness"] == 1
        assert index[("paper_chat", "en")].means["faithfulness"] is None
        assert ("agent", "manual") not in index
        assert [row.mode for row in aggregates if row.subset == "all"] == ["agent", "paper_chat"]

    def test_markdown_and_readme_table(self):
        samples = self._samples()
        aggregates = aggregate_scores(samples, ALL_METRICS)
        table = render_readme_table(aggregates)
        assert table.splitlines()[2] == "| agent | 0.700 | 0.800 | 0.700 | 0.600 |"
        assert table.splitlines()[3] == "| paper_chat | 0.900 | 0.800 | 0.700 | 0.600 |"

        markdown = render_markdown(aggregates, samples, ALL_METRICS, title="Generation", meta={"판정": "fake"})
        assert markdown.startswith("# Generation\n\n- 판정: fake\n")
        assert table in markdown
        assert "| agent | all | 2 | 0.700 (2) | 0.800 (2) | 0.700 (2) | 0.600 (1) | 0 | 0 |" in markdown
        assert "| paper_chat | en | 1 | n/a | n/a | n/a | n/a | 1 | 0 |" in markdown
        assert "| context_recall | no_reference_answer | 1 |" in markdown
        assert "| faithfulness | generation_error | 1 |" in markdown

    def test_readme_table_marks_uncomputed_metrics(self):
        samples = score_records(_records()[:1], ["faithfulness"], FakeScorer())
        table = render_readme_table(aggregate_scores(samples, ["faithfulness"]))
        assert table.splitlines()[2] == "| agent | 0.900 | - | - | - |"

    def test_empty_markdown(self):
        assert "집계할 결과가 없습니다." in render_markdown([], [], ALL_METRICS, title="t", meta={})

    def test_csv_outputs(self, tmp_path):
        samples = self._samples()
        aggregates = aggregate_scores(samples, ALL_METRICS)
        write_sample_csv(samples, ALL_METRICS, tmp_path / "samples.csv")
        write_summary_csv(aggregates, ALL_METRICS, tmp_path / "summary.csv")

        with (tmp_path / "samples.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows[1]["faithfulness"] == "0.5000"
        assert rows[1]["context_recall_skipped"] == SKIP_NO_REFERENCE_ANSWER
        assert rows[3]["generation_error"] == "LookupError: missing"
        with (tmp_path / "summary.csv").open(encoding="utf-8") as handle:
            summary = list(csv.DictReader(handle))
        assert summary[0]["mode"] == "agent" and summary[0]["subset"] == "all"
        assert summary[0]["faithfulness"] == "0.7000"
        assert summary[0]["context_recall_n"] == "1"


class TestJudgeModel:
    @pytest.mark.parametrize(
        ("model", "expected"), [("gpt-5-mini", True), ("o3-mini", True), ("gpt-4o", False), ("omni", False)]
    )
    def test_reasoning_detection(self, model, expected):
        assert is_reasoning_model(model) is expected

    def test_kwargs(self):
        assert judge_model_kwargs("gpt-5-mini") == {"max_tokens": 8192}
        assert judge_model_kwargs("gpt-4o-mini") == {}
        assert judge_model_kwargs("gpt-4o-mini", max_tokens=2048) == {"max_tokens": 2048}
        assert "temperature" not in judge_model_kwargs("gpt-5")


def _load_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_eval_script_eval_generation", REPO_ROOT / "scripts" / "eval_generation.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestEvalGenerationScript:
    @pytest.fixture(autouse=True)
    def _settings(self, monkeypatch):
        import src.shared

        self.settings = types.SimpleNamespace(openai_api_key="sk-test", openai_model="gpt-4o")
        monkeypatch.setattr(src.shared, "get_settings", lambda: self.settings)

    @pytest.fixture
    def script(self):
        return _load_script()

    def _answers_file(self, tmp_path: Path) -> Path:
        path = tmp_path / "answers.jsonl"
        for record in _records():
            append_answer_record(path, record)
        return path

    def _queries_file(self, tmp_path: Path) -> Path:
        path = tmp_path / "queries.jsonl"
        path.write_text(dump_queries(QUERIES), encoding="utf-8")
        return path

    def test_parse_args(self, script):
        args = script.parse_args(["--answers", "a.jsonl", "--metrics", "faithfulness", "--judge-model", "gpt-4o-mini"])
        assert args.answers == "a.jsonl" and not args.collect
        assert args.judge_model == "gpt-4o-mini"
        assert script.parse_args(["--collect"]).judge_model == "gpt-5-mini"
        assert script.parse_args(["--collect"]).mode == "both"

    @pytest.mark.parametrize(
        "argv",
        [
            [],
            ["--collect", "--answers", "a.jsonl"],
            ["--answers", "a.jsonl", "--no-score"],
            ["--collect", "--limit", "0"],
            ["--collect", "--concurrency", "0"],
        ],
    )
    def test_parse_args_rejects(self, script, argv):
        with pytest.raises(SystemExit):
            script.parse_args(argv)

    @pytest.mark.parametrize("key", [None, "", "change-me-openai-api-key"])
    def test_missing_key_exits_without_writing(self, script, tmp_path, capsys, key):
        self.settings.openai_api_key = key
        out_dir = tmp_path / "results"
        code = script.main(["--answers", str(self._answers_file(tmp_path)), "--out-dir", str(out_dir)])
        assert code == script.EXIT_PRECONDITION
        assert "OPENAI_API_KEY" in capsys.readouterr().err
        assert not out_dir.exists()

    def test_unknown_metric_exits(self, script, tmp_path):
        code = script.main(["--answers", str(self._answers_file(tmp_path)), "--metrics", "bleu"])
        assert code == script.EXIT_PRECONDITION

    def test_collect_with_unreachable_db_exits_without_writing(self, script, tmp_path, monkeypatch, capsys):
        def refuse(settings):
            raise OSError("connection refused")

        monkeypatch.setattr(script, "_connect", refuse)
        out_dir = tmp_path / "results"
        code = script.main(["--collect", "--queries", str(self._queries_file(tmp_path)), "--out-dir", str(out_dir)])
        assert code == script.EXIT_PRECONDITION
        assert "PostgreSQL에 연결할 수 없습니다" in capsys.readouterr().err
        assert not out_dir.exists()

    def test_scoring_needs_db_for_gold_chunks(self, script, tmp_path, monkeypatch, capsys):
        def refuse(settings):
            raise OSError("connection refused")

        monkeypatch.setattr(script, "_connect", refuse)
        out_dir = tmp_path / "results"
        code = script.main(["--answers", str(self._answers_file(tmp_path)), "--out-dir", str(out_dir)])
        assert code == script.EXIT_PRECONDITION
        assert "PostgreSQL에 연결할 수 없습니다" in capsys.readouterr().err
        assert not out_dir.exists()

    def test_score_existing_answers_with_fake_scorer(self, script, tmp_path, monkeypatch, capsys):
        scorer = FakeScorer()
        monkeypatch.setattr(script, "build_scorer", lambda args, api_key, metrics: scorer)
        fetched: list[list[int]] = []

        def fake_fetch(settings, chunk_ids):
            fetched.append(chunk_ids)
            return {11: "gold"}

        monkeypatch.setattr(script, "fetch_chunk_texts", fake_fetch)
        out_dir = tmp_path / "results"

        code = script.main(["--answers", str(self._answers_file(tmp_path)), "--out-dir", str(out_dir)])

        assert code == 1
        assert fetched == [[11]]
        names = sorted(path.name for path in out_dir.iterdir())
        assert len(names) == 3
        assert any(name.endswith("_summary.csv") for name in names)
        markdown = next(out_dir.glob("generation_*.md")).read_text(encoding="utf-8")
        assert "| agent | 0.900 | 0.800 | 0.700 | 0.600 |" in markdown
        assert "judge=gpt-5-mini" in markdown
        assert "agent 2, paper_chat 2" in markdown
        captured = capsys.readouterr()
        assert markdown in captured.out
        assert "답변 생성 실패 1건" in captured.err

    def test_metrics_without_references_do_not_touch_db(self, script, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "build_scorer", lambda args, api_key, metrics: FakeScorer())
        monkeypatch.setattr(script, "_connect", MagicMock(side_effect=AssertionError("no db")))
        out_dir = tmp_path / "results"
        code = script.main(
            [
                "--answers",
                str(self._answers_file(tmp_path)),
                "--metrics",
                "faithfulness,answer_relevancy",
                "--mode",
                "agent",
                "--out-dir",
                str(out_dir),
            ]
        )
        assert code == 0
        markdown = next(out_dir.glob("generation_*.md")).read_text(encoding="utf-8")
        assert "| agent | 0.900 | 0.800 | - | - |" in markdown
        assert "paper_chat" not in markdown.split("## 지표")[0]

    def test_collect_then_score_and_resume(self, script, tmp_path, monkeypatch, capsys):
        corpus = {"papers": 2, "chunks": 4, "embeddings": 4}
        monkeypatch.setattr(script, "preflight", lambda settings, queries, modes: corpus)
        monkeypatch.setattr(script, "fetch_chunk_texts", lambda settings, chunk_ids: {11: "gold"})
        monkeypatch.setattr(script, "build_scorer", lambda args, api_key, metrics: FakeScorer())
        calls: list[tuple[str, str | None]] = []

        def fake_generators(modes):
            from src.shared import get_runtime_openai_api_key, get_runtime_openai_model

            def generate(query):
                calls.append((query.id, get_runtime_openai_model()))
                assert get_runtime_openai_api_key() == "sk-test"
                return _result()

            return dict.fromkeys(modes, generate)

        monkeypatch.setattr(script, "build_product_generators", fake_generators)
        answers = tmp_path / "answers.jsonl"
        queries = self._queries_file(tmp_path)
        out_dir = tmp_path / "results"
        base = ["--collect", "--queries", str(queries), "--mode", "agent", "--out", str(answers)]

        code = script.main([*base, "--limit", "1", "--no-score", "--answer-model", "gpt-4o-mini"])
        assert code == 0
        assert calls == [("ki-2401.00001-ko", "gpt-4o-mini")]
        assert not out_dir.exists()

        code = script.main([*base, "--out-dir", str(out_dir)])
        assert code == 0
        assert calls[1:] == [("cs-11", "gpt-4o")]
        records = load_answer_records(answers)
        assert [(record["id"], record["model"]) for record in records] == [
            ("ki-2401.00001-ko", "gpt-4o-mini"),
            ("cs-11", "gpt-4o"),
        ]
        markdown = next(out_dir.glob("generation_*.md")).read_text(encoding="utf-8")
        assert "코퍼스: papers 2, chunks 4, embeddings 4" in markdown
        assert "gpt-4o, gpt-4o-mini" in markdown
        assert "이어서 수집합니다(완료 1건 건너뜀)" in capsys.readouterr().err

    def test_collect_keep_going_returns_one_on_failures(self, script, tmp_path, monkeypatch):
        monkeypatch.setattr(
            script, "preflight", lambda settings, queries, modes: {"papers": 1, "chunks": 1, "embeddings": 0}
        )

        def failing(modes):
            def generate(query):
                raise RuntimeError("llm down")

            return dict.fromkeys(modes, generate)

        monkeypatch.setattr(script, "build_product_generators", failing)
        answers = tmp_path / "answers.jsonl"
        code = script.main(
            [
                "--collect",
                "--queries",
                str(self._queries_file(tmp_path)),
                "--mode",
                "paper_chat",
                "--out",
                str(answers),
                "--keep-going",
                "--no-score",
            ]
        )
        assert code == 1
        assert all(record["error"] == "RuntimeError: llm down" for record in load_answer_records(answers))


def test_nan_is_not_averaged():
    samples = score_records(_records()[:2], ["faithfulness"], FakeScorer({"faithfulness": math.nan}))
    rows = aggregate_scores(samples, ["faithfulness"])
    assert rows[0].means["faithfulness"] is None
    assert rows[0].counts["faithfulness"] == 0
