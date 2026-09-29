import csv
import math
from collections.abc import Sequence
from typing import Any

import pytest

from arxplore_eval.answers import GenerationResult, build_answer_record
from arxplore_eval.dataset import EvalQuery
from arxplore_eval.generation import (
    REFERENCE_ANSWER,
    REFERENCE_GOLD_CHUNKS,
    SKIP_EMPTY_ANSWER,
    SKIP_GENERATION_ERROR,
    SKIP_NO_CONTEXTS,
    SKIP_NO_REFERENCE,
    SKIP_NO_REFERENCE_ANSWER,
    SKIP_NON_ANSWER_CASE,
    SKIP_REJECTED,
    SKIP_UNDEFINED,
    MetricRequest,
    NullScorer,
    RagasScorer,
    RagasUnavailableError,
    aggregate_scores,
    is_reasoning_model,
    judge_model_kwargs,
    plan_record,
    ragas_module,
    render_markdown,
    render_readme_table,
    request_kwargs,
    resolve_metrics,
    score_records,
    write_sample_csv,
    write_summary_csv,
)

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


def _result(answer: str = "answer", contexts: list[str] | None = None) -> GenerationResult:
    return GenerationResult(
        answer=answer,
        contexts=contexts if contexts is not None else ["ctx a", "ctx b"],
        citations=[{"arxiv_id": "2401.00001", "in_answer": True}],
        retrieval_mode="hybrid",
    )


def _record(**overrides: Any) -> dict[str, Any]:
    record = build_answer_record(QUERIES[1], "agent", result=_result(), latency_ms=10.0)
    record.update(overrides)
    return record


def _records() -> list[dict[str, Any]]:
    return [
        build_answer_record(QUERIES[0], "agent", result=_result(), reference_answer="ref"),
        build_answer_record(QUERIES[1], "agent", result=_result()),
        build_answer_record(QUERIES[0], "paper_chat", result=_result(), reference_answer="ref"),
        build_answer_record(QUERIES[1], "paper_chat", error="LookupError: missing"),
    ]


def _q(query_id: str, **kwargs: Any) -> EvalQuery:
    base: dict[str, Any] = {
        "query": "question",
        "lang": "en",
        "relevant_arxiv_ids": ("2401.00001",),
        "source": "manual",
    }
    base.update(kwargs)
    return EvalQuery(id=query_id, **base)


def _behavior_records() -> list[dict[str, Any]]:
    return [
        build_answer_record(
            _q("lang-1", category="language"),
            "agent",
            result=GenerationResult(
                answer="[P](https://arxiv.org/abs/2401.00001)", contexts=["ctx"], hit_arxiv_ids=["2401.00001"]
            ),
        ),
        build_answer_record(
            _q("lang-2", category="language"),
            "agent",
            result=GenerationResult(answer="[Fake](https://arxiv.org/abs/2099.00001)", contexts=["ctx"]),
        ),
        build_answer_record(
            _q("oc-1", relevant_arxiv_ids=(), category="out_of_corpus", expected_behavior="refuse", mode="agent"),
            "agent",
            result=GenerationResult(answer="관련 논문을 찾지 못했습니다.", contexts=[]),
        ),
        build_answer_record(
            _q("sf-1", query="", relevant_arxiv_ids=(), category="safety", expect_rejected=True),
            "agent",
            rejection="메시지를 입력하세요.",
        ),
    ]


class FakeScorer:
    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.values = values or {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6,
        }
        self.requests: list[MetricRequest] = []

    async def score(self, requests: Sequence[MetricRequest]) -> list[Any]:
        self.requests.extend(requests)
        return [self.values[request.metric] for request in requests]


class TestMetrics:
    def test_resolve(self) -> None:
        assert resolve_metrics(["all"]) == ALL_METRICS
        assert resolve_metrics(["faithfulness", "faithfulness"]) == ["faithfulness"]
        with pytest.raises(ValueError):
            resolve_metrics(["bleu"])
        with pytest.raises(ValueError):
            resolve_metrics([" "])


class TestPlanRecord:
    def test_all_metrics_with_reference_answer(self) -> None:
        requests, skipped, kind = plan_record(_record(reference_answer="3e-4"), ALL_METRICS, chunk_texts={11: "gold"})
        assert [request.metric for request in requests] == ALL_METRICS
        assert skipped == {}
        assert kind == REFERENCE_ANSWER
        by_metric = {request.metric: request for request in requests}
        assert by_metric["context_precision"].reference == "3e-4"
        assert by_metric["context_recall"].reference == "3e-4"
        assert by_metric["faithfulness"].reference is None
        assert by_metric["faithfulness"].retrieved_contexts == ("ctx a", "ctx b")

    def test_gold_chunks_feed_context_precision_but_not_recall(self) -> None:
        requests, skipped, kind = plan_record(_record(), ALL_METRICS, chunk_texts={11: "gold passage"})
        assert {request.metric: request.reference for request in requests}["context_precision"] == "gold passage"
        assert kind == REFERENCE_GOLD_CHUNKS
        assert skipped == {"context_recall": SKIP_NO_REFERENCE_ANSWER}

    def test_no_reference_skips_precision_and_recall(self) -> None:
        requests, skipped, kind = plan_record(_record(relevant_chunk_ids=[]), ALL_METRICS)
        assert [request.metric for request in requests] == ["faithfulness", "answer_relevancy"]
        assert skipped == {"context_precision": SKIP_NO_REFERENCE, "context_recall": SKIP_NO_REFERENCE_ANSWER}
        assert kind == ""
        assert plan_record(_record(), ["context_precision"], chunk_texts={})[1] == {
            "context_precision": SKIP_NO_REFERENCE
        }

    def test_generation_error_rejection_and_non_answer_cases_skip_everything(self) -> None:
        error = build_answer_record(QUERIES[1], "agent", error="RuntimeError: x", reference_answer="r")
        assert plan_record(error, ALL_METRICS, chunk_texts={11: "gold"})[1] == dict.fromkeys(
            ALL_METRICS, SKIP_GENERATION_ERROR
        )
        records = _behavior_records()
        requests, skipped, _ = plan_record(records[2], ["faithfulness", "answer_relevancy"])
        assert requests == [] and skipped == dict.fromkeys(["faithfulness", "answer_relevancy"], SKIP_NON_ANSWER_CASE)
        assert plan_record(records[3], ["answer_relevancy"])[1] == {"answer_relevancy": SKIP_REJECTED}

    def test_empty_answer_and_no_contexts(self) -> None:
        requests, skipped, _ = plan_record(_record(answer="  ", contexts=[" "], reference_answer="r"), ALL_METRICS)
        assert requests == []
        assert skipped == {
            "faithfulness": SKIP_EMPTY_ANSWER,
            "answer_relevancy": SKIP_EMPTY_ANSWER,
            "context_precision": SKIP_NO_CONTEXTS,
            "context_recall": SKIP_NO_CONTEXTS,
        }
        requests, skipped, _ = plan_record(_record(contexts=[]), ["faithfulness", "answer_relevancy"])
        assert [request.metric for request in requests] == ["answer_relevancy"]
        assert skipped == {"faithfulness": SKIP_NO_CONTEXTS}


class TestScoring:
    async def test_fake_scorer_scores_map_back_to_records(self) -> None:
        scorer = FakeScorer()
        samples = await score_records(_records(), ALL_METRICS, scorer, chunk_texts={11: "gold"})

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

    async def test_exceptions_and_nan(self) -> None:
        scorer = FakeScorer(
            {
                "faithfulness": float("nan"),
                "answer_relevancy": ValueError("bad json"),
                "context_precision": None,
                "context_recall": 1.0,
            }
        )
        (sample,) = await score_records(_records()[:1], ALL_METRICS, scorer)
        assert sample.scores == {"context_recall": 1.0}
        assert sample.errors == {"answer_relevancy": "ValueError: bad json"}
        assert sample.skipped == {"faithfulness": SKIP_UNDEFINED, "context_precision": SKIP_UNDEFINED}

    async def test_result_count_mismatch_raises(self) -> None:
        class Short:
            async def score(self, requests: Sequence[MetricRequest]) -> list[Any]:
                return []

        with pytest.raises(RuntimeError):
            await score_records(_records()[:1], ["faithfulness"], Short())

    async def test_no_requests_skip_the_scorer(self) -> None:
        samples = await score_records([_records()[3]], ALL_METRICS, NullScorer())
        assert samples[0].skipped["faithfulness"] == SKIP_GENERATION_ERROR
        with pytest.raises(RuntimeError):
            await NullScorer().score([MetricRequest("faithfulness", "q", "a", ("c",), None, "en")])

    async def test_behavior_is_scored_per_record_and_aggregated_by_category(self) -> None:
        scorer = FakeScorer()
        samples = await score_records(_behavior_records(), ["answer_relevancy"], scorer)
        assert [request.user_input for request in scorer.requests] == ["question", "question"]
        assert [sample.outcome for sample in samples] == ["answered", "answered", "refused", "rejected"]
        assert samples[1].behavior == {"no_fabricated_links": 0.0}
        assert samples[2].behavior == {"no_fabricated_links": 1.0, "refusal_correct": 1.0}
        assert samples[3].behavior == {"rejected_as_expected": 1.0}

        rows = {(row.mode, row.subset): row for row in aggregate_scores(samples, ["answer_relevancy"])}
        language = rows[("agent", "category:language")]
        assert language.n_records == 2
        assert language.means["answer_relevancy"] == 0.8
        assert language.behavior_means["no_fabricated_links"] == 0.5
        assert language.behavior_counts["no_fabricated_links"] == 2
        assert rows[("agent", "category:out_of_corpus")].behavior_means["refusal_correct"] == 1.0
        assert rows[("agent", "behavior:refuse")].n_records == 1
        assert rows[("agent", "behavior:answer")].n_records == 3
        assert rows[("agent", "all")].outcomes == {"answered": 2, "refused": 1, "rejected": 1, "error": 0}


class TestAggregationAndReport:
    async def _samples(self) -> list[Any]:
        samples = await score_records(_records(), ALL_METRICS, FakeScorer(), chunk_texts={11: "gold"})
        samples[1].scores["faithfulness"] = 0.5
        return samples

    async def test_aggregate_means_counts_and_subsets(self) -> None:
        aggregates = aggregate_scores(await self._samples(), ALL_METRICS)
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

    async def test_markdown_and_readme_table(self) -> None:
        samples = await self._samples()
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
        assert "집계할 결과가 없습니다." in render_markdown([], [], ALL_METRICS, title="t", meta={})

    async def test_readme_table_marks_uncomputed_metrics(self) -> None:
        samples = await score_records(_records()[:1], ["faithfulness"], FakeScorer())
        table = render_readme_table(aggregate_scores(samples, ["faithfulness"]))
        assert table.splitlines()[2] == "| agent | 0.900 | - | - | - |"

    async def test_behavior_table_and_csv(self, tmp_path) -> None:
        samples = await score_records(_behavior_records(), [], FakeScorer())
        aggregates = aggregate_scores(samples, [])
        markdown = render_markdown(aggregates, samples, [], title="t", meta={})
        section = markdown.split("## 행동 지표 (LLM 판정 없음)")[1].split("## 건너뛴 지표")[0]
        assert "| agent | all | 4 | 2 | 1 | 1 | 0 | 1.000 (1) | 0.667 (3) | - | - | 1.000 (1) |" in section
        assert "| agent | category:safety | 1 | 0 | 0 | 1 | 0 | - | - | - | - | 1.000 (1) |" in section

        write_sample_csv(samples, [], tmp_path / "s.csv")
        write_summary_csv(aggregates, [], tmp_path / "sum.csv")
        with (tmp_path / "s.csv").open(encoding="utf-8") as handle:
            sample_rows = list(csv.DictReader(handle))
        assert [row["outcome"] for row in sample_rows] == ["answered", "answered", "refused", "rejected"]
        assert sample_rows[1]["no_fabricated_links"] == "0.0000"
        with (tmp_path / "sum.csv").open(encoding="utf-8") as handle:
            summary = {row["subset"]: row for row in csv.DictReader(handle)}
        assert summary["all"]["n_rejected"] == "1"
        assert summary["category:language"]["no_fabricated_links"] == "0.5000"
        assert summary["category:language"]["no_fabricated_links_n"] == "2"

    async def test_metric_csv_outputs(self, tmp_path) -> None:
        samples = await self._samples()
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
        assert (summary[0]["mode"], summary[0]["subset"], summary[0]["faithfulness"]) == ("agent", "all", "0.7000")
        assert summary[0]["context_recall_n"] == "1"

    async def test_nan_is_not_averaged(self) -> None:
        samples = await score_records(_records()[:2], ["faithfulness"], FakeScorer({"faithfulness": math.nan}))
        rows = aggregate_scores(samples, ["faithfulness"])
        assert rows[0].means["faithfulness"] is None
        assert rows[0].counts["faithfulness"] == 0


class TestJudgeModel:
    @pytest.mark.parametrize(
        ("model", "expected"), [("gpt-5-mini", True), ("o3-mini", True), ("gpt-4o", False), ("omni", False)]
    )
    def test_reasoning_detection(self, model: str, expected: bool) -> None:
        assert is_reasoning_model(model) is expected

    def test_kwargs(self) -> None:
        assert judge_model_kwargs("gpt-5-mini") == {"max_tokens": 8192}
        assert judge_model_kwargs("gpt-4o-mini") == {}
        assert judge_model_kwargs("gpt-4o-mini", max_tokens=2048) == {"max_tokens": 2048}

    def test_request_kwargs_follow_each_metric(self) -> None:
        request = MetricRequest("faithfulness", "q", "a", ("c1", "c2"), "ref", "en")
        assert request_kwargs(request) == {"user_input": "q", "response": "a", "retrieved_contexts": ["c1", "c2"]}
        relevancy = MetricRequest("answer_relevancy", "q", "a", ("c1",), None, "en")
        assert request_kwargs(relevancy) == {"user_input": "q", "response": "a"}
        precision = MetricRequest("context_precision", "q", "a", ("c1",), "ref", "en")
        assert request_kwargs(precision) == {"user_input": "q", "reference": "ref", "retrieved_contexts": ["c1"]}


class FakeMetric:
    def __init__(self, name: str, language: str = "english") -> None:
        self.name = name
        self.language = language
        self.calls: list[dict[str, Any]] = []

    async def ascore(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("response") == "boom":
            raise ValueError("judge failed")
        if kwargs.get("response") == "undefined":
            return type("Result", (), {"value": None})()
        return type("Result", (), {"value": 0.25 if self.language == "korean" else 0.75})()


class OfflineRagasScorer(RagasScorer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="sk-fake-arxplore", base_url="http://127.0.0.1:9/v1", **kwargs)
        self.built: list[dict[str, FakeMetric]] = []
        self.adapted: list[str] = []

    def judge(self, client: Any) -> tuple[Any, Any]:
        return "llm", "embeddings"

    def build_metrics(self, llm: Any, embeddings: Any) -> dict[str, Any]:
        metrics = {metric: FakeMetric(metric) for metric in self.metrics}
        self.built.append(metrics)
        return metrics

    async def adapt(self, metrics: dict[str, Any], llm: Any, language: str) -> dict[str, Any]:
        self.adapted.append(language)
        for metric in metrics.values():
            metric.language = language
        return metrics


class TestRagasScorer:
    async def test_korean_requests_use_the_adapted_metric_set(self) -> None:
        progress: list[tuple[int, int, str]] = []
        scorer = OfflineRagasScorer(
            metrics=["faithfulness"], progress=lambda done, total, request: progress.append((done, total, request.lang))
        )
        requests = [
            MetricRequest("faithfulness", "q", "a", ("c",), None, "ko"),
            MetricRequest("faithfulness", "q", "a", ("c",), None, "en"),
            MetricRequest("faithfulness", "q", "boom", ("c",), None, "en"),
            MetricRequest("faithfulness", "q", "undefined", ("c",), None, "en"),
        ]

        results = await scorer.score(requests)

        assert results[:2] == [0.25, 0.75]
        assert isinstance(results[2], ValueError)
        assert results[3] is None
        assert scorer.adapted == ["korean"]
        assert len(scorer.built) == 2
        assert sorted(item[:2] for item in progress) == [(1, 4), (2, 4), (3, 4), (4, 4)]

    async def test_no_adaptation_without_korean_or_when_disabled(self) -> None:
        english = [MetricRequest("faithfulness", "q", "a", ("c",), None, "en")]
        scorer = OfflineRagasScorer(metrics=["faithfulness"])
        assert await scorer.score(english) == [0.75]
        assert scorer.adapted == []

        korean = [MetricRequest("faithfulness", "q", "a", ("c",), None, "ko")]
        disabled = OfflineRagasScorer(metrics=["faithfulness"], adapt_korean=False)
        assert await disabled.score(korean) == [0.75]
        assert disabled.adapted == []
        assert await disabled.score([]) == []

    def test_missing_ragas_is_reported(self) -> None:
        with pytest.raises(RagasUnavailableError, match="uv sync --group eval"):
            ragas_module("arxplore_eval_missing_ragas_module")
