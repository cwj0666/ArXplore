import itertools
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest

from arxplore_eval.dataset import EvalQuery
from arxplore_eval.embedders import CountingEmbedder, MemoEmbedder
from arxplore_eval.latency import (
    CONTROL,
    HYBRID_EVAL,
    PARALLEL,
    SEQUENTIAL,
    VECTOR_ONLY,
    BenchmarkPlan,
    Sample,
    StageRecorder,
    TimedEmbedder,
    embedding_cost,
    equality_rows,
    memo_equality,
    pair_order,
    paired_delta,
    render_markdown,
    run_benchmark,
    signature,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _query(query_id: str, lang: str) -> EvalQuery:
    return EvalQuery(
        id=query_id, query=f"policy loss {query_id}", lang=lang, relevant_arxiv_ids=("A",), source="manual"
    )


QUERIES = [_query("a-ko", "ko"), _query("a-en", "en"), _query("b-ko", "ko"), _query("b-en", "en")]


class FakeEmbedder:
    def __init__(self, clock: FakeClock | None = None) -> None:
        self.clock = clock
        self.calls: list[list[str]] = []

    def is_available(self) -> bool:
        return True

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.clock is not None:
            self.clock.advance(0.1)
        return [[float(len(text))] for text in texts]


def test_pair_order_alternates() -> None:
    assert pair_order(0) == (SEQUENTIAL, PARALLEL)
    assert pair_order(1) == (PARALLEL, SEQUENTIAL)
    assert pair_order(-1) == (PARALLEL, SEQUENTIAL)


async def test_run_benchmark_interleaves_and_skips_warmup(make_hit, make_context) -> None:
    calls: list[tuple[Any, str]] = []

    async def path(retriever: Any, query: str) -> tuple[list[Any], str]:
        calls.append((retriever, query))
        return [make_context(make_hit(1, arxiv_id="x"))], "hybrid"

    plan = BenchmarkPlan(
        queries=QUERIES[:2],
        retrievers={SEQUENTIAL: "A", PARALLEL: "B"},
        recorder=StageRecorder(),
        paired_paths={HYBRID_EVAL: path},
        control_paths={VECTOR_ONLY: path},
        repeats=2,
        warmup=1,
    )
    samples = await run_benchmark(plan)

    assert [retriever for retriever, _ in calls] == ["B", "A", "B"] * 2 + ["A", "B", "B"] * 2 + ["B", "A", "B"] * 2
    assert len(samples) == 2 * 2 * 3
    assert {sample.repeat for sample in samples} == {0, 1}
    assert [s.variant for s in samples if s.repeat == 1 and s.query_id == "a-ko"] == [PARALLEL, SEQUENTIAL, CONTROL]
    assert samples[0].signature == ((1, "x"),)


async def test_run_benchmark_records_errors_and_keeps_going() -> None:
    async def failing(retriever: Any, query: str) -> tuple[list[Any], str]:
        raise RuntimeError("boom")

    plan = BenchmarkPlan(
        queries=QUERIES[:1],
        retrievers={SEQUENTIAL: "A", PARALLEL: "B"},
        recorder=StageRecorder(),
        paired_paths={HYBRID_EVAL: failing},
        repeats=1,
        warmup=0,
    )
    samples = await run_benchmark(plan)
    assert [sample.error for sample in samples] == ["RuntimeError: boom"] * 2
    assert all(sample.signature == () for sample in samples)


async def test_stage_recorder_and_timed_embedder_measure_awaited_work() -> None:
    clock = FakeClock()
    recorder = StageRecorder(clock=clock)
    embedder = TimedEmbedder(FakeEmbedder(clock), recorder)

    assert embedder.is_available()
    assert await embedder.embed(["abc"]) == [[3.0]]
    assert recorder.snapshot_ms() == {"embedding": pytest.approx(100.0)}
    recorder.reset()
    assert recorder.snapshot_ms() == {}


async def test_memo_and_counting_embedders() -> None:
    inner = FakeEmbedder()
    counting = CountingEmbedder(inner, counter=len)
    memo = MemoEmbedder(counting)

    assert await memo.embed(["abcd"]) == await memo.embed(["abcd"]) == [[4.0]]
    await memo.embed(["xy"])

    assert inner.calls == [["abcd"], ["xy"]]
    assert (counting.calls, counting.texts, counting.tokens) == (2, 2, 6)
    assert memo.is_available() and counting.is_available()


async def test_memo_equality_compares_whole_results(make_hit, make_context) -> None:
    async def path(retriever: str, query: str) -> tuple[list[Any], str]:
        score = 0.5 if retriever == SEQUENTIAL or query.endswith("a-ko") else 0.4
        return [make_context(make_hit(1, score=score))], "hybrid"

    equality = await memo_equality({HYBRID_EVAL: path}, {SEQUENTIAL: SEQUENTIAL, PARALLEL: PARALLEL}, QUERIES)
    assert equality == {HYBRID_EVAL: (1, 4)}


def _samples(a_ms: dict[str, float], b_ms: dict[str, float], repeats: int = 3) -> list[Sample]:
    samples = []
    for repeat, query in itertools.product(range(repeats), QUERIES):
        for variant, table in ((SEQUENTIAL, a_ms), (PARALLEL, b_ms)):
            samples.append(
                Sample(
                    query.id, query.lang, repeat, HYBRID_EVAL, variant, 0, table[query.id], {}, ((1, "x"),), "hybrid"
                )
            )
    return samples


def test_paired_delta_uses_per_query_medians_and_clusters() -> None:
    a = {"a-ko": 300.0, "a-en": 500.0, "b-ko": 300.0, "b-en": 500.0}
    b = {"a-ko": 200.0, "a-en": 300.0, "b-ko": 200.0, "b-en": 300.0}
    row = paired_delta(_samples(a, b), QUERIES, HYBRID_EVAL, resamples=200, seed=1)

    assert (row.n, row.n_clusters) == (4, 2)
    assert (row.delta_ms, row.low_ms, row.high_ms) == pytest.approx((-150.0, -150.0, -150.0))
    assert row.relative == pytest.approx(((200 / 300 - 1) + (300 / 500 - 1)) / 2)

    en = paired_delta(_samples(a, b), QUERIES, HYBRID_EVAL, subset="en", resamples=200, seed=1)
    assert (en.n, en.n_clusters, en.delta_ms) == (2, 2, pytest.approx(-200.0))


def test_equality_rows_report_mismatches_and_baselines() -> None:
    ids = [query.id for query in QUERIES]
    samples = _samples(dict.fromkeys(ids, 1.0), dict.fromkeys(ids, 1.0), 2)
    samples = [
        replace(sample, signature=((2, "y"), (1, "x")))
        if (sample.variant == PARALLEL and sample.repeat == 1 and sample.query_id == "a-en")
        else sample
        for sample in samples
    ]
    rows = {row.comparison: row for row in equality_rows(samples, HYBRID_EVAL, 2)}

    assert rows["A vs B (반복 0)"].mismatches == 0
    assert rows["A vs B (반복 1)"].mismatch_ids == ("a-en",)
    assert rows["A vs B (반복 1)"].same_set_reordered == 0
    assert rows["A vs A (반복 0→1)"].mismatches == 0
    assert rows["B vs B (반복 0→1)"].mismatches == 1


def test_signature_and_cost(make_hit, make_context) -> None:
    assert signature([make_context(make_hit(3, arxiv_id="a"))]) == ((3, "a"),)
    assert embedding_cost(1_000_000) == pytest.approx(0.13)


def test_render_markdown_smoke() -> None:
    ids = [query.id for query in QUERIES]
    samples = _samples(dict.fromkeys(ids, 3.0), dict.fromkeys(ids, 2.0))
    text = render_markdown(
        samples,
        title="t",
        meta={"커밋": "abc"},
        deltas=[paired_delta(samples, QUERIES, HYBRID_EVAL, resamples=50)],
        equality=equality_rows(samples, HYBRID_EVAL, 3),
        memo={HYBRID_EVAL: (4, 4)},
        embedding_usage={"합계": "0회"},
        notes=["메모"],
    )
    assert "| hybrid (eval 경로, k=10, window=1) | sequential | 12 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 |" in text
    assert "4 / 4" in text
    assert "오류 표본: 0개" in text
    assert "## 메모" in text
