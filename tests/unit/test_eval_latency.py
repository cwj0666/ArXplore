from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest

from eval.latency import (
    CONTROL,
    HYBRID_EVAL,
    PARALLEL,
    SEQUENTIAL,
    VECTOR_ONLY,
    BenchmarkPlan,
    Sample,
    StageRecorder,
    embedding_cost,
    equality_rows,
    instrument,
    pair_order,
    paired_delta,
    render_markdown,
    run_benchmark,
    signature,
)
from src.integrations.paper_retriever import PaperRetriever


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _query(query_id: str, lang: str) -> SimpleNamespace:
    return SimpleNamespace(id=query_id, lang=lang, query=f"policy loss {query_id}")


QUERIES = [_query("a-ko", "ko"), _query("a-en", "en"), _query("b-ko", "ko"), _query("b-en", "en")]


def test_pair_order_alternates():
    assert pair_order(0) == (SEQUENTIAL, PARALLEL)
    assert pair_order(1) == (PARALLEL, SEQUENTIAL)
    assert pair_order(-1) == (PARALLEL, SEQUENTIAL)


def test_run_benchmark_interleaves_and_skips_warmup():
    calls: list[tuple[str, str]] = []

    def path(retriever, query):
        calls.append((retriever, query))
        return [{"chunk_id": 1, "arxiv_id": "x"}], "hybrid"

    plan = BenchmarkPlan(
        queries=QUERIES[:2],
        retrievers={SEQUENTIAL: "A", PARALLEL: "B"},
        recorder=StageRecorder(),
        paired_paths={HYBRID_EVAL: path},
        control_paths={VECTOR_ONLY: path},
        repeats=2,
        warmup=1,
    )
    samples = run_benchmark(plan)

    # 워밍업(B→A) 2질의 + 반복 0(A→B) + 반복 1(B→A), 블록마다 쌍 2 + 대조군 1
    assert [retriever for retriever, _ in calls] == ["B", "A", "B"] * 2 + ["A", "B", "B"] * 2 + ["B", "A", "B"] * 2
    assert len(samples) == 2 * 2 * 3
    assert {sample.repeat for sample in samples} == {0, 1}
    assert [s.variant for s in samples if s.repeat == 1 and s.query_id == "a-ko"] == [PARALLEL, SEQUENTIAL, CONTROL]


def test_run_benchmark_records_errors_and_keeps_going():
    def failing(retriever, query):
        raise RuntimeError("boom")

    plan = BenchmarkPlan(
        queries=QUERIES[:1],
        retrievers={SEQUENTIAL: "A", PARALLEL: "B"},
        recorder=StageRecorder(),
        paired_paths={HYBRID_EVAL: failing},
        repeats=1,
        warmup=0,
    )
    samples = run_benchmark(plan)
    assert [sample.error for sample in samples] == ["RuntimeError: boom"] * 2
    assert all(sample.signature == () for sample in samples)


def test_instrument_records_stages_without_changing_results():
    clock = FakeClock()

    class Repository:
        def list_chunk_candidates_by_query(self, query, *, limit, arxiv_id=None):
            clock.advance(0.010)
            return [{"chunk_id": 1, "arxiv_id": "2401.00001", "chunk_index": 0, "chunk_text": "policy loss"}]

        def list_chunk_windows(self, centers, *, window):
            clock.advance(0.002)
            return [[{"chunk_id": 1, "arxiv_id": a, "chunk_index": i, "chunk_text": "ctx"}] for a, i in centers]

    class Embeddings:
        def embed_texts(self, texts):
            clock.advance(0.100)
            return [[0.0] for _ in texts]

        def is_available(self):
            return True

    class Vectors:
        def search_paper_chunks(self, embedding, *, limit, arxiv_id=None):
            clock.advance(0.020)
            return [{"chunk_id": 2, "arxiv_id": "2401.00002", "chunk_index": 0, "chunk_text": "v", "score": 0.5}]

    plain = PaperRetriever(
        repository=Repository(), embedding_client=Embeddings(), vector_repository=Vectors(), parallel_channels=False
    )
    expected = plain.search_paper_contexts_by_hybrid("policy loss", limit=5)
    recorder = StageRecorder(clock=clock)
    timed = instrument(
        PaperRetriever(
            repository=Repository(), embedding_client=Embeddings(), vector_repository=Vectors(), parallel_channels=False
        ),
        recorder,
    )

    assert timed.search_paper_contexts_by_hybrid("policy loss", limit=5) == expected
    stages = recorder.snapshot_ms()
    assert stages["lexical_sql"] == pytest.approx(10.0)
    assert stages["embedding"] == pytest.approx(100.0)
    assert stages["vector_sql"] == pytest.approx(20.0)
    assert stages["vector_channel"] == pytest.approx(120.0)
    assert stages["context_window"] == pytest.approx(2.0)
    assert "fusion" in stages
    assert timed.embedding_client.is_available() is True


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


def test_paired_delta_uses_per_query_medians_and_clusters():
    a = {"a-ko": 300.0, "a-en": 500.0, "b-ko": 300.0, "b-en": 500.0}
    b = {"a-ko": 200.0, "a-en": 300.0, "b-ko": 200.0, "b-en": 300.0}
    row = paired_delta(_samples(a, b), QUERIES, HYBRID_EVAL, resamples=200, seed=1)

    assert (row.n, row.n_clusters) == (4, 2)
    assert row.delta_ms == pytest.approx(-150.0)
    assert row.low_ms == pytest.approx(-150.0)
    assert row.high_ms == pytest.approx(-150.0)
    assert row.relative == pytest.approx(((200 / 300 - 1) + (300 / 500 - 1)) / 2)

    en = paired_delta(_samples(a, b), QUERIES, HYBRID_EVAL, subset="en", resamples=200, seed=1)
    assert (en.n, en.n_clusters, en.delta_ms) == (2, 2, pytest.approx(-200.0))


def test_equality_rows_report_mismatches_and_baselines():
    samples = _samples(dict.fromkeys((q.id for q in QUERIES), 1.0), dict.fromkeys((q.id for q in QUERIES), 1.0), 2)
    samples = [
        Sample(**{**s.__dict__, "signature": ((2, "y"), (1, "x"))})
        if (s.variant == PARALLEL and s.repeat == 1 and s.query_id == "a-en")
        else s
        for s in samples
    ]
    rows = {row.comparison: row for row in equality_rows(samples, HYBRID_EVAL, 2)}

    assert rows["A vs B (반복 0)"].mismatches == 0
    assert rows["A vs B (반복 1)"].mismatch_ids == ("a-en",)
    assert rows["A vs B (반복 1)"].same_set_reordered == 0
    assert rows["A vs A (반복 0→1)"].mismatches == 0
    assert rows["B vs B (반복 0→1)"].mismatches == 1


def test_signature_and_cost():
    assert signature([{"chunk_id": 3, "arxiv_id": "a", "score": 1}]) == ((3, "a"),)
    assert embedding_cost(1_000_000) == pytest.approx(0.13)


def test_render_markdown_smoke():
    samples = _samples(dict.fromkeys((q.id for q in QUERIES), 3.0), dict.fromkeys((q.id for q in QUERIES), 2.0))
    text = render_markdown(
        samples,
        QUERIES,
        title="t",
        meta={"커밋": "abc"},
        deltas=[paired_delta(samples, QUERIES, HYBRID_EVAL, resamples=50)],
        equality=equality_rows(samples, HYBRID_EVAL, 3),
        memo_equality={HYBRID_EVAL: (4, 4)},
        embedding_usage={"합계": "0회"},
    )
    assert "| hybrid (eval 경로, k=10, window=1) | sequential | 12 | 3.0 | 3.0 |" in text
    assert "4 / 4" in text
