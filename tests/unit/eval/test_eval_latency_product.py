import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from arxplore.config import AppSettings
from arxplore_eval.commands.latency import api_line, is_local_base_url, select_latency_queries
from arxplore_eval.dataset import EvalQuery
from arxplore_eval.latency import (
    EMBEDDING_STAGE,
    PARALLEL,
    PRODUCT,
    BenchmarkPlan,
    Sample,
    StageRecorder,
    render_markdown,
    run_benchmark,
    write_samples_csv,
)


def _query(query_id: str, lang: str) -> EvalQuery:
    return EvalQuery(id=query_id, query=f"q {query_id}", lang=lang, relevant_arxiv_ids=("A",), source="manual")


QUERIES = [_query("a-ko", "ko"), _query("a-en", "en"), _query("b-ko", "ko"), _query("c-ko", "ko"), _query("b-en", "en")]


async def test_product_only_plan_runs_one_variant_and_keeps_cold_samples(make_hit, make_context) -> None:
    calls: list[tuple[Any, str]] = []

    async def product(retriever: Any, query: str) -> tuple[list[Any], str]:
        calls.append((retriever, query))
        return [make_context(make_hit(1, arxiv_id="x"))], "hybrid"

    plan = BenchmarkPlan(
        queries=QUERIES[:2],
        retrievers={PARALLEL: "B"},
        recorder=StageRecorder(),
        paired_paths={PRODUCT: product},
        repeats=2,
        warmup=1,
        variants=(PARALLEL,),
        keep_warmup=True,
    )
    samples = await run_benchmark(plan)

    assert [retriever for retriever, _ in calls] == ["B"] * 6
    assert {sample.variant for sample in samples} == {PARALLEL}
    assert [(sample.repeat, sample.cold) for sample in samples if sample.query_id == "a-ko"] == [
        (-1, True),
        (0, False),
        (1, False),
    ]


async def test_warmup_is_dropped_without_keep_warmup(make_hit, make_context) -> None:
    async def product(retriever: Any, query: str) -> tuple[list[Any], str]:
        return [make_context(make_hit(1, arxiv_id="x"))], "hybrid"

    plan = BenchmarkPlan(
        queries=QUERIES[:1],
        retrievers={PARALLEL: "B"},
        recorder=StageRecorder(),
        paired_paths={PRODUCT: product},
        repeats=1,
        warmup=2,
        variants=(PARALLEL,),
    )
    samples = await run_benchmark(plan)
    assert [(sample.repeat, sample.cold) for sample in samples] == [(0, False)]


def _sample(total: float, embedding: float, *, cold: bool = False, repeat: int = 0) -> Sample:
    return Sample(
        "a-ko", "ko", repeat, PRODUCT, PARALLEL, 0, total, {EMBEDDING_STAGE: embedding}, ((1, "x"),), cold=cold
    )


def test_db_ms_is_total_minus_embedding_wait() -> None:
    assert _sample(300.0, 200.0).db_ms == 100.0
    assert _sample(100.0, 0.0).embedding_ms == 0.0
    assert _sample(100.0, 150.0).db_ms == 0.0


def test_samples_csv_has_cold_and_db_columns(tmp_path: Path) -> None:
    path = tmp_path / "latency.csv"
    write_samples_csv([_sample(400.0, 300.0, cold=True, repeat=-1), _sample(300.0, 200.0)], path)
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["cold"], row["db_ms"], row["embedding_ms"]) for row in rows] == [
        ("1", "100.000", "300.000"),
        ("0", "100.000", "200.000"),
    ]


def test_markdown_splits_embedding_and_lists_cold_samples() -> None:
    text = render_markdown(
        [_sample(300.0, 200.0)],
        title="t",
        meta={},
        deltas=[],
        equality=[],
        memo={},
        embedding_usage={},
        cold_samples=[_sample(500.0, 400.0, cold=True, repeat=-1)],
    )
    assert (
        "| 제품 경로 `retrieve_contexts` (limit=5) | parallel | 1 | 300.0 | 300.0 | 200.0 | 200.0 | 100.0 | 100.0 |"
        in text
    )
    assert "## cold" in text
    assert (
        "| 제품 경로 `retrieve_contexts` (limit=5) | parallel | 1 | 500.0 | 500.0 | 400.0 | 400.0 | 100.0 | 100.0 |"
        in text
    )
    assert "## 질의별 차이" not in text
    assert "| 제품 경로 `retrieve_contexts` (limit=5) | sequential |" not in text


def test_per_lang_selection_keeps_file_order() -> None:
    chosen = select_latency_queries(QUERIES, limit=None, per_lang=2)
    assert [query.id for query in chosen] == ["a-ko", "a-en", "b-ko", "b-en"]
    assert [query.id for query in select_latency_queries(QUERIES, limit=2, per_lang=None)] == ["a-ko", "a-en"]
    assert len(select_latency_queries(QUERIES, limit=None, per_lang=None)) == len(QUERIES)


def test_api_line_marks_a_local_fake_server() -> None:
    settings = cast(
        AppSettings, SimpleNamespace(openai_embedding_model="text-embedding-3-small", openai_embedding_dimensions=1536)
    )
    assert is_local_base_url("http://127.0.0.1:18765/v1")
    assert is_local_base_url("http://localhost:1/v1")
    assert not is_local_base_url("https://api.openai.com/v1")
    assert not is_local_base_url(None)
    assert "실제 API 왕복 아님" in api_line(settings, "http://127.0.0.1:18765/v1")
    assert api_line(settings, None) == "질의 임베딩 text-embedding-3-small (1536d) 왕복 포함, 캐시 없음"
