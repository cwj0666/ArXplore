import csv
from typing import Any

import pytest

from arxplore.domain.retrieval import RetrievalContext
from arxplore_eval.dataset import EvalQuery
from arxplore_eval.report import render_markdown, render_readme_table, write_query_csv, write_summary_csv
from arxplore_eval.runner import (
    ABLATIONS,
    METHODS,
    RankedHit,
    aggregate,
    needs_embeddings,
    resolve_methods,
    run_evaluation,
    run_query,
    score_hits,
    split_retrieval_queries,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


QUERIES = [
    EvalQuery(id="q-en-1", query="first", lang="en", relevant_arxiv_ids=("A",), source="known_item"),
    EvalQuery(
        id="q-en-2",
        query="second",
        lang="en",
        relevant_arxiv_ids=("B",),
        relevant_chunk_ids=(21,),
        source="llm_synth",
    ),
    EvalQuery(id="q-ko-1", query="셋째", lang="ko", relevant_arxiv_ids=("C",), source="known_item"),
]
RANKINGS = {
    "lexical": {
        "first": [("A", 1, "body"), ("X", 2, "references")],
        "second": [("X", 3, "body"), ("Y", 4, "body"), ("B", 22, "body"), ("B", 21, "body")],
        "셋째": [],
    },
    "vector": {
        "first": [("X", 2, "body"), ("A", 1, "body")],
        "second": [("B", 21, "body")],
        "셋째": [("C", 31, "body")],
    },
    "hybrid": {
        "first": [("A", 1, "body")],
        "second": [("B", 21, "body")],
        "셋째": [("Y", 4, "body"), ("C", 31, "front_matter")],
    },
}
BASE_LATENCY_S = {"first": 0.010, "second": 0.020, "셋째": 0.030}
METHOD_SLOWDOWN = {"lexical": 1, "vector": 2, "hybrid": 3}


class FakeRetriever:
    def __init__(self, clock: FakeClock, make_hit, make_context, *, failing_query: str | None = None) -> None:
        self.clock = clock
        self.make_hit = make_hit
        self.make_context = make_context
        self.failing_query = failing_query
        self.calls: list[tuple[str, str, int, int]] = []

    async def _respond(self, method: str, query: str, limit: int, adjacency_window: int) -> list[RetrievalContext]:
        self.calls.append((method, query, limit, adjacency_window))
        self.clock.now += BASE_LATENCY_S[query] * METHOD_SLOWDOWN[method]
        if query == self.failing_query:
            raise RuntimeError("embedding API down")
        return [
            self.make_context(self.make_hit(chunk_id, arxiv_id=arxiv_id, content_role=role))
            for arxiv_id, chunk_id, role in RANKINGS[method][query][:limit]
        ]

    async def search_lexical_contexts(self, query: str, *, limit: int, adjacency_window: int) -> list[Any]:
        return await self._respond("lexical", query, limit, adjacency_window)

    async def search_vector_contexts(self, query: str, *, limit: int, adjacency_window: int) -> list[Any]:
        return await self._respond("vector", query, limit, adjacency_window)

    async def search_hybrid_contexts(self, query: str, *, limit: int, adjacency_window: int) -> list[Any]:
        return await self._respond("hybrid", query, limit, adjacency_window)


@pytest.fixture
def run(make_hit, make_context):
    async def execute(**kwargs: Any) -> tuple[FakeRetriever, list[Any]]:
        clock = FakeClock()
        retriever = FakeRetriever(clock, make_hit, make_context, **kwargs.pop("retriever_kwargs", {}))
        results = await run_evaluation(retriever, QUERIES, ["lexical", "vector", "hybrid"], k=10, clock=clock, **kwargs)
        return retriever, results

    return execute


class TestRunEvaluation:
    async def test_calls_the_product_context_search_with_k_and_window(self, run) -> None:
        retriever, results = await run(adjacency_window=0)
        assert len(results) == 9
        assert retriever.calls[:3] == [
            ("lexical", "first", 10, 0),
            ("vector", "first", 10, 0),
            ("hybrid", "first", 10, 0),
        ]

    async def test_per_query_scoring(self, run) -> None:
        _, results = await run()
        by_key = {(row.method, row.query_id): row for row in results}
        lexical_second = by_key[("lexical", "q-en-2")]
        assert lexical_second.paper_metrics["hit@1"] == 0.0
        assert lexical_second.paper_metrics["hit@5"] == 1.0
        assert lexical_second.paper_metrics["mrr@10"] == pytest.approx(1 / 3)
        assert lexical_second.chunk_metrics["mrr@10"] == pytest.approx(0.25)
        assert by_key[("lexical", "q-en-1")].chunk_metrics == {}
        assert by_key[("lexical", "q-en-1")].noise_count == 1
        assert by_key[("vector", "q-ko-1")].latency_ms == pytest.approx(60.0)

    async def test_aggregate_table(self, run) -> None:
        _, results = await run()
        rows = {(row.method, row.subset): row for row in aggregate(results)}
        subsets = ("all", "ko", "en", "known_item", "llm_synth")
        assert set(rows) == {(method, subset) for method in METHOD_SLOWDOWN for subset in subsets}
        assert rows[("lexical", "known_item")].n_queries == 2
        assert rows[("lexical", "llm_synth")].paper["mrr@10"] == pytest.approx(1 / 3)

        lexical = rows[("lexical", "all")]
        assert lexical.n_queries == 3
        assert lexical.paper["hit@1"] == pytest.approx(1 / 3)
        assert lexical.paper["hit@5"] == pytest.approx(2 / 3)
        assert lexical.paper["mrr@10"] == pytest.approx((1 + 1 / 3 + 0) / 3)
        assert lexical.paper["recall@10"] == pytest.approx(2 / 3)
        assert lexical.noise_rate == pytest.approx(1 / 6)
        assert lexical.latency_p50_ms == pytest.approx(20.0)
        assert lexical.latency_p95_ms == pytest.approx(29.0)
        assert lexical.n_chunk_queries == 1
        assert lexical.chunk["hit@1"] == 0.0 and lexical.chunk["hit@5"] == 1.0

        assert rows[("lexical", "ko")].paper["mrr@10"] == 0.0
        assert rows[("lexical", "ko")].noise_rate is None
        assert rows[("vector", "all")].paper["mrr@10"] == pytest.approx((0.5 + 1 + 1) / 3)
        assert rows[("hybrid", "ko")].paper["mrr@10"] == pytest.approx(0.5)
        assert rows[("hybrid", "all")].noise_rate == pytest.approx(1 / 4)
        assert rows[("vector", "ko")].chunk["hit@1"] is None

    async def test_markdown_report(self, run) -> None:
        _, results = await run()
        aggregates = aggregate(results)
        readme_table = render_readme_table(aggregates)
        assert readme_table.splitlines()[2] == "| lexical | 0.333 | 0.667 | 0.667 | 0.444 | 20 / 29 |"
        assert readme_table.splitlines()[3] == "| vector | 0.667 | 1.000 | 1.000 | 0.833 | 40 / 58 |"

        markdown = render_markdown(aggregates, title="Retrieval evaluation test", meta={"커밋": "abc123"})
        assert markdown.startswith("# Retrieval evaluation test\n\n- 커밋: abc123\n")
        assert readme_table in markdown
        assert "| lexical | ko | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | n/a | 30 | 30 | 0 |" in markdown
        assert "## 청크 단위" in markdown
        assert "- `lexical`: Retriever.search_lexical_contexts (제품 경로)" in markdown
        assert render_markdown([], title="t", meta={}).endswith("집계할 결과가 없습니다.\n")

    async def test_errors_propagate_by_default(self, run) -> None:
        with pytest.raises(RuntimeError, match="embedding API down"):
            await run(retriever_kwargs={"failing_query": "second"})

    async def test_keep_going_records_errors_and_excludes_them(self, run) -> None:
        _, results = await run(keep_going=True, retriever_kwargs={"failing_query": "second"})
        failed = [row for row in results if not row.ok]
        assert len(failed) == 3
        assert all("embedding API down" in (row.error or "") for row in failed)
        lexical = next(row for row in aggregate(results) if row.method == "lexical" and row.subset == "all")
        assert (lexical.n_queries, lexical.n_errors) == (2, 1)
        assert lexical.paper["mrr@10"] == pytest.approx(0.5)
        assert lexical.n_chunk_queries == 0

    async def test_csv_outputs(self, run, tmp_path) -> None:
        _, results = await run()
        write_query_csv(results, tmp_path / "run.csv")
        write_summary_csv(aggregate(results), tmp_path / "run_summary.csv")

        with (tmp_path / "run.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 9
        assert rows[0]["method"] == "lexical"
        assert rows[0]["paper_hit@1"] == "1.0000"
        assert rows[0]["chunk_hit@1"] == ""
        assert rows[0]["retrieved_roles"] == "body references"
        assert rows[0]["retrieved_chunk_ids"] == "1 2"

        with (tmp_path / "run_summary.csv").open(encoding="utf-8") as handle:
            summary = list(csv.DictReader(handle))
        assert (summary[0]["method"], summary[0]["subset"], summary[0]["noise_rate"]) == ("lexical", "all", "0.1667")

    async def test_small_k_limits_hit_cutoffs(self, make_hit, make_context) -> None:
        clock = FakeClock()
        result = await run_query(FakeRetriever(clock, make_hit, make_context), "lexical", QUERIES[1], k=3, clock=clock)
        assert set(result.paper_metrics) == {"hit@1", "mrr@3", "recall@3"}
        assert result.paper_metrics["mrr@3"] == pytest.approx(1 / 3)


class TestResolveMethods:
    def test_base_methods(self) -> None:
        assert resolve_methods(["lexical", " vector", ""]) == ["lexical", "vector"]

    def test_all_ablations(self) -> None:
        methods = resolve_methods(["lexical", "vector", "hybrid"], ["all"])
        expected_ablations = [name for names in ABLATIONS.values() for name in names]
        assert methods == ["lexical", "vector", "hybrid", *expected_ablations]
        assert len(methods) == len(METHODS) == 10

    def test_unknown_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown method"):
            resolve_methods(["bm25"])
        with pytest.raises(ValueError, match="unknown ablation"):
            resolve_methods(["lexical"], ["rerank"])

    def test_embedding_needs(self) -> None:
        assert not needs_embeddings(["lexical", "lexical_nodiv", "lexical_nofilter"])
        assert needs_embeddings(["lexical", "hybrid_rules"])


def _q(query_id: str, **kwargs: Any) -> EvalQuery:
    base: dict[str, Any] = {
        "query": "question",
        "lang": "en",
        "relevant_arxiv_ids": ("2401.00001",),
        "source": "manual",
    }
    base.update(kwargs)
    return EvalQuery(id=query_id, **base)


CATEGORY_QUERIES = [
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


class TestCategories:
    class Retriever:
        def __init__(self, make_hit, make_context) -> None:
            self.queries: list[str] = []
            self.make_hit = make_hit
            self.make_context = make_context

        async def search_lexical_contexts(self, query: str, *, limit: int, adjacency_window: int) -> list[Any]:
            self.queries.append(query)
            return [self.make_context(self.make_hit(1, arxiv_id="A"))]

    async def test_split_and_run_skip_ineligible_cases(self, make_hit, make_context) -> None:
        kept, skipped = split_retrieval_queries(CATEGORY_QUERIES)
        assert [query.id for query in kept] == ["lang-1", "lang-2", "qf-1"]
        assert [query.id for query in skipped] == ["oc-1", "sf-1", "pc-1", "cv-1"]
        retriever = self.Retriever(make_hit, make_context)
        results = await run_evaluation(retriever, CATEGORY_QUERIES, ["lexical"], k=5)
        assert retriever.queries == ["l1", "l2", "q1"]
        assert [(row.query_id, row.category, row.expected_behavior) for row in results] == [
            ("lang-1", "language", "answer"),
            ("lang-2", "language", "answer"),
            ("qf-1", "query_form", "clarify"),
        ]

    async def test_aggregate_by_category_and_behavior(self, make_hit, make_context) -> None:
        results = await run_evaluation(self.Retriever(make_hit, make_context), CATEGORY_QUERIES, ["lexical"], k=5)
        rows = {row.subset: row for row in aggregate(results)}
        assert rows["category:language"].n_queries == 2
        assert rows["category:language"].paper["hit@1"] == pytest.approx(0.5)
        assert rows["behavior:answer"].n_queries == 2
        assert rows["behavior:clarify"].n_queries == 1
        assert "category:out_of_corpus" not in rows

        single = await run_evaluation(self.Retriever(make_hit, make_context), CATEGORY_QUERIES[:2], ["lexical"], k=5)
        assert not any(row.subset.startswith("behavior:") for row in aggregate(single))


class TestScoreHits:
    def test_ranked_hits_from_mappings_skip_missing_chunk_ids(self) -> None:
        query = _q("q", relevant_arxiv_ids=("A",), relevant_chunk_ids=(3,))
        hits = [
            RankedHit.from_mapping({"chunk_id": None, "arxiv_id": "B"}),
            RankedHit.from_mapping({"chunk_id": 3, "arxiv_id": "A"}),
        ]

        result = score_hits(query, "replay", hits, k=5, latency_ms=None)

        assert result.retrieved_chunk_ids == [3]
        assert result.retrieved_arxiv_ids == ["B", "A"]
        assert result.paper_metrics["mrr@5"] == 0.5
        assert result.chunk_metrics["hit@1"] == 1.0
        assert result.retrieved_roles == ["", ""]
