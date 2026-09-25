from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from eval.dataset import (
    DatasetError,
    EvalQuery,
    chunk_synth_query,
    dump_queries,
    known_item_queries,
    load_queries,
    longest_shared_word_run,
    parse_lines,
)
from eval.report import render_markdown, render_readme_table, write_query_csv, write_summary_csv
from eval.runner import (
    ABLATIONS,
    METHODS,
    aggregate,
    plain_rrf,
    resolve_methods,
    run_evaluation,
    run_query,
)
from src.integrations.paper_retriever import PaperRetriever

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PATH = REPO_ROOT / "eval" / "queries.sample.jsonl"


def _line(**overrides) -> str:
    payload = {
        "id": "q1",
        "query": "sparse attention",
        "lang": "en",
        "relevant_arxiv_ids": ["2401.00001"],
        "source": "manual",
        "notes": "",
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


class TestDataset:
    def test_sample_file_loads(self):
        queries = load_queries(SAMPLE_PATH)
        assert len(queries) == 6
        assert {query.lang for query in queries} == {"ko", "en"}
        assert {query.source for query in queries} == {"known_item", "llm_synth", "manual"}
        assert any(query.relevant_chunk_ids for query in queries)

    def test_roundtrip(self):
        queries = load_queries(SAMPLE_PATH)
        assert parse_lines(dump_queries(queries).splitlines()) == queries

    def test_blank_lines_are_ignored(self):
        assert len(parse_lines(["", _line(), "   "])) == 1

    @pytest.mark.parametrize(
        "overrides, message",
        [
            ({"lang": "ja"}, "'lang'"),
            ({"source": "crawl"}, "'source'"),
            ({"relevant_arxiv_ids": []}, "relevant_arxiv_ids"),
            ({"relevant_chunk_ids": ["12"]}, "relevant_chunk_ids"),
            ({"relevant_chunk_ids": [True]}, "relevant_chunk_ids"),
            ({"query": "  "}, "'query'"),
            ({"notes": 3}, "'notes'"),
        ],
    )
    def test_validation_errors_name_the_field_and_line(self, overrides, message):
        with pytest.raises(DatasetError, match=message) as excinfo:
            parse_lines([_line(), _line(id="q2", **overrides)], origin="queries.jsonl")
        assert "queries.jsonl:2" in str(excinfo.value)

    def test_duplicate_ids_rejected(self):
        with pytest.raises(DatasetError, match="duplicate id"):
            parse_lines([_line(), _line()])

    def test_invalid_json_rejected(self):
        with pytest.raises(DatasetError, match="invalid JSON"):
            parse_lines(["{not json"])

    def test_duplicate_relevant_ids_collapse(self):
        (query,) = parse_lines([_line(relevant_arxiv_ids=["a", "a", "b"], relevant_chunk_ids=[3, 3])])
        assert query.relevant_arxiv_ids == ("a", "b")
        assert query.relevant_chunk_ids == (3,)

    def test_longest_shared_word_run(self):
        source = "We propose a sparse attention mechanism that scales linearly with sequence length."
        assert longest_shared_word_run("a sparse attention mechanism that scales", source) == 6
        assert longest_shared_word_run("Linear-time attention for long inputs", source) == 1
        assert longest_shared_word_run("", source) == 0

    def test_known_item_builder_skips_empty_language(self):
        records = known_item_queries(
            arxiv_id="2401.00001", title="T", queries_by_lang={"ko": "질의", "en": " "}, basis="abstract"
        )
        assert [record.id for record in records] == ["ki-2401.00001-ko"]
        assert records[0].source == "known_item"

    def test_chunk_synth_builder(self):
        record = chunk_synth_query(
            chunk_id=7, arxiv_id="2401.00001", chunk_index=3, section_title="Method", question="Q?", lang="en"
        )
        assert record is not None
        assert record.relevant_chunk_ids == (7,)
        assert record.source == "llm_synth"
        assert (
            chunk_synth_query(chunk_id=7, arxiv_id="a", chunk_index=3, section_title="", question=" ", lang="en")
            is None
        )


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _hit(arxiv_id: str, chunk_id: int, role: str = "body") -> dict:
    return {"arxiv_id": arxiv_id, "chunk_id": chunk_id, "content_role": role, "chunk_text": "", "score": 1.0}


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
        "first": [_hit("A", 1), _hit("X", 2, "references")],
        "second": [_hit("X", 3), _hit("Y", 4), _hit("B", 22), _hit("B", 21)],
        "셋째": [],
    },
    "vector": {
        "first": [_hit("X", 2), _hit("A", 1)],
        "second": [_hit("B", 21)],
        "셋째": [_hit("C", 31)],
    },
    "hybrid": {
        "first": [_hit("A", 1)],
        "second": [_hit("B", 21)],
        "셋째": [_hit("Y", 4), _hit("C", 31, "front_matter")],
    },
}
BASE_LATENCY_S = {"first": 0.010, "second": 0.020, "셋째": 0.030}
METHOD_SLOWDOWN = {"lexical": 1, "vector": 2, "hybrid": 3}


class FakeRetriever:
    def __init__(self, clock: FakeClock, *, failing_query: str | None = None) -> None:
        self.clock = clock
        self.failing_query = failing_query
        self.calls: list[tuple[str, str, int, int]] = []

    def _respond(self, method: str, query: str, limit: int, adjacency_window: int) -> list[dict]:
        self.calls.append((method, query, limit, adjacency_window))
        self.clock.now += BASE_LATENCY_S[query] * METHOD_SLOWDOWN[method]
        if query == self.failing_query:
            raise RuntimeError("embedding API down")
        return RANKINGS[method][query][:limit]

    def search_paper_contexts(self, query, *, limit, adjacency_window):
        return self._respond("lexical", query, limit, adjacency_window)

    def search_paper_contexts_by_vector(self, query, *, limit, adjacency_window):
        return self._respond("vector", query, limit, adjacency_window)

    def search_paper_contexts_by_hybrid(self, query, *, limit, adjacency_window):
        return self._respond("hybrid", query, limit, adjacency_window)


def _run(**kwargs):
    clock = FakeClock()
    retriever = FakeRetriever(clock, **kwargs.pop("retriever_kwargs", {}))
    results = run_evaluation(retriever, QUERIES, ["lexical", "vector", "hybrid"], k=10, clock=clock, **kwargs)
    return retriever, results


class TestRunEvaluation:
    def test_calls_public_context_search_with_k_and_window(self):
        retriever, results = _run(adjacency_window=0)
        assert len(results) == 9
        assert retriever.calls[:3] == [
            ("lexical", "first", 10, 0),
            ("vector", "first", 10, 0),
            ("hybrid", "first", 10, 0),
        ]

    def test_per_query_scoring(self):
        _, results = _run()
        by_key = {(row.method, row.query_id): row for row in results}
        lexical_second = by_key[("lexical", "q-en-2")]
        assert lexical_second.paper_metrics["hit@1"] == 0.0
        assert lexical_second.paper_metrics["hit@5"] == 1.0
        assert lexical_second.paper_metrics["mrr@10"] == pytest.approx(1 / 3)
        assert lexical_second.chunk_metrics["mrr@10"] == pytest.approx(0.25)
        assert by_key[("lexical", "q-en-1")].chunk_metrics == {}
        assert by_key[("lexical", "q-en-1")].noise_count == 1
        assert by_key[("vector", "q-ko-1")].latency_ms == pytest.approx(60.0)

    def test_aggregate_table(self):
        _, results = _run()
        rows = {(row.method, row.subset): row for row in aggregate(results)}
        subsets = ("all", "ko", "en", "known_item", "llm_synth")
        assert set(rows) == {(method, subset) for method in METHOD_SLOWDOWN for subset in subsets}
        assert rows[("lexical", "known_item")].n_queries == 2
        assert rows[("lexical", "llm_synth")].paper["mrr@10"] == pytest.approx(1 / 3)

        lexical = rows[("lexical", "all")]
        assert lexical.n_queries == 3
        assert lexical.paper["hit@1"] == pytest.approx(1 / 3)
        assert lexical.paper["hit@5"] == pytest.approx(2 / 3)
        assert lexical.paper["hit@10"] == pytest.approx(2 / 3)
        assert lexical.paper["mrr@10"] == pytest.approx((1 + 1 / 3 + 0) / 3)
        assert lexical.paper["recall@10"] == pytest.approx(2 / 3)
        assert lexical.noise_rate == pytest.approx(1 / 6)
        assert lexical.latency_p50_ms == pytest.approx(20.0)
        assert lexical.latency_p95_ms == pytest.approx(29.0)
        assert lexical.n_chunk_queries == 1
        assert lexical.chunk["hit@1"] == 0.0
        assert lexical.chunk["hit@5"] == 1.0

        assert rows[("lexical", "ko")].paper["mrr@10"] == 0.0
        assert rows[("lexical", "ko")].noise_rate is None
        assert rows[("vector", "all")].paper["mrr@10"] == pytest.approx((0.5 + 1 + 1) / 3)
        assert rows[("hybrid", "ko")].paper["mrr@10"] == pytest.approx(0.5)
        assert rows[("hybrid", "all")].noise_rate == pytest.approx(1 / 4)
        assert rows[("vector", "ko")].n_chunk_queries == 0
        assert rows[("vector", "ko")].chunk["hit@1"] is None

    def test_markdown_report(self):
        _, results = _run()
        aggregates = aggregate(results)
        readme_table = render_readme_table(aggregates)
        assert readme_table.splitlines()[2] == "| lexical | 0.333 | 0.667 | 0.667 | 0.444 | 20 / 29 |"
        assert readme_table.splitlines()[3] == "| vector | 0.667 | 1.000 | 1.000 | 0.833 | 40 / 58 |"

        markdown = render_markdown(aggregates, title="Retrieval evaluation test", meta={"커밋": "abc123"})
        assert markdown.startswith("# Retrieval evaluation test\n\n- 커밋: abc123\n")
        assert readme_table in markdown
        assert "| lexical | ko | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | n/a | 30 | 30 | 0 |" in markdown
        assert "## 청크 단위" in markdown

    def test_errors_propagate_by_default(self):
        with pytest.raises(RuntimeError, match="embedding API down"):
            _run(retriever_kwargs={"failing_query": "second"})

    def test_keep_going_records_errors_and_excludes_them(self):
        _, results = _run(keep_going=True, retriever_kwargs={"failing_query": "second"})
        failed = [row for row in results if not row.ok]
        assert len(failed) == 3
        assert all("embedding API down" in (row.error or "") for row in failed)
        lexical = next(row for row in aggregate(results) if row.method == "lexical" and row.subset == "all")
        assert lexical.n_queries == 2
        assert lexical.n_errors == 1
        assert lexical.paper["mrr@10"] == pytest.approx(0.5)
        assert lexical.n_chunk_queries == 0

    def test_csv_outputs(self, tmp_path):
        _, results = _run()
        write_query_csv(results, tmp_path / "run.csv")
        write_summary_csv(aggregate(results), tmp_path / "run_summary.csv")

        with (tmp_path / "run.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 9
        first = rows[0]
        assert first["method"] == "lexical"
        assert first["paper_hit@1"] == "1.0000"
        assert first["chunk_hit@1"] == ""
        assert first["retrieved_roles"] == "body references"

        with (tmp_path / "run_summary.csv").open(encoding="utf-8") as handle:
            summary = list(csv.DictReader(handle))
        assert summary[0]["method"] == "lexical"
        assert summary[0]["subset"] == "all"
        assert summary[0]["noise_rate"] == "0.1667"

    def test_small_k_limits_hit_cutoffs(self):
        clock = FakeClock()
        result = run_query(FakeRetriever(clock), "lexical", QUERIES[1], k=3, clock=clock)
        assert set(result.paper_metrics) == {"hit@1", "mrr@3", "recall@3"}
        assert result.paper_metrics["hit@1"] == 0.0
        assert result.paper_metrics["mrr@3"] == pytest.approx(1 / 3)


class TestResolveMethods:
    def test_base_methods(self):
        assert resolve_methods(["lexical", " vector", ""]) == ["lexical", "vector"]

    def test_all_ablations(self):
        methods = resolve_methods(["lexical", "vector", "hybrid"], ["all"])
        expected_ablations = [name for names in ABLATIONS.values() for name in names]
        assert methods == ["lexical", "vector", "hybrid", *expected_ablations]

    def test_unknown_names_rejected(self):
        with pytest.raises(ValueError, match="unknown method"):
            resolve_methods(["bm25"])
        with pytest.raises(ValueError, match="unknown ablation"):
            resolve_methods(["lexical"], ["rerank"])


def _row(chunk_id: int, arxiv_id: str, score: float, *, role: str = "body", section: str = "Method") -> dict:
    return {
        "chunk_id": chunk_id,
        "arxiv_id": arxiv_id,
        "paper_title": f"Paper {arxiv_id}",
        "paper_abstract": "",
        "chunk_text": "We evaluate the proposed model on standard benchmarks.",
        "chunk_index": chunk_id,
        "section_title": section,
        "content_role": role,
        "score": score,
    }


class FakeRepository:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.fetch_limits: list[int] = []

    def list_chunk_candidates_by_query(self, query, *, limit, arxiv_id=None):
        self.fetch_limits.append(limit)
        return [dict(row) for row in self.rows[:limit]]

    def list_chunk_window(self, arxiv_id, center_chunk_index, *, window=1):
        return [{"chunk_text": f"context {center_chunk_index}", "metadata": {"content_role": "body"}}]


class FakeEmbeddingClient:
    def embed_texts(self, texts):
        return [[0.0] for _ in texts]


class FakeVectorRepository:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def search_paper_chunks(self, embedding, *, limit, arxiv_id=None):
        return [dict(row) for row in self.rows[:limit]]


LEXICAL_ROWS = [
    _row(10, "A", 0.99, role="references", section="References"),
    _row(1, "A", 0.9),
    _row(2, "A", 0.8),
    _row(3, "A", 0.7),
    _row(4, "B", 0.6),
    _row(5, "C", 0.5),
]
VECTOR_ROWS = [
    _row(6, "D", 0.52, section="Appendix B"),
    _row(1, "A", 0.5),
    _row(2, "A", 0.49),
    _row(3, "A", 0.48),
    _row(4, "B", 0.47),
]


def _real_retriever() -> PaperRetriever:
    return PaperRetriever(
        repository=FakeRepository(LEXICAL_ROWS),
        embedding_client=FakeEmbeddingClient(),
        vector_repository=FakeVectorRepository(VECTOR_ROWS),
    )


def _chunk_ids(method: str, k: int = 3) -> list[int]:
    query = EvalQuery(id="q", query="benchmark evaluation", lang="en", relevant_arxiv_ids=("A",), source="manual")
    return run_query(_real_retriever(), method, query, k=k).retrieved_chunk_ids


class TestAblationsOnRealRetriever:
    def test_base_lexical_filters_references_and_diversifies(self):
        assert _chunk_ids("lexical") == [1, 2, 4]

    def test_lexical_nodiv_uses_the_same_candidate_pool(self):
        retriever = _real_retriever()
        retriever.search_paper_chunks("benchmark evaluation", limit=3)
        query = EvalQuery(id="q", query="benchmark evaluation", lang="en", relevant_arxiv_ids=("A",), source="manual")
        result = run_query(retriever, "lexical_nodiv", query, k=3)
        assert retriever.repository.fetch_limits == [10, 10]
        assert result.retrieved_chunk_ids == [1, 2, 3]

    def test_lexical_nofilter_keeps_reference_chunk(self):
        query = EvalQuery(id="q", query="benchmark evaluation", lang="en", relevant_arxiv_ids=("A",), source="manual")
        result = run_query(_real_retriever(), "lexical_nofilter", query, k=3)
        assert result.retrieved_chunk_ids == [10, 1, 4]
        assert result.noise_count == 1

    def test_vector_rerank_demotes_appendix_and_norerank_keeps_sql_order(self):
        assert _chunk_ids("vector")[0] == 1
        assert _chunk_ids("vector_norerank") == [6, 1, 2]

    def test_vector_nodiv(self):
        assert _chunk_ids("vector_nodiv") == [1, 2, 3]

    def test_hybrid_nodiv_lets_one_paper_fill_top_k(self):
        assert _chunk_ids("hybrid") == [1, 2, 4]
        assert _chunk_ids("hybrid_nodiv") == [1, 2, 3]

    def test_hybrid_plainrrf_runs_through_retriever(self):
        assert _chunk_ids("hybrid_plainrrf") == [1, 2, 4]

    def test_ablation_hits_carry_context_text(self):
        query = EvalQuery(id="q", query="benchmark evaluation", lang="en", relevant_arxiv_ids=("A",), source="manual")
        retriever = _real_retriever()
        hits = METHODS["lexical_nodiv"].search(retriever, query.query, 2, 1)
        assert [hit["context_text"] for hit in hits] == ["context 1", "context 2"]


class TestPlainRrf:
    def test_sums_reciprocal_ranks_without_weights(self):
        merged = plain_rrf([[{"chunk_id": 1}, {"chunk_id": 2}], [{"chunk_id": 2}, {"chunk_id": 3}]])
        assert [item["chunk_id"] for item in merged] == [2, 1, 3]
        assert merged[0]["score"] == pytest.approx(1 / 62 + 1 / 61)
        assert merged[1]["score"] == pytest.approx(1 / 61)

    def test_ties_break_by_chunk_id_descending(self):
        merged = plain_rrf([[{"chunk_id": 1}], [{"chunk_id": 5}]])
        assert [item["chunk_id"] for item in merged] == [5, 1]


def _load_script(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"_eval_script_{name}", REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestEvalRetrievalScript:
    @pytest.fixture(autouse=True)
    def _settings(self, monkeypatch):
        import types

        import src.shared

        settings = types.SimpleNamespace(
            openai_api_key="sk-test", openai_embedding_model="text-embedding-3-large", openai_embedding_dimensions=1536
        )
        monkeypatch.setattr(src.shared, "get_settings", lambda: settings)

    def _queries_file(self, tmp_path: Path) -> Path:
        path = tmp_path / "queries.jsonl"
        path.write_text(dump_queries(QUERIES), encoding="utf-8")
        return path

    def test_unreachable_db_exits_without_writing(self, tmp_path, monkeypatch, capsys):
        script = _load_script("eval_retrieval")

        def refuse(settings):
            raise OSError("connection refused")

        monkeypatch.setattr(script, "_connect", refuse)
        out_dir = tmp_path / "results"
        code = script.main(["--queries", str(self._queries_file(tmp_path)), "--out-dir", str(out_dir)])
        assert code == script.EXIT_PRECONDITION
        assert "PostgreSQL에 연결할 수 없습니다" in capsys.readouterr().err
        assert not out_dir.exists()

    def test_end_to_end_with_fake_retriever(self, tmp_path, monkeypatch, capsys):
        import src.integrations.paper_retriever as retriever_module

        script = _load_script("eval_retrieval")
        corpus = {"papers": 3, "papers_with_chunks": 3, "chunks": 9, "embeddings": 9, "fulltext_sources": {"pdf": 3}}
        monkeypatch.setattr(script, "preflight", lambda settings, queries, needs_embeddings: corpus)
        monkeypatch.setattr(retriever_module, "PaperRetriever", lambda: FakeRetriever(FakeClock()))
        out_dir = tmp_path / "results"

        code = script.main(["--queries", str(self._queries_file(tmp_path)), "--out-dir", str(out_dir)])

        assert code == 0
        written = sorted(path.name for path in out_dir.iterdir())
        assert len(written) == 3
        assert any(name.endswith("_summary.csv") for name in written)
        markdown = next(out_dir.glob("*.md")).read_text(encoding="utf-8")
        assert "| lexical | 0.333 | 0.667 | 0.667 | 0.444 |" in markdown
        assert "(ko 1 / en 2; known_item 2, llm_synth 1)" in markdown
        assert markdown in capsys.readouterr().out


class TestBuildQueriesScript:
    def test_sampling_is_deterministic_and_streams_are_independent(self):
        script = _load_script("eval_build_queries")
        ids = list(range(100))
        first = script.sample_ids(ids, 5, seed=42, stream="papers")
        assert first == script.sample_ids(ids, 5, seed=42, stream="papers")
        assert first != script.sample_ids(ids, 5, seed=42, stream="chunks")
        assert first != script.sample_ids(ids, 5, seed=7, stream="papers")
        assert script.sample_ids(ids[:3], 10, seed=42, stream="papers") == script.sample_ids(
            ids[:3], 3, seed=42, stream="papers"
        )

    def test_chunk_language_policy(self):
        script = _load_script("eval_build_queries")
        assert [script.chunk_lang_for(index, "alternate") for index in range(3)] == ["en", "ko", "en"]
        assert script.chunk_lang_for(1, "en") == "en"

    def test_generate_rejects_verbatim_and_empty_outputs(self, monkeypatch):
        import types

        script = _load_script("eval_build_queries")
        abstract = "We introduce a routing scheme that keeps expert load balanced without any auxiliary loss term."
        responses = iter(
            [
                {"ko": "보조 손실 없이 전문가 부하를 고르게 맞추는 MoE 라우팅", "en": abstract},
                {"question": "   "},
                {"question": "Which batch size was used for the ablation runs?"},
            ]
        )

        class FakeStructuredLlm:
            def invoke(self, messages):
                return types.SimpleNamespace(**next(responses))

        class FakeLlm:
            def with_structured_output(self, schema):
                return FakeStructuredLlm()

        monkeypatch.setattr(script, "build_llm", lambda: FakeLlm())
        paper = script.PaperSample(arxiv_id="2401.00001", title="Balanced MoE", abstract=abstract, key_findings=())
        chunks = [
            script.ChunkSample(1, "2401.00001", 0, "Method", "text", "Balanced MoE", "en"),
            script.ChunkSample(2, "2401.00001", 5, "Experiments", "We use batch size 512.", "Balanced MoE", "en"),
        ]

        records, rejected = script.generate_queries([paper], chunks, max_shared_words=5)

        assert [record.id for record in records] == ["ki-2401.00001-ko", "cs-2"]
        assert records[1].relevant_chunk_ids == (2,)
        assert len(rejected) == 2
        assert rejected[0].startswith("ki-2401.00001-en")
        assert rejected[1].startswith("cs-1")
