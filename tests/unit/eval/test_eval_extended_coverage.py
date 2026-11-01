import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from arxplore_eval import embed_cost
from arxplore_eval.cli import app
from arxplore_eval.commands.latency import _usage
from arxplore_eval.compare_runs import (
    CHUNK,
    PAPER,
    CompareError,
    QueryPair,
    RunRow,
    compare_runs,
    draws_cache,
    metric_delta,
    pair_runs,
    read_run_csv,
    render_markdown,
    write_queries_csv,
)
from arxplore_eval.dataset import EvalQuery, dump_queries
from arxplore_eval.embed_cost import CostReport, CostSummary, measure_cost
from arxplore_eval.embedders import token_counter
from arxplore_eval.fusion_sweep import cluster_key
from arxplore_eval.metrics import hit_at_k, mrr_at_k, ndcg_at_k, precision_at_k
from arxplore_eval.report import render_markdown as render_report_markdown
from arxplore_eval.report import write_query_csv, write_summary_csv
from arxplore_eval.runner import RankedHit, aggregate, all_metric_names, extended_metric_names, metric_names, score_hits

runner = CliRunner()


def _dcg(*positions: int) -> float:
    return sum(1 / math.log2(position + 1) for position in positions)


class TestRankMetricsEdges:
    def test_several_relevant_at_several_ranks(self) -> None:
        ranked = ["a", "x", "b", "y", "c"]
        expected = _dcg(1, 3, 5) / _dcg(1, 2, 3)
        assert ndcg_at_k(ranked, {"a", "b", "c"}, 5) == pytest.approx(expected)

    def test_ideal_is_capped_at_k_when_more_relevant_than_k(self) -> None:
        relevant = {"a", "b", "c", "d", "e"}
        assert ndcg_at_k(["x", "a", "b"], relevant, 3) == pytest.approx(_dcg(2, 3) / _dcg(1, 2, 3))

    def test_relevant_beyond_k_is_ignored(self) -> None:
        assert ndcg_at_k(["x", "y", "a"], {"a"}, 2) == 0.0
        assert precision_at_k(["x", "y", "a"], {"a"}, 2) == 0.0

    def test_k_equal_one_boundaries(self) -> None:
        assert precision_at_k(["a", "b"], {"a", "b"}, 1) == 1.0
        assert precision_at_k(["x", "a"], {"a"}, 1) == 0.0
        assert ndcg_at_k(["a", "b"], {"a", "b"}, 1) == pytest.approx(1.0)
        assert ndcg_at_k(["x", "a"], {"a"}, 1) == 0.0

    def test_invalid_k_rejected_by_ndcg(self) -> None:
        with pytest.raises(ValueError):
            ndcg_at_k(["a"], {"a"}, 0)

    def test_none_for_no_relevant_matches_hit_and_mrr(self) -> None:
        for relevant in ((), None, set()):
            empty = cast(Any, relevant or ())
            assert precision_at_k(["a"], empty, 3) is None
            assert ndcg_at_k(["a"], empty, 3) is None
            assert hit_at_k(["a"], empty, 3) is None
            assert mrr_at_k(["a"], empty, 3) is None

    def test_ranking_shorter_than_k_gets_partial_ndcg(self) -> None:
        assert ndcg_at_k(["a"], {"a", "b"}, 10) == pytest.approx(1 / _dcg(1, 2))

    def test_duplicate_paper_chunks_are_consistent_with_hit_and_mrr(self) -> None:
        ranked = ["x", "a", "a", "a", "y"]
        relevant = {"a"}
        assert hit_at_k(ranked, relevant, 5) == 1.0
        assert mrr_at_k(ranked, relevant, 5) == pytest.approx(0.5)
        assert ndcg_at_k(ranked, relevant, 5) == pytest.approx(1 / math.log2(3))
        assert precision_at_k(ranked, relevant, 5) == pytest.approx(0.2)

    def test_duplicates_crowding_out_other_relevant_lowers_ndcg(self) -> None:
        crowded = ndcg_at_k(["a", "a", "a", "b"], {"a", "b"}, 4)
        clean = ndcg_at_k(["a", "b", "x", "x"], {"a", "b"}, 4)
        assert crowded == pytest.approx(_dcg(1, 4) / _dcg(1, 2))
        assert crowded is not None and clean is not None and crowded < clean

    def test_ndcg_never_exceeds_one(self) -> None:
        value = ndcg_at_k(["a", "a", "b", "b"], {"a", "b"}, 4)
        assert value is not None and value <= 1.0 + 1e-12


def _query(query_id: str, relevant: tuple[str, ...], chunks: tuple[int, ...] = (), lang: str = "en") -> EvalQuery:
    return EvalQuery(
        id=query_id, query="q", lang=lang, relevant_arxiv_ids=relevant, source="manual", relevant_chunk_ids=chunks
    )


def _hits(*pairs: tuple[int, str]) -> list[RankedHit]:
    return [RankedHit(chunk_id=chunk_id, arxiv_id=arxiv_id, content_role="body") for chunk_id, arxiv_id in pairs]


class TestRunnerExtendedMetrics:
    def test_cutoffs_are_truncated_below_ten(self) -> None:
        assert extended_metric_names(5) == ["precision@1", "precision@5", "ndcg@5"]
        assert extended_metric_names(4) == ["precision@1", "ndcg@4"]
        assert extended_metric_names(1) == ["precision@1", "ndcg@1"]

    def test_cutoffs_above_ten_keep_hit_cutoffs_and_use_k_for_ndcg(self) -> None:
        assert extended_metric_names(20) == ["precision@1", "precision@5", "precision@10", "ndcg@20"]

    def test_all_metric_names_is_legacy_then_extended(self) -> None:
        for k in (1, 5, 10, 20):
            assert all_metric_names(k) == [*metric_names(k), *extended_metric_names(k)]

    def test_legacy_paper_and_chunk_key_sets_are_unchanged(self) -> None:
        result = score_hits(_query("q", ("A",), (1,)), "lexical", _hits((1, "A")), k=10, latency_ms=1.0)
        legacy = {"hit@1", "hit@5", "hit@10", "mrr@10", "recall@10"}
        assert set(result.paper_metrics) == legacy
        assert set(result.chunk_metrics) == legacy
        assert set(result.paper_extended) == {"precision@1", "precision@5", "precision@10", "ndcg@10"}
        assert set(result.chunk_extended) == set(result.paper_extended)

    def test_k_below_ten_scores_truncated_extended_keys(self) -> None:
        result = score_hits(_query("q", ("A",)), "lexical", _hits((1, "A")), k=3, latency_ms=1.0)
        assert set(result.paper_extended) == {"precision@1", "ndcg@3"}

    def test_chunk_extended_is_empty_without_relevant_chunks(self) -> None:
        result = score_hits(_query("q", ("A",)), "lexical", _hits((1, "A")), k=10, latency_ms=1.0)
        assert result.chunk_extended == {}
        assert result.chunk_metrics == {}

    def test_hits_beyond_k_do_not_count(self) -> None:
        result = score_hits(_query("q", ("B",)), "lexical", _hits((1, "A"), (2, "B")), k=1, latency_ms=1.0)
        assert result.paper_extended == {"precision@1": 0.0, "ndcg@1": 0.0}

    def test_aggregate_means_chunk_extended_over_chunk_queries_only(self) -> None:
        results = [
            score_hits(_query("c1", ("A",), (1,)), "vector", _hits((1, "A")), k=10, latency_ms=1.0),
            score_hits(_query("c2", ("A",), (9,)), "vector", _hits((1, "A")), k=10, latency_ms=1.0),
            score_hits(_query("n1", ("A",)), "vector", _hits((1, "A")), k=10, latency_ms=1.0),
        ]
        overall = next(row for row in aggregate(results) if row.subset == "all")
        assert overall.n_chunk_queries == 2
        assert overall.chunk_extended["precision@1"] == pytest.approx(0.5)
        assert overall.chunk_extended["ndcg@10"] == pytest.approx(0.5)
        assert overall.paper_extended["precision@1"] == pytest.approx(1.0)

    def test_aggregate_chunk_extended_is_none_without_chunk_queries(self) -> None:
        results = [score_hits(_query("n1", ("A",)), "vector", _hits((1, "A")), k=10, latency_ms=1.0)]
        overall = next(row for row in aggregate(results) if row.subset == "all")
        assert overall.chunk_extended["ndcg@10"] is None

    def test_aggregate_extended_means_per_language_subset(self) -> None:
        results = [
            score_hits(_query("q-ko", ("A",), lang="ko"), "vector", _hits((1, "A")), k=10, latency_ms=1.0),
            score_hits(_query("q-en", ("A",), lang="en"), "vector", _hits((1, "X")), k=10, latency_ms=1.0),
        ]
        by_subset = {row.subset: row for row in aggregate(results)}
        assert by_subset["ko"].paper_extended["precision@1"] == 1.0
        assert by_subset["en"].paper_extended["precision@1"] == 0.0
        assert by_subset["all"].paper_extended["precision@1"] == pytest.approx(0.5)


def _results(k: int = 10, *, with_chunks: bool = True) -> list:
    chunks = (1,) if with_chunks else ()
    return [score_hits(_query("q1", ("A",), chunks), "lexical", _hits((1, "A")), k=k, latency_ms=1.0)]


class TestReportExtendedColumns:
    def test_query_csv_extended_columns_follow_legacy_columns(self, tmp_path: Path) -> None:
        write_query_csv(_results(), tmp_path / "run.csv")
        with (tmp_path / "run.csv").open(encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        legacy = [f"{prefix}_{name}" for prefix in ("paper", "chunk") for name in metric_names(10)]
        extended = [f"{prefix}_{name}" for prefix in ("paper", "chunk") for name in extended_metric_names(10)]
        start = header.index(legacy[0])
        assert header[start : start + len(legacy)] == legacy
        assert header[start + len(legacy) : start + len(legacy) + len(extended)] == extended
        for column in ("paper_precision@1", "paper_precision@5", "paper_precision@10", "paper_ndcg@10"):
            assert column in header
        for column in ("chunk_precision@1", "chunk_precision@5", "chunk_precision@10", "chunk_ndcg@10"):
            assert column in header
        assert header.index("paper_ndcg@10") > header.index("chunk_recall@10")
        assert header.index("noise_count") > header.index("chunk_ndcg@10")

    def test_query_csv_legacy_columns_keep_their_original_prefix_of_header(self, tmp_path: Path) -> None:
        write_query_csv(_results(), tmp_path / "run.csv")
        with (tmp_path / "run.csv").open(encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        assert header[:9] == [
            "query_id",
            "lang",
            "source",
            "category",
            "expected_behavior",
            "method",
            "k",
            "latency_ms",
            "error",
        ]
        assert header[9 : 9 + 5] == ["paper_hit@1", "paper_hit@5", "paper_hit@10", "paper_mrr@10", "paper_recall@10"]

    def test_query_csv_cells_and_empty_chunk_cells(self, tmp_path: Path) -> None:
        write_query_csv(_results(with_chunks=False), tmp_path / "run.csv")
        with (tmp_path / "run.csv").open(encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        assert row["paper_precision@1"] == "1.0000"
        assert row["paper_precision@10"] == "0.1000"
        assert row["paper_ndcg@10"] == "1.0000"
        assert row["chunk_precision@1"] == ""
        assert row["chunk_ndcg@10"] == ""

    def test_query_csv_uses_truncated_extended_names_for_small_k(self, tmp_path: Path) -> None:
        write_query_csv(_results(k=3), tmp_path / "run.csv")
        with (tmp_path / "run.csv").open(encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        assert "paper_ndcg@3" in header
        assert "paper_precision@5" not in header

    def test_summary_csv_extended_columns_follow_each_legacy_block(self, tmp_path: Path) -> None:
        write_summary_csv(aggregate(_results()), tmp_path / "summary.csv")
        with (tmp_path / "summary.csv").open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            row = next(reader)
        assert header.index("paper_precision@1") == header.index("paper_recall@10") + 1
        assert header.index("n_chunk_queries") == header.index("paper_ndcg@10") + 1
        assert header.index("chunk_precision@1") == header.index("chunk_recall@10") + 1
        assert header.index("noise_rate") == header.index("chunk_ndcg@10") + 1
        assert row["paper_precision@10"] == "0.1000"
        assert row["chunk_ndcg@10"] == "1.0000"

    def test_markdown_has_precision_section_and_skips_chunk_table_without_chunks(self) -> None:
        text = render_report_markdown(aggregate(_results(with_chunks=False)), title="t", meta={})
        assert "## precision·nDCG (binary 관련성)" in text
        assert "| method | subset | n | precision@1 | precision@5 | precision@10 | nDCG@10 |" in text
        assert "| lexical | all | 1 | 1.000 | 0.200 | 0.100 | 1.000 |" in text
        assert "청크 단위 (relevant_chunk_ids가 있는 질의만)\n\n| method" not in text.split("## precision")[1]

    def test_markdown_chunk_precision_table_when_chunks_exist(self) -> None:
        text = render_report_markdown(aggregate(_results()), title="t", meta={})
        section = text.split("## precision·nDCG")[1]
        assert "청크 단위 (relevant_chunk_ids가 있는 질의만)" in section


QUERIES = [
    EvalQuery(id="p1-ko", query="q", lang="ko", relevant_arxiv_ids=("A",), source="known_item", category="cat1"),
    EvalQuery(id="p1-en", query="q", lang="en", relevant_arxiv_ids=("A",), source="known_item", category="cat1"),
    EvalQuery(id="p2-ko", query="q", lang="ko", relevant_arxiv_ids=("B",), source="llm_synth", category="cat2"),
    EvalQuery(id="p2-en", query="q", lang="en", relevant_arxiv_ids=("B",), source="llm_synth", category="cat2"),
    EvalQuery(id="p3-ko", query="q", lang="ko", relevant_arxiv_ids=("C",), source="manual"),
    EvalQuery(id="p3-en", query="q", lang="en", relevant_arxiv_ids=("C",), source="manual"),
    EvalQuery(id="cs-1", query="q", lang="en", relevant_arxiv_ids=("D",), source="llm_synth", relevant_chunk_ids=(40,)),
    EvalQuery(id="cs-2", query="q", lang="ko", relevant_arxiv_ids=("E",), source="manual", relevant_chunk_ids=(50,)),
]
CHUNK_OF = {"D": 40, "E": 50}


def _ranked(papers: list[str]) -> list[RankedHit]:
    return [
        RankedHit(chunk_id=CHUNK_OF.get(paper, 100 + index), arxiv_id=paper, content_role="body")
        for index, paper in enumerate(papers)
    ]


def _write(
    path: Path,
    rankings: dict[str, dict[str, list[str]]],
    *,
    queries: list[EvalQuery] | None = None,
    k: int = 10,
) -> Path:
    chosen = queries or QUERIES
    results = [
        score_hits(query, method, _ranked(rankings[method][query.id]), k=k, latency_ms=1.0)
        for method in rankings
        for query in chosen
        if query.id in rankings[method]
    ]
    write_query_csv(results, path)
    return path


GOOD = {query.id: [next(iter(query.relevant_arxiv_ids))] for query in QUERIES}
BAD = {query.id: ["X"] for query in QUERIES}
LEXICAL = {query.id: ["X", next(iter(query.relevant_arxiv_ids))] for query in QUERIES}


def _paired(a: Path, b: Path, queries: list[EvalQuery] | None = None, **kwargs: Any):
    return pair_runs(read_run_csv(a), read_run_csv(b), queries or QUERIES, label_a="large", label_b="small", **kwargs)


@pytest.fixture
def run_paths(tmp_path: Path) -> tuple[Path, Path]:
    a = _write(tmp_path / "a.csv", {"lexical": LEXICAL, "vector": BAD, "hybrid": GOOD})
    b = _write(tmp_path / "b.csv", {"lexical": LEXICAL, "vector": GOOD, "hybrid": BAD})
    return a, b


class TestCompareDeltas:
    def test_delta_sign_follows_b_minus_a(self, run_paths: tuple[Path, Path]) -> None:
        deltas = {
            (row.method, row.level, row.subset, row.metric): row
            for row in compare_runs(_paired(*run_paths), resamples=100, seed=1).deltas
        }
        improved = deltas[("vector", PAPER, "all", "mrr@10")]
        regressed = deltas[("hybrid", PAPER, "all", "mrr@10")]
        assert (improved.mean_a, improved.mean_b, improved.delta) == pytest.approx((0.0, 1.0, 1.0))
        assert (regressed.mean_a, regressed.mean_b, regressed.delta) == pytest.approx((1.0, 0.0, -1.0))
        assert improved.excludes_zero and regressed.excludes_zero
        flat = deltas[("lexical", PAPER, "all", "mrr@10")]
        assert flat.delta == 0.0 and not flat.excludes_zero

    def test_extended_metrics_are_compared_too(self, run_paths: tuple[Path, Path]) -> None:
        deltas = {
            (row.method, row.level, row.subset, row.metric): row
            for row in compare_runs(_paired(*run_paths), resamples=50, seed=1).deltas
        }
        assert deltas[("vector", PAPER, "all", "ndcg@10")].delta == pytest.approx(1.0)
        assert deltas[("vector", PAPER, "all", "precision@1")].delta == pytest.approx(1.0)

    def test_same_seed_gives_identical_intervals(self, tmp_path: Path) -> None:
        mixed = {query.id: ["A"] if index % 2 else ["X"] for index, query in enumerate(QUERIES)}
        a = _write(tmp_path / "a.csv", {"vector": BAD})
        b = _write(tmp_path / "b.csv", {"vector": mixed})
        first = compare_runs(_paired(a, b), resamples=500, seed=11).deltas
        second = compare_runs(_paired(a, b), resamples=500, seed=11).deltas
        assert [(row.low, row.high) for row in first] == [(row.low, row.high) for row in second]

    def test_resampling_draws_whole_translation_pairs(self, tmp_path: Path) -> None:
        a_rank = {query.id: ["X"] for query in QUERIES}
        b_rank = {query.id: ["X"] for query in QUERIES}
        b_rank["p1-en"] = ["A"]
        a = _write(tmp_path / "a.csv", {"vector": a_rank})
        b = _write(tmp_path / "b.csv", {"vector": b_rank})
        paired = _paired(a, b)
        pairs = [pair for pair in paired.pairs["vector"] if pair.query.id.startswith("p")]

        def only_first_cluster(n_clusters: int) -> list[list[int]]:
            return [[0]]

        row = metric_delta(pairs, method="vector", level=PAPER, subset="all", metric="hit@1", draws=only_first_cluster)
        assert row is not None
        assert cluster_key("p1-ko") == cluster_key("p1-en")
        assert row.n == 6 and row.n_clusters == 3
        assert (row.low, row.high) == pytest.approx((0.5, 0.5))

    def test_cluster_count_pairs_translations(self, run_paths: tuple[Path, Path]) -> None:
        deltas = compare_runs(_paired(*run_paths), resamples=20, seed=1).deltas
        row = next(r for r in deltas if (r.method, r.level, r.subset, r.metric) == ("vector", PAPER, "all", "hit@1"))
        assert (row.n, row.n_clusters) == (8, 5)
        ko = next(r for r in deltas if (r.method, r.level, r.subset, r.metric) == ("vector", PAPER, "ko", "hit@1"))
        assert (ko.n, ko.n_clusters) == (4, 4)

    def test_draws_cache_reuses_draws_per_cluster_count(self) -> None:
        draws = draws_cache(10, 5)
        assert draws(4) is draws(4)
        assert draws(4) is not draws(3)
        assert len(draws(4)) == 10


class TestCompareOutcomesAndSubsets:
    def test_outcome_counts_improved_tied_worse(self, tmp_path: Path) -> None:
        a_rank = {query.id: ["X"] for query in QUERIES}
        b_rank = dict(a_rank)
        b_rank["p1-ko"] = ["A"]
        b_rank["p1-en"] = ["A"]
        a_rank["p2-ko"] = ["B"]
        a = _write(tmp_path / "a.csv", {"vector": a_rank})
        b = _write(tmp_path / "b.csv", {"vector": b_rank})
        comparison = compare_runs(_paired(a, b), resamples=20)
        by_key = {(row.subset, row.metric): row for row in comparison.outcomes}
        overall = by_key[("all", "mrr@10")].counts
        assert overall == {"improved": 2, "tied": 5, "worse": 1}
        assert by_key[("ko", "mrr@10")].counts == {"improved": 1, "tied": 2, "worse": 1}
        assert by_key[("en", "mrr@10")].counts == {"improved": 1, "tied": 3, "worse": 0}

    def test_subset_rows_exist_for_lang_source_and_category(self, run_paths: tuple[Path, Path]) -> None:
        comparison = compare_runs(_paired(*run_paths), resamples=20)
        subsets = {row.subset for row in comparison.deltas if row.level == PAPER}
        assert {"all", "ko", "en", "known_item", "llm_synth", "manual", "category:cat1", "category:cat2"} <= subsets
        text = render_markdown(comparison, title="t", meta={})
        section = text.split("## 부분집합 (논문 단위)")[1].split("## 질의별")[0]
        for subset in ("all", "ko", "en", "known_item", "llm_synth", "manual", "category:cat1"):
            assert f"| vector | {subset} |" in section

    def test_chunk_rows_only_for_chunk_labelled_queries_and_lang_subsets(self, run_paths: tuple[Path, Path]) -> None:
        comparison = compare_runs(_paired(*run_paths), resamples=20)
        chunk_rows = [row for row in comparison.deltas if row.level == CHUNK]
        assert chunk_rows
        assert {row.subset for row in chunk_rows} <= {"all", "ko", "en"}
        overall = next(r for r in chunk_rows if (r.method, r.subset, r.metric) == ("vector", "all", "mrr@10"))
        assert overall.n == 2
        text = render_markdown(comparison, title="t", meta={})
        chunk_section = text.split("## 청크 단위")[1].split("## 부분집합")[0]
        assert "| 2 |" in chunk_section

    def test_chunk_section_says_none_without_chunk_labels(self, tmp_path: Path) -> None:
        plain = [query for query in QUERIES if not query.relevant_chunk_ids]
        a = _write(tmp_path / "a.csv", {"vector": BAD}, queries=plain)
        b = _write(tmp_path / "b.csv", {"vector": GOOD}, queries=plain)
        comparison = compare_runs(_paired(a, b, plain), resamples=20)
        assert not [row for row in comparison.deltas if row.level == CHUNK]
        assert "청크 정답이 있는 질의가 없습니다." in render_markdown(comparison, title="t", meta={})


class TestCompareWarnings:
    def test_lexical_difference_prints_warning_line(self, tmp_path: Path) -> None:
        changed = dict(LEXICAL, **{"p1-ko": ["A"]})
        a = _write(tmp_path / "a.csv", {"lexical": LEXICAL, "vector": BAD})
        b = _write(tmp_path / "b.csv", {"lexical": changed, "vector": GOOD})
        comparison = compare_runs(_paired(a, b), resamples=20)
        text = render_markdown(comparison, title="t", meta={})
        assert any(line.startswith("- 경고: lexical top-k가 1/8") and "p1-ko" in line for line in text.splitlines())

    def test_identical_lexical_prints_no_warning(self, run_paths: tuple[Path, Path]) -> None:
        comparison = compare_runs(_paired(*run_paths), resamples=20)
        text = render_markdown(comparison, title="t", meta={})
        assert comparison.identity_warnings == []
        assert "경고" not in text
        assert "lexical: 8/8 질의의 top-k가 large·small에서 같다." in text

    def test_no_lexical_method_skips_identity_check(self, tmp_path: Path) -> None:
        a = _write(tmp_path / "a.csv", {"vector": BAD})
        b = _write(tmp_path / "b.csv", {"vector": GOOD})
        text = render_markdown(compare_runs(_paired(a, b), resamples=20), title="t", meta={})
        assert "lexical 계열 방식이 없어 동일성 검사를 하지 않았습니다." in text

    def test_recorded_mrr_that_disagrees_with_queries_warns_for_that_side(self, run_paths: tuple[Path, Path]) -> None:
        relabelled = [
            *QUERIES[:6],
            EvalQuery(id="cs-1", query="q", lang="en", relevant_arxiv_ids=("Z",), source="llm_synth"),
            QUERIES[7],
        ]
        paired = _paired(*run_paths, relabelled)
        stale = [line for line in paired.warnings if "cs-1" in line]
        assert stale and any("large" in line for line in stale) and any("small" in line for line in stale)
        text = render_markdown(compare_runs(paired, resamples=20), title="t", meta={})
        assert "## 짝짓기 경고" in text

    def test_consistent_labels_produce_no_pairing_warnings(self, run_paths: tuple[Path, Path]) -> None:
        paired = _paired(*run_paths)
        assert paired.warnings == ()
        assert "## 짝짓기 경고" not in render_markdown(compare_runs(paired, resamples=20), title="t", meta={})

    def test_query_present_in_one_run_only_is_excluded_with_warning(self, tmp_path: Path) -> None:
        partial = {key: value for key, value in GOOD.items() if key != "p3-en"}
        a = _write(tmp_path / "a.csv", {"vector": BAD})
        b = _write(tmp_path / "b.csv", {"vector": partial})
        paired = _paired(a, b)
        assert "p3-en" not in [pair.query.id for pair in paired.pairs["vector"]]
        assert len(paired.pairs["vector"]) == 7
        assert any("한쪽 실행에만 있는 질의 1개" in line and "p3-en" in line for line in paired.warnings)

    def test_method_present_in_one_run_only_is_dropped_silently(self, tmp_path: Path) -> None:
        a = _write(tmp_path / "a.csv", {"vector": BAD, "hybrid": GOOD})
        b = _write(tmp_path / "b.csv", {"vector": GOOD})
        assert _paired(a, b).methods == ("vector",)

    def test_explicit_method_missing_from_one_run_is_an_error(self, tmp_path: Path) -> None:
        a = _write(tmp_path / "a.csv", {"vector": BAD, "hybrid": GOOD})
        b = _write(tmp_path / "b.csv", {"vector": GOOD})
        with pytest.raises(CompareError, match="방식"):
            _paired(a, b, methods=["hybrid"])

    def test_no_common_method_is_an_error(self, tmp_path: Path) -> None:
        a = _write(tmp_path / "a.csv", {"vector": BAD})
        b = _write(tmp_path / "b.csv", {"hybrid": GOOD})
        with pytest.raises(CompareError, match="공통 방식"):
            _paired(a, b)

    def test_error_rows_are_excluded_with_warning(self, tmp_path: Path) -> None:
        a = _write(tmp_path / "a.csv", {"vector": BAD})
        b = _write(tmp_path / "b.csv", {"vector": GOOD})
        with b.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        rows[0]["error"] = "RuntimeError: boom"
        with b.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        paired = _paired(a, b)
        assert len(paired.pairs["vector"]) == 7
        assert any("오류 행이 있는 질의 1개" in line and "p1-ko" in line for line in paired.warnings)

    def test_k_mismatch_is_an_error(self, tmp_path: Path) -> None:
        a = _write(tmp_path / "a.csv", {"vector": BAD}, k=10)
        b = _write(tmp_path / "b.csv", {"vector": GOOD}, k=5)
        with pytest.raises(CompareError, match="k가 다릅니다"):
            _paired(a, b)

    def test_duplicate_method_query_rows_are_an_error(self, tmp_path: Path) -> None:
        a = _write(tmp_path / "a.csv", {"vector": BAD})
        lines = a.read_text(encoding="utf-8").splitlines()
        a.write_text("\n".join([*lines, lines[1]]) + "\n", encoding="utf-8")
        with pytest.raises(CompareError, match="두 번"):
            read_run_csv(a)

    def test_empty_csv_is_an_error(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.csv"
        empty.write_text("query_id,method,k\n", encoding="utf-8")
        with pytest.raises(CompareError, match="행이 없습니다"):
            read_run_csv(empty)


class TestCompareOutputs:
    def test_queries_csv_ranks_and_chunk_ranks(self, run_paths: tuple[Path, Path], tmp_path: Path) -> None:
        paired = _paired(*run_paths)
        path = tmp_path / "queries.csv"
        write_queries_csv(paired, path)
        with path.open(encoding="utf-8") as handle:
            rows = {(row["method"], row["query_id"]): row for row in csv.DictReader(handle)}
        vector = rows[("vector", "p1-ko")]
        assert (vector["a_rank"], vector["b_rank"], vector["rank_change"]) == ("", "1", "improved")
        assert rows[("hybrid", "p1-ko")]["rank_change"] == "worse"
        assert rows[("lexical", "p1-ko")]["rank_change"] == "tied"
        assert rows[("vector", "cs-1")]["b_chunk_rank"] == "1"
        assert rows[("vector", "p1-ko")]["b_chunk_rank"] == ""
        assert float(vector["delta_hit@1"]) == 1.0

    def test_pair_values_use_the_requested_level(self) -> None:
        row = RunRow("q", "en", "manual", "", "answer", "vector", 10, "", {"hit@1": 1.0}, {"hit@1": 0.0}, (), ())
        pair = QueryPair("vector", QUERIES[0], row, row, 10)
        assert pair.values(PAPER, "hit@1") == (1.0, 1.0)
        assert pair.values(CHUNK, "hit@1") == (0.0, 0.0)
        assert pair.values(CHUNK, "missing") == (None, None)

    def test_cli_options_are_registered(self) -> None:
        result = runner.invoke(app, ["compare-runs", "--help"])
        assert result.exit_code == 0
        for option in (
            "--a",
            "--b",
            "--queries",
            "--label-a",
            "--label-b",
            "--methods",
            "--resamples",
            "--seed",
            "--confidence",
            "--out-dir",
        ):
            assert option in result.output

    def test_cli_rejects_bad_option_values(self, run_paths: tuple[Path, Path], tmp_path: Path) -> None:
        queries = tmp_path / "q.jsonl"
        queries.write_text(dump_queries(QUERIES), encoding="utf-8")
        base = ["compare-runs", "--a", str(run_paths[0]), "--b", str(run_paths[1]), "--queries", str(queries)]
        assert runner.invoke(app, [*base, "--resamples", "0"]).exit_code == 2
        assert runner.invoke(app, [*base, "--confidence", "0.1"]).exit_code == 2
        assert runner.invoke(app, [*base, "--methods", "nonexistent", "--resamples", "5"]).exit_code == 2
        assert runner.invoke(app, ["compare-runs", "--a", str(run_paths[0])]).exit_code == 2

    def test_cli_methods_filter_and_stderr_warnings(self, tmp_path: Path) -> None:
        changed = dict(LEXICAL, **{"p1-ko": ["A"]})
        a = _write(tmp_path / "a.csv", {"lexical": LEXICAL, "vector": BAD})
        b = _write(tmp_path / "b.csv", {"lexical": changed, "vector": GOOD})
        queries = tmp_path / "q.jsonl"
        queries.write_text(dump_queries(QUERIES), encoding="utf-8")
        out = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "compare-runs",
                "--a",
                str(a),
                "--b",
                str(b),
                "--queries",
                str(queries),
                "--resamples",
                "20",
                "--out-dir",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "경고: lexical top-k" in result.stderr
        only = tmp_path / "only"
        filtered = runner.invoke(
            app,
            [
                "compare-runs",
                "--a",
                str(a),
                "--b",
                str(b),
                "--queries",
                str(queries),
                "--methods",
                "vector",
                "--resamples",
                "20",
                "--out-dir",
                str(only),
            ],
        )
        assert filtered.exit_code == 0, filtered.output
        with (only / "compare_metrics.csv").open(encoding="utf-8") as handle:
            assert {row["method"] for row in csv.DictReader(handle)} == {"vector"}
        assert "경고" not in filtered.stderr


class FakeCursor:
    def __init__(self, connection: FakeConnection, name: str | None) -> None:
        self.connection = connection
        self.named = name is not None
        self.itersize = 0
        self.rows: list[tuple[Any, ...]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, query: bytes) -> None:
        sql = query.decode()
        self.connection.executed.append(sql)
        if "paper_embeddings GROUP BY" in sql.replace("\n", " "):
            self.rows = list(self.connection.embeddings)
        else:
            self.rows = [(text,) for text in self.connection.texts]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows

    def __iter__(self):
        return iter(self.rows)


class FakeConnection:
    def __init__(self, texts: list[str | None], embeddings: list[tuple[str, int]]) -> None:
        self.texts = texts
        self.embeddings = embeddings
        self.executed: list[str] = []
        self.read_only = False
        self.rolled_back = False

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def cursor(self, name: str | None = None) -> FakeCursor:
        return FakeCursor(self, name)

    def rollback(self) -> None:
        self.rolled_back = True


def _patch_connect(monkeypatch: pytest.MonkeyPatch, connection: FakeConnection) -> None:
    monkeypatch.setattr(embed_cost.psycopg, "connect", lambda dsn: connection)


class TestEmbedCost:
    def test_measure_cost_sums_tokens_with_injected_counter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connection = FakeConnection(["abc", "defgh", "  ij  "], [("text-embedding-3-large", 7)])
        _patch_connect(monkeypatch, connection)
        report = measure_cost("dsn", all_chunks=False, counter=len, token_method="len")
        assert report.summary == CostSummary(chunks=3, empty_chunks=0, characters=10, tokens=10, token_method="len")
        assert report.embeddings == {"text-embedding-3-large": 7}
        assert connection.read_only and connection.rolled_back

    def test_missing_and_all_selection_use_different_sql(self, monkeypatch: pytest.MonkeyPatch) -> None:
        missing_connection = FakeConnection(["a"], [])
        _patch_connect(monkeypatch, missing_connection)
        measure_cost("dsn", all_chunks=False, counter=len, token_method="len")
        all_connection = FakeConnection(["a"], [])
        _patch_connect(monkeypatch, all_connection)
        measure_cost("dsn", all_chunks=True, counter=len, token_method="len")
        missing_sql = next(sql for sql in missing_connection.executed if "chunk_text" in sql)
        all_sql = next(sql for sql in all_connection.executed if "chunk_text" in sql)
        assert "e.chunk_id IS NULL" in missing_sql
        assert "e.chunk_id IS NULL" not in all_sql
        assert missing_sql != all_sql

    def test_empty_result_is_zero_everywhere(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_connect(monkeypatch, FakeConnection([], []))
        report = measure_cost("dsn", all_chunks=True, counter=len, token_method="len")
        assert report.summary == CostSummary(chunks=0, empty_chunks=0, characters=0, tokens=0, token_method="len")
        assert report.summary.usd(0.13) == 0.0
        assert report.embeddings == {}

    def test_null_and_blank_texts_count_as_empty_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_connect(monkeypatch, FakeConnection([None, "   ", "abcd"], []))
        summary = measure_cost("dsn", all_chunks=False, counter=len, token_method="len").summary
        assert (summary.chunks, summary.empty_chunks, summary.tokens, summary.characters) == (3, 2, 4, 4)

    def test_zero_price_costs_nothing(self) -> None:
        summary = CostSummary(chunks=1, empty_chunks=0, characters=5, tokens=1_000_000, token_method="len")
        assert summary.usd(0.0) == 0.0
        assert summary.usd(0.02) == pytest.approx(0.02)
        assert summary.as_dict(0.0)["usd"] == 0.0

    def test_special_token_text_through_real_tiktoken_path(self) -> None:
        summary = embed_cost.summarize_texts(
            ["before <|endoftext|> after", "<|endoftext|>"],
            *token_counter(),
        )
        assert summary.tokens > 0 and summary.empty_chunks == 0

    def _run_cli(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *extra: str, price: str = "0.02") -> Any:
        report = CostReport(
            summary=CostSummary(chunks=3, empty_chunks=1, characters=40, tokens=500_000, token_method="len"),
            embeddings={"text-embedding-3-large": 9, "text-embedding-3-small": 2},
        )
        calls: list[bool] = []

        def fake_measure(dsn: str, *, all_chunks: bool, counter: Any, token_method: str) -> CostReport:
            calls.append(all_chunks)
            return report

        monkeypatch.setattr("arxplore_eval.commands.cost.measure_cost", fake_measure)
        result = runner.invoke(
            app,
            [
                "embed-cost",
                "--db",
                "postgresql://x/y",
                "--model",
                "text-embedding-3-small",
                "--price-per-million",
                price,
                *extra,
            ],
        )
        return result, calls

    def test_cli_default_is_missing_and_prints_cost_lines(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        result, calls = self._run_cli(monkeypatch, tmp_path)
        assert result.exit_code == 0, result.output
        assert calls == [False]
        assert "청크: 3개 (빈 텍스트 1개)" in result.stdout
        assert "토큰: 500000 (len)" in result.stdout
        assert "추정 비용: $0.0100 ($0.02/1M 토큰)" in result.stdout
        assert "text-embedding-3-large 9, text-embedding-3-small 2" in result.stdout
        assert "임베딩 없는 청크" in result.stdout

    def test_cli_all_flag_selects_all_chunks(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        result, calls = self._run_cli(monkeypatch, tmp_path, "--all")
        assert calls == [True]
        assert "references 제외 전체 청크" in result.stdout

    def test_cli_json_output_shape(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "cost.json"
        result, _ = self._run_cli(monkeypatch, tmp_path, "--all", "--json-out", str(target))
        assert result.exit_code == 0, result.output
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload == {
            "scope": "all",
            "model": "text-embedding-3-small",
            "chunks": 3,
            "empty_chunks": 1,
            "characters": 40,
            "tokens": 500_000,
            "token_method": "len",
            "price_per_million_usd": 0.02,
            "usd": pytest.approx(0.01),
            "paper_embeddings": {"text-embedding-3-large": 9, "text-embedding-3-small": 2},
        }

    def test_cli_zero_price_and_empty_embeddings(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        result, _ = self._run_cli(monkeypatch, tmp_path, price="0")
        assert result.exit_code == 0
        assert "추정 비용: $0.0000" in result.stdout

    def test_cli_rejects_negative_price(self) -> None:
        result = runner.invoke(
            app, ["embed-cost", "--db", "postgresql://x/y", "--model", "m", "--price-per-million", "-1"]
        )
        assert result.exit_code == 2


def _sample_usage_run(tokens: int) -> Any:
    embeddings = SimpleNamespace(calls=12, texts=12, tokens=tokens, token_method="tiktoken")
    return SimpleNamespace(embeddings=embeddings, timed_calls=10, timed_tokens=tokens - 20)


class TestLatencyUsageLines:
    def test_usage_reports_tokens_and_usd(self) -> None:
        usage = _usage(_sample_usage_run(1_000_020), 0.13)
        assert usage["합계"] == "12회 (12개 텍스트), 1000020 토큰 (tiktoken)"
        assert usage["측정(워밍업 포함) 호출"] == "10회, 1000000 토큰"
        assert usage["임베딩 고정 패스 호출"] == "2회, 20 토큰"
        assert usage["추정 비용"] == "$0.13000 ($0.13/1M 토큰)"

    def test_usage_zero_price(self) -> None:
        assert _usage(_sample_usage_run(500), 0.0)["추정 비용"] == "$0.00000 ($0.0/1M 토큰)"
