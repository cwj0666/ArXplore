import csv
from pathlib import Path

import pytest
from typer.testing import CliRunner

from arxplore_eval.cli import app
from arxplore_eval.compare_runs import (
    BOTH,
    CHUNK,
    IMPROVED,
    LEXICAL_ONLY,
    NEITHER,
    PAPER,
    TIED,
    VECTOR_ONLY,
    WORSE,
    CompareError,
    compare_runs,
    pair_runs,
    read_run_csv,
    render_markdown,
)
from arxplore_eval.dataset import EvalQuery, dump_queries
from arxplore_eval.report import write_query_csv
from arxplore_eval.runner import RankedHit, score_hits

runner = CliRunner()
QUERIES = [
    EvalQuery(id="p1-ko", query="q", lang="ko", relevant_arxiv_ids=("A",), source="known_item"),
    EvalQuery(id="p1-en", query="q", lang="en", relevant_arxiv_ids=("A",), source="known_item"),
    EvalQuery(id="p2-ko", query="q", lang="ko", relevant_arxiv_ids=("B",), source="known_item"),
    EvalQuery(id="p2-en", query="q", lang="en", relevant_arxiv_ids=("B",), source="known_item"),
    EvalQuery(id="cs-1", query="q", lang="en", relevant_arxiv_ids=("C",), source="llm_synth", relevant_chunk_ids=(30,)),
]
LEXICAL = {"p1-ko": ["X", "A"], "p1-en": ["A"], "p2-ko": ["X"], "p2-en": ["B"], "cs-1": ["C"]}
VECTOR_A = {"p1-ko": ["A"], "p1-en": ["X", "X", "A"], "p2-ko": ["X"], "p2-en": ["X"], "cs-1": ["X", "C"]}
VECTOR_B = {"p1-ko": ["A"], "p1-en": ["A"], "p2-ko": ["B"], "p2-en": ["X"], "cs-1": ["C"]}


def _hits(papers: list[str]) -> list[RankedHit]:
    return [
        RankedHit(chunk_id=30 if paper == "C" else 100 + index, arxiv_id=paper, content_role="body")
        for index, paper in enumerate(papers)
    ]


def _write_run(path: Path, rankings: dict[str, dict[str, list[str]]]) -> Path:
    results = [
        score_hits(query, method, _hits(rankings[method][query.id]), k=10, latency_ms=1.0)
        for method in rankings
        for query in QUERIES
    ]
    write_query_csv(results, path)
    return path


@pytest.fixture
def runs(tmp_path: Path) -> tuple[Path, Path, Path]:
    a = _write_run(tmp_path / "a.csv", {"lexical": LEXICAL, "vector": VECTOR_A})
    b = _write_run(tmp_path / "b.csv", {"lexical": LEXICAL, "vector": VECTOR_B})
    queries = tmp_path / "queries.jsonl"
    queries.write_text(dump_queries(QUERIES), encoding="utf-8")
    return a, b, queries


def _paired(a: Path, b: Path):
    return pair_runs(read_run_csv(a), read_run_csv(b), QUERIES, label_a="large", label_b="small")


def test_read_run_csv_picks_up_every_metric_column(runs) -> None:
    table = read_run_csv(runs[0])
    assert table.k == 10
    assert table.methods == ["lexical", "vector"]
    assert "ndcg@10" in table.metrics and "hit@1" in table.metrics
    assert table.rows[("vector", "cs-1")].retrieved_chunk_ids == (100, 30)


def test_deltas_are_b_minus_a_with_cluster_bootstrap(runs) -> None:
    comparison = compare_runs(_paired(*runs[:2]), resamples=300, seed=7)
    by_key = {(row.method, row.level, row.subset, row.metric): row for row in comparison.deltas}

    hit1 = by_key[("vector", PAPER, "all", "hit@1")]
    assert (hit1.n, hit1.n_clusters) == (5, 3)
    assert (hit1.mean_a, hit1.mean_b) == pytest.approx((1 / 5, 4 / 5))
    assert hit1.delta == pytest.approx(3 / 5)
    assert hit1.low <= hit1.delta <= hit1.high

    lexical = by_key[("lexical", PAPER, "all", "mrr@10")]
    assert (lexical.delta, lexical.low, lexical.high) == (0.0, 0.0, 0.0)

    chunk = by_key[("vector", CHUNK, "all", "mrr@10")]
    assert (chunk.n, chunk.mean_a, chunk.mean_b) == (1, 0.5, 1.0)
    assert ("vector", CHUNK, "category:x", "mrr@10") not in by_key
    assert ("vector", PAPER, "known_item", "hit@1") in by_key


def test_bootstrap_is_reproducible_for_a_seed(runs) -> None:
    first = compare_runs(_paired(*runs[:2]), resamples=200, seed=3).deltas
    again = compare_runs(_paired(*runs[:2]), resamples=200, seed=3).deltas
    assert first == again


def test_outcomes_complement_and_identity(runs) -> None:
    comparison = compare_runs(_paired(*runs[:2]), resamples=50)
    outcomes = {(row.method, row.subset, row.metric): row for row in comparison.outcomes}
    assert outcomes[("vector", "all", "mrr@10")].counts == {IMPROVED: 3, TIED: 2, WORSE: 0}
    assert outcomes[("vector", "all", "mrr@10")].query_ids[IMPROVED] == ("p1-en", "p2-ko", "cs-1")
    assert outcomes[("vector", "all", "hit@5")].counts == {IMPROVED: 1, TIED: 4, WORSE: 0}

    assert comparison.complement["large"] == {BOTH: 3, LEXICAL_ONLY: 1, VECTOR_ONLY: 0, NEITHER: 1}
    assert comparison.complement["small"] == {BOTH: 3, LEXICAL_ONLY: 1, VECTOR_ONLY: 1, NEITHER: 0}
    assert all(check.identical for check in comparison.identity)
    assert comparison.identity_warnings == []

    markdown = render_markdown(comparison, title="t", meta={"k": "10"})
    assert "lexical: 5/5 질의의 top-k가 large·small에서 같다." in markdown
    assert "| vector | hit@1 | 5 | 3 | 0.200 | 0.800 |" in markdown
    assert "| large | 3 | 1 | 0 | 1 |" in markdown


def test_lexical_mismatch_and_stale_labels_warn(tmp_path: Path, runs) -> None:
    changed = dict(LEXICAL, **{"p2-ko": ["B"]})
    b = _write_run(tmp_path / "b2.csv", {"lexical": changed, "vector": VECTOR_B})
    comparison = compare_runs(_paired(runs[0], b), resamples=50)
    assert comparison.identity_warnings and "p2-ko" in comparison.identity_warnings[0]

    relabelled = [
        *QUERIES[:4],
        EvalQuery(id="cs-1", query="q", lang="en", relevant_arxiv_ids=("X",), source="llm_synth"),
    ]
    paired = pair_runs(read_run_csv(runs[0]), read_run_csv(runs[1]), relabelled, label_a="large", label_b="small")
    assert any("cs-1" in line and "large" in line for line in paired.warnings)


def test_pairing_errors(tmp_path: Path, runs) -> None:
    with pytest.raises(CompareError, match="질의셋에 없는"):
        pair_runs(read_run_csv(runs[0]), read_run_csv(runs[1]), QUERIES[:2], label_a="a", label_b="b")
    with pytest.raises(CompareError, match="방식"):
        pair_runs(read_run_csv(runs[0]), read_run_csv(runs[1]), QUERIES, label_a="a", label_b="b", methods=["hybrid"])
    only_lexical = _write_run(tmp_path / "lex.csv", {"lexical": LEXICAL})
    paired = pair_runs(read_run_csv(runs[0]), read_run_csv(only_lexical), QUERIES, label_a="a", label_b="b")
    assert paired.methods == ("lexical",)


def test_cli_writes_three_files(tmp_path: Path, runs) -> None:
    a, b, queries = runs
    out = tmp_path / "compare"
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
            "--label-a",
            "large",
            "--label-b",
            "small",
            "--resamples",
            "50",
            "--out-dir",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert {path.name for path in out.iterdir()} == {"compare.md", "compare_metrics.csv", "compare_queries.csv"}
    with (out / "compare_queries.csv").open(encoding="utf-8") as handle:
        rows = {(row["method"], row["query_id"]): row for row in csv.DictReader(handle)}
    assert (rows[("vector", "p1-en")]["a_rank"], rows[("vector", "p1-en")]["b_rank"]) == ("3", "1")
    assert rows[("vector", "p2-en")]["rank_change"] == TIED
    assert rows[("vector", "cs-1")]["b_chunk_rank"] == "1"

    missing = runner.invoke(
        app, ["compare-runs", "--a", str(tmp_path / "none.csv"), "--b", str(b), "--queries", str(queries)]
    )
    assert missing.exit_code == 2
