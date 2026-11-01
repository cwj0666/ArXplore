import csv
import math
from pathlib import Path

import pytest

from arxplore_eval.dataset import EvalQuery
from arxplore_eval.metrics import ndcg_at_k, precision_at_k
from arxplore_eval.report import render_markdown, write_query_csv, write_summary_csv
from arxplore_eval.runner import RankedHit, aggregate, all_metric_names, extended_metric_names, score_hits


class TestPrecisionAtK:
    def test_counts_distinct_relevant_over_k(self) -> None:
        assert precision_at_k(["a", "x", "b", "y"], {"a", "b"}, 4) == pytest.approx(0.5)

    def test_duplicates_do_not_inflate(self) -> None:
        assert precision_at_k(["a", "a", "x"], {"a"}, 3) == pytest.approx(1 / 3)

    def test_short_ranking_still_divides_by_k(self) -> None:
        assert precision_at_k(["a"], {"a"}, 5) == pytest.approx(0.2)

    def test_single_answer_is_hit_over_k(self) -> None:
        assert precision_at_k(["x", "a"], {"a"}, 1) == 0.0
        assert precision_at_k(["x", "a"], {"a"}, 2) == pytest.approx(0.5)

    def test_empty_relevant_is_undefined(self) -> None:
        assert precision_at_k(["a"], [], 5) is None

    def test_invalid_k(self) -> None:
        with pytest.raises(ValueError):
            precision_at_k(["a"], {"a"}, 0)


class TestNdcgAtK:
    def test_perfect_ranking_is_one(self) -> None:
        assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3) == pytest.approx(1.0)

    def test_single_answer_is_discount_of_first_rank(self) -> None:
        assert ndcg_at_k(["x", "y", "a"], {"a"}, 10) == pytest.approx(1 / math.log2(4))

    def test_duplicates_score_once(self) -> None:
        assert ndcg_at_k(["a", "a"], {"a", "b"}, 2) == pytest.approx(1 / (1 + 1 / math.log2(3)))

    def test_ideal_uses_min_of_relevant_and_k(self) -> None:
        relevant = {"a", "b", "c", "d"}
        assert ndcg_at_k(["a", "b"], relevant, 2) == pytest.approx(1.0)

    def test_miss_and_empty_ranking(self) -> None:
        assert ndcg_at_k(["x"], {"a"}, 5) == 0.0
        assert ndcg_at_k([], {"a"}, 5) == 0.0

    def test_empty_relevant_is_undefined(self) -> None:
        assert ndcg_at_k(["a"], set(), 5) is None


def _query(query_id: str, relevant: tuple[str, ...], chunks: tuple[int, ...] = ()) -> EvalQuery:
    return EvalQuery(
        id=query_id, query="q", lang="en", relevant_arxiv_ids=relevant, source="manual", relevant_chunk_ids=chunks
    )


def _hits(*pairs: tuple[int, str]) -> list[RankedHit]:
    return [RankedHit(chunk_id=chunk_id, arxiv_id=arxiv_id, content_role="body") for chunk_id, arxiv_id in pairs]


def test_extended_names_follow_hit_cutoffs() -> None:
    assert extended_metric_names(10) == ["precision@1", "precision@5", "precision@10", "ndcg@10"]
    assert extended_metric_names(3) == ["precision@1", "ndcg@3"]
    assert all_metric_names(3) == ["hit@1", "mrr@3", "recall@3", "precision@1", "ndcg@3"]


def test_score_hits_keeps_legacy_metrics_and_adds_extended_ones() -> None:
    result = score_hits(_query("q1", ("B",), (2,)), "vector", _hits((1, "A"), (2, "B"), (3, "B")), k=10, latency_ms=1.0)
    assert set(result.paper_metrics) == {"hit@1", "hit@5", "hit@10", "mrr@10", "recall@10"}
    assert result.paper_extended["precision@5"] == pytest.approx(0.2)
    assert result.paper_extended["ndcg@10"] == pytest.approx(1 / math.log2(3))
    assert result.chunk_extended["ndcg@10"] == pytest.approx(1 / math.log2(3))
    no_chunks = score_hits(_query("q2", ("A",)), "vector", _hits((1, "A")), k=10, latency_ms=1.0)
    assert no_chunks.chunk_extended == {}


def test_reports_carry_extended_metrics(tmp_path: Path) -> None:
    results = [
        score_hits(_query("q1", ("A",)), "lexical", _hits((1, "A")), k=10, latency_ms=1.0),
        score_hits(_query("q2", ("A",)), "lexical", _hits((1, "X"), (2, "A")), k=10, latency_ms=1.0),
    ]
    aggregates = aggregate(results)
    overall = next(row for row in aggregates if row.subset == "all")
    assert overall.paper_extended["precision@1"] == pytest.approx(0.5)
    assert overall.paper_extended["ndcg@10"] == pytest.approx((1 + 1 / math.log2(3)) / 2)

    write_query_csv(results, tmp_path / "run.csv")
    write_summary_csv(aggregates, tmp_path / "summary.csv")
    with (tmp_path / "run.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["paper_ndcg@10"] == f"{1 / math.log2(3):.4f}"
    assert rows[1]["chunk_ndcg@10"] == ""
    with (tmp_path / "summary.csv").open(encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert summary[0]["paper_precision@1"] == "0.5000"

    markdown = render_markdown(aggregates, title="t", meta={})
    assert "## precision·nDCG (binary 관련성)" in markdown
    assert "| method | subset | n | precision@1 | precision@5 | precision@10 | nDCG@10 |" in markdown
