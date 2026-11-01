import csv
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from arxplore_eval.answers import GenerationResult, append_answer_record, build_answer_record
from arxplore_eval.cli import app
from arxplore_eval.compare_generation import (
    CSV_METRICS,
    GENERATION_METRICS,
    STRATUM_PREFIX,
    CompareGenerationError,
    RunValues,
    change_counts,
    compare_generation,
    pair_generation,
    read_answers_jsonl,
    read_run,
    read_sample_csv,
    render_markdown,
)
from arxplore_eval.dataset import EvalQuery, dump_queries

runner = CliRunner()
IDS = ["a1-ko", "a1-en", "a2-ko", "a2-en", "a3-en", "a4-en"]
STRATA = {
    "a1-ko": "same",
    "a1-en": "same",
    "a2-ko": "changed",
    "a2-en": "changed",
    "a3-en": "reordered",
    "a4-en": "no_search",
}
QUERIES = [
    EvalQuery(
        id=query_id,
        query=f"question {query_id}",
        lang=query_id.rsplit("-", 1)[1],
        relevant_arxiv_ids=("2401.00001",),
        source="manual",
    )
    for query_id in IDS
]


def _csv(path: Path, values: dict[str, dict[str, str]], *, mode: str = "agent") -> Path:
    fields = ["id", "mode", "lang", "outcome", "faithfulness", "answer_relevancy", "lang_match", "cited_relevant"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        for query_id, row in values.items():
            writer.writerow({"id": query_id, "mode": mode, "outcome": "answered", **row})
    return path


def _cited(a: list[str], b: list[str], tmp_path: Path) -> tuple[Path, Path]:
    left = _csv(
        tmp_path / "a.csv", {query_id: {"cited_relevant": value} for query_id, value in zip(IDS, a, strict=True)}
    )
    right = _csv(
        tmp_path / "b.csv", {query_id: {"cited_relevant": value} for query_id, value in zip(IDS, b, strict=True)}
    )
    return left, right


def _paired(a: Path, b: Path, strata: dict[str, str] | None = None) -> Any:
    return pair_generation(
        [read_run(a, mode="agent")], [read_run(b, mode="agent")], QUERIES, label_a="A", label_b="B", strata=strata
    )


class TestCitedRelevantMetric:
    def test_metric_lists_include_cited_relevant(self) -> None:
        assert "cited_relevant" in GENERATION_METRICS
        assert "cited_relevant" in CSV_METRICS

    def test_read_from_the_sample_csv(self, tmp_path: Path) -> None:
        left, _ = _cited(["1", "0", "1", "", "0", "1"], ["1"] * 6, tmp_path)
        run = read_sample_csv(left, mode="agent")
        assert [run.values[query_id]["cited_relevant"] for query_id in IDS] == [1.0, 0.0, 1.0, None, 0.0, 1.0]
        assert run.mean("cited_relevant") == pytest.approx(3 / 5)

    def test_csv_without_the_column_gives_none(self, tmp_path: Path) -> None:
        path = tmp_path / "old.csv"
        path.write_text("id,mode,outcome,faithfulness\na1-en,agent,answered,0.5\n", encoding="utf-8")
        run = read_sample_csv(path, mode="agent")
        assert run.values["a1-en"]["cited_relevant"] is None
        assert run.values["a1-en"]["faithfulness"] == 0.5

    def _agent_record(self, query: EvalQuery, answer: str, citations: list[dict[str, Any]]) -> dict[str, Any]:
        return build_answer_record(
            query, "agent", result=GenerationResult(answer=answer, contexts=["c"], citations=citations)
        )

    def test_read_from_answers_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "answers.jsonl"
        hit = {"arxiv_id": "2401.00001", "title": "t", "url": "u", "in_answer": True}
        miss = {"arxiv_id": "2401.99999", "title": "t", "url": "u", "in_answer": True}
        append_answer_record(path, self._agent_record(QUERIES[1], "This paper studies policy loss.", [hit]))
        append_answer_record(path, self._agent_record(QUERIES[3], "This paper studies something.", [miss]))
        append_answer_record(
            path, self._agent_record(QUERIES[4], "See https://arxiv.org/abs/2401.00001v2 for details.", [])
        )
        append_answer_record(
            path,
            build_answer_record(
                EvalQuery(id="n-en", query="q", lang="en", relevant_arxiv_ids=(), source="manual"),
                "agent",
                result=GenerationResult(answer="An answer without relevant ids.", contexts=[]),
            ),
        )
        run = read_answers_jsonl(path, mode="agent")
        assert run.values["a1-en"]["cited_relevant"] == 1.0
        assert run.values["a2-en"]["cited_relevant"] == 0.0
        assert run.values["a3-en"]["cited_relevant"] == 1.0
        assert run.values["n-en"]["cited_relevant"] is None

    def test_paper_chat_answers_have_no_cited_relevant(self, tmp_path: Path) -> None:
        path = tmp_path / "answers.jsonl"
        record = build_answer_record(
            QUERIES[1], "paper_chat", result=GenerationResult(answer="The rate was 3e-4 [1].", contexts=["c"])
        )
        append_answer_record(path, record)
        assert read_answers_jsonl(path).values["a1-en"]["cited_relevant"] is None

    def test_it_is_compared_like_any_metric(self, tmp_path: Path) -> None:
        left, right = _cited(["0"] * 6, ["1", "1", "1", "1", "0", "0"], tmp_path)
        paired = _paired(left, right)
        assert "cited_relevant" in paired.metrics
        comparison = compare_generation(paired, resamples=100, seed=3)
        overall = {row.metric: row.result for row in comparison.deltas if row.subset == "all"}["cited_relevant"]
        assert overall.delta == pytest.approx(4 / 6)
        assert overall.n == 6

    def test_markdown_mentions_it(self, tmp_path: Path) -> None:
        left, right = _cited(["0"] * 6, ["1"] * 6, tmp_path)
        markdown = render_markdown(compare_generation(_paired(left, right), resamples=50, seed=1), title="t", meta={})
        assert "cited_relevant" in markdown


class TestStrata:
    def _pair(self, tmp_path: Path, strata: dict[str, str] | None = STRATA) -> Any:
        left, right = _cited(["0", "0", "1", "0", "0", "1"], ["1", "1", "0", "0", "1", "1"], tmp_path)
        return _paired(left, right, strata)

    def test_subsets_add_one_entry_per_stratum(self, tmp_path: Path) -> None:
        subsets = self._pair(tmp_path).subsets()
        added = [name for name in subsets if name.startswith(STRATUM_PREFIX)]
        assert added == ["stratum:changed", "stratum:no_search", "stratum:reordered", "stratum:same"]
        assert subsets[: len(subsets) - len(added)] == self._pair(tmp_path, None).subsets()

    def test_no_strata_means_no_stratum_subsets(self, tmp_path: Path) -> None:
        assert not [name for name in self._pair(tmp_path, None).subsets() if name.startswith(STRATUM_PREFIX)]

    def test_strata_for_queries_outside_the_pairs_are_ignored(self, tmp_path: Path) -> None:
        strata = {**STRATA, "ghost-en": "weird"}
        assert "stratum:weird" not in self._pair(tmp_path, strata).subsets()

    def test_select_filters_by_stratum(self, tmp_path: Path) -> None:
        paired = self._pair(tmp_path)
        assert [pair.query.id for pair in paired.select("stratum:same")] == ["a1-ko", "a1-en"]
        assert [pair.query.id for pair in paired.select("stratum:changed")] == ["a2-ko", "a2-en"]
        assert [pair.query.id for pair in paired.select("stratum:no_search")] == ["a4-en"]
        assert paired.select("stratum:missing") == []

    def test_select_still_handles_ordinary_subsets(self, tmp_path: Path) -> None:
        paired = self._pair(tmp_path)
        assert len(paired.select("all")) == 6
        assert [pair.query.id for pair in paired.select("ko")] == ["a1-ko", "a2-ko"]

    def test_unstratified_queries_fall_out_of_every_stratum(self, tmp_path: Path) -> None:
        paired = self._pair(tmp_path, {"a1-en": "same"})
        assert [pair.query.id for pair in paired.select("stratum:same")] == ["a1-en"]

    def test_deltas_include_the_strata_subsets(self, tmp_path: Path) -> None:
        comparison = compare_generation(self._pair(tmp_path), resamples=100, seed=5)
        rows = {(row.subset, row.metric): row.result for row in comparison.deltas}
        assert rows[("stratum:same", "cited_relevant")].delta == pytest.approx(1.0)
        assert rows[("stratum:same", "cited_relevant")].n == 2
        assert rows[("stratum:changed", "cited_relevant")].delta == pytest.approx(-0.5)
        assert rows[("stratum:no_search", "cited_relevant")].n == 1
        assert rows[("stratum:reordered", "cited_relevant")].delta == pytest.approx(1.0)

    def test_deltas_without_strata_have_no_stratum_rows(self, tmp_path: Path) -> None:
        comparison = compare_generation(self._pair(tmp_path, None), resamples=50, seed=5)
        assert not [row for row in comparison.deltas if row.subset.startswith(STRATUM_PREFIX)]

    def test_change_counts_include_the_strata_subsets(self, tmp_path: Path) -> None:
        rows = {(row.subset, row.metric): row.counts for row in change_counts(self._pair(tmp_path))}
        assert sum(rows[("stratum:same", "cited_relevant")].values()) == 2
        assert sum(rows[("stratum:changed", "cited_relevant")].values()) == 2
        assert sum(rows[("all", "cited_relevant")].values()) == 6

    def test_change_counts_without_strata_only_have_outcome_subsets(self, tmp_path: Path) -> None:
        subsets = {row.subset for row in change_counts(self._pair(tmp_path, None))}
        assert subsets <= {"all", "ko", "en"}

    def test_markdown_shows_the_strata(self, tmp_path: Path) -> None:
        comparison = compare_generation(self._pair(tmp_path), resamples=50, seed=5)
        markdown = render_markdown(comparison, title="t", meta={})
        assert "stratum:same" in markdown and "stratum:changed" in markdown


class TestCompareCommandStrata:
    def _argv(self, tmp_path: Path, *extra: str) -> list[str]:
        left, right = _cited(["0", "0", "1", "0", "0", "1"], ["1", "1", "0", "0", "1", "1"], tmp_path)
        queries = tmp_path / "queries.jsonl"
        queries.write_text(dump_queries(QUERIES), encoding="utf-8")
        return [
            "compare-generation",
            "--queries",
            str(queries),
            "--a",
            str(left),
            "--b",
            str(right),
            "--mode",
            "agent",
            "--resamples",
            "50",
            *extra,
        ]

    def _strata_file(self, tmp_path: Path) -> Path:
        path = tmp_path / "strata.csv"
        lines = ["id,stratum", *(f"{query_id},{name}" for query_id, name in STRATA.items())]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_strata_adds_a_meta_line_and_subset_rows(self, tmp_path: Path) -> None:
        strata = self._strata_file(tmp_path)
        out = tmp_path / "out"
        result = runner.invoke(app, self._argv(tmp_path, "--strata", str(strata), "--out-dir", str(out)))
        assert result.exit_code == 0, result.output
        markdown = (out / "compare_generation.md").read_text(encoding="utf-8")
        assert f"| 질의 층 | `{strata}` sha256:" in markdown
        assert "stratum:same" in markdown
        assert "stratum:same" in (out / "compare_generation_metrics.csv").read_text(encoding="utf-8")

    def test_without_strata_there_is_no_meta_line(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        result = runner.invoke(app, self._argv(tmp_path, "--out-dir", str(out)))
        assert result.exit_code == 0, result.output
        markdown = (out / "compare_generation.md").read_text(encoding="utf-8")
        assert "| 질의 층 |" not in markdown and "stratum:" not in markdown

    def test_strata_csv_missing_columns_exits_2(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        bad.write_text("id,group\na1-en,same\n", encoding="utf-8")
        result = runner.invoke(app, self._argv(tmp_path, "--strata", str(bad), "--out-dir", str(tmp_path / "o")))
        assert result.exit_code == 2
        assert "id, stratum" in result.stderr

    def test_missing_strata_file_exits_2(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, self._argv(tmp_path, "--strata", str(tmp_path / "none.csv"), "--out-dir", str(tmp_path / "o"))
        )
        assert result.exit_code == 2


def test_run_values_type_is_exported() -> None:
    assert RunValues and CompareGenerationError
