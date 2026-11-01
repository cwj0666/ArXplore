import csv
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from arxplore_eval.agent_plan import AgentPlan, PlannedSearch, plan_payload
from arxplore_eval.answers import AgentTrace, GenerationResult, append_answer_record, build_answer_record
from arxplore_eval.cli import app
from arxplore_eval.dataset import EvalQuery, dump_queries
from arxplore_eval.fusion_variants import FUSION_CONVEX, FUSION_LEGACY_RULES, fuse_inputs
from arxplore_eval.plan_diff import (
    CHANGED,
    NO_SEARCH,
    REORDERED,
    SAME,
    STRATA,
    QueryDiff,
    SearchDiff,
    diff_plan,
    diff_plans,
    gold_rank,
    read_strata_csv,
    render_markdown,
    write_searches_csv,
    write_strata_csv,
)
from tests.unit.eval.plan_support import lexical_hit, sample_inputs

runner = CliRunner()


def _search(
    a: tuple[int, ...], b: tuple[int, ...], *, a_rank: int | None = None, b_rank: int | None = None
) -> SearchDiff:
    return SearchDiff(
        query_id="q",
        tool_call_id="call",
        tool_query="tq",
        a_chunks=a,
        b_chunks=b,
        a_papers=tuple(str(chunk) for chunk in a),
        b_papers=tuple(str(chunk) for chunk in b),
        a_gold_rank=a_rank,
        b_gold_rank=b_rank,
    )


def _query(query_id: str = "q-en", **kwargs: Any) -> EvalQuery:
    base: dict[str, Any] = {
        "query": "What is policy loss?",
        "lang": "en",
        "relevant_arxiv_ids": ("2401.00001",),
        "source": "manual",
    }
    base.update(kwargs)
    return EvalQuery(id=query_id, **base)


class TestSearchDiff:
    def test_same_when_orders_are_equal(self) -> None:
        diff = _search((1, 2, 3), (1, 2, 3))
        assert diff.relation == SAME
        assert diff.jaccard == 1.0

    def test_reordered_when_the_set_is_equal(self) -> None:
        diff = _search((1, 2, 3), (3, 1, 2))
        assert diff.relation == REORDERED
        assert diff.jaccard == 1.0

    def test_changed_when_the_sets_differ(self) -> None:
        diff = _search((1, 2, 3), (2, 3, 4))
        assert diff.relation == CHANGED
        assert diff.jaccard == pytest.approx(2 / 4)

    def test_disjoint_sets_have_zero_jaccard(self) -> None:
        assert _search((1, 2), (3, 4)).jaccard == 0.0

    def test_both_empty_is_same_with_jaccard_one(self) -> None:
        diff = _search((), ())
        assert diff.relation == SAME
        assert diff.jaccard == 1.0

    def test_one_side_empty(self) -> None:
        diff = _search((), (1, 2))
        assert diff.relation == CHANGED
        assert diff.jaccard == 0.0


class TestQueryDiffStratum:
    def test_no_searches_is_no_search(self) -> None:
        assert QueryDiff(query=_query(), kind="direct", searches=()).stratum == NO_SEARCH

    def test_single_relations(self) -> None:
        for searches, expected in (
            ((_search((1,), (1,)),), SAME),
            ((_search((1, 2), (2, 1)),), REORDERED),
            ((_search((1,), (2,)),), CHANGED),
        ):
            assert QueryDiff(query=_query(), kind="tools", searches=searches).stratum == expected

    def test_worst_relation_wins(self) -> None:
        same = _search((1,), (1,))
        reordered = _search((1, 2), (2, 1))
        changed = _search((1,), (2,))
        assert QueryDiff(_query(), "tools", (same, reordered)).stratum == REORDERED
        assert QueryDiff(_query(), "tools", (reordered, same)).stratum == REORDERED
        assert QueryDiff(_query(), "tools", (same, reordered, changed)).stratum == CHANGED
        assert QueryDiff(_query(), "tools", (changed, same)).stratum == CHANGED

    def test_gold_hit_per_side(self) -> None:
        diff = QueryDiff(_query(), "tools", (_search((1,), (1,), a_rank=None, b_rank=2),))
        assert diff.gold_hit("a") is False
        assert diff.gold_hit("b") is True

    def test_strata_constant(self) -> None:
        assert STRATA == (SAME, REORDERED, CHANGED, NO_SEARCH)


class TestGoldRank:
    def test_first_relevant_rank_is_one_based(self) -> None:
        hits = [lexical_hit(1, 0.9, "A"), lexical_hit(2, 0.8, "B"), lexical_hit(3, 0.7, "B")]
        assert gold_rank(hits, {"B"}) == 2
        assert gold_rank(hits, {"A", "B"}) == 1

    def test_none_when_absent_or_empty(self) -> None:
        assert gold_rank([lexical_hit(1, 0.9, "A")], {"Z"}) is None
        assert gold_rank([], {"A"}) is None
        assert gold_rank([lexical_hit(1, 0.9, "A")], set()) is None


def _plan(query_id: str = "q-en", *, queries: tuple[str, ...] = ("policy loss",)) -> AgentPlan:
    searches = tuple(
        PlannedSearch(tool_call_id=f"call_{index}", query=text, inputs=sample_inputs(text))
        for index, text in enumerate(queries)
    )
    calls = tuple(
        {"name": "search_paper_chunks_tool", "args": {"query": s.query}, "id": s.tool_call_id} for s in searches
    )
    return AgentPlan(query_id=query_id, content="", tool_calls=calls, searches=searches, search_limit=5)


class TestDiffPlan:
    def test_a_is_legacy_rules_and_b_is_convex_on_the_same_inputs(self) -> None:
        plan = _plan()
        diff = diff_plan(plan, _query())
        inputs = plan.searches[0].inputs
        legacy = fuse_inputs(inputs, FUSION_LEGACY_RULES, limit=5)
        convex = fuse_inputs(inputs, FUSION_CONVEX, limit=5)
        (search,) = diff.searches
        assert search.a_chunks == tuple(hit.chunk_id for hit in legacy)
        assert search.b_chunks == tuple(hit.chunk_id for hit in convex)
        assert search.tool_call_id == "call_0" and search.tool_query == "policy loss"
        assert search.a_papers == tuple(dict.fromkeys(hit.arxiv_id for hit in legacy))
        assert search.a_gold_rank == gold_rank(legacy, {"2401.00001"})
        assert search.b_gold_rank == gold_rank(convex, {"2401.00001"})
        assert diff.kind == "tools"

    def test_plan_without_searches_is_no_search(self) -> None:
        plan = AgentPlan(query_id="q-en", content="바로", tool_calls=(), searches=(), search_limit=5)
        diff = diff_plan(plan, _query())
        assert diff.stratum == NO_SEARCH and diff.kind == "direct"

    def test_diff_plans_follows_query_order_and_skips_unplanned_queries(self) -> None:
        queries = [_query("b-en"), _query("a-en"), _query("c-en")]
        plans = {"a-en": _plan("a-en"), "b-en": _plan("b-en")}
        assert [diff.query.id for diff in diff_plans(plans, queries)] == ["b-en", "a-en"]


def _diffs() -> list[QueryDiff]:
    ko = _query("p1-ko", lang="ko", query="질문")
    en = _query("p1-en", source="llm_synth")
    none = _query("p2-en", relevant_arxiv_ids=())
    return [
        QueryDiff(ko, "tools", (_search((1, 2), (2, 1), a_rank=1, b_rank=None),)),
        QueryDiff(en, "tools", (_search((1,), (2,), a_rank=None, b_rank=1), _search((1,), (1,)))),
        QueryDiff(none, "direct", ()),
    ]


class TestRenderMarkdown:
    def test_smoke(self) -> None:
        markdown = render_markdown(_diffs(), title="제목", meta={"커밋": "abc"})
        assert markdown.startswith("# 제목\n")
        assert "- 커밋: abc" in markdown
        for heading in ("## 질의 층", "## 검색 호출", "## 질의별"):
            assert heading in markdown
        assert "검색 호출 3개: same 1, reordered 1, changed 1" in markdown
        assert "A 1/2, B 1/2" in markdown
        for stratum in STRATA:
            assert f"`{stratum}`" in markdown
        for query_id in ("p1-ko", "p1-en", "p2-en"):
            assert query_id in markdown
        assert markdown.endswith("\n")

    def test_groups_cover_langs_and_sources(self) -> None:
        lines = render_markdown(_diffs(), title="t", meta={}).splitlines()
        labels = {line.split("|")[1].strip() for line in lines if line.startswith("| ") and "|" in line[2:]}
        assert {"all", "ko", "en", "llm_synth", "manual"} <= labels

    def test_without_any_search(self) -> None:
        diffs = [QueryDiff(_query(), "direct", ())]
        markdown = render_markdown(diffs, title="t", meta={})
        assert "- 검색 호출이 없습니다." in markdown
        assert "A 0/0, B 0/0" in markdown


class TestCsv:
    def test_strata_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "strata.csv"
        write_strata_csv(_diffs(), path)
        assert read_strata_csv(path) == {"p1-ko": REORDERED, "p1-en": CHANGED, "p2-en": NO_SEARCH}

    def test_read_skips_rows_missing_values(self, tmp_path: Path) -> None:
        path = tmp_path / "s.csv"
        path.write_text("id,stratum\na,same\n,same\nb,\n", encoding="utf-8")
        assert read_strata_csv(path) == {"a": "same"}

    @pytest.mark.parametrize("header", ["id,group", "stratum,x", "name"])
    def test_missing_columns_raise(self, tmp_path: Path, header: str) -> None:
        path = tmp_path / "s.csv"
        path.write_text(f"{header}\na,b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="id, stratum"):
            read_strata_csv(path)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "s.csv"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="id, stratum"):
            read_strata_csv(path)

    def test_searches_csv_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "searches.csv"
        write_searches_csv(_diffs(), path)
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [(row["id"], row["relation"], row["stratum"]) for row in rows] == [
            ("p1-ko", REORDERED, REORDERED),
            ("p1-en", CHANGED, CHANGED),
            ("p1-en", SAME, CHANGED),
        ]
        assert rows[0]["a_chunk_ids"] == "1 2" and rows[0]["b_chunk_ids"] == "2 1"
        assert rows[0]["jaccard"] == "1.0000" and rows[1]["jaccard"] == "0.0000"
        assert (rows[0]["a_gold_rank"], rows[0]["b_gold_rank"]) == ("1", "")
        assert (rows[1]["a_gold_rank"], rows[1]["b_gold_rank"]) == ("", "1")


def _write_plans(path: Path, queries: list[EvalQuery], plans: dict[str, AgentPlan]) -> None:
    for query in queries:
        plan = plans.get(query.id)
        if plan is None:
            continue
        trace = AgentTrace(fusion=None, origin="plan_only", plan=plan_payload(plan))
        record = build_answer_record(query, "agent", result=GenerationResult(answer="", contexts=[], agent=trace))
        append_answer_record(path, record)


class TestPlanDiffCommand:
    def _setup(self, tmp_path: Path) -> tuple[Path, Path, list[EvalQuery]]:
        queries = [_query("p1-en"), _query("p1-ko", lang="ko", query="정책 손실이란?"), _query("p2-en", query="Other?")]
        direct = AgentPlan(query_id="p2-en", content="바로", tool_calls=(), searches=(), search_limit=5)
        plans = {"p1-en": _plan("p1-en"), "p1-ko": _plan("p1-ko", queries=("policy loss", "dpo")), "p2-en": direct}
        queries_path = tmp_path / "queries.jsonl"
        queries_path.write_text(dump_queries(queries), encoding="utf-8")
        plan_path = tmp_path / "plans.jsonl"
        _write_plans(plan_path, queries, plans)
        return queries_path, plan_path, queries

    def test_writes_the_three_files(self, tmp_path: Path) -> None:
        queries_path, plan_path, queries = self._setup(tmp_path)
        out = tmp_path / "out"
        result = runner.invoke(
            app, ["plan-diff", "--plan", str(plan_path), "--queries", str(queries_path), "--out-dir", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert sorted(path.name for path in out.iterdir()) == ["plan_diff.md", "plan_diff_searches.csv", "strata.csv"]
        markdown = (out / "plan_diff.md").read_text(encoding="utf-8")
        assert "legacy_rules vs convex" in markdown
        assert "sha256:" in markdown and "(3건)" in markdown
        strata = read_strata_csv(out / "strata.csv")
        assert set(strata) == {"p1-en", "p1-ko", "p2-en"}
        assert strata["p2-en"] == NO_SEARCH
        assert strata["p1-en"] in (SAME, REORDERED, CHANGED)
        with (out / "plan_diff_searches.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["id"] for row in rows] == ["p1-en", "p1-ko", "p1-ko"]
        assert [row["tool_query"] for row in rows] == ["policy loss", "policy loss", "dpo"]
        assert queries and "plan_diff.md" in result.stderr

    def test_unknown_plan_id_exits_2(self, tmp_path: Path) -> None:
        queries_path, plan_path, queries = self._setup(tmp_path)
        queries_path.write_text(dump_queries(queries[:2]), encoding="utf-8")
        out = tmp_path / "out"
        result = runner.invoke(
            app, ["plan-diff", "--plan", str(plan_path), "--queries", str(queries_path), "--out-dir", str(out)]
        )
        assert result.exit_code == 2
        assert "질의셋에 없는 계획 id" in result.stderr and "p2-en" in result.stderr
        assert not out.exists()

    def test_changed_question_exits_2(self, tmp_path: Path) -> None:
        queries_path, plan_path, queries = self._setup(tmp_path)
        edited = [*queries[:2], _query("p2-en", query="A different question?")]
        queries_path.write_text(dump_queries(edited), encoding="utf-8")
        result = runner.invoke(
            app,
            ["plan-diff", "--plan", str(plan_path), "--queries", str(queries_path), "--out-dir", str(tmp_path / "o")],
        )
        assert result.exit_code == 2
        assert "질문이 다른 계획 id" in result.stderr and "p2-en" in result.stderr

    def test_missing_plan_file_exits_2(self, tmp_path: Path) -> None:
        queries_path, _, _ = self._setup(tmp_path)
        result = runner.invoke(
            app, ["plan-diff", "--plan", str(tmp_path / "none.jsonl"), "--queries", str(queries_path)]
        )
        assert result.exit_code == 2

    def test_a_non_plan_record_exits_2(self, tmp_path: Path) -> None:
        queries_path, _, queries = self._setup(tmp_path)
        other = tmp_path / "answers.jsonl"
        append_answer_record(
            other, build_answer_record(queries[0], "agent", result=GenerationResult(answer="a", contexts=[]))
        )
        result = runner.invoke(app, ["plan-diff", "--plan", str(other), "--queries", str(queries_path)])
        assert result.exit_code == 2
        assert "입력 형식 오류" in result.stderr
