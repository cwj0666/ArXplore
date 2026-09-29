from typing import Any

import pytest
from typer.testing import CliRunner

from arxplore.retrieval.lexical import build_lexical_candidates_query
from arxplore.retrieval.query import candidate_fetch_limit, hybrid_branch_limit
from arxplore.retrieval.service import RetrievalConfig
from arxplore.retrieval.vector import build_vector_search_query, resolve_hnsw_ef_search
from arxplore_eval.cli import app
from arxplore_eval.explain import (
    HNSW_EF_SEARCH_SQL,
    LEXICAL,
    SEQ_SCAN_NOTE,
    VECTOR,
    channel_limit,
    lexical_target,
    render_plan,
    report_lines,
    sql_fetch_limit,
    summarize_plan,
    vector_target,
)

EMBEDDING = [0.1] * 8
PLAN: list[dict[str, Any]] = [
    {
        "Plan": {
            "Node Type": "Limit",
            "Actual Rows": 5,
            "Actual Loops": 1,
            "Actual Total Time": 3.5,
            "Shared Hit Blocks": 40,
            "Shared Read Blocks": 2,
            "Plans": [
                {
                    "Node Type": "Bitmap Heap Scan",
                    "Relation Name": "paper_chunks",
                    "Actual Rows": 9,
                    "Actual Loops": 1,
                    "Actual Total Time": 1.0,
                    "Plans": [{"Node Type": "Bitmap Index Scan", "Index Name": "idx_paper_chunks_chunk_vector"}],
                },
                {"Node Type": "Seq Scan", "Relation Name": "papers", "Actual Rows": 3, "Actual Loops": 1},
                {"Node Type": "CTE Scan", "CTE Name": "query_terms", "Actual Rows": 1, "Actual Loops": 1},
            ],
        },
        "Planning Time": 0.4,
        "Execution Time": 3.6,
    }
]


class TestLimits:
    @pytest.mark.parametrize("limit", [1, 5, 10, 40])
    def test_limits_follow_the_retriever_rules(self, limit: int) -> None:
        assert sql_fetch_limit(channel_limit(limit, hybrid=True), arxiv_id=None) == candidate_fetch_limit(
            hybrid_branch_limit(limit)
        )
        assert sql_fetch_limit(channel_limit(limit, hybrid=False), arxiv_id=None) == candidate_fetch_limit(limit)
        assert sql_fetch_limit(channel_limit(limit, hybrid=True), arxiv_id="2401.00001") == hybrid_branch_limit(limit)
        assert sql_fetch_limit(channel_limit(limit, hybrid=False), arxiv_id="2401.00001") == limit

    def test_product_defaults(self) -> None:
        assert sql_fetch_limit(channel_limit(5, hybrid=True), arxiv_id=None) == 75
        assert sql_fetch_limit(channel_limit(5, hybrid=False), arxiv_id=None) == 30
        assert channel_limit(0, hybrid=False) == 1


class TestTargets:
    def test_lexical_target_is_the_product_sql(self) -> None:
        target = lexical_target("policy loss", sql_limit=75)
        assert target is not None
        assert (target.sql, target.params) == build_lexical_candidates_query("policy loss", limit=75)
        assert target.channel == LEXICAL and target.setup == ()
        assert target.describe() == "SQL LIMIT 75, 부분 일치 후보 상한 1500"
        assert lexical_target("   ", sql_limit=10) is None

    def test_vector_target_sets_ef_search_like_the_product(self) -> None:
        config = RetrievalConfig(embedding_model="text-embedding-3-large", vector_min_similarity=0.2)
        target = vector_target(EMBEDDING, sql_limit=75, config=config)

        sql, params = build_vector_search_query(
            EMBEDDING, limit=75, model_name="text-embedding-3-large", min_similarity=0.2
        )
        assert (target.sql, target.params) == (sql, params)
        assert target.setup == ((HNSW_EF_SEARCH_SQL, (str(resolve_hnsw_ef_search(600)),)),)
        assert target.describe() == "SQL LIMIT 75, 1단계 후보 600, hnsw.ef_search 600"

    def test_paper_scoped_vector_search_keeps_the_default_ef_search(self) -> None:
        target = vector_target(EMBEDDING, sql_limit=15, config=RetrievalConfig(), arxiv_id="2401.00001")
        assert target.channel == VECTOR and target.setup == ()
        assert target.params["arxiv_id"] == "2401.00001"
        assert "기본값" in target.describe()

    def test_negative_similarity_floor_is_clamped(self) -> None:
        target = vector_target(EMBEDDING, sql_limit=10, config=RetrievalConfig(vector_min_similarity=-0.5))
        assert "min_similarity" not in target.params


class TestPlanSummary:
    def test_summary_collects_indexes_scans_and_timings(self) -> None:
        summary = summarize_plan(PLAN)
        assert summary.indexes == {"idx_paper_chunks_chunk_vector"}
        assert summary.seq_scans == {"papers"}
        assert (summary.planning_ms, summary.execution_ms) == (0.4, 3.6)
        assert (summary.rows, summary.shared_hit, summary.shared_read) == (5, 40, 2)
        assert summarize_plan(PLAN[0]) == summary

    def test_render_plan_indents_children(self) -> None:
        lines = render_plan(PLAN[0]["Plan"]).splitlines()
        assert lines[0] == "- Limit: rows=5 loops=1 time=3.5ms"
        assert lines[1] == "  - Bitmap Heap Scan [paper_chunks]: rows=9 loops=1 time=1.0ms"
        assert lines[2].startswith("    - Bitmap Index Scan [idx_paper_chunks_chunk_vector]")
        assert lines[4] == "  - CTE Scan [query_terms]: rows=1 loops=1 time=Nonems"

    def test_report_marks_used_and_unused_indexes(self) -> None:
        summary = summarize_plan(PLAN)
        lines = report_lines(
            "lexical", [summary, summary], ("idx_paper_chunks_chunk_vector", "idx_papers_title_abstract_vector"), "TEXT"
        )
        assert "=== lexical ===" in lines and "TEXT" in lines
        assert "  idx_paper_chunks_chunk_vector: USED" in lines
        assert "  idx_papers_title_abstract_vector: not used" in lines
        assert SEQ_SCAN_NOTE in lines
        assert "  seq scans: papers" in lines
        assert "  execution ms per run: 3.600, 3.600" in lines
        assert "  execution ms median: 3.600" in lines
        assert lines[-1] == "  result rows: 5  shared buffers hit/read: 40/2"

        single = report_lines("vector", [summary], ())
        assert SEQ_SCAN_NOTE not in single
        assert not any(line.startswith("  execution ms median") for line in single)
        assert "  other indexes: idx_paper_chunks_chunk_vector" in single


class TestExplainCommand:
    def test_is_registered(self) -> None:
        assert "explain" in CliRunner().invoke(app, ["--help"]).output

    def test_empty_query_and_missing_key(self, clean_env) -> None:
        empty = CliRunner().invoke(app, ["explain", "--query", "\x00 "])
        assert empty.exit_code == 2 and "빈 질의" in empty.stderr
        keyless = CliRunner().invoke(app, ["explain", "--embed"])
        assert keyless.exit_code == 2 and "OPENAI_API_KEY" in keyless.stderr

    def test_unreachable_database(self, clean_env, monkeypatch: pytest.MonkeyPatch) -> None:
        for key, value in {
            "POSTGRES_HOST": "127.0.0.1:9",
            "POSTGRES_DB": "arxplore",
            "POSTGRES_USER": "arxplore",
            "POSTGRES_PASSWORD": "unused",
            "POSTGRES_CONNECT_TIMEOUT": "1",
        }.items():
            monkeypatch.setenv(key, value)
        result = CliRunner().invoke(app, ["explain", "--channel", "lexical"])
        assert result.exit_code == 2
        assert "EXPLAIN을 실행할 수 없습니다" in result.stderr
