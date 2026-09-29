import psycopg
import pytest
from psycopg import errors
from typer.testing import CliRunner

from arxplore.retrieval.service import RetrievalConfig
from arxplore_eval.cli import app
from arxplore_eval.explain import (
    LEXICAL,
    VECTOR,
    ExplainTarget,
    channel_indexes,
    explain_target,
    lexical_target,
    stored_query_embedding,
    vector_target,
)

pytestmark = pytest.mark.integration


@pytest.fixture
async def connection(empty_database: str, chunk_ids):
    async with await psycopg.AsyncConnection.connect(empty_database, autocommit=True) as opened:
        yield opened


async def test_explains_both_channels_and_rolls_back(connection) -> None:
    lexical = lexical_target("policy loss", sql_limit=30)
    assert lexical is not None
    run = await explain_target(connection, lexical, text=True)

    assert run.summary.rows is not None and run.summary.execution_ms is not None
    assert run.text_plan is not None and "Execution Time" in run.text_plan
    assert connection.info.transaction_status == psycopg.pq.TransactionStatus.IDLE

    stored = await stored_query_embedding(connection)
    assert stored is not None
    embedding, source = stored
    assert len(embedding) == 1536 and source.startswith("stored embedding of chunk")
    vector = await explain_target(connection, vector_target(embedding, sql_limit=30, config=RetrievalConfig()))
    assert vector.summary.rows is not None


async def test_the_transaction_is_read_only(connection) -> None:
    target = ExplainTarget(
        channel=LEXICAL, sql="SELECT 1", params={}, setup=(("CREATE TABLE eval_explain_probe (id int)", ()),)
    )
    with pytest.raises(errors.ReadOnlySqlTransaction):
        await explain_target(connection, target)
    async with connection.cursor() as cursor:
        await cursor.execute("SELECT to_regclass('eval_explain_probe')")
        assert await cursor.fetchone() == (None,)


async def test_expected_indexes_come_from_the_search_index_tables(connection) -> None:
    indexes = await channel_indexes(connection)
    assert set(indexes[LEXICAL]) == {"idx_papers_title_abstract_vector", "idx_paper_chunks_chunk_vector"}
    assert indexes[VECTOR] == ("paper_embeddings_embedding_hnsw",)


def test_explain_command_reports_both_channels(clean_env, empty_database: str, chunk_ids, monkeypatch) -> None:
    params = psycopg.conninfo.conninfo_to_dict(empty_database)
    for key, value in {
        "POSTGRES_HOST": f"{params.get('host', 'localhost')}:{params.get('port', 5432)}",
        "APP_POSTGRES_DB": str(params["dbname"]),
        "POSTGRES_USER": str(params.get("user", "")),
        "POSTGRES_PASSWORD": str(params.get("password") or "unused-by-trust-auth"),
    }.items():
        monkeypatch.setenv(key, value)

    result = CliRunner().invoke(app, ["explain", "--query", "policy loss", "--runs", "2"])

    assert result.exit_code == 0, result.output
    assert "=== lexical — SQL LIMIT 75, 부분 일치 후보 상한 1500 ===" in result.output
    assert "=== vector — SQL LIMIT 75, 1단계 후보 600, hnsw.ef_search 600 ===" in result.output
    assert "search indexes: idx_papers_title_abstract_vector present" in result.output
    assert result.output.count("execution ms median") == 2

    scoped = CliRunner().invoke(app, ["explain", "--query", "policy loss", "--arxiv-id", "2401.00001", "--no-hybrid"])
    assert scoped.exit_code == 0, scoped.output
    assert "=== vector (논문 범위, 정확 검색) — SQL LIMIT 5" in scoped.output
    assert "query: 'policy loss' sql_limit=5 arxiv_id=2401.00001" in scoped.output
