import pytest
from typer.testing import CliRunner

from arxplore_eval.cli import app
from arxplore_eval.embed_cost import CostSummary, summarize_texts, target_sql
from arxplore_eval.embedders import token_counter

runner = CliRunner()


def test_missing_targets_reuse_the_worker_condition() -> None:
    sql = target_sql(all_chunks=False)
    assert "e.chunk_id IS NULL" in sql
    assert "<> 'references'" in sql
    assert sql.rstrip().endswith("ORDER BY c.arxiv_id ASC, c.chunk_index ASC) AS targets")


def test_all_targets_ignore_existing_embeddings() -> None:
    sql = target_sql(all_chunks=True)
    assert "paper_embeddings" not in sql
    assert "<> 'references'" in sql


def test_summarize_texts_cleans_like_the_embedding_client() -> None:
    summary = summarize_texts(["  ab  ", "", "cdef", "\ud800x"], len, "len")
    assert summary == CostSummary(chunks=4, empty_chunks=1, characters=7, tokens=7, token_method="len")
    assert summary.usd(0.02) == pytest.approx(7 / 1_000_000 * 0.02)
    assert summary.as_dict(0.13)["usd"] == pytest.approx(7 / 1_000_000 * 0.13)


def test_token_counter_accepts_special_token_text() -> None:
    count, _ = token_counter()
    assert count("text with <|endoftext|> inside") > 0


def test_embed_cost_requires_database_and_price() -> None:
    result = runner.invoke(app, ["embed-cost", "--model", "text-embedding-3-small"])
    assert result.exit_code == 2


def test_embed_cost_reports_unreachable_database() -> None:
    result = runner.invoke(
        app,
        [
            "embed-cost",
            "--db",
            "postgresql://nobody@127.0.0.1:9/none?connect_timeout=1",
            "--model",
            "text-embedding-3-small",
            "--price-per-million",
            "0.02",
        ],
    )
    assert result.exit_code == 2
    assert "PostgreSQL에 연결할 수 없습니다" in result.stderr
