from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row

from arxplore.db.migrate import migrate
from arxplore.db.repositories import generation_cache

pytestmark = pytest.mark.integration

ARXIV_ID = "2401.00001"


async def prepared(url: str, *, target: int | None = None) -> psycopg.AsyncConnection[Any]:
    connection = await psycopg.AsyncConnection.connect(url, autocommit=True, row_factory=dict_row)
    await migrate(connection, target=target)
    await connection.execute("INSERT INTO papers (arxiv_id, title) VALUES (%s, 'T')", (ARXIV_ID,))
    return connection


async def test_rows_written_before_0004_become_prompt_version_1(empty_database: str) -> None:
    connection = await prepared(empty_database, target=3)
    async with connection:
        await connection.execute(
            "INSERT INTO paper_ai_overviews (arxiv_id, overview, key_findings, model) VALUES (%s, 'old', '[\"k\"]', 'm')",
            (ARXIV_ID,),
        )
        await connection.execute(
            "INSERT INTO paper_ai_detailed_summaries (arxiv_id, model, summary) VALUES (%s, 'gpt-5', 'old summary')",
            (ARXIV_ID,),
        )

        assert [migration.version for migration in await migrate(connection, target=4)] == [4]

        overview = await generation_cache.get_overview(connection, ARXIV_ID, prompt_version=1)
        summary = await generation_cache.get_detailed_summary(connection, ARXIV_ID, "gpt-5", prompt_version=1)
        assert overview is not None and (overview.overview, overview.key_findings) == ("old", ["k"])
        assert summary is not None and summary.summary == "old summary"
        with pytest.raises(psycopg.errors.CheckViolation):
            await connection.execute("UPDATE paper_ai_overviews SET prompt_version = 0")


async def test_other_prompt_versions_miss_and_saving_replaces_the_row(empty_database: str) -> None:
    connection = await prepared(empty_database)
    async with connection:
        await generation_cache.save_overview(
            connection, ARXIV_ID, overview="v1", key_findings=["a"], model="gpt-5-mini", prompt_version=1
        )
        await generation_cache.save_detailed_summary(
            connection, ARXIV_ID, summary="s1", model="gpt-5", prompt_version=1
        )

        assert await generation_cache.get_overview(connection, ARXIV_ID, prompt_version=2) is None
        assert await generation_cache.get_detailed_summary(connection, ARXIV_ID, "gpt-5", prompt_version=2) is None
        assert await generation_cache.get_detailed_summary(connection, ARXIV_ID, "gpt-5-mini", prompt_version=1) is None

        await generation_cache.save_overview(
            connection, ARXIV_ID, overview="v2", key_findings=[], model="gpt-5-mini", prompt_version=2
        )
        await generation_cache.save_detailed_summary(
            connection, ARXIV_ID, summary="s2", model="gpt-5", prompt_version=2
        )

        overview = await generation_cache.get_overview(connection, ARXIV_ID, prompt_version=2)
        summary = await generation_cache.get_detailed_summary(connection, ARXIV_ID, "gpt-5", prompt_version=2)
        assert overview is not None and (overview.overview, overview.key_findings) == ("v2", [])
        assert summary is not None and summary.summary == "s2"
        assert await generation_cache.get_overview(connection, ARXIV_ID, prompt_version=1) is None
        cursor = await connection.execute("SELECT count(*) AS rows FROM paper_ai_detailed_summaries")
        assert (await cursor.fetchone()) == {"rows": 1}
