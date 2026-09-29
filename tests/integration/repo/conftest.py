from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from typing import Any

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row

from arxplore.db.migrate import MigrationState, fetch_applied, load_migrations, migrate, migration_status
from arxplore.db.pool import Pool, create_pool, open_pool
from arxplore.db.repositories.jobs import JOB_CHANNEL
from arxplore.domain.jobs import QueuePolicy

REPO_TABLES = (
    "papers",
    "paper_fulltexts",
    "paper_chunks",
    "paper_embeddings",
    "paper_ai_overviews",
    "paper_ai_detailed_summaries",
    "raw_daily_papers",
    "pipeline_state",
    "prepare_jobs",
)

SENTINEL_PREFIX = "sentinel:"

CLEAR_TABLES_SQL = """
DELETE FROM papers;
DELETE FROM raw_daily_papers;
DELETE FROM pipeline_state;
DELETE FROM prepare_jobs;
ALTER SEQUENCE paper_chunks_id_seq RESTART;
ALTER SEQUENCE prepare_jobs_id_seq RESTART;
ALTER SEQUENCE paper_ai_detailed_summaries_id_seq RESTART;
"""


async def _schema_is_current(connection: psycopg.AsyncConnection[Any]) -> bool:
    statuses = migration_status(load_migrations(), await fetch_applied(connection))
    if not statuses or any(status.state is not MigrationState.APPLIED for status in statuses):
        return False
    cursor = await connection.execute(
        "SELECT count(*) AS present FROM unnest(%s::text[]) AS name WHERE to_regclass(name) IS NOT NULL",
        (list(REPO_TABLES),),
    )
    row = await cursor.fetchone()
    return row is not None and row["present"] == len(REPO_TABLES)


@pytest.fixture
async def repo_database(database_url: str) -> str:
    async with await psycopg.AsyncConnection.connect(database_url, autocommit=True, row_factory=dict_row) as connection:
        if await _schema_is_current(connection):
            await connection.execute(CLEAR_TABLES_SQL)
        else:
            await connection.execute("DROP SCHEMA IF EXISTS public CASCADE")
            await connection.execute("CREATE SCHEMA public")
            await migrate(connection)
    return database_url


@pytest.fixture
async def pool(repo_database: str) -> AsyncIterator[Pool]:
    repo_pool = create_pool(repo_database, role="test-repo", max_size=8, timeout=10, connect_timeout=5)
    async with open_pool(repo_pool):
        yield repo_pool


class SqlProbe:
    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    async def execute(self, query: str, params: Sequence[Any] = ()) -> None:
        async with self._pool.connection() as connection:
            await connection.execute(query, params)

    async def rows(self, query: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(query, params)
            return list(await cursor.fetchall())

    async def row(self, query: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        rows = await self.rows(query, params)
        return rows[0] if rows else None

    async def scalar(self, query: str, params: Sequence[Any] = ()) -> Any:
        row = await self.row(query, params)
        assert row is not None
        return next(iter(row.values()))


@pytest.fixture
def probe(pool: Pool) -> SqlProbe:
    return SqlProbe(pool)


@pytest.fixture
def policy() -> QueuePolicy:
    return QueuePolicy(max_attempts=3, stale_seconds=900)


class NotificationProbe:
    def __init__(self, connection: psycopg.AsyncConnection[Any], probe: SqlProbe) -> None:
        self._connection = connection
        self._probe = probe
        self._sentinels = 0

    async def drain(self) -> list[str]:
        self._sentinels += 1
        sentinel = f"{SENTINEL_PREFIX}{self._sentinels}"
        await self._probe.execute("SELECT pg_notify(%s, %s)", (JOB_CHANNEL, sentinel))
        received: list[str] = []
        async with aclosing(self._connection.notifies(timeout=5.0)) as stream:
            async for notify in stream:
                if notify.payload == sentinel:
                    return received
                received.append(notify.payload)
        raise AssertionError("the sentinel notification did not arrive")


@pytest.fixture
async def notifications(repo_database: str, probe: SqlProbe) -> AsyncIterator[NotificationProbe]:
    async with await psycopg.AsyncConnection.connect(repo_database, autocommit=True) as connection:
        await connection.execute(sql.SQL("LISTEN {}").format(sql.Identifier(JOB_CHANNEL)))
        yield NotificationProbe(connection, probe)
