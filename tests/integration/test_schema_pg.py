import asyncio
from typing import Any

import psycopg
import pytest

from arxplore.db.migrate import (
    Migration,
    SchemaNotReadyError,
    compute_checksum,
    load_migrations,
    migrate,
    verify_schema,
)

pytestmark = pytest.mark.integration

CATALOG_QUERIES = {
    "columns": """
        SELECT table_name, column_name, ordinal_position, data_type, udt_name, is_nullable,
               column_default, is_generated, generation_expression
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name <> 'schema_migrations'
        ORDER BY table_name, ordinal_position
    """,
    "indexes": """
        SELECT tablename, indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = 'public' AND tablename <> 'schema_migrations'
        ORDER BY tablename, indexname
    """,
    "constraints": """
        SELECT conrelid::regclass::text AS table_name, conname, pg_get_constraintdef(oid) AS definition
        FROM pg_constraint
        WHERE connamespace = 'public'::regnamespace AND conrelid::regclass::text <> 'schema_migrations'
        ORDER BY 1, 2
    """,
    "extensions": "SELECT extname, extversion FROM pg_extension ORDER BY extname",
}


async def _catalog(connection: psycopg.AsyncConnection) -> dict[str, list[dict[str, Any]]]:
    snapshot = {}
    for name, sql in CATALOG_QUERIES.items():
        cursor = await connection.execute(sql)
        snapshot[name] = await cursor.fetchall()
    return snapshot


async def _relation_exists(connection: psycopg.AsyncConnection, name: str) -> bool:
    cursor = await connection.execute("SELECT to_regclass(%s) IS NOT NULL AS present", (name,))
    return bool((await cursor.fetchone())["present"])


async def _column_type(connection: psycopg.AsyncConnection, table: str, column: str) -> str:
    cursor = await connection.execute(
        "SELECT data_type FROM information_schema.columns WHERE table_name = %s AND column_name = %s",
        (table, column),
    )
    return str((await cursor.fetchone())["data_type"])


async def _latest_version(connection: psycopg.AsyncConnection) -> int:
    cursor = await connection.execute("SELECT max(version) AS latest FROM schema_migrations")
    return int((await cursor.fetchone())["latest"])


async def _apply_baseline_without_recording(connection: psycopg.AsyncConnection) -> None:
    baseline = load_migrations()[0]
    async with connection.transaction():
        await connection.execute(baseline.sql.encode())


async def test_baseline_is_a_no_op_on_an_existing_legacy_schema(open_connection) -> None:
    connection = await open_connection()
    await _apply_baseline_without_recording(connection)
    before = await _catalog(connection)

    assert [migration.version for migration in await migrate(connection, target=1)] == [1]

    assert await _catalog(connection) == before


async def test_hnsw_failure_only_warns(open_connection) -> None:
    setup = await open_connection()
    await _apply_baseline_without_recording(setup)
    await setup.execute("DROP INDEX paper_embeddings_embedding_hnsw")
    holder = await open_connection(autocommit=False)
    await holder.execute("LOCK TABLE paper_embeddings IN ACCESS EXCLUSIVE MODE")
    connection = await open_connection(options="-c lock_timeout=300")
    warnings: list[str] = []

    def collect_warning(diagnostic: psycopg.errors.Diagnostic) -> None:
        if diagnostic.severity_nonlocalized == "WARNING":
            warnings.append(diagnostic.message_primary or "")

    connection.add_notice_handler(collect_warning)

    applied = await asyncio.wait_for(migrate(connection, target=1), timeout=30)

    await holder.rollback()
    assert [migration.version for migration in applied] == [1]
    assert any("HNSW" in message for message in warnings)
    assert not await _relation_exists(setup, "paper_embeddings_embedding_hnsw")


async def test_verify_schema_accepts_a_migrated_database(open_connection) -> None:
    connection = await open_connection()
    await migrate(connection)

    await verify_schema(connection, embedding_dimensions=1536)


async def test_verify_schema_rejects_pending_migrations(open_connection) -> None:
    connection = await open_connection()
    await migrate(connection)
    sql = "SELECT 1;"
    future = Migration(version=9999, name="9999_future", sql=sql, checksum=compute_checksum(sql))

    with pytest.raises(SchemaNotReadyError, match="pending migrations: 9999_future"):
        await verify_schema(connection, embedding_dimensions=1536, migrations=[*load_migrations(), future])


async def test_verify_schema_rejects_an_empty_database(open_connection) -> None:
    connection = await open_connection()

    with pytest.raises(SchemaNotReadyError, match=r"0001_baseline.*paper_embeddings\.embedding is missing"):
        await verify_schema(connection, embedding_dimensions=1536)


async def test_verify_schema_rejects_a_dimension_mismatch(open_connection) -> None:
    connection = await open_connection()
    await migrate(connection)

    with pytest.raises(SchemaNotReadyError, match=r"vector\(1536\) but OPENAI_EMBEDDING_DIMENSIONS is 3072"):
        await verify_schema(connection, embedding_dimensions=3072)


async def test_verify_schema_rejects_modified_migrations(open_connection) -> None:
    connection = await open_connection()
    await migrate(connection)
    await connection.execute("UPDATE schema_migrations SET checksum = %s WHERE version = 1", ("0" * 64,))

    with pytest.raises(SchemaNotReadyError, match="modified migrations: 0001_baseline"):
        await verify_schema(connection, embedding_dimensions=1536)


async def test_verify_schema_rejects_unknown_database_versions(open_connection) -> None:
    connection = await open_connection()
    await migrate(connection)
    await connection.execute("INSERT INTO schema_migrations (version, checksum) VALUES (9998, 'future')")

    with pytest.raises(SchemaNotReadyError, match="unknown migrations in database: 9998"):
        await verify_schema(connection, embedding_dimensions=1536)


async def test_0002_drops_the_legacy_topic_tables(open_connection) -> None:
    connection = await open_connection()
    topic_tables = ("topics", "topic_papers", "topic_documents")
    await migrate(connection, target=1)
    assert all([await _relation_exists(connection, table) for table in topic_tables])

    assert [migration.version for migration in await migrate(connection, target=2)] == [2]

    assert not any([await _relation_exists(connection, table) for table in topic_tables])
    assert await _relation_exists(connection, "papers")


LEGACY_JOBS = [
    ("auto", "2026-04-06", "done", "{}"),
    ("auto", "2026-04-07", "processing", '{"force": true}'),
    ("auto", "2026-04-08", "pending", "{}"),
    ("auto", "2026-04-09", "failed", '{"force": true}'),
    ("backfill", "2026-04-07", "pending", "{}"),
    ("auto", "2025-12-31", "done", "{}"),
]


async def _insert_legacy_jobs(connection: psycopg.AsyncConnection, jobs: list[tuple[str, str, str, str]]) -> None:
    async with connection.cursor() as cursor:
        await cursor.executemany(
            "INSERT INTO prepare_jobs (mode, target_date, status, payload) VALUES (%s, %s, %s, %s::jsonb)",
            jobs,
        )


async def test_0003_converts_legacy_target_dates(open_connection) -> None:
    connection = await open_connection()
    await migrate(connection, target=2)
    await _insert_legacy_jobs(connection, LEGACY_JOBS)

    assert [migration.version for migration in await migrate(connection, target=3)] == [3]

    assert await _column_type(connection, "prepare_jobs", "target_date") == "date"
    cursor = await connection.execute("SELECT mode, target_date::text AS target_date, status FROM prepare_jobs")
    assert sorted(tuple(row.values()) for row in await cursor.fetchall()) == sorted(
        (mode, target_date, status) for mode, target_date, status, _ in LEGACY_JOBS
    )
    cursor = await connection.execute(
        "SELECT target_date FROM prepare_jobs WHERE mode = 'auto' AND status = 'pending' AND target_date >= %s",
        ("2026-04-07",),
    )
    assert [str(row["target_date"]) for row in await cursor.fetchall()] == ["2026-04-08"]
    with pytest.raises(psycopg.errors.CheckViolation):
        await connection.execute(
            "INSERT INTO prepare_jobs (mode, target_date, status) VALUES ('auto', '2026-05-01', 'running')"
        )
    with pytest.raises(psycopg.errors.UniqueViolation):
        await connection.execute("INSERT INTO prepare_jobs (mode, target_date) VALUES ('auto', '2026-04-06')")


@pytest.mark.parametrize(
    ("job", "error"),
    [
        (("auto", "2026/04/07", "pending", "{}"), psycopg.errors.RaiseException),
        (("auto", "2026-02-30", "pending", "{}"), psycopg.errors.DatetimeFieldOverflow),
        (("auto", "2026-04-07", "running", "{}"), psycopg.errors.CheckViolation),
    ],
)
async def test_0003_refuses_rows_it_cannot_convert(
    open_connection, job: tuple[str, str, str, str], error: type[Exception]
) -> None:
    connection = await open_connection()
    await migrate(connection, target=2)
    await _insert_legacy_jobs(connection, [job])

    with pytest.raises(error):
        await migrate(connection, target=3)

    assert await _latest_version(connection) == 2
    assert await _column_type(connection, "prepare_jobs", "target_date") == "text"
