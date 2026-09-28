import asyncio
from typing import Any

import psycopg
import pytest

from arxplore.db.migrate import load_migrations, migrate

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
