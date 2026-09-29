from pathlib import Path

import psycopg
import pytest

from arxplore.db.migrate import (
    MIGRATION_LOCK_KEY,
    ChecksumMismatchError,
    Migration,
    MigrationError,
    load_migrations,
    migrate,
)

pytestmark = pytest.mark.integration


async def _scalar(connection: psycopg.AsyncConnection, sql: str, params: tuple[object, ...] = ()) -> object:
    cursor = await connection.execute(sql, params)
    row = await cursor.fetchone()
    assert row is not None
    return next(iter(row.values()))


async def _relation_exists(connection: psycopg.AsyncConnection, name: str) -> bool:
    return bool(await _scalar(connection, "SELECT to_regclass(%s) IS NOT NULL AS present", (name,)))


async def _recorded(connection: psycopg.AsyncConnection) -> list[tuple[int, str]]:
    cursor = await connection.execute("SELECT version, checksum FROM schema_migrations ORDER BY version")
    return [(row["version"], row["checksum"]) for row in await cursor.fetchall()]


def _write(directory: Path, files: dict[str, str]) -> list[Migration]:
    for name, sql in files.items():
        (directory / name).write_text(sql, encoding="utf-8")
    return load_migrations(directory)


THREE_TABLES = {
    "0001_first.sql": "CREATE TABLE first_table (id INT);",
    "0002_second.sql": "CREATE TABLE second_table (id INT);",
    "0003_third.sql": "CREATE TABLE third_table (id INT);",
}


async def test_fresh_database_gets_every_packaged_migration(open_connection) -> None:
    connection = await open_connection()
    packaged = load_migrations()

    assert await migrate(connection) == packaged

    assert await _recorded(connection) == [(migration.version, migration.checksum) for migration in packaged]
    assert await _relation_exists(connection, "paper_embeddings_embedding_hnsw")


async def test_second_run_is_a_no_op(open_connection) -> None:
    connection = await open_connection()
    await migrate(connection)
    cursor = await connection.execute("SELECT version, applied_at FROM schema_migrations ORDER BY version")
    before = await cursor.fetchall()

    assert await migrate(connection) == []
    cursor = await connection.execute("SELECT version, applied_at FROM schema_migrations ORDER BY version")
    assert await cursor.fetchall() == before


async def test_target_stops_at_the_requested_version(open_connection, migrations_dir: Path) -> None:
    connection = await open_connection()
    migrations = _write(migrations_dir, THREE_TABLES)

    assert [migration.version for migration in await migrate(connection, migrations=migrations, target=2)] == [1, 2]
    assert not await _relation_exists(connection, "third_table")

    assert [migration.version for migration in await migrate(connection, migrations=migrations)] == [3]
    assert [version for version, _ in await _recorded(connection)] == [1, 2, 3]


async def test_dry_run_changes_nothing(open_connection) -> None:
    connection = await open_connection()

    assert await migrate(connection, dry_run=True) == load_migrations()

    assert not await _relation_exists(connection, "schema_migrations")
    assert not await _relation_exists(connection, "papers")


async def test_dry_run_after_partial_migration(open_connection, migrations_dir: Path) -> None:
    connection = await open_connection()
    migrations = _write(migrations_dir, THREE_TABLES)
    await migrate(connection, migrations=migrations, target=1)

    planned = await migrate(connection, migrations=migrations, dry_run=True, target=2)

    assert [migration.version for migration in planned] == [2]
    assert [version for version, _ in await _recorded(connection)] == [1]
    assert not await _relation_exists(connection, "second_table")


async def test_modified_applied_migration_is_rejected(open_connection) -> None:
    connection = await open_connection()
    await migrate(connection)
    await connection.execute("UPDATE schema_migrations SET checksum = %s WHERE version = 1", ("0" * 64,))
    recorded = await _recorded(connection)

    with pytest.raises(ChecksumMismatchError, match="0001_baseline"):
        await migrate(connection)
    with pytest.raises(ChecksumMismatchError, match="0001_baseline"):
        await migrate(connection, dry_run=True)
    assert await _recorded(connection) == recorded


async def test_edited_migration_file_is_detected(open_connection, migrations_dir: Path) -> None:
    connection = await open_connection()
    await migrate(
        connection, migrations=_write(migrations_dir, {"0001_first.sql": "CREATE TABLE first_table (id INT);"})
    )
    edited = _write(
        migrations_dir,
        {
            "0001_first.sql": "CREATE TABLE first_table (id BIGINT);",
            "0002_second.sql": "CREATE TABLE second_table (id INT);",
        },
    )

    with pytest.raises(ChecksumMismatchError, match="0001_first"):
        await migrate(connection, migrations=edited)
    assert not await _relation_exists(connection, "second_table")


async def test_failing_file_rolls_back_only_itself(open_connection, migrations_dir: Path) -> None:
    connection = await open_connection()
    other = await open_connection()
    migrations = _write(
        migrations_dir,
        {
            "0001_first.sql": "CREATE TABLE first_table (id INT);",
            "0002_second.sql": "CREATE TABLE second_table (id INT);\nINSERT INTO second_table VALUES (1 / 0);",
            "0003_third.sql": "CREATE TABLE third_table (id INT);",
        },
    )

    with pytest.raises(psycopg.errors.DivisionByZero):
        await migrate(connection, migrations=migrations)

    assert [version for version, _ in await _recorded(connection)] == [1]
    assert await _relation_exists(connection, "first_table")
    assert not await _relation_exists(connection, "second_table")
    assert not await _relation_exists(connection, "third_table")
    assert await _scalar(other, "SELECT pg_try_advisory_lock(%s) AS locked", (MIGRATION_LOCK_KEY,)) is True


async def test_statements_with_percent_signs_and_dollar_quotes_run_verbatim(
    open_connection, migrations_dir: Path
) -> None:
    connection = await open_connection()
    migrations = _write(
        migrations_dir,
        {
            "0001_literal.sql": "CREATE TABLE literal_table (value TEXT);\n"
            "INSERT INTO literal_table VALUES ('100% done'), ($body$it's $1 %s$body$);",
        },
    )

    await migrate(connection, migrations=migrations)

    cursor = await connection.execute("SELECT value FROM literal_table ORDER BY value")
    assert [row["value"] for row in await cursor.fetchall()] == ["100% done", "it's $1 %s"]


async def test_non_autocommit_connection_is_refused(open_connection) -> None:
    connection = await open_connection(autocommit=False)

    with pytest.raises(MigrationError, match="autocommit"):
        await migrate(connection)
