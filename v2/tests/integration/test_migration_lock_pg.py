import asyncio

import pytest

from arxplore.db.migrate import MIGRATION_LOCK_KEY, MigrationLockTimeoutError, load_migrations, migrate

pytestmark = pytest.mark.integration


async def _try_lock(connection) -> bool:
    cursor = await connection.execute("SELECT pg_try_advisory_lock(%s) AS locked", (MIGRATION_LOCK_KEY,))
    row = await cursor.fetchone()
    return bool(row["locked"])


async def test_gives_up_when_the_lock_is_held(open_connection) -> None:
    holder = await open_connection()
    connection = await open_connection()
    await holder.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_KEY,))

    with pytest.raises(MigrationLockTimeoutError):
        await migrate(connection, lock_timeout=0.3)

    cursor = await connection.execute("SELECT to_regclass('schema_migrations') IS NULL AS absent")
    assert (await cursor.fetchone())["absent"] is True


async def test_waits_for_the_lock_to_be_released(open_connection) -> None:
    holder = await open_connection()
    connection = await open_connection()
    await holder.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_KEY,))

    run = asyncio.create_task(migrate(connection, lock_timeout=10))
    await asyncio.sleep(0.5)
    assert not run.done()
    await holder.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK_KEY,))

    assert await run == load_migrations()


async def test_concurrent_runs_apply_each_migration_once(open_connection) -> None:
    connections = [await open_connection() for _ in range(3)]
    packaged = load_migrations()

    results = await asyncio.gather(*(migrate(connection, lock_timeout=30) for connection in connections))

    assert sorted(len(result) for result in results) == [0, 0, len(packaged)]
    cursor = await connections[0].execute("SELECT count(*) AS applied FROM schema_migrations")
    assert (await cursor.fetchone())["applied"] == len(packaged)


async def test_lock_is_released_after_a_run(open_connection) -> None:
    connection = await open_connection()
    other = await open_connection()

    await migrate(connection, target=1)

    assert await _try_lock(other) is True
