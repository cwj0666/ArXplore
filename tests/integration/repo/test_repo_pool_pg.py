import psycopg
import pytest

from arxplore.db.pool import create_pool, open_pool, transaction, use_connection

pytestmark = pytest.mark.integration


async def _backend_pid(connection: psycopg.AsyncConnection) -> int:
    cursor = await connection.execute("SELECT pg_backend_pid() AS pid")
    row = await cursor.fetchone()
    assert row is not None
    return int(row["pid"])


async def test_transaction_commits_reuses_the_connection_and_rolls_back_on_error(repo_database: str) -> None:
    single = create_pool(repo_database, role="test-repo", max_size=1, timeout=5, connect_timeout=5)
    async with open_pool(single):
        async with transaction(single) as connection:
            await connection.execute("CREATE TABLE pool_probe (value INTEGER)")
            first_pid = await _backend_pid(connection)

        with pytest.raises(RuntimeError, match="abort"):
            async with transaction(single) as connection:
                assert await _backend_pid(connection) == first_pid
                await connection.execute("INSERT INTO pool_probe VALUES (1)")
                raise RuntimeError("abort")

        async with transaction(single) as connection:
            cursor = await connection.execute("SELECT count(*) AS rows FROM pool_probe")
            assert (await cursor.fetchone())["rows"] == 0
            assert await _backend_pid(connection) == first_pid

        with pytest.raises(psycopg.OperationalError):
            async with transaction(single) as connection:
                await connection.execute("SELECT pg_terminate_backend(pg_backend_pid())")

        async with transaction(single) as connection:
            assert await _backend_pid(connection) != first_pid
            await connection.execute("DROP TABLE pool_probe")


async def test_use_connection_joins_the_callers_transaction_without_committing(pool, probe) -> None:
    async with pool.connection() as connection:
        async with use_connection(connection) as joined:
            assert joined is connection
            await joined.execute("CREATE TEMPORARY TABLE scratch (value INTEGER) ON COMMIT DROP")
            await joined.execute("INSERT INTO scratch VALUES (1)")
        cursor = await connection.execute("SELECT count(*) AS rows FROM scratch")
        assert (await cursor.fetchone())["rows"] == 1
        await connection.rollback()

    async with use_connection(pool) as borrowed:
        await borrowed.execute("INSERT INTO pipeline_state (key, value) VALUES ('probe:default', '{}')")
    assert await probe.scalar("SELECT count(*) FROM pipeline_state") == 1

    with pytest.raises(RuntimeError, match="abort"):
        async with use_connection(pool) as borrowed:
            await borrowed.execute("DELETE FROM pipeline_state")
            raise RuntimeError("abort")
    assert await probe.scalar("SELECT count(*) FROM pipeline_state") == 1
