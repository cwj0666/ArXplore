import psycopg
import pytest
from psycopg_pool import PoolTimeout

from arxplore.db.pool import connect, create_pool, create_pool_from_settings, open_pool

pytestmark = pytest.mark.integration


async def _session_details(connection: psycopg.AsyncConnection) -> dict[str, object]:
    cursor = await connection.execute(
        "SELECT current_setting('application_name') AS application_name, pg_backend_pid() AS pid"
    )
    row = await cursor.fetchone()
    assert isinstance(row, dict)
    return row


async def test_pool_yields_dict_rows_with_the_role_name(empty_database: str) -> None:
    pool = create_pool(empty_database, role="test", max_size=2, timeout=5, connect_timeout=4)

    async with open_pool(pool), pool.connection() as connection:
        details = await _session_details(connection)
        parameters = connection.info.get_parameters()

    assert details["application_name"] == "arxplore-test"
    assert parameters["connect_timeout"] == "4"
    assert pool.closed


async def test_pool_waits_then_times_out_when_exhausted(empty_database: str) -> None:
    pool = create_pool(empty_database, role="test", max_size=1, timeout=0.3, connect_timeout=4)

    async with open_pool(pool), pool.connection():
        with pytest.raises(PoolTimeout):
            async with pool.connection():
                pass


async def test_pool_replaces_a_connection_killed_by_the_server(empty_database: str) -> None:
    pool = create_pool(empty_database, role="test", max_size=1, timeout=5, connect_timeout=4)

    async with open_pool(pool):
        async with pool.connection() as connection:
            first_pid = (await _session_details(connection))["pid"]
        async with await psycopg.AsyncConnection.connect(empty_database, autocommit=True) as killer:
            await killer.execute("SELECT pg_terminate_backend(%s)", (first_pid,))
        async with pool.connection() as connection:
            second_pid = (await _session_details(connection))["pid"]

    assert second_pid != first_pid


async def test_pool_from_settings_uses_pool_settings(database_settings) -> None:
    settings = database_settings(POSTGRES_POOL_MAX="3", POSTGRES_POOL_TIMEOUT="2.5", POSTGRES_CONNECT_TIMEOUT="7")
    pool = create_pool_from_settings(settings, role="api")

    assert (pool.max_size, pool.min_size, pool.timeout) == (3, 1, 2.5)
    async with open_pool(pool), pool.connection() as connection:
        details = await _session_details(connection)
        parameters = connection.info.get_parameters()

    assert details["application_name"] == "arxplore-api"
    assert parameters["connect_timeout"] == "7"
    assert parameters["dbname"] == settings.app_postgres_db


async def test_open_pool_closes_the_pool_when_the_database_is_unreachable() -> None:
    pool = create_pool(
        "postgresql://arxplore@127.0.0.1:9/v2_test_unreachable",
        role="test",
        max_size=1,
        timeout=0.5,
        connect_timeout=2,
    )

    with pytest.raises(PoolTimeout):
        async with open_pool(pool):
            pass

    assert pool.closed


async def test_dedicated_connection(empty_database: str) -> None:
    connection = await connect(empty_database, role="worker", connect_timeout=5, autocommit=True)
    async with connection:
        details = await _session_details(connection)
        assert connection.autocommit is True

    assert details["application_name"] == "arxplore-worker"
