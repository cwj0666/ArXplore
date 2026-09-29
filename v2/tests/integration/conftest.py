import os
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import DictRow, dict_row

from arxplore.config import AppSettings

DATABASE_URL_VARIABLE = "V2_TEST_DATABASE_URL"
REQUIRE_DATABASE_VARIABLE = "V2_TEST_REQUIRE_DB"
SAFE_DATABASE_PREFIX = "v2_test"


def _configured_url() -> str:
    return os.environ.get(DATABASE_URL_VARIABLE, "").strip()


@pytest.fixture(scope="session")
def database_url() -> str:
    url = _configured_url()
    if not url:
        message = f"{DATABASE_URL_VARIABLE} is not set; point it at a dedicated {SAFE_DATABASE_PREFIX}* database"
        if os.environ.get(REQUIRE_DATABASE_VARIABLE):
            pytest.fail(message)
        pytest.skip(message)
    dbname = str(conninfo_to_dict(url).get("dbname", ""))
    if not dbname.startswith(SAFE_DATABASE_PREFIX):
        pytest.fail(f"integration tests drop the public schema; use a database named {SAFE_DATABASE_PREFIX}*: {dbname}")
    try:
        with psycopg.connect(url, connect_timeout=3):
            pass
    except psycopg.OperationalError as exc:
        if os.environ.get(REQUIRE_DATABASE_VARIABLE):
            pytest.fail(f"PostgreSQL is not reachable at {DATABASE_URL_VARIABLE}: {exc}")
        pytest.skip(f"PostgreSQL is not reachable at {DATABASE_URL_VARIABLE}: {exc}")
    return url


@pytest.fixture
async def empty_database(database_url: str) -> str:
    async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as connection:
        await connection.execute("DROP SCHEMA IF EXISTS public CASCADE")
        await connection.execute("CREATE SCHEMA public")
    return database_url


type ConnectionFactory = Callable[..., Awaitable[psycopg.AsyncConnection[DictRow]]]


@pytest.fixture
async def open_connection(empty_database: str) -> AsyncIterator[ConnectionFactory]:
    opened: list[psycopg.AsyncConnection[DictRow]] = []

    async def factory(*, autocommit: bool = True, **options: object) -> psycopg.AsyncConnection[DictRow]:
        connection = await psycopg.AsyncConnection.connect(
            empty_database,
            autocommit=autocommit,
            row_factory=dict_row,
            **options,
        )
        opened.append(connection)
        return connection

    yield factory
    for connection in opened:
        await connection.close()


@pytest.fixture
def database_settings(empty_database: str) -> Callable[..., AppSettings]:
    params = conninfo_to_dict(empty_database)

    def build(**overrides: str) -> AppSettings:
        values = {
            "POSTGRES_HOST": f"{params.get('host', 'localhost')}:{params.get('port', 5432)}",
            "APP_POSTGRES_DB": str(params["dbname"]),
            "POSTGRES_USER": str(params.get("user", "")),
            "POSTGRES_PASSWORD": str(params.get("password") or "unused-by-trust-auth"),
        }
        values.update(overrides)
        return AppSettings.model_validate(values)

    return build


@pytest.fixture
def migrations_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "migrations"
    directory.mkdir()
    return directory
