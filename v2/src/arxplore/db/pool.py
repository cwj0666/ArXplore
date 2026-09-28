from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from psycopg import AsyncConnection
from psycopg.conninfo import make_conninfo
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from arxplore.config import AppSettings, build_postgres_connection_params

type DictConnection = AsyncConnection[DictRow]
type Pool = AsyncConnectionPool[AsyncConnection[DictRow]]
type AnyPool = AsyncConnectionPool[Any]
type ConnectionSource = AnyPool | AsyncConnection[Any]

APPLICATION_NAME_PREFIX = "arxplore-"


def application_name(role: str) -> str:
    return f"{APPLICATION_NAME_PREFIX}{role}"


def settings_conninfo(settings: AppSettings) -> str:
    params = build_postgres_connection_params(settings)
    return make_conninfo(
        "",
        dbname=params["dbname"],
        user=params["user"],
        password=params["password"],
        host=params["host"],
        port=params["port"],
    )


def connection_options(role: str, connect_timeout: int) -> dict[str, Any]:
    return {"application_name": application_name(role), "connect_timeout": connect_timeout}


def create_pool(
    conninfo: str,
    *,
    role: str,
    max_size: int,
    timeout: float,
    connect_timeout: int,
    min_size: int = 1,
) -> Pool:
    return AsyncConnectionPool(
        conninfo,
        connection_class=AsyncConnection[DictRow],
        kwargs={**connection_options(role, connect_timeout), "row_factory": dict_row},
        min_size=min(min_size, max_size),
        max_size=max_size,
        timeout=timeout,
        check=AsyncConnectionPool.check_connection,
        name=application_name(role),
        open=False,
    )


def create_pool_from_settings(settings: AppSettings, *, role: str, min_size: int = 1) -> Pool:
    return create_pool(
        settings_conninfo(settings),
        role=role,
        max_size=settings.postgres_pool_max,
        timeout=settings.postgres_pool_timeout,
        connect_timeout=settings.postgres_connect_timeout,
        min_size=min_size,
    )


@asynccontextmanager
async def open_pool(pool: Pool) -> AsyncIterator[Pool]:
    try:
        await pool.open(wait=True, timeout=pool.timeout)
        yield pool
    finally:
        await pool.close()


async def connect(
    conninfo: str,
    *,
    role: str,
    connect_timeout: int,
    autocommit: bool = False,
) -> DictConnection:
    return await AsyncConnection.connect(
        conninfo,
        autocommit=autocommit,
        row_factory=dict_row,
        **connection_options(role, connect_timeout),
    )


async def connect_from_settings(settings: AppSettings, *, role: str, autocommit: bool = False) -> DictConnection:
    return await connect(
        settings_conninfo(settings),
        role=role,
        connect_timeout=settings.postgres_connect_timeout,
        autocommit=autocommit,
    )


def transaction(pool: AnyPool) -> AbstractAsyncContextManager[AsyncConnection[Any]]:
    return pool.connection()


@asynccontextmanager
async def use_connection(source: ConnectionSource) -> AsyncIterator[AsyncConnection[Any]]:
    if isinstance(source, AsyncConnection):
        yield source
        return
    async with source.connection() as connection:
        yield connection
