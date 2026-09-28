import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from urllib.parse import unquote, urlsplit

DATABASE_URL_ENV = "GOLDEN_DATABASE_URL"
DEFAULT_DATABASE_URL = "postgresql://arxplore@localhost:55432/arxplore"
TEMPLATE_DATABASE = "arxplore_tpl"
TRUST_PASSWORD = "golden-trust"
READ_ONLY_OPTION = "-c default_transaction_read_only=on"
EXACT_SEARCH_OPTION = "-c enable_indexscan=off"
EXACT_SEARCH_ENV = "GOLDEN_EXACT_SEARCH"
COPY_PREFIX = "golden_"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def database_url() -> str:
    return os.environ.get(DATABASE_URL_ENV, "").strip() or DEFAULT_DATABASE_URL


def with_database(url: str, name: str) -> str:
    return urlsplit(url)._replace(path=f"/{name}").geturl()


def database_name(url: str) -> str:
    return urlsplit(url).path.lstrip("/")


def database_environ(url: str) -> dict[str, str]:
    parts = urlsplit(url)
    port = parts.port or 5432
    name = database_name(url)
    return {
        "POSTGRES_HOST": f"{parts.hostname or 'localhost'}:{port}",
        "SERVER_POSTGRES_PORT": str(port),
        "POSTGRES_DB": name,
        "APP_POSTGRES_DB": name,
        "POSTGRES_USER": unquote(parts.username or "arxplore"),
        "POSTGRES_PASSWORD": unquote(parts.password or TRUST_PASSWORD),
    }


def harness_environ(
    *,
    openai_environ: dict[str, str],
    database_url: str,
    pg_options: str = READ_ONLY_OPTION,
) -> dict[str, str]:
    return {
        **openai_environ,
        **database_environ(database_url),
        "LAYOUT_PARSER_BASE_URL": "",
        "RETRIEVAL_MODE": "hybrid",
        "PGOPTIONS": pg_options,
    }


def database_reachable(url: str, timeout: float = 1.0) -> bool:
    parts = urlsplit(url)
    try:
        with socket.create_connection((parts.hostname or "localhost", parts.port or 5432), timeout=timeout):
            return True
    except OSError:
        return False


def connect(url: str) -> Any:
    try:
        import psycopg
    except ModuleNotFoundError:
        import psycopg2

        connection = psycopg2.connect(url)
        connection.autocommit = True
        return connection
    return psycopg.connect(url, autocommit=True)


def query_rows(url: str, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    connection = connect(url)
    try:
        cursor = connection.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        cursor.close()
        return rows
    finally:
        connection.close()


def create_copy(url: str, name: str) -> str:
    if not name.startswith(COPY_PREFIX) or not name.replace("_", "").isalnum():
        raise ValueError(f"golden database copies must be named {COPY_PREFIX}<alnum>: {name!r}")
    maintenance = connect(with_database(url, "postgres"))
    try:
        cursor = maintenance.cursor()
        cursor.execute(f'DROP DATABASE IF EXISTS "{name}"')
        cursor.execute(f'CREATE DATABASE "{name}" TEMPLATE "{TEMPLATE_DATABASE}"')
        cursor.close()
    finally:
        maintenance.close()
    return with_database(url, name)


def drop_copy(url: str, name: str) -> None:
    if not name.startswith(COPY_PREFIX):
        raise ValueError(f"refusing to drop a database that is not a golden copy: {name!r}")
    maintenance = connect(with_database(url, "postgres"))
    try:
        cursor = maintenance.cursor()
        cursor.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        cursor.close()
    finally:
        maintenance.close()


@contextmanager
def database_copy(url: str, name: str) -> Iterator[str]:
    copy_url = create_copy(url, name)
    try:
        yield copy_url
    finally:
        drop_copy(url, name)


@contextmanager
def patched_environ(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _reject_non_loopback(address: Any) -> None:
    if isinstance(address, tuple) and address and address[0] not in LOOPBACK_HOSTS:
        raise RuntimeError(f"golden harness may only open loopback connections, not {address!r}")


@contextmanager
def loopback_only() -> Iterator[None]:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def connect(sock: socket.socket, address: Any) -> None:
        _reject_non_loopback(address)
        return original_connect(sock, address)

    def connect_ex(sock: socket.socket, address: Any) -> int:
        _reject_non_loopback(address)
        return original_connect_ex(sock, address)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
