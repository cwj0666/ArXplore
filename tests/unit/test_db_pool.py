from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import psycopg2
import psycopg2.extensions
import pytest

from src.integrations import db
from src.integrations import paper_repository as paper_repository_module
from src.integrations import prepare_job_repository as prepare_job_repository_module
from src.integrations import vector_repository as vector_repository_module

IDLE = psycopg2.extensions.TRANSACTION_STATUS_IDLE
INTRANS = psycopg2.extensions.TRANSACTION_STATUS_INTRANS
UNKNOWN = psycopg2.extensions.TRANSACTION_STATUS_UNKNOWN


class FakeCursor:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    def execute(self, sql, params=None):
        if self.connection.ping_fails:
            raise psycopg2.OperationalError("server closed the connection")
        self.connection.executed.append(sql)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnection:
    def __init__(self, serial: int) -> None:
        self.serial = serial
        self.closed = 0
        self.autocommit = False
        self.commits = 0
        self.rollbacks = 0
        self.ping_fails = False
        self.executed: list[str] = []
        self.info = SimpleNamespace(transaction_status=IDLE)

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1
        self.info.transaction_status = IDLE

    def close(self):
        self.closed = 1


@pytest.fixture
def connections(monkeypatch):
    created: list[FakeConnection] = []
    lock = threading.Lock()

    def fake_connect(**params):
        with lock:
            connection = FakeConnection(len(created))
            created.append(connection)
        return connection

    monkeypatch.setattr(db.psycopg2, "connect", fake_connect)
    monkeypatch.delenv("POSTGRES_POOL_MAX", raising=False)
    monkeypatch.delenv("POSTGRES_POOL_TIMEOUT", raising=False)
    db.close_all_pools()
    yield created
    db.close_all_pools()


PARAMS = {"dbname": "arxplore", "host": "localhost", "port": 5432}


def test_connection_is_reused_and_committed(connections):
    with db.get_connection(PARAMS) as first:
        pass
    with db.get_connection(PARAMS) as second:
        pass

    assert first is second
    assert len(connections) == 1
    assert first.commits == 2
    assert first.closed == 0
    assert db.get_pool(PARAMS).idle_count == 1


def test_exception_rolls_back_and_returns_connection(connections):
    with pytest.raises(ValueError):
        with db.get_connection(PARAMS) as connection:
            raise ValueError("boom")

    assert connection.rollbacks == 1
    assert connection.commits == 0
    assert connection.closed == 0
    with db.get_connection(PARAMS) as again:
        assert again is connection


@pytest.mark.parametrize("error", [psycopg2.OperationalError, psycopg2.InterfaceError])
def test_connection_errors_discard_the_connection(connections, error):
    with pytest.raises(error):
        with db.get_connection(PARAMS) as connection:
            raise error("server closed the connection unexpectedly")

    assert connection.closed == 1
    assert db.get_pool(PARAMS).idle_count == 0
    with db.get_connection(PARAMS) as replacement:
        assert replacement is not connection


def test_closed_idle_connection_is_skipped(connections):
    with db.get_connection(PARAMS) as connection:
        pass
    connection.closed = 1
    with db.get_connection(PARAMS) as replacement:
        assert replacement is not connection
    assert len(connections) == 2


def test_long_idle_connection_is_pinged_and_replaced_when_dead(connections, monkeypatch):
    monkeypatch.setattr(db, "IDLE_PING_AFTER_SECONDS", 0.0)
    with db.get_connection(PARAMS) as connection:
        pass

    with db.get_connection(PARAMS) as same:
        assert same is connection
    assert "SELECT 1" in connection.executed

    connection.ping_fails = True
    with db.get_connection(PARAMS) as replacement:
        assert replacement is not connection
    assert connection.closed == 1


def test_release_resets_open_transaction_and_autocommit(connections):
    pool = db.get_pool(PARAMS)
    connection = pool.acquire()
    connection.info.transaction_status = INTRANS
    connection.autocommit = True
    pool.release(connection)

    assert connection.rollbacks == 1
    assert connection.autocommit is False
    assert pool.idle_count == 1


def test_release_closes_connection_in_unknown_state(connections):
    pool = db.get_pool(PARAMS)
    connection = pool.acquire()
    connection.info.transaction_status = UNKNOWN
    pool.release(connection)

    assert connection.closed == 1
    assert pool.idle_count == 0


def test_pool_is_bounded_and_times_out(connections):
    settings = SimpleNamespace(postgres_pool_max=1, postgres_pool_timeout=0.05)
    pool = db.get_pool(PARAMS, settings=settings)
    held = pool.acquire()

    with pytest.raises(db.PoolError):
        pool.acquire()

    pool.release(held)
    with db.get_connection(PARAMS) as connection:
        assert connection is held


def test_waiting_thread_gets_released_connection(connections):
    pool = db.get_pool(PARAMS, settings=SimpleNamespace(postgres_pool_max=1, postgres_pool_timeout=2))
    held = pool.acquire()
    acquired: list[FakeConnection] = []

    waiter = threading.Thread(target=lambda: acquired.append(pool.acquire()))
    waiter.start()
    time.sleep(0.05)
    assert acquired == []
    pool.release(held)
    waiter.join(timeout=2)

    assert acquired == [held]
    pool.release(held)


def test_concurrent_use_never_exceeds_max(connections):
    settings = SimpleNamespace(postgres_pool_max=3, postgres_pool_timeout=5)
    db.get_pool(PARAMS, settings=settings)
    in_use = 0
    peak = 0
    lock = threading.Lock()

    def work():
        nonlocal in_use, peak
        for _ in range(30):
            with db.get_connection(PARAMS):
                with lock:
                    in_use += 1
                    peak = max(peak, in_use)
                time.sleep(0.0005)
                with lock:
                    in_use -= 1

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert peak <= 3
    assert len(connections) <= 3


def test_pools_are_keyed_by_connection_params(connections):
    assert db.get_pool(PARAMS) is db.get_pool(dict(reversed(list(PARAMS.items()))))
    assert db.get_pool(PARAMS) is not db.get_pool({**PARAMS, "dbname": "other"})


def test_child_process_does_not_reuse_or_close_parent_pool(connections):
    with db.get_connection(PARAMS) as parent_connection:
        pass
    parent_pool = db.get_pool(PARAMS)

    db._forget_inherited_pools()

    child_pool = db.get_pool(PARAMS)
    assert child_pool is not parent_pool
    assert parent_pool in db._inherited_pools
    assert parent_connection.closed == 0
    with db.get_connection(PARAMS) as child_connection:
        assert child_connection is not parent_connection
    db._inherited_pools.remove(parent_pool)


def test_pool_max_resolution(monkeypatch):
    monkeypatch.delenv("POSTGRES_POOL_MAX", raising=False)
    assert db.resolve_pool_max(object()) == db.DEFAULT_POOL_MAX
    monkeypatch.setenv("POSTGRES_POOL_MAX", "3")
    assert db.resolve_pool_max(None) == 3
    assert db.resolve_pool_max(SimpleNamespace(postgres_pool_max=5)) == 5
    monkeypatch.setenv("POSTGRES_POOL_MAX", "zero")
    assert db.resolve_pool_max(SimpleNamespace(postgres_pool_max=None)) == db.DEFAULT_POOL_MAX


@pytest.mark.parametrize(
    "module,cls_name",
    [
        (paper_repository_module, "PaperRepository"),
        (vector_repository_module, "VectorRepository"),
        (prepare_job_repository_module, "PrepareJobRepository"),
    ],
)
def test_repositories_borrow_from_the_pool(monkeypatch, module, cls_name):
    calls = []
    settings = SimpleNamespace()

    def fake_get_connection(params, *, settings=None):
        calls.append((params, settings))
        return "pooled-context"

    monkeypatch.setattr(module, "get_connection", fake_get_connection)
    repository = getattr(module, cls_name)(settings=settings)
    repository._build_postgres_connection_params = lambda: {"dsn": "postgresql://x"}

    assert repository._connection() == "pooled-context"
    assert calls == [({"dsn": "postgresql://x"}, settings)]
