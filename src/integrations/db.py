"""프로세스 단위 PostgreSQL 커넥션 풀.

- 풀은 접속 파라미터별로 첫 `get_connection()` 호출 시 만든다. gunicorn은 워커를 fork한 뒤 요청을 받으므로
  워커마다 자기 풀을 갖는다. fork된 자식은 부모에게서 물려받은 풀을 쓰지도 닫지도 않는다
  (닫으면 같은 소켓으로 Terminate를 보내 부모 세션을 끊는다). 그래서 참조만 남겨 두고 새 풀을 만든다.
- 스레드 안전하다(gthread 워커). 풀이 가득 차면 `POSTGRES_POOL_TIMEOUT`초까지 기다린 뒤 `PoolError`를 낸다.
- 프로세스당 최대 연결 수는 `POSTGRES_POOL_MAX`(기본 8). 서버 `max_connections`는 워커 수 x 이 값 이상이어야 한다.
- LISTEN/NOTIFY처럼 세션을 오래 붙잡는 용도는 풀을 쓰지 말고 전용 연결을 연다.

`psycopg2.pool.ThreadedConnectionPool`은 반납된 연결을 `minconn`개까지만 보관하고(나머지는 닫음)
고갈 시 기다리지 않고 바로 예외를 내서, 요청마다 새 연결을 여는 것과 차이가 작다. 그래서 직접 구현한다.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import psycopg2
import psycopg2.extensions
from psycopg2.pool import PoolError

logger = logging.getLogger(__name__)

DEFAULT_POOL_MAX = 8
DEFAULT_ACQUIRE_TIMEOUT_SECONDS = 30.0
IDLE_PING_AFTER_SECONDS = 60.0

__all__ = [
    "ConnectionPool",
    "PoolError",
    "close_all_pools",
    "get_connection",
    "get_pool",
    "resolve_pool_max",
]


def _positive_number(value: Any, cast: type) -> Any | None:
    try:
        number = cast(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def resolve_pool_max(settings: Any = None) -> int:
    for candidate in (getattr(settings, "postgres_pool_max", None), os.environ.get("POSTGRES_POOL_MAX")):
        number = _positive_number(candidate, int)
        if number is not None:
            return number
    return DEFAULT_POOL_MAX


def resolve_acquire_timeout(settings: Any = None) -> float:
    for candidate in (getattr(settings, "postgres_pool_timeout", None), os.environ.get("POSTGRES_POOL_TIMEOUT")):
        number = _positive_number(candidate, float)
        if number is not None:
            return number
    return DEFAULT_ACQUIRE_TIMEOUT_SECONDS


class ConnectionPool:
    """최대 `maxconn`개의 연결을 LIFO로 재사용하는 스레드 안전 풀."""

    def __init__(self, params: Mapping[str, Any], *, maxconn: int, acquire_timeout: float) -> None:
        self._params = dict(params)
        self.maxconn = max(1, int(maxconn))
        self.acquire_timeout = float(acquire_timeout)
        self._slots = threading.BoundedSemaphore(self.maxconn)
        self._lock = threading.Lock()
        self._idle: list[tuple[Any, float]] = []
        self._closed = False

    @property
    def idle_count(self) -> int:
        with self._lock:
            return len(self._idle)

    def acquire(self) -> Any:
        if self._closed:
            raise PoolError("connection pool is closed")
        if not self._slots.acquire(timeout=self.acquire_timeout):
            raise PoolError(f"no PostgreSQL connection available within {self.acquire_timeout:g}s")
        try:
            return self._checkout()
        except BaseException:
            self._slots.release()
            raise

    def release(self, connection: Any, *, discard: bool = False) -> None:
        try:
            if discard or self._closed or connection.closed or not self._reset(connection):
                self._close_quietly(connection)
                return
            with self._lock:
                self._idle.append((connection, time.monotonic()))
        finally:
            self._slots.release()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            idle, self._idle = self._idle, []
        for connection, _ in idle:
            self._close_quietly(connection)

    def _checkout(self) -> Any:
        while True:
            with self._lock:
                entry = self._idle.pop() if self._idle else None
            if entry is None:
                return psycopg2.connect(**self._params)
            connection, returned_at = entry
            if connection.closed:
                continue
            if time.monotonic() - returned_at < IDLE_PING_AFTER_SECONDS or self._ping(connection):
                return connection
            self._close_quietly(connection)

    @staticmethod
    def _ping(connection: Any) -> bool:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            connection.rollback()
            return True
        except psycopg2.Error:
            return False

    @staticmethod
    def _reset(connection: Any) -> bool:
        try:
            status = connection.info.transaction_status
            if status == psycopg2.extensions.TRANSACTION_STATUS_UNKNOWN:
                return False
            if status != psycopg2.extensions.TRANSACTION_STATUS_IDLE:
                connection.rollback()
            if connection.autocommit:
                connection.autocommit = False
            return True
        except psycopg2.Error:
            return False

    @staticmethod
    def _close_quietly(connection: Any) -> None:
        try:
            connection.close()
        except Exception:  # noqa: BLE001
            logger.debug("failed to close pooled connection", exc_info=True)


_registry_lock = threading.Lock()
_pools: dict[tuple[tuple[str, str], ...], ConnectionPool] = {}
_owner_pid = os.getpid()
_inherited_pools: list[ConnectionPool] = []


def _forget_inherited_pools() -> None:
    global _registry_lock, _owner_pid
    _registry_lock = threading.Lock()
    _inherited_pools.extend(_pools.values())
    _pools.clear()
    _owner_pid = os.getpid()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_inherited_pools)


def _pool_key(params: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in params.items()))


def get_pool(params: Mapping[str, Any], *, settings: Any = None) -> ConnectionPool:
    if os.getpid() != _owner_pid:
        _forget_inherited_pools()
    key = _pool_key(params)
    with _registry_lock:
        pool = _pools.get(key)
        if pool is None:
            pool = ConnectionPool(
                params,
                maxconn=resolve_pool_max(settings),
                acquire_timeout=resolve_acquire_timeout(settings),
            )
            _pools[key] = pool
        return pool


@contextmanager
def get_connection(params: Mapping[str, Any], *, settings: Any = None) -> Iterator[Any]:
    """풀에서 연결을 빌려 준다. 정상 종료 시 commit, 예외 시 rollback 후 풀에 반납한다.

    연결 계열 오류(OperationalError/InterfaceError)가 나면 그 연결은 반납하지 않고 닫는다.
    """
    pool = get_pool(params, settings=settings)
    connection = pool.acquire()
    discard = False
    try:
        yield connection
        connection.commit()
    except BaseException as exc:
        discard = isinstance(exc, (psycopg2.OperationalError, psycopg2.InterfaceError))
        try:
            if not connection.closed:
                connection.rollback()
        except psycopg2.Error:
            discard = True
        raise
    finally:
        pool.release(connection, discard=discard)


def close_all_pools() -> None:
    with _registry_lock:
        pools = list(_pools.values())
        _pools.clear()
    for pool in pools:
        pool.close()
