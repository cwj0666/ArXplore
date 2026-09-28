from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from psycopg import AsyncConnection, sql
from psycopg.rows import dict_row

from arxplore.db.pool import ConnectionSource, use_connection
from arxplore.db.repositories.raw import raw_jsonb
from arxplore.text import strip_nul_and_surrogates, strip_nul_and_surrogates_deep

SCHEDULER_LOCK_KEY = 0x41525850_53434844
ABANDONED_ERROR = "the scheduler stopped before this run finished"

START_SCHEDULED_SQL = """
INSERT INTO scheduler_runs (job, trigger, scheduled_for, params)
VALUES (%s, 'scheduled', %s, %s)
ON CONFLICT (job, scheduled_for) WHERE trigger = 'scheduled' DO NOTHING
RETURNING id
"""

START_MANUAL_SQL = """
INSERT INTO scheduler_runs (job, trigger, scheduled_for, params)
VALUES (%s, 'manual', NULL, %s)
RETURNING id
"""

RECORD_ATTEMPT_SQL = "UPDATE scheduler_runs SET attempts = %s, error = %s WHERE id = %s"

FINISH_SQL = """
UPDATE scheduler_runs
SET status = %s, result = %s, error = %s, attempts = %s, finished_at = NOW()
WHERE id = %s
"""

HAS_SCHEDULED_SQL = """
SELECT EXISTS (
    SELECT 1 FROM scheduler_runs WHERE job = %s AND trigger = 'scheduled' AND scheduled_for = %s
) AS present
"""

ABANDON_RUNNING_SQL = """
UPDATE scheduler_runs
SET status = 'abandoned', error = COALESCE(error, %s), finished_at = NOW()
WHERE status = 'running' AND trigger = 'scheduled'
"""

LIST_RECENT_SQL = """
SELECT id, job, trigger, scheduled_for, status, params, result, error, attempts, started_at, finished_at
FROM scheduler_runs
{where}
ORDER BY started_at DESC, id DESC
LIMIT %s
"""

TRY_LOCK_SQL = "SELECT pg_try_advisory_lock(%s) AS locked"
UNLOCK_SQL = "SELECT pg_advisory_unlock(%s)"


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ABANDONED = "abandoned"


class RunTrigger(StrEnum):
    SCHEDULED = "scheduled"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True, kw_only=True)
class SchedulerRun:
    run_id: int
    job: str
    trigger: RunTrigger
    scheduled_for: datetime | None
    status: RunStatus
    params: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None
    attempts: int
    started_at: datetime
    finished_at: datetime | None


def _json(value: Mapping[str, Any] | None) -> Any:
    return None if value is None else raw_jsonb(strip_nul_and_surrogates_deep(dict(value)))


def _text(value: str | None) -> str | None:
    return None if value is None else strip_nul_and_surrogates(value)


async def start_scheduled_run(
    db: ConnectionSource, *, job: str, scheduled_for: datetime, params: Mapping[str, Any]
) -> int | None:
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(START_SCHEDULED_SQL, (job, scheduled_for, _json(params)))
        row = await cursor.fetchone()
    return int(row["id"]) if row else None


async def start_manual_run(db: ConnectionSource, *, job: str, params: Mapping[str, Any]) -> int:
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(START_MANUAL_SQL, (job, _json(params)))
        row = await cursor.fetchone()
    if row is None:
        raise RuntimeError("scheduler_runs insert returned no row")
    return int(row["id"])


async def record_attempt(db: ConnectionSource, run_id: int, *, attempts: int, error: str | None) -> None:
    async with use_connection(db) as connection, connection.cursor() as cursor:
        await cursor.execute(RECORD_ATTEMPT_SQL, (attempts, _text(error), run_id))


async def finish_run(
    db: ConnectionSource,
    run_id: int,
    *,
    status: RunStatus,
    result: Mapping[str, Any] | None,
    error: str | None,
    attempts: int,
) -> None:
    async with use_connection(db) as connection, connection.cursor() as cursor:
        await cursor.execute(FINISH_SQL, (status.value, _json(result), _text(error), attempts, run_id))


async def has_scheduled_run(db: ConnectionSource, *, job: str, scheduled_for: datetime) -> bool:
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(HAS_SCHEDULED_SQL, (job, scheduled_for))
        row = await cursor.fetchone()
    return bool(row and row["present"])


async def abandon_running_runs(db: ConnectionSource) -> int:
    async with use_connection(db) as connection, connection.cursor() as cursor:
        await cursor.execute(ABANDON_RUNNING_SQL, (ABANDONED_ERROR,))
        return cursor.rowcount


def _scheduler_run(row: Mapping[str, Any]) -> SchedulerRun:
    return SchedulerRun(
        run_id=int(row["id"]),
        job=row["job"],
        trigger=RunTrigger(row["trigger"]),
        scheduled_for=row["scheduled_for"],
        status=RunStatus(row["status"]),
        params=row["params"] or {},
        result=row["result"],
        error=row["error"],
        attempts=int(row["attempts"] or 0),
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


async def list_recent_runs(db: ConnectionSource, *, job: str | None = None, limit: int = 20) -> list[SchedulerRun]:
    where = sql.SQL("WHERE job = %s") if job else sql.SQL("")
    params: list[Any] = [job] if job else []
    params.append(max(1, limit))
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(sql.SQL(LIST_RECENT_SQL).format(where=where), params)
        rows = await cursor.fetchall()
    return [_scheduler_run(row) for row in rows]


async def try_acquire_scheduler_lock(connection: AsyncConnection[Any]) -> bool:
    async with connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(TRY_LOCK_SQL, (SCHEDULER_LOCK_KEY,))
        row = await cursor.fetchone()
    return bool(row and row["locked"])


async def release_scheduler_lock(connection: AsyncConnection[Any]) -> None:
    if connection.closed or connection.broken:
        return
    await connection.execute(UNLOCK_SQL, (SCHEDULER_LOCK_KEY,))
