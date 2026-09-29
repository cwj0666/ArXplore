from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import date
from typing import Any

import psycopg
import pytest
from psycopg import sql

from arxplore.db.repositories import jobs
from arxplore.pipeline.requeue import requeue_failed_jobs

pytestmark = pytest.mark.integration

APRIL_6 = date(2026, 4, 6)
APRIL_7 = date(2026, 4, 7)
APRIL_8 = date(2026, 4, 8)
APRIL_9 = date(2026, 4, 9)

JOB_COLUMNS = """
    SELECT mode, target_date, status, attempt_count, payload, error, worker_id,
           next_attempt_at, claimed_at, heartbeat_at, finished_at, pending_refresh
    FROM prepare_jobs ORDER BY mode, target_date
"""


async def _job(pool: Any, probe: Any, target_date: date, *, status: str, mode: str = "auto", **fields: Any) -> int:
    job_id = (await jobs.enqueue(pool, target_date=target_date, mode=mode, raw_revision=1)).job_id
    assignments = {"status": status, **fields}
    columns = ", ".join(f"{name} = %s" for name in assignments)
    await probe.execute(f"UPDATE prepare_jobs SET {columns} WHERE id = %s", (*assignments.values(), job_id))
    return job_id


@pytest.fixture
async def queue(pool: Any, probe: Any) -> dict[str, int]:
    failed_fields = {
        "attempt_count": 3,
        "error": "all 2 paper(s) failed",
        "worker_id": "w1",
        "payload": '{"force": true}',
    }
    return {
        "old_failed": await _job(pool, probe, APRIL_6, status="failed", **failed_fields),
        "failed": await _job(pool, probe, APRIL_7, status="failed", **failed_fields),
        "done": await _job(pool, probe, APRIL_8, status="done", attempt_count=1),
        "busy": await _job(pool, probe, APRIL_9, status="processing", attempt_count=1, worker_id="w2"),
        "backfill_failed": await _job(pool, probe, APRIL_7, status="failed", mode="backfill", attempt_count=3),
    }


@pytest.fixture
async def listener(pipeline_database: str) -> AsyncIterator[psycopg.AsyncConnection[Any]]:
    async with await psycopg.AsyncConnection.connect(pipeline_database, autocommit=True) as connection:
        await connection.execute(sql.SQL("LISTEN {}").format(sql.Identifier(jobs.JOB_CHANNEL)))
        yield connection


async def _notifications(connection: psycopg.AsyncConnection[Any]) -> list[str]:
    async with aclosing(connection.notifies(timeout=1.0)) as stream:
        return [notify.payload async for notify in stream]


async def test_the_default_run_lists_failed_jobs_without_touching_them(pool, probe, queue, listener) -> None:
    before = await probe.rows(JOB_COLUMNS)

    result = (await requeue_failed_jobs(pool)).as_dict()

    assert result == {
        "stage": "requeue_failed",
        "mode": "auto",
        "since": None,
        "force": False,
        "applied": False,
        "job_count": 2,
        "jobs": [
            {"job_id": queue["old_failed"], "target_date": "2026-04-06", "attempt_count": 3},
            {"job_id": queue["failed"], "target_date": "2026-04-07", "attempt_count": 3},
        ],
    }
    assert await probe.rows(JOB_COLUMNS) == before
    assert await _notifications(listener) == []


async def test_apply_reopens_only_failed_jobs_with_the_attempts_reset(pool, probe, queue, listener) -> None:
    result = (await requeue_failed_jobs(pool, since=APRIL_7, apply=True)).as_dict()

    assert result["applied"] is True
    assert result["since"] == "2026-04-07"
    assert result["jobs"] == [{"job_id": queue["failed"], "target_date": "2026-04-07", "attempt_count": 3}]
    rows = {(row["mode"], row["target_date"]): row for row in await probe.rows(JOB_COLUMNS)}
    reopened = rows[("auto", APRIL_7)]
    assert (reopened["status"], reopened["attempt_count"], reopened["payload"], reopened["error"]) == (
        "pending",
        0,
        {},
        None,
    )
    assert reopened["worker_id"] is None
    assert [reopened[column] for column in ("next_attempt_at", "claimed_at", "heartbeat_at", "finished_at")] == [
        None
    ] * 4
    assert reopened["pending_refresh"] is False
    assert rows[("auto", APRIL_6)]["status"] == "failed"
    assert rows[("auto", APRIL_8)]["status"] == "done"
    assert rows[("auto", APRIL_9)]["status"] == "processing"
    assert rows[("backfill", APRIL_7)]["status"] == "failed"
    assert await _notifications(listener) == ["auto:2026-04-07"]


async def test_force_sets_the_flag_on_the_reopened_jobs(pool, probe, queue) -> None:
    await probe.execute("UPDATE prepare_jobs SET payload = '{}'::jsonb")

    await requeue_failed_jobs(pool, force=True, apply=True)

    payloads = {row["target_date"]: row["payload"] for row in await probe.rows(JOB_COLUMNS) if row["mode"] == "auto"}
    assert payloads == {APRIL_6: {"force": True}, APRIL_7: {"force": True}, APRIL_8: {}, APRIL_9: {}}


async def test_the_backfill_mode_is_requeued_separately(pool, probe, queue) -> None:
    result = (await requeue_failed_jobs(pool, mode="backfill", apply=True)).as_dict()

    assert result["jobs"] == [{"job_id": queue["backfill_failed"], "target_date": "2026-04-07", "attempt_count": 3}]
    statuses = {(row["mode"], row["target_date"]): row["status"] for row in await probe.rows(JOB_COLUMNS)}
    assert statuses[("backfill", APRIL_7)] == "pending"
    assert statuses[("auto", APRIL_7)] == "failed"


async def test_nothing_to_requeue_is_an_empty_list(pool) -> None:
    result = (await requeue_failed_jobs(pool, apply=True)).as_dict()
    assert (result["job_count"], result["jobs"]) == (0, [])
