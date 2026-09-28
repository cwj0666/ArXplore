from datetime import date
from typing import Any

from psycopg import AsyncCursor, sql
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb

from arxplore.db.pool import AnyPool, ConnectionSource, use_connection
from arxplore.domain.jobs import (
    DEFAULT_JOB_MODE,
    DEFAULT_JOB_SOURCE,
    DEFAULT_WORKER_ID,
    STALE_EXHAUSTED_ERROR,
    ClaimedJob,
    ClaimToken,
    EnqueueResult,
    JobStatus,
    QueuePolicy,
    RequeuedJob,
    job_notification_payload,
    resolve_failure_transition,
)

JOB_CHANNEL = "arxplore_prepare_jobs"

_ENQUEUE_KEEP_STATE = (
    "(prepare_jobs.status = 'processing' "
    "OR (prepare_jobs.status = 'done' AND EXCLUDED.raw_revision <= prepare_jobs.raw_revision))"
)

ENQUEUE_SQL = f"""
INSERT INTO prepare_jobs (
    mode, target_date, source, payload, status, attempt_count, raw_revision, created_at, updated_at
)
VALUES (%s, %s, %s, %s, 'pending', 0, COALESCE(%s, 0), NOW(), NOW())
ON CONFLICT (mode, target_date)
DO UPDATE SET
    source = EXCLUDED.source,
    payload = prepare_jobs.payload || EXCLUDED.payload,
    status = CASE
        WHEN {_ENQUEUE_KEEP_STATE} THEN prepare_jobs.status
        ELSE 'pending'
    END,
    pending_refresh = CASE
        WHEN prepare_jobs.status = 'processing'
            THEN prepare_jobs.pending_refresh OR EXCLUDED.raw_revision > prepare_jobs.raw_revision
        ELSE FALSE
    END,
    attempt_count = CASE
        WHEN {_ENQUEUE_KEEP_STATE} THEN prepare_jobs.attempt_count
        WHEN prepare_jobs.status IN ('failed', 'done') THEN 0
        ELSE prepare_jobs.attempt_count
    END,
    next_attempt_at = CASE
        WHEN {_ENQUEUE_KEEP_STATE} THEN prepare_jobs.next_attempt_at
        WHEN prepare_jobs.status IN ('failed', 'done') THEN NULL
        ELSE prepare_jobs.next_attempt_at
    END,
    worker_id = CASE WHEN {_ENQUEUE_KEEP_STATE} THEN prepare_jobs.worker_id ELSE NULL END,
    error = CASE WHEN {_ENQUEUE_KEEP_STATE} THEN prepare_jobs.error ELSE NULL END,
    claimed_at = CASE WHEN {_ENQUEUE_KEEP_STATE} THEN prepare_jobs.claimed_at ELSE NULL END,
    finished_at = CASE WHEN {_ENQUEUE_KEEP_STATE} THEN prepare_jobs.finished_at ELSE NULL END,
    raw_revision = GREATEST(prepare_jobs.raw_revision, EXCLUDED.raw_revision),
    updated_at = NOW()
RETURNING id, status, created_at = updated_at AS inserted, raw_revision, pending_refresh
"""

NOTIFY_SQL = "SELECT pg_notify(%s, %s)"

CLAIM_SQL = """
WITH next_job AS (
    SELECT id
    FROM prepare_jobs
    WHERE mode = %s
      AND status = 'pending'
      AND (next_attempt_at IS NULL OR next_attempt_at <= NOW())
    ORDER BY target_date ASC, id ASC
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
UPDATE prepare_jobs j
SET
    status = 'processing',
    worker_id = %s,
    claimed_at = NOW(),
    heartbeat_at = NOW(),
    next_attempt_at = NULL,
    updated_at = NOW(),
    attempt_count = j.attempt_count + 1,
    claim_generation = j.claim_generation + 1
FROM next_job
WHERE j.id = next_job.id
RETURNING
    j.id,
    j.mode,
    j.target_date,
    j.source,
    j.payload,
    j.status,
    j.attempt_count,
    j.worker_id,
    j.created_at,
    j.claimed_at,
    j.updated_at,
    j.claim_generation,
    j.raw_revision
"""

HEARTBEAT_SQL = """
UPDATE prepare_jobs
SET heartbeat_at = NOW()
WHERE id = %s
  AND worker_id = %s
  AND claim_generation = %s
  AND status = 'processing'
"""

RESET_STALE_SQL = """
UPDATE prepare_jobs
SET
    status = CASE
        WHEN pending_refresh OR attempt_count < %s THEN 'pending'
        ELSE 'failed'
    END,
    error = CASE
        WHEN pending_refresh OR attempt_count < %s THEN NULL
        ELSE %s
    END,
    finished_at = CASE
        WHEN pending_refresh OR attempt_count < %s THEN NULL
        ELSE NOW()
    END,
    attempt_count = CASE WHEN pending_refresh THEN 0 ELSE attempt_count END,
    pending_refresh = FALSE,
    worker_id = NULL,
    claimed_at = NULL,
    heartbeat_at = NULL,
    next_attempt_at = NULL,
    updated_at = NOW()
WHERE mode = %s
  AND status = 'processing'
  AND COALESCE(heartbeat_at, claimed_at) IS NOT NULL
  AND COALESCE(heartbeat_at, claimed_at) < NOW() - (%s * INTERVAL '1 second')
RETURNING target_date, status
"""

COMPLETE_SQL = """
UPDATE prepare_jobs
SET
    status = CASE WHEN pending_refresh THEN 'pending' ELSE 'done' END,
    attempt_count = CASE WHEN pending_refresh THEN 0 ELSE attempt_count END,
    worker_id = CASE WHEN pending_refresh THEN NULL ELSE worker_id END,
    claimed_at = CASE WHEN pending_refresh THEN NULL ELSE claimed_at END,
    finished_at = CASE WHEN pending_refresh THEN NULL ELSE NOW() END,
    pending_refresh = FALSE,
    heartbeat_at = NULL,
    next_attempt_at = NULL,
    payload = payload - 'force',
    result = %s,
    error = NULL,
    updated_at = NOW()
WHERE id = %s
  AND worker_id = %s
  AND claim_generation = %s
  AND status = 'processing'
RETURNING mode, target_date, status
"""

LOCK_CLAIMED_JOB_SQL = """
SELECT attempt_count, pending_refresh
FROM prepare_jobs
WHERE id = %s
  AND worker_id = %s
  AND claim_generation = %s
  AND status = 'processing'
FOR UPDATE
"""

FAIL_SQL = """
UPDATE prepare_jobs
SET
    status = %s,
    attempt_count = %s,
    next_attempt_at = CASE
        WHEN %s::integer IS NULL THEN NULL
        ELSE NOW() + (%s::integer * INTERVAL '1 second')
    END,
    error = %s,
    pending_refresh = FALSE,
    worker_id = CASE WHEN %s THEN worker_id ELSE NULL END,
    claimed_at = CASE WHEN %s THEN claimed_at ELSE NULL END,
    finished_at = CASE WHEN %s THEN NOW() ELSE NULL END,
    heartbeat_at = NULL,
    payload = CASE WHEN %s THEN payload - 'force' ELSE payload END,
    updated_at = NOW()
WHERE id = %s
RETURNING mode, target_date
"""

REQUEUE_PREVIEW_SQL = """
SELECT id, target_date, attempt_count
FROM prepare_jobs
{where}
ORDER BY target_date ASC, id ASC
"""

REQUEUE_APPLY_SQL = """
WITH targets AS (
    SELECT id, attempt_count AS previous_attempt_count
    FROM prepare_jobs
    {where}
    FOR UPDATE
)
UPDATE prepare_jobs j
SET
    status = 'pending',
    payload = {payload},
    attempt_count = 0,
    next_attempt_at = NULL,
    pending_refresh = FALSE,
    worker_id = NULL,
    error = NULL,
    claimed_at = NULL,
    heartbeat_at = NULL,
    finished_at = NULL,
    updated_at = NOW()
FROM targets
WHERE j.id = targets.id
RETURNING j.id, j.target_date, targets.previous_attempt_count AS attempt_count
"""

REQUEUE_FORCE_PAYLOAD = "j.payload || '{\"force\": true}'::jsonb"
REQUEUE_PLAIN_PAYLOAD = "j.payload - 'force'"


async def _notify(cursor: AsyncCursor[Any], mode: str, target_date: date) -> None:
    await cursor.execute(NOTIFY_SQL, (JOB_CHANNEL, job_notification_payload(mode, target_date)))


def _claimed_job(row: DictRow) -> ClaimedJob:
    return ClaimedJob(
        job_id=row["id"],
        mode=row["mode"],
        target_date=row["target_date"],
        source=row["source"],
        payload=row["payload"] or {},
        status=JobStatus(row["status"]),
        attempt_count=int(row["attempt_count"] or 0),
        worker_id=row["worker_id"],
        created_at=row["created_at"],
        claimed_at=row["claimed_at"],
        updated_at=row["updated_at"],
        claim_generation=int(row["claim_generation"] or 0),
        raw_revision=int(row["raw_revision"] or 0),
    )


async def enqueue(
    db: ConnectionSource,
    *,
    target_date: date,
    mode: str = DEFAULT_JOB_MODE,
    source: str = DEFAULT_JOB_SOURCE,
    payload: dict[str, Any] | None = None,
    raw_revision: int | None = None,
) -> EnqueueResult:
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(ENQUEUE_SQL, (mode, target_date, source, Jsonb(payload or {}), raw_revision))
        row = await cursor.fetchone()
        await _notify(cursor, mode, target_date)
    if row is None:
        raise RuntimeError("prepare_jobs upsert returned no row")
    return EnqueueResult(
        enqueued=bool(row["inserted"]),
        job_id=int(row["id"]),
        mode=mode,
        target_date=target_date,
        status=JobStatus(row["status"]),
        raw_revision=int(row["raw_revision"] or 0),
        pending_refresh=bool(row["pending_refresh"]),
    )


async def reset_stale(
    db: ConnectionSource,
    *,
    policy: QueuePolicy,
    mode: str = DEFAULT_JOB_MODE,
    stale_seconds: int | None = None,
) -> int:
    threshold = int(stale_seconds if stale_seconds is not None else policy.stale_seconds)
    if threshold <= 0:
        return 0
    attempts = policy.max_attempts
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(
            RESET_STALE_SQL,
            (attempts, attempts, STALE_EXHAUSTED_ERROR, attempts, mode, threshold),
        )
        reset_rows = await cursor.fetchall()
        for row in reset_rows:
            if row["status"] == JobStatus.PENDING:
                await _notify(cursor, mode, row["target_date"])
    return len(reset_rows)


async def claim(
    pool: AnyPool,
    *,
    policy: QueuePolicy,
    mode: str = DEFAULT_JOB_MODE,
    worker_id: str = DEFAULT_WORKER_ID,
) -> ClaimedJob | None:
    await reset_stale(pool, policy=policy, mode=mode)
    async with pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(CLAIM_SQL, (mode, worker_id))
        row = await cursor.fetchone()
    return _claimed_job(row) if row is not None else None


async def heartbeat(db: ConnectionSource, token: ClaimToken) -> bool:
    async with use_connection(db) as connection, connection.cursor() as cursor:
        await cursor.execute(HEARTBEAT_SQL, (token.job_id, token.worker_id, token.claim_generation))
        return cursor.rowcount > 0


async def complete(db: ConnectionSource, token: ClaimToken, *, result: dict[str, Any] | None = None) -> bool:
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(
            COMPLETE_SQL,
            (Jsonb(result or {}), token.job_id, token.worker_id, token.claim_generation),
        )
        row = await cursor.fetchone()
        if row is not None and row["status"] == JobStatus.PENDING:
            await _notify(cursor, row["mode"], row["target_date"])
    return row is not None


async def fail(db: ConnectionSource, token: ClaimToken, *, policy: QueuePolicy, error: str) -> bool:
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(LOCK_CLAIMED_JOB_SQL, (token.job_id, token.worker_id, token.claim_generation))
        row = await cursor.fetchone()
        if row is None:
            return False
        transition = resolve_failure_transition(
            attempt_count=int(row["attempt_count"] or 0),
            pending_refresh=bool(row["pending_refresh"]),
            max_attempts=policy.max_attempts,
        )
        is_final = transition.is_final
        await cursor.execute(
            FAIL_SQL,
            (
                transition.status.value,
                transition.attempt_count,
                transition.backoff_seconds,
                transition.backoff_seconds,
                error,
                is_final,
                is_final,
                is_final,
                is_final,
                token.job_id,
            ),
        )
        updated = await cursor.fetchone()
        if updated is not None and transition.retries_now:
            await _notify(cursor, updated["mode"], updated["target_date"])
    return True


async def requeue_failed(
    db: ConnectionSource,
    *,
    mode: str,
    since_date: date | None = None,
    dry_run: bool = True,
    force: bool = False,
) -> list[RequeuedJob]:
    where: sql.Composable = sql.SQL("WHERE mode = %s AND status = 'failed'")
    params: list[Any] = [mode]
    if since_date:
        where = sql.Composed([where, sql.SQL(" AND target_date >= %s")])
        params.append(since_date)
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        if dry_run:
            await cursor.execute(sql.SQL(REQUEUE_PREVIEW_SQL).format(where=where), params)
            rows = await cursor.fetchall()
        else:
            payload = sql.SQL(REQUEUE_FORCE_PAYLOAD if force else REQUEUE_PLAIN_PAYLOAD)
            await cursor.execute(sql.SQL(REQUEUE_APPLY_SQL).format(where=where, payload=payload), params)
            rows = sorted(await cursor.fetchall(), key=lambda row: (row["target_date"], row["id"]))
            for row in rows:
                await _notify(cursor, mode, row["target_date"])
    return [
        RequeuedJob(job_id=int(row["id"]), target_date=row["target_date"], attempt_count=int(row["attempt_count"] or 0))
        for row in rows
    ]
