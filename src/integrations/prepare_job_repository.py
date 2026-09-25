from __future__ import annotations

import select
from datetime import date as date_cls
from typing import Any

import psycopg2
from psycopg2.extras import Json

from src.integrations.db import get_connection
from src.shared import AppSettings, build_postgres_connection_params, get_settings

RETRY_BACKOFF_BASE_SECONDS = 60
RETRY_BACKOFF_MAX_SECONDS = 3600
STALE_EXHAUSTED_ERROR = "stale claim reset after max attempts"
_ENQUEUE_KEEP_STATE = (
    "(prepare_jobs.status = 'processing' "
    "OR (prepare_jobs.status = 'done' AND EXCLUDED.raw_revision <= prepare_jobs.raw_revision))"
)


def compute_retry_backoff_seconds(
    attempt_count: int,
    *,
    base_seconds: int = RETRY_BACKOFF_BASE_SECONDS,
    max_seconds: int = RETRY_BACKOFF_MAX_SECONDS,
) -> int:
    """attempt_count번째 시도가 실패한 뒤 다음 시도까지 기다릴 시간(초). 60s * 2^(n-1), 상한 1시간."""
    exponent = max(0, int(attempt_count) - 1)
    if exponent >= 32:
        return int(max_seconds)
    return int(min(max_seconds, base_seconds * (2**exponent)))


def resolve_failure_transition(
    *,
    attempt_count: int,
    pending_refresh: bool,
    max_attempts: int,
) -> dict[str, Any]:
    """실패한 claim을 재시도(pending)할지 종료(failed)할지 결정한다.

    처리 중 새 raw revision이 들어온 잡(pending_refresh)은 새 입력이므로 시도 횟수를 초기화하고 즉시 재시도한다.
    """
    if pending_refresh:
        return {"status": "pending", "attempt_count": 0, "backoff_seconds": None}
    if int(attempt_count) < int(max_attempts):
        return {
            "status": "pending",
            "attempt_count": int(attempt_count),
            "backoff_seconds": compute_retry_backoff_seconds(attempt_count),
        }
    return {"status": "failed", "attempt_count": int(attempt_count), "backoff_seconds": None}


class PrepareJobRepository:
    """prepare job queue를 PostgreSQL에 저장하고 소비하는 진입점"""

    channel_name = "arxplore_prepare_jobs"

    def __init__(self, *, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def max_attempts(self) -> int:
        return max(1, int(self.settings.prepare_job_max_attempts))

    def enqueue_prepare_job(
        self,
        *,
        target_date: str,
        mode: str = "auto",
        source: str = "collect",
        payload: dict[str, Any] | None = None,
        raw_revision: int | None = None,
    ) -> dict[str, Any]:
        """날짜 단위 prepare 작업을 큐에 추가한다.

        raw_revision이 기존 잡보다 크면 done 잡은 pending으로 되돌리고, processing 잡은 pending_refresh로 표시한다.
        failed 잡은 새 입력이므로 시도 횟수를 초기화해 pending으로 되돌린다.
        payload는 기존 payload에 병합되므로 requeue가 넣은 force는 작업이 완료될 때까지 유지된다.
        """
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
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
                """,
                (mode, target_date, source, Json(payload or {}), raw_revision),
            )
            row = cursor.fetchone()
            cursor.execute("SELECT pg_notify(%s, %s)", (self.channel_name, f"{mode}:{target_date}"))

        return {
            "enqueued": bool(row[2]) if row is not None else False,
            "job_id": int(row[0]) if row is not None else None,
            "mode": mode,
            "date": target_date,
            "status": row[1] if row is not None else None,
            "raw_revision": int(row[3] or 0) if row is not None else None,
            "pending_refresh": bool(row[4]) if row is not None else False,
        }

    def claim_prepare_job(
        self,
        *,
        mode: str = "auto",
        worker_id: str = "local_prepare_worker",
    ) -> dict[str, Any] | None:
        """재시도 대기 시간이 지난 pending 작업 1건을 선점하고 claim 토큰(job_id, worker_id, claim_generation)을 발급한다"""
        self.reset_stale_prepare_jobs(mode=mode)
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
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
                """,
                (mode, worker_id),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return {
            "job_id": row[0],
            "mode": row[1],
            "date": row[2],
            "source": row[3],
            "payload": row[4] or {},
            "status": row[5],
            "attempt_count": int(row[6] or 0),
            "worker_id": row[7],
            "created_at": row[8],
            "claimed_at": row[9],
            "updated_at": row[10],
            "claim_generation": int(row[11] or 0),
            "raw_revision": int(row[12] or 0),
        }

    def heartbeat_prepare_job(
        self,
        *,
        job_id: int,
        worker_id: str,
        claim_generation: int,
    ) -> bool:
        """claim이 아직 유효하면 heartbeat_at을 갱신하고 True를 반환한다"""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE prepare_jobs
                SET heartbeat_at = NOW()
                WHERE id = %s
                  AND worker_id = %s
                  AND claim_generation = %s
                  AND status = 'processing'
                """,
                (job_id, worker_id, claim_generation),
            )
            return cursor.rowcount > 0

    def reset_stale_prepare_jobs(
        self,
        *,
        mode: str = "auto",
        stale_seconds: int | None = None,
    ) -> int:
        """heartbeat가 끊긴 processing 작업을 pending으로 되돌리고, 시도 횟수를 소진한 작업은 failed로 닫는다"""
        normalized_stale_seconds = int(
            stale_seconds if stale_seconds is not None else self.settings.prepare_job_stale_seconds
        )
        if normalized_stale_seconds <= 0:
            return 0

        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
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
                """,
                (
                    self.max_attempts,
                    self.max_attempts,
                    STALE_EXHAUSTED_ERROR,
                    self.max_attempts,
                    mode,
                    normalized_stale_seconds,
                ),
            )
            reset_rows = cursor.fetchall()
            for target_date, status in reset_rows:
                if status == "pending":
                    cursor.execute("SELECT pg_notify(%s, %s)", (self.channel_name, f"{mode}:{target_date}"))

        return len(reset_rows)

    def complete_prepare_job(
        self,
        *,
        job_id: int,
        worker_id: str,
        claim_generation: int,
        result: dict[str, Any] | None = None,
    ) -> bool:
        """claim이 유효할 때만 작업을 완료한다. 처리 중 새 raw revision이 들어왔으면 pending으로 1회 되돌린다.

        claim을 잃었으면(다른 worker가 재선점했거나 stale reset됨) 아무것도 바꾸지 않고 False를 반환한다.
        """
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
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
                """,
                (Json(result or {}), job_id, worker_id, claim_generation),
            )
            row = cursor.fetchone()
            if row is not None and row[2] == "pending":
                cursor.execute("SELECT pg_notify(%s, %s)", (self.channel_name, f"{row[0]}:{row[1]}"))
        return row is not None

    def fail_prepare_job(
        self,
        *,
        job_id: int,
        worker_id: str,
        claim_generation: int,
        error: str,
    ) -> bool:
        """claim이 유효할 때만 실패를 기록한다.

        시도 횟수가 max_attempts 미만이면 backoff 후 재시도되도록 pending으로, 소진했으면 failed로 전환한다.
        claim을 잃었으면 아무것도 바꾸지 않고 False를 반환한다.
        """
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT attempt_count, pending_refresh
                FROM prepare_jobs
                WHERE id = %s
                  AND worker_id = %s
                  AND claim_generation = %s
                  AND status = 'processing'
                FOR UPDATE
                """,
                (job_id, worker_id, claim_generation),
            )
            row = cursor.fetchone()
            if row is None:
                return False

            transition = resolve_failure_transition(
                attempt_count=int(row[0] or 0),
                pending_refresh=bool(row[1]),
                max_attempts=self.max_attempts,
            )
            is_final = transition["status"] == "failed"
            cursor.execute(
                """
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
                """,
                (
                    transition["status"],
                    transition["attempt_count"],
                    transition["backoff_seconds"],
                    transition["backoff_seconds"],
                    error,
                    is_final,
                    is_final,
                    is_final,
                    is_final,
                    job_id,
                ),
            )
            updated = cursor.fetchone()
            if updated is not None and transition["status"] == "pending" and transition["backoff_seconds"] is None:
                cursor.execute("SELECT pg_notify(%s, %s)", (self.channel_name, f"{updated[0]}:{updated[1]}"))
        return True

    def requeue_failed_prepare_jobs(
        self,
        *,
        mode: str,
        since_date: str | None = None,
        dry_run: bool = True,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        """failed 상태 작업을 조회하고, dry_run이 아니면 시도 횟수를 초기화해 pending으로 되돌린다.

        force면 payload에 "force": true를 넣어 재처리 시 본문 source 순위·content_hash 검사를 건너뛰게 하고,
        아니면 payload의 force 키를 지운다.
        반환값은 대상(또는 실제로 전환된) 작업의 id, target_date, 전환 전 attempt_count 목록이다.
        """
        payload_sql = "j.payload || '{\"force\": true}'::jsonb" if force else "j.payload - 'force'"
        normalized_since = date_cls.fromisoformat(since_date.strip()).isoformat() if since_date else None
        where_sql = "WHERE mode = %s AND status = 'failed'"
        params: list[Any] = [mode]
        if normalized_since:
            where_sql += " AND target_date >= %s"
            params.append(normalized_since)

        with self._connection() as connection, connection.cursor() as cursor:
            if dry_run:
                cursor.execute(
                    f"""
                    SELECT id, target_date, attempt_count
                    FROM prepare_jobs
                    {where_sql}
                    ORDER BY target_date ASC, id ASC
                    """,
                    tuple(params),
                )
                rows = cursor.fetchall()
            else:
                cursor.execute(
                    f"""
                    WITH targets AS (
                        SELECT id, attempt_count AS previous_attempt_count
                        FROM prepare_jobs
                        {where_sql}
                        FOR UPDATE
                    )
                    UPDATE prepare_jobs j
                    SET
                        status = 'pending',
                        payload = {payload_sql},
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
                    RETURNING j.id, j.target_date, targets.previous_attempt_count
                    """,
                    tuple(params),
                )
                rows = sorted(cursor.fetchall(), key=lambda row: (str(row[1]), row[0]))
                for row in rows:
                    cursor.execute("SELECT pg_notify(%s, %s)", (self.channel_name, f"{mode}:{row[1]}"))

        return [
            {"id": int(row[0]), "target_date": str(row[1]), "attempt_count": int(row[2] or 0)}
            for row in rows
        ]

    def wait_for_prepare_job(
        self,
        *,
        timeout_seconds: float = 120.0,
    ) -> bool:
        """새 prepare 작업 알림이 올 때까지 기다린다. LISTEN은 세션에 묶이므로 풀이 아닌 전용 연결을 쓴다."""
        connection = psycopg2.connect(**self._build_postgres_connection_params())
        connection.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"LISTEN {self.channel_name};")
            ready, _, _ = select.select([connection], [], [], max(0.0, float(timeout_seconds)))
            if not ready:
                return False
            connection.poll()
            while connection.notifies:
                connection.notifies.pop(0)
            return True
        finally:
            connection.close()

    def _connection(self):
        return get_connection(self._build_postgres_connection_params(), settings=self.settings)

    def ensure_schema(self) -> None:
        """prepare_jobs 테이블·컬럼·인덱스를 멱등하게 생성한다. 프로세스 시작/마이그레이션 시 1회만 호출한다."""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS prepare_jobs (
                    id BIGSERIAL PRIMARY KEY,
                    mode TEXT NOT NULL,
                    target_date TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'collect',
                    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                    result JSONB NOT NULL DEFAULT '{}'::jsonb,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT,
                    error TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    claimed_at TIMESTAMPTZ,
                    finished_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT uq_prepare_jobs_mode_target_date UNIQUE (mode, target_date)
                )
                """
            )
            cursor.execute(
                """
                ALTER TABLE prepare_jobs
                    ADD COLUMN IF NOT EXISTS claim_generation INTEGER NOT NULL DEFAULT 0,
                    ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ,
                    ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ,
                    ADD COLUMN IF NOT EXISTS raw_revision INTEGER NOT NULL DEFAULT 0,
                    ADD COLUMN IF NOT EXISTS pending_refresh BOOLEAN NOT NULL DEFAULT FALSE
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_prepare_jobs_mode_status_target_date
                ON prepare_jobs (mode, status, target_date)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_prepare_jobs_status_updated_at
                ON prepare_jobs (status, updated_at DESC)
                """
            )

    def _build_postgres_connection_params(self) -> dict[str, Any]:
        return build_postgres_connection_params(self.settings)
