from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import psycopg2
import pytest

from src.integrations.prepare_job_repository import STALE_EXHAUSTED_ERROR, PrepareJobRepository

pytestmark = pytest.mark.integration

LEGACY_PREPARE_JOBS_DDL = """
CREATE TABLE prepare_jobs (
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


class DsnPrepareJobRepository(PrepareJobRepository):
    def __init__(self, dsn: str, **settings: Any) -> None:
        values = {"prepare_job_stale_seconds": 900, "prepare_job_max_attempts": 3}
        values.update(settings)
        super().__init__(settings=SimpleNamespace(**values))
        self.dsn = dsn

    def _build_postgres_connection_params(self) -> dict[str, Any]:
        return {"dsn": self.dsn}


def _execute(dsn: str, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    connection = psycopg2.connect(dsn=dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall() if cursor.description else []
        connection.commit()
        return rows
    finally:
        connection.close()


def _job(dsn: str, job_id: int) -> dict[str, Any]:
    columns = (
        "status",
        "attempt_count",
        "claim_generation",
        "worker_id",
        "raw_revision",
        "pending_refresh",
        "error",
        "heartbeat_at",
        "next_attempt_at",
        "finished_at",
    )
    row = _execute(dsn, f"SELECT {', '.join(columns)} FROM prepare_jobs WHERE id = %s", (job_id,))[0]
    job = dict(zip(columns, row, strict=True))
    delay = _execute(
        dsn,
        "SELECT EXTRACT(EPOCH FROM (next_attempt_at - NOW())) FROM prepare_jobs WHERE id = %s",
        (job_id,),
    )[0][0]
    job["next_attempt_in_seconds"] = float(delay) if delay is not None else None
    return job


def _make_due(dsn: str, job_id: int) -> None:
    _execute(dsn, "UPDATE prepare_jobs SET next_attempt_at = NOW() - INTERVAL '1 second' WHERE id = %s", (job_id,))


def _token(job: dict[str, Any]) -> dict[str, Any]:
    return {"job_id": job["job_id"], "worker_id": job["worker_id"], "claim_generation": job["claim_generation"]}


@pytest.fixture
def dsn(test_database_url: str) -> str:
    _execute(test_database_url, "DROP TABLE IF EXISTS prepare_jobs")
    yield test_database_url
    _execute(test_database_url, "DROP TABLE IF EXISTS prepare_jobs")


@pytest.fixture
def repository(dsn: str) -> DsnPrepareJobRepository:
    repository = DsnPrepareJobRepository(dsn)
    repository.ensure_schema()
    return repository


def test_ensure_schema_upgrades_legacy_table_idempotently(dsn):
    _execute(dsn, LEGACY_PREPARE_JOBS_DDL)
    _execute(dsn, "INSERT INTO prepare_jobs (mode, target_date, status) VALUES ('auto', '2026-04-07', 'done')")

    repository = DsnPrepareJobRepository(dsn)
    repository.ensure_schema()
    repository.ensure_schema()

    columns = {
        row[0]
        for row in _execute(dsn, "SELECT column_name FROM information_schema.columns WHERE table_name = 'prepare_jobs'")
    }
    assert {"claim_generation", "heartbeat_at", "next_attempt_at", "raw_revision", "pending_refresh"} <= columns
    legacy = _execute(dsn, "SELECT claim_generation, raw_revision, pending_refresh FROM prepare_jobs")[0]
    assert legacy == (0, 0, False)


def test_wrong_claim_token_cannot_complete_or_fail(repository, dsn):
    job_id = repository.enqueue_prepare_job(target_date="2026-04-07", raw_revision=1)["job_id"]
    claimed = repository.claim_prepare_job(worker_id="w1")
    assert claimed["job_id"] == job_id
    assert claimed["claim_generation"] == 1
    token = _token(claimed)

    assert repository.complete_prepare_job(**{**token, "claim_generation": 0}) is False
    assert repository.complete_prepare_job(**{**token, "worker_id": "w2"}) is False
    assert repository.fail_prepare_job(**{**token, "claim_generation": 2}, error="x") is False
    assert repository.heartbeat_prepare_job(**{**token, "worker_id": "w2"}) is False
    assert _job(dsn, job_id)["status"] == "processing"

    assert repository.complete_prepare_job(**token, result={"ok": 1}) is True
    assert _job(dsn, job_id)["status"] == "done"
    assert repository.complete_prepare_job(**token) is False


def test_stale_reclaim_fences_out_zombie_worker(repository, dsn):
    job_id = repository.enqueue_prepare_job(target_date="2026-04-07")["job_id"]
    zombie = repository.claim_prepare_job(worker_id="w1")
    _execute(
        dsn,
        "UPDATE prepare_jobs SET claimed_at = NOW() - INTERVAL '2 hours', heartbeat_at = NOW() - INTERVAL '1 hour'"
        " WHERE id = %s",
        (job_id,),
    )

    owner = repository.claim_prepare_job(worker_id="w1")
    assert owner["job_id"] == job_id
    assert owner["claim_generation"] == zombie["claim_generation"] + 1

    assert repository.heartbeat_prepare_job(**_token(zombie)) is False
    assert repository.complete_prepare_job(**_token(zombie)) is False
    assert repository.fail_prepare_job(**_token(zombie), error="late") is False
    assert _job(dsn, job_id)["status"] == "processing"
    assert repository.complete_prepare_job(**_token(owner)) is True
    assert _job(dsn, job_id)["status"] == "done"


def test_stale_reset_uses_heartbeat_not_claimed_at(repository, dsn):
    job_id = repository.enqueue_prepare_job(target_date="2026-04-07")["job_id"]
    claimed = repository.claim_prepare_job(worker_id="w1")
    _execute(dsn, "UPDATE prepare_jobs SET claimed_at = NOW() - INTERVAL '2 hours' WHERE id = %s", (job_id,))

    assert repository.heartbeat_prepare_job(**_token(claimed)) is True
    assert repository.reset_stale_prepare_jobs(stale_seconds=900) == 0
    assert _job(dsn, job_id)["status"] == "processing"

    _execute(dsn, "UPDATE prepare_jobs SET heartbeat_at = NOW() - INTERVAL '20 minutes' WHERE id = %s", (job_id,))
    assert repository.reset_stale_prepare_jobs(stale_seconds=900) == 1
    job = _job(dsn, job_id)
    assert job["status"] == "pending"
    assert job["worker_id"] is None
    assert job["heartbeat_at"] is None
    assert job["attempt_count"] == 1


def test_stale_reset_falls_back_to_claimed_at_for_legacy_rows(repository, dsn):
    job_id = repository.enqueue_prepare_job(target_date="2026-04-07")["job_id"]
    repository.claim_prepare_job(worker_id="w1")
    _execute(
        dsn,
        "UPDATE prepare_jobs SET heartbeat_at = NULL, claimed_at = NOW() - INTERVAL '20 minutes' WHERE id = %s",
        (job_id,),
    )
    assert repository.reset_stale_prepare_jobs(stale_seconds=900) == 1
    assert _job(dsn, job_id)["status"] == "pending"


def test_stale_reset_closes_job_that_exhausted_attempts(repository, dsn):
    job_id = repository.enqueue_prepare_job(target_date="2026-04-07")["job_id"]
    repository.claim_prepare_job(worker_id="w1")
    _execute(
        dsn,
        "UPDATE prepare_jobs SET attempt_count = 3, heartbeat_at = NOW() - INTERVAL '1 hour' WHERE id = %s",
        (job_id,),
    )
    assert repository.reset_stale_prepare_jobs(stale_seconds=900) == 1
    job = _job(dsn, job_id)
    assert job["status"] == "failed"
    assert job["error"] == STALE_EXHAUSTED_ERROR
    assert repository.claim_prepare_job(worker_id="w1") is None


def test_failures_back_off_then_fail_and_requeue_resets(repository, dsn):
    job_id = repository.enqueue_prepare_job(target_date="2026-04-07")["job_id"]

    first = repository.claim_prepare_job(worker_id="w1")
    assert first["attempt_count"] == 1
    assert repository.fail_prepare_job(**_token(first), error="boom 1") is True
    job = _job(dsn, job_id)
    assert job["status"] == "pending"
    assert job["error"] == "boom 1"
    assert 55 <= job["next_attempt_in_seconds"] <= 61
    assert repository.claim_prepare_job(worker_id="w1") is None

    _make_due(dsn, job_id)
    second = repository.claim_prepare_job(worker_id="w1")
    assert (second["attempt_count"], second["claim_generation"]) == (2, 2)
    assert _job(dsn, job_id)["next_attempt_at"] is None
    assert repository.fail_prepare_job(**_token(second), error="boom 2") is True
    assert 115 <= _job(dsn, job_id)["next_attempt_in_seconds"] <= 121

    _make_due(dsn, job_id)
    third = repository.claim_prepare_job(worker_id="w1")
    assert third["attempt_count"] == 3
    assert repository.fail_prepare_job(**_token(third), error="boom 3") is True
    job = _job(dsn, job_id)
    assert job["status"] == "failed"
    assert job["next_attempt_at"] is None
    assert job["finished_at"] is not None
    assert repository.claim_prepare_job(worker_id="w1") is None

    requeued = repository.requeue_failed_prepare_jobs(mode="auto", dry_run=False)
    assert requeued == [{"id": job_id, "target_date": "2026-04-07", "attempt_count": 3}]
    job = _job(dsn, job_id)
    assert (job["status"], job["attempt_count"], job["next_attempt_at"]) == ("pending", 0, None)
    assert repository.claim_prepare_job(worker_id="w1")["attempt_count"] == 1


def test_recollect_refreshes_done_job_only_for_newer_revision(repository, dsn):
    job_id = repository.enqueue_prepare_job(target_date="2026-04-07", raw_revision=1)["job_id"]
    first = repository.claim_prepare_job(worker_id="w1")
    assert first["raw_revision"] == 1
    assert repository.complete_prepare_job(**_token(first)) is True

    same = repository.enqueue_prepare_job(target_date="2026-04-07", raw_revision=1)
    assert same["status"] == "done"
    assert repository.enqueue_prepare_job(target_date="2026-04-07")["status"] == "done"
    job = _job(dsn, job_id)
    assert job["worker_id"] == "w1"
    assert job["finished_at"] is not None

    refreshed = repository.enqueue_prepare_job(target_date="2026-04-07", raw_revision=2)
    assert (refreshed["job_id"], refreshed["status"], refreshed["raw_revision"]) == (job_id, "pending", 2)
    job = _job(dsn, job_id)
    assert (job["attempt_count"], job["worker_id"], job["finished_at"]) == (0, None, None)


def test_recollect_during_processing_requeues_once_after_complete(repository, dsn):
    job_id = repository.enqueue_prepare_job(target_date="2026-04-07", raw_revision=1)["job_id"]
    first = repository.claim_prepare_job(worker_id="w1")

    during = repository.enqueue_prepare_job(target_date="2026-04-07", raw_revision=2)
    assert (during["status"], during["pending_refresh"], during["raw_revision"]) == ("processing", True, 2)
    older = repository.enqueue_prepare_job(target_date="2026-04-07", raw_revision=1)
    assert (older["pending_refresh"], older["raw_revision"]) == (True, 2)
    assert repository.heartbeat_prepare_job(**_token(first)) is True

    assert repository.complete_prepare_job(**_token(first)) is True
    job = _job(dsn, job_id)
    assert (job["status"], job["pending_refresh"], job["attempt_count"]) == ("pending", False, 0)

    second = repository.claim_prepare_job(worker_id="w1")
    assert second["claim_generation"] == 2
    assert repository.complete_prepare_job(**_token(second)) is True
    assert _job(dsn, job_id)["status"] == "done"


def test_recollect_during_processing_gives_fresh_attempts_on_failure(repository, dsn):
    job_id = repository.enqueue_prepare_job(target_date="2026-04-07", raw_revision=1)["job_id"]
    _execute(dsn, "UPDATE prepare_jobs SET attempt_count = 2 WHERE id = %s", (job_id,))
    claimed = repository.claim_prepare_job(worker_id="w1")
    assert claimed["attempt_count"] == 3
    repository.enqueue_prepare_job(target_date="2026-04-07", raw_revision=2)

    assert repository.fail_prepare_job(**_token(claimed), error="boom") is True
    job = _job(dsn, job_id)
    assert (job["status"], job["attempt_count"], job["next_attempt_at"], job["pending_refresh"]) == (
        "pending",
        0,
        None,
        False,
    )


def test_enqueue_failed_job_resets_attempts(repository, dsn):
    job_id = repository.enqueue_prepare_job(target_date="2026-04-07")["job_id"]
    _execute(dsn, "UPDATE prepare_jobs SET status = 'failed', attempt_count = 3, error = 'x' WHERE id = %s", (job_id,))

    result = repository.enqueue_prepare_job(target_date="2026-04-07", raw_revision=1)
    assert result["status"] == "pending"
    job = _job(dsn, job_id)
    assert (job["attempt_count"], job["error"], job["raw_revision"]) == (0, None, 1)


def test_requeue_force_sets_payload_flag_and_plain_requeue_clears_it(repository, dsn):
    job_id = repository.enqueue_prepare_job(target_date="2026-04-07", payload={"note": "x"})["job_id"]
    _execute(dsn, "UPDATE prepare_jobs SET status = 'failed', attempt_count = 3 WHERE id = %s", (job_id,))

    repository.requeue_failed_prepare_jobs(mode="auto", dry_run=False, force=True)
    claimed = repository.claim_prepare_job(worker_id="w1")
    assert claimed["payload"] == {"note": "x", "force": True}

    _execute(dsn, "UPDATE prepare_jobs SET status = 'failed' WHERE id = %s", (job_id,))
    repository.requeue_failed_prepare_jobs(mode="auto", dry_run=False)
    assert repository.claim_prepare_job(worker_id="w1")["payload"] == {"note": "x"}


def test_collect_enqueue_keeps_requeued_force_until_job_completes(repository, dsn):
    job_id = repository.enqueue_prepare_job(target_date="2026-04-07", payload={"note": "x"})["job_id"]
    _execute(dsn, "UPDATE prepare_jobs SET status = 'failed', attempt_count = 3 WHERE id = %s", (job_id,))
    repository.requeue_failed_prepare_jobs(mode="auto", dry_run=False, force=True)

    repository.enqueue_prepare_job(target_date="2026-04-07", payload={"collected": 2}, raw_revision=1)
    claimed = repository.claim_prepare_job(worker_id="w1")
    assert claimed["payload"] == {"note": "x", "force": True, "collected": 2}

    assert repository.complete_prepare_job(job_id=job_id, worker_id="w1", claim_generation=claimed["claim_generation"])
    payload = _execute(dsn, "SELECT payload FROM prepare_jobs WHERE id = %s", (job_id,))[0][0]
    assert payload == {"note": "x", "collected": 2}
