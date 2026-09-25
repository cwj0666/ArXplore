from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from src.integrations import prepare_job_repository as prepare_job_repository_module
from src.integrations.prepare_job_repository import (
    STALE_EXHAUSTED_ERROR,
    PrepareJobRepository,
    compute_retry_backoff_seconds,
    resolve_failure_transition,
)


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.split())


class RecordingCursor:
    def __init__(
        self,
        rows: list[tuple[Any, ...]] | None = None,
        *,
        fetchone_results: list[tuple[Any, ...] | None] | None = None,
        rowcount: int = 0,
    ) -> None:
        self.executed: list[tuple[str, Any]] = []
        self.rows = rows or []
        self.fetchone_results = list(fetchone_results) if fetchone_results is not None else None
        self.rowcount = rowcount

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return list(self.rows)

    def fetchone(self):
        if self.fetchone_results is not None:
            return self.fetchone_results.pop(0) if self.fetchone_results else None
        return self.rows[0] if self.rows else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class RecordingConnection:
    def __init__(self, cursor: RecordingCursor) -> None:
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def _settings(**overrides: Any) -> SimpleNamespace:
    values = {"prepare_job_stale_seconds": 900, "prepare_job_max_attempts": 3}
    values.update(overrides)
    return SimpleNamespace(**values)


def _repository_with_cursor(cursor: RecordingCursor, **settings_overrides: Any) -> PrepareJobRepository:
    repository = PrepareJobRepository(settings=_settings(**settings_overrides))

    @contextmanager
    def fake_connection():
        yield RecordingConnection(cursor)

    repository._connection = fake_connection  # type: ignore[method-assign]
    return repository


def _notifies(cursor: RecordingCursor) -> list[Any]:
    return [params for sql, params in cursor.executed if "pg_notify" in sql]


def test_constructor_runs_no_ddl(monkeypatch):
    def fail_connect(**kwargs):
        raise AssertionError("PrepareJobRepository() must not open a DB connection")

    monkeypatch.setattr(prepare_job_repository_module.psycopg2, "connect", fail_connect)
    PrepareJobRepository(settings=object())
    assert not hasattr(PrepareJobRepository, "_ensure_schema")


def test_ensure_schema_creates_prepare_jobs():
    cursor = RecordingCursor()
    _repository_with_cursor(cursor).ensure_schema()
    statements = [_normalize_sql(sql) for sql, _ in cursor.executed]
    assert any("CREATE TABLE IF NOT EXISTS prepare_jobs" in statement for statement in statements)
    assert not any("DROP" in statement.upper() for statement in statements)


def test_requeue_dry_run_only_selects():
    cursor = RecordingCursor(rows=[(11, "2026-04-07", 1), (12, "2026-04-08", 2)])
    rows = _repository_with_cursor(cursor).requeue_failed_prepare_jobs(mode="auto", since_date="2026-04-07")

    assert rows == [
        {"id": 11, "target_date": "2026-04-07", "attempt_count": 1},
        {"id": 12, "target_date": "2026-04-08", "attempt_count": 2},
    ]
    assert len(cursor.executed) == 1
    sql, params = cursor.executed[0]
    normalized = _normalize_sql(sql)
    assert normalized.startswith("SELECT id, target_date, attempt_count FROM prepare_jobs")
    assert "WHERE mode = %s AND status = 'failed' AND target_date >= %s" in normalized
    assert "UPDATE" not in normalized.upper()
    assert params == ("auto", "2026-04-07")


def test_requeue_apply_resets_fields_and_notifies():
    cursor = RecordingCursor(rows=[(12, "2026-04-08", 3), (11, "2026-04-07", 3)])
    rows = _repository_with_cursor(cursor).requeue_failed_prepare_jobs(
        mode="auto", since_date="2026-04-07", dry_run=False
    )

    assert [row["target_date"] for row in rows] == ["2026-04-07", "2026-04-08"]
    assert [row["attempt_count"] for row in rows] == [3, 3]
    sql, params = cursor.executed[0]
    normalized = _normalize_sql(sql)
    assert "SELECT id, attempt_count AS previous_attempt_count FROM prepare_jobs" in normalized
    assert "FOR UPDATE" in normalized
    assert "UPDATE prepare_jobs j SET" in normalized
    for clause in (
        "status = 'pending'",
        "attempt_count = 0",
        "next_attempt_at = NULL",
        "pending_refresh = FALSE",
        "worker_id = NULL",
        "error = NULL",
        "claimed_at = NULL",
        "heartbeat_at = NULL",
        "finished_at = NULL",
        "updated_at = NOW()",
    ):
        assert clause in normalized
    assert "WHERE mode = %s AND status = 'failed' AND target_date >= %s" in normalized
    assert "RETURNING j.id, j.target_date, targets.previous_attempt_count" in normalized
    assert params == ("auto", "2026-04-07")

    assert _notifies(cursor) == [
        ("arxplore_prepare_jobs", "auto:2026-04-07"),
        ("arxplore_prepare_jobs", "auto:2026-04-08"),
    ]


def test_requeue_apply_sets_or_clears_force_in_payload():
    cursor = RecordingCursor(rows=[(11, "2026-04-07", 3)])
    _repository_with_cursor(cursor).requeue_failed_prepare_jobs(mode="auto", dry_run=False, force=True)
    forced_sql, forced_params = cursor.executed[0]
    assert """payload = j.payload || '{"force": true}'::jsonb""" in _normalize_sql(forced_sql)
    assert forced_params == ("auto",)

    cursor = RecordingCursor(rows=[(11, "2026-04-07", 3)])
    _repository_with_cursor(cursor).requeue_failed_prepare_jobs(mode="auto", dry_run=False)
    assert "payload = j.payload - 'force'" in _normalize_sql(cursor.executed[0][0])


def test_requeue_without_since_has_no_date_predicate():
    cursor = RecordingCursor()
    assert _repository_with_cursor(cursor).requeue_failed_prepare_jobs(mode="auto") == []
    sql, params = cursor.executed[0]
    assert "target_date >=" not in sql
    assert params == ("auto",)


def test_requeue_rejects_invalid_since_date():
    with pytest.raises(ValueError):
        _repository_with_cursor(RecordingCursor()).requeue_failed_prepare_jobs(mode="auto", since_date="2026/04/07")


def test_ensure_schema_adds_queue_columns_idempotently():
    cursor = RecordingCursor()
    _repository_with_cursor(cursor).ensure_schema()
    statements = [_normalize_sql(sql) for sql, _ in cursor.executed]
    alter = next(statement for statement in statements if statement.startswith("ALTER TABLE prepare_jobs"))
    for column in (
        "ADD COLUMN IF NOT EXISTS claim_generation INTEGER NOT NULL DEFAULT 0",
        "ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ",
        "ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ",
        "ADD COLUMN IF NOT EXISTS raw_revision INTEGER NOT NULL DEFAULT 0",
        "ADD COLUMN IF NOT EXISTS pending_refresh BOOLEAN NOT NULL DEFAULT FALSE",
    ):
        assert column in alter
    create_index = statements.index(
        next(statement for statement in statements if "idx_prepare_jobs_mode_status_target_date" in statement)
    )
    assert statements.index(alter) < create_index


@pytest.mark.parametrize(
    ("attempt_count", "expected"),
    [(0, 60), (1, 60), (2, 120), (3, 240), (6, 1920), (7, 3600), (50, 3600), (10_000, 3600)],
)
def test_compute_retry_backoff_seconds(attempt_count, expected):
    assert compute_retry_backoff_seconds(attempt_count) == expected


def test_resolve_failure_transition_retries_with_backoff_below_max():
    assert resolve_failure_transition(attempt_count=1, pending_refresh=False, max_attempts=3) == {
        "status": "pending",
        "attempt_count": 1,
        "backoff_seconds": 60,
    }
    assert resolve_failure_transition(attempt_count=2, pending_refresh=False, max_attempts=3)["backoff_seconds"] == 120


def test_resolve_failure_transition_fails_when_attempts_exhausted():
    assert resolve_failure_transition(attempt_count=3, pending_refresh=False, max_attempts=3) == {
        "status": "failed",
        "attempt_count": 3,
        "backoff_seconds": None,
    }


def test_resolve_failure_transition_refresh_resets_attempts():
    assert resolve_failure_transition(attempt_count=3, pending_refresh=True, max_attempts=3) == {
        "status": "pending",
        "attempt_count": 0,
        "backoff_seconds": None,
    }


def test_max_attempts_has_floor_of_one():
    assert PrepareJobRepository(settings=_settings(prepare_job_max_attempts=0)).max_attempts == 1
    assert PrepareJobRepository(settings=_settings(prepare_job_max_attempts=5)).max_attempts == 5


def test_enqueue_passes_raw_revision_and_applies_refresh_rules():
    cursor = RecordingCursor(fetchone_results=[(7, "pending", False, 2, False)])
    result = _repository_with_cursor(cursor).enqueue_prepare_job(target_date="2026-04-07", raw_revision=2)

    assert result == {
        "enqueued": False,
        "job_id": 7,
        "mode": "auto",
        "date": "2026-04-07",
        "status": "pending",
        "raw_revision": 2,
        "pending_refresh": False,
    }
    sql, params = cursor.executed[0]
    normalized = _normalize_sql(sql)
    assert params[0:2] == ("auto", "2026-04-07")
    assert params[-1] == 2
    assert "COALESCE(%s, 0)" in normalized
    keep = (
        "(prepare_jobs.status = 'processing' OR (prepare_jobs.status = 'done' "
        "AND EXCLUDED.raw_revision <= prepare_jobs.raw_revision))"
    )
    assert f"status = CASE WHEN {keep} THEN prepare_jobs.status ELSE 'pending' END" in normalized
    assert (
        "pending_refresh = CASE WHEN prepare_jobs.status = 'processing' THEN prepare_jobs.pending_refresh "
        "OR EXCLUDED.raw_revision > prepare_jobs.raw_revision ELSE FALSE END"
    ) in normalized
    assert (
        f"attempt_count = CASE WHEN {keep} THEN prepare_jobs.attempt_count "
        "WHEN prepare_jobs.status IN ('failed', 'done') THEN 0 ELSE prepare_jobs.attempt_count END"
    ) in normalized
    assert "raw_revision = GREATEST(prepare_jobs.raw_revision, EXCLUDED.raw_revision)" in normalized
    assert _notifies(cursor) == [("arxplore_prepare_jobs", "auto:2026-04-07")]


def test_enqueue_merges_payload_so_requeued_force_survives():
    cursor = RecordingCursor(fetchone_results=[(7, "pending", False, 0, False)])
    _repository_with_cursor(cursor).enqueue_prepare_job(target_date="2026-04-07", payload={"collected": 3})
    sql, params = cursor.executed[0]
    normalized = _normalize_sql(sql)
    assert "payload = prepare_jobs.payload || EXCLUDED.payload," in normalized
    assert "payload = EXCLUDED.payload," not in normalized
    assert params[3].adapted == {"collected": 3}


def test_enqueue_without_revision_passes_null():
    cursor = RecordingCursor(fetchone_results=[(7, "done", False, 3, False)])
    result = _repository_with_cursor(cursor).enqueue_prepare_job(target_date="2026-04-07")
    assert cursor.executed[0][1][-1] is None
    assert result["status"] == "done"


def test_claim_issues_generation_token_and_skips_backoff():
    claimed_row = (
        5, "auto", "2026-04-07", "collect", {}, "processing", 2, "w1", "c", "cl", "u", 4, 3,
    )
    cursor = RecordingCursor(fetchone_results=[claimed_row])
    job = _repository_with_cursor(cursor).claim_prepare_job(mode="auto", worker_id="w1")

    assert job is not None
    assert (job["job_id"], job["worker_id"], job["claim_generation"]) == (5, "w1", 4)
    assert job["attempt_count"] == 2
    assert job["raw_revision"] == 3

    reset_sql = _normalize_sql(cursor.executed[0][0])
    assert reset_sql.startswith("UPDATE prepare_jobs SET")
    claim_sql, claim_params = cursor.executed[1]
    normalized = _normalize_sql(claim_sql)
    assert "AND (next_attempt_at IS NULL OR next_attempt_at <= NOW())" in normalized
    assert "FOR UPDATE SKIP LOCKED" in normalized
    assert "claim_generation = j.claim_generation + 1" in normalized
    assert "attempt_count = j.attempt_count + 1" in normalized
    assert "heartbeat_at = NOW()" in normalized
    assert "next_attempt_at = NULL" in normalized
    assert "j.claim_generation, j.raw_revision" in normalized
    assert claim_params == ("auto", "w1")


def test_claim_returns_none_when_queue_empty():
    cursor = RecordingCursor(fetchone_results=[None])
    assert _repository_with_cursor(cursor).claim_prepare_job() is None


@pytest.mark.parametrize(("rowcount", "expected"), [(1, True), (0, False)])
def test_heartbeat_is_fenced_by_claim_token(rowcount, expected):
    cursor = RecordingCursor(rowcount=rowcount)
    alive = _repository_with_cursor(cursor).heartbeat_prepare_job(job_id=5, worker_id="w1", claim_generation=4)

    assert alive is expected
    sql, params = cursor.executed[0]
    normalized = _normalize_sql(sql)
    assert normalized == (
        "UPDATE prepare_jobs SET heartbeat_at = NOW() WHERE id = %s AND worker_id = %s "
        "AND claim_generation = %s AND status = 'processing'"
    )
    assert params == (5, "w1", 4)


def test_reset_stale_uses_heartbeat_and_closes_exhausted_jobs():
    cursor = RecordingCursor(rows=[("2026-04-07", "pending"), ("2026-04-08", "failed")])
    reset = _repository_with_cursor(cursor, prepare_job_max_attempts=3).reset_stale_prepare_jobs(mode="auto")

    assert reset == 2
    sql, params = cursor.executed[0]
    normalized = _normalize_sql(sql)
    assert "AND COALESCE(heartbeat_at, claimed_at) IS NOT NULL" in normalized
    assert "AND COALESCE(heartbeat_at, claimed_at) < NOW() - (%s * INTERVAL '1 second')" in normalized
    assert "claimed_at < NOW()" not in normalized
    assert "status = CASE WHEN pending_refresh OR attempt_count < %s THEN 'pending' ELSE 'failed' END" in normalized
    assert "attempt_count = CASE WHEN pending_refresh THEN 0 ELSE attempt_count END" in normalized
    assert "heartbeat_at = NULL" in normalized
    assert "RETURNING target_date, status" in normalized
    assert params == (3, 3, STALE_EXHAUSTED_ERROR, 3, "auto", 900)
    assert _notifies(cursor) == [("arxplore_prepare_jobs", "auto:2026-04-07")]


def test_reset_stale_disabled_when_interval_not_positive():
    cursor = RecordingCursor()
    assert _repository_with_cursor(cursor).reset_stale_prepare_jobs(stale_seconds=0) == 0
    assert cursor.executed == []


def test_complete_is_fenced_and_reports_applied():
    cursor = RecordingCursor(fetchone_results=[("auto", "2026-04-07", "done")])
    applied = _repository_with_cursor(cursor).complete_prepare_job(
        job_id=5, worker_id="w1", claim_generation=4, result={"saved_papers": 1}
    )

    assert applied is True
    sql, params = cursor.executed[0]
    normalized = _normalize_sql(sql)
    assert "WHERE id = %s AND worker_id = %s AND claim_generation = %s AND status = 'processing'" in normalized
    assert "status = CASE WHEN pending_refresh THEN 'pending' ELSE 'done' END" in normalized
    assert "attempt_count = CASE WHEN pending_refresh THEN 0 ELSE attempt_count END" in normalized
    assert "pending_refresh = FALSE" in normalized
    assert "payload = payload - 'force'" in normalized
    assert "WHERE mode = %s AND target_date = %s" not in normalized
    assert params[1:] == (5, "w1", 4)
    assert _notifies(cursor) == []


def test_complete_with_lost_claim_changes_nothing():
    cursor = RecordingCursor(fetchone_results=[None])
    applied = _repository_with_cursor(cursor).complete_prepare_job(job_id=5, worker_id="w1", claim_generation=3)
    assert applied is False
    assert len(cursor.executed) == 1
    assert _notifies(cursor) == []


def test_complete_with_pending_refresh_requeues_and_notifies():
    cursor = RecordingCursor(fetchone_results=[("auto", "2026-04-07", "pending")])
    applied = _repository_with_cursor(cursor).complete_prepare_job(job_id=5, worker_id="w1", claim_generation=4)
    assert applied is True
    assert _notifies(cursor) == [("arxplore_prepare_jobs", "auto:2026-04-07")]


def test_fail_with_lost_claim_does_not_update():
    cursor = RecordingCursor(fetchone_results=[None])
    applied = _repository_with_cursor(cursor).fail_prepare_job(
        job_id=5, worker_id="w1", claim_generation=3, error="boom"
    )

    assert applied is False
    assert len(cursor.executed) == 1
    sql, params = cursor.executed[0]
    normalized = _normalize_sql(sql)
    assert normalized.startswith("SELECT attempt_count, pending_refresh FROM prepare_jobs")
    assert "WHERE id = %s AND worker_id = %s AND claim_generation = %s AND status = 'processing' FOR UPDATE" in normalized
    assert params == (5, "w1", 3)


def _fail_update(cursor: RecordingCursor) -> tuple[str, Any]:
    sql, params = cursor.executed[1]
    return _normalize_sql(sql), params


def test_fail_below_max_attempts_schedules_backoff_retry():
    cursor = RecordingCursor(fetchone_results=[(2, False), ("auto", "2026-04-07")])
    applied = _repository_with_cursor(cursor, prepare_job_max_attempts=3).fail_prepare_job(
        job_id=5, worker_id="w1", claim_generation=4, error="boom"
    )

    assert applied is True
    normalized, params = _fail_update(cursor)
    assert normalized.startswith("UPDATE prepare_jobs SET status = %s, attempt_count = %s")
    assert "NOW() + (%s::integer * INTERVAL '1 second')" in normalized
    assert normalized.endswith("WHERE id = %s RETURNING mode, target_date")
    assert params == ("pending", 2, 120, 120, "boom", False, False, False, False, 5)
    assert _notifies(cursor) == []


def test_fail_at_max_attempts_marks_failed():
    cursor = RecordingCursor(fetchone_results=[(3, False), ("auto", "2026-04-07")])
    applied = _repository_with_cursor(cursor, prepare_job_max_attempts=3).fail_prepare_job(
        job_id=5, worker_id="w1", claim_generation=4, error="boom"
    )

    assert applied is True
    _, params = _fail_update(cursor)
    assert params == ("failed", 3, None, None, "boom", True, True, True, True, 5)
    assert _notifies(cursor) == []


def test_fail_with_pending_refresh_retries_immediately():
    cursor = RecordingCursor(fetchone_results=[(3, True), ("auto", "2026-04-07")])
    applied = _repository_with_cursor(cursor, prepare_job_max_attempts=3).fail_prepare_job(
        job_id=5, worker_id="w1", claim_generation=4, error="boom"
    )

    assert applied is True
    _, params = _fail_update(cursor)
    assert params == ("pending", 0, None, None, "boom", False, False, False, False, 5)
    assert _notifies(cursor) == [("arxplore_prepare_jobs", "auto:2026-04-07")]


# ---- scripts/requeue_failed_prepare_jobs.py ----

from scripts import requeue_failed_prepare_jobs as requeue_script  # noqa: E402


class FakeJobRepository:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def requeue_failed_prepare_jobs(self, *, mode, since_date=None, dry_run=True, force=False):
        self.calls.append({"mode": mode, "since_date": since_date, "dry_run": dry_run, "force": force})
        return [{"id": 1, "target_date": "2026-04-07", "attempt_count": 1}]


def test_requeue_cli_defaults_to_dry_run(capsys):
    repository = FakeJobRepository()
    assert requeue_script.main(["--since", "2026-04-07", "--mode", "auto"], repository=repository) == 0
    assert repository.calls == [{"mode": "auto", "since_date": "2026-04-07", "dry_run": True, "force": False}]
    output = capsys.readouterr().out
    assert "[DRY-RUN]" in output
    assert "target_date=2026-04-07" in output


def test_requeue_cli_apply(capsys):
    repository = FakeJobRepository()
    assert requeue_script.main(["--since", "2026-04-07", "--mode", "auto", "--apply"], repository=repository) == 0
    assert repository.calls[0]["dry_run"] is False
    assert repository.calls[0]["force"] is False
    assert "[APPLY]" in capsys.readouterr().out


def test_requeue_cli_force_is_passed_through(capsys):
    repository = FakeJobRepository()
    assert requeue_script.main(["--apply", "--force"], repository=repository) == 0
    assert repository.calls[0]["force"] is True
    assert "force=True" in capsys.readouterr().out
