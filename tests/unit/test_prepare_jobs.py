from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

from src.integrations import prepare_job_repository as prepare_job_repository_module
from src.integrations.prepare_job_repository import PrepareJobRepository


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.split())


class RecordingCursor:
    def __init__(self, rows: list[tuple[Any, ...]] | None = None) -> None:
        self.executed: list[tuple[str, Any]] = []
        self.rows = rows or []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return list(self.rows)

    def fetchone(self):
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


def _repository_with_cursor(cursor: RecordingCursor) -> PrepareJobRepository:
    repository = PrepareJobRepository(settings=object())

    @contextmanager
    def fake_connection():
        yield RecordingConnection(cursor)

    repository._connection = fake_connection  # type: ignore[method-assign]
    return repository


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
    cursor = RecordingCursor(rows=[(12, "2026-04-08", 2), (11, "2026-04-07", 1)])
    rows = _repository_with_cursor(cursor).requeue_failed_prepare_jobs(
        mode="auto", since_date="2026-04-07", dry_run=False
    )

    assert [row["target_date"] for row in rows] == ["2026-04-07", "2026-04-08"]
    sql, params = cursor.executed[0]
    normalized = _normalize_sql(sql)
    assert normalized.startswith("UPDATE prepare_jobs SET")
    for clause in (
        "status = 'pending'",
        "worker_id = NULL",
        "error = NULL",
        "claimed_at = NULL",
        "finished_at = NULL",
        "updated_at = NOW()",
    ):
        assert clause in normalized
    assert "WHERE mode = %s AND status = 'failed' AND target_date >= %s" in normalized
    assert "RETURNING id, target_date, attempt_count" in normalized
    assert params == ("auto", "2026-04-07")

    notifies = cursor.executed[1:]
    assert [params for _, params in notifies] == [
        ("arxplore_prepare_jobs", "auto:2026-04-07"),
        ("arxplore_prepare_jobs", "auto:2026-04-08"),
    ]


def test_requeue_without_since_has_no_date_predicate():
    cursor = RecordingCursor()
    assert _repository_with_cursor(cursor).requeue_failed_prepare_jobs(mode="auto") == []
    sql, params = cursor.executed[0]
    assert "target_date >=" not in sql
    assert params == ("auto",)


def test_requeue_rejects_invalid_since_date():
    with pytest.raises(ValueError):
        _repository_with_cursor(RecordingCursor()).requeue_failed_prepare_jobs(mode="auto", since_date="2026/04/07")


# ---- scripts/requeue_failed_prepare_jobs.py ----

from scripts import requeue_failed_prepare_jobs as requeue_script  # noqa: E402


class FakeJobRepository:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def requeue_failed_prepare_jobs(self, *, mode, since_date=None, dry_run=True):
        self.calls.append({"mode": mode, "since_date": since_date, "dry_run": dry_run})
        return [{"id": 1, "target_date": "2026-04-07", "attempt_count": 1}]


def test_requeue_cli_defaults_to_dry_run(capsys):
    repository = FakeJobRepository()
    assert requeue_script.main(["--since", "2026-04-07", "--mode", "auto"], repository=repository) == 0
    assert repository.calls == [{"mode": "auto", "since_date": "2026-04-07", "dry_run": True}]
    output = capsys.readouterr().out
    assert "[DRY-RUN]" in output
    assert "target_date=2026-04-07" in output


def test_requeue_cli_apply(capsys):
    repository = FakeJobRepository()
    assert requeue_script.main(["--since", "2026-04-07", "--mode", "auto", "--apply"], repository=repository) == 0
    assert repository.calls[0]["dry_run"] is False
    assert "[APPLY]" in capsys.readouterr().out
