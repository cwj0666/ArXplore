from __future__ import annotations

import select
import threading
from types import SimpleNamespace
from typing import Any

import psycopg2
import pytest

from src.integrations.db import get_connection
from src.integrations.prepare_job_repository import PrepareJobRepository
from src.integrations.raw_store import RawPaperStore, compute_payload_hash
from src.pipeline import collect_papers

pytestmark = pytest.mark.integration


class DsnRawPaperStore(RawPaperStore):
    def __init__(self, dsn: str) -> None:
        super().__init__(settings=SimpleNamespace())
        self.dsn = dsn

    def _connection(self):
        return get_connection({"dsn": self.dsn}, settings=self.settings)


class DsnPrepareJobRepository(PrepareJobRepository):
    def __init__(self, dsn: str) -> None:
        super().__init__(settings=SimpleNamespace(prepare_job_stale_seconds=900, prepare_job_max_attempts=3))
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


def _drop_tables(dsn: str) -> None:
    _execute(dsn, "DROP TABLE IF EXISTS raw_daily_papers, pipeline_state, prepare_jobs")


@pytest.fixture
def dsn(test_database_url: str) -> str:
    _drop_tables(test_database_url)
    yield test_database_url
    _drop_tables(test_database_url)


@pytest.fixture
def store(dsn: str) -> DsnRawPaperStore:
    store = DsnRawPaperStore(dsn)
    store.ensure_schema()
    store.ensure_schema()
    return store


def _raw_row(dsn: str, date: str) -> tuple[Any, ...] | None:
    rows = _execute(
        dsn,
        "SELECT revision, payload_hash, fetched_count, collected_at, payload FROM raw_daily_papers "
        "WHERE source = 'hf_daily_papers' AND date = %s",
        (date,),
    )
    return rows[0] if rows else None


def test_first_save_inserts_revision_one(store, dsn):
    saved = store.save_daily_papers_response(date="2026-04-07", payload=[{"paper": {"id": "a"}}])

    assert saved == {"record_id": "hf_daily_papers:2026-04-07", "revision": 1, "changed": True}
    revision, payload_hash, fetched_count, _, payload = _raw_row(dsn, "2026-04-07")
    assert (revision, fetched_count) == (1, 1)
    assert payload_hash == compute_payload_hash([{"paper": {"id": "a"}}])
    assert payload == [{"paper": {"id": "a"}}]
    assert store.has_daily_papers_response(date="2026-04-07") is True
    assert store.has_daily_papers_response(date="2026-04-08") is False


def test_unchanged_payload_keeps_revision_and_refreshes_collected_at(store, dsn):
    store.save_daily_papers_response(date="2026-04-07", payload=[{"paper": {"id": "a", "title": "T"}}])
    _execute(dsn, "UPDATE raw_daily_papers SET collected_at = NOW() - INTERVAL '1 day'")
    before = _raw_row(dsn, "2026-04-07")[3]

    saved = store.save_daily_papers_response(date="2026-04-07", payload=[{"paper": {"title": "T", "id": "a"}}])

    assert saved == {"record_id": "hf_daily_papers:2026-04-07", "revision": 1, "changed": False}
    revision, _, _, collected_at, _ = _raw_row(dsn, "2026-04-07")
    assert revision == 1
    assert collected_at > before


def test_changed_payload_increments_revision_per_date(store, dsn):
    store.save_daily_papers_response(date="2026-04-07", payload=[{"paper": {"id": "a"}}])
    store.save_daily_papers_response(date="2026-04-07", payload=[{"paper": {"id": "a"}}])

    saved = store.save_daily_papers_response(
        date="2026-04-07", payload=[{"paper": {"id": "a"}}, {"paper": {"id": "b"}}]
    )
    other_date = store.save_daily_papers_response(date="2026-04-08", payload=[])

    assert saved == {"record_id": "hf_daily_papers:2026-04-07", "revision": 2, "changed": True}
    assert other_date == {"record_id": "hf_daily_papers:2026-04-08", "revision": 1, "changed": True}
    assert _raw_row(dsn, "2026-04-07")[2] == 2
    assert store.load_daily_papers_response(date="2026-04-07") == [{"paper": {"id": "a"}}, {"paper": {"id": "b"}}]
    assert store.load_daily_papers_response(date="2026-04-09") == []
    assert store.list_daily_papers_dates() == ["2026-04-07", "2026-04-08"]
    assert store.list_daily_papers_dates(date_gt="2026-04-07", ascending=False) == ["2026-04-08"]
    assert store.list_daily_papers_dates(ascending=False, limit=1) == ["2026-04-08"]


def test_nul_characters_are_stored_without_error(store):
    store.save_daily_papers_response(date="2026-04-07", payload=[{"title": "a\x00b"}])
    assert store.load_daily_papers_response(date="2026-04-07") == [{"title": "ab"}]


def test_concurrent_distinct_payloads_get_distinct_revisions(store, dsn):
    worker_count = 6
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(worker_count)

    def save(index: int) -> None:
        try:
            barrier.wait()
            results.append(store.save_daily_papers_response(date="2026-04-07", payload=[{"n": index}]))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=save, args=(index,)) for index in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert sorted(result["revision"] for result in results) == list(range(1, worker_count + 1))
    assert _raw_row(dsn, "2026-04-07")[0] == worker_count


def test_pipeline_state_round_trip(store):
    assert store.load_pipeline_state(pipeline="hf_daily_papers_backfill", name="default") is None

    store.save_pipeline_state(
        pipeline="hf_daily_papers_backfill",
        name="default",
        state={"cursor_date": "2026-04-06", "last_failure": {"date": "2026-04-07", "error": "429"}},
    )
    store.save_pipeline_state(
        pipeline="hf_daily_papers_backfill",
        name="default",
        state={"cursor_date": "2026-04-05", "last_processed_dates": ["2026-04-06"], "last_failure": None},
    )
    loaded = store.load_pipeline_state(pipeline="hf_daily_papers_backfill", name="default")

    assert loaded is not None
    assert loaded["cursor_date"] == "2026-04-05"
    assert loaded["last_processed_dates"] == ["2026-04-06"]
    assert loaded["last_failure"] is None
    assert loaded["pipeline"] == "hf_daily_papers_backfill"
    assert loaded["name"] == "default"
    assert loaded["updated_at"] is not None
    assert store.load_pipeline_state(pipeline="hf_daily_papers_backfill", name="other") is None


class FakeSearchClient:
    def fetch_daily_papers(self, target_date: str) -> list[dict[str, Any]]:
        return [{"paper": {"id": "2604.00001"}}]


class FailingPrepareJobRepository(DsnPrepareJobRepository):
    def enqueue_prepare_job(self, **kwargs: Any) -> dict[str, Any]:
        super().enqueue_prepare_job(**kwargs)
        raise RuntimeError("enqueue failed after insert")


def _patch_collect(monkeypatch, dsn: str, repository_class: type[DsnPrepareJobRepository]) -> None:
    monkeypatch.setattr(collect_papers, "PaperSearchClient", FakeSearchClient)
    monkeypatch.setattr(collect_papers, "RawPaperStore", lambda: DsnRawPaperStore(dsn))
    monkeypatch.setattr(collect_papers, "PrepareJobRepository", lambda: repository_class(dsn))


def _listen(dsn: str):
    connection = psycopg2.connect(dsn=dsn)
    connection.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    with connection.cursor() as cursor:
        cursor.execute(f"LISTEN {PrepareJobRepository.channel_name}")
    return connection


def _drain_notifications(connection, timeout: float) -> list[str]:
    select.select([connection], [], [], timeout)
    connection.poll()
    payloads = [notify.payload for notify in connection.notifies]
    connection.notifies.clear()
    return payloads


def test_collect_commits_raw_and_job_together_and_notifies(monkeypatch, store, dsn):
    DsnPrepareJobRepository(dsn).ensure_schema()
    _patch_collect(monkeypatch, dsn, DsnPrepareJobRepository)
    listener = _listen(dsn)
    try:
        result = collect_papers.run_collect_papers(runtime="test", target_date="2026-04-07")
        notifications = _drain_notifications(listener, 5.0)
    finally:
        listener.close()

    assert result["raw_revision"] == 1
    assert result["raw_payload_changed"] is True
    assert result["prepare_job_enqueued"] is True
    assert _raw_row(dsn, "2026-04-07")[0] == 1
    jobs = _execute(dsn, "SELECT target_date, status, raw_revision FROM prepare_jobs")
    assert jobs == [("2026-04-07", "pending", 1)]
    assert notifications == ["auto:2026-04-07"]


def test_collect_enqueue_failure_rolls_back_raw_and_sends_no_notification(monkeypatch, store, dsn):
    DsnPrepareJobRepository(dsn).ensure_schema()
    _patch_collect(monkeypatch, dsn, FailingPrepareJobRepository)
    listener = _listen(dsn)
    try:
        with pytest.raises(RuntimeError, match="enqueue failed after insert"):
            collect_papers.run_collect_papers(runtime="test", target_date="2026-04-07")
        notifications = _drain_notifications(listener, 0.5)
    finally:
        listener.close()

    assert _raw_row(dsn, "2026-04-07") is None
    assert _execute(dsn, "SELECT COUNT(*) FROM prepare_jobs") == [(0,)]
    assert notifications == []


def test_collect_enqueue_database_error_rolls_back_raw(monkeypatch, store, dsn):
    _patch_collect(monkeypatch, dsn, DsnPrepareJobRepository)

    with pytest.raises(psycopg2.errors.UndefinedTable):
        collect_papers.run_collect_papers(runtime="test", target_date="2026-04-07")

    assert _raw_row(dsn, "2026-04-07") is None
