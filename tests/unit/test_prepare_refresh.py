from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from src.integrations import raw_store as raw_store_module
from src.integrations.raw_store import (
    SAVE_DAILY_PAPERS_SQL,
    RawPaperStore,
    compute_payload_hash,
    pipeline_state_key,
    sanitize_payload,
)
from src.pipeline import collect_papers, enrich_papers_metadata


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.split())


class RecordingCursor:
    def __init__(self, fetchone_results: list[Any] | None = None, rows: list[tuple[Any, ...]] | None = None) -> None:
        self.executed: list[tuple[str, Any]] = []
        self.fetchone_results = list(fetchone_results or [])
        self.rows = rows or []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self.fetchone_results.pop(0) if self.fetchone_results else None

    def fetchall(self):
        return list(self.rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class RecordingConnection:
    def __init__(self, cursor: RecordingCursor) -> None:
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def _store_with_cursor(cursor: RecordingCursor) -> tuple[RawPaperStore, list[str]]:
    store = RawPaperStore(settings=SimpleNamespace())
    opened: list[str] = []

    @contextmanager
    def fake_connection():
        opened.append("pool")
        yield RecordingConnection(cursor)

    store._connection = fake_connection  # type: ignore[method-assign]
    return store, opened


def test_constructor_opens_no_connection(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("RawPaperStore() must not open a DB connection")

    monkeypatch.setattr(raw_store_module, "get_connection", fail)
    RawPaperStore(settings=SimpleNamespace())


def test_save_runs_single_upsert_with_hash_and_revision_case():
    cursor = RecordingCursor(fetchone_results=[(1, True)])
    store, _ = _store_with_cursor(cursor)

    saved = store.save_daily_papers_response(date="2026-04-07", payload=[{"paper": {"id": "a"}}])

    assert saved == {"record_id": "hf_daily_papers:2026-04-07", "revision": 1, "changed": True}
    assert len(cursor.executed) == 1
    sql, params = cursor.executed[0]
    assert sql is SAVE_DAILY_PAPERS_SQL
    normalized = _normalize_sql(sql)
    assert "WITH previous AS ( SELECT payload_hash FROM raw_daily_papers" in normalized
    assert "ON CONFLICT (source, date) DO UPDATE SET" in normalized
    assert (
        "revision = CASE WHEN raw_daily_papers.payload_hash = EXCLUDED.payload_hash "
        "THEN raw_daily_papers.revision ELSE raw_daily_papers.revision + 1 END"
    ) in normalized
    assert "collected_at = NOW()" in normalized
    assert "IS DISTINCT FROM upserted.payload_hash AS changed" in normalized
    assert "xmax" not in normalized
    assert params["source"] == "hf_daily_papers"
    assert params["date"] == "2026-04-07"
    assert params["payload_hash"] == compute_payload_hash([{"paper": {"id": "a"}}])
    assert params["payload"].adapted == [{"paper": {"id": "a"}}]
    assert params["fetched_count"] == 1


def test_save_hash_ignores_key_order_so_reordered_payload_is_not_a_new_revision():
    cursor = RecordingCursor(fetchone_results=[(1, True), (1, False)])
    store, _ = _store_with_cursor(cursor)

    store.save_daily_papers_response(date="2026-04-07", payload=[{"paper": {"id": "a", "title": "T"}}])
    saved = store.save_daily_papers_response(date="2026-04-07", payload=[{"paper": {"title": "T", "id": "a"}}])

    first_hash = cursor.executed[0][1]["payload_hash"]
    second_hash = cursor.executed[1][1]["payload_hash"]
    assert first_hash == second_hash
    assert saved == {"record_id": "hf_daily_papers:2026-04-07", "revision": 1, "changed": False}


def test_save_reports_database_revision_for_changed_payload():
    cursor = RecordingCursor(fetchone_results=[(3, True)])
    store, _ = _store_with_cursor(cursor)

    saved = store.save_daily_papers_response(
        date="2026-04-07", payload=[{"paper": {"id": "a"}}, {"paper": {"id": "b"}}]
    )

    assert saved == {"record_id": "hf_daily_papers:2026-04-07", "revision": 3, "changed": True}
    params = cursor.executed[0][1]
    assert params["fetched_count"] == 2
    assert params["payload_hash"] == compute_payload_hash([{"paper": {"id": "a"}}, {"paper": {"id": "b"}}])


def test_save_dict_payload_counts_as_one():
    cursor = RecordingCursor(fetchone_results=[(1, True)])
    store, _ = _store_with_cursor(cursor)

    store.save_daily_papers_response(date="2026-04-07", payload={"paper": {"id": "a"}})

    assert cursor.executed[0][1]["fetched_count"] == 1


def test_save_strips_characters_jsonb_rejects_before_hashing():
    cursor = RecordingCursor(fetchone_results=[(1, True)])
    store, _ = _store_with_cursor(cursor)

    store.save_daily_papers_response(date="2026-04-07", payload=[{"title": "a\x00b\ud800c"}])

    params = cursor.executed[0][1]
    assert params["payload"].adapted == [{"title": "abc"}]
    assert params["payload_hash"] == compute_payload_hash([{"title": "abc"}])


def test_save_uses_given_connection_without_borrowing_from_pool():
    pooled = RecordingCursor()
    store, opened = _store_with_cursor(pooled)
    shared = RecordingCursor(fetchone_results=[(2, False)])

    saved = store.save_daily_papers_response(date="2026-04-07", payload=[], connection=RecordingConnection(shared))

    assert opened == []
    assert pooled.executed == []
    assert len(shared.executed) == 1
    assert saved["revision"] == 2


def test_save_raises_when_upsert_returns_nothing():
    store, _ = _store_with_cursor(RecordingCursor(fetchone_results=[None]))
    with pytest.raises(RuntimeError):
        store.save_daily_papers_response(date="2026-04-07", payload=[])


def test_load_wraps_dict_payload_and_returns_empty_when_missing():
    cursor = RecordingCursor(fetchone_results=[([{"paper": {"id": "a"}}],), ({"paper": {"id": "b"}},), None])
    store, _ = _store_with_cursor(cursor)

    assert store.load_daily_papers_response(date="2026-04-07") == [{"paper": {"id": "a"}}]
    assert store.load_daily_papers_response(date="2026-04-08") == [{"paper": {"id": "b"}}]
    assert store.load_daily_papers_response(date="2026-04-09") == []
    assert cursor.executed[0][1] == ("hf_daily_papers", "2026-04-07")


def test_has_daily_papers_response_uses_exists():
    cursor = RecordingCursor(fetchone_results=[(True,), (False,)])
    store, _ = _store_with_cursor(cursor)

    assert store.has_daily_papers_response(date="2026-04-07") is True
    assert store.has_daily_papers_response(date="2026-04-08") is False
    assert "SELECT EXISTS" in cursor.executed[0][0]


def test_list_dates_builds_filters_order_and_limit():
    cursor = RecordingCursor(rows=[(date(2026, 4, 9),), (date(2026, 4, 8),)])
    store, _ = _store_with_cursor(cursor)

    dates = store.list_daily_papers_dates(date_gt="2026-04-01", date_lte="2026-04-30", limit=2, ascending=False)

    assert dates == ["2026-04-09", "2026-04-08"]
    sql, params = cursor.executed[0]
    assert _normalize_sql(sql) == (
        "SELECT date FROM raw_daily_papers WHERE source = %s AND date > %s AND date <= %s ORDER BY date DESC LIMIT %s"
    )
    assert params == ("hf_daily_papers", "2026-04-01", "2026-04-30", 2)


def test_pipeline_state_round_trip_shapes():
    cursor = RecordingCursor(fetchone_results=[({"cursor_date": "2026-04-06", "status": "success"}, "ts"), None])
    store, _ = _store_with_cursor(cursor)

    store.save_pipeline_state(
        pipeline="hf_daily_papers_backfill",
        name="default",
        state={"cursor_date": "2026-04-06", "status": "success", "pipeline": "ignored"},
    )
    loaded = store.load_pipeline_state(pipeline="hf_daily_papers_backfill", name="default")
    missing = store.load_pipeline_state(pipeline="prepare_papers_backfill", name="other")

    save_sql, save_params = cursor.executed[0]
    assert "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()" in _normalize_sql(save_sql)
    assert save_params[0] == "hf_daily_papers_backfill:default"
    assert save_params[1].adapted == {"cursor_date": "2026-04-06", "status": "success"}
    assert cursor.executed[1][1] == ("hf_daily_papers_backfill:default",)
    assert loaded == {
        "cursor_date": "2026-04-06",
        "status": "success",
        "pipeline": "hf_daily_papers_backfill",
        "name": "default",
        "updated_at": "ts",
    }
    assert missing is None
    assert pipeline_state_key("prepare_papers_backfill") == "prepare_papers_backfill:default"


def test_ensure_schema_creates_raw_and_state_tables():
    cursor = RecordingCursor()
    store, _ = _store_with_cursor(cursor)

    store.ensure_schema()

    statements = [_normalize_sql(sql) for sql, _ in cursor.executed]
    raw_ddl = next(sql for sql in statements if "CREATE TABLE IF NOT EXISTS raw_daily_papers" in sql)
    state_ddl = next(sql for sql in statements if "CREATE TABLE IF NOT EXISTS pipeline_state" in sql)
    assert "payload JSONB NOT NULL" in raw_ddl
    assert "payload_hash TEXT NOT NULL" in raw_ddl
    assert "revision INTEGER NOT NULL DEFAULT 1" in raw_ddl
    assert "PRIMARY KEY (source, date)" in raw_ddl
    assert "key TEXT PRIMARY KEY" in state_ddl
    assert "value JSONB NOT NULL" in state_ddl
    assert not any("DROP" in sql for sql in statements)


def test_sanitize_payload_handles_nested_keys_and_tuples():
    assert sanitize_payload({"k\x00": ("a\x00", 1, None)}) == {"k": ["a", 1, None]}


def test_payload_hash_ignores_key_order_but_not_values():
    assert compute_payload_hash({"a": 1, "b": [1, 2]}) == compute_payload_hash({"b": [1, 2], "a": 1})
    assert compute_payload_hash({"a": 1}) != compute_payload_hash({"a": 2})


class FakeSearchClient:
    def fetch_daily_papers(self, target_date):
        return [{"paper": {"id": "2604.00001"}}]


SHARED_CONNECTION = object()


class FakeRawStore:
    def __init__(self, revision: int = 4, changed: bool = True) -> None:
        self.revision = revision
        self.changed = changed
        self.saved: list[str] = []
        self.transactions: list[str] = []

    @contextmanager
    def transaction(self):
        self.transactions.append("begin")
        try:
            yield SHARED_CONNECTION
        except BaseException:
            self.transactions.append("rollback")
            raise
        self.transactions.append("commit")

    def save_daily_papers_response(self, *, date, payload, connection=None):
        assert connection is SHARED_CONNECTION
        self.saved.append(date)
        return {"record_id": "hf_daily_papers:" + date, "revision": self.revision, "changed": self.changed}


class FakePrepareJobRepository:
    instances: list[FakePrepareJobRepository] = []
    fail_enqueue = False

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        FakePrepareJobRepository.instances.append(self)

    def ensure_schema(self):
        raise AssertionError("Airflow task paths must not run DDL")

    def enqueue_prepare_job(self, **kwargs):
        assert kwargs.pop("connection") is SHARED_CONNECTION
        self.calls.append(("enqueue", kwargs))
        if self.fail_enqueue:
            raise RuntimeError("enqueue failed")
        return {"enqueued": False, "job_id": 9, "status": "pending"}


def _patch_collect(monkeypatch, raw_store: FakeRawStore, *, fail_enqueue: bool = False) -> None:
    FakePrepareJobRepository.instances = []
    monkeypatch.setattr(FakePrepareJobRepository, "fail_enqueue", fail_enqueue)
    monkeypatch.setattr(collect_papers, "PaperSearchClient", FakeSearchClient)
    monkeypatch.setattr(collect_papers, "RawPaperStore", lambda: raw_store)
    monkeypatch.setattr(collect_papers, "PrepareJobRepository", FakePrepareJobRepository)


def test_collect_passes_raw_revision_to_enqueue_without_ddl(monkeypatch):
    _patch_collect(monkeypatch, FakeRawStore(revision=4))

    result = collect_papers.run_collect_papers(runtime="test", target_date="2026-04-07")

    assert len(FakePrepareJobRepository.instances) == 1
    calls = FakePrepareJobRepository.instances[0].calls
    assert calls == [
        ("enqueue", {"target_date": "2026-04-07", "mode": "auto", "source": "collect", "raw_revision": 4}),
    ]
    assert result["raw_revision"] == 4
    assert result["raw_payload_changed"] is True
    assert result["stored_record_id"] == "hf_daily_papers:2026-04-07"
    assert result["prepare_job_id"] == 9
    assert result["prepare_job_status"] == "pending"


def test_collect_unchanged_payload_still_enqueues_with_same_revision(monkeypatch):
    _patch_collect(monkeypatch, FakeRawStore(revision=3, changed=False))

    result = collect_papers.run_collect_papers(runtime="test", target_date="2026-04-07")

    enqueue_calls = [kwargs for name, kwargs in FakePrepareJobRepository.instances[0].calls if name == "enqueue"]
    assert enqueue_calls == [{"target_date": "2026-04-07", "mode": "auto", "source": "collect", "raw_revision": 3}]
    assert result["raw_payload_changed"] is False
    assert result["raw_revision"] == 3


def test_collect_backfill_path_skips_queue_entirely(monkeypatch):
    raw_store = FakeRawStore(revision=2)
    _patch_collect(monkeypatch, raw_store)

    result = collect_papers.run_collect_papers(runtime="test", target_date="2026-04-07", enqueue_prepare=False)

    assert FakePrepareJobRepository.instances == []
    assert raw_store.saved == ["2026-04-07"]
    assert result["prepare_job_enqueued"] is False
    assert result["prepare_job_id"] is None
    assert result["raw_revision"] == 2
    assert raw_store.transactions == ["begin", "commit"]


def test_collect_saves_raw_and_enqueues_in_one_transaction(monkeypatch):
    raw_store = FakeRawStore(revision=4)
    _patch_collect(monkeypatch, raw_store)

    collect_papers.run_collect_papers(runtime="test", target_date="2026-04-07")

    assert raw_store.transactions == ["begin", "commit"]
    assert raw_store.saved == ["2026-04-07"]
    assert len(FakePrepareJobRepository.instances[0].calls) == 1


def test_collect_enqueue_failure_rolls_back_the_raw_transaction(monkeypatch):
    raw_store = FakeRawStore(revision=4)
    _patch_collect(monkeypatch, raw_store, fail_enqueue=True)

    with pytest.raises(RuntimeError, match="enqueue failed"):
        collect_papers.run_collect_papers(runtime="test", target_date="2026-04-07")

    assert raw_store.transactions == ["begin", "rollback"]


class FakePaperRepository:
    instances: list[FakePaperRepository] = []

    def __init__(self) -> None:
        self.ensure_calls = 0
        FakePaperRepository.instances.append(self)

    def ensure_schema(self):
        self.ensure_calls += 1

    def list_papers_missing_arxiv_metadata(self, *, limit):
        return [{"arxiv_id": "2604.00001"}, {"arxiv_id": "2604.00002"}]

    def save_paper(self, paper):
        return paper["arxiv_id"]


class FakeArxivClient:
    def fetch_arxiv_metadata(self, arxiv_ids):
        return {arxiv_id: {"title": "T", "primary_category": "cs.AI"} for arxiv_id in arxiv_ids}


def test_enrich_does_not_run_ddl_on_constructed_repository(monkeypatch):
    FakePaperRepository.instances = []
    monkeypatch.setattr(enrich_papers_metadata, "PaperRepository", FakePaperRepository)

    result = enrich_papers_metadata.run_enrich_papers_metadata(runtime="test", search_client=FakeArxivClient())

    assert result["updated_count"] == 2
    assert [repository.ensure_calls for repository in FakePaperRepository.instances] == [0]


def test_enrich_does_not_run_ddl_on_injected_repository():
    repository = FakePaperRepository()
    enrich_papers_metadata.run_enrich_papers_metadata(
        runtime="test", paper_repository=repository, search_client=FakeArxivClient()
    )
    assert repository.ensure_calls == 0
