from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from src.integrations.raw_store import RawPaperStore, compute_payload_hash
from src.pipeline import collect_papers, enrich_papers_metadata


class FakeMongoCollection:
    def __init__(self) -> None:
        self.documents: dict[tuple[str, str], dict[str, Any]] = {}
        self.calls: list[dict[str, Any]] = []

    def find_one_and_update(self, filter, update, *, projection=None, upsert=False, return_document=None):
        self.calls.append({"filter": filter, "update": update, "upsert": upsert, "return_document": return_document})
        key = (filter["source"], filter["date"])
        document = self.documents.get(key)
        extra_conditions = {name: value for name, value in filter.items() if name not in ("source", "date")}
        if document is not None and any(document.get(name) != value for name, value in extra_conditions.items()):
            document = None
            if upsert:
                raise AssertionError("upsert with a non-key filter would duplicate (source, date)")
        if document is None:
            if not upsert:
                return None
            document = {"_id": f"oid-{len(self.documents) + 1}", **filter}
            self.documents[key] = document
        document.update(update["$set"])
        for field, amount in update.get("$inc", {}).items():
            document[field] = document.get(field, 0) + amount
        return {name: document[name] for name in ("_id", *(projection or {})) if name in document}

    def find_one(self, filter, projection=None, sort=None):
        return self.documents.get((filter["source"], filter["date"]))


def _raw_store(collection: FakeMongoCollection) -> RawPaperStore:
    settings = SimpleNamespace(mongo_db="db", mongo_daily_papers_collection="raw")
    return RawPaperStore(settings=settings, client={"db": {"raw": collection}})


def test_first_save_starts_at_revision_one():
    collection = FakeMongoCollection()
    saved = _raw_store(collection).save_daily_papers_response(date="2026-04-07", payload=[{"paper": {"id": "a"}}])

    assert saved == {"record_id": "oid-1", "revision": 1, "changed": True}
    stored = collection.documents[("hf_daily_papers", "2026-04-07")]
    assert stored["payload_hash"] == compute_payload_hash([{"paper": {"id": "a"}}])
    assert stored["fetched_count"] == 1


def test_same_payload_keeps_revision_and_refreshes_collected_at():
    collection = FakeMongoCollection()
    store = _raw_store(collection)
    store.save_daily_papers_response(date="2026-04-07", payload=[{"paper": {"id": "a", "title": "T"}}])
    first_collected_at = collection.documents[("hf_daily_papers", "2026-04-07")]["collected_at"]

    saved = store.save_daily_papers_response(date="2026-04-07", payload=[{"paper": {"title": "T", "id": "a"}}])

    assert saved == {"record_id": "oid-1", "revision": 1, "changed": False}
    stored = collection.documents[("hf_daily_papers", "2026-04-07")]
    assert stored["revision"] == 1
    assert stored["collected_at"] >= first_collected_at
    assert "$inc" not in collection.calls[-1]["update"]
    assert collection.calls[-1]["upsert"] is False


def test_different_payload_increments_revision():
    collection = FakeMongoCollection()
    store = _raw_store(collection)
    store.save_daily_papers_response(date="2026-04-07", payload=[{"paper": {"id": "a"}}])
    store.save_daily_papers_response(date="2026-04-07", payload=[{"paper": {"id": "a"}}])

    saved = store.save_daily_papers_response(
        date="2026-04-07", payload=[{"paper": {"id": "a"}}, {"paper": {"id": "b"}}]
    )
    other_date = store.save_daily_papers_response(date="2026-04-08", payload=[])

    assert saved == {"record_id": "oid-1", "revision": 2, "changed": True}
    assert other_date == {"record_id": "oid-2", "revision": 1, "changed": True}
    assert len(collection.documents) == 2
    stored = collection.documents[("hf_daily_papers", "2026-04-07")]
    assert stored["fetched_count"] == 2
    assert stored["payload_hash"] == compute_payload_hash([{"paper": {"id": "a"}}, {"paper": {"id": "b"}}])
    assert store.load_daily_papers_response(date="2026-04-07") == [{"paper": {"id": "a"}}, {"paper": {"id": "b"}}]


def test_changed_save_upserts_on_source_and_date():
    collection = FakeMongoCollection()
    _raw_store(collection).save_daily_papers_response(date="2026-04-07", payload={"paper": {"id": "a"}})

    call = collection.calls[-1]
    assert call["filter"] == {"source": "hf_daily_papers", "date": "2026-04-07"}
    assert call["upsert"] is True
    assert call["update"]["$inc"] == {"revision": 1}
    assert set(call["update"]["$set"]) == {"source", "date", "payload", "payload_hash", "fetched_count", "collected_at"}
    assert call["update"]["$set"]["fetched_count"] == 1


def test_legacy_document_without_hash_or_revision_starts_at_one():
    collection = FakeMongoCollection()
    collection.documents[("hf_daily_papers", "2026-04-07")] = {
        "_id": "legacy",
        "source": "hf_daily_papers",
        "date": "2026-04-07",
        "payload": [],
    }
    saved = _raw_store(collection).save_daily_papers_response(date="2026-04-07", payload=[])
    assert saved == {"record_id": "legacy", "revision": 1, "changed": True}


def test_payload_hash_ignores_key_order_but_not_values():
    assert compute_payload_hash({"a": 1, "b": [1, 2]}) == compute_payload_hash({"b": [1, 2], "a": 1})
    assert compute_payload_hash({"a": 1}) != compute_payload_hash({"a": 2})


class FakeSearchClient:
    def fetch_daily_papers(self, target_date):
        return [{"paper": {"id": "2604.00001"}}]


class FakeRawStore:
    def __init__(self, revision: int = 4, changed: bool = True) -> None:
        self.revision = revision
        self.changed = changed
        self.saved: list[str] = []

    def save_daily_papers_response(self, *, date, payload):
        self.saved.append(date)
        return {"record_id": "oid-1", "revision": self.revision, "changed": self.changed}


class FakePrepareJobRepository:
    instances: list[FakePrepareJobRepository] = []

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        FakePrepareJobRepository.instances.append(self)

    def ensure_schema(self):
        self.calls.append(("ensure_schema", {}))

    def enqueue_prepare_job(self, **kwargs):
        self.calls.append(("enqueue", kwargs))
        return {"enqueued": False, "job_id": 9, "status": "pending"}


def _patch_collect(monkeypatch, raw_store: FakeRawStore) -> None:
    FakePrepareJobRepository.instances = []
    monkeypatch.setattr(collect_papers, "PaperSearchClient", FakeSearchClient)
    monkeypatch.setattr(collect_papers, "RawPaperStore", lambda: raw_store)
    monkeypatch.setattr(collect_papers, "PrepareJobRepository", FakePrepareJobRepository)


def test_collect_passes_raw_revision_to_enqueue_after_ensure_schema(monkeypatch):
    _patch_collect(monkeypatch, FakeRawStore(revision=4))

    result = collect_papers.run_collect_papers(runtime="test", target_date="2026-04-07")

    assert len(FakePrepareJobRepository.instances) == 1
    calls = FakePrepareJobRepository.instances[0].calls
    assert calls == [
        ("ensure_schema", {}),
        ("enqueue", {"target_date": "2026-04-07", "mode": "auto", "source": "collect", "raw_revision": 4}),
    ]
    assert result["raw_revision"] == 4
    assert result["raw_payload_changed"] is True
    assert result["stored_record_id"] == "oid-1"
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


def test_enrich_ensures_schema_once_on_constructed_repository(monkeypatch):
    FakePaperRepository.instances = []
    monkeypatch.setattr(enrich_papers_metadata, "PaperRepository", FakePaperRepository)

    result = enrich_papers_metadata.run_enrich_papers_metadata(runtime="test", search_client=FakeArxivClient())

    assert result["updated_count"] == 2
    assert [repository.ensure_calls for repository in FakePaperRepository.instances] == [1]


def test_enrich_does_not_run_ddl_on_injected_repository():
    repository = FakePaperRepository()
    enrich_papers_metadata.run_enrich_papers_metadata(
        runtime="test", paper_repository=repository, search_client=FakeArxivClient()
    )
    assert repository.ensure_calls == 0
