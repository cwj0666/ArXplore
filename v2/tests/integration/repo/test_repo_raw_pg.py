import asyncio
from datetime import date
from typing import Any

import psycopg
import pytest

from arxplore.db.pool import transaction
from arxplore.db.repositories import jobs, pipeline_state, raw
from arxplore.db.repositories.raw import RawSaveResult, compute_payload_hash

pytestmark = pytest.mark.integration

APRIL_7 = date(2026, 4, 7)
APRIL_8 = date(2026, 4, 8)
APRIL_9 = date(2026, 4, 9)


async def _raw_row(probe, target_date: date) -> dict[str, Any] | None:
    return await probe.row(
        "SELECT revision, payload_hash, fetched_count, collected_at, payload FROM raw_daily_papers "
        "WHERE source = 'hf_daily_papers' AND date = %s",
        (target_date,),
    )


async def test_first_save_inserts_revision_one(pool, probe) -> None:
    saved = await raw.save_daily_papers(pool, target_date=APRIL_7, payload=[{"paper": {"id": "a"}}])

    assert saved == RawSaveResult(record_id="hf_daily_papers:2026-04-07", revision=1, changed=True)
    row = await _raw_row(probe, APRIL_7)
    assert row is not None
    assert (row["revision"], row["fetched_count"]) == (1, 1)
    assert row["payload_hash"] == compute_payload_hash([{"paper": {"id": "a"}}])
    assert row["payload"] == [{"paper": {"id": "a"}}]
    assert await raw.has_daily_papers(pool, target_date=APRIL_7) is True
    assert await raw.has_daily_papers(pool, target_date=APRIL_8) is False


async def test_unchanged_payload_keeps_revision_and_refreshes_collected_at(pool, probe) -> None:
    await raw.save_daily_papers(pool, target_date=APRIL_7, payload=[{"paper": {"id": "a", "title": "T"}}])
    await probe.execute("UPDATE raw_daily_papers SET collected_at = NOW() - INTERVAL '1 day'")
    before = await _raw_row(probe, APRIL_7)
    assert before is not None

    saved = await raw.save_daily_papers(pool, target_date=APRIL_7, payload=[{"paper": {"title": "T", "id": "a"}}])

    assert saved == RawSaveResult(record_id="hf_daily_papers:2026-04-07", revision=1, changed=False)
    after = await _raw_row(probe, APRIL_7)
    assert after is not None
    assert after["revision"] == 1
    assert after["collected_at"] > before["collected_at"]


async def test_changed_payload_increments_revision_per_date(pool, probe) -> None:
    await raw.save_daily_papers(pool, target_date=APRIL_7, payload=[{"paper": {"id": "a"}}])
    await raw.save_daily_papers(pool, target_date=APRIL_7, payload=[{"paper": {"id": "a"}}])

    saved = await raw.save_daily_papers(
        pool, target_date=APRIL_7, payload=[{"paper": {"id": "a"}}, {"paper": {"id": "b"}}]
    )
    other_date = await raw.save_daily_papers(pool, target_date=APRIL_8, payload=[])

    assert saved == RawSaveResult(record_id="hf_daily_papers:2026-04-07", revision=2, changed=True)
    assert other_date == RawSaveResult(record_id="hf_daily_papers:2026-04-08", revision=1, changed=True)
    row = await _raw_row(probe, APRIL_7)
    assert row is not None
    assert row["fetched_count"] == 2
    assert await raw.load_daily_papers(pool, target_date=APRIL_7) == [{"paper": {"id": "a"}}, {"paper": {"id": "b"}}]
    assert await raw.load_daily_papers(pool, target_date=APRIL_9) == []
    assert await raw.list_daily_papers_dates(pool) == [APRIL_7, APRIL_8]
    assert await raw.list_daily_papers_dates(pool, date_gt=APRIL_7, ascending=False) == [APRIL_8]
    assert await raw.list_daily_papers_dates(pool, ascending=False, limit=1) == [APRIL_8]
    assert await raw.list_daily_papers_dates(pool, date_gte=APRIL_7, date_lte=APRIL_7, limit=0) == [APRIL_7]


async def test_nul_characters_are_stored_without_error(pool) -> None:
    await raw.save_daily_papers(pool, target_date=APRIL_7, payload=[{"title": "a\x00b"}])

    assert await raw.load_daily_papers(pool, target_date=APRIL_7) == [{"title": "ab"}]


async def test_payload_is_cleaned_before_hashing_and_kept_as_unicode(pool, probe) -> None:
    payload = [{"title\x00": "논문 \ud800x", "tags": ("a\x00", 1, None)}]
    cleaned = [{"title": "논문 x", "tags": ["a", 1, None]}]

    await raw.save_daily_papers(pool, target_date=APRIL_7, payload=payload)

    row = await _raw_row(probe, APRIL_7)
    assert row is not None
    assert row["payload"] == cleaned
    assert row["payload_hash"] == compute_payload_hash(cleaned)
    assert await probe.scalar("SELECT payload::text LIKE '%%논문%%' FROM raw_daily_papers") is True


async def test_dict_payload_counts_as_one_and_loads_as_a_list(pool, probe) -> None:
    await raw.save_daily_papers(pool, target_date=APRIL_7, payload={"dailyPapers": []})

    row = await _raw_row(probe, APRIL_7)
    assert row is not None
    assert row["fetched_count"] == 1
    assert await raw.load_daily_papers(pool, target_date=APRIL_7) == [{"dailyPapers": []}]


async def test_concurrent_distinct_payloads_get_distinct_revisions(pool, probe) -> None:
    worker_count = 6
    barrier = asyncio.Barrier(worker_count)

    async def save(index: int) -> RawSaveResult:
        async with pool.connection() as connection:
            await barrier.wait()
            return await raw.save_daily_papers(connection, target_date=APRIL_7, payload=[{"n": index}])

    results = await asyncio.gather(*(save(index) for index in range(worker_count)))

    assert sorted(result.revision for result in results) == list(range(1, worker_count + 1))
    row = await _raw_row(probe, APRIL_7)
    assert row is not None
    assert row["revision"] == worker_count


async def test_pipeline_state_round_trip(pool) -> None:
    assert await pipeline_state.load_pipeline_state(pool, pipeline="hf_daily_papers_backfill", name="default") is None

    await pipeline_state.save_pipeline_state(
        pool,
        pipeline="hf_daily_papers_backfill",
        name="default",
        state={"cursor_date": "2026-04-06", "last_failure": {"date": "2026-04-07", "error": "429"}},
    )
    await pipeline_state.save_pipeline_state(
        pool,
        pipeline="hf_daily_papers_backfill",
        name="default",
        state={"cursor_date": "2026-04-05", "last_processed_dates": ["2026-04-06"], "last_failure": None},
    )
    loaded = await pipeline_state.load_pipeline_state(pool, pipeline="hf_daily_papers_backfill", name="default")

    assert loaded is not None
    assert loaded.values == {"cursor_date": "2026-04-05", "last_processed_dates": ["2026-04-06"], "last_failure": None}
    assert (loaded.pipeline, loaded.name) == ("hf_daily_papers_backfill", "default")
    assert loaded.updated_at is not None
    assert await pipeline_state.load_pipeline_state(pool, pipeline="hf_daily_papers_backfill", name="other") is None


async def test_pipeline_state_drops_reserved_keys_and_cleans_text(pool, probe) -> None:
    await pipeline_state.save_pipeline_state(
        pool,
        pipeline="prepare_papers_backfill",
        name="gpu",
        state={"pipeline": "x", "name": "y", "updated_at": "z", "status": "ok\x00", "note": "\ud800수집"},
    )

    stored = await probe.row("SELECT key, value FROM pipeline_state")
    assert stored == {"key": "prepare_papers_backfill:gpu", "value": {"status": "ok", "note": "수집"}}
    loaded = await pipeline_state.load_pipeline_state(pool, pipeline="prepare_papers_backfill", name="gpu")
    assert loaded is not None
    assert loaded.values == {"status": "ok", "note": "수집"}
    assert pipeline_state.pipeline_state_key("hf_daily_papers_backfill") == "hf_daily_papers_backfill:default"


async def _collect(
    pool, *, target_date: date, payload: Any, fail_after_enqueue: bool = False, revision_offset: int = 0
):
    async with transaction(pool) as connection:
        saved = await raw.save_daily_papers(connection, target_date=target_date, payload=payload)
        queued = await jobs.enqueue(
            connection,
            target_date=target_date,
            mode="auto",
            source="collect",
            raw_revision=saved.revision + revision_offset,
        )
        if fail_after_enqueue:
            raise RuntimeError("enqueue failed after insert")
    return saved, queued


async def test_collect_commits_raw_and_job_together_and_notifies(pool, probe, notifications) -> None:
    saved, queued = await _collect(pool, target_date=APRIL_7, payload=[{"paper": {"id": "2604.00001"}}])

    assert (saved.revision, saved.changed, queued.enqueued) == (1, True, True)
    row = await _raw_row(probe, APRIL_7)
    assert row is not None
    assert row["revision"] == 1
    job_rows = await probe.rows("SELECT target_date, status, raw_revision FROM prepare_jobs")
    assert job_rows == [{"target_date": APRIL_7, "status": "pending", "raw_revision": 1}]
    assert await notifications.drain() == ["auto:2026-04-07"]


async def test_collect_enqueue_failure_rolls_back_raw_and_sends_no_notification(pool, probe, notifications) -> None:
    with pytest.raises(RuntimeError, match="enqueue failed after insert"):
        await _collect(pool, target_date=APRIL_7, payload=[{"paper": {"id": "2604.00001"}}], fail_after_enqueue=True)

    assert await _raw_row(probe, APRIL_7) is None
    assert await probe.scalar("SELECT count(*) FROM prepare_jobs") == 0
    assert await notifications.drain() == []


async def test_collect_enqueue_database_error_rolls_back_raw(pool, probe, notifications) -> None:
    with pytest.raises(psycopg.errors.NumericValueOutOfRange):
        await _collect(pool, target_date=APRIL_7, payload=[{"paper": {"id": "2604.00001"}}], revision_offset=2**40)

    assert await _raw_row(probe, APRIL_7) is None
    assert await probe.scalar("SELECT count(*) FROM prepare_jobs") == 0
    assert await notifications.drain() == []
