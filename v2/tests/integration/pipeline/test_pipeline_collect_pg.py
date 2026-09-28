from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import psycopg
import pytest
from psycopg import sql

from arxplore.db.repositories import jobs, pipeline_state, raw
from arxplore.pipeline.backfill import BACKFILL_PIPELINE, backfill_daily_papers
from arxplore.pipeline.collect import collect_daily_papers

pytestmark = pytest.mark.integration

DAY = date(2026, 9, 15)
TODAY = date(2026, 9, 20)


@dataclass
class FakeDailyPapers:
    payloads: dict[date, list[Any]] = field(default_factory=dict)
    errors: dict[date, Exception] = field(default_factory=dict)
    requested: list[date] = field(default_factory=list)
    default: Callable[[date], list[Any]] = lambda day: [{"paper": {"id": f"2609.{day.day:05d}"}}]

    async def fetch_daily_papers(self, target_date: date) -> list[Any]:
        self.requested.append(target_date)
        if target_date in self.errors:
            raise self.errors[target_date]
        return self.payloads.get(target_date) or self.default(target_date)


async def _raw_dates(probe: Any) -> list[date]:
    return [row["date"] for row in await probe.rows("SELECT date FROM raw_daily_papers ORDER BY date")]


async def test_collect_saves_the_raw_payload_and_enqueues_the_date(pool, probe) -> None:
    payload = [{"paper": {"id": f"2609.0000{number}"}} for number in range(1, 8)] + ["not a dict"]
    source = FakeDailyPapers(payloads={DAY: payload})

    result = await collect_daily_papers(pool, source, target_date=DAY)

    assert result.as_dict() == {
        "stage": "collect_papers",
        "status": "success",
        "target_date": "2026-09-15",
        "fetched_count": 8,
        "stored_record_id": "hf_daily_papers:2026-09-15",
        "raw_revision": 1,
        "raw_payload_changed": True,
        "prepare_job_enqueued": True,
        "prepare_job_id": 1,
        "prepare_job_status": "pending",
        "sample_arxiv_ids": ["2609.00001", "2609.00002", "2609.00003", "2609.00004", "2609.00005"],
    }
    job = await probe.row("SELECT mode, target_date, source, raw_revision, status FROM prepare_jobs")
    assert job == {"mode": "auto", "target_date": DAY, "source": "collect", "raw_revision": 1, "status": "pending"}


async def test_an_unchanged_payload_keeps_its_revision_and_still_enqueues(pool, probe) -> None:
    source = FakeDailyPapers(payloads={DAY: [{"paper": {"id": "2609.00001"}}]})
    await collect_daily_papers(pool, source, target_date=DAY)
    await probe.execute("UPDATE prepare_jobs SET status = 'done'")

    again = await collect_daily_papers(pool, source, target_date=DAY)

    assert (again.raw_revision, again.raw_payload_changed) == (1, False)
    assert again.job is not None and again.job.enqueued is False
    assert await probe.scalar("SELECT status FROM prepare_jobs") == "done"


async def test_a_changed_payload_bumps_the_revision_and_reopens_the_job(pool, probe) -> None:
    source = FakeDailyPapers(payloads={DAY: [{"paper": {"id": "2609.00001"}}]})
    await collect_daily_papers(pool, source, target_date=DAY)
    await probe.execute("UPDATE prepare_jobs SET status = 'done'")
    source.payloads[DAY] = [{"paper": {"id": "2609.00001"}}, {"paper": {"id": "2609.00002"}}]

    again = await collect_daily_papers(pool, source, target_date=DAY)

    assert (again.raw_revision, again.raw_payload_changed) == (2, True)
    assert await probe.row("SELECT status, raw_revision FROM prepare_jobs") == {"status": "pending", "raw_revision": 2}


async def test_an_enqueue_failure_rolls_back_the_raw_save(pool, probe, monkeypatch) -> None:
    async def broken_enqueue(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("enqueue failed")

    monkeypatch.setattr(jobs, "enqueue", broken_enqueue)

    with pytest.raises(RuntimeError, match="enqueue failed"):
        await collect_daily_papers(pool, FakeDailyPapers(), target_date=DAY)

    assert await _raw_dates(probe) == []


async def test_the_notification_is_delivered_after_commit(pool, pipeline_database) -> None:
    async with await psycopg.AsyncConnection.connect(pipeline_database, autocommit=True) as listener:
        await listener.execute(sql.SQL("LISTEN {}").format(sql.Identifier(jobs.JOB_CHANNEL)))
        await collect_daily_papers(pool, FakeDailyPapers(), target_date=DAY)
        payloads = [notify.payload async for notify in listener.notifies(timeout=5.0, stop_after=1)]
    assert payloads == ["auto:2026-09-15"]


async def test_the_http_request_happens_before_the_transaction(pool, probe) -> None:
    class Source:
        async def fetch_daily_papers(self, target_date: date) -> list[Any]:
            active = await probe.scalar(
                "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND state LIKE 'idle in%%'"
            )
            assert active == 0
            return []

    result = await collect_daily_papers(pool, Source(), target_date=DAY)
    assert result.fetched_count == 0


async def test_collect_without_enqueue_saves_only_the_raw_payload(pool, probe) -> None:
    result = await collect_daily_papers(pool, FakeDailyPapers(), target_date=DAY, enqueue_prepare=False)

    assert (result.job, result.as_dict()["prepare_job_id"], result.as_dict()["prepare_job_status"]) == (
        None,
        None,
        None,
    )
    assert await probe.scalar("SELECT count(*) FROM prepare_jobs") == 0
    assert await _raw_dates(probe) == [DAY]


async def test_backfill_collects_missing_days_newest_first_and_skips_existing(pool, probe) -> None:
    await raw.save_daily_papers(pool, target_date=date(2026, 9, 18), payload=[])
    source = FakeDailyPapers()

    result = await backfill_daily_papers(pool, source, today=TODAY, oldest_date=date(2026, 9, 10), batch_days=4)

    assert source.requested == [date(2026, 9, 19), date(2026, 9, 17), date(2026, 9, 16)]
    assert result.as_dict() == {
        "stage": "backfill_collect_papers",
        "status": "success",
        "state_name": "default",
        "cursor_date": "2026-09-19",
        "next_cursor_date": "2026-09-15",
        "oldest_date": "2026-09-10",
        "batch_days": 4,
        "success_count": 3,
        "skipped_existing_count": 1,
        "failure_count": 0,
        "successes": [
            {"date": "2026-09-19", "fetched_count": 1, "stored_record_id": "hf_daily_papers:2026-09-19"},
            {"date": "2026-09-17", "fetched_count": 1, "stored_record_id": "hf_daily_papers:2026-09-17"},
            {"date": "2026-09-16", "fetched_count": 1, "stored_record_id": "hf_daily_papers:2026-09-16"},
        ],
        "skipped_existing": ["2026-09-18"],
        "failures": [],
        "stopped_reason": "batch_complete",
    }
    queued = await probe.rows("SELECT mode, source, target_date, raw_revision, status FROM prepare_jobs ORDER BY id")
    assert queued == [
        {"mode": "backfill", "source": "backfill", "target_date": day, "raw_revision": 1, "status": "pending"}
        for day in (date(2026, 9, 19), date(2026, 9, 17), date(2026, 9, 16))
    ]
    state = await pipeline_state.load_pipeline_state(pool, pipeline=BACKFILL_PIPELINE)
    assert state is not None
    assert state.values == {
        "cursor_date": "2026-09-15",
        "oldest_date": "2026-09-10",
        "batch_days": 4,
        "status": "success",
        "last_processed_dates": ["2026-09-19", "2026-09-17", "2026-09-16", "2026-09-18"],
        "last_failure": None,
    }


async def test_backfill_resumes_from_the_saved_cursor_and_completes(pool) -> None:
    source = FakeDailyPapers()
    await backfill_daily_papers(pool, source, today=TODAY, oldest_date=date(2026, 9, 16), batch_days=2)

    second = await backfill_daily_papers(pool, source, today=TODAY, batch_days=2)

    assert source.requested == [date(2026, 9, 19), date(2026, 9, 18), date(2026, 9, 17), date(2026, 9, 16)]
    assert (second.status, second.next_cursor_date, second.stopped_reason) == ("completed", None, "batch_complete")


async def test_backfill_with_the_cursor_past_the_oldest_date_is_exhausted(pool) -> None:
    source = FakeDailyPapers()

    result = await backfill_daily_papers(
        pool, source, today=TODAY, cursor_date=date(2026, 9, 1), oldest_date=date(2026, 9, 2)
    )

    assert source.requested == []
    assert (result.status, result.stopped_reason, result.next_cursor_date) == ("completed", "cursor_exhausted", None)
    state = await pipeline_state.load_pipeline_state(pool, pipeline=BACKFILL_PIPELINE)
    assert state is not None and state.values["status"] == "completed"


async def test_a_finished_cursor_restarts_from_yesterday(pool) -> None:
    await pipeline_state.save_pipeline_state(
        pool, pipeline=BACKFILL_PIPELINE, state={"cursor_date": None, "oldest_date": "2026-09-18"}
    )
    source = FakeDailyPapers()

    result = await backfill_daily_papers(pool, source, today=TODAY)

    assert source.requested == [date(2026, 9, 19), date(2026, 9, 18)]
    assert result.status == "completed"


@pytest.mark.parametrize(
    "error,status,reason",
    [
        (RuntimeError("Client error '429 Too Many Requests' for url"), "rate_limited", "rate_limited"),
        (RuntimeError("Too Many Requests"), "rate_limited", "rate_limited"),
        (RuntimeError("503 Service Unavailable"), "failed", "collection_failed"),
    ],
)
async def test_backfill_stops_at_the_first_failure(pool, error, status, reason) -> None:
    source = FakeDailyPapers(errors={date(2026, 9, 18): error})

    result = await backfill_daily_papers(pool, source, today=TODAY, oldest_date=date(2026, 9, 10), batch_days=5)

    assert source.requested == [date(2026, 9, 19), date(2026, 9, 18)]
    assert (result.status, result.stopped_reason) == (status, reason)
    assert result.next_cursor_date == date(2026, 9, 18)
    assert result.failures == [{"date": "2026-09-18", "error": str(error)}]
    state = await pipeline_state.load_pipeline_state(pool, pipeline=BACKFILL_PIPELINE)
    assert state is not None
    assert state.values["cursor_date"] == "2026-09-18"
    assert state.values["last_failure"] == {"date": "2026-09-18", "error": str(error)}
